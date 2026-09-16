"""Disposable exact numeric work; never a source of graph lifecycle authority."""
from __future__ import annotations

import hashlib
import math
import shutil
import tempfile
import time
from contextvars import ContextVar
from functools import wraps
from pathlib import Path
from typing import Any, Callable

import numpy as np
from threadpoolctl import threadpool_limits
from app.services.build_performance import measure

NUMERIC_PROTOCOL = "graph_build_numeric_float64_v1"
_CURRENT: ContextVar[GraphBuildWorkspace | None] = ContextVar("graph_build_workspace", default=None)


def current_workspace() -> GraphBuildWorkspace | None:
    return _CURRENT.get()


def workspace_protocol_cache(fn):
    """Cache only immutable code protocol cards, within this build's lifetime."""
    @wraps(fn)
    def wrapped():
        workspace = current_workspace()
        if workspace is None:
            return fn()
        return workspace.cached("code_protocol:"+fn.__module__+"."+fn.__name__, fn)
    return wrapped


def numeric_distances(vectors: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """Direct differences avoid the cancellation in x²+c²-2xc."""
    result = np.empty((len(vectors), len(centers)), dtype=np.float64)
    for offset in range(0, len(vectors), 128):
        if len(vectors) > 1:
            checkpoint("rq_training", offset, len(vectors))
        with measure("rq_distance_block"):
            delta = vectors[offset:offset + 128, None, :] - centers[None, :, :]
            result[offset:offset + 128] = np.einsum("bkd,bkd->bk", delta, delta, optimize=False)
    # Refine ambiguous nearest-center choices using the original scalar sum.
    if len(centers) > 1:
        nearest = np.partition(result, 1, axis=1)[:, :2]
        tolerance = max(1e-12, 8 * vectors.shape[1] * np.finfo(np.float64).eps)
        ambiguous = np.flatnonzero(np.abs(nearest[:, 1] - nearest[:, 0]) <= tolerance * np.maximum(1, nearest[:, 1]))
        for row in ambiguous:
            for column, center in enumerate(centers):
                result[row, column] = sum((float(a) - float(b)) ** 2 for a, b in zip(vectors[row], center))
        if current_workspace():
            current_workspace().counts["rq_boundary_refinements"] += len(ambiguous)
    return result


def exact_decimal_round(values: np.ndarray, digits: int) -> np.ndarray:
    """Vectorize decimal rounding, replaying uncertain cases with Python round.

    The fast domain keeps scaled integers exactly representable. Values near
    a half-way rounding boundary, very large values and non-finite input take
    the original scalar path; no initialization hash may change silently.
    """
    source=np.asarray(values,dtype=np.float64)
    if digits not in (6,12) or source.ndim not in (1,2):
        raise ValueError("Unsupported graph decimal rounding domain")
    scale=float(10**digits)
    with np.errstate(over="ignore",invalid="ignore"):
        scaled=source*scale
        rounded=np.round(source,decimals=digits)
        margin=np.abs(np.abs(scaled-np.trunc(scaled))-.5)
        tolerance=8*np.spacing(np.maximum(np.abs(scaled),1.))
        uncertain=(~np.isfinite(scaled)) | (np.abs(scaled)>2**46) | (margin<=tolerance)
    for index in zip(*np.nonzero(uncertain)):
        rounded[index]=round(float(source[index]),digits)
    if current_workspace():
        current_workspace().counts["decimal_round_refinements"] += int(np.count_nonzero(uncertain))
    return rounded


def checkpoint(phase: str, completed: int = 0, total: int = 0) -> None:
    from app.services.build_performance import current_performance
    performance = current_performance()
    if performance is not None:
        performance.check()
    workspace = current_workspace()
    if workspace is not None:
        workspace.checkpoint(phase, completed, total)


class ScoredRow:
    """A view, avoiding n² Python tuples and Chunk references."""
    def __init__(self, workspace: GraphBuildWorkspace, row: int, indices: np.ndarray):
        self.workspace, self.row, self.indices = workspace, row, indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, key):
        if isinstance(key, slice):
            return ScoredRow(self.workspace, self.row, self.indices[key])
        index = int(self.indices[key])
        return float(self.workspace.scores[self.row, index]), self.workspace.chunks[index]

    def filtered(self, mask: np.ndarray):
        return ScoredRow(self.workspace, self.row, self.indices[mask[self.indices]])


class GraphBuildWorkspace:
    def __init__(self, *, memory_mb: int = 256, threads: int = 2, progress_seconds: int = 5,
                 scratch_root: Path | None = None, control: Callable | None = None):
        from collections import Counter
        self.memory_bytes = int(memory_mb) * 1024**2
        self.threads = int(threads)
        self.progress_seconds = int(progress_seconds)
        self.scratch_root = scratch_root
        self.control = control
        self.counts = Counter()
        self.cache: dict[str, Any] = {}
        self.chunks: list[Any] = []
        self.scores = self.order = self.matrix = None
        self.vector_digest = None
        self.vector_object = None
        self.ids: list[str] = []
        self.last_poll = self.last_progress = 0.0
        self.phase = ""
        self.prepare_seconds = 0.0
        self.best_candidate = None
        self.last_candidate = None
        self.scalar_cache = {}
        self.temp_directory = None
        self.started = time.perf_counter()

    def __enter__(self):
        self.token = _CURRENT.set(self)
        self.thread_limit = threadpool_limits(limits=self.threads)
        return self

    def __exit__(self, *exc):
        try:
            from app.services.build_performance import current_performance
            performance = current_performance()
            if performance is not None:
                performance.numeric_workspace = self.summary()
            for array in (self.scores, self.order):
                if isinstance(array, np.memmap):
                    array.flush()
                    array._mmap.close()
            self.scores = self.order = self.matrix = None
            self.cache.clear()
            self.chunks = []
            self.vector_object = None
            self.scalar_cache.clear()
            self.best_candidate = self.last_candidate = None
            if self.temp_directory:
                if performance is not None:
                    performance.sample_temporary()
                    performance.temporary_roots.discard(self.temp_directory)
                shutil.rmtree(self.temp_directory)
        finally:
            self.thread_limit.restore_original_limits()
            _CURRENT.reset(self.token)
            from app.services.resource_guard import release_unused_memory
            release_unused_memory()

    def checkpoint(self, phase: str, completed: int = 0, total: int = 0):
        now = time.perf_counter()
        changed = phase != self.phase
        progress = changed or now - self.last_progress >= self.progress_seconds
        if self.control and (progress or now - self.last_poll >= 1.0):
            self.control(phase, completed, total, progress)
            self.last_poll = now
        if progress:
            self.last_progress, self.phase = now, phase

    def cached(self, key: str, loader: Callable):
        if key not in self.cache:
            self.cache[key] = loader()
        return self.cache[key]

    def bind(self, chunks, vectors):
        ids = [str(chunk.id) for chunk in chunks]
        scope = [(str(chunk.id), *(getattr(chunk, key, None) for key in ("document_version_id", "chunk_version", "char_start", "char_end", "token_start", "token_end", "text_hash", "state")), hashlib.sha256(str(getattr(chunk,"text","")).encode("utf-8")).hexdigest()) for chunk in chunks]
        if self.matrix is not None:
            if ids != self.ids or vectors is not self.vector_object or scope != self.chunk_scope:
                raise RuntimeError("Graph numeric workspace input identity changed")
            if any(len(vectors[key]) != self.matrix.shape[1] for key in ids):
                raise RuntimeError("Graph numeric workspace vector shape changed")
            observed = np.asarray([vectors[key] for key in ids], dtype=np.float64)
            if hashlib.sha256(observed.tobytes()).hexdigest() != self.vector_digest:
                raise RuntimeError("Graph numeric workspace vector values changed")
            return
        started = time.perf_counter()
        self.chunks, self.ids, self.vector_object = list(chunks), ids, vectors
        self.chunk_scope = scope
        self.index = {key: index for index, key in enumerate(ids)}
        n=len(chunks)
        d=len(vectors[chunks[0].id]) if n else 0
        if n*d*16 + max(1,n)*128*32 > self.memory_bytes:
            raise MemoryError("Graph numeric input and block exceed workspace memory budget")
        self.matrix = np.asarray([vectors[chunk.id] for chunk in chunks], dtype=np.float64, order="C")
        if self.matrix.ndim != 2 or not self.matrix.shape[1] or not np.isfinite(self.matrix).all():
            raise ValueError("Graph numeric workspace requires finite rectangular vectors")
        self.vector_digest = hashlib.sha256(self.matrix.tobytes()).hexdigest()
        self.row_digests = [hashlib.sha256(row.tobytes()).digest() for row in self.matrix]
        n, d = self.matrix.shape
        block_reserve = self.matrix.nbytes * 2 + max(1, n) * 128 * 32
        if block_reserve > self.memory_bytes:
            raise MemoryError("Graph numeric input and block exceed workspace memory budget")
        required = n*n*12
        if required + block_reserve > self.memory_bytes:
            root = self.scratch_root or Path(tempfile.gettempdir())
            root.mkdir(parents=True, exist_ok=True)
            if shutil.disk_usage(root).free < required * 2 + 64 * 1024**2:
                raise MemoryError("Insufficient scratch space for exact graph workspace")
            self.temp_directory = Path(tempfile.mkdtemp(prefix="graph-numeric-", dir=root))
            self.scores = np.memmap(self.temp_directory / "scores", mode="w+", dtype="float64", shape=(n, n))
            self.order = np.memmap(self.temp_directory / "order", mode="w+", dtype="int32", shape=(n, n))
            self.counts["mapped_bytes"] = required
            from app.services.build_performance import current_performance
            performance = current_performance()
            if performance is not None:
                performance.temporary_roots.add(self.temp_directory)
                performance.sample_temporary()
        else:
            self.scores = np.empty((n, n), dtype=np.float64)
            self.order = np.empty((n, n), dtype=np.int32)
        norms = np.sqrt(np.einsum("ij,ij->i", self.matrix, self.matrix, optimize=False))
        if np.any(norms <= 0) or not np.isfinite(norms).all():
            raise ValueError("Graph numeric workspace requires nonzero finite norms")
        self.norms = norms
        self.epsilon = max(1e-12, 8*d*np.finfo(np.float64).eps)
        # Stable id order is the existing relation tie-break contract.
        tie_order = np.asarray(sorted(range(n), key=lambda i: ids[i]), dtype=np.int32)
        for offset in range(0, n, 128):
            self.checkpoint("similarity", offset, n)
            with measure("similarity_block"):
                block = self.matrix[offset:offset+128] @ self.matrix.T
                block /= norms[offset:offset+128, None] * norms[None, :]
                self.scores[offset:offset+len(block)] = block
        for row in range(n):
            self.checkpoint("neighbor_order", row, n)
            ordered = tie_order[np.argsort(-self.scores[row, tie_order], kind="stable")]
            # Refine adjacent near-ties, including duplicate vectors.
            close = np.flatnonzero(np.abs(np.diff(self.scores[row, ordered])) <= self.epsilon)
            indices = set(ordered[close].tolist() + ordered[close+1].tolist())
            for column in indices:
                self.scores[row, column] = self.scalar_cosine(row, column)
            if indices:
                ordered = tie_order[np.argsort(-self.scores[row, tie_order], kind="stable")]
            self.order[row] = ordered
        self.counts["similarity_preparations"] += 1
        self.counts["neighbor_sort_passes"] += 1
        self.prepare_seconds += time.perf_counter() - started

    def scalar_cosine(self, row, column):
        self.counts["cosine_boundary_refinements"] += 1
        key = tuple(sorted((self.row_digests[row], self.row_digests[column])))
        if key in self.scalar_cache:
            return self.scalar_cache[key]
        left, right = self.matrix[row].tolist(), self.matrix[column].tolist()
        result = sum(a*b for a,b in zip(left,right)) / (math.sqrt(sum(a*a for a in left))*math.sqrt(sum(b*b for b in right)))
        if len(self.scalar_cache) < 8192:
            self.scalar_cache[key] = result
        return result

    def rows(self):
        return {key: ScoredRow(self, i, self.order[i][self.order[i] != i]) for i,key in enumerate(self.ids)}

    def classify(self, documents, languages):
        self.document_codes = np.asarray([str(c.document_id) for c in self.chunks])
        self.languages = np.asarray([languages(str(c.id)) for c in self.chunks])

    def masks(self, row):
        cross_language = (self.languages[row] != "unknown") & (self.languages != "unknown") & (self.languages[row] != self.languages)
        cross_document = self.document_codes[row] != self.document_codes
        return cross_document, cross_language

    def refine_thresholds(self, thresholds):
        for row in range(len(self.ids)):
            self.checkpoint("candidate_thresholds", row, len(self.ids))
            values = self.scores[row]
            close = np.zeros(len(values), dtype=bool)
            for threshold in thresholds:
                close |= np.abs(values - threshold) <= self.epsilon
            for column in np.flatnonzero(close):
                values[column] = self.scalar_cosine(row, int(column))

    def pressure(self, thresholds):
        n = len(self.ids)
        inbound = np.zeros(n, dtype=np.int64)
        bridges = np.zeros(n, dtype=np.float64)
        available = np.zeros(n, dtype=bool)
        top4 = [[] for _ in range(n)]
        bridge_counts = np.zeros(n, dtype=np.int64)
        for row in range(n):
            self.checkpoint("quota_signals", row, n)
            doc, lang = self.masks(row)
            threshold = np.where(lang, thresholds["dense_cross_language_bridge"], np.where(doc, thresholds["dense_cross_document_bridge"], thresholds["dense_semantic"]))
            eligible = self.scores[row] >= threshold
            eligible[row] = False
            inbound += eligible
            domain = doc | lang
            available[row] |= bool(domain.any())
            available |= domain
            bridge = domain & eligible
            bridge_counts += bridge
            bridge_counts[row] += int(bridge.sum())
            strength = np.where(bridge, np.clip((self.scores[row]-threshold)/np.maximum(1-threshold, 1e-6), 0, 1), 0)
            bridges[row] = max(bridges[row], float(strength.max(initial=0)))
            bridges = np.maximum(bridges, strength)
            ordered = self.order[row]
            values = self.scores[row, ordered[(ordered != row) & (self.scores[row, ordered] > 0)]][:4]
            top4[row] = np.clip(values, 0, 1).tolist()
        return inbound, bridges, available, top4, bridge_counts

    def summary(self):
        return {"protocol_version": NUMERIC_PROTOCOL, "prepare_seconds": self.prepare_seconds,
                "elapsed_seconds": time.perf_counter()-self.started, "counts": dict(self.counts),
                "n":len(self.ids), "d":int(self.matrix.shape[1]) if self.matrix is not None else 0,
                "matrix_bytes": int(self.scores.nbytes) if self.scores is not None else 0,
                "memory_budget_bytes": self.memory_bytes, "numeric_threads": self.threads}


def graph_workspace_scope(fn):
    @wraps(fn)
    def wrapped(db, knowledge_base_id, chunks, *args, **kwargs):
        if current_workspace() is not None:
            return fn(db, knowledge_base_id, chunks, *args, **kwargs)
        from app.core.config import get_settings
        settings = get_settings()
        batch_id = kwargs.get("batch_id")
        def control(phase, completed, total, progress):
            from app.services import context_graph
            context_graph.ensure_not_cancelled(db, batch_id)
            if progress and batch_id and kwargs.get("emit_heartbeats", True):
                context_graph.context_graph_batch_heartbeat(batch_id, "chunk_relation:"+phase, {"completed": completed, "total": total})
        with GraphBuildWorkspace(memory_mb=settings.graph_compute_memory_mb,
                                 threads=settings.graph_compute_threads,
                                 progress_seconds=settings.graph_progress_interval_seconds,
                                 scratch_root=Path(settings.data_root)/"build_scratch", control=control) as workspace:
            result = fn(db, knowledge_base_id, chunks, *args, **kwargs)
            result.diagnostics_json = {**dict(result.diagnostics_json or {}), "numeric_workspace": workspace.summary()}
            return result
    return wrapped
