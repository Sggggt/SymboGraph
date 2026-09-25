"""One request's immutable raw-text/vector view and independent facet discovery."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import re
import unicodedata

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import load_only
from threadpoolctl import threadpool_limits

from app.core.config import get_settings
from app.models import Chunk, Document, DocumentVersion, VectorRecord
from app.retrieval_control_contracts import LexicalRepairCandidate, TaskContract, control_hash
from app.services.qa_performance import qa_stage
from app.services.retrieval_constraints import formal_literal, formal_literals_match


def normalized_surface(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


@dataclass(frozen=True)
class CorpusSource:
    chunk_id: str
    document_id: str
    document_version_id: str
    title: str
    text: str
    char_start: int
    char_end: int
    text_hash: str
    section_path: str | None = None


@dataclass(frozen=True)
class SourceLocator:
    id: str
    chunk_id: str
    document_version_id: str
    char_start: int
    char_end: int
    surface: str
    source_scope_hash: str


class EmptyFilteredRetrievalScope(ValueError):
    def __init__(self, checked_source_count):
        self.checked_source_count = checked_source_count
        super().__init__('retrieval_filtered_scope_empty')


class RetrievalCorpus:
    def __init__(self, *, knowledge_base_id, sources, vectors, target, scope_hash, threads):
        self.knowledge_base_id = knowledge_base_id
        self.sources = tuple(sources)
        self.by_id = {source.chunk_id: source for source in sources}
        self.positions = {source.chunk_id: index for index, source in enumerate(sources)}
        self.vectors = vectors
        self.target = target
        self.scope_hash = scope_hash
        self.threads = threads
        norms = np.linalg.norm(vectors, axis=1)
        if np.any(norms <= 0) or not np.isfinite(vectors).all():
            raise ValueError("retrieval_corpus_vectors_invalid")
        self.unit_vectors = vectors / norms[:, None]
        self.vectors.flags.writeable = self.unit_vectors.flags.writeable = False
        self.locators: dict[str, SourceLocator] = {}
        self.source_use_cache = {}
        self.vector_identity_hash = control_hash({
            "runtime_state_hash": target.runtime_state_hash,
            "activation_generation": target.activation_generation,
            "schema_hash": target.vector_schema_hash,
        })
        self._normalized_texts: tuple[str, ...] | None = None

    @classmethod
    def load(cls, db, *, knowledge_base_id, filters):
        from app.services.context_graph import (
            _active_vector_record_protocol_reasons, _vector_runtime_target_for_kb,
            canonical_embedding_vector, passes_filters,
        )
        with qa_stage("candidate_discovery"):
            target = _vector_runtime_target_for_kb(db, knowledge_base_id)
            schema = target.schema
            chunk_columns = [Chunk.id, Chunk.knowledge_base_id, Chunk.document_id, Chunk.document_version_id,
                Chunk.state, Chunk.text, Chunk.char_start, Chunk.char_end, Chunk.text_hash, Chunk.section_path]
            if filters.page_range:
                chunk_columns.extend((Chunk.page_start,Chunk.page_end))
            if filters.chunk_version is not None:
                chunk_columns.append(Chunk.chunk_version)
            if filters.partition or filters.content_kinds:
                chunk_columns.append(Chunk.metadata_json)
            raw = list(db.execute(select(Chunk, DocumentVersion, Document, VectorRecord)
                .options(load_only(*chunk_columns),
                    load_only(DocumentVersion.id, DocumentVersion.document_id, DocumentVersion.checksum, DocumentVersion.is_active),
                    load_only(Document.id, Document.title, Document.source_type, Document.source_path, Document.tags, Document.is_active))
                .join(DocumentVersion, DocumentVersion.id == Chunk.document_version_id)
                .join(Document, Document.id == Chunk.document_id)
                .join(VectorRecord, VectorRecord.chunk_id == Chunk.id)
                .where(Chunk.knowledge_base_id == knowledge_base_id, Chunk.state == "active",
                       DocumentVersion.is_active.is_(True), Document.is_active.is_(True),
                       VectorRecord.embedding_model == schema.embedding_model,
                       VectorRecord.embedding_dimension == schema.embedding_dimension,
                       VectorRecord.embedding_text_version == schema.embedding_text_version,
                       VectorRecord.chunk_schema_version == schema.chunk_schema_version,
                       VectorRecord.vector_status == target.ready_vector_status)
                .order_by(Chunk.id)))
            eligible = [row for row in raw if passes_filters(db, row[0], filters)]
            if raw and not eligible:
                raise EmptyFilteredRetrievalScope(len(raw))
            settings = get_settings()
            size = len(eligible) * schema.embedding_dimension * np.dtype(np.float64).itemsize * 2
            if size > settings.graph_compute_memory_mb * 1024 * 1024:
                raise ValueError("retrieval_corpus_numeric_budget_exceeded")
            sources, facts, seen = [], [], set()
            array = np.empty((len(eligible), schema.embedding_dimension), dtype=np.float64)
            for row_index, (chunk, version, document, record) in enumerate(eligible):
                if chunk.id in seen:
                    raise ValueError("retrieval_corpus_vector_scope_ambiguous")
                if version.document_id != document.id or _active_vector_record_protocol_reasons(record, vector_runtime_target=target):
                    raise ValueError("retrieval_corpus_vector_provenance_invalid")
                vector = canonical_embedding_vector((record.diagnostics_json or {}).get("embedding_vector"),
                                                    source="retrieval corpus canonical vector")
                if len(vector) != schema.embedding_dimension:
                    raise ValueError("retrieval_corpus_vector_dimension_invalid")
                seen.add(chunk.id)
                sources.append(CorpusSource(chunk.id, document.id, version.id, document.title, chunk.text,
                                            chunk.char_start, chunk.char_end, chunk.text_hash, chunk.section_path))
                array[row_index] = vector
                facts.append((chunk.id, version.id, version.checksum, chunk.text_hash, record.id,
                              hashlib.sha256(np.asarray(vector, dtype=np.float32).tobytes()).hexdigest(),
                              document.title, chunk.section_path, chunk.char_start, chunk.char_end))
            if not sources:
                raise ValueError("retrieval_corpus_has_no_active_sources")
            scope_hash = control_hash({"protocol_version": "retrieval_raw_source_scope_v2",
                                       "filter_protocol": "active_source_filters_v2",
                                       "kb": knowledge_base_id, "vector_state": target.runtime_state_hash,
                                       "filter_scope": filters.model_dump(mode="json"), "sources": facts})
            return cls(knowledge_base_id=knowledge_base_id, sources=sources, vectors=array, target=target,
                       scope_hash=scope_hash, threads=settings.graph_compute_threads)

    def cosine_scores(self, query_vectors):
        values = np.ascontiguousarray(query_vectors, dtype=np.float64)
        if values.ndim == 1:
            values = values[None, :]
        if values.shape[1] != self.vectors.shape[1] or not np.isfinite(values).all():
            raise ValueError("retrieval_query_vector_shape_invalid")
        norms = np.linalg.norm(values, axis=1)
        if np.any(norms <= 0):
            raise ValueError("retrieval_query_zero_vector")
        with threadpool_limits(limits=self.threads), qa_stage("path_features", item_count=len(self.sources)):
            return np.clip((values / norms[:, None]) @ self.unit_vectors.T, 0, 1)

    def literal_lookup(self, surface: str):
        normalized = normalized_surface(surface)
        if not normalized:
            raise ValueError("literal_lookup_empty_surface")
        if self._normalized_texts is None:
            self._normalized_texts = tuple(normalized_surface(source.text) for source in self.sources)
        hits = tuple(source.chunk_id for source, text in zip(self.sources, self._normalized_texts) if normalized in text)
        # The view is a retrieval/ready-vector scope, not proof that all raw
        # PDFs (images, failed parses or other scopes) have been represented.
        return {"matched_chunk_ids": hits, "observed_source_count": len(self.sources),
                "source_scope_hash": self.scope_hash, "representation": "active_retrievable_text",
                "matcher": "nfkc_casefold_whitespace_substring_v1", "corpus_fact_absence_proven": False}

    def locate(self, source: CorpusSource, start: int, end: int):
        if not 0 <= start < end <= len(source.text):
            raise ValueError("lexical_locator_span_invalid")
        surface = source.text[start:end]
        identity = {"chunk_id": source.chunk_id, "document_version_id": source.document_version_id,
                    "char_span": [source.char_start + start, source.char_start + end],
                    "surface": surface, "source_scope_hash": self.scope_hash}
        locator = SourceLocator("w_" + control_hash(identity), source.chunk_id, source.document_version_id,
                                source.char_start + start, source.char_start + end, surface, self.scope_hash)
        self.locators[locator.id] = locator
        return locator

    def discover(self, *, task: TaskContract, missing_facet_ids, facet_scores, excluded_chunk_ids=(),
                 per_facet_candidates=3):
        """Independent canonical-facet retrieval, not the failed lexical route."""
        if len(missing_facet_ids) > 2 or not 1 <= per_facet_candidates <= 3:
            raise ValueError("lexical_discovery_budget_invalid")
        excluded = set(excluded_chunk_ids)
        candidates = []
        by_facet = {item.id: (index, item) for index, item in enumerate(task.requirements)}
        with qa_stage("candidate_discovery"):
            from app.services.source_use import analyze_source_use, source_use_decision
            from app.services.storage import raise_if_source_io_cancelled
            for facet_id in missing_facet_ids:
                if facet_id not in by_facet:
                    raise ValueError("lexical_discovery_outside_task")
                position, facet = by_facet[facet_id]
                eligible = [index for index, source in enumerate(self.sources)
                            if formal_literals_match(facet, source.title, source.text)]
                if not eligible and any(formal_literal(value) for value in facet.protected_literals) and not task.allow_partial:
                    # There is no attested direction for a mandatory identifier
                    # in this observed scope. Do not ask the model to change it.
                    # This is a bounded stop, not semantic corpus-absence proof.
                    return ()
                useful = []
                for ordinal, index in enumerate(eligible):
                    if ordinal % 128 == 0:
                        raise_if_source_io_cancelled()
                    source = self.sources[index]
                    if source.chunk_id not in self.source_use_cache:
                        self.source_use_cache[source.chunk_id] = analyze_source_use(source.text)
                    # Discovery has no native structure witness yet. Preserve
                    # explicit structural requests for the actual-package gate.
                    if set(facet.source_roles) & {'table', 'formula', 'code'} or source_use_decision(
                            task, facet, self.source_use_cache[source.chunk_id]).allowed:
                        useful.append(index)
                ranked = sorted(useful, key=lambda index: (-float(facet_scores[position, index]),
                                                                            self.sources[index].chunk_id))
                used_surfaces = set()
                for index in ranked[:32]:
                    source = self.sources[index]
                    if source.chunk_id in excluded or float(facet_scores[position, index]) <= 0:
                        continue
                    # Use a short actual line/phrase as a locator. It remains
                    # related_locator until stronger equivalence is attested.
                    pieces = list(re.finditer(r"[^\n!?。！？]{8,96}", source.text))
                    if not pieces:
                        continue
                    preferred = next((match for match in pieces if any(
                        normalized_surface(word) in normalized_surface(match.group())
                        for word in facet.protected_literals if word)), pieces[0])
                    text = preferred.group()
                    left = preferred.start() + len(text) - len(text.lstrip())
                    right = preferred.end() - len(text) + len(text.rstrip())
                    if right <= left:
                        continue
                    surface = source.text[left:right]
                    normalized = normalized_surface(surface)
                    if normalized in used_surfaces:
                        continue
                    used_surfaces.add(normalized)
                    locator = self.locate(source, left, right)
                    context_start = max(0, left - 64)
                    context_end = min(len(source.text), context_start + 240)
                    words = surface.split()
                    proposed_equivalence = (
                        facet.role not in {"source_role", "comparison"}
                        and 2 <= len(words) <= 8 and len(surface) <= 64
                        and not any(character in surface for character in (":", ";", "=", "|"))
                    )
                    candidates.append(LexicalRepairCandidate(id="c_" + control_hash({
                        "facet": facet_id, "witness": locator.id})[:24],
                        facet_id=facet_id, surface=surface, witness_id=locator.id,
                        context=source.text[context_start:context_end],
                        source_title=" ".join(source.title.split())[:160],
                        source_section=" ".join(source.section_path.split())[:160] if source.section_path else None,
                        relation="proposed_equivalence" if proposed_equivalence else "related_locator",
                        permitted_operations=(("replace_surface", "add_attested_alias", "qualify", "locator_probe")
                                              if proposed_equivalence else ("locator_probe",)),
                        scope_kind="new_scope_proposal"))
                    if sum(item.facet_id == facet_id for item in candidates) >= per_facet_candidates:
                        break
        return tuple(candidates)
