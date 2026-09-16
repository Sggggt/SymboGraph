"""Replay materialized evidence scope from a frozen bounded packing input."""
from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session
from collections.abc import Sequence

from app.models import Chunk, ContextPackage, RetrievalTrace
from app.retrieval_control_contracts import (
    EvidenceCompletionPlan, EvidenceInterval, EvidenceIntervalCandidate, EvidenceScopeCoverage, control_hash,
)

PACKING_PROTOCOL = "whole_chunk_then_empty_package_raw_prefix_v3"
MAX_PACKING_CANDIDATES = 4096


def plan_interval_scope_completion(*, knowledge_base_id: str, coverage: EvidenceScopeCoverage,
        candidates: Sequence[EvidenceIntervalCandidate], budget: int | None = None) -> EvidenceCompletionPlan:
    """Minimum additive cost interval cover; a plan, never execution authority."""
    from bisect import bisect_right
    from collections import defaultdict
    from app.services.source_use import _merge_scope_intervals, _missing_scope_intervals
    from app.services.storage import raise_if_source_io_cancelled
    if coverage.mode != 'complete' or len(candidates) > MAX_PACKING_CANDIDATES:
        raise ValueError('scope_completion_mode_or_capacity_invalid')
    if budget is not None and (type(budget) is not int or budget < 0):
        raise ValueError('scope_completion_budget_invalid')
    if (len({item.id for item in candidates}) != len(candidates) or any(
            item.interval.knowledge_base_id != knowledge_base_id for item in candidates)
            or any(item.knowledge_base_id != knowledge_base_id for item in (*coverage.usable_intervals, *coverage.missing_intervals))):
        raise ValueError('scope_completion_candidate_scope_invalid')
    candidates = sorted(candidates,key=lambda item:item.id)
    input_hash = control_hash({'protocol_version':'interval_scope_completion_v1',
        'knowledge_base_id':knowledge_base_id,'coverage':coverage.model_dump(mode='json'),
        'candidates':[item.model_dump(mode='json') for item in candidates],'budget':budget})
    def result(status, selected=(), cost=None, uncovered=()):
        return EvidenceCompletionPlan(status=status,input_hash=input_hash,selected_ids=tuple(selected),
            total_cost=cost,uncovered_intervals=tuple(uncovered))
    if not coverage.scope_extent_known or coverage.state == 'unknown':
        return result('scope_unresolved')
    if coverage.state == 'satisfied':
        return result('already_covered',cost=0)
    required = _merge_scope_intervals((item.knowledge_base_id,item.document_version_id,item.start,item.end)
                                    for item in coverage.missing_intervals)
    available = _merge_scope_intervals((item.interval.knowledge_base_id,item.interval.document_version_id,
        item.interval.start,item.interval.end) for item in candidates)
    missing = _missing_scope_intervals(required,available)
    if missing or not required:
        return result('not_coverable',uncovered=(EvidenceInterval(knowledge_base_id=kb,document_version_id=version,start=start,end=end)
            for kb,version,start,end in missing))
    ranges, groups = defaultdict(list), defaultdict(list)
    for kb,version,start,end in required:
        ranges[(kb,version)].append((start,end))
    for candidate in candidates:
        groups[(knowledge_base_id,candidate.interval.document_version_id)].append(candidate)
    selected, total_cost = [], 0
    for namespace, needed in sorted(ranges.items()):
        raise_if_source_io_cancelled()
        group = groups[namespace]
        ends = [end for _,end in needed]
        first, terminal = needed[0][0], needed[-1][1]
        def advance(position):
            index = bisect_right(ends,position)
            return terminal if index == len(needed) else max(position,needed[index][0])
        destinations = {item.id:advance(item.interval.end) for item in group}
        positions = sorted({first,terminal,*destinations.values()})
        best, previous = {first:(0,0)}, {}
        for position in positions:
            raise_if_source_io_cancelled()
            if position == terminal or position not in best:
                continue
            for item in group:
                if not item.interval.start <= position < item.interval.end:
                    continue
                target = destinations[item.id]
                score = (best[position][0]+item.cost,best[position][1]+1)
                if target not in best or score < best[target]:
                    best[target], previous[target] = score, (position,item.id)
        if terminal not in best:
            raise ValueError('scope_completion_coverability_invariant_failed')
        path, position = [], terminal
        while position != first:
            position, cid = previous[position]
            path.append(cid)
        selected.extend(reversed(path))
        total_cost += best[terminal][0]
    return result('over_budget' if budget is not None and total_cost > budget else 'ready',selected,total_cost)


def audit_context_packing(
    db: Session, *, package: ContextPackage, trace: RetrievalTrace | None,
    for_update: bool = False, retained_chunk_ids: set[str] | None = None,
) -> list[str]:
    audit = (package.diagnostics_json or {}).get("token_budget_audit") or {}
    trace_hits = list(trace.result_chunk_ids_json or []) if trace else []
    if audit.get("packing_protocol") != PACKING_PROTOCOL:
        return [] if set(package.hit_chunk_ids_json or []) == set(trace_hits) else ["context_package_hit_trace_scope_mismatch"]
    from app.services.chunking import rough_token_count
    from app.services.context_graph import _fit_context_package_chunk

    candidates = audit.get("candidate_chunk_ids")
    cap = audit.get("selection_token_budget")
    if (not isinstance(candidates, list) or len(candidates) > MAX_PACKING_CANDIDATES
        or any(not isinstance(cid, str) or not cid for cid in candidates)
        or len(candidates) != len(set(candidates)) or not set(trace_hits).issubset(candidates)
        or type(cap) is not int or cap <= 0):
        return ["context_package_packing_input_invalid"]
    statement = select(Chunk).where(Chunk.id.in_(candidates)).order_by(Chunk.id)
    if for_update:
        statement = statement.with_for_update().execution_options(populate_existing=True)
    chunks = {chunk.id: chunk for chunk in db.scalars(statement)}
    if len(chunks) != len(candidates) or any(chunk.knowledge_base_id != package.knowledge_base_id or chunk.state != "active" for chunk in chunks.values()):
        return ["context_package_packing_source_scope_invalid"]
    expected, skipped = {}, []
    used = 0
    for cid in candidates:
        chunk = chunks[cid]
        remaining = cap - used
        fitted = _fit_context_package_chunk(chunk, token_limit=remaining, allow_clipping=not expected)
        if fitted is None:
            if rough_token_count(chunk.text) > remaining:
                skipped.append(cid)
            continue
        expected[cid] = fitted
        used += fitted.token_count
    items = (package.package_json or {}).get("chunks") or []
    actual = {item["chunk_id"]: item for item in items}
    retained = retained_chunk_ids or set()
    if (len(actual) != len(items) or not set(expected).issubset(actual)
        or not (set(actual) - set(expected)).issubset(retained)
        or [cid for cid in actual if cid in expected] != list(expected)
        or list(package.hit_chunk_ids_json or []) != [cid for cid in trace_hits if cid in expected]):
        return ["context_package_packing_selection_mismatch"]
    for cid, fitted in expected.items():
        item = actual[cid]
        if cid not in retained:
            valid = (item.get("content") == fitted.content and item.get("content_token_count") == fitted.token_count
                and item.get("content_clipped") is fitted.clipped and item.get("char_span") == fitted.char_span
                and item.get("raw_chunk_char_span") == fitted.raw_chunk_char_span)
        else:
            span = item.get("char_span") or []
            valid = (len(span) == 2 and span[0] == fitted.char_span[0] and span[1] >= fitted.char_span[1]
                and str(item.get("content") or "").startswith(fitted.content))
        if not valid:
            return ["context_package_packing_span_mismatch"]
    actual_tokens = sum(rough_token_count(item.get("content") or "") for item in items)
    if (audit.get("skipped_chunk_ids") != [cid for cid in skipped if cid not in actual]
        or audit.get("clipped_chunk_ids") != [item["chunk_id"] for item in items if item.get("content_clipped")]
        or audit.get("token_count") != actual_tokens or package.token_count != actual_tokens
        or audit.get("token_budget") != package.token_budget or actual_tokens > package.token_budget
        or audit.get("within_budget") is not True):
        return ["context_package_packing_budget_mismatch"]
    return []
