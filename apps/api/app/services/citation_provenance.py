from __future__ import annotations

import hashlib
from collections import Counter
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import (
    Chunk,
    ChunkCoordinate,
    ChunkRelationEdge,
    ChunkSpan,
    ChunkStructureMapping,
    ChunkStructureNode,
    ContextPackage,
    Document,
    DocumentVersion,
    GraphRetrievalStep,
    KnowledgeBase,
    RetrievalTrace,
)
from app.services.chunking import CHUNK_TEXT_HASH_PROTOCOL_VERSION, stable_hash, text_hash
from app.services.context_graph import (
    RAW_SPAN_TEXT_HASH_PROTOCOL_VERSION,
    ChunkSourceProvenanceError,
    SnapshotIntegrityVerifier,
    chunk_source_span,
)


CITATION_PROVENANCE_PROTOCOL_VERSION = "citation_provenance_db_gate_v1"
CITATION_PROVENANCE_SESSION_HASH_PROTOCOL_VERSION = (
    "citation_provenance_session_hash_v1"
)
CITATION_ANSWER_SESSION_BINDING_PROTOCOL_VERSION = (
    "citation_answer_session_binding_v1"
)


def _raw_text_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _pair(value: Any) -> list[int] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    if any(isinstance(item, bool) or not isinstance(item, int) for item in value):
        return None
    return [int(value[0]), int(value[1])]


def _page_pair(value: Any) -> list[int | None] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    if any(
        item is not None and (isinstance(item, bool) or not isinstance(item, int))
        for item in value
    ):
        return None
    return [int(item) if item is not None else None for item in value]


def _as_string_list(value: Any) -> list[str] | None:
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item for item in value
    ):
        return None
    return list(value)


def _append(reasons: list[str], reason: str, condition: bool = True) -> None:
    if condition and reason not in reasons:
        reasons.append(reason)


def _structure_payload_forged(
    payload: dict[str, Any],
    *,
    node: ChunkStructureNode | None,
    mapping: ChunkStructureMapping | None,
    allowed_types: set[str] | None = None,
) -> bool:
    if node is None or mapping is None:
        return True
    if allowed_types is not None and node.node_type not in allowed_types:
        return True
    expected = {
        "node_id": node.id,
        "node_type": node.node_type,
        "title": node.title,
        "path": node.path,
        "depth": node.depth,
        "page_number": node.page_number,
        "bbox": node.bbox_json or {},
        "layout": node.layout_json or {},
        "mapping_role": mapping.mapping_role,
        "coverage_ratio": mapping.coverage_ratio,
        "span_overlap": mapping.span_overlap,
        "bbox_iou": mapping.bbox_iou,
        "path_match": mapping.path_match,
        "mapping_weight": mapping.mapping_weight,
        "mapping_protocol_version": mapping.mapping_protocol_version,
        "mapping_diagnostics": mapping.metadata_json or {},
    }
    return any(payload.get(field) != value for field, value in expected.items())


def _load(
    db: Session,
    model: type[Any],
    object_id: str | None,
    *,
    for_update: bool,
) -> Any | None:
    if not object_id:
        return None
    statement = select(model).where(model.id == object_id)
    if for_update and db.get_bind().dialect.name == "postgresql":
        statement = statement.with_for_update().execution_options(
            populate_existing=True
        )
    return db.scalar(statement)


def _package_binding_payload(package: ContextPackage) -> dict[str, Any]:
    return {
        "id": package.id,
        "knowledge_base_id": package.knowledge_base_id,
        "retrieval_trace_id": package.retrieval_trace_id,
        "query": package.query,
        "hit_chunk_ids": package.hit_chunk_ids_json or [],
        "restored_chunk_ids": package.restored_chunk_ids_json or [],
        "bridge_chunk_ids": package.bridge_chunk_ids_json or [],
        "parent_structure_node_ids": package.parent_structure_node_ids_json or [],
        "graph_path_ids": package.graph_path_ids_json or [],
        "why_selected": package.why_selected_json or {},
        "package": package.package_json or {},
        "citation_spans": package.citation_spans_json or [],
        "runtime_settings_hash": package.runtime_settings_hash,
        "profile_hash": package.profile_hash,
    }


def _trace_binding_payload(trace: RetrievalTrace) -> dict[str, Any]:
    return {
        "id": trace.id,
        "knowledge_base_id": trace.knowledge_base_id,
        "query": trace.query,
        "retrieval_mode": trace.retrieval_mode,
        "result_chunk_ids": trace.result_chunk_ids_json or [],
        "path_labels": trace.path_labels_json or [],
        "chunk_scope_hash": trace.chunk_scope_hash,
        "structure_graph_hash": trace.structure_graph_hash,
        "chunk_relation_graph_hash": trace.chunk_relation_graph_hash,
        "rq_membership_hash": trace.rq_membership_hash,
        "mid_concept_hash": trace.mid_concept_hash,
        "coarse_concept_hash": trace.coarse_concept_hash,
        "runtime_settings_hash": trace.runtime_settings_hash,
        "agent_operating_envelope_hash": trace.agent_operating_envelope_hash,
        "edge_distance_protocol_hash": trace.edge_distance_protocol_hash,
        "edge_projection_protocol_hash": trace.edge_projection_protocol_hash,
        "traversal_protocol_hash": trace.traversal_protocol_hash,
    }


def _package_chunks(package: ContextPackage) -> tuple[dict[str, dict[str, Any]], set[str]]:
    chunks = list((package.package_json or {}).get("chunks") or [])
    counts = Counter(
        str(item.get("chunk_id"))
        for item in chunks
        if isinstance(item, dict) and item.get("chunk_id")
    )
    duplicates = {chunk_id for chunk_id, count in counts.items() if count != 1}
    return {
        str(item["chunk_id"]): item
        for item in chunks
        if isinstance(item, dict) and item.get("chunk_id")
    }, duplicates


def _contexts_by_chunk(
    contexts: list[dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], set[str]]:
    counts = Counter(
        str(item.get("chunk_id"))
        for item in contexts
        if isinstance(item, dict) and item.get("chunk_id")
    )
    duplicates = {chunk_id for chunk_id, count in counts.items() if count != 1}
    return {
        str(item["chunk_id"]): item
        for item in contexts
        if isinstance(item, dict) and item.get("chunk_id")
    }, duplicates


def _retrieval_steps(
    db: Session,
    *,
    package: ContextPackage,
    trace: RetrievalTrace | None,
    for_update: bool,
) -> list[GraphRetrievalStep]:
    if trace is None:
        return []
    statement = (
        select(GraphRetrievalStep)
        .where(
            GraphRetrievalStep.retrieval_trace_id == trace.id,
            GraphRetrievalStep.knowledge_base_id == package.knowledge_base_id,
        )
        .order_by(GraphRetrievalStep.step_index.asc())
    )
    if for_update and db.get_bind().dialect.name == "postgresql":
        statement = statement.with_for_update().execution_options(
            populate_existing=True
        )
    return list(db.scalars(statement).all())


def _active_bridge_proof(
    db: Session,
    *,
    knowledge_base_id: str,
    hit_chunk_ids: set[str],
    bridge_chunk_id: str,
    for_update: bool,
) -> bool:
    if not hit_chunk_ids:
        return False
    statement = select(ChunkRelationEdge.id).where(
        ChunkRelationEdge.knowledge_base_id == knowledge_base_id,
        ChunkRelationEdge.is_bridge.is_(True),
        (
            (
                ChunkRelationEdge.source_chunk_id.in_(hit_chunk_ids)
                & (ChunkRelationEdge.target_chunk_id == bridge_chunk_id)
            )
            | (
                ChunkRelationEdge.target_chunk_id.in_(hit_chunk_ids)
                & (ChunkRelationEdge.source_chunk_id == bridge_chunk_id)
            )
        ),
    )
    if for_update and db.get_bind().dialect.name == "postgresql":
        statement = statement.with_for_update()
    return db.scalar(statement) is not None


def _graph_path_proof(trace: RetrievalTrace, chunk_id: str) -> bool:
    for label in trace.path_labels_json or []:
        if not isinstance(label, dict):
            continue
        if str(label.get("chunk_id") or "") == chunk_id:
            return True
        if chunk_id in {str(value) for value in (label.get("path") or [])}:
            return True
    return False


def _neighbor_proof(
    db: Session,
    *,
    hit_chunk_ids: set[str],
    restored_chunk_id: str,
    for_update: bool,
) -> bool:
    for hit_id in sorted(hit_chunk_ids):
        hit = _load(db, Chunk, hit_id, for_update=for_update)
        if hit is not None and restored_chunk_id in {
            hit.previous_chunk_id,
            hit.next_chunk_id,
        }:
            return True
    return False


def _citation_span_rows(package: ContextPackage, chunk_id: str) -> list[dict[str, Any]]:
    return [
        item
        for item in (package.citation_spans_json or [])
        if isinstance(item, dict) and str(item.get("chunk_id") or "") == chunk_id
    ]


def audit_citation_provenance(
    db: Session,
    *,
    knowledge_base_id: str,
    package: ContextPackage,
    citations: list[dict[str, Any]],
    contexts: list[dict[str, Any]],
    for_update: bool = False,
) -> dict[str, Any]:
    """Validate every citation against current persisted provenance.

    The routine never accepts a caller-provided address as fact.  It rebuilds
    the canonical source span from PostgreSQL and the immutable source
    snapshot, then binds the result to the persisted retrieval trace and
    context package.  ``for_update`` is used immediately before answer-audit
    persistence so lifecycle rows cannot change between the final replay and
    commit on PostgreSQL.
    """

    if for_update:
        db.flush()
    persisted_package = _load(
        db,
        ContextPackage,
        str(getattr(package, "id", "") or ""),
        for_update=for_update,
    )
    package_row = persisted_package or package
    knowledge_base = _load(
        db,
        KnowledgeBase,
        knowledge_base_id,
        for_update=for_update,
    )
    trace = _load(
        db,
        RetrievalTrace,
        str(getattr(package_row, "retrieval_trace_id", "") or ""),
        for_update=for_update,
    )
    retrieval_steps = _retrieval_steps(
        db,
        package=package_row,
        trace=trace,
        for_update=for_update,
    )
    structure_steps = [
        step
        for step in retrieval_steps
        if step.layer == "structure"
        and step.action_type == "restore_context_package"
    ]
    structure_step = structure_steps[0] if len(structure_steps) == 1 else None
    package_chunks, duplicate_package_chunks = _package_chunks(package_row)
    contexts_by_chunk, duplicate_context_chunks = _contexts_by_chunk(contexts)
    package_binding_hash = stable_hash(_package_binding_payload(package_row))
    trace_binding_hash = stable_hash(_trace_binding_payload(trace)) if trace else None
    global_reasons: list[str] = []
    _append(global_reasons, "context_package_not_persisted", persisted_package is None)
    _append(global_reasons, "knowledge_base_missing", knowledge_base is None)
    _append(
        global_reasons,
        "context_package_knowledge_base_mismatch",
        package_row.knowledge_base_id != knowledge_base_id,
    )
    _append(global_reasons, "retrieval_trace_missing", trace is None)
    if trace is not None:
        _append(
            global_reasons,
            "retrieval_trace_knowledge_base_mismatch",
            trace.knowledge_base_id != knowledge_base_id,
        )
        _append(
            global_reasons,
            "context_package_trace_query_mismatch",
            package_row.query != trace.query,
        )
        _append(
            global_reasons,
            "context_package_runtime_settings_hash_mismatch",
            not package_row.runtime_settings_hash
            or package_row.runtime_settings_hash != trace.runtime_settings_hash,
        )
    _append(
        global_reasons,
        "structure_restoration_step_invalid",
        len(structure_steps) != 1,
    )
    if structure_step is not None:
        output = structure_step.output_json or {}
        input_payload = structure_step.input_json or {}
        _append(
            global_reasons,
            "structure_step_context_package_mismatch",
            str(output.get("context_package_id") or "") != package_row.id,
        )
        _append(
            global_reasons,
            "structure_step_hit_scope_mismatch",
            list(input_payload.get("hit_chunk_ids") or [])
            != list(package_row.hit_chunk_ids_json or [])
            or list(structure_step.selected_topk_ids_json or [])
            != list(package_row.hit_chunk_ids_json or [])
            or list(output.get("restored_chunk_ids") or [])
            != list(package_row.restored_chunk_ids_json or [])
            or list(output.get("bridge_chunk_ids") or [])
            != list(package_row.bridge_chunk_ids_json or [])
            or list(output.get("graph_path_ids") or [])
            != list(package_row.graph_path_ids_json or [])
            or list(structure_step.expanded_edge_ids_json or [])
            != list(package_row.graph_path_ids_json or []),
        )

    hit_ids = {str(value) for value in (package_row.hit_chunk_ids_json or [])}
    restored_ids = {
        str(value) for value in (package_row.restored_chunk_ids_json or [])
    }
    bridge_ids = {str(value) for value in (package_row.bridge_chunk_ids_json or [])}
    trace_result_ids = {
        str(value) for value in ((trace.result_chunk_ids_json if trace else None) or [])
    }
    _append(
        global_reasons,
        "context_package_hit_trace_scope_mismatch",
        hit_ids != trace_result_ids,
    )
    expected_graph_path_ids = list(
        dict.fromkeys(
            str(edge_id)
            for label in ((trace.path_labels_json if trace else None) or [])
            if isinstance(label, dict)
            for edge_id in (label.get("path_edge_ids") or [])
            if str(edge_id)
        )
    )
    _append(
        global_reasons,
        "context_package_graph_path_scope_mismatch",
        list(package_row.graph_path_ids_json or []) != expected_graph_path_ids,
    )
    step_path_labels = [
        label
        for step in retrieval_steps
        if step.layer != "structure"
        for label in (
            (step.diagnostics_json or {}).get("path_labels")
            if isinstance(step.diagnostics_json, dict)
            else []
        )
        or []
    ]
    _append(
        global_reasons,
        "retrieval_trace_executor_path_mismatch",
        step_path_labels != list((trace.path_labels_json if trace else None) or []),
    )
    snapshot_verifier = SnapshotIntegrityVerifier()
    audits: list[dict[str, Any]] = []

    for citation_index, citation in enumerate(citations, start=1):
        reasons = list(global_reasons)
        if not isinstance(citation, dict):
            citation = {}
            _append(reasons, "citation_payload_not_object")
        source_span = citation.get("source_span")
        if not isinstance(source_span, dict):
            source_span = {}
            _append(reasons, "citation_source_span_not_object")
        chunk_id = str(citation.get("chunk_id") or "")
        _append(reasons, "citation_chunk_id_missing", not chunk_id)
        _append(
            reasons,
            "citation_source_chunk_mismatch",
            str(source_span.get("chunk_id") or "") != chunk_id,
        )
        _append(
            reasons,
            "duplicate_context_package_chunk",
            chunk_id in duplicate_package_chunks,
        )
        _append(
            reasons,
            "duplicate_context_chunk",
            chunk_id in duplicate_context_chunks,
        )
        package_item = package_chunks.get(chunk_id)
        context = contexts_by_chunk.get(chunk_id)
        _append(reasons, "citation_chunk_outside_context_package", package_item is None)
        _append(reasons, "citation_chunk_outside_context_input", context is None)
        chunk = _load(db, Chunk, chunk_id, for_update=for_update)
        _append(reasons, "citation_chunk_missing", chunk is None)

        canonical_source_span: dict[str, Any] = {}
        raw_span: list[int] | None = None
        expected_text = ""
        structure_support: dict[str, Any] = {}
        if chunk is not None:
            document = _load(db, Document, chunk.document_id, for_update=for_update)
            version = _load(
                db,
                DocumentVersion,
                chunk.document_version_id,
                for_update=for_update,
            )
            _append(
                reasons,
                "citation_chunk_knowledge_base_mismatch",
                chunk.knowledge_base_id != knowledge_base_id,
            )
            _append(reasons, "citation_chunk_inactive", chunk.state != "active")
            _append(reasons, "citation_document_missing", document is None)
            _append(reasons, "citation_document_version_missing", version is None)
            if document is not None:
                _append(
                    reasons,
                    "citation_document_knowledge_base_mismatch",
                    document.knowledge_base_id != knowledge_base_id,
                )
                _append(reasons, "citation_document_inactive", not document.is_active)
                _append(
                    reasons,
                    "citation_document_identity_mismatch",
                    str(citation.get("document_id") or "") != document.id,
                )
                if version is not None:
                    _append(
                        reasons,
                        "citation_document_checksum_mismatch",
                        document.checksum != version.checksum,
                    )
            if version is not None:
                _append(
                    reasons,
                    "citation_document_version_inactive",
                    not version.is_active,
                )
                _append(
                    reasons,
                    "citation_document_version_document_mismatch",
                    version.document_id != chunk.document_id,
                )
                _append(
                    reasons,
                    "citation_document_version_number_mismatch",
                    int(version.version) != int(chunk.chunk_version),
                )
                _append(
                    reasons,
                    "citation_document_version_identity_mismatch",
                    str(citation.get("document_version_id") or "") != version.id
                    or str(source_span.get("document_version_id") or "")
                    != version.id,
                )
            _append(
                reasons,
                "citation_chunk_text_hash_mismatch",
                not chunk.text_hash or chunk.text_hash != text_hash(chunk.text or ""),
            )
            raw_span = _pair(
                (package_item or {}).get("char_span")
                if package_item is not None
                else source_span.get("char_span")
            )
            _append(reasons, "citation_char_span_invalid", raw_span is None)
            if raw_span is not None:
                start, end = raw_span
                _append(
                    reasons,
                    "citation_char_span_out_of_chunk_bounds",
                    start < int(chunk.char_start)
                    or end <= start
                    or end > int(chunk.char_end),
                )
                relative_start = start - int(chunk.char_start)
                relative_end = end - int(chunk.char_start)
                _append(
                    reasons,
                    "citation_char_span_out_of_text_bounds",
                    relative_start < 0
                    or relative_end <= relative_start
                    or relative_end > len(chunk.text or ""),
                )
                if (
                    relative_start >= 0
                    and relative_end > relative_start
                    and relative_end <= len(chunk.text or "")
                ):
                    expected_text = (chunk.text or "")[relative_start:relative_end]
            span_statement = select(ChunkSpan).where(
                ChunkSpan.chunk_id == chunk.id,
                ChunkSpan.document_version_id == chunk.document_version_id,
                ChunkSpan.span_type == "raw_text",
            )
            if for_update and db.get_bind().dialect.name == "postgresql":
                span_statement = span_statement.with_for_update().execution_options(
                    populate_existing=True
                )
            span_rows = list(db.scalars(span_statement).all())
            exact_span_rows = [
                row
                for row in span_rows
                if int(row.char_start) == int(chunk.char_start)
                and int(row.char_end) == int(chunk.char_end)
                and int(row.token_start) == int(chunk.token_start)
                and int(row.token_end) == int(chunk.token_end)
                and (row.metadata_json or {}).get("text_hash") == chunk.text_hash
            ]
            _append(
                reasons,
                "citation_raw_chunk_span_record_invalid",
                len(exact_span_rows) != 1,
            )
            mapping_statement = (
                select(ChunkStructureMapping, ChunkStructureNode)
                .join(
                    ChunkStructureNode,
                    ChunkStructureMapping.structure_node_id
                    == ChunkStructureNode.id,
                )
                .where(ChunkStructureMapping.chunk_id == chunk.id)
            )
            if for_update and db.get_bind().dialect.name == "postgresql":
                mapping_statement = mapping_statement.with_for_update().execution_options(
                    populate_existing=True
                )
            mapping_rows = db.execute(mapping_statement).all()
            valid_mapping_rows = [
                (mapping, node)
                for mapping, node in mapping_rows
                if mapping.document_version_id == chunk.document_version_id
                and node.knowledge_base_id == knowledge_base_id
                and node.document_id == chunk.document_id
                and node.document_version_id == chunk.document_version_id
            ]
            _append(
                reasons,
                "citation_structure_mapping_missing",
                not valid_mapping_rows,
            )
            structure_support = {
                "mapping_ids": sorted(mapping.id for mapping, _node in valid_mapping_rows),
                "node_ids": sorted(node.id for _mapping, node in valid_mapping_rows),
            }
            structure_nodes_by_id = {
                node.id: node for _mapping, node in valid_mapping_rows
            }
            structure_mappings_by_node_id = {
                node.id: mapping for mapping, node in valid_mapping_rows
            }
            coordinate_statement = select(ChunkCoordinate).where(
                ChunkCoordinate.chunk_id == chunk.id
            )
            if for_update and db.get_bind().dialect.name == "postgresql":
                coordinate_statement = (
                    coordinate_statement.with_for_update().execution_options(
                        populate_existing=True
                    )
                )
            # Lock every persisted coordinate before the canonical helper picks
            # the highest-confidence address used for page/bbox validation.
            list(db.scalars(coordinate_statement).all())
            try:
                canonical_source_span = chunk_source_span(
                    db,
                    chunk,
                    context_package_id=package_row.id,
                    retrieval_trace_id=package_row.retrieval_trace_id,
                    snapshot_verifier=snapshot_verifier,
                )
            except ChunkSourceProvenanceError:
                _append(reasons, "citation_source_snapshot_provenance_invalid")
                canonical_source_span = {}
            if canonical_source_span and raw_span is not None:
                canonical_source_span["char_span"] = list(raw_span)
                canonical_source_span["raw_chunk_char_span"] = [
                    int(chunk.char_start),
                    int(chunk.char_end),
                ]
                canonical_source_span["raw_span_text_hash"] = _raw_text_hash(
                    expected_text
                )
        else:
            structure_nodes_by_id = {}
            structure_mappings_by_node_id = {}

        if package_item is not None:
            item_source_span = package_item.get("source_span")
            _append(
                reasons,
                "context_package_source_span_not_object",
                not isinstance(item_source_span, dict),
            )
            _append(
                reasons,
                "context_package_chunk_id_mismatch",
                str(package_item.get("chunk_id") or "") != chunk_id,
            )
            _append(
                reasons,
                "context_package_document_id_mismatch",
                chunk is not None
                and str(package_item.get("document_id") or "") != chunk.document_id,
            )
            _append(
                reasons,
                "context_package_document_version_id_mismatch",
                chunk is not None
                and str(package_item.get("document_version_id") or "")
                != chunk.document_version_id,
            )
            _append(
                reasons,
                "context_package_content_mismatch",
                not expected_text or str(package_item.get("content") or "") != expected_text,
            )
            _append(
                reasons,
                "context_package_raw_span_hash_mismatch",
                str(package_item.get("raw_span_text_hash") or "")
                != _raw_text_hash(expected_text),
            )
            _append(
                reasons,
                "context_package_chunk_hash_mismatch",
                chunk is not None
                and (
                    package_item.get("chunk_text_hash_protocol_version")
                    != CHUNK_TEXT_HASH_PROTOCOL_VERSION
                    or package_item.get("chunk_text_hash") != chunk.text_hash
                ),
            )
            _append(
                reasons,
                "context_package_source_span_mismatch",
                bool(canonical_source_span)
                and any(
                    (item_source_span or {}).get(field)
                    != canonical_source_span.get(field)
                    for field in canonical_source_span
                    if field != "verification_id"
                ),
            )
            _append(
                reasons,
                "citation_source_span_mismatch",
                bool(canonical_source_span)
                and any(
                    source_span.get(field) != canonical_source_span.get(field)
                    for field in canonical_source_span
                    if field != "verification_id"
                ),
            )
            _append(
                reasons,
                "citation_raw_span_hash_protocol_mismatch",
                source_span.get("raw_span_text_hash_protocol_version")
                != RAW_SPAN_TEXT_HASH_PROTOCOL_VERSION,
            )
            _append(
                reasons,
                "context_package_structure_node_ids_mismatch",
                bool(canonical_source_span)
                and _as_string_list(package_item.get("structure_node_ids"))
                != _as_string_list(canonical_source_span.get("structure_node_ids")),
            )
            _append(
                reasons,
                "context_package_page_range_mismatch",
                bool(canonical_source_span)
                and _page_pair(package_item.get("page_range"))
                != _page_pair(canonical_source_span.get("page_range")),
            )
            _append(
                reasons,
                "context_package_section_path_mismatch",
                bool(canonical_source_span)
                and package_item.get("section_path")
                != canonical_source_span.get("section_path"),
            )
            _append(
                reasons,
                "context_package_bbox_mismatch",
                bool(canonical_source_span)
                and (package_item.get("bbox") or {})
                != (canonical_source_span.get("bbox") or {}),
            )
            closure = package_item.get("structure_closure")
            _append(
                reasons,
                "context_package_structure_closure_missing",
                not isinstance(closure, dict),
            )
            if isinstance(closure, dict) and chunk is not None:
                _append(
                    reasons,
                    "context_package_previous_closure_mismatch",
                    closure.get("previous_chunk_id") != chunk.previous_chunk_id,
                )
                _append(
                    reasons,
                    "context_package_next_closure_mismatch",
                    closure.get("next_chunk_id") != chunk.next_chunk_id,
                )
                for neighbor_field, neighbor_id in (
                    ("previous", chunk.previous_chunk_id),
                    ("next", chunk.next_chunk_id),
                ):
                    if neighbor_id is None:
                        continue
                    neighbor = _load(
                        db,
                        Chunk,
                        neighbor_id,
                        for_update=for_update,
                    )
                    _append(
                        reasons,
                        f"context_package_{neighbor_field}_closure_invalid",
                        neighbor is None
                        or neighbor.state != "active"
                        or neighbor.knowledge_base_id != knowledge_base_id
                        or neighbor.document_id != chunk.document_id
                        or neighbor.document_version_id
                        != chunk.document_version_id,
                    )
                _append(
                    reasons,
                    "context_package_parent_section_mismatch",
                    closure.get("parent_section") != package_item.get("parent_section"),
                )
                _append(
                    reasons,
                    "context_package_bridge_closure_mismatch",
                    {
                        str(value)
                        for value in (closure.get("bridge_chunk_ids") or [])
                    }
                    != bridge_ids,
                )
                for bridge_chunk_id in sorted(bridge_ids):
                    bridge_chunk = _load(
                        db,
                        Chunk,
                        bridge_chunk_id,
                        for_update=for_update,
                    )
                    _append(
                        reasons,
                        "context_package_bridge_closure_invalid",
                        bridge_chunk is None
                        or bridge_chunk.state != "active"
                        or bridge_chunk.knowledge_base_id != knowledge_base_id
                        or not _active_bridge_proof(
                            db,
                            knowledge_base_id=knowledge_base_id,
                            hit_chunk_ids=hit_ids,
                            bridge_chunk_id=bridge_chunk_id,
                            for_update=for_update,
                        ),
                    )
                for closure_field, allowed_types in (
                    ("same_page_region", {"page", "region"}),
                    ("table_formula_caption", {"table", "formula", "caption"}),
                    ("code_blocks", {"code_block"}),
                ):
                    closure_nodes = closure.get(closure_field)
                    if not isinstance(closure_nodes, list):
                        _append(
                            reasons,
                            f"context_package_{closure_field}_invalid",
                        )
                        continue
                    for closure_node in closure_nodes:
                        if not isinstance(closure_node, dict):
                            _append(
                                reasons,
                                f"context_package_{closure_field}_invalid",
                            )
                            continue
                        node = structure_nodes_by_id.get(
                            str(closure_node.get("node_id") or "")
                        )
                        _append(
                            reasons,
                            f"context_package_{closure_field}_forged",
                            _structure_payload_forged(
                                closure_node,
                                node=node,
                                mapping=structure_mappings_by_node_id.get(
                                    str(closure_node.get("node_id") or "")
                                ),
                                allowed_types=allowed_types,
                            ),
                        )
                parent_section = closure.get("parent_section")
                if parent_section is not None:
                    parent_node = (
                        structure_nodes_by_id.get(
                            str(parent_section.get("node_id") or "")
                        )
                        if isinstance(parent_section, dict)
                        else None
                    )
                    _append(
                        reasons,
                        "context_package_parent_section_forged",
                        _structure_payload_forged(
                            parent_section,
                            node=parent_node,
                            mapping=structure_mappings_by_node_id.get(
                                str(parent_section.get("node_id") or "")
                            ),
                            allowed_types={"section", "heading", "document"},
                        ),
                    )

            structure_nodes = package_item.get("structure_nodes")
            _append(
                reasons,
                "context_package_structure_nodes_invalid",
                not isinstance(structure_nodes, list),
            )
            if isinstance(structure_nodes, list):
                payload_node_ids = [
                    str(item.get("node_id") or "")
                    for item in structure_nodes
                    if isinstance(item, dict)
                ]
                _append(
                    reasons,
                    "context_package_structure_nodes_scope_mismatch",
                    len(payload_node_ids) != len(structure_nodes)
                    or len(payload_node_ids) != len(set(payload_node_ids))
                    or set(payload_node_ids) != set(structure_nodes_by_id),
                )
                for structure_node in structure_nodes:
                    if not isinstance(structure_node, dict):
                        continue
                    node = structure_nodes_by_id.get(
                        str(structure_node.get("node_id") or "")
                    )
                    _append(
                        reasons,
                        "context_package_structure_node_forged",
                        _structure_payload_forged(
                            structure_node,
                            node=node,
                            mapping=structure_mappings_by_node_id.get(
                                str(structure_node.get("node_id") or "")
                            ),
                        ),
                    )

            role = str(package_item.get("role") or "")
            _append(reasons, "context_package_role_missing", not role)
            if role == "hit":
                _append(reasons, "citation_hit_membership_missing", chunk_id not in hit_ids)
                _append(
                    reasons,
                    "citation_role_scope_overlap",
                    chunk_id in restored_ids or chunk_id in bridge_ids,
                )
                _append(
                    reasons,
                    "citation_hit_trace_result_missing",
                    chunk_id not in trace_result_ids,
                )
            elif role == "bridge":
                _append(
                    reasons,
                    "citation_role_scope_overlap",
                    chunk_id in hit_ids,
                )
                _append(
                    reasons,
                    "citation_bridge_membership_missing",
                    chunk_id not in bridge_ids or chunk_id not in restored_ids,
                )
                _append(
                    reasons,
                    "citation_bridge_support_missing",
                    not _active_bridge_proof(
                        db,
                        knowledge_base_id=knowledge_base_id,
                        hit_chunk_ids=hit_ids,
                        bridge_chunk_id=chunk_id,
                        for_update=for_update,
                    ),
                )
            elif role == "graph_path":
                _append(
                    reasons,
                    "citation_role_scope_overlap",
                    chunk_id in hit_ids or chunk_id in bridge_ids,
                )
                _append(
                    reasons,
                    "citation_graph_path_membership_missing",
                    chunk_id not in restored_ids,
                )
                _append(
                    reasons,
                    "citation_graph_path_support_missing",
                    trace is None or not _graph_path_proof(trace, chunk_id),
                )
            elif role == "restored_context":
                _append(
                    reasons,
                    "citation_role_scope_overlap",
                    chunk_id in hit_ids or chunk_id in bridge_ids,
                )
                _append(
                    reasons,
                    "citation_restored_membership_missing",
                    chunk_id not in restored_ids,
                )
                _append(
                    reasons,
                    "citation_neighbor_support_missing",
                    not _neighbor_proof(
                        db,
                        hit_chunk_ids=hit_ids,
                        restored_chunk_id=chunk_id,
                        for_update=for_update,
                    ),
                )
            else:
                _append(reasons, "context_package_role_invalid")

            why_selected = package_item.get("why_selected")
            _append(
                reasons,
                "context_package_why_selected_missing",
                not isinstance(why_selected, dict),
            )
            _append(
                reasons,
                "context_package_why_selected_mismatch",
                (package_row.why_selected_json or {}).get(chunk_id) != why_selected,
            )
            if isinstance(why_selected, dict):
                path_edge_ids = {
                    str(value) for value in (why_selected.get("path_edge_ids") or [])
                }
                _append(
                    reasons,
                    "citation_graph_edge_support_outside_package",
                    not path_edge_ids.issubset(
                        {str(value) for value in (package_row.graph_path_ids_json or [])}
                    ),
                )

        if context is not None and package_item is not None:
            metadata = context.get("metadata")
            _append(reasons, "citation_context_metadata_not_object", not isinstance(metadata, dict))
            _append(
                reasons,
                "citation_context_content_mismatch",
                str(context.get("content") or "") != str(package_item.get("content") or ""),
            )
            if isinstance(metadata, dict):
                _append(
                    reasons,
                    "citation_context_package_id_mismatch",
                    str(metadata.get("context_package_id") or "") != package_row.id,
                )
                _append(
                    reasons,
                    "citation_context_source_span_mismatch",
                    metadata.get("source_span") != package_item.get("source_span"),
                )
                _append(
                    reasons,
                    "citation_context_structure_closure_mismatch",
                    metadata.get("structure_closure")
                    != package_item.get("structure_closure"),
                )

        span_rows = _citation_span_rows(package_row, chunk_id)
        _append(reasons, "context_package_citation_span_missing", len(span_rows) != 1)
        if len(span_rows) == 1 and package_item is not None:
            package_span = span_rows[0]
            _append(
                reasons,
                "context_package_citation_span_mismatch",
                any(
                    package_span.get(field)
                    != (package_item.get("source_span") or {}).get(field)
                    for field in (
                        "chunk_id",
                        "document_version_id",
                        "char_span",
                        "page_range",
                        "section_path",
                        "structure_node_ids",
                        "bbox",
                        "context_package_id",
                        "retrieval_trace_id",
                        "source_checksum",
                        "chunk_text_hash",
                        "raw_span_text_hash",
                    )
                ),
            )

        _append(
            reasons,
            "citation_context_package_id_mismatch",
            str(source_span.get("context_package_id") or "") != package_row.id,
        )
        if canonical_source_span:
            canonical_page_range = _page_pair(
                canonical_source_span.get("page_range")
            )
            _append(
                reasons,
                "citation_page_number_mismatch",
                citation.get("page_number")
                != (
                    canonical_page_range[0]
                    if canonical_page_range is not None
                    else None
                ),
            )
            _append(
                reasons,
                "citation_section_mismatch",
                citation.get("section")
                != canonical_source_span.get("section_path"),
            )
            _append(
                reasons,
                "citation_source_path_mismatch",
                citation.get("source_path")
                != canonical_source_span.get("source_path"),
            )
        _append(
            reasons,
            "citation_retrieval_trace_id_mismatch",
            str(citation.get("retrieval_trace_id") or "")
            != str(package_row.retrieval_trace_id or "")
            or str(source_span.get("retrieval_trace_id") or "")
            != str(package_row.retrieval_trace_id or ""),
        )
        if raw_span is not None:
            _append(
                reasons,
                "citation_source_char_span_mismatch",
                _pair(source_span.get("char_span")) != raw_span,
            )
        reasons = sorted(set(reasons))
        identity = {
            "protocol_version": CITATION_PROVENANCE_PROTOCOL_VERSION,
            "citation_index": citation_index,
            "knowledge_base_id": knowledge_base_id,
            "context_package_id": package_row.id,
            "retrieval_trace_id": package_row.retrieval_trace_id,
            "chunk_id": chunk_id or None,
            "document_id": chunk.document_id if chunk is not None else None,
            "document_version_id": (
                chunk.document_version_id if chunk is not None else None
            ),
            "char_span": raw_span,
            "raw_span_text_hash": _raw_text_hash(expected_text),
            "chunk_text_hash": chunk.text_hash if chunk is not None else None,
            "source_checksum": canonical_source_span.get("source_checksum"),
            "structure_support": structure_support,
            "package_binding_hash": package_binding_hash,
            "trace_binding_hash": trace_binding_hash,
            "reasons": reasons,
        }
        audits.append(
            {
                **identity,
                "valid": not reasons,
                "provenance_hash": stable_hash(identity),
                "fail_closed": True,
                "llm_override_allowed": False,
            }
        )

    session_identity = {
        "protocol_version": CITATION_PROVENANCE_SESSION_HASH_PROTOCOL_VERSION,
        "knowledge_base_id": knowledge_base_id,
        "context_package_id": package_row.id,
        "retrieval_trace_id": package_row.retrieval_trace_id,
        "package_binding_hash": package_binding_hash,
        "trace_binding_hash": trace_binding_hash,
        "citation_provenance_hashes": [item["provenance_hash"] for item in audits],
    }
    return {
        "protocol_version": CITATION_PROVENANCE_PROTOCOL_VERSION,
        "session_hash_protocol_version": (
            CITATION_PROVENANCE_SESSION_HASH_PROTOCOL_VERSION
        ),
        "provenance_session_hash": stable_hash(session_identity),
        "package_binding_hash": package_binding_hash,
        "trace_binding_hash": trace_binding_hash,
        "citation_count": len(audits),
        "citation_set_present": bool(audits),
        "valid_count": sum(bool(item["valid"]) for item in audits),
        "invalid_count": sum(not bool(item["valid"]) for item in audits),
        "all_valid": bool(audits) and all(bool(item["valid"]) for item in audits),
        "fail_closed": True,
        "audits": audits,
    }


def replay_citation_provenance_for_persistence(
    db: Session,
    *,
    knowledge_base_id: str,
    package: ContextPackage,
    citations: list[dict[str, Any]],
    contexts: list[dict[str, Any]],
    expected_session_hash: str,
) -> dict[str, Any]:
    lock_backend = db.get_bind().dialect.name
    replay = audit_citation_provenance(
        db,
        knowledge_base_id=knowledge_base_id,
        package=package,
        citations=citations,
        contexts=contexts,
        for_update=True,
    )
    replay["matches_pre_entailment_session_hash"] = (
        bool(expected_session_hash)
        and replay["provenance_session_hash"] == expected_session_hash
    )
    replay["persistence_gate_passed"] = bool(
        replay["all_valid"] and replay["matches_pre_entailment_session_hash"]
    )
    replay["transactional_replay"] = True
    replay["lock_backend"] = lock_backend
    replay["rows_locked"] = lock_backend == "postgresql"
    return replay


def citation_answer_session_binding_hash(
    *,
    knowledge_base_id: str,
    qa_session_id: str,
    question: str,
    answer: str,
    context_package_id: str,
    retrieval_trace_id: str | None,
    provenance_session_hash: str,
) -> str:
    return stable_hash(
        {
            "protocol_version": CITATION_ANSWER_SESSION_BINDING_PROTOCOL_VERSION,
            "knowledge_base_id": knowledge_base_id,
            "qa_session_id": qa_session_id,
            "question_hash": _raw_text_hash(question),
            "answer_hash": _raw_text_hash(answer),
            "context_package_id": context_package_id,
            "retrieval_trace_id": retrieval_trace_id,
            "provenance_session_hash": provenance_session_hash,
        }
    )
