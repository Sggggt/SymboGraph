"""Bounded, read-only coarse navigation for pre-retrieval planning."""
from __future__ import annotations

import json
import math
import time
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator
from sqlalchemy import select

from app.models import Chunk, CoarseConcept, ContextGraphState, Document
from app.retrieval_control_contracts import control_hash
from app.schemas import SearchFilters
from app.services.qa_performance import qa_stage


PROTOCOL = "coarse_resource_read_v1"
MAX_DIRECTORY_CHARACTERS = 64_000
MAX_DETAILS = 4
MAX_DETAIL_FIELD_CHARACTERS = 800
MAX_DETAILS_CHARACTERS = 20_000


class ResourceReadBudgetError(ValueError):
    def __init__(self, mode: str, node_count: int, characters: int):
        super().__init__(f"resource_read_{mode}_over_budget")
        self.mode = mode
        self.node_count = node_count
        self.characters = characters


class ResourceReadAction(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    action: Literal["resource_read"]
    mode: Literal["titles", "details"]
    keys: tuple[str, ...] = Field(default=(), max_length=MAX_DETAILS)

    @model_validator(mode="after")
    def valid_keys(self):
        if self.mode == "titles" and self.keys:
            raise ValueError("resource_read_titles_keys_forbidden")
        if self.mode == "details" and (not self.keys or len(set(self.keys)) != len(self.keys)):
            raise ValueError("resource_read_details_keys_invalid")
        return self


def _active_state(db, knowledge_base_id: str, graph_identity: str) -> ContextGraphState:
    state = db.scalar(
        select(ContextGraphState).where(
            ContextGraphState.knowledge_base_id == knowledge_base_id,
            ContextGraphState.state == "active",
        )
    )
    if (
        state is None
        or state.context_graph_hash != graph_identity
        or not state.coarse_concept_state_id
    ):
        raise ValueError("resource_read_graph_identity_changed")
    return state


def verify_resource_snapshot(db, *, knowledge_base_id: str, graph_identity: str) -> None:
    _active_state(db, knowledge_base_id, graph_identity)


def _matches_filters(chunk: Chunk, document: Document, filters: SearchFilters) -> bool:
    if filters.document_ids and chunk.document_id not in filters.document_ids:
        return False
    if filters.source_paths and document.source_path not in filters.source_paths:
        return False
    if filters.source_type and document.source_type != filters.source_type:
        return False
    if filters.chunk_version is not None and chunk.chunk_version != filters.chunk_version:
        return False
    if filters.tags and not set(filters.tags).intersection(document.tags or []):
        return False
    if filters.page_range and any(value is not None for value in filters.page_range):
        lower, upper = filters.page_range
        if (
            chunk.page_start is None
            or chunk.page_end is None
            or (lower is not None and chunk.page_end < lower)
            or (upper is not None and chunk.page_start > upper)
        ):
            return False
    metadata = chunk.metadata_json or {}
    if filters.content_kinds and not {
        metadata.get("content_kind"), *(metadata.get("protected_object_kinds") or [])
    }.intersection(filters.content_kinds):
        return False
    if (
        filters.partition
        and filters.partition not in (document.tags or [])
        and filters.partition != metadata.get("partition")
    ):
        return False
    return True


def _eligible_support_ids(db, *, knowledge_base_id: str, concepts: list[CoarseConcept], filters: SearchFilters) -> set[str] | None:
    if filters == SearchFilters():
        return None
    ids = sorted({str(chunk_id) for concept in concepts for chunk_id in (concept.support_chunk_ids_json or [])})
    eligible: set[str] = set()
    for offset in range(0, len(ids), 500):
        rows = db.execute(
            select(Chunk, Document)
            .join(Document, Chunk.document_id == Document.id)
            .where(
                Chunk.id.in_(ids[offset:offset + 500]),
                Chunk.knowledge_base_id == knowledge_base_id,
                Chunk.state == "active",
                Document.knowledge_base_id == knowledge_base_id,
                Document.is_active.is_(True),
            )
        )
        eligible.update(
            chunk.id for chunk, document in rows
            if _matches_filters(chunk, document, filters)
        )
    return eligible


def read_coarse_titles(db, *, knowledge_base_id: str, graph_identity: str, filters: SearchFilters) -> tuple[dict, dict[str, str], dict]:
    started = time.monotonic()
    with qa_stage("resource_read"):
        state = _active_state(db, knowledge_base_id, graph_identity)
        concepts = list(db.scalars(
            select(CoarseConcept).where(
                CoarseConcept.knowledge_base_id == knowledge_base_id,
                CoarseConcept.coarse_state_id == state.coarse_concept_state_id,
                CoarseConcept.state == "active",
            ).order_by(CoarseConcept.node_weight.desc(), CoarseConcept.id.asc())
        ))
        eligible = _eligible_support_ids(db, knowledge_base_id=knowledge_base_id, concepts=concepts, filters=filters)
        if eligible is not None:
            concepts = [
                item for item in concepts
                if (support := set(item.support_chunk_ids_json or [])) and support <= eligible
            ]
        nodes = []
        key_to_id = {}
        for index, item in enumerate(concepts, 1):
            weight = float(item.node_weight)
            confidence = float(item.confidence)
            if not (math.isfinite(weight) and math.isfinite(confidence)):
                raise ValueError("resource_read_node_metadata_invalid")
            key = f"c{index}"
            key_to_id[key] = item.id
            nodes.append({
                "key": key,
                "title": item.canonical_label,
                "node_weight": weight,
                "confidence": confidence,
                "mid_count": len(item.included_mid_concept_ids_json or []),
                "support_chunk_count": len(item.support_chunk_ids_json or []),
            })
        payload = {"protocol_version": PROTOCOL, "mode": "titles", "complete": True, "nodes": nodes, "is_answer_evidence": False}
        size = len(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
        if size > MAX_DIRECTORY_CHARACTERS:
            raise ResourceReadBudgetError("directory", len(nodes), size)
        _active_state(db, knowledge_base_id, graph_identity)
    audit = {
        "mode": "titles", "graph_identity": graph_identity,
        "filter_hash": control_hash(filters.model_dump(mode="json")),
        "node_count": len(nodes), "characters": size,
        "result_hash": control_hash(payload),
        "local_duration_ms": round((time.monotonic() - started) * 1000, 3),
    }
    return payload, key_to_id, audit


def _bounded_text(value: str, limit: int = MAX_DETAIL_FIELD_CHARACTERS) -> dict:
    text = str(value or "")
    return {"text": text[:limit], "truncated": len(text) > limit}


def read_coarse_details(db, *, knowledge_base_id: str, graph_identity: str, keys: tuple[str, ...], key_to_id: dict[str, str]) -> tuple[dict, dict]:
    if not keys or len(keys) > MAX_DETAILS or len(set(keys)) != len(keys) or any(key not in key_to_id for key in keys):
        raise ValueError("resource_read_key_outside_directory")
    started = time.monotonic()
    with qa_stage("resource_read"):
        state = _active_state(db, knowledge_base_id, graph_identity)
        ids = [key_to_id[key] for key in keys]
        concepts = {
            item.id: item for item in db.scalars(
                select(CoarseConcept).where(
                    CoarseConcept.id.in_(ids),
                    CoarseConcept.knowledge_base_id == knowledge_base_id,
                    CoarseConcept.coarse_state_id == state.coarse_concept_state_id,
                    CoarseConcept.state == "active",
                )
            )
        }
        if set(concepts) != set(ids):
            raise ValueError("resource_read_node_identity_changed")
        nodes = []
        for key in keys:
            item = concepts[key_to_id[key]]
            nodes.append({
                "key": key, "title": item.canonical_label,
                "summary": _bounded_text(item.summary),
                "definition": _bounded_text(item.definition, 400),
                "scope_note": _bounded_text(item.scope_note, 400),
                "inclusion_criteria": [_bounded_text(value, 200) for value in (item.inclusion_criteria_json or [])[:4]],
                "exclusion_criteria": [_bounded_text(value, 200) for value in (item.exclusion_criteria_json or [])[:4]],
                "inclusion_criteria_truncated": len(item.inclusion_criteria_json or []) > 4,
                "exclusion_criteria_truncated": len(item.exclusion_criteria_json or []) > 4,
            })
        payload = {"protocol_version": PROTOCOL, "mode": "details", "nodes": nodes, "is_answer_evidence": False}
        size = len(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
        if size > MAX_DETAILS_CHARACTERS:
            raise ResourceReadBudgetError("details", len(nodes), size)
        _active_state(db, knowledge_base_id, graph_identity)
    audit = {
        "mode": "details", "graph_identity": graph_identity,
        "keys": list(keys), "node_count": len(nodes),
        "characters": size,
        "result_hash": control_hash(payload),
        "local_duration_ms": round((time.monotonic() - started) * 1000, 3),
    }
    return payload, audit
