from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterable, Mapping
from typing import Any


DESTRUCTIVE_INVENTORY_PROTOCOL_VERSION = (
    "destructive_cleanup_exact_inventory_v1"
)
INVENTORY_QUERY_BATCH_SIZE = 256
INVENTORY_SAMPLE_LIMIT = 20
INVENTORY_HARD_ENTRY_LIMIT = 100_000


class DestructiveInventoryError(RuntimeError):
    pass


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
        default=str,
    ).encode("utf-8")


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def hashed_value(value: Any) -> str:
    """Return an audit-safe digest without exposing the underlying value."""

    return canonical_hash(value)


def iter_query_identities(
    db,
    statement,
    identity: Callable[[Any], Mapping[str, Any] | None],
) -> Iterable[dict[str, Any]]:
    result = db.execute(
        statement.execution_options(
            yield_per=INVENTORY_QUERY_BATCH_SIZE,
            stream_results=True,
        )
    )
    for row in result:
        candidate = identity(row)
        if candidate is None:
            continue
        if not isinstance(candidate, Mapping):
            raise DestructiveInventoryError(
                "Inventory identity factory returned a non-object"
            )
        yield {str(key): value for key, value in candidate.items()}


class InventoryBuilder:
    def __init__(
        self,
        *,
        operation: str,
        knowledge_base_id: str,
        knowledge_base_name: str,
        options: Mapping[str, Any],
        impact: str,
        hard_entry_limit: int = INVENTORY_HARD_ENTRY_LIMIT,
    ) -> None:
        bounded_limit = int(hard_entry_limit)
        if bounded_limit < 1 or bounded_limit > INVENTORY_HARD_ENTRY_LIMIT:
            raise DestructiveInventoryError(
                "Inventory hard entry limit must be between 1 and "
                f"{INVENTORY_HARD_ENTRY_LIMIT}"
            )
        self._operation = str(operation)
        self._knowledge_base_id = str(knowledge_base_id)
        self._knowledge_base_name = str(knowledge_base_name)
        self._options = dict(options)
        self._impact = str(impact)
        self._hard_entry_limit = bounded_limit
        self._total_entries = 0
        self._scopes: dict[str, dict[str, Any]] = {}

    def add_scope(
        self,
        name: str,
        identities: Iterable[Mapping[str, Any]],
    ) -> None:
        scope_name = str(name)
        if scope_name in self._scopes:
            raise DestructiveInventoryError(
                f"Duplicate inventory scope: {scope_name}"
            )
        digest = hashlib.sha256()
        count = 0
        sample: list[dict[str, Any]] = []
        for raw_identity in identities:
            identity = {
                str(key): value for key, value in dict(raw_identity).items()
            }
            encoded = canonical_json(identity)
            count += 1
            self._total_entries += 1
            if self._total_entries > self._hard_entry_limit:
                raise DestructiveInventoryError(
                    "Destructive inventory exceeds its hard entry limit; "
                    "split the operation into a smaller exact target"
                )
            digest.update(len(encoded).to_bytes(8, "big"))
            digest.update(encoded)
            if len(sample) < INVENTORY_SAMPLE_LIMIT:
                sample.append(identity)
        self._scopes[scope_name] = {
            "count": count,
            "scope_hash": digest.hexdigest(),
            "sample": sample,
            "sample_truncated": count > len(sample),
        }

    def finalize(self) -> dict[str, Any]:
        scope_commitments = [
            {
                "name": name,
                "count": int(card["count"]),
                "scope_hash": str(card["scope_hash"]),
            }
            for name, card in sorted(self._scopes.items())
        ]
        commitment = {
            "protocol_version": DESTRUCTIVE_INVENTORY_PROTOCOL_VERSION,
            "operation": self._operation,
            "knowledge_base": {
                "id": self._knowledge_base_id,
                "name": self._knowledge_base_name,
            },
            "options": self._options,
            "impact": self._impact,
            "hard_entry_limit": self._hard_entry_limit,
            "total_entries": self._total_entries,
            "scopes": scope_commitments,
        }
        return {
            **commitment,
            "complete": True,
            "query_batch_size": INVENTORY_QUERY_BATCH_SIZE,
            "sample_limit_per_scope": INVENTORY_SAMPLE_LIMIT,
            "inventory_hash": canonical_hash(commitment),
            "scope_inventory": {
                name: dict(card)
                for name, card in sorted(self._scopes.items())
            },
        }


def combine_inventories(
    *,
    operation: str,
    knowledge_base_id: str,
    knowledge_base_name: str,
    options: Mapping[str, Any],
    impact: str,
    components: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    component_cards = [
        {
            "name": str(name),
            "protocol_version": str(component.get("protocol_version") or ""),
            "inventory_hash": str(component.get("inventory_hash") or ""),
            "total_entries": int(component.get("total_entries") or 0),
            "complete": component.get("complete") is True,
        }
        for name, component in sorted(components.items())
    ]
    if not component_cards or not all(
        card["complete"]
        and len(card["inventory_hash"]) == 64
        and bool(card["protocol_version"])
        for card in component_cards
    ):
        raise DestructiveInventoryError(
            "Composite destructive inventory contains an incomplete component"
        )
    commitment = {
        "protocol_version": DESTRUCTIVE_INVENTORY_PROTOCOL_VERSION,
        "operation": str(operation),
        "knowledge_base": {
            "id": str(knowledge_base_id),
            "name": str(knowledge_base_name),
        },
        "options": dict(options),
        "impact": str(impact),
        "components": component_cards,
    }
    return {
        **commitment,
        "complete": True,
        "total_entries": sum(card["total_entries"] for card in component_cards),
        "inventory_hash": canonical_hash(commitment),
        "component_inventory": {
            str(name): dict(component)
            for name, component in sorted(components.items())
        },
    }


def require_exact_confirmation(
    *,
    actual_knowledge_base_id: str,
    actual_inventory_hash: str,
    confirmed_knowledge_base_id: str | None,
    confirmed_inventory_hash: str | None,
) -> None:
    if confirmed_knowledge_base_id != actual_knowledge_base_id:
        raise SystemExit(
            "--confirm-knowledge-base-id must exactly repeat the resolved "
            "knowledge-base id from the latest dry-run"
        )
    if confirmed_inventory_hash != actual_inventory_hash:
        raise SystemExit(
            "--confirm-inventory-hash must exactly repeat the complete "
            "inventory hash from the latest dry-run"
        )


def _ordered_scope(
    db,
    statement,
    identity: Callable[[Any], Mapping[str, Any] | None],
) -> Iterable[dict[str, Any]]:
    return iter_query_identities(db, statement, identity)


def build_stale_cleanup_inventory(
    db,
    knowledge_base,
    *,
    delete_inactive_chunks: bool,
    hard_entry_limit: int = INVENTORY_HARD_ENTRY_LIMIT,
) -> dict[str, Any]:
    from sqlalchemy import or_, select

    from app.models import (
        Chunk,
        ChunkContextText,
        ChunkCoordinate,
        ChunkRelationEdge,
        ChunkSpan,
        ChunkStructureEdge,
        ChunkStructureMapping,
        ChunkStructureNode,
        ChunkVersion,
        CitationVerification,
        Document,
        DocumentVersion,
        IngestionCompensationLog,
        RQPrefixMembership,
        VectorRecord,
    )
    from app.services.qdrant_outbox import (
        LEGACY_QDRANT_DELETE_BLOCKING_STATES,
        LEGACY_QDRANT_DELETE_OPERATION,
        QDRANT_DELETE_ACTIVE_STATES,
        QDRANT_DELETE_OPERATION,
    )

    refresh = getattr(db, "refresh", None)
    if callable(refresh):
        refresh(knowledge_base)
    knowledge_base_id = str(knowledge_base.id)
    current_chunk_version = int(
        knowledge_base.current_chunk_version or 0
    )
    impact = (
        "normalize ChunkVersion active state, delete stale VectorRecords and "
        "their owner-fenced Qdrant points, and delete the exact inactive "
        "chunk/document-version/dependent rows listed in this inventory"
        if delete_inactive_chunks
        else "normalize ChunkVersion active state and delete stale VectorRecords "
        "plus their owner-fenced Qdrant points; retain inactive chunks and "
        "document versions"
    )
    builder = InventoryBuilder(
        operation="cleanup_stale_data",
        knowledge_base_id=knowledge_base_id,
        knowledge_base_name=str(knowledge_base.name),
        options={
            "delete_inactive_chunks": bool(delete_inactive_chunks),
            "current_chunk_version": current_chunk_version,
        },
        impact=impact,
        hard_entry_limit=hard_entry_limit,
    )

    inactive_chunk_ids = select(Chunk.id).where(
        Chunk.knowledge_base_id == knowledge_base_id,
        Chunk.state != "active",
    )
    inactive_document_version_ids = (
        select(DocumentVersion.id)
        .join(Document, DocumentVersion.document_id == Document.id)
        .where(
            Document.knowledge_base_id == knowledge_base_id,
            DocumentVersion.is_active.is_(False),
        )
    )

    chunk_version_state_query = (
        select(ChunkVersion)
        .where(
            ChunkVersion.knowledge_base_id == knowledge_base_id,
            or_(
                (
                    (ChunkVersion.chunk_version != current_chunk_version)
                    & (ChunkVersion.state == "active")
                ),
                (
                    (ChunkVersion.chunk_version == current_chunk_version)
                    & (ChunkVersion.state != "active")
                ),
            ),
        )
        .order_by(ChunkVersion.id.asc())
    )
    builder.add_scope(
        "chunk_version_state_updates",
        _ordered_scope(
            db,
            chunk_version_state_query,
            lambda row: {
                "id": str(row[0].id),
                "chunk_version": int(row[0].chunk_version),
                "current_state": str(row[0].state),
                "target_state": (
                    "active"
                    if int(row[0].chunk_version) == current_chunk_version
                    else "inactive"
                ),
            },
        ),
    )

    stale_vector_query = (
        select(VectorRecord)
        .join(Chunk, VectorRecord.chunk_id == Chunk.id)
        .where(
            VectorRecord.knowledge_base_id == knowledge_base_id,
            Chunk.state != "active",
        )
        .order_by(VectorRecord.id.asc())
    )
    builder.add_scope(
        "stale_vector_records_and_qdrant_points",
        _ordered_scope(
            db,
            stale_vector_query,
            lambda row: {
                "id": str(row[0].id),
                "chunk_id": str(row[0].chunk_id),
                "collection_name": str(row[0].collection_name),
                "qdrant_point_id": str(row[0].qdrant_point_id),
                "vector_status": str(row[0].vector_status),
            },
        ),
    )

    pending_delete_query = (
        select(IngestionCompensationLog)
        .where(
            IngestionCompensationLog.knowledge_base_id == knowledge_base_id,
            (
                (
                    IngestionCompensationLog.operation
                    == QDRANT_DELETE_OPERATION
                )
                & IngestionCompensationLog.status.in_(
                    sorted(QDRANT_DELETE_ACTIVE_STATES)
                )
            )
            | (
                (
                    IngestionCompensationLog.operation
                    == LEGACY_QDRANT_DELETE_OPERATION
                )
                & IngestionCompensationLog.status.in_(
                    sorted(LEGACY_QDRANT_DELETE_BLOCKING_STATES)
                )
            ),
        )
        .order_by(IngestionCompensationLog.id.asc())
    )
    builder.add_scope(
        "pending_qdrant_delete_intents",
        _ordered_scope(
            db,
            pending_delete_query,
            lambda row: {
                "id": str(row[0].id),
                "operation": str(row[0].operation),
                "status": str(row[0].status),
                "target_ids_hash": hashed_value(
                    list(row[0].target_ids_json or [])
                ),
                "payload_hash": hashed_value(dict(row[0].payload_json or {})),
            },
        ),
    )

    destructive_scopes: list[
        tuple[str, Any, Callable[[Any], Mapping[str, Any] | None]]
    ] = []
    if delete_inactive_chunks:
        destructive_scopes.extend(
            [
                (
                    "inactive_chunks",
                    select(Chunk)
                    .where(Chunk.id.in_(inactive_chunk_ids))
                    .order_by(Chunk.id.asc()),
                    lambda row: {
                        "id": str(row[0].id),
                        "document_version_id": str(
                            row[0].document_version_id
                        ),
                        "chunk_version": int(row[0].chunk_version),
                        "state": str(row[0].state),
                    },
                ),
                (
                    "inactive_document_versions",
                    select(DocumentVersion)
                    .where(
                        DocumentVersion.id.in_(
                            inactive_document_version_ids
                        )
                    )
                    .order_by(DocumentVersion.id.asc()),
                    lambda row: {
                        "id": str(row[0].id),
                        "document_id": str(row[0].document_id),
                        "version": int(row[0].version),
                        "is_active": bool(row[0].is_active),
                    },
                ),
                (
                    "inactive_chunk_versions",
                    select(ChunkVersion)
                    .where(
                        ChunkVersion.knowledge_base_id == knowledge_base_id,
                        ChunkVersion.chunk_version != current_chunk_version,
                    )
                    .order_by(ChunkVersion.id.asc()),
                    lambda row: {
                        "id": str(row[0].id),
                        "chunk_version": int(row[0].chunk_version),
                        "state": str(row[0].state),
                    },
                ),
                (
                    "citation_chunk_nullifications",
                    select(CitationVerification)
                    .where(
                        CitationVerification.chunk_id.in_(
                            inactive_chunk_ids
                        )
                    )
                    .order_by(CitationVerification.id.asc()),
                    lambda row: {
                        "id": str(row[0].id),
                        "chunk_id": str(row[0].chunk_id),
                    },
                ),
                (
                    "chunk_relation_edges",
                    select(ChunkRelationEdge)
                    .where(
                        (
                            ChunkRelationEdge.source_chunk_id.in_(
                                inactive_chunk_ids
                            )
                        )
                        | (
                            ChunkRelationEdge.target_chunk_id.in_(
                                inactive_chunk_ids
                            )
                        )
                    )
                    .order_by(ChunkRelationEdge.id.asc()),
                    lambda row: {
                        "id": str(row[0].id),
                        "source_chunk_id": str(row[0].source_chunk_id),
                        "target_chunk_id": str(row[0].target_chunk_id),
                    },
                ),
                (
                    "rq_prefix_memberships",
                    select(RQPrefixMembership)
                    .where(
                        RQPrefixMembership.chunk_id.in_(inactive_chunk_ids)
                    )
                    .order_by(RQPrefixMembership.id.asc()),
                    lambda row: {
                        "id": str(row[0].id),
                        "chunk_id": str(row[0].chunk_id),
                        "rq_prefix_id": str(row[0].rq_prefix_id),
                    },
                ),
                (
                    "chunk_structure_mappings",
                    select(ChunkStructureMapping)
                    .where(
                        ChunkStructureMapping.chunk_id.in_(
                            inactive_chunk_ids
                        )
                    )
                    .order_by(ChunkStructureMapping.id.asc()),
                    lambda row: {
                        "id": str(row[0].id),
                        "chunk_id": str(row[0].chunk_id),
                        "structure_node_id": str(row[0].structure_node_id),
                    },
                ),
                (
                    "chunk_context_texts",
                    select(ChunkContextText)
                    .where(
                        ChunkContextText.chunk_id.in_(inactive_chunk_ids)
                    )
                    .order_by(ChunkContextText.id.asc()),
                    lambda row: {
                        "id": str(row[0].id),
                        "chunk_id": str(row[0].chunk_id),
                        "context_hash": str(row[0].context_hash),
                    },
                ),
                (
                    "chunk_coordinates",
                    select(ChunkCoordinate)
                    .where(
                        ChunkCoordinate.chunk_id.in_(inactive_chunk_ids)
                    )
                    .order_by(ChunkCoordinate.id.asc()),
                    lambda row: {
                        "id": str(row[0].id),
                        "chunk_id": str(row[0].chunk_id),
                    },
                ),
                (
                    "chunk_spans",
                    select(ChunkSpan)
                    .where(ChunkSpan.chunk_id.in_(inactive_chunk_ids))
                    .order_by(ChunkSpan.id.asc()),
                    lambda row: {
                        "id": str(row[0].id),
                        "chunk_id": str(row[0].chunk_id),
                    },
                ),
                (
                    "chunk_structure_edges",
                    select(ChunkStructureEdge)
                    .where(
                        ChunkStructureEdge.document_version_id.in_(
                            inactive_document_version_ids
                        )
                    )
                    .order_by(ChunkStructureEdge.id.asc()),
                    lambda row: {
                        "id": str(row[0].id),
                        "document_version_id": str(
                            row[0].document_version_id
                        ),
                    },
                ),
                (
                    "chunk_structure_nodes",
                    select(ChunkStructureNode)
                    .where(
                        ChunkStructureNode.document_version_id.in_(
                            inactive_document_version_ids
                        )
                    )
                    .order_by(ChunkStructureNode.id.asc()),
                    lambda row: {
                        "id": str(row[0].id),
                        "document_version_id": str(
                            row[0].document_version_id
                        ),
                    },
                ),
            ]
        )
    for scope_name, statement, identity in destructive_scopes:
        builder.add_scope(
            scope_name,
            _ordered_scope(db, statement, identity),
        )
    return builder.finalize()
