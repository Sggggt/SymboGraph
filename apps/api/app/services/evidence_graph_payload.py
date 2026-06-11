from __future__ import annotations

import json
from collections import Counter, defaultdict
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.models import (
    ActiveChunk,
    CommunityMembership,
    CommunityState,
    Document,
    EvidenceAtom,
    EvidenceEdge,
    EvidenceGraphState,
    KnowledgeBase,
    SignalEdge,
    SignalNode,
    SignalState,
)

GRAPH_TYPES = {"evidence"}


def _graph_response(
    graph_type: str,
    nodes: list[dict],
    edges: list[dict],
    partition: str | None = None,
    *,
    signal_state: SignalState | None = None,
    view: str | None = None,
    diagnostics: dict[str, Any] | None = None,
) -> dict:
    return {
        "graph_type": graph_type,
        "schema_version": "evidence_graph_v1" if graph_type == "evidence" else "typed_graph_v1",
        "view": view,
        "nodes": nodes,
        "edges": edges,
        "node_counts": dict(Counter(str(node.get("category") or "unknown") for node in nodes)),
        "edge_counts": dict(Counter(str(edge.get("category") or edge.get("label") or "unknown") for edge in edges)),
        "focus_partition": partition,
        "freshness": {},
        "signal_layer_status": signal_state.status if signal_state else None,
        "signal_state_id": signal_state.id if signal_state else None,
        "signal_state_hash": signal_state.signal_state_hash if signal_state else None,
        "signal_layer_complete": bool(signal_state and signal_state.status == "active"),
        "diagnostics": diagnostics or {},
    }


def _active_global_graph_state(db: Session, knowledge_base_id: str) -> EvidenceGraphState | None:
    return db.scalar(
        select(EvidenceGraphState)
        .where(
            EvidenceGraphState.knowledge_base_id == knowledge_base_id,
            EvidenceGraphState.scope_type == "global",
            EvidenceGraphState.state == "active",
        )
        .order_by(EvidenceGraphState.created_at.desc())
    )


def _load_documents(db: Session, knowledge_base_id: str, partition: str | None = None) -> tuple[KnowledgeBase | None, list[Document]]:
    knowledge_base = db.get(KnowledgeBase, knowledge_base_id)
    query = select(Document).where(Document.knowledge_base_id == knowledge_base_id, Document.is_active.is_(True))
    if partition:
        query = query.where(Document.tags.contains([partition]))
    documents = db.scalars(query.order_by(Document.title.asc(), Document.id.asc())).all()
    return knowledge_base, documents


def graph_freshness(db: Session, knowledge_base_id: str) -> dict[str, Any]:
    latest_state = _active_global_graph_state(db, knowledge_base_id)
    active_atoms = set(
        db.scalars(
            select(EvidenceAtom.id).where(EvidenceAtom.knowledge_base_id == knowledge_base_id, EvidenceAtom.state == "active")
        ).all()
    )
    if latest_state is None:
        return {
            "is_stale": bool(active_atoms),
            "reason": "missing_evidence_graph_state" if active_atoms else None,
            "active_atom_count": len(active_atoms),
            "graph_atom_count": 0,
        }
    graph_atom_ids = {str(item) for item in (latest_state.active_atom_ids or []) if item}
    missing_atoms = sorted(graph_atom_ids - active_atoms)
    new_atoms = sorted(active_atoms - graph_atom_ids)
    edge_count = db.scalar(select(func.count(EvidenceEdge.id)).where(EvidenceEdge.graph_state_id == latest_state.id))
    is_stale = bool(missing_atoms or new_atoms)
    return {
        "is_stale": is_stale,
        "reason": "atom_scope_changed" if is_stale else None,
        "active_atom_count": len(active_atoms),
        "graph_atom_count": len(graph_atom_ids),
        "missing_graph_atoms": len(missing_atoms),
        "new_active_atoms": len(new_atoms),
        "evidence_edge_count": int(edge_count or 0),
        "graph_state_id": latest_state.id,
        "graph_state_hash": latest_state.state_hash,
        "graph_built_at": latest_state.created_at.isoformat() if latest_state.created_at else None,
    }


def _apply_overview_limits(payload: dict) -> dict:
    settings = get_settings()
    max_nodes = int(settings.graph_overview_max_nodes)
    max_edges = int(settings.graph_overview_max_edges)
    nodes = payload["nodes"]
    edges = payload["edges"]
    by_category: dict[str, list[dict]] = defaultdict(list)
    for node in nodes:
        by_category[str(node.get("category") or "unknown")].append(node)

    def rank(items: list[dict]) -> list[dict]:
        return sorted(
            items,
            key=lambda node: (
                int(node.get("support_count") or 0),
                float(node.get("confidence") or 0.0),
                float(node.get("value") or 0.0),
                str(node.get("name") or ""),
            ),
            reverse=True,
        )

    kept_nodes: list[dict] = []
    seen: set[str] = set()

    def take(category: str, limit: int) -> None:
        nonlocal kept_nodes
        for node in rank(by_category.get(category, [])):
            if len(kept_nodes) >= max_nodes or len([item for item in kept_nodes if item.get("category") == category]) >= limit:
                break
            node_id = str(node.get("id"))
            if node_id in seen:
                continue
            seen.add(node_id)
            kept_nodes.append(node)

    take("evidence_graph_state", 1)
    take("community_region", max(8, min(len(by_category.get("community_region", [])), max_nodes // 5)))
    take("signal_node", max(20, int(max_nodes * 0.55)))
    take("active_chunk", max(12, int(max_nodes * 0.22)))
    take("document_version", max(4, int(max_nodes * 0.06)))
    take("evidence_atom", max(0, max_nodes - len(kept_nodes)))
    if len(kept_nodes) < max_nodes:
        for category in sorted(by_category):
            take(category, max_nodes)
    kept_ids = {str(node.get("id")) for node in kept_nodes}
    visible_edges = [edge for edge in edges if str(edge.get("source")) in kept_ids and str(edge.get("target")) in kept_ids]
    edges_by_category: dict[str, list[dict]] = defaultdict(list)
    for edge in visible_edges:
        edges_by_category[str(edge.get("category") or edge.get("label") or "unknown")].append(edge)

    def rank_edges(items: list[dict]) -> list[dict]:
        return sorted(
            items,
            key=lambda edge: (float(edge.get("weight") or 0.0), float(edge.get("confidence") or 0.0), int(edge.get("support_count") or 0)),
            reverse=True,
        )

    ranked_edges: list[dict] = []
    seen_edge_keys: set[tuple[str, str, str]] = set()

    def take_edges(category: str, limit: int) -> None:
        for edge in rank_edges(edges_by_category.get(category, [])):
            if len(ranked_edges) >= max_edges or len([item for item in ranked_edges if (item.get("category") or item.get("label")) == category]) >= limit:
                break
            key = (str(edge.get("source")), str(edge.get("target")), str(edge.get("label")))
            if key in seen_edge_keys:
                continue
            seen_edge_keys.add(key)
            ranked_edges.append(edge)

    take_edges("community", max(30, int(max_edges * 0.20)))
    take_edges("active_chunk", max(30, int(max_edges * 0.18)))
    take_edges("signal_projection", max(80, int(max_edges * 0.45)))
    take_edges("observation", max(20, int(max_edges * 0.10)))
    take_edges("graph_scope", max(10, int(max_edges * 0.04)))
    if len(ranked_edges) < max_edges:
        for category in sorted(edges_by_category):
            take_edges(category, max_edges)
    payload = {
        **payload,
        "nodes": kept_nodes,
        "edges": ranked_edges,
        "node_counts": dict(Counter(str(node.get("category") or "unknown") for node in kept_nodes)),
        "edge_counts": dict(Counter(str(edge.get("category") or edge.get("label") or "unknown") for edge in ranked_edges)),
        "diagnostics": {
            **(payload.get("diagnostics") or {}),
            "overview_limits": {"max_nodes": max_nodes, "max_edges": max_edges},
            "overview_truncated": len(nodes) > len(kept_nodes) or len(edges) > len(ranked_edges),
            "full_node_count": len(nodes),
            "full_edge_count": len(edges),
        },
    }
    return payload


def get_evidence_graph_payload(db: Session, knowledge_base_id: str, partition: str | None = None, view: str | None = None) -> dict:
    view = view or ("detail" if partition else "overview")
    latest_state = _active_global_graph_state(db, knowledge_base_id)
    if latest_state is None:
        return _graph_response("evidence", [], [], partition=partition, view=view)
    signal_state = db.scalar(
        select(SignalState)
        .where(
            SignalState.knowledge_base_id == knowledge_base_id,
            SignalState.evidence_graph_state_id == latest_state.id,
        )
        .order_by(SignalState.created_at.desc())
    )
    active_signal_state = signal_state if signal_state is not None and signal_state.status == "active" else None

    atoms = db.scalars(
        select(EvidenceAtom)
        .where(
            EvidenceAtom.knowledge_base_id == knowledge_base_id,
            EvidenceAtom.id.in_(latest_state.active_atom_ids or []),
            EvidenceAtom.state == "active",
        )
        .order_by(EvidenceAtom.document_id.asc(), EvidenceAtom.atom_index.asc())
    ).all()
    if partition:
        documents = {document.id: document for document in db.scalars(select(Document).where(Document.knowledge_base_id == knowledge_base_id)).all()}
        atoms = [
            atom
            for atom in atoms
            if (documents.get(atom.document_id) and partition in (documents[atom.document_id].tags or []))
            or (atom.source_span_json or {}).get("section") == partition
        ]
    atom_ids = {atom.id for atom in atoms}
    edges = db.scalars(
        select(EvidenceEdge).where(
            EvidenceEdge.graph_state_id == latest_state.id,
            EvidenceEdge.source_atom_id.in_(atom_ids) if atom_ids else False,
            EvidenceEdge.target_atom_id.in_(atom_ids) if atom_ids else False,
        )
    ).all() if atom_ids else []
    active_chunks = db.scalars(
        select(ActiveChunk).where(
            ActiveChunk.knowledge_base_id == knowledge_base_id,
            ActiveChunk.graph_state_hash == latest_state.state_hash,
            ActiveChunk.state == "active",
        )
    ).all()
    memberships = db.scalars(
        select(CommunityMembership)
        .join(CommunityState, CommunityState.id == CommunityMembership.community_state_id)
        .where(
            CommunityState.knowledge_base_id == knowledge_base_id,
            CommunityState.graph_state_id == latest_state.id,
            CommunityMembership.atom_id.in_(atom_ids) if atom_ids else False,
        )
    ).all() if atom_ids else []

    nodes: list[dict] = [
        {
            "id": f"evidence_graph:{latest_state.id}",
            "name": f"Evidence graph {latest_state.state_hash[:8]}",
            "category": "evidence_graph_state",
            "value": 3,
            "support_count": len(atoms),
            "summary": json.dumps(latest_state.stats_json or {}, ensure_ascii=False),
        }
    ]
    node_ids = {nodes[0]["id"]}
    edge_items: list[dict] = []
    seen_edges: set[tuple[str, str, str]] = set()

    def add_node(node: dict) -> None:
        if node["id"] in node_ids:
            return
        node_ids.add(node["id"])
        nodes.append(node)

    def add_edge(source: str, target: str, label: str, category: str, **extra: Any) -> None:
        key = (source, target, label)
        if key in seen_edges:
            return
        seen_edges.add(key)
        edge_items.append({"source": source, "target": target, "label": label, "category": category, **extra})

    for atom in atoms:
        span = atom.source_span_json or {}
        atom_node_id = f"atom:{atom.id}"
        version_node_id = f"document_version:{atom.document_version_id}"
        add_node(
            {
                "id": atom_node_id,
                "name": atom.text[:80].replace("\n", " ") or atom.id,
                "category": "evidence_atom",
                "value": 1.2 if atom.atom_type != "heading" else 1.6,
                "document_id": atom.document_id,
                "document_version_id": atom.document_version_id,
                "snippet": atom.text[:240],
                "page_number": span.get("page_number"),
                "source_type": (atom.metadata_json or {}).get("content_kind"),
                "entity_type": atom.atom_type,
                "confidence": atom.parser_confidence,
                "summary": json.dumps({"source_span": span}, ensure_ascii=False),
            }
        )
        add_node(
            {
                "id": version_node_id,
                "name": atom.document_version_id,
                "category": "document_version",
                "value": 2,
                "document_id": atom.document_id,
                "document_version_id": atom.document_version_id,
            }
        )
        add_edge(f"evidence_graph:{latest_state.id}", atom_node_id, "contains_atom", "graph_scope")
        add_edge(atom_node_id, version_node_id, "from_version", "traceability")

    for edge in edges:
        add_edge(
            f"atom:{edge.source_atom_id}",
            f"atom:{edge.target_atom_id}",
            edge.edge_type,
            "observation",
            confidence=edge.confidence,
            weight=edge.weight,
            relation_source="evidence_graph",
        )

    atom_to_community: dict[str, list[str]] = defaultdict(list)
    for membership in memberships:
        atom_to_community[membership.atom_id].append(membership.community_id)
    for community_id in sorted({membership.community_id for membership in memberships}):
        community_node_id = f"community:{community_id}"
        add_node(
            {
                "id": community_node_id,
                "name": community_id,
                "category": "community_region",
                "value": 2,
                "support_count": sum(1 for membership in memberships if membership.community_id == community_id),
            }
        )
    for atom_id, community_ids in atom_to_community.items():
        for community_id in community_ids:
            add_edge(f"atom:{atom_id}", f"community:{community_id}", "member_of", "community")

    for active_chunk in active_chunks:
        chunk_node_id = f"active_chunk:{active_chunk.id}"
        add_node(
            {
                "id": chunk_node_id,
                "name": active_chunk.text[:80].replace("\n", " "),
                "category": "active_chunk",
                "value": 2,
                "snippet": active_chunk.text[:240],
                "support_count": len(active_chunk.atom_ids_json or []),
                "summary": json.dumps({"source_span_union": active_chunk.source_span_union_json}, ensure_ascii=False),
            }
        )
        for atom_id in active_chunk.atom_ids_json or []:
            if atom_id in atom_ids:
                add_edge(chunk_node_id, f"atom:{atom_id}", "grounded_by", "active_chunk")
        for community_id in (active_chunk.community_ids_json or (active_chunk.metadata_json or {}).get("community_ids") or [])[:4]:
            add_edge(chunk_node_id, f"community:{community_id}", "located_in", "community", relation_source="active_chunk_metadata")
    if active_signal_state is not None:
        signal_nodes = db.scalars(
            select(SignalNode).where(SignalNode.signal_state_id == active_signal_state.id)
        ).all()
        visible_signal_nodes = [
            node
            for node in signal_nodes
            if set(str(atom_id) for atom_id in (node.support_atom_ids_json or [])).intersection(atom_ids)
        ]
        signal_node_ids = {node.id for node in visible_signal_nodes}
        for node in visible_signal_nodes:
            support_atom_ids = sorted(set(str(atom_id) for atom_id in (node.support_atom_ids_json or [])).intersection(atom_ids))
            signal_node_id = f"signal:{node.id}"
            add_node(
                {
                    "id": signal_node_id,
                    "name": node.canonical_label,
                    "category": "signal_node",
                    "value": 2.4 + min(4.0, len(support_atom_ids) * 0.35),
                    "entity_type": node.signal_type,
                    "confidence": node.confidence,
                    "support_count": len(support_atom_ids),
                    "support_atom_ids": support_atom_ids,
                    "support_active_chunk_ids": node.support_active_chunk_ids_json or [],
                    "source_span_union": node.source_span_union_json,
                    "summary": json.dumps(node.quality_json or {}, ensure_ascii=False),
                }
            )
            for atom_id in support_atom_ids[:12]:
                add_edge(signal_node_id, f"atom:{atom_id}", "supported_by", "signal_projection", confidence=node.confidence, relation_source="signal_projection")
            for active_chunk_id in node.support_active_chunk_ids_json or []:
                add_edge(signal_node_id, f"active_chunk:{active_chunk_id}", "appears_in", "signal_projection", confidence=node.confidence, relation_source="signal_projection")
        signal_edges = db.scalars(
            select(SignalEdge).where(
                SignalEdge.signal_state_id == active_signal_state.id,
                SignalEdge.source_signal_id.in_(signal_node_ids) if signal_node_ids else False,
                SignalEdge.target_signal_id.in_(signal_node_ids) if signal_node_ids else False,
            )
        ).all() if signal_node_ids else []
        for edge in signal_edges:
            support_atom_ids = sorted(set(str(atom_id) for atom_id in (edge.support_atom_ids_json or [])).intersection(atom_ids))
            if not support_atom_ids:
                continue
            add_edge(
                f"signal:{edge.source_signal_id}",
                f"signal:{edge.target_signal_id}",
                edge.edge_type,
                "signal_projection",
                confidence=edge.confidence,
                support_count=len(support_atom_ids),
                support_atom_ids=support_atom_ids,
                support_active_chunk_ids=edge.support_active_chunk_ids_json or [],
                source_span_union=edge.source_span_union_json,
                relation_source="signal_projection",
            )
    payload = _graph_response("evidence", nodes, edge_items, partition=partition, signal_state=signal_state, view=view)
    if view == "overview":
        payload = _apply_overview_limits(payload)
    return payload


def get_query_evidence_graph_payload(db: Session, knowledge_base_id: str, chunk_ids: list[str], query: str | None = None) -> dict:
    del query
    requested_chunk_ids = list(dict.fromkeys(str(chunk_id) for chunk_id in chunk_ids if chunk_id))
    if not requested_chunk_ids:
        return _graph_response("evidence", [], [])

    chunks = db.scalars(
        select(ActiveChunk).where(
            ActiveChunk.knowledge_base_id == knowledge_base_id,
            ActiveChunk.id.in_(requested_chunk_ids),
            ActiveChunk.state == "active",
        )
    ).all()
    if not chunks:
        return _graph_response("evidence", [], [])

    documents = {
        document.id: document
        for document in db.scalars(
            select(Document).where(
                Document.knowledge_base_id == knowledge_base_id,
                Document.id.in_({chunk.document_id for chunk in chunks if chunk.document_id}),
            )
        ).all()
    }
    nodes: list[dict] = []
    edges: list[dict] = []
    node_ids: set[str] = set()
    seen_edges: set[tuple[str, str, str]] = set()

    def add_node(node: dict) -> None:
        if node["id"] in node_ids:
            return
        node_ids.add(node["id"])
        nodes.append(node)

    def add_edge(source: str, target: str, label: str) -> None:
        key = (source, target, label)
        if key in seen_edges:
            return
        seen_edges.add(key)
        edges.append({"source": source, "target": target, "label": label, "category": "evidence"})

    for chunk in chunks:
        document = documents.get(chunk.document_id)
        chunk_node_id = f"evidence_chunk:{chunk.id}"
        add_node(
            {
                "id": chunk_node_id,
                "name": chunk.snippet[:80] if chunk.snippet else chunk.id,
                "category": "evidence_chunk",
                "value": 1.5,
                "partition": chunk.partition,
                "document_id": chunk.document_id,
                "document_version_id": chunk.document_version_id,
                "snippet": chunk.snippet,
                "page_number": chunk.page_number,
                "source_type": chunk.source_type,
            }
        )
        if chunk.document_version_id:
            version_node_id = f"document_version:{chunk.document_version_id}"
            add_node(
                {
                    "id": version_node_id,
                    "name": document.title if document else chunk.document_version_id,
                    "category": "document_version",
                    "value": 2,
                    "document_id": chunk.document_id,
                    "document_version_id": chunk.document_version_id,
                    "source_type": document.source_type if document else None,
                }
            )
            add_edge(chunk_node_id, version_node_id, "from_version")

    partitions = {chunk.partition for chunk in chunks if chunk.partition}
    return _graph_response("evidence", nodes, edges, partition=next(iter(partitions)) if len(partitions) == 1 else None)


def get_graph_payload(db: Session, knowledge_base_id: str, partition: str | None = None, graph_type: str = "evidence", view: str | None = None) -> dict:
    if graph_type not in GRAPH_TYPES:
        raise ValueError(f"invalid graph_type {graph_type!r}")
    payload = get_evidence_graph_payload(db, knowledge_base_id, partition=partition, view=view)
    payload["freshness"] = graph_freshness(db, knowledge_base_id)
    return payload
