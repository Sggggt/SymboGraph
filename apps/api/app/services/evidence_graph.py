from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session
import networkx as nx

from app.core.config import get_settings
from app.models import (
    ActiveChunk,
    ChunkCandidate,
    ChunkDecision,
    CommunityMembership,
    CommunityState,
    CommunitySummary,
    Document,
    DocumentVersion,
    EvidenceAtom,
    EvidenceEdge,
    EvidenceGraphState,
    ParseJob,
    ProjectionCommunity,
    ProjectionEdge,
    ProjectionNode,
    PolicyObservation,
    PolicyState,
    ProjectionState,
    QualityDecision,
    RewardEvent,
    SignalCandidate,
    SignalCommunity,
    SignalCommunityMembership,
    SignalDecision,
    SignalEdge,
    SignalNode,
    SignalRelationSpec,
    SignalSchemaState,
    SourceFile,
    SignalState,
    SignalTypeSpec,
    VectorRecord,
)
from app.services.chunking import CURRENT_EMBEDDING_TEXT_VERSION
from app.services.evidence_signal_projection import (
    attach_active_chunks_to_signal_layer,
    build_evidence_signal_layer,
    signal_features_for_atoms,
)
from app.services.parsers import ParsedSection


PARSER_PROTOCOL_VERSION = "parser_sections_to_atoms_v1"
ATOM_PROTOCOL_VERSION = "evidence_atom_v1"
EDGE_PROTOCOL_VERSION = "deterministic_observation_edges_v1"
GRAPH_FEATURE_PROTOCOL_VERSION = "graph_features_v1"
CHUNK_GENERATOR_VERSION = "evidence_chunk_region_v1"
SIGNAL_REGION_GENERATOR_VERSION = "signal_region_v1"
SIGNAL_BRIDGE_REPAIR_GENERATOR_VERSION = "signal_bridge_repair_v1"
QUALITY_POLICY_VERSION = "quality_gate_v1"
CHUNK_DECISION_VERSION = "chunk_decision_v1"
COMMUNITY_PROTOCOL_VERSION = "modularity_louvain_v1"
BANDIT_POLICY_VERSION = "constrained_linucb_v1"
PROMPT_PROTOCOL_VERSION = "answer_grounding_v1"
REWARD_PROTOCOL_VERSION = "reward_protocol_v1"
BANDIT_CONTEXT_FEATURES = (
    "bias",
    "internal_cohesion",
    "boundary_safety",
    "signal_coverage",
    "signal_boundary_safety",
    "community_modularity_gain",
    "community_boundary_safety",
    "reference_closure",
    "layout_integrity",
    "token_efficiency",
    "table_code_integrity",
    "signal_region_affinity",
    "quality_confidence",
)
BANDIT_ARMS = (
    "atomic_parent_context",
    "community_region",
    "heading_preserving",
    "semantic_cut",
    "signal_region",
    "signal_bridge_repair",
    "high_recall_overlap",
    "low_overlap_precise",
    "table_code_preserving",
)

OBSERVED_EDGE_TYPES = {
    "ADJACENT",
    "CONTAINS",
    "LAYOUT_CONTINUES",
    "SEMANTIC_SIMILAR",
    "REFERENCE_DEPENDS_ON",
    "MODALITY_LINK",
    "DISCOURSE_SHIFT",
    "LEXICAL_OVERLAP",
    "TOPIC_OVERLAP",
    "DEFINITION_SUPPORT",
    "SYMBOL_REFERENCE",
}

@dataclass
class EvidencePipelineResult:
    graph_state: EvidenceGraphState
    policy_state: PolicyState
    community_state: CommunityState | None
    signal_state: SignalState | None
    projection_state: ProjectionState | None
    chunk_to_active: dict[str, ActiveChunk]
    stats: dict[str, Any]


@dataclass
class ChunkDraft:
    """Parse-time chunk candidate. This is intentionally not a persisted model."""

    id: str
    knowledge_base_id: str
    document_id: str
    document_version_id: str
    chunk_version: int
    content: str
    snippet: str
    partition: str | None
    section: str | None
    page_number: int | None
    token_count: int
    source_type: str
    metadata_json: dict[str, Any]
    parent_chunk_id: str | None = None
    summary: str | None = None
    keywords: list[str] | None = None
    embedding_text_version: str = CURRENT_EMBEDDING_TEXT_VERSION
    embedding_status: str = "pending"
    is_active: bool = False
    created_at: datetime | None = None


def stable_hash(value: Any) -> str:
    data = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def text_hash(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    # Conservative mixed CJK/Latin estimate without adding tokenizer dependency here.
    cjk = sum(1 for char in text if "\u4e00" <= char <= "\u9fff")
    latin_words = len(re.findall(r"[A-Za-z0-9_]+", text))
    symbols = max(len(text) - cjk - sum(len(item) for item in re.findall(r"[A-Za-z0-9_]+", text)), 0)
    return max(1, cjk + latin_words + symbols // 4)


def lexical_terms(text: str) -> set[str]:
    terms = {item.lower() for item in re.findall(r"[\w\u4e00-\u9fff]{2,}", text or "")}
    return {term for term in terms if not re.fullmatch(r"[\d_]+", term)}


def _span_for_block(
    *,
    section: ParsedSection,
    section_index: int,
    block: str,
    cursor: int,
    source_path: str,
    extracted_path: str | None,
    atom_role: str,
) -> tuple[dict[str, Any], int]:
    section_text = section.text or ""
    start = section_text.find(block, cursor)
    if start < 0:
        start = min(max(cursor, 0), len(section_text))
    end = min(start + len(block), len(section_text))
    if start < 0 or end < start or end > len(section_text):
        raise ValueError(f"Invalid source span for section {section_index}: {start}:{end}")
    return (
        {
            "source_path": source_path,
            "extracted_path": extracted_path,
            "section_index": section_index,
            "section": section.section,
            "page_number": section.page_number,
            "start": start,
            "end": end,
            "atom_role": atom_role,
            "parser_protocol_version": PARSER_PROTOCOL_VERSION,
        },
        end,
    )


def infer_atom_type(text: str, metadata: dict[str, Any], *, role: str = "body") -> str:
    if role == "heading":
        return "heading"
    content_kind = str(metadata.get("content_kind") or "").lower()
    if content_kind == "code" or "[Code Cell]" in text:
        return "code_block"
    if metadata.get("has_table") or ("|" in text and re.search(r"\|\s*-{2,}\s*\|", text)):
        return "table_block"
    if metadata.get("has_formula") or re.search(r"[=∑∫√∞≈≠≤≥±×÷→←]", text):
        return "formula"
    if re.match(r"^\s*(?:[-*+]|\d+[.)])\s+", text):
        return "list_item"
    if content_kind in {"slide", "pdf_page"} and len(text) > 1200:
        return "page_block"
    return "paragraph"


def section_blocks(section: ParsedSection) -> list[str]:
    raw_blocks = [block.strip() for block in re.split(r"\n\s*\n", section.text or "") if block.strip()]
    if not raw_blocks and section.text.strip():
        raw_blocks = [section.text.strip()]
    blocks: list[str] = []
    for block in raw_blocks:
        if len(block) <= 1800:
            blocks.append(block)
            continue
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        if len(lines) > 1:
            blocks.extend(lines)
        else:
            blocks.extend(block[i : i + 1500].strip() for i in range(0, len(block), 1500) if block[i : i + 1500].strip())
    return blocks


def build_atom_drafts(
    *,
    sections: list[ParsedSection],
    source_path: str,
    extracted_path: str | None,
) -> list[dict[str, Any]]:
    drafts: list[dict[str, Any]] = []
    atom_index = 0
    for section_index, section in enumerate(sections):
        metadata = dict(section.metadata or {})
        cursor = 0
        title = (section.title or section.section or "").strip()
        if title:
            span = {
                "source_path": source_path,
                "extracted_path": extracted_path,
                "section_index": section_index,
                "section": section.section,
                "page_number": section.page_number,
                "start": 0,
                "end": min(len(title), len(section.text or "")),
                "atom_role": "heading",
                "parser_protocol_version": PARSER_PROTOCOL_VERSION,
            }
            drafts.append(
                {
                    "atom_index": atom_index,
                    "atom_type": "heading",
                    "text": title,
                    "source_span_json": span,
                    "layout_json": {"section_index": section_index, "role": "heading"},
                    "metadata_json": {
                        **metadata,
                        "section_index": section_index,
                        "atom_protocol_version": ATOM_PROTOCOL_VERSION,
                    },
                }
            )
            atom_index += 1
        for block in section_blocks(section):
            span, cursor = _span_for_block(
                section=section,
                section_index=section_index,
                block=block,
                cursor=cursor,
                source_path=source_path,
                extracted_path=extracted_path,
                atom_role="body",
            )
            atom_type = infer_atom_type(block, metadata)
            drafts.append(
                {
                    "atom_index": atom_index,
                    "atom_type": atom_type,
                    "text": block,
                    "source_span_json": span,
                    "layout_json": {
                        "section_index": section_index,
                        "role": "body",
                        "page_number": section.page_number,
                    },
                    "metadata_json": {
                        **metadata,
                        "section_index": section_index,
                        "atom_protocol_version": ATOM_PROTOCOL_VERSION,
                    },
                }
            )
            atom_index += 1
    return drafts


def upsert_source_file(
    db: Session,
    *,
    knowledge_base_id: str,
    document_id: str,
    source_path: Path,
    checksum: str,
    source_type: str,
) -> SourceFile:
    source = db.scalar(
        select(SourceFile).where(
            SourceFile.knowledge_base_id == knowledge_base_id,
            SourceFile.source_path == str(source_path),
        )
    )
    if source is None:
        source = SourceFile(
            knowledge_base_id=knowledge_base_id,
            document_id=document_id,
            source_path=str(source_path),
            checksum=checksum,
            source_type=source_type,
            size_bytes=source_path.stat().st_size if source_path.exists() else 0,
            metadata_json={},
            state="active",
        )
        db.add(source)
    else:
        source.document_id = document_id
        source.checksum = checksum
        source.source_type = source_type
        source.size_bytes = source_path.stat().st_size if source_path.exists() else source.size_bytes
        source.state = "active"
    db.flush()
    return source


def record_parse_job(
    db: Session,
    *,
    knowledge_base_id: str,
    document_id: str,
    document_version_id: str,
    ingestion_job_id: str | None,
    source_file_id: str | None,
    sections: list[ParsedSection],
) -> ParseJob:
    parse_job = ParseJob(
        knowledge_base_id=knowledge_base_id,
        document_id=document_id,
        document_version_id=document_version_id,
        ingestion_job_id=ingestion_job_id,
        source_file_id=source_file_id,
        parser_protocol_version=PARSER_PROTOCOL_VERSION,
        status="completed",
        stats_json={
            "section_count": len(sections),
            "atom_protocol_version": ATOM_PROTOCOL_VERSION,
            "discard_diagnostics": [],
        },
        diagnostics_json={},
        started_at=datetime.utcnow(),
        completed_at=datetime.utcnow(),
    )
    db.add(parse_job)
    db.flush()
    return parse_job


def clear_document_version_evidence(db: Session, *, knowledge_base_id: str, document_version_id: str) -> None:
    states = db.scalars(select(EvidenceGraphState).where(EvidenceGraphState.knowledge_base_id == knowledge_base_id)).all()
    graph_state_ids = [
        state.id
        for state in states
        if getattr(state, "scope_type", "document") == "document"
        if document_version_id in set(str(item) for item in (state.active_document_version_ids or []))
    ]
    if graph_state_ids:
        decision_ids = [
            row[0]
            for row in db.execute(select(ChunkDecision.id).where(ChunkDecision.graph_state_id.in_(graph_state_ids))).all()
        ]
        if decision_ids:
            db.query(ActiveChunk).filter(ActiveChunk.chunk_decision_id.in_(decision_ids)).delete(synchronize_session=False)
        candidate_ids = [
            row[0]
            for row in db.execute(select(ChunkCandidate.id).where(ChunkCandidate.graph_state_id.in_(graph_state_ids))).all()
        ]
        if candidate_ids:
            db.query(QualityDecision).filter(QualityDecision.candidate_id.in_(candidate_ids)).delete(synchronize_session=False)
            db.query(ChunkCandidate).filter(ChunkCandidate.id.in_(candidate_ids)).delete(synchronize_session=False)
        db.query(ChunkDecision).filter(ChunkDecision.graph_state_id.in_(graph_state_ids)).delete(synchronize_session=False)
        db.query(EvidenceEdge).filter(EvidenceEdge.graph_state_id.in_(graph_state_ids)).delete(synchronize_session=False)
        db.query(CommunityMembership).filter(
            CommunityMembership.community_state_id.in_(
                select(CommunityState.id).where(CommunityState.graph_state_id.in_(graph_state_ids))
            )
        ).delete(synchronize_session=False)
        db.query(CommunitySummary).filter(
            CommunitySummary.community_state_id.in_(
                select(CommunityState.id).where(CommunityState.graph_state_id.in_(graph_state_ids))
            )
        ).delete(synchronize_session=False)
        db.query(CommunityState).filter(CommunityState.graph_state_id.in_(graph_state_ids)).delete(synchronize_session=False)
        signal_state_ids = list(
            db.scalars(select(SignalState.id).where(SignalState.evidence_graph_state_id.in_(graph_state_ids))).all()
        )
        if signal_state_ids:
            projection_state_ids = list(
                db.scalars(select(ProjectionState.id).where(ProjectionState.signal_state_id.in_(signal_state_ids))).all()
            )
            if projection_state_ids:
                db.query(ProjectionEdge).filter(ProjectionEdge.projection_state_id.in_(projection_state_ids)).delete(synchronize_session=False)
                db.query(ProjectionNode).filter(ProjectionNode.projection_state_id.in_(projection_state_ids)).delete(synchronize_session=False)
                db.query(ProjectionCommunity).filter(ProjectionCommunity.projection_state_id.in_(projection_state_ids)).delete(synchronize_session=False)
                db.query(ProjectionState).filter(ProjectionState.id.in_(projection_state_ids)).delete(synchronize_session=False)
            signal_community_ids = list(
                db.scalars(select(SignalCommunity.id).where(SignalCommunity.signal_state_id.in_(signal_state_ids))).all()
            )
            if signal_community_ids:
                db.query(SignalCommunityMembership).filter(SignalCommunityMembership.signal_community_id.in_(signal_community_ids)).delete(synchronize_session=False)
                db.query(SignalCommunity).filter(SignalCommunity.id.in_(signal_community_ids)).delete(synchronize_session=False)
            db.query(SignalEdge).filter(SignalEdge.signal_state_id.in_(signal_state_ids)).delete(synchronize_session=False)
            db.query(SignalNode).filter(SignalNode.signal_state_id.in_(signal_state_ids)).delete(synchronize_session=False)
            db.query(SignalDecision).filter(SignalDecision.signal_state_id.in_(signal_state_ids)).delete(synchronize_session=False)
            db.query(SignalCandidate).filter(SignalCandidate.signal_state_id.in_(signal_state_ids)).delete(synchronize_session=False)
            schema_state_ids = list(
                db.scalars(select(SignalSchemaState.id).where(SignalSchemaState.evidence_graph_state_id.in_(graph_state_ids))).all()
            )
            if schema_state_ids:
                db.query(SignalTypeSpec).filter(SignalTypeSpec.schema_state_id.in_(schema_state_ids)).delete(synchronize_session=False)
                db.query(SignalRelationSpec).filter(SignalRelationSpec.schema_state_id.in_(schema_state_ids)).delete(synchronize_session=False)
            db.query(SignalState).filter(SignalState.id.in_(signal_state_ids)).delete(synchronize_session=False)
            if schema_state_ids:
                db.query(SignalSchemaState).filter(SignalSchemaState.id.in_(schema_state_ids)).delete(synchronize_session=False)
        db.query(EvidenceGraphState).filter(EvidenceGraphState.id.in_(graph_state_ids)).delete(synchronize_session=False)
    db.query(EvidenceAtom).filter(
        EvidenceAtom.knowledge_base_id == knowledge_base_id,
        EvidenceAtom.document_version_id == document_version_id,
    ).delete(synchronize_session=False)
    db.flush()


def create_atoms(
    db: Session,
    *,
    knowledge_base_id: str,
    document_id: str,
    document_version_id: str,
    sections: list[ParsedSection],
    source_path: str,
    extracted_path: str | None,
) -> list[EvidenceAtom]:
    drafts = build_atom_drafts(sections=sections, source_path=source_path, extracted_path=extracted_path)
    atoms: list[EvidenceAtom] = []
    for draft in drafts:
        text = draft["text"]
        atom = EvidenceAtom(
            knowledge_base_id=knowledge_base_id,
            document_id=document_id,
            document_version_id=document_version_id,
            atom_index=draft["atom_index"],
            atom_type=draft["atom_type"],
            text=text,
            text_hash=text_hash(text),
            source_span_json=draft["source_span_json"],
            layout_json=draft["layout_json"],
            parser_confidence=1.0,
            metadata_json=draft["metadata_json"],
            state="active",
        )
        db.add(atom)
        atoms.append(atom)
    db.flush()
    return atoms


def _policy_identity_hash(*, knowledge_base_id: str, objective_hash: str) -> str:
    return stable_hash(
        {
            "knowledge_base_id": knowledge_base_id,
            "policy_version": BANDIT_POLICY_VERSION,
            "profile_objective_hash": objective_hash,
        }
    )


def _fresh_bandit_posterior() -> dict[str, Any]:
    dimension = len(BANDIT_CONTEXT_FEATURES)
    return {
        "protocol_version": REWARD_PROTOCOL_VERSION,
        "policy_algorithm": "diagonal_linucb",
        "context_features": list(BANDIT_CONTEXT_FEATURES),
        "arms": {
            arm: {
                "A_diag": [1.0 for _ in range(dimension)],
                "b": [0.0 for _ in range(dimension)],
                "count": 0,
                "reward_sum": 0.0,
            }
            for arm in BANDIT_ARMS
        },
    }


def _normalize_policy_posterior(policy_state: PolicyState) -> dict[str, Any]:
    posterior = dict(policy_state.posterior_json or {})
    dimension = len(BANDIT_CONTEXT_FEATURES)
    arms = dict(posterior.get("arms") or {})
    changed = False
    for arm in BANDIT_ARMS:
        arm_state = dict(arms.get(arm) or {})
        a_diag = [float(value) for value in list(arm_state.get("A_diag") or [])[:dimension]]
        b_vec = [float(value) for value in list(arm_state.get("b") or [])[:dimension]]
        if len(a_diag) < dimension:
            a_diag.extend([1.0 for _ in range(dimension - len(a_diag))])
            changed = True
        if len(b_vec) < dimension:
            b_vec.extend([0.0 for _ in range(dimension - len(b_vec))])
            changed = True
        if "count" not in arm_state:
            arm_state["count"] = 0
            changed = True
        if "reward_sum" not in arm_state:
            arm_state["reward_sum"] = 0.0
            changed = True
        arm_state["A_diag"] = a_diag
        arm_state["b"] = b_vec
        arms[arm] = arm_state
    if posterior.get("context_features") != list(BANDIT_CONTEXT_FEATURES):
        posterior["context_features"] = list(BANDIT_CONTEXT_FEATURES)
        changed = True
    if posterior.get("policy_algorithm") != "diagonal_linucb":
        posterior["policy_algorithm"] = "diagonal_linucb"
        changed = True
    if posterior.get("protocol_version") != REWARD_PROTOCOL_VERSION:
        posterior["protocol_version"] = REWARD_PROTOCOL_VERSION
        changed = True
    posterior["arms"] = arms
    if changed:
        policy_state.posterior_json = posterior
    return posterior


def _policy_posterior_hash(policy_state: PolicyState) -> str:
    return stable_hash(policy_state.posterior_json or {})


def ensure_policy_state(db: Session, *, knowledge_base_id: str, profile_objective_hash: str | None = None) -> PolicyState:
    objective_hash = profile_objective_hash or stable_hash({"objective": "evidence_first", "domain_hint": "general"})
    state_hash = _policy_identity_hash(knowledge_base_id=knowledge_base_id, objective_hash=objective_hash)
    existing = db.scalar(
        select(PolicyState)
        .where(
            PolicyState.knowledge_base_id == knowledge_base_id,
            PolicyState.policy_version == BANDIT_POLICY_VERSION,
            PolicyState.profile_objective_hash == objective_hash,
        )
        .order_by(PolicyState.created_at.desc())
    )
    if existing is not None:
        _normalize_policy_posterior(existing)
        existing.reward_summary_json = {
            **(existing.reward_summary_json or {}),
            "posterior_hash": _policy_posterior_hash(existing),
            "context_features": list(BANDIT_CONTEXT_FEATURES),
        }
        return existing
    policy = PolicyState(
        knowledge_base_id=knowledge_base_id,
        policy_family="constrained_linucb",
        policy_version=BANDIT_POLICY_VERSION,
        profile_objective_hash=objective_hash,
        posterior_json=_fresh_bandit_posterior(),
        constraints_json={
            "safe_baseline": "atomic_parent_context",
            "max_candidate_tokens": int(getattr(get_settings(), "chunk_token_budget", 2400) or 2400),
            "allowed_arms": list(BANDIT_ARMS),
        },
        exploration_json={"alpha": 0.25, "rate": 0.05, "algorithm": "linucb_ucb"},
        reward_summary_json={
            "protocol_version": REWARD_PROTOCOL_VERSION,
            "events": 0,
            "observations": 0,
            "posterior_hash": stable_hash(_fresh_bandit_posterior()),
            "context_features": list(BANDIT_CONTEXT_FEATURES),
        },
        drift_status="fresh",
        state_hash=state_hash,
    )
    db.add(policy)
    db.flush()
    return policy


def create_graph_state(
    db: Session,
    *,
    knowledge_base_id: str,
    atoms: list[EvidenceAtom],
    policy_state: PolicyState,
    scope_type: str = "document",
    initial_state: str = "active",
) -> EvidenceGraphState:
    atom_scope = [
        {
            "id": atom.id,
            "document_version_id": atom.document_version_id,
            "text_hash": atom.text_hash,
            "state": atom.state,
        }
        for atom in atoms
    ]
    atom_scope_hash = stable_hash(atom_scope)
    document_version_ids = sorted({atom.document_version_id for atom in atoms})
    state_hash = stable_hash(
        {
            "atom_scope_hash": atom_scope_hash,
            "edge_protocol_version": EDGE_PROTOCOL_VERSION,
            "parser_protocol_version": PARSER_PROTOCOL_VERSION,
            "embedding_text_version": CURRENT_EMBEDDING_TEXT_VERSION,
            "policy_state_hash": policy_state.state_hash,
            "prompt_protocol_version": PROMPT_PROTOCOL_VERSION,
            "scope_type": scope_type,
        }
    )
    graph_state = EvidenceGraphState(
        knowledge_base_id=knowledge_base_id,
        scope_type=scope_type,
        state_hash=state_hash,
        atom_scope_hash=atom_scope_hash,
        edge_protocol_version=EDGE_PROTOCOL_VERSION,
        parser_protocol_version=PARSER_PROTOCOL_VERSION,
        embedding_text_version=CURRENT_EMBEDDING_TEXT_VERSION,
        active_document_version_ids=document_version_ids,
        active_atom_ids=[atom.id for atom in atoms],
        policy_state_id=policy_state.id,
        prompt_protocol_version=PROMPT_PROTOCOL_VERSION,
        stats_json={"atom_count": len(atoms), "edge_count": 0},
        diagnostics_json={},
        state=initial_state,
    )
    db.add(graph_state)
    db.flush()
    return graph_state


def _refresh_graph_state_hash(graph_state: EvidenceGraphState, *, signal_state_hash: str | None = None) -> None:
    signal_hash = signal_state_hash or (graph_state.stats_json or {}).get("signal_state_hash")
    graph_state.state_hash = stable_hash(
        {
            "atom_scope_hash": graph_state.atom_scope_hash,
            "edge_protocol_version": graph_state.edge_protocol_version,
            "parser_protocol_version": graph_state.parser_protocol_version,
            "embedding_text_version": graph_state.embedding_text_version,
            "policy_state_id": graph_state.policy_state_id,
            "prompt_protocol_version": graph_state.prompt_protocol_version,
            "signal_state_hash": signal_hash,
            "community_state_id": graph_state.community_state_id,
            "scope_type": getattr(graph_state, "scope_type", "document"),
        }
    )


def attach_signal_layer_to_graph_state(graph_state: EvidenceGraphState, signal_state: SignalState, projection_state: ProjectionState | None = None) -> None:
    _refresh_graph_state_hash(graph_state, signal_state_hash=signal_state.signal_state_hash)
    graph_state.stats_json = {
        **(graph_state.stats_json or {}),
        "signal_state_id": signal_state.id,
        "signal_state_hash": signal_state.signal_state_hash,
        "signal_layer_status": signal_state.status,
        "signal_node_count": int((signal_state.stats_json or {}).get("signal_node_count") or 0),
        "signal_edge_count": int((signal_state.stats_json or {}).get("signal_edge_count") or 0),
        "signal_candidate_count": int((signal_state.stats_json or {}).get("signal_candidate_count") or 0),
        "projection_state_id": projection_state.id if projection_state else None,
        "projection_hash": projection_state.projection_hash if projection_state else None,
    }


def create_edge(
    graph_state: EvidenceGraphState,
    source: EvidenceAtom,
    target: EvidenceAtom,
    edge_type: str,
    *,
    weight: float,
    confidence: float,
    features: dict[str, Any] | None = None,
    evidence: dict[str, Any] | None = None,
) -> EvidenceEdge:
    if edge_type not in OBSERVED_EDGE_TYPES:
        raise ValueError(f"Unsupported evidence edge type: {edge_type}")
    return EvidenceEdge(
        graph_state_id=graph_state.id,
        source_atom_id=source.id,
        target_atom_id=target.id,
        edge_type=edge_type,
        weight=max(0.0, min(1.0, weight)),
        confidence=max(0.0, min(1.0, confidence)),
        features_json=features or {},
        evidence_json=evidence or {},
    )


def _global_observation_edge_candidates(atoms: list[EvidenceAtom]) -> list[tuple[EvidenceAtom, EvidenceAtom, str, float, float, dict[str, Any]]]:
    by_id = {atom.id: atom for atom in atoms}
    term_index: dict[str, list[EvidenceAtom]] = defaultdict(list)
    definition_index: dict[str, list[EvidenceAtom]] = defaultdict(list)
    symbol_index: dict[str, list[EvidenceAtom]] = defaultdict(list)
    for atom in atoms:
        terms = lexical_terms(atom.text)
        for term in terms:
            if len(term) >= 4:
                term_index[term].append(atom)
        if re.search(r"\b(?:is|are|means|refers to|denotes|represents)\b|(?:鏄寚|瀹氫箟|琛ㄧず)", atom.text or ""):
            for term in terms:
                definition_index[term].append(atom)
        if atom.atom_type in {"formula", "code_block"}:
            for symbol in re.findall(r"\b[A-Za-z_][A-Za-z0-9_]{1,24}\b", atom.text or ""):
                symbol_index[symbol.lower()].append(atom)

    pair_scores: dict[tuple[str, str, str], dict[str, Any]] = {}

    def add_pair(left: EvidenceAtom, right: EvidenceAtom, edge_type: str, increment: float, reason: str) -> None:
        if left.id == right.id:
            return
        ordered = tuple(sorted([left.id, right.id]))
        key = (ordered[0], ordered[1], edge_type)
        current = pair_scores.setdefault(
            key,
            {
                "score": 0.0,
                "reasons": set(),
                "source": by_id[ordered[0]],
                "target": by_id[ordered[1]],
            },
        )
        current["score"] += increment
        current["reasons"].add(reason)

    for term, grouped in term_index.items():
        if len(grouped) < 2 or len(grouped) > 80:
            continue
        ordered = sorted(grouped, key=lambda atom: (atom.document_id, atom.atom_index))
        for index, left in enumerate(ordered):
            for right in ordered[index + 1 : index + 9]:
                if left.document_id == right.document_id and abs(left.atom_index - right.atom_index) <= 3:
                    continue
                add_pair(left, right, "TOPIC_OVERLAP", 0.18, term)

    for term, grouped in definition_index.items():
        if len(grouped) < 2 or len(grouped) > 40:
            continue
        for index, left in enumerate(grouped[:40]):
            for right in grouped[index + 1 : index + 8]:
                add_pair(left, right, "DEFINITION_SUPPORT", 0.24, term)

    for symbol, grouped in symbol_index.items():
        if len(grouped) < 2 or len(grouped) > 60:
            continue
        for index, left in enumerate(grouped[:60]):
            for right in grouped[index + 1 : index + 8]:
                add_pair(left, right, "SYMBOL_REFERENCE", 0.22, symbol)

    per_atom_counts: Counter[str] = Counter()
    candidates: list[tuple[EvidenceAtom, EvidenceAtom, str, float, float, dict[str, Any]]] = []
    for (_left_id, _right_id, edge_type), payload in sorted(pair_scores.items(), key=lambda item: item[1]["score"], reverse=True):
        left = payload["source"]
        right = payload["target"]
        if per_atom_counts[left.id] >= 8 or per_atom_counts[right.id] >= 8:
            continue
        score = float(payload["score"])
        weight = min(0.82, 0.28 + score)
        confidence = min(0.86, 0.52 + score * 0.45)
        reasons = sorted(str(reason) for reason in payload["reasons"])[:12]
        candidates.append((left, right, edge_type, weight, confidence, {"shared_evidence_keys": reasons, "global_observation": True}))
        per_atom_counts[left.id] += 1
        per_atom_counts[right.id] += 1
        if len(candidates) >= max(500, len(atoms) * 8):
            break
    return candidates


def generate_edges(db: Session, *, graph_state: EvidenceGraphState, atoms: list[EvidenceAtom]) -> list[EvidenceEdge]:
    edges: list[EvidenceEdge] = []
    ordered = sorted(atoms, key=lambda atom: (atom.document_id, atom.atom_index))
    last_heading_by_section: dict[tuple[str, int], EvidenceAtom] = {}
    for left, right in zip(ordered, ordered[1:]):
        if left.document_version_id == right.document_version_id:
            distance = max(1, right.atom_index - left.atom_index)
            edges.append(
                create_edge(
                    graph_state,
                    left,
                    right,
                    "ADJACENT",
                    weight=1.0 / distance,
                    confidence=1.0,
                    features={"adjacency_distance": distance},
                    evidence={"observation": "sequential_parser_order"},
                )
            )
    for atom in ordered:
        section_index = int((atom.metadata_json or {}).get("section_index") or 0)
        key = (atom.document_version_id, section_index)
        if atom.atom_type == "heading":
            last_heading_by_section[key] = atom
            continue
        heading = last_heading_by_section.get(key)
        if heading is not None:
            edges.append(
                create_edge(
                    graph_state,
                    heading,
                    atom,
                    "CONTAINS",
                    weight=0.92,
                    confidence=0.95,
                    features={"layout_role": "heading_contains_body"},
                    evidence={"section_index": section_index},
                )
            )
    for left, right in zip(ordered, ordered[1:]):
        if left.document_version_id != right.document_version_id:
            continue
        left_meta = left.metadata_json or {}
        right_meta = right.metadata_json or {}
        if left_meta.get("has_table") and right_meta.get("has_table"):
            edges.append(create_edge(graph_state, left, right, "LAYOUT_CONTINUES", weight=0.86, confidence=0.85))
        if left.atom_type in {"table_block", "formula", "code_block"} or right.atom_type in {"table_block", "formula", "code_block"}:
            edges.append(create_edge(graph_state, left, right, "MODALITY_LINK", weight=0.74, confidence=0.8))
    max_pairs = 2500
    pair_count = 0
    for index, left in enumerate(ordered):
        left_terms = lexical_terms(left.text)
        if not left_terms:
            continue
        for right in ordered[index + 1 : min(len(ordered), index + 24)]:
            pair_count += 1
            if pair_count > max_pairs:
                break
            if left.document_version_id != right.document_version_id:
                continue
            right_terms = lexical_terms(right.text)
            if not right_terms:
                continue
            overlap = len(left_terms & right_terms) / max(len(left_terms | right_terms), 1)
            if overlap >= 0.22:
                edges.append(
                    create_edge(
                        graph_state,
                        left,
                        right,
                        "LEXICAL_OVERLAP",
                        weight=min(1.0, overlap),
                        confidence=0.75,
                        features={"jaccard": round(overlap, 4)},
                    )
                )
        if pair_count > max_pairs:
            break
    for edge in edges:
        db.add(edge)
    if getattr(graph_state, "scope_type", "document") == "global":
        for left, right, edge_type, weight, confidence, features in _global_observation_edge_candidates(ordered):
            edge = create_edge(
                graph_state,
                left,
                right,
                edge_type,
                weight=weight,
                confidence=confidence,
                features=features,
                evidence={"observation": "bounded_global_inverted_index"},
            )
            db.add(edge)
            edges.append(edge)
    graph_state.stats_json = {**(graph_state.stats_json or {}), "edge_count": len(edges)}
    db.flush()
    return edges


def source_span_union(atoms: list[EvidenceAtom]) -> dict[str, Any]:
    spans = [atom.source_span_json or {} for atom in atoms if atom.source_span_json]
    document_version_ids = sorted({atom.document_version_id for atom in atoms})
    return {
        "document_version_ids": document_version_ids,
        "source_paths": sorted({str(span.get("source_path")) for span in spans if span.get("source_path")}),
        "spans": spans,
        "atom_count": len(atoms),
    }


def subgraph_features(
    db: Session,
    candidate_atoms: list[EvidenceAtom],
    edges: list[EvidenceEdge],
    *,
    graph_state: EvidenceGraphState | None = None,
    signal_state: SignalState | None = None,
) -> dict[str, Any]:
    atom_ids = {atom.id for atom in candidate_atoms}
    internal = [edge for edge in edges if edge.source_atom_id in atom_ids and edge.target_atom_id in atom_ids]
    outgoing = [
        edge
        for edge in edges
        if (edge.source_atom_id in atom_ids) != (edge.target_atom_id in atom_ids)
    ]
    internal_cohesion = sum(edge.weight * edge.confidence for edge in internal) / max(len(internal), 1)
    boundary_cut_cost = sum(edge.weight * edge.confidence for edge in outgoing)
    reference_edges = [edge for edge in edges if edge.source_atom_id in atom_ids and edge.edge_type == "REFERENCE_DEPENDS_ON"]
    broken_reference_edges = [edge for edge in reference_edges if edge.target_atom_id not in atom_ids]
    layout_edges = [edge for edge in edges if edge.edge_type == "LAYOUT_CONTINUES" and (edge.source_atom_id in atom_ids or edge.target_atom_id in atom_ids)]
    broken_layout_edges = [edge for edge in layout_edges if (edge.source_atom_id in atom_ids) != (edge.target_atom_id in atom_ids)]
    token_count = sum(estimate_tokens(atom.text) for atom in candidate_atoms)
    signal_features = signal_features_for_atoms(db, signal_state=signal_state, atoms=candidate_atoms)
    community_features = community_features_for_atoms(db, graph_state=graph_state, atoms=candidate_atoms)
    return {
        "feature_protocol_version": GRAPH_FEATURE_PROTOCOL_VERSION,
        "internal_cohesion": round(internal_cohesion, 6),
        "boundary_cut_cost": round(boundary_cut_cost, 6),
        "reference_closure": round(1.0 - (len(broken_reference_edges) / max(len(reference_edges), 1)), 6),
        "layout_integrity": round(1.0 - (len(broken_layout_edges) / max(len(layout_edges), 1)), 6),
        "token_efficiency": round(len(candidate_atoms) / max(token_count, 1), 6),
        "atom_count": len(candidate_atoms),
        "token_count": token_count,
        "has_table": any(atom.atom_type == "table_block" for atom in candidate_atoms),
        "has_code": any(atom.atom_type == "code_block" for atom in candidate_atoms),
        "has_formula": any(atom.atom_type == "formula" for atom in candidate_atoms),
        **community_features,
        **signal_features,
    }


def community_features_for_atoms(
    db: Session,
    *,
    graph_state: EvidenceGraphState | None,
    atoms: list[EvidenceAtom],
) -> dict[str, Any]:
    atom_ids = {atom.id for atom in atoms}
    if graph_state is None or not graph_state.community_state_id or not atom_ids:
        return {
            "community_state_id": None,
            "community_modularity_q": 0.0,
            "community_modularity_gain": 0.0,
            "community_boundary_penalty": 0.0,
            "dominant_community_ids": [],
        }
    state = db.get(CommunityState, graph_state.community_state_id)
    memberships = db.scalars(
        select(CommunityMembership).where(
            CommunityMembership.community_state_id == graph_state.community_state_id,
            CommunityMembership.atom_id.in_(atom_ids),
        )
    ).all()
    counts = Counter(membership.community_id for membership in memberships)
    dominant = [community_id for community_id, _count in counts.most_common(4)]
    modularity_q = float((state.diagnostics_json or {}).get("modularity_q") or 0.0) if state else 0.0
    largest_share = (counts.most_common(1)[0][1] / max(len(atom_ids), 1)) if counts else 0.0
    boundary_penalty = max(0.0, min(1.0, 1.0 - largest_share))
    return {
        "community_state_id": graph_state.community_state_id,
        "community_modularity_q": round(modularity_q, 6),
        "community_modularity_gain": round(max(0.0, modularity_q) * largest_share, 6),
        "community_boundary_penalty": round(boundary_penalty, 6),
        "dominant_community_ids": dominant,
    }


def atoms_for_chunk(chunk: ChunkDraft, atoms: list[EvidenceAtom]) -> list[EvidenceAtom]:
    scoped_atoms = [atom for atom in atoms if atom.document_version_id == chunk.document_version_id]
    atoms = scoped_atoms or atoms
    metadata = chunk.metadata_json or {}
    section_index = metadata.get("section_index")
    if section_index is not None:
        matched = [atom for atom in atoms if (atom.metadata_json or {}).get("section_index") == section_index]
        if matched:
            return matched
    chunk_terms = lexical_terms(chunk.content)
    scored: list[tuple[float, EvidenceAtom]] = []
    for atom in atoms:
        atom_terms = lexical_terms(atom.text)
        overlap = len(chunk_terms & atom_terms) / max(len(atom_terms | chunk_terms), 1)
        if overlap > 0:
            scored.append((overlap, atom))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [atom for _score, atom in scored[:8]] or atoms[:1]


def create_candidate_for_chunk(
    db: Session,
    *,
    graph_state: EvidenceGraphState,
    chunk: ChunkDraft,
    atoms: list[EvidenceAtom],
    edges: list[EvidenceEdge],
    signal_state: SignalState | None = None,
    generator_name: str = "evidence_chunk_region",
    generator_version: str = CHUNK_GENERATOR_VERSION,
    feedback_driven: bool = False,
) -> ChunkCandidate:
    candidate_atoms = atoms_for_chunk(chunk, atoms)
    features = subgraph_features(db, candidate_atoms, edges, graph_state=graph_state, signal_state=signal_state)
    candidate = ChunkCandidate(
        graph_state_id=graph_state.id,
        generator_name=generator_name,
        generator_version=generator_version,
        atom_ids_json=[atom.id for atom in candidate_atoms],
        source_span_union_json=source_span_union(candidate_atoms),
        token_count=estimate_tokens(chunk.content),
        graph_features_json=features,
        cost_json={
            "token_count": estimate_tokens(chunk.content),
            "latency_cost": min(1.0, estimate_tokens(chunk.content) / 4000.0),
            "duplicated_atoms": 0,
        },
        diagnostics_json={
            "chunk_draft_id": chunk.id,
            "connected_or_explained": True,
            "bridge_reason": generator_name,
            "signal_state_id": signal_state.id if signal_state else None,
            "signal_state_hash": signal_state.signal_state_hash if signal_state else None,
            "feedback_driven": feedback_driven,
        },
        feedback_driven=feedback_driven,
    )
    db.add(candidate)
    db.flush()
    return candidate


def create_signal_candidates_for_chunk(
    db: Session,
    *,
    graph_state: EvidenceGraphState,
    chunk: ChunkDraft,
    atoms: list[EvidenceAtom],
    edges: list[EvidenceEdge],
    signal_state: SignalState | None,
    previous_feedback: dict[str, Any] | None = None,
) -> list[ChunkCandidate]:
    if signal_state is None or signal_state.status != "active":
        return []
    candidate_atoms = atoms_for_chunk(chunk, atoms)
    features = subgraph_features(db, candidate_atoms, edges, graph_state=graph_state, signal_state=signal_state)
    if not features.get("dominant_signal_ids"):
        return []
    suggested_generators = set((previous_feedback or {}).get("suggested_generators") or [])
    candidates: list[ChunkCandidate] = []
    signal_region = ChunkCandidate(
        graph_state_id=graph_state.id,
        generator_name="signal_region",
        generator_version=SIGNAL_REGION_GENERATOR_VERSION,
        atom_ids_json=[atom.id for atom in candidate_atoms],
        source_span_union_json=source_span_union(candidate_atoms),
        token_count=estimate_tokens(chunk.content),
        graph_features_json={**features, "candidate_generator": "signal_region"},
        cost_json={
            "token_count": estimate_tokens(chunk.content),
            "latency_cost": min(1.0, estimate_tokens(chunk.content) / 4000.0),
            "duplicated_atoms": 0,
        },
        diagnostics_json={
            "chunk_draft_id": chunk.id,
            "connected_or_explained": True,
            "bridge_reason": "signal_region",
            "signal_state_id": signal_state.id,
        },
        feedback_driven=False,
    )
    db.add(signal_region)
    candidates.append(signal_region)
    should_repair_signal_bridge = float(features.get("signal_boundary_cut_cost") or 0.0) > 0 or "signal_bridge_repair" in suggested_generators
    if should_repair_signal_bridge:
        repair = ChunkCandidate(
            graph_state_id=graph_state.id,
            generator_name="signal_bridge_repair",
            generator_version=SIGNAL_BRIDGE_REPAIR_GENERATOR_VERSION,
            atom_ids_json=[atom.id for atom in candidate_atoms],
            source_span_union_json=source_span_union(candidate_atoms),
            token_count=estimate_tokens(chunk.content),
            graph_features_json={
                **features,
                "candidate_generator": "signal_bridge_repair",
                "repair_reason": "signal_boundary_cut" if float(features.get("signal_boundary_cut_cost") or 0.0) > 0 else "quality_feedback",
            },
            cost_json={
                "token_count": estimate_tokens(chunk.content),
                "latency_cost": min(1.0, estimate_tokens(chunk.content) / 3600.0),
                "duplicated_atoms": 0,
            },
            diagnostics_json={
                "chunk_draft_id": chunk.id,
                "connected_or_explained": True,
                "bridge_reason": "signal_bridge_repair",
                "signal_state_id": signal_state.id,
                "feedback_driven": "signal_bridge_repair" in suggested_generators,
            },
            feedback_driven="signal_bridge_repair" in suggested_generators,
        )
        db.add(repair)
        candidates.append(repair)
    db.flush()
    return candidates


def latest_quality_feedback(db: Session, knowledge_base_id: str) -> dict[str, Any] | None:
    decision = db.scalar(
        select(QualityDecision)
        .join(ChunkCandidate, ChunkCandidate.id == QualityDecision.candidate_id)
        .join(EvidenceGraphState, EvidenceGraphState.id == ChunkCandidate.graph_state_id)
        .where(EvidenceGraphState.knowledge_base_id == knowledge_base_id)
        .order_by(QualityDecision.created_at.desc())
        .limit(1)
    )
    if decision is None or not isinstance(decision.feedback_json, dict):
        return None
    return dict(decision.feedback_json)


def create_feedback_candidates_for_chunk(
    db: Session,
    *,
    graph_state: EvidenceGraphState,
    chunk: ChunkDraft,
    atoms: list[EvidenceAtom],
    edges: list[EvidenceEdge],
    signal_state: SignalState | None,
    previous_feedback: dict[str, Any] | None,
) -> list[ChunkCandidate]:
    suggested_generators = set((previous_feedback or {}).get("suggested_generators") or [])
    candidates: list[ChunkCandidate] = []
    for generator_name in ("dependency_closure", "community_region"):
        if generator_name not in suggested_generators:
            continue
        candidates.append(
            create_candidate_for_chunk(
                db,
                graph_state=graph_state,
                chunk=chunk,
                atoms=atoms,
                edges=edges,
                signal_state=signal_state,
                generator_name=generator_name,
                generator_version=f"{CHUNK_GENERATOR_VERSION}:feedback",
                feedback_driven=True,
            )
        )
    return candidates


def decide_candidate_quality(
    candidate: ChunkCandidate,
    atoms: list[EvidenceAtom],
    policy_state: PolicyState,
    previous_feedback: dict[str, Any] | None = None,
) -> dict[str, Any]:
    atom_ids = set(candidate.atom_ids_json or [])
    candidate_atoms = [atom for atom in atoms if atom.id in atom_ids]
    risk_flags: list[str] = []
    if not candidate_atoms:
        risk_flags.append("missing_atoms")
    if not (candidate.source_span_union_json or {}).get("spans"):
        risk_flags.append("missing_source_span")
    if any(atom.state != "active" for atom in candidate_atoms):
        risk_flags.append("inactive_atom")
    features = candidate.graph_features_json or {}
    if float(features.get("layout_integrity", 1.0)) < 1.0:
        risk_flags.append("broken_layout_integrity")
    if float(features.get("reference_closure", 1.0)) < 1.0:
        risk_flags.append("broken_reference_closure")
    max_tokens = int((policy_state.constraints_json or {}).get("max_candidate_tokens") or 2400)
    if (candidate.token_count or 0) > max_tokens:
        risk_flags.append("token_budget_exceeded")
    signal_feedback_flags: list[str] = []
    if float(features.get("signal_boundary_cut_cost") or 0.0) > 0:
        signal_feedback_flags.append("signal_boundary_cut")
    if float(features.get("signal_fragmentation") or 0.0) > 2.5:
        signal_feedback_flags.append("high_signal_fragmentation")
    if float(features.get("community_boundary_penalty") or 0.0) > 0.45:
        signal_feedback_flags.append("community_boundary_cut")
    gate_passed = not risk_flags
    if not gate_passed and risk_flags == ["token_budget_exceeded"]:
        action = "needs_rechunk"
    elif not gate_passed:
        action = "reject"
    elif features.get("has_table") or features.get("has_code") or features.get("has_formula"):
        action = "answer_candidate"
    else:
        action = "retrieval_candidate"
    return {
        "gate_passed": gate_passed,
        "decision_action": action,
        "risk_flags": risk_flags,
        "diagnostics": {
            "quality_policy_version": QUALITY_POLICY_VERSION,
            "feedback_driven": bool(candidate.feedback_driven),
            "previous_feedback_generators": list((previous_feedback or {}).get("suggested_generators") or []),
            "hard_constraints": [
                "source_span",
                "active_atoms",
                "layout_integrity",
                "reference_closure",
                "token_budget",
                "traceable_evidence",
            ],
        },
        "reward_features": {
            "retrieval_hit": None,
            "context_precision": None,
            "context_recall": None,
            "citation_utilization": None,
            "latency_cost": (candidate.cost_json or {}).get("latency_cost"),
            "token_cost": candidate.token_count,
            "signal_coverage": features.get("signal_coverage"),
            "signal_boundary_cut_cost": features.get("signal_boundary_cut_cost"),
            "community_modularity_gain": features.get("community_modularity_gain"),
            "community_boundary_penalty": features.get("community_boundary_penalty"),
        },
        "feedback": {
            "next_action": "commit_active_chunk" if gate_passed else "repair_candidate",
            "suggested_generators": (
                ["dependency_closure"] if "broken_reference_closure" in risk_flags else []
            )
            + (["signal_bridge_repair"] if "signal_boundary_cut" in signal_feedback_flags else [])
            + (["community_region"] if "community_boundary_cut" in signal_feedback_flags else []),
            "signal_feedback_flags": signal_feedback_flags,
        },
    }


def create_quality_decision(
    db: Session,
    *,
    candidate: ChunkCandidate,
    atoms: list[EvidenceAtom],
    policy_state: PolicyState,
    previous_feedback: dict[str, Any] | None = None,
) -> QualityDecision:
    decision = decide_candidate_quality(candidate, atoms, policy_state, previous_feedback=previous_feedback)
    record = QualityDecision(
        candidate_id=candidate.id,
        policy_state_id=policy_state.id,
        decision_action=decision["decision_action"],
        gate_passed=decision["gate_passed"],
        confidence=0.95 if decision["gate_passed"] else 0.75,
        risk_flags_json=decision["risk_flags"],
        diagnostics_json=decision["diagnostics"],
        reward_features_json=decision["reward_features"],
        feedback_json=decision["feedback"],
    )
    db.add(record)
    db.flush()
    return record


def create_active_chunk_for_chunk(
    db: Session,
    *,
    knowledge_base_id: str,
    graph_state: EvidenceGraphState,
    policy_state: PolicyState,
    candidate: ChunkCandidate,
    quality_decision: QualityDecision,
    chunk: ChunkDraft,
    atoms: list[EvidenceAtom],
    signal_state: SignalState | None = None,
) -> ActiveChunk | None:
    if not quality_decision.gate_passed:
        return None
    chunk_decision = ChunkDecision(
        knowledge_base_id=knowledge_base_id,
        graph_state_id=graph_state.id,
        candidate_id=candidate.id,
        quality_decision_id=quality_decision.id,
        policy_state_id=policy_state.id,
        action="activate",
        decision_protocol_version=CHUNK_DECISION_VERSION,
        diagnostics_json={"chunk_draft_id": chunk.id, "selected_generator": candidate.generator_name},
    )
    db.add(chunk_decision)
    db.flush()
    atom_ids = list(candidate.atom_ids_json or [])
    document_version_scope_hash = stable_hash(candidate.source_span_union_json.get("document_version_ids", []))
    active_chunk = ActiveChunk(
        knowledge_base_id=knowledge_base_id,
        chunk_decision_id=chunk_decision.id,
        document_version_scope_hash=document_version_scope_hash,
        graph_state_hash=graph_state.state_hash,
        atom_ids_json=atom_ids,
        text=chunk.content,
        source_span_union_json=candidate.source_span_union_json,
        boundary_policy_version=CHUNK_DECISION_VERSION,
        quality_decision_id=quality_decision.id,
        policy_state_id=policy_state.id,
        community_ids_json=(candidate.graph_features_json or {}).get("dominant_community_ids") or [],
        metadata_json={
            "chunk_draft_id": chunk.id,
            "document_id": chunk.document_id,
            "document_version_id": chunk.document_version_id,
            "chunk_version": chunk.chunk_version,
            "partition": chunk.partition,
            "section": chunk.section,
            "page_number": chunk.page_number,
            "source_type": chunk.source_type,
            "snippet": chunk.snippet,
            "content_kind": (chunk.metadata_json or {}).get("content_kind"),
            "is_parent": bool((chunk.metadata_json or {}).get("is_parent")),
            "parent_chunk_id": chunk.parent_chunk_id,
            "policy_state_hash": policy_state.state_hash,
            "bandit_policy_version": policy_state.policy_version,
            "bandit_selection": (candidate.diagnostics_json or {}).get("bandit_selection"),
            "graph_features": candidate.graph_features_json,
            "quality_action": quality_decision.decision_action,
            "quality_gate_passed": quality_decision.gate_passed,
            "quality_policy_version": QUALITY_POLICY_VERSION,
            "signal_state_id": signal_state.id if signal_state else (candidate.graph_features_json or {}).get("signal_state_id"),
            "signal_state_hash": signal_state.signal_state_hash if signal_state else (candidate.graph_features_json or {}).get("signal_state_hash"),
            "signal_node_ids": (candidate.graph_features_json or {}).get("dominant_signal_ids") or [],
            "community_state_id": (candidate.graph_features_json or {}).get("community_state_id"),
            "community_ids": (candidate.graph_features_json or {}).get("dominant_community_ids") or [],
            "modularity_q": (candidate.graph_features_json or {}).get("community_modularity_q"),
            "signal_boundary_features": {
                "signal_coverage": (candidate.graph_features_json or {}).get("signal_coverage"),
                "signal_fragmentation": (candidate.graph_features_json or {}).get("signal_fragmentation"),
                "signal_boundary_cut_cost": (candidate.graph_features_json or {}).get("signal_boundary_cut_cost"),
                "signal_support_closure": (candidate.graph_features_json or {}).get("signal_support_closure"),
                "community_modularity_gain": (candidate.graph_features_json or {}).get("community_modularity_gain"),
                "community_boundary_penalty": (candidate.graph_features_json or {}).get("community_boundary_penalty"),
            },
        },
        state="staged",
    )
    db.add(active_chunk)
    db.flush()
    chunk.metadata_json = {
        **(chunk.metadata_json or {}),
        "evidence_atom_ids": atom_ids,
        "active_chunk_id": active_chunk.id,
        "chunk_decision_id": chunk_decision.id,
        "quality_decision_id": quality_decision.id,
        "graph_state_hash": graph_state.state_hash,
        "source_span_union": candidate.source_span_union_json,
        "quality_action": quality_decision.decision_action,
        "quality_gate_passed": quality_decision.gate_passed,
        "boundary_policy_version": CHUNK_DECISION_VERSION,
        "policy_state_hash": policy_state.state_hash,
        "bandit_policy_version": policy_state.policy_version,
        "bandit_selection": (candidate.diagnostics_json or {}).get("bandit_selection"),
        "selected_candidate_generator": candidate.generator_name,
        "signal_state_id": signal_state.id if signal_state else (candidate.graph_features_json or {}).get("signal_state_id"),
        "signal_state_hash": signal_state.signal_state_hash if signal_state else (candidate.graph_features_json or {}).get("signal_state_hash"),
        "signal_node_ids": (candidate.graph_features_json or {}).get("dominant_signal_ids") or [],
        "community_state_id": (candidate.graph_features_json or {}).get("community_state_id"),
        "community_ids": (candidate.graph_features_json or {}).get("dominant_community_ids") or [],
        "modularity_q": (candidate.graph_features_json or {}).get("community_modularity_q"),
    }
    from app.services.cache_manager import get_cache_manager

    get_cache_manager().invalidate_knowledge_base(knowledge_base_id)
    return active_chunk


def create_community_state(
    db: Session,
    *,
    knowledge_base_id: str,
    graph_state: EvidenceGraphState,
    atoms: list[EvidenceAtom],
    edges: list[EvidenceEdge] | None = None,
) -> CommunityState | None:
    if not atoms:
        return None
    settings = get_settings()
    atom_by_id = {atom.id: atom for atom in atoms}
    graph = nx.Graph()
    graph.add_nodes_from(atom_by_id)
    edge_weight_by_type = {
        "ADJACENT": 0.48,
        "CONTAINS": 0.84,
        "LAYOUT_CONTINUES": 0.72,
        "SEMANTIC_SIMILAR": 0.62,
        "REFERENCE_DEPENDS_ON": 0.68,
        "MODALITY_LINK": 0.55,
        "DISCOURSE_SHIFT": 0.22,
        "LEXICAL_OVERLAP": 0.46,
        "TOPIC_OVERLAP": 0.58,
        "DEFINITION_SUPPORT": 0.72,
        "SYMBOL_REFERENCE": 0.64,
    }
    edge_type_counts: Counter[str] = Counter()
    for edge in edges or []:
        if edge.source_atom_id not in atom_by_id or edge.target_atom_id not in atom_by_id:
            continue
        protocol_weight = edge_weight_by_type.get(edge.edge_type, 0.4)
        weight = max(0.001, float(edge.weight or 0.0) * float(edge.confidence or 0.0) * protocol_weight)
        if graph.has_edge(edge.source_atom_id, edge.target_atom_id):
            graph[edge.source_atom_id][edge.target_atom_id]["weight"] += weight
        else:
            graph.add_edge(edge.source_atom_id, edge.target_atom_id, weight=weight)
        edge_type_counts[edge.edge_type] += 1

    reason = None
    modularity_q = 0.0
    if graph.number_of_edges() <= 0 or graph.number_of_nodes() < 3:
        reason = "insufficient_edges"
        communities = [set(graph.nodes)]
    else:
        communities = list(
            nx.algorithms.community.louvain_communities(
                graph,
                weight="weight",
                resolution=float(settings.community_louvain_resolution),
                seed=17,
            )
        )
        if not communities:
            reason = "louvain_empty"
            communities = [set(graph.nodes)]
        else:
            modularity_q = float(nx.algorithms.community.modularity(graph, communities, weight="weight", resolution=float(settings.community_louvain_resolution)))
            if modularity_q < float(settings.community_min_modularity_warn):
                reason = "low_modularity"
    groups: dict[str, list[EvidenceAtom]] = {}
    for index, community in enumerate(sorted(communities, key=lambda item: (-len(item), sorted(item)[0] if item else ""))):
        groups[f"community:{index:04d}"] = [atom_by_id[atom_id] for atom_id in sorted(community) if atom_id in atom_by_id]
    isolated_atom_count = sum(1 for atom_id in atom_by_id if graph.degree(atom_id) == 0)
    state_hash = stable_hash(
        {
            "graph_state_hash": graph_state.state_hash,
            "protocol": COMMUNITY_PROTOCOL_VERSION,
            "groups": {key: [atom.text_hash for atom in value] for key, value in sorted(groups.items())},
            "modularity_q": round(modularity_q, 8),
            "resolution": float(settings.community_louvain_resolution),
        }
    )
    community_state = CommunityState(
        knowledge_base_id=knowledge_base_id,
        graph_state_id=graph_state.id,
        community_protocol_version=COMMUNITY_PROTOCOL_VERSION,
        state_hash=state_hash,
        diagnostics_json={
            "community_count": len(groups),
            "algorithm": "louvain_modularity",
            "modularity_q": round(modularity_q, 6),
            "resolution": float(settings.community_louvain_resolution),
            "edge_weight_protocol": edge_weight_by_type,
            "edge_type_counts": dict(edge_type_counts),
            "community_size_distribution": sorted([len(value) for value in groups.values()], reverse=True),
            "isolated_atom_count": isolated_atom_count,
            "reason": reason,
        },
        state="active",
    )
    db.add(community_state)
    db.flush()
    for community_id, members in groups.items():
        for atom in members:
            db.add(
                CommunityMembership(
                    community_state_id=community_state.id,
                    community_id=community_id,
                    atom_id=atom.id,
                    membership_score=1.0,
                    diagnostics_json={
                        "reason": "louvain_modularity" if reason is None else reason,
                        "modularity_q": round(modularity_q, 6),
                    },
                )
            )
        citation_atoms = [atom for atom in members if atom.atom_type != "heading"] or members
        summary_text = " ".join(atom.text.strip() for atom in citation_atoms[:3])[:500]
        db.add(
            CommunitySummary(
                community_state_id=community_state.id,
                community_id=community_id,
                summary=summary_text,
                citations_json=[
                    {
                        "evidence_atom_id": atom.id,
                        "source_span": atom.source_span_json,
                    }
                    for atom in citation_atoms[:5]
                ],
                evidence_atom_ids_json=[atom.id for atom in citation_atoms[:5]],
                quality_json={"citation_backed": bool(citation_atoms), "derived_view": True},
            )
        )
    graph_state.community_state_id = community_state.id
    graph_state.stats_json = {
        **(graph_state.stats_json or {}),
        "community_count": len(groups),
        "modularity_q": round(modularity_q, 6),
        "community_protocol_version": COMMUNITY_PROTOCOL_VERSION,
        "community_state_hash": community_state.state_hash,
    }
    _refresh_graph_state_hash(graph_state)
    db.flush()
    return community_state


def _clamp01(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = default
    return max(0.0, min(1.0, number))


def bandit_arm_for_candidate(candidate: ChunkCandidate) -> str:
    features = candidate.graph_features_json or {}
    if features.get("has_table") or features.get("has_code") or features.get("has_formula"):
        return "table_code_preserving"
    if candidate.generator_name == "signal_bridge_repair":
        return "signal_bridge_repair"
    if candidate.generator_name == "signal_region":
        return "signal_region"
    if candidate.generator_name == "community_region":
        return "community_region"
    if candidate.generator_name in {"semantic_cut", "heading_preserving", "high_recall_overlap", "low_overlap_precise"}:
        return candidate.generator_name
    return "atomic_parent_context"


def bandit_context_for_candidate(candidate: ChunkCandidate, quality: QualityDecision, policy_state: PolicyState | None = None) -> list[float]:
    features = candidate.graph_features_json or {}
    max_tokens = 2400
    if policy_state is not None:
        max_tokens = int((policy_state.constraints_json or {}).get("max_candidate_tokens") or max_tokens)
    token_efficiency = 1.0 - min(1.0, float(candidate.token_count or 0) / max(float(max_tokens), 1.0))
    signal_arm = 1.0 if bandit_arm_for_candidate(candidate) in {"signal_region", "signal_bridge_repair"} else 0.0
    return [
        1.0,
        _clamp01(features.get("internal_cohesion")),
        1.0 - _clamp01(features.get("boundary_cut_cost")),
        _clamp01(features.get("signal_coverage")),
        1.0 - _clamp01(features.get("signal_boundary_cut_cost")),
        _clamp01(features.get("community_modularity_gain")),
        1.0 - _clamp01(features.get("community_boundary_penalty")),
        _clamp01(features.get("reference_closure"), 1.0),
        _clamp01(features.get("layout_integrity"), 1.0),
        _clamp01(token_efficiency),
        1.0 if features.get("has_table") or features.get("has_code") or features.get("has_formula") else 0.0,
        signal_arm,
        _clamp01(quality.confidence, 1.0) if quality.gate_passed else 0.0,
    ]


def linucb_score(policy_state: PolicyState, arm: str, context: list[float]) -> tuple[float, dict[str, Any]]:
    posterior = _normalize_policy_posterior(policy_state)
    arms = posterior.get("arms") or {}
    constraints = policy_state.constraints_json or {}
    allowed_arms = set(constraints.get("allowed_arms") or BANDIT_ARMS)
    if arm not in allowed_arms:
        return float("-inf"), {
            "algorithm": "diagonal_linucb",
            "arm": arm,
            "score": float("-inf"),
            "filtered": True,
            "filter_reason": "arm_not_allowed",
            "allowed_arms": sorted(allowed_arms),
            "context_features": list(BANDIT_CONTEXT_FEATURES),
            "context": [round(value, 6) for value in context],
        }
    arm_state = arms.get(arm) or arms.get("atomic_parent_context") or {}
    max_candidate_tokens = constraints.get("max_candidate_tokens")
    count = int(arm_state.get("count") or 0)
    if max_candidate_tokens is not None and count > 0:
        token_cost_sum = float(arm_state.get("token_cost_sum") or 0.0)
        average_token_cost = token_cost_sum / max(count, 1)
        if average_token_cost > float(max_candidate_tokens):
            return float("-inf"), {
                "algorithm": "diagonal_linucb",
                "arm": arm,
                "score": float("-inf"),
                "filtered": True,
                "filter_reason": "arm_token_cost_exceeded",
                "average_token_cost": round(average_token_cost, 6),
                "max_candidate_tokens": float(max_candidate_tokens),
                "context_features": list(BANDIT_CONTEXT_FEATURES),
                "context": [round(value, 6) for value in context],
            }
    a_diag = [max(float(value), 1e-6) for value in arm_state.get("A_diag", [])]
    b_vec = [float(value) for value in arm_state.get("b", [])]
    alpha = float((policy_state.exploration_json or {}).get("alpha") or 0.25)
    theta = [b / a for a, b in zip(a_diag, b_vec)]
    exploitation = sum(weight * value for weight, value in zip(theta, context))
    exploration = math.sqrt(sum((value * value) / a for a, value in zip(a_diag, context)))
    score = exploitation + alpha * exploration
    return score, {
        "algorithm": "diagonal_linucb",
        "arm": arm,
        "exploitation": round(exploitation, 6),
        "exploration": round(exploration, 6),
        "alpha": alpha,
        "score": round(score, 6),
        "arm_count": int(arm_state.get("count") or 0),
        "context_features": list(BANDIT_CONTEXT_FEATURES),
        "context": [round(value, 6) for value in context],
    }


def candidate_selection_score(candidate: ChunkCandidate, quality: QualityDecision, policy_state: PolicyState) -> float:
    if not quality.gate_passed:
        return -1_000_000.0
    arm = bandit_arm_for_candidate(candidate)
    context = bandit_context_for_candidate(candidate, quality, policy_state)
    score, diagnostics = linucb_score(policy_state, arm, context)
    candidate.diagnostics_json = {
        **(candidate.diagnostics_json or {}),
        "bandit_selection": diagnostics,
    }
    quality.diagnostics_json = {
        **(quality.diagnostics_json or {}),
        "bandit_selection": diagnostics,
    }
    return score


def record_policy_selection_observation(
    db: Session,
    *,
    knowledge_base_id: str,
    policy_state: PolicyState,
    candidate: ChunkCandidate,
    quality: QualityDecision,
    selected_score: float,
) -> PolicyObservation:
    arm = bandit_arm_for_candidate(candidate)
    context = bandit_context_for_candidate(candidate, quality, policy_state)
    observation = PolicyObservation(
        knowledge_base_id=knowledge_base_id,
        policy_state_id=policy_state.id,
        context_json={
            "context_features": list(BANDIT_CONTEXT_FEATURES),
            "context": context,
            "candidate_id": candidate.id,
            "quality_decision_id": quality.id,
            "generator_name": candidate.generator_name,
        },
        action_json={
            "arm": arm,
            "selected_score": selected_score,
            "policy_version": policy_state.policy_version,
            "selection_algorithm": "constrained_diagonal_linucb",
        },
        reward_json={"status": "pending_reward"},
        propensity=1.0,
        diagnostics_json={"source": "chunk_candidate_selection_v1"},
    )
    db.add(observation)
    db.flush()
    return observation


def _reward_value(reward_json: dict[str, Any]) -> float:
    weighted_terms: list[tuple[float, float]] = []
    for key, weight in (
        ("retrieval_hit", 0.25),
        ("context_precision", 0.15),
        ("context_recall", 0.15),
        ("citation_utilization", 0.2),
        ("answer_groundedness", 0.2),
        ("answer_completeness", 0.15),
        ("rerank_gain", 0.1),
        ("user_acceptance", 0.2),
    ):
        value = reward_json.get(key)
        if value is not None:
            weighted_terms.append((_clamp01(value), weight))
    if not weighted_terms:
        reward = 0.0
    else:
        reward = sum(value * weight for value, weight in weighted_terms) / sum(weight for _value, weight in weighted_terms)
    latency_value = reward_json.get("latency_cost")
    try:
        normalized_latency = min(1.0, max(0.0, float(latency_value) / 30.0)) if latency_value is not None else 0.0
    except (TypeError, ValueError):
        normalized_latency = 0.0
    latency_penalty = 0.05 * normalized_latency
    token_cost = reward_json.get("token_cost")
    token_penalty = 0.0
    if token_cost is not None:
        try:
            token_penalty = 0.05 * min(1.0, max(0.0, float(token_cost)))
        except (TypeError, ValueError):
            token_penalty = 0.0
    rechunk_penalty = 0.1 * _clamp01(reward_json.get("rechunk_rate"))
    return max(0.0, min(1.0, reward - latency_penalty - token_penalty - rechunk_penalty))


def _mean(values: list[float]) -> float:
    return sum(values) / max(len(values), 1)


def _variance(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = _mean(values)
    return sum((value - mean) ** 2 for value in values) / len(values)


def _jaccard_distance(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 0.0
    return 1.0 - (len(left & right) / max(len(left | right), 1))


def _policy_drift_status(db: Session, policy_state: PolicyState, current_reward: float) -> tuple[str, dict[str, Any]]:
    recent_events = db.scalars(
        select(RewardEvent)
        .where(
            RewardEvent.knowledge_base_id == policy_state.knowledge_base_id,
            RewardEvent.policy_state_id == policy_state.id,
        )
        .order_by(RewardEvent.created_at.desc())
        .limit(50)
    ).all()
    reward_values = [_reward_value(event.reward_json or {}) for event in recent_events]
    if current_reward is not None and (not reward_values or reward_values[0] != current_reward):
        reward_values.insert(0, current_reward)
    recent_window = reward_values[:10]
    history_window = reward_values[10:]
    recent_mean = _mean(recent_window)
    history_mean = _mean(history_window) if history_window else recent_mean
    recent_variance = _variance(recent_window)
    history_variance = _variance(history_window) if history_window else max(recent_variance, 1e-9)
    reward_drift = (
        len(history_window) >= 5
        and (
            recent_variance > max(history_variance * 2.0, 1e-9)
            or recent_mean < max(history_mean * 0.5, 1e-9)
        )
    )

    graph_states = db.scalars(
        select(EvidenceGraphState)
        .where(EvidenceGraphState.knowledge_base_id == policy_state.knowledge_base_id)
        .order_by(EvidenceGraphState.created_at.desc())
        .limit(2)
    ).all()
    graph_jaccard_distance = 0.0
    if len(graph_states) >= 2:
        current_atoms = {str(item) for item in (graph_states[0].active_atom_ids or []) if item}
        previous_atoms = {str(item) for item in (graph_states[1].active_atom_ids or []) if item}
        graph_jaccard_distance = _jaccard_distance(current_atoms, previous_atoms)

    if graph_jaccard_distance > 0.3:
        status = "graph_drift"
    elif reward_drift:
        status = "drift_detected"
    else:
        status = "fresh"
    return status, {
        "recent_reward_mean": round(recent_mean, 6),
        "history_reward_mean": round(history_mean, 6),
        "recent_reward_variance": round(recent_variance, 6),
        "history_reward_variance": round(history_variance, 6),
        "reward_event_window": len(reward_values),
        "graph_jaccard_distance": round(graph_jaccard_distance, 6),
    }


def update_policy_from_reward_event(db: Session, reward_event: RewardEvent) -> int:
    diagnostics = dict(reward_event.diagnostics_json or {})
    if diagnostics.get("policy_update_applied"):
        return 0
    if not reward_event.policy_state_id:
        diagnostics["policy_update_skipped"] = "missing_policy_state_id"
        reward_event.diagnostics_json = diagnostics
        return 0
    policy_state = db.get(PolicyState, reward_event.policy_state_id)
    if policy_state is None:
        diagnostics["policy_update_skipped"] = "policy_state_not_found"
        reward_event.diagnostics_json = diagnostics
        return 0
    posterior = _normalize_policy_posterior(policy_state)
    reward = _reward_value(reward_event.reward_json or {})
    active_chunk_ids = [str(item) for item in reward_event.active_chunk_ids_json or [] if item]
    active_chunks = db.scalars(select(ActiveChunk).where(ActiveChunk.id.in_(active_chunk_ids))).all() if active_chunk_ids else []
    observations = 0
    updated_arms: list[str] = []
    for active_chunk in active_chunks:
        chunk_decision = db.get(ChunkDecision, active_chunk.chunk_decision_id)
        if chunk_decision is None:
            continue
        candidate = db.get(ChunkCandidate, chunk_decision.candidate_id)
        quality = db.get(QualityDecision, chunk_decision.quality_decision_id)
        if candidate is None or quality is None:
            continue
        arm = bandit_arm_for_candidate(candidate)
        context = bandit_context_for_candidate(candidate, quality, policy_state)
        arm_state = posterior["arms"][arm]
        propensity = max(float(reward_event.propensity or 1.0), 0.05)
        weighted_reward = reward / propensity
        arm_state["A_diag"] = [
            float(a_value) + (context_value * context_value)
            for a_value, context_value in zip(arm_state["A_diag"], context)
        ]
        arm_state["b"] = [
            float(b_value) + (weighted_reward * context_value)
            for b_value, context_value in zip(arm_state["b"], context)
        ]
        arm_state["count"] = int(arm_state.get("count") or 0) + 1
        arm_state["reward_sum"] = float(arm_state.get("reward_sum") or 0.0) + reward
        arm_state["last_reward"] = reward
        arm_state["token_cost_sum"] = float(arm_state.get("token_cost_sum") or 0.0) + float(candidate.token_count or 0)
        updated_arms.append(arm)
        db.add(
            PolicyObservation(
                knowledge_base_id=reward_event.knowledge_base_id,
                policy_state_id=policy_state.id,
                context_json={
                    "context_features": list(BANDIT_CONTEXT_FEATURES),
                    "context": context,
                    "candidate_id": candidate.id,
                    "active_chunk_id": active_chunk.id,
                    "retrieval_trace_id": reward_event.retrieval_trace_id,
                    "answer_session_id": reward_event.answer_session_id,
                },
                action_json={
                    "arm": arm,
                    "policy_version": policy_state.policy_version,
                    "update_algorithm": "diagonal_linucb_reward_update",
                },
                reward_json={
                    "reward": reward,
                    **(reward_event.reward_json or {}),
                },
                propensity=propensity,
                diagnostics_json={
                    "source": "reward_event_policy_update_v1",
                    "reward_event_id": reward_event.id,
                },
            )
        )
        observations += 1
    if observations:
        policy_state.posterior_json = posterior
        drift_status, drift_diagnostics = _policy_drift_status(db, policy_state, reward)
        exploration = dict(policy_state.exploration_json or {})
        if drift_status != "fresh":
            current_alpha = float(exploration.get("alpha") or 0.25)
            exploration["alpha"] = round(min(0.5, max(current_alpha * 1.5, current_alpha + 0.05)), 6)
            policy_state.drift_detected_at = datetime.utcnow()
        policy_state.exploration_json = exploration
        summary = dict(policy_state.reward_summary_json or {})
        previous_events = int(summary.get("events") or 0)
        previous_observations = int(summary.get("observations") or 0)
        previous_reward_sum = float(summary.get("reward_sum") or 0.0)
        summary.update(
            {
                "protocol_version": REWARD_PROTOCOL_VERSION,
                "events": previous_events + 1,
                "observations": previous_observations + observations,
                "reward_sum": previous_reward_sum + reward,
                "reward_mean": round((previous_reward_sum + reward) / max(previous_events + 1, 1), 6),
                "last_reward": reward,
                "last_reward_event_id": reward_event.id,
                "updated_arms": sorted(set(updated_arms)),
                "posterior_hash": _policy_posterior_hash(policy_state),
                "context_features": list(BANDIT_CONTEXT_FEATURES),
                "drift": drift_diagnostics,
            }
        )
        policy_state.reward_summary_json = summary
        policy_state.drift_status = drift_status
        diagnostics["policy_update_applied"] = True
        diagnostics["policy_update_observations"] = observations
        diagnostics["policy_updated_arms"] = sorted(set(updated_arms))
        diagnostics["policy_reward"] = reward
        diagnostics["policy_drift_status"] = drift_status
        diagnostics["policy_drift"] = drift_diagnostics
    else:
        diagnostics["policy_update_skipped"] = "no_resolvable_active_chunks"
    reward_event.diagnostics_json = diagnostics
    db.flush()
    return observations


def update_policy_from_rewards(db: Session, *, knowledge_base_id: str, limit: int = 200) -> int:
    events = db.scalars(
        select(RewardEvent)
        .where(RewardEvent.knowledge_base_id == knowledge_base_id)
        .order_by(RewardEvent.created_at.asc())
        .limit(limit)
    ).all()
    updates = 0
    for event in events:
        if (event.diagnostics_json or {}).get("policy_update_applied"):
            continue
        updates += update_policy_from_reward_event(db, event)
    return updates


def _apply_evidence_pipeline_for_chunks_impl(
    db: Session,
    *,
    knowledge_base_id: str,
    document: Document,
    version: DocumentVersion,
    sections: list[ParsedSection],
    created_chunks: list[ChunkDraft],
    source_path: Path,
    extracted_path: Path | None,
    checksum: str,
    source_type: str,
    ingestion_job_id: str | None = None,
    batch_id: str | None = None,
    profile_objective_hash: str | None = None,
) -> EvidencePipelineResult:
    clear_document_version_evidence(db, knowledge_base_id=knowledge_base_id, document_version_id=version.id)
    source_file = upsert_source_file(
        db,
        knowledge_base_id=knowledge_base_id,
        document_id=document.id,
        source_path=source_path,
        checksum=checksum,
        source_type=source_type,
    )
    record_parse_job(
        db,
        knowledge_base_id=knowledge_base_id,
        document_id=document.id,
        document_version_id=version.id,
        ingestion_job_id=ingestion_job_id,
        source_file_id=source_file.id,
        sections=sections,
    )
    atoms = create_atoms(
        db,
        knowledge_base_id=knowledge_base_id,
        document_id=document.id,
        document_version_id=version.id,
        sections=sections,
        source_path=str(source_path),
        extracted_path=str(extracted_path) if extracted_path else None,
    )
    policy_state = ensure_policy_state(db, knowledge_base_id=knowledge_base_id, profile_objective_hash=profile_objective_hash)
    graph_state = create_graph_state(db, knowledge_base_id=knowledge_base_id, atoms=atoms, policy_state=policy_state)
    edges = generate_edges(db, graph_state=graph_state, atoms=atoms)
    community_state = create_community_state(db, knowledge_base_id=knowledge_base_id, graph_state=graph_state, atoms=atoms, edges=edges)
    signal_layer = build_evidence_signal_layer(
        db,
        knowledge_base_id=knowledge_base_id,
        graph_state_id=graph_state.id,
        graph_state_hash=graph_state.state_hash,
        atom_scope_hash=graph_state.atom_scope_hash,
        atoms=atoms,
        community_state=community_state,
        batch_id=batch_id,
    )
    signal_state = signal_layer.state
    projection_state = signal_layer.projection_state
    attach_signal_layer_to_graph_state(graph_state, signal_state, projection_state)
    previous_feedback = latest_quality_feedback(db, knowledge_base_id)
    chunk_to_active: dict[str, ActiveChunk] = {}
    rejected = 0
    created_candidate_count = 0
    created_quality_count = 0
    for chunk in created_chunks:
        candidates = [
            create_candidate_for_chunk(
                db,
                graph_state=graph_state,
                chunk=chunk,
                atoms=atoms,
                edges=edges,
                signal_state=signal_state,
            )
        ]
        candidates.extend(
            create_signal_candidates_for_chunk(
                db,
                graph_state=graph_state,
                chunk=chunk,
                atoms=atoms,
                edges=edges,
                signal_state=signal_state,
                previous_feedback=previous_feedback,
            )
        )
        candidates.extend(
            create_feedback_candidates_for_chunk(
                db,
                graph_state=graph_state,
                chunk=chunk,
                atoms=atoms,
                edges=edges,
                signal_state=signal_state,
                previous_feedback=previous_feedback,
            )
        )
        created_candidate_count += len(candidates)
        quality_pairs = [
            (
                candidate,
                create_quality_decision(
                    db,
                    candidate=candidate,
                    atoms=atoms,
                    policy_state=policy_state,
                    previous_feedback=previous_feedback,
                ),
            )
            for candidate in candidates
        ]
        created_quality_count += len(quality_pairs)
        candidate, quality = max(quality_pairs, key=lambda pair: candidate_selection_score(pair[0], pair[1], policy_state))
        selected_score = candidate_selection_score(candidate, quality, policy_state)
        record_policy_selection_observation(
            db,
            knowledge_base_id=knowledge_base_id,
            policy_state=policy_state,
            candidate=candidate,
            quality=quality,
            selected_score=selected_score,
        )
        active_chunk = create_active_chunk_for_chunk(
            db,
            knowledge_base_id=knowledge_base_id,
            graph_state=graph_state,
            policy_state=policy_state,
            candidate=candidate,
            quality_decision=quality,
            chunk=chunk,
            atoms=atoms,
            signal_state=signal_state,
        )
        if active_chunk is None:
            rejected += 1
        else:
            chunk_to_active[chunk.id] = active_chunk
    attach_active_chunks_to_signal_layer(db, signal_state=signal_state, active_chunks=chunk_to_active)
    graph_state.stats_json = {
        **(graph_state.stats_json or {}),
        "candidate_count": created_candidate_count,
        "active_chunk_count": len(chunk_to_active),
        "rejected_candidate_count": rejected,
    }
    db.flush()
    return EvidencePipelineResult(
        graph_state=graph_state,
        policy_state=policy_state,
        community_state=community_state,
        signal_state=signal_state,
        projection_state=projection_state,
        chunk_to_active=chunk_to_active,
        stats={
            "evidence_atoms": len(atoms),
            "evidence_edges": len(edges),
            "chunk_candidates": created_candidate_count,
            "quality_decisions": created_quality_count,
            "active_chunks": len(chunk_to_active),
            "community_regions": int((graph_state.stats_json or {}).get("community_count") or 0),
            "signal_layer_status": signal_state.status,
            "signal_layer_complete": signal_state.status == "active",
            "signal_state_id": signal_state.id,
            "signal_state_hash": signal_state.signal_state_hash,
            "signal_schema_state_id": signal_layer.schema_state.id,
            "signal_schema_hash": signal_layer.schema_state.schema_hash,
            "signal_candidates": signal_layer.stats.get("signal_candidate_count", 0),
            "signal_nodes": signal_layer.stats.get("signal_node_count", 0),
            "signal_edges": signal_layer.stats.get("signal_edge_count", 0),
            "signal_communities": signal_layer.stats.get("signal_community_count", 0),
            "signal_model_external_called": signal_layer.stats.get("llm_external_called", False),
            "signal_fallback_used": signal_layer.stats.get("fallback_used", False),
            "signal_estimated_tokens": signal_layer.stats.get("estimated_tokens", 0),
            "policy_state_id": policy_state.id,
            "policy_state_hash": policy_state.state_hash,
            "graph_state_id": graph_state.id,
            "graph_state_hash": graph_state.state_hash,
            "community_state_id": community_state.id if community_state else None,
        "community_state_hash": community_state.state_hash if community_state else None,
        },
    )


def apply_evidence_pipeline_for_chunks(
    db: Session,
    *,
    knowledge_base_id: str,
    document: Document,
    version: DocumentVersion,
    sections: list[ParsedSection],
    created_chunks: list[ChunkDraft],
    source_path: Path,
    extracted_path: Path | None,
    checksum: str,
    source_type: str,
    ingestion_job_id: str | None = None,
    batch_id: str | None = None,
    profile_objective_hash: str | None = None,
) -> EvidencePipelineResult:
    with db.begin_nested():
        return _apply_evidence_pipeline_for_chunks_impl(
            db,
            knowledge_base_id=knowledge_base_id,
            document=document,
            version=version,
            sections=sections,
            created_chunks=created_chunks,
            source_path=source_path,
            extracted_path=extracted_path,
            checksum=checksum,
            source_type=source_type,
            ingestion_job_id=ingestion_job_id,
            batch_id=batch_id,
            profile_objective_hash=profile_objective_hash,
        )


def load_active_atoms_for_knowledge_base(db: Session, knowledge_base_id: str) -> list[EvidenceAtom]:
    return db.scalars(
        select(EvidenceAtom)
        .join(DocumentVersion, DocumentVersion.id == EvidenceAtom.document_version_id)
        .where(
            EvidenceAtom.knowledge_base_id == knowledge_base_id,
            EvidenceAtom.state == "active",
            DocumentVersion.is_active.is_(True),
        )
        .order_by(EvidenceAtom.document_id.asc(), EvidenceAtom.atom_index.asc())
    ).all()


def _update_existing_active_chunk_for_candidate(
    db: Session,
    *,
    knowledge_base_id: str,
    graph_state: EvidenceGraphState,
    policy_state: PolicyState,
    candidate: ChunkCandidate,
    quality_decision: QualityDecision,
    chunk: ChunkDraft,
    active_chunk: ActiveChunk,
    signal_state: SignalState | None,
) -> ActiveChunk | None:
    if not quality_decision.gate_passed:
        return None
    chunk_decision = ChunkDecision(
        knowledge_base_id=knowledge_base_id,
        graph_state_id=graph_state.id,
        candidate_id=candidate.id,
        quality_decision_id=quality_decision.id,
        policy_state_id=policy_state.id,
        action="activate",
        decision_protocol_version=CHUNK_DECISION_VERSION,
        diagnostics_json={"chunk_draft_id": chunk.id, "selected_generator": candidate.generator_name, "global_publish": True},
    )
    db.add(chunk_decision)
    db.flush()
    atom_ids = list(candidate.atom_ids_json or [])
    active_chunk.chunk_decision_id = chunk_decision.id
    active_chunk.document_version_scope_hash = stable_hash(candidate.source_span_union_json.get("document_version_ids", []))
    active_chunk.graph_state_hash = graph_state.state_hash
    active_chunk.atom_ids_json = atom_ids
    active_chunk.text = chunk.content
    active_chunk.source_span_union_json = candidate.source_span_union_json
    active_chunk.boundary_policy_version = CHUNK_DECISION_VERSION
    active_chunk.quality_decision_id = quality_decision.id
    active_chunk.policy_state_id = policy_state.id
    active_chunk.community_ids_json = (candidate.graph_features_json or {}).get("dominant_community_ids") or []
    active_chunk.metadata_json = {
        **(active_chunk.metadata_json or {}),
        "chunk_draft_id": chunk.id,
        "document_id": chunk.document_id,
        "document_version_id": chunk.document_version_id,
        "chunk_version": chunk.chunk_version,
        "partition": chunk.partition,
        "section": chunk.section,
        "page_number": chunk.page_number,
        "source_type": chunk.source_type,
        "snippet": chunk.snippet,
        "content_kind": (chunk.metadata_json or {}).get("content_kind"),
        "is_parent": bool((chunk.metadata_json or {}).get("is_parent")),
        "parent_chunk_id": chunk.parent_chunk_id,
        "global_graph_state_id": graph_state.id,
        "policy_state_hash": policy_state.state_hash,
        "bandit_policy_version": policy_state.policy_version,
        "bandit_selection": (candidate.diagnostics_json or {}).get("bandit_selection"),
        "graph_state_hash": graph_state.state_hash,
        "graph_features": candidate.graph_features_json,
        "quality_action": quality_decision.decision_action,
        "quality_gate_passed": quality_decision.gate_passed,
        "quality_policy_version": QUALITY_POLICY_VERSION,
        "signal_state_id": signal_state.id if signal_state else (candidate.graph_features_json or {}).get("signal_state_id"),
        "signal_state_hash": signal_state.signal_state_hash if signal_state else (candidate.graph_features_json or {}).get("signal_state_hash"),
        "signal_node_ids": (candidate.graph_features_json or {}).get("dominant_signal_ids") or [],
        "community_state_id": (candidate.graph_features_json or {}).get("community_state_id"),
        "community_ids": (candidate.graph_features_json or {}).get("dominant_community_ids") or [],
        "modularity_q": (candidate.graph_features_json or {}).get("community_modularity_q"),
        "signal_boundary_features": {
            "signal_coverage": (candidate.graph_features_json or {}).get("signal_coverage"),
            "signal_fragmentation": (candidate.graph_features_json or {}).get("signal_fragmentation"),
            "signal_boundary_cut_cost": (candidate.graph_features_json or {}).get("signal_boundary_cut_cost"),
            "signal_support_closure": (candidate.graph_features_json or {}).get("signal_support_closure"),
            "community_modularity_gain": (candidate.graph_features_json or {}).get("community_modularity_gain"),
            "community_boundary_penalty": (candidate.graph_features_json or {}).get("community_boundary_penalty"),
        },
    }
    chunk.metadata_json = {
        **(chunk.metadata_json or {}),
        "evidence_atom_ids": atom_ids,
        "active_chunk_id": active_chunk.id,
        "chunk_decision_id": chunk_decision.id,
        "quality_decision_id": quality_decision.id,
        "global_graph_state_id": graph_state.id,
        "graph_state_hash": graph_state.state_hash,
        "source_span_union": candidate.source_span_union_json,
        "quality_action": quality_decision.decision_action,
        "quality_gate_passed": quality_decision.gate_passed,
        "boundary_policy_version": CHUNK_DECISION_VERSION,
        "policy_state_hash": policy_state.state_hash,
        "bandit_policy_version": policy_state.policy_version,
        "bandit_selection": (candidate.diagnostics_json or {}).get("bandit_selection"),
        "selected_candidate_generator": candidate.generator_name,
        "signal_state_id": signal_state.id if signal_state else (candidate.graph_features_json or {}).get("signal_state_id"),
        "signal_state_hash": signal_state.signal_state_hash if signal_state else (candidate.graph_features_json or {}).get("signal_state_hash"),
        "signal_node_ids": (candidate.graph_features_json or {}).get("dominant_signal_ids") or [],
        "community_state_id": (candidate.graph_features_json or {}).get("community_state_id"),
        "community_ids": (candidate.graph_features_json or {}).get("dominant_community_ids") or [],
        "modularity_q": (candidate.graph_features_json or {}).get("community_modularity_q"),
    }
    return active_chunk


def publish_global_evidence_graph_state(
    db: Session,
    *,
    knowledge_base_id: str,
    batch_id: str | None = None,
    profile_objective_hash: str | None = None,
) -> EvidencePipelineResult:
    from app.services.ingestion_logs import emit_ingestion_log

    atoms = load_active_atoms_for_knowledge_base(db, knowledge_base_id)
    policy_state = ensure_policy_state(db, knowledge_base_id=knowledge_base_id, profile_objective_hash=profile_objective_hash)
    graph_state = create_graph_state(
        db,
        knowledge_base_id=knowledge_base_id,
        atoms=atoms,
        policy_state=policy_state,
        scope_type="global",
        initial_state="staging",
    )
    emit_ingestion_log(batch_id, "global_graph_scanning", "Global evidence graph publish started", graph_state_id=graph_state.id, atom_count=len(atoms))
    edges = generate_edges(db, graph_state=graph_state, atoms=atoms)
    community_state = create_community_state(db, knowledge_base_id=knowledge_base_id, graph_state=graph_state, atoms=atoms, edges=edges)
    signal_layer = build_evidence_signal_layer(
        db,
        knowledge_base_id=knowledge_base_id,
        graph_state_id=graph_state.id,
        graph_state_hash=graph_state.state_hash,
        atom_scope_hash=graph_state.atom_scope_hash,
        atoms=atoms,
        community_state=community_state,
        batch_id=batch_id,
    )
    signal_state = signal_layer.state
    projection_state = signal_layer.projection_state
    attach_signal_layer_to_graph_state(graph_state, signal_state, projection_state)

    active_chunks = db.scalars(
        select(ActiveChunk).where(
            ActiveChunk.knowledge_base_id == knowledge_base_id,
            ActiveChunk.state == "active",
        )
    ).all()
    chunk_to_active: dict[str, ActiveChunk] = {}
    created_candidate_count = 0
    created_quality_count = 0
    rejected = 0
    previous_feedback = latest_quality_feedback(db, knowledge_base_id)
    for active_chunk in active_chunks:
        metadata = active_chunk.metadata_json or {}
        chunk = ChunkDraft(
            id=active_chunk.id,
            knowledge_base_id=active_chunk.knowledge_base_id,
            document_id=str(metadata.get("document_id") or ""),
            document_version_id=str(metadata.get("document_version_id") or ""),
            chunk_version=int(metadata.get("chunk_version") or 0),
            content=active_chunk.text,
            snippet=str(metadata.get("snippet") or active_chunk.text[:240]),
            partition=metadata.get("partition"),
            section=metadata.get("section"),
            page_number=metadata.get("page_number"),
            token_count=estimate_tokens(active_chunk.text),
            source_type=str(metadata.get("source_type") or "unknown"),
            metadata_json={**metadata, "active_chunk_id": active_chunk.id},
            parent_chunk_id=metadata.get("parent_chunk_id"),
            summary=metadata.get("summary"),
            keywords=list(metadata.get("keywords") or []),
        )
        candidates = [
            create_candidate_for_chunk(
                db,
                graph_state=graph_state,
                chunk=chunk,
                atoms=atoms,
                edges=edges,
                signal_state=signal_state,
                generator_name="global_evidence_chunk_region",
                generator_version=f"{CHUNK_GENERATOR_VERSION}:global",
            )
        ]
        candidates.extend(
            create_signal_candidates_for_chunk(
                db,
                graph_state=graph_state,
                chunk=chunk,
                atoms=atoms,
                edges=edges,
                signal_state=signal_state,
                previous_feedback=previous_feedback,
            )
        )
        candidates.extend(
            create_feedback_candidates_for_chunk(
                db,
                graph_state=graph_state,
                chunk=chunk,
                atoms=atoms,
                edges=edges,
                signal_state=signal_state,
                previous_feedback=previous_feedback,
            )
        )
        created_candidate_count += len(candidates)
        chunk_atoms = [atom for atom in atoms if atom.document_version_id == chunk.document_version_id]
        quality_pairs = [
            (
                candidate,
                create_quality_decision(
                    db,
                    candidate=candidate,
                    atoms=chunk_atoms or atoms,
                    policy_state=policy_state,
                    previous_feedback=previous_feedback,
                ),
            )
            for candidate in candidates
        ]
        created_quality_count += len(quality_pairs)
        candidate, quality = max(quality_pairs, key=lambda pair: candidate_selection_score(pair[0], pair[1], policy_state))
        selected_score = candidate_selection_score(candidate, quality, policy_state)
        record_policy_selection_observation(
            db,
            knowledge_base_id=knowledge_base_id,
            policy_state=policy_state,
            candidate=candidate,
            quality=quality,
            selected_score=selected_score,
        )
        updated = _update_existing_active_chunk_for_candidate(
            db,
            knowledge_base_id=knowledge_base_id,
            graph_state=graph_state,
            policy_state=policy_state,
            candidate=candidate,
            quality_decision=quality,
            chunk=chunk,
            active_chunk=active_chunk,
            signal_state=signal_state,
        )
        if updated is None:
            rejected += 1
        else:
            chunk_to_active[chunk.id] = updated
    attach_active_chunks_to_signal_layer(db, signal_state=signal_state, active_chunks=chunk_to_active)
    old_global_states = db.scalars(
        select(EvidenceGraphState).where(
            EvidenceGraphState.knowledge_base_id == knowledge_base_id,
            EvidenceGraphState.scope_type == "global",
            EvidenceGraphState.state == "active",
            EvidenceGraphState.id != graph_state.id,
        )
    ).all()
    for old_state in old_global_states:
        old_state.state = "inactive"
    graph_state.state = "active"
    graph_state.stats_json = {
        **(graph_state.stats_json or {}),
        "candidate_count": created_candidate_count,
        "quality_decision_count": created_quality_count,
        "active_chunk_count": len(chunk_to_active),
        "rejected_candidate_count": rejected,
        "global_publish": True,
    }
    db.flush()
    from app.services.cache_manager import get_cache_manager

    get_cache_manager().invalidate_knowledge_base(knowledge_base_id)
    stats = {
        "graph_rebuilt": True,
        "graph_runtime": "evidence_graph",
        "graph_scope_type": "global",
        "graph_nodes": len(atoms) + len(chunk_to_active),
        "graph_edges": len(edges),
        "evidence_atoms": len(atoms),
        "evidence_edges": len(edges),
        "chunk_candidates": created_candidate_count,
        "quality_decisions": created_quality_count,
        "active_chunks": len(chunk_to_active),
        "community_regions": int((graph_state.stats_json or {}).get("community_count") or 0),
        "modularity_q": (graph_state.stats_json or {}).get("modularity_q"),
        "signal_layer_status": signal_state.status,
        "signal_layer_complete": signal_state.status == "active",
        "signal_state_id": signal_state.id,
        "signal_state_hash": signal_state.signal_state_hash,
        "signal_schema_state_id": signal_layer.schema_state.id,
        "signal_schema_hash": signal_layer.schema_state.schema_hash,
        "signal_candidates": signal_layer.stats.get("signal_candidate_count", 0),
        "signal_nodes": signal_layer.stats.get("signal_node_count", 0),
        "signal_edges": signal_layer.stats.get("signal_edge_count", 0),
        "signal_communities": signal_layer.stats.get("signal_community_count", 0),
        "signal_model_external_called": signal_layer.stats.get("llm_external_called", False),
        "signal_fallback_used": signal_layer.stats.get("fallback_used", False),
        "signal_estimated_tokens": signal_layer.stats.get("estimated_tokens", 0),
        "policy_state_id": policy_state.id,
        "policy_state_hash": policy_state.state_hash,
        "graph_state_id": graph_state.id,
        "graph_state_hash": graph_state.state_hash,
        "community_state_id": community_state.id if community_state else None,
        "community_state_hash": community_state.state_hash if community_state else None,
    }
    emit_ingestion_log(batch_id, "global_graph_active", "Global evidence graph activated", **stats)
    return EvidencePipelineResult(
        graph_state=graph_state,
        policy_state=policy_state,
        community_state=community_state,
        signal_state=signal_state,
        projection_state=projection_state,
        chunk_to_active=chunk_to_active,
        stats=stats,
    )


def activate_evidence_chunks(db: Session, active_chunks: dict[str, ActiveChunk]) -> None:
    for active_chunk in active_chunks.values():
        active_chunk.state = "active"
    db.flush()


def active_chunk_scope_hash(
    active_chunk_ids: list[str],
    graph_state_hash: str | None,
    policy_state_hash: str | None,
    signal_state_hash: str | None = None,
) -> str:
    return stable_hash(
        {
            "active_chunk_ids": sorted(active_chunk_ids),
            "graph_state_hash": graph_state_hash,
            "quality_policy_version": QUALITY_POLICY_VERSION,
            "policy_state_hash": policy_state_hash,
            "signal_state_hash": signal_state_hash,
        }
    )


def enrich_vector_payload(
    *,
    payload: dict[str, Any],
    chunk: ChunkDraft,
    active_chunk: ActiveChunk | None,
    graph_state: EvidenceGraphState | None,
    policy_state: PolicyState | None,
    community_state: CommunityState | None,
) -> dict[str, Any]:
    metadata = chunk.metadata_json or {}
    enriched = {
        **payload,
        "active_chunk_id": active_chunk.id if active_chunk else metadata.get("active_chunk_id"),
        "evidence_atom_ids": metadata.get("evidence_atom_ids") or [],
        "source_span_union": metadata.get("source_span_union") or {},
        "graph_state_hash": graph_state.state_hash if graph_state else metadata.get("graph_state_hash"),
        "policy_state_id": policy_state.id if policy_state else metadata.get("policy_state_id"),
        "policy_state_hash": policy_state.state_hash if policy_state else None,
        "community_state_id": community_state.id if community_state else None,
        "community_state_hash": community_state.state_hash if community_state else None,
        "quality_decision_id": metadata.get("quality_decision_id"),
        "quality_action": metadata.get("quality_action"),
        "quality_gate_passed": metadata.get("quality_gate_passed"),
        "chunk_decision_id": metadata.get("chunk_decision_id"),
        "selected_candidate_generator": metadata.get("selected_candidate_generator"),
        "boundary_policy_version": metadata.get("boundary_policy_version"),
        "signal_state_hash": metadata.get("signal_state_hash"),
        "signal_node_ids": metadata.get("signal_node_ids") or [],
        "embedding_text_version": CURRENT_EMBEDDING_TEXT_VERSION,
    }
    return enriched


def record_vector_records(
    db: Session,
    *,
    knowledge_base_id: str,
    chunks: list[ChunkDraft],
    vector_points: list[dict[str, Any]],
    chunk_to_active: dict[str, ActiveChunk],
    embedding_model: str,
) -> None:
    for chunk, point in zip(chunks, vector_points):
        payload_hash = stable_hash(point.get("payload") or {})
        active_chunk = chunk_to_active.get(chunk.id)
        if active_chunk is None:
            continue
        db.query(VectorRecord).filter(VectorRecord.active_chunk_id == active_chunk.id).delete(synchronize_session=False)
        db.add(
            VectorRecord(
                knowledge_base_id=knowledge_base_id,
                active_chunk_id=active_chunk.id,
                qdrant_point_id=str(point["id"]),
                embedding_model=embedding_model,
                embedding_text_version=CURRENT_EMBEDDING_TEXT_VERSION,
                payload_hash=payload_hash,
                vector_status="ready",
                diagnostics_json={"payload_hash": payload_hash},
            )
        )
    db.flush()
