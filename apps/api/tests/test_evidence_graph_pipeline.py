from __future__ import annotations

import json
from pathlib import Path


def _seed_document(db_session, sample_knowledge_base, tmp_path: Path, text: str = "Alpha heading\n\nAlpha connects to beta.\n\nBeta closes the topic."):
    from app.models import Document, DocumentVersion, generate_uuid
    from app.services.evidence_graph import ChunkDraft

    source_path = tmp_path / "general-note.md"
    source_path.write_text(text, encoding="utf-8")
    document = Document(
        knowledge_base_id=sample_knowledge_base.id,
        title="General note",
        source_path=str(source_path),
        source_type="markdown",
        tags=["General"],
        checksum="checksum-evidence",
    )
    db_session.add(document)
    db_session.flush()
    version = DocumentVersion(
        document_id=document.id,
        version=1,
        checksum="checksum-evidence",
        storage_path=str(source_path),
        extracted_path=str(tmp_path / "general-note.extracted.json"),
        is_active=False,
    )
    db_session.add(version)
    db_session.flush()
    chunk = ChunkDraft(
        id=generate_uuid(),
        knowledge_base_id=sample_knowledge_base.id,
        document_id=document.id,
        document_version_id=version.id,
        chunk_version=1,
        content=text,
        snippet="Alpha connects to beta.",
        partition="General",
        section="Overview",
        page_number=None,
        token_count=24,
        source_type="markdown",
        metadata_json={"content_kind": "markdown", "section_index": 0},
        is_active=False,
    )
    return source_path, document, version, [chunk]


def test_evidence_pipeline_creates_traceable_atoms_edges_and_active_chunks(db_session, sample_knowledge_base, tmp_path):
    from app.models import ActiveChunk, ChunkCandidate, EvidenceAtom, EvidenceEdge, PolicyObservation, SignalNode, SignalState
    from app.services.evidence_graph import apply_evidence_pipeline_for_chunks
    from app.services.parsers import ParsedSection

    source_path, document, version, chunks = _seed_document(db_session, sample_knowledge_base, tmp_path)
    section_text = chunks[0].content
    result = apply_evidence_pipeline_for_chunks(
        db_session,
        knowledge_base_id=sample_knowledge_base.id,
        document=document,
        version=version,
        sections=[
            ParsedSection(
                title="Overview",
                text=section_text,
                section="Overview",
                metadata={"content_kind": "markdown"},
            )
        ],
        created_chunks=chunks,
        source_path=source_path,
        extracted_path=Path(version.extracted_path),
        checksum=version.checksum,
        source_type=document.source_type,
        ingestion_job_id=None,
        profile_objective_hash="profile-hash",
    )

    atoms = db_session.query(EvidenceAtom).filter(EvidenceAtom.document_version_id == version.id).all()
    edges = db_session.query(EvidenceEdge).filter(EvidenceEdge.graph_state_id == result.graph_state.id).all()
    active_chunks = db_session.query(ActiveChunk).filter(ActiveChunk.knowledge_base_id == sample_knowledge_base.id).all()

    assert result.stats["evidence_atoms"] == len(atoms)
    assert result.stats["evidence_edges"] == len(edges)
    assert result.stats["active_chunks"] == len(active_chunks) == 1
    assert result.stats["signal_layer_complete"] is True
    assert result.stats["signal_nodes"] >= 1
    assert result.stats["signal_model_external_called"] is False
    assert result.stats["signal_fallback_used"] is False
    assert result.stats["signal_estimated_tokens"] >= 1
    assert all((atom.source_span_json or {}).get("end", 0) <= len(section_text) for atom in atoms)
    assert {edge.source_atom_id for edge in edges}.issubset({atom.id for atom in atoms})
    assert {edge.target_atom_id for edge in edges}.issubset({atom.id for atom in atoms})
    assert active_chunks[0].atom_ids_json
    assert active_chunks[0].source_span_union_json["spans"]
    assert active_chunks[0].metadata_json["signal_state_hash"] == result.signal_state.signal_state_hash
    assert active_chunks[0].metadata_json["quality_gate_passed"] is True
    assert active_chunks[0].metadata_json["bandit_selection"]["algorithm"] == "diagonal_linucb"
    assert "signal_coverage" in active_chunks[0].metadata_json["signal_boundary_features"]
    assert chunks[0].metadata_json["active_chunk_id"] == active_chunks[0].id
    assert chunks[0].metadata_json["graph_state_hash"] == result.graph_state.state_hash
    posterior = result.policy_state.posterior_json
    assert posterior["policy_algorithm"] == "diagonal_linucb"
    assert len(posterior["context_features"]) == 13
    assert all(len(arm["A_diag"]) == 13 and len(arm["b"]) == 13 for arm in posterior["arms"].values())
    observations = db_session.query(PolicyObservation).filter_by(policy_state_id=result.policy_state.id).all()
    assert any((observation.diagnostics_json or {}).get("source") == "chunk_candidate_selection_v1" for observation in observations)
    signal_state = db_session.query(SignalState).filter_by(evidence_graph_state_id=result.graph_state.id).one()
    assert signal_state.status == "active"
    assert set(signal_state.eligible_atom_ids_json) == set(signal_state.processed_atom_ids_json)
    nodes = db_session.query(SignalNode).filter_by(signal_state_id=signal_state.id).all()
    assert nodes
    assert all(node.support_atom_ids_json and node.source_span_union_json["spans"] for node in nodes)
    candidates = db_session.query(ChunkCandidate).filter_by(graph_state_id=result.graph_state.id).all()
    assert any(candidate.generator_name == "signal_region" for candidate in candidates)
    assert all("signal_coverage" in (candidate.graph_features_json or {}) for candidate in candidates)


def test_generic_quality_decision_has_no_course_bias(db_session, sample_knowledge_base, tmp_path):
    from app.models import ChunkCandidate
    from app.services.evidence_graph import apply_evidence_pipeline_for_chunks
    from app.services.parsers import ParsedSection

    source_path, document, version, chunks = _seed_document(db_session, sample_knowledge_base, tmp_path, text="System design note\n\nEvidence atoms preserve source spans.")
    result = apply_evidence_pipeline_for_chunks(
        db_session,
        knowledge_base_id=sample_knowledge_base.id,
        document=document,
        version=version,
        sections=[ParsedSection(title="System design note", text=chunks[0].content, section="System design note", metadata={"content_kind": "markdown"})],
        created_chunks=chunks,
        source_path=source_path,
        extracted_path=Path(version.extracted_path),
        checksum=version.checksum,
        source_type=document.source_type,
        profile_objective_hash="profile-hash",
    )
    candidate = db_session.query(ChunkCandidate).filter(ChunkCandidate.graph_state_id == result.graph_state.id).first()
    assert candidate is not None
    diagnostics = json.dumps(candidate.graph_features_json, ensure_ascii=False).lower()
    quality_diagnostics = json.dumps(chunks[0].metadata_json, ensure_ascii=False).lower()
    forbidden = {"course", "chapter", "lecture", "homework", "assignment", "exam"}
    assert not any(term in diagnostics for term in forbidden)
    assert not any(term in quality_diagnostics for term in forbidden)


def test_signal_layer_publishes_complete_algorithmic_view(db_session, sample_knowledge_base, tmp_path):
    from app.models import SignalState
    from app.services.evidence_graph import apply_evidence_pipeline_for_chunks
    from app.services.parsers import ParsedSection

    text = "Graph Centrality\n\nGraph Centrality is a ranking signal. Centrality appears in retrieval diagnostics."
    source_path, document, version, chunks = _seed_document(db_session, sample_knowledge_base, tmp_path, text=text)
    result = apply_evidence_pipeline_for_chunks(
        db_session,
        knowledge_base_id=sample_knowledge_base.id,
        document=document,
        version=version,
        sections=[ParsedSection(title="Graph Centrality", text=text, section="Graph Centrality", metadata={"content_kind": "markdown"})],
        created_chunks=chunks,
        source_path=source_path,
        extracted_path=Path(version.extracted_path),
        checksum=version.checksum,
        source_type=document.source_type,
        profile_objective_hash="profile-hash",
    )

    signal_state = db_session.query(SignalState).filter_by(id=result.signal_state.id).one()
    assert signal_state.status == "active"
    assert signal_state.model_audit_json["llm_external_called"] is False
    assert signal_state.model_audit_json["fallback_used"] is False
    assert len(signal_state.processed_atom_ids_json) == len(signal_state.eligible_atom_ids_json)
    assert result.stats["signal_layer_complete"] is True


def test_graph_payload_hides_incomplete_signal_layer(db_session, sample_knowledge_base, tmp_path):
    from app.models import SignalState
    from app.models import EvidenceGraphState
    from app.services.evidence_graph import activate_evidence_chunks, apply_evidence_pipeline_for_chunks, publish_global_evidence_graph_state
    from app.services.evidence_graph_payload import get_graph_payload
    from app.services.parsers import ParsedSection

    source_path, document, version, chunks = _seed_document(db_session, sample_knowledge_base, tmp_path, text="Alpha Evidence\n\nAlpha Evidence is traceable.")
    result = apply_evidence_pipeline_for_chunks(
        db_session,
        knowledge_base_id=sample_knowledge_base.id,
        document=document,
        version=version,
        sections=[ParsedSection(title="Alpha Evidence", text=chunks[0].content, section="Alpha Evidence", metadata={"content_kind": "markdown"})],
        created_chunks=chunks,
        source_path=source_path,
        extracted_path=Path(version.extracted_path),
        checksum=version.checksum,
        source_type=document.source_type,
        profile_objective_hash="profile-hash",
    )
    graph_without_global = get_graph_payload(db_session, sample_knowledge_base.id, graph_type="evidence")
    assert graph_without_global["nodes"] == []
    version.is_active = True
    for chunk in chunks:
        chunk.is_active = True
    activate_evidence_chunks(db_session, result.chunk_to_active)
    global_result = publish_global_evidence_graph_state(db_session, knowledge_base_id=sample_knowledge_base.id)
    graph = get_graph_payload(db_session, sample_knowledge_base.id, graph_type="evidence")
    assert graph["signal_layer_complete"] is True
    assert graph["view"] == "overview"
    assert any(node["category"] == "signal_node" for node in graph["nodes"])
    assert all((node.get("support_atom_ids") or []) for node in graph["nodes"] if node["category"] == "signal_node")
    assert db_session.query(EvidenceGraphState).filter_by(id=global_result.graph_state.id).one().scope_type == "global"

    signal_state = db_session.query(SignalState).filter_by(id=global_result.signal_state.id).one()
    signal_state.status = "normalizing"
    db_session.flush()
    graph = get_graph_payload(db_session, sample_knowledge_base.id, graph_type="evidence")
    assert graph["signal_layer_complete"] is False
    assert graph["signal_layer_status"] == "normalizing"
    assert not any(node["category"] == "signal_node" for node in graph["nodes"])


def test_global_publish_replaces_document_scope_for_graph_api(db_session, sample_knowledge_base, tmp_path):
    from app.models import ActiveChunk, CommunityState, EvidenceGraphState
    from app.services.evidence_graph import activate_evidence_chunks, apply_evidence_pipeline_for_chunks, publish_global_evidence_graph_state
    from app.services.evidence_graph_payload import get_graph_payload
    from app.services.parsers import ParsedSection

    source_path, document, version, chunks = _seed_document(
        db_session,
        sample_knowledge_base,
        tmp_path,
        text="Bayesian Inference\n\nPosterior inference combines likelihood and prior evidence.\n\nGraph modularity separates posterior topics.",
    )
    result = apply_evidence_pipeline_for_chunks(
        db_session,
        knowledge_base_id=sample_knowledge_base.id,
        document=document,
        version=version,
        sections=[ParsedSection(title="Bayesian Inference", text=chunks[0].content, section="Bayesian Inference", metadata={"content_kind": "markdown"})],
        created_chunks=chunks,
        source_path=source_path,
        extracted_path=Path(version.extracted_path),
        checksum=version.checksum,
        source_type=document.source_type,
        profile_objective_hash="profile-hash",
    )
    version.is_active = True
    for chunk in chunks:
        chunk.is_active = True
    activate_evidence_chunks(db_session, result.chunk_to_active)
    global_result = publish_global_evidence_graph_state(db_session, knowledge_base_id=sample_knowledge_base.id)

    document_state = db_session.query(EvidenceGraphState).filter_by(id=result.graph_state.id).one()
    assert document_state.scope_type == "document"
    assert global_result.graph_state.scope_type == "global"
    graph = get_graph_payload(db_session, sample_knowledge_base.id, graph_type="evidence", view="detail")
    assert graph["freshness"]["graph_state_id"] == global_result.graph_state.id
    assert graph["signal_layer_complete"] is True
    assert graph["diagnostics"] == {}
    community_state = db_session.query(CommunityState).filter_by(id=global_result.community_state.id).one()
    assert community_state.community_protocol_version == "modularity_louvain_v1"
    assert "modularity_q" in community_state.diagnostics_json
    active_chunk = db_session.query(ActiveChunk).filter_by(knowledge_base_id=sample_knowledge_base.id, state="active").one()
    assert active_chunk.graph_state_hash == global_result.graph_state.state_hash
    assert active_chunk.metadata_json["global_graph_state_id"] == global_result.graph_state.id
    assert active_chunk.metadata_json["signal_state_hash"] == global_result.signal_state.signal_state_hash
    assert active_chunk.metadata_json["quality_gate_passed"] is True
    assert "community_boundary_penalty" in active_chunk.metadata_json["signal_boundary_features"]


def test_signal_gate_rejects_low_signal_fragments(db_session, sample_knowledge_base, tmp_path):
    from app.models import SignalCandidate, SignalNode
    from app.services.evidence_graph import apply_evidence_pipeline_for_chunks
    from app.services.parsers import ParsedSection

    text = "Posterior Distribution\n\nPosterior Distribution is a probability measure.\n\nreturn float values theta_cand.\n\nfloat values theta_cand return."
    source_path, document, version, chunks = _seed_document(db_session, sample_knowledge_base, tmp_path, text=text)
    result = apply_evidence_pipeline_for_chunks(
        db_session,
        knowledge_base_id=sample_knowledge_base.id,
        document=document,
        version=version,
        sections=[ParsedSection(title="Posterior Distribution", text=text, section="Posterior Distribution", metadata={"content_kind": "markdown"})],
        created_chunks=chunks,
        source_path=source_path,
        extracted_path=Path(version.extracted_path),
        checksum=version.checksum,
        source_type=document.source_type,
        profile_objective_hash="profile-hash",
    )
    nodes = db_session.query(SignalNode).filter_by(signal_state_id=result.signal_state.id).all()
    names = {node.canonical_label.lower() for node in nodes}
    assert "posterior distribution" in names
    assert "values" not in names
    rejected = db_session.query(SignalCandidate).filter_by(signal_state_id=result.signal_state.id, status="rejected").all()
    assert any((candidate.features_json or {}).get("candidate_features", {}).get("confidence") is not None for candidate in rejected)


def test_reward_events_update_bandit_policy_posterior(db_session, sample_knowledge_base, tmp_path):
    from app.models import ActiveChunk, ChunkCandidate, ChunkDecision, PolicyObservation, RewardEvent
    from app.services.evidence_graph import apply_evidence_pipeline_for_chunks, bandit_arm_for_candidate, update_policy_from_reward_event
    from app.services.parsers import ParsedSection

    text = "Policy Learning\n\nEvidence rewards should update the selected chunk policy."
    source_path, document, version, chunks = _seed_document(db_session, sample_knowledge_base, tmp_path, text=text)
    result = apply_evidence_pipeline_for_chunks(
        db_session,
        knowledge_base_id=sample_knowledge_base.id,
        document=document,
        version=version,
        sections=[ParsedSection(title="Policy Learning", text=text, section="Policy Learning", metadata={"content_kind": "markdown"})],
        created_chunks=chunks,
        source_path=source_path,
        extracted_path=Path(version.extracted_path),
        checksum=version.checksum,
        source_type=document.source_type,
        profile_objective_hash="profile-hash",
    )
    active_chunk = db_session.query(ActiveChunk).filter_by(knowledge_base_id=sample_knowledge_base.id).one()
    decision = db_session.get(ChunkDecision, active_chunk.chunk_decision_id)
    candidate = db_session.get(ChunkCandidate, decision.candidate_id)
    arm = bandit_arm_for_candidate(candidate)
    before = dict(result.policy_state.posterior_json["arms"][arm])
    reward = RewardEvent(
        knowledge_base_id=sample_knowledge_base.id,
        policy_state_id=result.policy_state.id,
        active_chunk_ids_json=[active_chunk.id],
        context_json={"question_length": 42, "used_chunks": 1},
        action_json={"route": "agent_answer"},
        reward_json={
            "retrieval_hit": 1.0,
            "citation_utilization": 1.0,
            "answer_groundedness": 1.0,
            "token_cost": active_chunk.metadata_json.get("token_count"),
        },
        propensity=1.0,
        diagnostics_json={"source": "unit_reward"},
    )
    db_session.add(reward)
    db_session.flush()

    updates = update_policy_from_reward_event(db_session, reward)

    assert updates == 1
    assert reward.diagnostics_json["policy_update_applied"] is True
    after = result.policy_state.posterior_json["arms"][arm]
    assert after["count"] == int(before.get("count") or 0) + 1
    assert after["reward_sum"] > float(before.get("reward_sum") or 0.0)
    assert any(value > 0 for value in after["b"])
    assert result.policy_state.reward_summary_json["events"] == 1
    assert result.policy_state.reward_summary_json["observations"] >= 1
    assert result.policy_state.reward_summary_json["posterior_hash"]
    reward_observations = db_session.query(PolicyObservation).filter_by(policy_state_id=result.policy_state.id).all()
    assert any((observation.diagnostics_json or {}).get("reward_event_id") == reward.id for observation in reward_observations)
