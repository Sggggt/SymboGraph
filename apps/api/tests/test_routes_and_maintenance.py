from __future__ import annotations

import json
from pathlib import Path
import sys
import types
from types import SimpleNamespace

import pytest


def _coarse_graph_node_role_payload() -> dict:
    return {
        "contract_kind": "coarse_concept_node",
        "id": "coarse-1",
        "label": "Coarse concept",
        "type": "coarse_concept",
        "raw_node_weight": 0.8,
        "node_weight": 1.0,
        "node_weight_normalization_scope": "coarse_concept_state",
        "node_weight_diagnostics": {},
        "included_mid_concept_ids": ["mid-core"],
        "boundary_mid_concept_ids": ["mid-boundary"],
        "bridge_mid_concept_ids": ["mid-bridge"],
        "outlier_mid_concept_ids": ["mid-outlier"],
        "low_confidence_mid_concept_ids": ["mid-low-confidence"],
        "all_mid_concept_ids": [
            "mid-core",
            "mid-boundary",
            "mid-bridge",
            "mid-outlier",
            "mid-low-confidence",
        ],
    }


def test_graph_node_coarse_role_union_is_closed_and_replayable():
    from pydantic import ValidationError

    from app.schemas import GraphNode

    node = GraphNode.model_validate(_coarse_graph_node_role_payload())
    assert node.all_mid_concept_ids == [
        "mid-core",
        "mid-boundary",
        "mid-bridge",
        "mid-outlier",
        "mid-low-confidence",
    ]

    wrong_union = _coarse_graph_node_role_payload()
    wrong_union["all_mid_concept_ids"] = ["mid-core"]
    with pytest.raises(ValidationError, match="membership-role union"):
        GraphNode.model_validate(wrong_union)

    non_coarse = {
        "contract_kind": "chunk_node",
        "id": "chunk-1",
        "label": "Chunk",
        "type": "chunk",
        "boundary_mid_concept_ids": ["mid-smuggled"],
    }
    with pytest.raises(ValidationError, match="only coarse"):
        GraphNode.model_validate(non_coarse)


@pytest.mark.asyncio
async def test_context_graph_stats_and_layer_payloads(db_session, populated_context_graph):
    from app.services.context_graph import context_graph_stats, graph_layer_payload
    from app.schemas import GraphResponse

    kb = populated_context_graph["knowledge_base"]
    stats = context_graph_stats(db_session, kb.id)
    assert stats["counts"]["active_chunks"] >= 3
    assert stats["counts"]["mid_concepts"] >= 1
    assert "codebooks" not in (stats["diagnostics"]["relation"].get("rq_kmeans") or {})
    assert "path_by_chunk" not in (stats["diagnostics"]["relation"].get("rq_kmeans") or {})
    hash_keys = {
        "chunk-structure": "structure_graph_hash",
        "chunk-relation": "chunk_relation_graph_hash",
        "mid-concepts": "mid_concept_hash",
        "coarse-concepts": "coarse_concept_hash",
    }
    for layer in ("chunk-structure", "chunk-relation", "mid-concepts", "coarse-concepts"):
        payload = graph_layer_payload(db_session, kb.id, layer, limit=20)
        GraphResponse.model_validate(payload)
        assert payload["knowledge_base_id"] == kb.id
        assert payload["graph_type"] == layer
        assert payload["counts"]
        assert "freshness" in payload
        assert "grounding" in payload
        assert payload["hash"] == payload["hashes"][hash_keys[layer]]
        assert payload["diagnostics"]["layer_hash_key"] == hash_keys[layer]
        assert payload["diagnostics"]["layer_hash"] == payload["hash"]
        assert "edge_distance_diagnostics" in payload
        assert "projection_diagnostics" in payload
        if layer == "coarse-concepts":
            for node in payload["nodes"]:
                expected_all_mid_ids = list(
                    dict.fromkeys(
                        [
                            *node["included_mid_concept_ids"],
                            *node["boundary_mid_concept_ids"],
                            *node["bridge_mid_concept_ids"],
                            *node["outlier_mid_concept_ids"],
                            *node["low_confidence_mid_concept_ids"],
                        ]
                    )
                )
                assert node["all_mid_concept_ids"] == expected_all_mid_ids

        if layer == "chunk-relation":
            distance = payload["edge_distance_diagnostics"]
            assert distance["applicable"] is True
            assert distance["protocol_hash"]
            assert distance["distribution"]["count"] == payload["counts"][
                "chunk_relation_edges"
            ]
            assert distance["by_edge_type"]
            assert payload["projection_diagnostics"]["applicable"] is False
        elif layer in {"mid-concepts", "coarse-concepts"}:
            projection = payload["projection_diagnostics"]
            assert projection["applicable"] is True
            projection_count_key = (
                "mid_concept_edges" if layer == "mid-concepts" else "coarse_concept_edges"
            )
            membership_count_key = (
                "mid_concept_memberships"
                if layer == "mid-concepts"
                else "coarse_concept_memberships"
            )
            assert projection["full_edge_count"] == payload["counts"][projection_count_key]
            assert projection["graph_total_edge_count"] == payload["full_counts"]["edges"]
            assert projection["non_projection_edge_count"] == payload["counts"][
                membership_count_key
            ]
            assert projection["raw_projected_distance_distribution"]["count"] == projection[
                "full_edge_count"
            ]
            assert projection["calibrated_projected_distance_distribution"][
                "count"
            ] == projection["full_edge_count"]
            assert projection["protocol_hash_coverage"] == 1.0
            assert projection["protocol_hash_consistent"] is True
            assert sum(
                item["full_edge_count"]
                for item in projection["by_edge_type"].values()
            ) == projection["full_edge_count"]
            assert all(
                item["protocol_hash_coverage"] == 1.0
                and item["protocol_hash_consistent"] is True
                for item in projection["by_edge_type"].values()
            )
        else:
            assert payload["edge_distance_diagnostics"]["applicable"] is False
            assert payload["projection_diagnostics"]["applicable"] is False


def test_graph_layer_payload_reuses_its_freshness_admission_gate(
    monkeypatch,
    db_session,
    populated_context_graph,
):
    from app.services import context_graph

    knowledge_base = populated_context_graph["knowledge_base"]
    original_gate = context_graph.active_graph_admission_gate
    gate_calls = 0

    def counting_gate(*args, **kwargs):
        nonlocal gate_calls
        gate_calls += 1
        return original_gate(*args, **kwargs)

    monkeypatch.setattr(
        context_graph,
        "active_graph_admission_gate",
        counting_gate,
    )

    payload = context_graph.graph_layer_payload(
        db_session,
        knowledge_base.id,
        "mid-concepts",
        limit=20,
    )

    assert payload["freshness"]["is_admissible"] is True
    assert gate_calls == 1


def test_retrieval_trace_route_maps_replay_invariant_to_closed_409(
    monkeypatch,
    db_session,
):
    from fastapi import HTTPException

    from app.routers import knowledge
    from app.services.context_graph import EntrySelectionTraceInvariantError

    def fail_replay(_db, _trace_id):
        raise EntrySelectionTraceInvariantError(
            "private replay details and row identifiers"
        )

    monkeypatch.setattr(knowledge, "get_retrieval_trace_steps", fail_replay)

    with pytest.raises(HTTPException) as exc_info:
        knowledge.retrieval_trace_steps("stale-trace", db_session)

    assert exc_info.value.status_code == 409
    assert exc_info.value.detail == {
        "code": "retrieval_trace_replay_failed",
        "message": (
            "Persisted retrieval trace is not replayable against the "
            "current PostgreSQL evidence state."
        ),
        "failure_type": "EntrySelectionTraceInvariantError",
    }
    assert "private replay details" not in str(exc_info.value.detail)


def test_context_package_route_maps_replay_invariant_to_closed_409(
    monkeypatch,
    db_session,
):
    from fastapi import HTTPException

    from app.routers import knowledge
    from app.services.retrieval import ContextPackagePublicIntegrityError

    def fail_replay(_db, _package_id):
        raise ContextPackagePublicIntegrityError(
            "private source span and provider-adjacent diagnostics"
        )

    monkeypatch.setattr(knowledge, "get_context_package", fail_replay)

    with pytest.raises(HTTPException) as exc_info:
        knowledge.context_package("stale-package", db_session)

    assert exc_info.value.status_code == 409
    assert exc_info.value.detail == {
        "code": "context_package_replay_failed",
        "message": (
            "Persisted context package is not replayable against the "
            "current PostgreSQL evidence state."
        ),
        "failure_type": "ContextPackagePublicIntegrityError",
    }
    assert "private source span" not in str(exc_info.value.detail)


def test_layer_payloads_without_active_graph_states_remain_closed_contracts(
    monkeypatch,
    db_session,
    populated_context_graph,
):
    from app.schemas import GraphResponse
    from app.services import context_graph

    knowledge_base = populated_context_graph["knowledge_base"]
    monkeypatch.setattr(
        context_graph,
        "_state_bound_layers",
        lambda _db, _knowledge_base_id, _state: (None, None, None),
    )

    expected_reasons = {
        "chunk-relation": "active_chunk_relation_graph_state_missing",
        "mid-concepts": "active_mid_concept_graph_state_missing",
        "coarse-concepts": "active_coarse_concept_graph_state_missing",
    }
    for layer, reason in expected_reasons.items():
        payload = context_graph.graph_layer_payload(
            db_session,
            knowledge_base.id,
            layer,
            limit=20,
        )
        validated = GraphResponse.model_validate(payload)

        assert validated.graph_type == layer
        assert validated.edge_distance_diagnostics.applicable is False
        assert validated.edge_distance_diagnostics.reason == reason
        assert validated.projection_diagnostics.applicable is False
        assert validated.grounding.mid_total == 0
        assert validated.grounding.mid_grounded_rate == 0.0
        assert validated.grounding.coarse_total == 0
        assert validated.grounding.coarse_grounded_rate == 0.0
        if layer == "chunk-relation":
            assert (
                validated.projection_diagnostics.reason
                == "chunk_relation_is_the_projection_source_layer"
            )
        else:
            assert validated.projection_diagnostics.reason == reason


def test_context_graph_stats_without_state_or_vector_pointer_remains_observable(
    monkeypatch,
    db_session,
    populated_context_graph,
):
    from sqlalchemy import func, select

    from app.models import KnowledgeBaseVectorRuntimeState
    from app.services import context_graph

    knowledge_base = populated_context_graph["knowledge_base"]
    pointer = db_session.scalar(
        select(KnowledgeBaseVectorRuntimeState).where(
            KnowledgeBaseVectorRuntimeState.knowledge_base_id
            == knowledge_base.id
        )
    )
    assert pointer is not None
    db_session.delete(pointer)
    db_session.flush()
    monkeypatch.setattr(
        context_graph,
        "latest_context_graph_state",
        lambda _db, _knowledge_base_id: None,
    )

    stats = context_graph.context_graph_stats(db_session, knowledge_base.id)

    assert stats["counts"]["active_chunks"] > 0
    assert stats["context_graph_state_id"] is None
    assert stats["hashes"]["contextual_index_hash"] is None
    assert stats["hashes"]["contextual_index_business_hash"] is None
    assert stats["freshness"]["current_contextual_index_hash"] is None
    assert stats["freshness"]["current_contextual_index_business_hash"] is None
    assert stats["freshness"]["is_stale"] is True
    assert stats["freshness"]["is_admissible"] is False
    assert "context_graph_state_missing" in stats["freshness"][
        "admission_reasons"
    ]
    assert stats["freshness"]["model_call_count"] == 0
    assert stats["freshness"]["gray_zone_rule_inputs_modified"] is False
    assert (
        db_session.scalar(
            select(func.count(KnowledgeBaseVectorRuntimeState.id)).where(
                KnowledgeBaseVectorRuntimeState.knowledge_base_id
                == knowledge_base.id
            )
        )
        == 0
    )


@pytest.mark.asyncio
async def test_knowledge_base_summary_includes_active_graph_state(
    monkeypatch,
    db_session,
    populated_context_graph,
):
    from app.schemas import KnowledgeBaseSummary
    from app.services import context_graph
    from app.services.ingestion import summarize_knowledge_base
    from app.services.retrieval import get_dashboard_snapshot

    def forbidden_admission(*_args, **_kwargs):
        raise AssertionError("list/dashboard overview must not replay full admission")

    monkeypatch.setattr(
        context_graph,
        "active_graph_admission_gate",
        forbidden_admission,
    )

    kb = populated_context_graph["knowledge_base"]
    payload = summarize_knowledge_base(db_session, kb)
    validated = KnowledgeBaseSummary.model_validate(payload)

    assert validated.chunk_count >= 3
    assert validated.active_chunk_count == validated.chunk_count
    assert validated.context_graph_state_id == populated_context_graph["state"].id
    assert validated.context_graph_hash == populated_context_graph["state"].context_graph_hash
    assert validated.stale_reason is None
    assert validated.has_parsed_chunks is True
    assert validated.can_full_reparse is True

    dashboard = get_dashboard_snapshot(db_session, kb.id, include_graph=False)
    overview = dashboard["context_graph"]
    assert overview["counts"]["active_chunks"] == validated.active_chunk_count
    assert overview["counts"]["chunk_relation_edges"] > 0
    assert overview["freshness"]["is_admissible"] is True
    assert overview["freshness"]["admission_replayed"] is False
    assert overview["freshness"]["model_call_count"] == 0
    assert overview["diagnostics"] == {
        "protocol_version": "context_graph_persisted_overview_v1",
        "source": "postgresql_active_state_and_persisted_freshness",
        "summary_only": True,
        "active_graph_admission_replayed": False,
        "provider_request_count": 0,
        "model_call_count": 0,
    }


def test_uploaded_staging_job_is_listed_as_pending_until_batch_starts(db_session, sample_knowledge_base):
    from app.core.config import get_settings
    from app.schemas import KnowledgeBaseFileSummary
    from app.services.ingestion import create_job, create_uploaded_files_batch, list_knowledge_base_files, register_uploaded_file

    storage_root = get_settings().knowledge_base_paths_for_source_root(
        sample_knowledge_base.source_root
    )["storage_root"]
    storage_root.mkdir(parents=True, exist_ok=True)
    source_path = storage_root / "complex-network.md"
    source_path.write_text("# Complex network\n\nA source waiting for parse.\n", encoding="utf-8")

    document, upload_job = register_uploaded_file(db_session, sample_knowledge_base, source_path)
    assert upload_job.status == "queued"
    assert upload_job.batch_id is None

    files = list_knowledge_base_files(db_session, sample_knowledge_base.id)
    uploaded_item = next(item for item in files if item["source_path"] == str(source_path))
    uploaded_contract = KnowledgeBaseFileSummary.model_validate(uploaded_item)
    assert uploaded_item["document_id"] == document.id
    assert uploaded_item["status"] == "pending"
    assert uploaded_item["batch_id"] is None
    assert uploaded_contract.chunk_count == 0
    assert uploaded_contract.current_version == 0
    assert uploaded_contract.active_chunks == 0
    assert uploaded_contract.checksum is None
    assert uploaded_contract.chunk_version is None

    batch = create_uploaded_files_batch(db_session, sample_knowledge_base.id, [source_path])
    create_job(
        db_session,
        knowledge_base_id=sample_knowledge_base.id,
        document_id=document.id,
        trigger_source="upload",
        batch_id=batch.id,
        source_path=str(source_path),
    )

    files = list_knowledge_base_files(db_session, sample_knowledge_base.id)
    batched_item = next(item for item in files if item["source_path"] == str(source_path))
    assert batched_item["status"] == "parsing"
    assert batched_item["batch_id"] == batch.id
    KnowledgeBaseFileSummary.model_validate(batched_item)


def test_knowledge_base_file_summary_accepts_unregistered_pending_storage_file(
    db_session,
    sample_knowledge_base,
):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from app.core.config import get_settings
    from app.schemas import KnowledgeBaseFileSummary
    from app.services.ingestion import list_knowledge_base_files

    storage_root = get_settings().knowledge_base_paths_for_source_root(
        sample_knowledge_base.source_root
    )["storage_root"]
    storage_root.mkdir(parents=True, exist_ok=True)
    source_path = storage_root / "unregistered-pending.md"
    source_path.write_text("# Pending\n", encoding="utf-8")

    item = next(
        item
        for item in list_knowledge_base_files(db_session, sample_knowledge_base.id)
        if item["source_path"] == str(source_path)
    )
    validated = KnowledgeBaseFileSummary.model_validate(item)

    assert validated.document_id is None
    assert validated.status == "pending"
    assert validated.chunk_count == 0
    assert validated.current_version == 0
    assert validated.active_chunks == 0
    assert validated.chunk_version is None
    assert validated.checksum is None

    app = FastAPI()

    @app.get("/files", response_model=list[KnowledgeBaseFileSummary])
    def files_route():
        return [item]

    with TestClient(app) as client:
        response = client.get("/files")
    assert response.status_code == 200
    assert set(response.json()[0]) == set(KnowledgeBaseFileSummary.model_fields)
    assert response.json()[0]["document_id"] is None


def test_knowledge_base_file_summary_uses_document_active_version_not_kb_max(
    db_session,
    populated_context_graph,
):
    from sqlalchemy import func, select

    from app.models import Chunk, Document, DocumentVersion
    from app.schemas import KnowledgeBaseFileSummary
    from app.services.ingestion import list_knowledge_base_files

    knowledge_base = populated_context_graph["knowledge_base"]
    document = db_session.scalar(
        select(Document).where(Document.knowledge_base_id == knowledge_base.id)
    )
    assert document is not None
    active_version = db_session.scalar(
        select(DocumentVersion).where(
            DocumentVersion.document_id == document.id,
            DocumentVersion.is_active.is_(True),
        )
    )
    assert active_version is not None
    chunk_count = db_session.scalar(
        select(func.count(Chunk.id)).where(
            Chunk.document_id == document.id,
            Chunk.state == "active",
        )
    ) or 0
    knowledge_base.current_chunk_version = int(active_version.version) + 1
    db_session.commit()

    item = next(
        item
        for item in list_knowledge_base_files(db_session, knowledge_base.id)
        if item["document_id"] == document.id
    )
    validated = KnowledgeBaseFileSummary.model_validate(item)

    assert validated.current_version == active_version.version
    assert validated.chunk_version == active_version.version
    assert validated.current_version != knowledge_base.current_chunk_version
    assert validated.chunk_count == chunk_count
    assert validated.active_chunks == chunk_count
    assert validated.checksum == active_version.checksum
    assert validated.last_ingested_at == active_version.created_at


def test_knowledge_base_delete_blocks_active_upload_replacement_before_side_effects(
    monkeypatch,
    db_session,
    sample_knowledge_base,
):
    from app.models import IngestionCompensationLog
    from app.services import maintenance, qdrant_outbox, upload_replacement

    blocker = IngestionCompensationLog(
        knowledge_base_id=sample_knowledge_base.id,
        operation=upload_replacement.UPLOAD_SOURCE_REPLACEMENT_OPERATION,
        target_ids_json=[],
        payload_json={},
        status="manual_review",
    )
    db_session.add(blocker)
    db_session.commit()

    def forbidden_qdrant_preflight(*_args, **_kwargs):
        raise AssertionError("upload replacement blockers must gate deletion before Qdrant work")

    monkeypatch.setattr(qdrant_outbox, "pending_qdrant_intents", forbidden_qdrant_preflight)

    with pytest.raises(maintenance.MaintenanceConflict, match="durable upload replacement intents") as blocked:
        maintenance.delete_knowledge_base_data(db_session, sample_knowledge_base)

    assert f"{blocker.id}:manual_review" in str(blocked.value)
    assert db_session.get(type(sample_knowledge_base), sample_knowledge_base.id) is not None
    assert db_session.get(IngestionCompensationLog, blocker.id) is not None


@pytest.mark.asyncio
async def test_knowledge_base_delete_route_maps_ingestion_advisory_contention_to_409(
    monkeypatch,
    db_session,
    sample_knowledge_base,
):
    from contextlib import asynccontextmanager

    from fastapi import HTTPException

    from app.routers import knowledge
    from app.services.ingestion_resource_lock import IngestionResourceBusyError

    @asynccontextmanager
    async def busy_ingestion_lock(*_args, **_kwargs):
        raise IngestionResourceBusyError(
            {
                "wait_seconds": 0.01,
                "resource_key": f"knowledge_base:{sample_knowledge_base.id}",
                "operation": "delete_knowledge_base_data",
                "reason": "resource_lock_timeout",
                "retryable": True,
            }
        )
        yield

    monkeypatch.setattr(
        knowledge,
        "knowledge_base_ingestion_resource_lock",
        busy_ingestion_lock,
    )

    with pytest.raises(HTTPException) as blocked:
        await knowledge.delete_knowledge_base(sample_knowledge_base.id, db_session)

    assert blocked.value.status_code == 409
    assert blocked.value.detail["code"] == "ingestion_resource_busy"
    assert "retry" in str(blocked.value.detail).lower()
    assert db_session.get(type(sample_knowledge_base), sample_knowledge_base.id) is not None


@pytest.mark.asyncio
async def test_search_route_commits_retrieval_trace(db_session, populated_context_graph):
    from sqlalchemy import func, select

    from app.models import GraphRetrievalStep, RetrievalTrace
    from app.routers.search import search
    from app.schemas import SearchFilters, SearchRequest

    kb = populated_context_graph["knowledge_base"]
    payload = await search(
        SearchRequest(knowledge_base_id=kb.id, query="Bayes theorem prior posterior", filters=SearchFilters(), top_k=3),
        db_session,
    )
    trace_id = payload["model_audit"]["retrieval_trace_id"]

    assert payload["retrieval_granularity"] == "mid"
    assert payload["model_audit"]["retrieval_granularity"] == "mid"
    assert db_session.get(RetrievalTrace, trace_id) is not None
    trace = db_session.get(RetrievalTrace, trace_id)
    assert trace.diagnostics_json["retrieval_granularity"] == "mid"
    assert db_session.scalar(select(func.count(GraphRetrievalStep.id)).where(GraphRetrievalStep.retrieval_trace_id == trace_id)) >= 4


@pytest.mark.asyncio
async def test_rebuild_graph_endpoint_queues_worker_batch(monkeypatch, db_session, populated_context_graph):
    from app.models import IngestionBatch
    from app.routers import knowledge
    from app.schemas import RebuildGraphRequest

    kb = populated_context_graph["knowledge_base"]
    captured: dict[str, object] = {}

    class Settings:
        ingestion_execution_mode = "celery"
        ingestion_task_queue = "test-queue"
        data_root = Path("/tmp/data")

    class FakeTask:
        def apply_async(self, args, queue):
            captured["args"] = args
            captured["queue"] = queue
            return types.SimpleNamespace(id="task-1")

    fake_worker = types.ModuleType("worker_app")
    fake_tasks = types.ModuleType("worker_app.tasks")
    fake_tasks.rebuild_context_graph_batch = FakeTask()
    monkeypatch.setitem(sys.modules, "worker_app", fake_worker)
    monkeypatch.setitem(sys.modules, "worker_app.tasks", fake_tasks)
    monkeypatch.setattr(knowledge, "get_settings", lambda: Settings())

    payload = await knowledge.rebuild_graph_endpoint(RebuildGraphRequest(dry_run=False), kb.id, db_session)

    assert payload["state"] == "queued"
    assert payload["batch_id"]
    assert captured == {"args": [payload["batch_id"]], "queue": "test-queue"}
    batch = db_session.get(IngestionBatch, payload["batch_id"])
    assert batch is not None
    assert batch.trigger_source == "graph_rebuild"
    assert (batch.stats or {}).get("celery_task_name") == "rebuild_context_graph_batch"


def test_policy_reconcile_creates_context_graph_policy(db_session, sample_knowledge_base):
    from sqlalchemy import func, select

    from app.models import PolicyState
    from app.services.maintenance import reconcile_policy_state

    assert reconcile_policy_state(db_session, sample_knowledge_base.id) == 1
    db_session.commit()
    policy = db_session.scalar(select(PolicyState).where(PolicyState.knowledge_base_id == sample_knowledge_base.id))
    assert policy is not None
    assert policy.policy_family == "context_graph_bandit"
    assert "bridge_edge_exploration" in policy.weights_json


@pytest.mark.asyncio
async def test_agent_planner_repairs_invalid_json_shape(monkeypatch):
    from app.services import agent_graph

    class FakePlanner:
        calls = 0

        async def classify_json(self, system_prompt, user_prompt, fallback=None):
            self.calls += 1
            if self.calls == 1:
                return {"analysis": "not the contract"}
            return {
                "typed_actions": [
                    {
                        "action_type": "select_entry_nodes",
                        "target_ids": [],
                        "reason": "repair retry returned typed actions",
                        "budget_request": {},
                        "expected_evidence": {},
                        "stop_condition": {},
                    }
                ]
            }

    planner = FakePlanner()
    monkeypatch.setattr(agent_graph, "ChatProvider", lambda: planner)

    actions, raw = await agent_graph.propose_agent_plan("What is a posterior?", [], {"intent": "definition"}, agent_graph.agent_operating_envelope())

    assert planner.calls == 2
    assert actions[0]["action_type"] == "select_entry_nodes"
    assert raw["planner_repair"]["attempted"] is True


def test_runtime_check_payload_matches_response_schema(monkeypatch):
    from app.schemas import RuntimeCheckResponse
    from app.services import runtime_settings

    monkeypatch.setattr(
        runtime_settings,
        "env_sync_status",
        lambda: {"synced": True, "missing_keys": [], "extra_keys": [], "deprecated_keys": [], "bom_keys": []},
    )
    monkeypatch.setattr(runtime_settings, "_check_postgres", lambda: True)
    monkeypatch.setattr(runtime_settings, "_check_qdrant", lambda: True)
    monkeypatch.setattr(runtime_settings, "_check_redis", lambda: True)
    monkeypatch.setattr(runtime_settings, "_check_model_bridge", lambda status=None: None)

    payload = runtime_settings.runtime_check_payload()
    validated = RuntimeCheckResponse.model_validate(payload)

    assert validated.env_sync["synced"] is True
    assert validated.infrastructure["postgres"] is True
    assert validated.blocking_issues == []


@pytest.mark.asyncio
async def test_api_lifespan_syncs_model_bridge_runtime_config(monkeypatch):
    from app import main

    calls: list[tuple[str, object]] = []
    monkeypatch.setattr(
        main,
        "refresh_runtime_settings_if_needed",
        lambda force=False, sync_bridge=True: calls.append(
            ("refresh", (force, sync_bridge))
        ),
    )
    monkeypatch.setattr(
        main,
        "get_settings",
        lambda: SimpleNamespace(
            app_env="development",
            api_key_list=[],
            model_bridge_enabled=True,
            data_root=Path("/tmp/test-api-storage-root"),
        ),
    )
    monkeypatch.setattr(
        main,
        "ensure_storage_durability_ready",
        lambda settings=None, force_probe=False: calls.append(("storage", force_probe)) or {},
    )
    monkeypatch.setattr(main, "sync_model_bridge_runtime_config", lambda settings=None: calls.append(("bridge", settings)))
    monkeypatch.setattr(main, "ensure_schema", lambda: calls.append(("schema", None)))
    monkeypatch.setattr(
        main,
        "reconcile_builtin_default_profile_startup",
        lambda: calls.append(("profile_default", None)) or {},
    )
    monkeypatch.setattr(
        main,
        "reconcile_profile_lifecycle_events_startup",
        lambda: calls.append(("profile_lifecycle", None)) or {"failed": 0},
    )

    async def reconcile_uploads():
        calls.append(("upload_replacement", None))
        return {}

    async def reconcile_storage():
        calls.append(("storage_maintenance", None))
        return {}

    monkeypatch.setattr(
        main,
        "reconcile_pending_upload_replacements_startup",
        reconcile_uploads,
    )
    monkeypatch.setattr(
        main,
        "reconcile_pending_storage_maintenance_startup",
        reconcile_storage,
    )
    monkeypatch.setattr(main, "finalize_interrupted_batches", lambda: calls.append(("finalize", None)))

    async with main.lifespan(main.app):
        pass

    assert calls[0] == ("storage", False)
    assert calls[1] == ("schema", None)
    assert calls[2] == ("refresh", (True, False))
    assert calls[3][0] == "bridge"
    assert [name for name, _ in calls[4:]] == [
        "profile_default",
        "profile_lifecycle",
        "upload_replacement",
        "storage_maintenance",
        "finalize",
    ]


def test_runtime_env_sync_treats_legacy_runtime_keys_as_deprecated(monkeypatch, tmp_path):
    from app.services import runtime_settings

    env_path = tmp_path / ".env"
    example_path = tmp_path / ".env.example"
    env_path.write_text(
        "DATABASE_URL=sqlite:///runtime.db\n"
        "FIXED_CHUNK_SIZE_TOKENS=512\n"
        "FIXED_CHUNK_OVERLAP_TOKENS=80\n"
        "CONTEXT_PACKAGE_TOKEN_BUDGET=2400\n"
        "MID_CONCEPT_EXTRACTION_MAX_MODEL_BATCHES=4\n"
        "CHUNK_TOKEN_BUDGET=2400\n"
        "SEMANTIC_CHUNKING_ENABLED=false\n"
        "COMMUNITY_LOUVAIN_RESOLUTION=1.0\n"
        "SIGNAL_EXTRACTION_MAX_MODEL_BATCHES=4\n"
        "INGESTION_FILE_CONCURRENCY=3\n"
        "RETRIEVAL_LAYER_ENABLED=true\n"
        "RETRIEVAL_CACHE_TTL_SECONDS=120\n"
        "CITATION_VERIFICATION_SAMPLE_MAX=3\n"
        "GRAPH_OVERVIEW_MAX_NODES=260\n"
        "RERANKER_ENABLED=true\n"
        "RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2\n"
        "HF_HUB_OFFLINE=1\n",
        encoding="utf-8",
    )
    example_path.write_text(
        "DATABASE_URL=\n"
        "FIXED_CHUNK_SIZE_TOKENS=512\n"
        "FIXED_CHUNK_OVERLAP_TOKENS=80\n"
        "CONTEXT_PACKAGE_TOKEN_BUDGET=2400\n"
        "MID_CONCEPT_EXTRACTION_MAX_MODEL_BATCHES=4\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(runtime_settings, "ENV_PATH", env_path)
    monkeypatch.setattr(runtime_settings, "ENV_EXAMPLE_PATH", example_path)

    status = runtime_settings.env_sync_status()
    assert status["synced"] is True
    assert status["deprecated_keys"] == [
        "CHUNK_TOKEN_BUDGET",
        "CITATION_VERIFICATION_SAMPLE_MAX",
        "COMMUNITY_LOUVAIN_RESOLUTION",
        "GRAPH_OVERVIEW_MAX_NODES",
        "HF_HUB_OFFLINE",
        "INGESTION_FILE_CONCURRENCY",
        "RERANKER_ENABLED",
        "RERANKER_MODEL",
        "RETRIEVAL_CACHE_TTL_SECONDS",
        "RETRIEVAL_LAYER_ENABLED",
        "SEMANTIC_CHUNKING_ENABLED",
        "SIGNAL_EXTRACTION_MAX_MODEL_BATCHES",
    ]

    assert runtime_settings.normalize_env_file() is True
    cleaned = env_path.read_text(encoding="utf-8")
    assert "SIGNAL_EXTRACTION_MAX_MODEL_BATCHES" not in cleaned
    assert "CHUNK_TOKEN_BUDGET" not in cleaned
    assert "SEMANTIC_CHUNKING_ENABLED" not in cleaned
    assert "COMMUNITY_LOUVAIN_RESOLUTION" not in cleaned
    assert "INGESTION_FILE_CONCURRENCY" not in cleaned
    assert "RETRIEVAL_LAYER_ENABLED" not in cleaned
    assert "RETRIEVAL_CACHE_TTL_SECONDS" not in cleaned
    assert "CITATION_VERIFICATION_SAMPLE_MAX" not in cleaned
    assert "GRAPH_OVERVIEW_MAX_NODES" not in cleaned
    assert "RERANKER_ENABLED" not in cleaned
    assert "RERANKER_MODEL" not in cleaned
    assert "HF_HUB_OFFLINE" not in cleaned


def test_runtime_env_sync_allows_compose_only_keys_in_the_single_root_env(
    monkeypatch,
    tmp_path,
):
    from app.services import runtime_settings

    env_path = tmp_path / ".env"
    example_path = tmp_path / ".env.example"
    env_path.write_text("DATABASE_URL=sqlite:///runtime.db\n", encoding="utf-8")
    example_path.write_text(
        "DATABASE_URL=\nSAMPLE_IMPORT_PATH=./sample-import\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(runtime_settings, "ENV_PATH", env_path)
    monkeypatch.setattr(runtime_settings, "ENV_EXAMPLE_PATH", example_path)

    status = runtime_settings.env_sync_status()

    assert status["synced"] is True
    assert status["missing_keys"] == []
    assert status["extra_keys"] == []

    env_path.write_text(
        "DATABASE_URL=sqlite:///runtime.db\nSAMPLE_IMPORT_PATH=./sample-import\n",
        encoding="utf-8",
    )
    persisted_status = runtime_settings.env_sync_status()
    assert persisted_status["synced"] is True
    assert persisted_status["extra_keys"] == []


def test_runtime_env_sync_does_not_require_deployment_only_postgres_keys(
    monkeypatch,
    tmp_path,
):
    from app.services import runtime_settings

    env_path = tmp_path / ".env"
    example_path = tmp_path / ".env.example"
    env_path.write_text("CHAT_MODEL=unit-model\nGRAPH_RESOLVE_IP=\n", encoding="utf-8")
    example_path.write_text(
        "CHAT_MODEL=unit-model\nGRAPH_RESOLVE_IP=\n"
        "POSTGRES_DB=\nPOSTGRES_USER=\nPOSTGRES_PASSWORD=\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(runtime_settings, "ENV_PATH", env_path)
    monkeypatch.setattr(runtime_settings, "ENV_EXAMPLE_PATH", example_path)

    status = runtime_settings.env_sync_status()

    assert status["settings_file_schema_synced"] is True
    assert status["missing_keys"] == []


def test_model_settings_payload_uses_fixed_chunk_and_context_budget(monkeypatch):
    from app.core.config import get_settings, model_bridge_client_base_url
    from app.services import runtime_settings

    get_settings.cache_clear()
    monkeypatch.setenv("FIXED_CHUNK_SIZE_TOKENS", "512")
    monkeypatch.setenv("FIXED_CHUNK_OVERLAP_TOKENS", "80")
    monkeypatch.setenv("CONTEXT_PACKAGE_TOKEN_BUDGET", "2400")
    monkeypatch.setenv("UPLOAD_MAX_BYTES", "104857600")
    monkeypatch.setenv("AGENT_COARSE_INITIAL_BUDGET", "6")
    monkeypatch.setenv("AGENT_COARSE_TOP_K", "4")
    monkeypatch.setenv("AGENT_MID_INITIAL_BUDGET", "9")
    monkeypatch.setenv("AGENT_CHUNK_INITIAL_BUDGET", "11")
    monkeypatch.setenv("AGENT_STRUCTURE_RESTORE_PER_CHUNK_BUDGET", "3")
    monkeypatch.setenv("CONCEPT_I18N_ENABLED", "false")
    monkeypatch.setenv("QUERY_FACET_BILINGUAL_ENABLED", "false")
    monkeypatch.setattr(runtime_settings, "current_runtime_settings_version", lambda: "unit-version")
    monkeypatch.setattr(
        runtime_settings,
        "_env_entries",
        lambda _path: {
            "FIXED_CHUNK_SIZE_TOKENS": "512",
            "FIXED_CHUNK_OVERLAP_TOKENS": "80",
            "CONTEXT_PACKAGE_TOKEN_BUDGET": "2400",
            "UPLOAD_MAX_BYTES": "104857600",
            "AGENT_REQUEST_CONCURRENCY": "5",
            "SOURCE_IO_CONCURRENCY": "7",
            "AGENT_REQUEST_QUEUE_LIMIT": "9",
            "AGENT_REQUEST_QUEUE_TIMEOUT_SECONDS": "45",
            "AGENT_REQUEST_LEASE_TTL_SECONDS": "360",
            "RETRIEVAL_RESULT_TOP_K_DEFAULT": "7",
            "AGENT_COARSE_INITIAL_BUDGET": "6",
            "AGENT_COARSE_TOP_K": "4",
            "AGENT_COARSE_DRILLDOWN_MID_INITIAL_BUDGET": "10",
            "AGENT_MID_INITIAL_BUDGET": "9",
            "AGENT_CHUNK_INITIAL_BUDGET": "11",
            "AGENT_STRUCTURE_RESTORE_PER_CHUNK_BUDGET": "3",
            "CONCEPT_I18N_ENABLED": "false",
            "QUERY_FACET_BILINGUAL_ENABLED": "false",
            "RQ_KMEANS_LEVELS": "4",
        },
    )

    payload = runtime_settings.model_settings_payload()

    assert payload["fixed_chunk_size_tokens"] == 512
    assert payload["fixed_chunk_overlap_tokens"] == 80
    assert payload["context_package_token_budget"] == 2400
    assert payload["upload_max_bytes"] == 104857600
    assert payload["agent_request_concurrency"] == 5
    assert payload["source_io_concurrency"] == 7
    assert payload["agent_request_queue_limit"] == 9
    assert payload["agent_request_queue_timeout_seconds"] == 45
    assert payload["agent_request_lease_ttl_seconds"] == 360
    assert payload["retrieval_result_top_k_default"] == 7
    assert payload["agent_coarse_initial_budget"] == 6
    assert payload["agent_coarse_top_k"] == 4
    assert payload["agent_coarse_drilldown_mid_initial_budget"] == 10
    assert payload["agent_mid_initial_budget"] == 9
    assert payload["agent_chunk_initial_budget"] == 11
    assert payload["agent_structure_restore_per_chunk_budget"] == 3
    assert payload["concept_i18n_enabled"] is False
    assert payload["query_facet_bilingual_enabled"] is False
    assert payload["rq_kmeans_levels"] == 3
    assert "agent_coarse_initial_budget" in payload["lifecycle"]["hot_reloadable"]
    assert "agent_coarse_top_k" in payload["lifecycle"]["hot_reloadable"]
    assert "agent_coarse_drilldown_mid_initial_budget" in payload["lifecycle"]["hot_reloadable"]
    assert "agent_mid_initial_budget" in payload["lifecycle"]["hot_reloadable"]
    assert "agent_chunk_initial_budget" in payload["lifecycle"]["hot_reloadable"]
    assert "agent_structure_restore_per_chunk_budget" in payload["lifecycle"]["hot_reloadable"]
    assert "upload_max_bytes" in payload["lifecycle"]["hot_reloadable"]
    assert "agent_request_concurrency" in payload["lifecycle"]["hot_reloadable"]
    assert "source_io_concurrency" in payload["lifecycle"]["hot_reloadable"]
    assert "agent_request_queue_limit" in payload["lifecycle"]["hot_reloadable"]
    assert "agent_request_queue_timeout_seconds" in payload["lifecycle"]["hot_reloadable"]
    assert "agent_request_lease_ttl_seconds" in payload["lifecycle"]["hot_reloadable"]
    assert "chat_json_max_tokens" in payload["lifecycle"]["hot_reloadable"]
    assert payload["chat_json_max_tokens"] == 12000
    assert payload["agent_coarse_total_budget"] > 0
    assert payload["agent_chunk_top_k"] > 0
    assert "lifecycle" in payload
    assert "chunk_token_budget" not in payload
    assert "semantic_chunking_enabled" not in payload
    assert "agent_coarse_entry_budget" not in payload


def test_model_settings_payload_keeps_bridge_targets_editable(monkeypatch):
    from app.core import config
    from app.core.config import get_settings, model_bridge_client_base_url
    from app.services import runtime_settings

    get_settings.cache_clear()
    monkeypatch.setenv("MODEL_BRIDGE_ENABLED", "true")
    monkeypatch.setenv("MODEL_BRIDGE_PORT", "8765")
    monkeypatch.setenv("CHAT_API_PROTOCOL", "anthropic")
    monkeypatch.setenv("GRAPH_API_PROTOCOL", "openai")
    monkeypatch.setenv("CHAT_BASE_URL", "https://chat.example.test/v1")
    monkeypatch.setenv("EMBEDDING_BASE_URL", "https://embedding.example.test/v1")
    monkeypatch.setenv("CHAT_RESOLVE_IP", "1.1.1.1")
    monkeypatch.setenv("EMBEDDING_RESOLVE_IP", "2.2.2.2")
    monkeypatch.setattr(
        runtime_settings,
        "_env_entries",
        lambda _path: {
            "MODEL_BRIDGE_ENABLED": "true",
            "MODEL_BRIDGE_PORT": "8765",
            "CHAT_API_PROTOCOL": "anthropic",
            "GRAPH_API_PROTOCOL": "openai",
            "CHAT_BASE_URL": "https://chat.example.test/v1",
            "EMBEDDING_BASE_URL": "https://embedding.example.test/v1",
            "CHAT_RESOLVE_IP": "1.1.1.1",
            "EMBEDDING_RESOLVE_IP": "2.2.2.2",
        },
    )
    monkeypatch.setattr(
        config,
        "_read_workspace_env",
        lambda: runtime_settings._env_entries(runtime_settings.ENV_PATH),
    )
    get_settings.cache_clear()
    monkeypatch.setattr(runtime_settings, "current_runtime_settings_version", lambda: "unit-version")
    monkeypatch.setattr(runtime_settings, "model_bridge_status_payload", lambda settings=None, env_entries=None: {"enabled": True, "config_matches": True})

    payload = runtime_settings.model_settings_payload()

    assert payload["provider"] == "multi_protocol"
    assert payload["chat_api_protocol"] == "anthropic"
    assert payload["graph_api_protocol"] == "openai"
    assert payload["chat_base_url"] == "https://chat.example.test/v1"
    assert payload["embedding_base_url"] == "https://embedding.example.test/v1"
    assert payload["effective_chat_base_url"] == model_bridge_client_base_url(8765)
    assert payload["effective_embedding_base_url"] == model_bridge_client_base_url(8765)
    assert payload["chat_resolve_ip"] == "1.1.1.1"
    assert payload["embedding_resolve_ip"] == "2.2.2.2"
    assert "chat_api_key" not in payload
    assert "graph_api_key" not in payload
    assert "embedding_api_key" not in payload
    assert "Authorization" not in json.dumps(payload, ensure_ascii=False)


def test_update_model_settings_reloads_model_bridge(monkeypatch, tmp_path):
    from app.core.config import get_settings
    from app.services import runtime_settings

    env_path = tmp_path / ".env"
    example_path = tmp_path / ".env.example"
    env_path.write_text(
        "\n".join(
            [
                "MODEL_BRIDGE_ENABLED=true",
                "MODEL_BRIDGE_PORT=8765",
                "MODEL_BRIDGE_ADMIN_TOKEN=unit-token",
                "CHAT_API_PROTOCOL=openai",
                "CHAT_BASE_URL=https://chat.example.test/v1",
                "EMBEDDING_BASE_URL=https://embedding.example.test/v1",
                "CHAT_RESOLVE_IP=1.1.1.1",
                "EMBEDDING_RESOLVE_IP=2.2.2.2",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    example_path.write_text(env_path.read_text(encoding="utf-8"), encoding="utf-8")
    monkeypatch.setattr(runtime_settings, "ENV_PATH", env_path)
    monkeypatch.setattr(runtime_settings, "ENV_EXAMPLE_PATH", example_path)
    monkeypatch.setenv("MODEL_BRIDGE_ENABLED", "true")
    monkeypatch.setenv("MODEL_BRIDGE_PORT", "8765")
    monkeypatch.setenv("MODEL_BRIDGE_ADMIN_TOKEN", "unit-token")
    monkeypatch.setenv("CHAT_API_PROTOCOL", "openai")
    monkeypatch.setattr(runtime_settings, "publish_runtime_settings_version", lambda changed_keys, source="api": {"changed_keys": changed_keys, "source": source})
    monkeypatch.setattr(runtime_settings, "model_bridge_status_payload", lambda settings=None, env_entries=None: {"enabled": True, "config_matches": True})
    reload_calls: list[dict] = []

    def fake_reload_model_bridge(settings=None, env_entries=None):
        reload_calls.append(dict(env_entries or {}))
        return {"attempted": True, "ok": True, "config_version": "bridge-version"}

    monkeypatch.setattr(runtime_settings, "reload_model_bridge", fake_reload_model_bridge)
    monkeypatch.setattr(
        runtime_settings,
        "preflight_model_bridge_reload",
        lambda **_kwargs: {
            "desired": {},
            "previous": {
                "chat_api_protocol": "openai",
                "chat_target_base_url": "https://chat.example.test:443/v1",
                "chat_resolve_ip": "1.1.1.1",
                "embedding_target_base_url": "https://embedding.example.test:443/v1",
                "embedding_resolve_ip": "2.2.2.2",
                "timeout": 180,
            },
        },
    )
    get_settings.cache_clear()

    result = runtime_settings.update_model_settings(
        {
            # Chat endpoint changes are hot. Embedding endpoint changes alter
            # vector identity and must use the rebuild candidate lifecycle.
            "chat_api_protocol": "anthropic",
            "chat_base_url": "https://chat.example.test/v2",
            "upload_max_bytes": 209715200,
            "agent_request_concurrency": 5,
            "source_io_concurrency": 7,
            "agent_request_queue_limit": 12,
            "agent_request_queue_timeout_seconds": 40,
            "agent_request_lease_ttl_seconds": 360,
            "concept_i18n_enabled": True,
            "query_facet_bilingual_enabled": True,
            "agent_coarse_initial_budget": 6,
            "agent_coarse_top_k": 4,
            "agent_coarse_drilldown_mid_initial_budget": 10,
            "agent_mid_initial_budget": 9,
            "agent_chunk_initial_budget": 11,
            "agent_structure_restore_per_chunk_budget": 3,
        }
    )

    assert reload_calls
    assert reload_calls[-1]["CHAT_API_PROTOCOL"] == "anthropic"
    assert reload_calls[-1]["CHAT_BASE_URL"] == "https://chat.example.test/v2"
    assert reload_calls[-1]["UPLOAD_MAX_BYTES"] == "209715200"
    assert reload_calls[-1]["AGENT_REQUEST_CONCURRENCY"] == "5"
    assert reload_calls[-1]["SOURCE_IO_CONCURRENCY"] == "7"
    assert reload_calls[-1]["AGENT_REQUEST_QUEUE_LIMIT"] == "12"
    assert reload_calls[-1]["AGENT_REQUEST_QUEUE_TIMEOUT_SECONDS"] == "40"
    assert reload_calls[-1]["AGENT_REQUEST_LEASE_TTL_SECONDS"] == "360"
    assert reload_calls[-1]["CONCEPT_I18N_ENABLED"] == "true"
    assert reload_calls[-1]["QUERY_FACET_BILINGUAL_ENABLED"] == "true"
    assert reload_calls[-1]["AGENT_COARSE_INITIAL_BUDGET"] == "6"
    assert reload_calls[-1]["AGENT_COARSE_TOP_K"] == "4"
    assert reload_calls[-1]["AGENT_COARSE_DRILLDOWN_MID_INITIAL_BUDGET"] == "10"
    assert reload_calls[-1]["AGENT_MID_INITIAL_BUDGET"] == "9"
    assert reload_calls[-1]["AGENT_CHUNK_INITIAL_BUDGET"] == "11"
    assert reload_calls[-1]["AGENT_STRUCTURE_RESTORE_PER_CHUNK_BUDGET"] == "3"
    assert result["concept_i18n_enabled"] is True
    assert result["chat_api_protocol"] == "anthropic"
    assert result["upload_max_bytes"] == 209715200
    assert result["agent_request_concurrency"] == 5
    assert result["source_io_concurrency"] == 7
    assert result["agent_request_queue_limit"] == 12
    assert result["agent_request_queue_timeout_seconds"] == 40
    assert result["agent_request_lease_ttl_seconds"] == 360
    assert result["query_facet_bilingual_enabled"] is True
    assert result["agent_coarse_initial_budget"] == 6
    assert result["agent_coarse_top_k"] == 4
    assert result["agent_coarse_drilldown_mid_initial_budget"] == 10
    assert result["agent_mid_initial_budget"] == 9
    assert result["agent_chunk_initial_budget"] == 11
    assert result["agent_structure_restore_per_chunk_budget"] == 3
    assert result["model_bridge_status"]["last_reload"]["ok"] is True


def test_runtime_refresh_syncs_model_bridge_after_version_change(monkeypatch):
    from app.core.config import get_settings
    from app.services import runtime_settings

    get_settings.cache_clear()
    monkeypatch.setenv("MODEL_BRIDGE_ENABLED", "true")
    monkeypatch.setenv("MODEL_BRIDGE_PORT", "8765")
    monkeypatch.setattr(runtime_settings, "_LAST_RUNTIME_SETTINGS_VERSION", "old-version")
    monkeypatch.setattr(runtime_settings, "_redis_client", lambda: SimpleNamespace(get=lambda key: "new-version"))
    sync_calls: list[dict] = []

    def fake_sync_model_bridge_runtime_config(settings=None, env_entries=None, raise_on_error=True):
        sync_calls.append({"enabled": settings.model_bridge_enabled, "raise_on_error": raise_on_error})
        return {"attempted": True, "ok": True, "config_version": "bridge-version"}

    monkeypatch.setattr(runtime_settings, "sync_model_bridge_runtime_config", fake_sync_model_bridge_runtime_config)

    result = runtime_settings.refresh_runtime_settings_if_needed()

    assert result["refreshed"] is True
    assert result["runtime_settings_version"] == "new-version"
    assert result["model_bridge_sync"]["ok"] is True
    assert sync_calls == [{"enabled": True, "raise_on_error": True}]


def test_runtime_refresh_fails_fast_when_model_bridge_sync_fails(monkeypatch):
    from app.core.config import get_settings
    from app.services import runtime_settings

    get_settings.cache_clear()
    monkeypatch.setenv("MODEL_BRIDGE_ENABLED", "true")
    monkeypatch.setenv("MODEL_BRIDGE_PORT", "8765")
    monkeypatch.setattr(runtime_settings, "_LAST_RUNTIME_SETTINGS_VERSION", "old-version")
    monkeypatch.setattr(runtime_settings, "_redis_client", lambda: SimpleNamespace(get=lambda key: "new-version"))

    def fake_sync_model_bridge_runtime_config(settings=None, env_entries=None, raise_on_error=True):
        raise RuntimeError("bridge sync failed")

    monkeypatch.setattr(runtime_settings, "sync_model_bridge_runtime_config", fake_sync_model_bridge_runtime_config)

    with pytest.raises(RuntimeError, match="bridge sync failed"):
        runtime_settings.refresh_runtime_settings_if_needed()


def test_runtime_check_repairs_model_bridge_config_mismatch(monkeypatch):
    from app.core.config import get_settings
    from app.services import runtime_settings

    get_settings.cache_clear()
    monkeypatch.setenv("MODEL_BRIDGE_ENABLED", "true")
    monkeypatch.setenv("MODEL_BRIDGE_PORT", "8765")
    monkeypatch.setattr(
        runtime_settings,
        "env_sync_status",
        lambda: {"synced": True, "missing_keys": [], "extra_keys": [], "deprecated_keys": [], "bom_keys": []},
    )
    monkeypatch.setattr(runtime_settings, "_check_postgres", lambda: True)
    monkeypatch.setattr(runtime_settings, "_check_qdrant", lambda: True)
    monkeypatch.setattr(runtime_settings, "_check_redis", lambda: True)
    statuses = [
        {"enabled": True, "reachable": True, "admin_available": True, "config_matches": False, "warnings": ["mismatch"]},
        {"enabled": True, "reachable": True, "admin_available": True, "config_matches": True, "warnings": []},
    ]
    sync_calls: list[bool] = []

    def fake_model_bridge_status_payload(settings=None, env_entries=None):
        return statuses.pop(0) if statuses else {"enabled": True, "reachable": True, "admin_available": True, "config_matches": True, "warnings": []}

    def fake_sync_model_bridge_runtime_config(settings=None, env_entries=None, raise_on_error=True):
        sync_calls.append(raise_on_error)
        return {"attempted": True, "ok": True, "config_version": "bridge-version"}

    monkeypatch.setattr(runtime_settings, "model_bridge_status_payload", fake_model_bridge_status_payload)
    monkeypatch.setattr(runtime_settings, "sync_model_bridge_runtime_config", fake_sync_model_bridge_runtime_config)

    payload = runtime_settings.runtime_check_payload()

    assert payload["infrastructure"]["model_bridge"] is True
    assert payload["model_bridge_status"]["config_matches"] is True
    assert payload["model_bridge_status"]["last_sync"]["ok"] is True
    assert sync_calls == [False]
    assert payload["warnings"] == []


def test_update_model_settings_writes_isolated_chat_and_graph_keys(monkeypatch, tmp_path):
    from app.core import config
    from app.core.config import get_settings
    from app.services import runtime_settings

    env_path = tmp_path / ".env"
    example_path = tmp_path / ".env.example"
    env_path.write_text(
        "\n".join(
            [
                    "CHAT_API_KEY=",
                    "GRAPH_API_KEY=",
                    "CHAT_API_PROTOCOL=openai",
                    "GRAPH_API_PROTOCOL=openai",
                    "CHAT_BASE_URL=https://chat.example.test/v1",
                "GRAPH_BASE_URL=https://graph.example.test/v1",
                "CHAT_MODEL=chat-model",
                "GRAPH_MODEL=graph-model",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    example_path.write_text(env_path.read_text(encoding="utf-8"), encoding="utf-8")
    monkeypatch.setattr(runtime_settings, "ENV_PATH", env_path)
    monkeypatch.setattr(runtime_settings, "ENV_EXAMPLE_PATH", example_path)
    monkeypatch.setattr(config, "_read_workspace_env", lambda: runtime_settings._env_entries(env_path))
    monkeypatch.setattr(runtime_settings, "publish_runtime_settings_version", lambda changed_keys, source="api": {"changed_keys": changed_keys, "source": source})
    get_settings.cache_clear()

    result = runtime_settings.update_model_settings({"chat_api_key": "chat-secret", "graph_api_key": "graph-secret"})
    env_text = env_path.read_text(encoding="utf-8")

    assert "CHAT_API_KEY=chat-secret" in env_text
    assert "GRAPH_API_KEY=graph-secret" in env_text
    assert "OPENAI_API_KEY" not in env_text
    assert result["has_chat_api_key"] is True
    assert result["has_graph_api_key"] is True


def test_agent_run_status_includes_trace_for_frontend_recovery(db_session, sample_knowledge_base):
    from app.models import AgentRun, AgentTraceEvent
    from app.routers.search import agent_run_status

    run = AgentRun(
        knowledge_base_id=sample_knowledge_base.id,
        question="recover this run",
        status="completed",
        route="layered_context_graph",
        final_answer="Recovered answer",
    )
    db_session.add(run)
    db_session.flush()
    db_session.add(
        AgentTraceEvent(
            run_id=run.id,
            sequence_index=0,
            node="query_understanding",
            status="completed",
            output_summary="intent=question",
            scores={"ok": True},
        )
    )
    db_session.commit()

    payload = agent_run_status(run.id, db_session)

    assert payload["status"] == "completed"
    assert payload["answer"] == "Recovered answer"
    assert payload["trace"][0]["node"] == "query_understanding"


def test_retrieval_granularity_schema_defaults_and_rejects_invalid_values():
    from pydantic import ValidationError

    from app.schemas import AgentRequest, QARequest, SearchRequest

    assert SearchRequest(query="default").retrieval_granularity == "mid"
    assert QARequest(question="default").retrieval_granularity == "mid"
    assert AgentRequest(question="default").retrieval_granularity == "mid"
    assert SearchRequest(query="coarse", retrieval_granularity="coarse").retrieval_granularity == "coarse"
    for invalid in ("hybrid", "summary", "normal", "普通模式", ""):
        with pytest.raises(ValidationError):
            SearchRequest(query="invalid", retrieval_granularity=invalid)


def test_model_settings_update_rejects_legacy_generic_api_key():
    from pydantic import ValidationError

    from app.schemas import ModelSettingsUpdate

    assert ModelSettingsUpdate(clear_embedding_api_key=False).clear_embedding_api_key is False
    assert ModelSettingsUpdate(retrieval_result_top_k_default=8).retrieval_result_top_k_default == 8
    for invalid_depth in (1, 2, 3, 4):
        with pytest.raises(ValidationError):
            ModelSettingsUpdate(rq_kmeans_levels=invalid_depth)
    with pytest.raises(ValidationError):
        ModelSettingsUpdate(retrieval_result_top_k_default=0)
    with pytest.raises(ValidationError):
        ModelSettingsUpdate(retrieval_result_top_k_default=51)
    with pytest.raises(ValidationError):
        ModelSettingsUpdate(api_key="legacy-secret")


@pytest.mark.parametrize("submitted_depth", [1, 2, 3, 4])
def test_update_model_settings_rejects_fixed_rq_depth_before_any_side_effect(monkeypatch, submitted_depth):
    from app.services import cache_manager
    from app.services import runtime_settings

    normalize_calls: list[bool] = []
    env_write_calls: list[dict] = []
    settings_cache_clear_calls: list[bool] = []
    cache_manager_clear_calls: list[bool] = []
    broadcast_calls: list[list[str]] = []
    monkeypatch.setattr(runtime_settings, "normalize_env_file", lambda: normalize_calls.append(True))
    monkeypatch.setattr(runtime_settings, "_update_env_file", lambda updates: env_write_calls.append(dict(updates)))
    monkeypatch.setattr(
        runtime_settings.get_settings,
        "cache_clear",
        lambda: settings_cache_clear_calls.append(True),
    )
    monkeypatch.setattr(
        cache_manager,
        "clear_cache_manager",
        lambda: cache_manager_clear_calls.append(True),
    )
    monkeypatch.setattr(
        runtime_settings,
        "publish_runtime_settings_version",
        lambda changed_keys, source="api": broadcast_calls.append(list(changed_keys)),
    )

    with pytest.raises(ValueError, match="read-only fixed protocol"):
        runtime_settings.update_model_settings({"rq_kmeans_levels": submitted_depth})

    assert normalize_calls == []
    assert env_write_calls == []
    assert settings_cache_clear_calls == []
    assert cache_manager_clear_calls == []
    assert broadcast_calls == []


def test_model_settings_put_rejects_fixed_rq_depth_without_env_write_or_version_broadcast(monkeypatch, tmp_path):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from app.routers import settings as settings_router
    from app.services import cache_manager
    from app.services import runtime_settings

    env_path = tmp_path / ".env"
    example_path = tmp_path / ".env.example"
    initial_env = "CHAT_MODEL=unit-test-chat-model\nRQ_KMEANS_LEVELS=3\n"
    env_path.write_text(initial_env, encoding="utf-8")
    example_path.write_text(initial_env, encoding="utf-8")
    service_calls: list[dict] = []
    normalize_calls: list[bool] = []
    env_write_calls: list[dict] = []
    settings_cache_clear_calls: list[bool] = []
    cache_manager_clear_calls: list[bool] = []
    broadcasts: list[list[str]] = []
    monkeypatch.setattr(runtime_settings, "ENV_PATH", env_path)
    monkeypatch.setattr(runtime_settings, "ENV_EXAMPLE_PATH", example_path)
    monkeypatch.setattr(
        settings_router,
        "save_model_settings_to_root_env",
        lambda db, payload: service_calls.append(dict(payload)),
    )
    monkeypatch.setattr(
        runtime_settings,
        "normalize_env_file",
        lambda: normalize_calls.append(True),
    )
    monkeypatch.setattr(
        runtime_settings,
        "_update_env_file",
        lambda updates: env_write_calls.append(dict(updates)),
    )
    monkeypatch.setattr(
        runtime_settings.get_settings,
        "cache_clear",
        lambda: settings_cache_clear_calls.append(True),
    )
    monkeypatch.setattr(
        cache_manager,
        "clear_cache_manager",
        lambda: cache_manager_clear_calls.append(True),
    )
    monkeypatch.setattr(
        runtime_settings,
        "publish_runtime_settings_version",
        lambda changed_keys, source="api": broadcasts.append(list(changed_keys)),
    )

    app = FastAPI()
    app.include_router(settings_router.router, prefix="/api")
    with TestClient(app) as client:
        response = client.put("/api/settings/model", json={"rq_kmeans_levels": 3})

    assert response.status_code == 422
    assert any(item.get("type") == "extra_forbidden" for item in response.json()["detail"])
    assert service_calls == []
    assert normalize_calls == []
    assert env_write_calls == []
    assert settings_cache_clear_calls == []
    assert cache_manager_clear_calls == []
    assert env_path.read_text(encoding="utf-8") == initial_env
    assert broadcasts == []


def test_runtime_lifecycle_marks_rq_depth_as_fixed_protocol():
    from app.core.config import HOT_RELOAD_SETTINGS
    from app.services.runtime_settings import runtime_lifecycle_payload

    lifecycle = runtime_lifecycle_payload()

    assert lifecycle["fixed_protocol"]["rq_kmeans_levels"] == 3
    assert "rq_kmeans_levels" not in lifecycle["hot_reloadable"]
    assert "rq_kmeans_levels" not in lifecycle["rebuild_required"]
    assert "rq_kmeans_levels" not in lifecycle["service_recreate_required"]
    assert "rq_kmeans_levels" not in HOT_RELOAD_SETTINGS


def test_model_api_protocol_contract_is_closed_and_lifecycle_correct(
    monkeypatch,
):
    from pydantic import ValidationError

    from app.core.config import (
        HOT_RELOAD_SETTINGS,
        MODEL_API_PROTOCOL_ALLOWLIST,
        REBUILD_REQUIRED_SETTINGS,
        Settings,
        _validate_fixed_protocol_env,
    )
    from app.services.runtime_settings import runtime_lifecycle_payload

    assert MODEL_API_PROTOCOL_ALLOWLIST == frozenset({"openai", "anthropic"})
    monkeypatch.delenv("CHAT_API_PROTOCOL", raising=False)
    monkeypatch.delenv("GRAPH_API_PROTOCOL", raising=False)
    assert Settings(_env_file=None).chat_api_protocol == "openai"
    assert Settings(_env_file=None).graph_api_protocol == "openai"
    assert "chat_api_protocol" in HOT_RELOAD_SETTINGS
    assert "graph_api_protocol" in REBUILD_REQUIRED_SETTINGS
    lifecycle = runtime_lifecycle_payload()
    assert "chat_api_protocol" in lifecycle["hot_reloadable"]
    assert "graph_api_protocol" in lifecycle["rebuild_required"]

    for invalid in ("", "OpenAI", " anthropic", "unknown"):
        with pytest.raises(ValueError, match="allowlisted model API protocol"):
            _validate_fixed_protocol_env({"CHAT_API_PROTOCOL": invalid})
        with pytest.raises(ValidationError):
            Settings(chat_api_protocol=invalid, _env_file=None)


def test_fixed_rq_protocol_env_validation_is_independent_of_hot_reload_settings():
    from app.core.config import HOT_RELOAD_SETTINGS, _validate_fixed_protocol_env

    assert "rq_kmeans_levels" not in HOT_RELOAD_SETTINGS
    _validate_fixed_protocol_env({"RQ_KMEANS_LEVELS": "3"})
    for invalid_depth in ("1", "2", "4", "not-an-integer"):
        with pytest.raises(ValueError, match="fixed active protocol depth 3"):
            _validate_fixed_protocol_env({"RQ_KMEANS_LEVELS": invalid_depth})


def test_settings_parses_fixed_rq_protocol_from_string_environment(monkeypatch):
    from pydantic import ValidationError

    from app.core.config import get_settings

    monkeypatch.setenv("RQ_KMEANS_LEVELS", "3")
    get_settings.cache_clear()
    assert get_settings().rq_kmeans_levels == 3

    monkeypatch.setenv("RQ_KMEANS_LEVELS", "4")
    get_settings.cache_clear()
    with pytest.raises(ValidationError, match="fixed active protocol depth 3"):
        get_settings()
    get_settings.cache_clear()


def test_model_bridge_status_blocks_self_target(monkeypatch):
    from app.core.config import get_settings
    from app.services import runtime_settings

    get_settings.cache_clear()
    monkeypatch.setenv("MODEL_BRIDGE_ENABLED", "true")
    monkeypatch.setenv("MODEL_BRIDGE_PORT", "8765")
    monkeypatch.setattr(
        runtime_settings,
        "_env_entries",
        lambda _path: {
            "MODEL_BRIDGE_ENABLED": "true",
            "MODEL_BRIDGE_PORT": "8765",
            "CHAT_BASE_URL": "https://chat.example.test/v1",
            "EMBEDDING_BASE_URL": "http://host.docker.internal:8765",
        },
    )
    monkeypatch.setattr(
        runtime_settings.httpx,
        "get",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("offline")),
    )

    payload = runtime_settings.model_bridge_status_payload()

    assert payload["self_target_blocked"] is True
    assert payload["embedding_target_is_bridge"] is True
    assert payload["config_matches"] is not True
    assert any("bridge itself" in warning for warning in payload["warnings"])


def test_model_bridge_process_admin_token_overrides_stale_managed_value(
    monkeypatch,
):
    from app.core import config
    from app.core.config import get_settings

    process_token = "process-only-unit-token"
    monkeypatch.setenv("MODEL_BRIDGE_ADMIN_TOKEN", process_token)
    monkeypatch.setattr(
        config,
        "_read_workspace_env",
        lambda: {
            "MODEL_BRIDGE_ENABLED": "true",
            "MODEL_BRIDGE_PORT": "8765",
            "MODEL_BRIDGE_ADMIN_TOKEN": "local-model-bridge-admin",
        },
    )
    get_settings.cache_clear()

    assert get_settings().model_bridge_admin_token == process_token
    get_settings.cache_clear()


@pytest.mark.parametrize(
    "process_token",
    [
        "",
        " ",
        "\tunit-token",
        "unit-token\n",
        "unit-token\x01",
        "default",
        "local-model-bridge-admin",
        '\"\"',
        '\"   \"',
        '\"default\"',
        '\"unit-token\x01\"',
        "''",
        "'   '",
        "'default'",
        "'unit-token\x01'",
    ],
)
def test_model_bridge_process_admin_token_invalid_override_never_revives_managed_value(
    monkeypatch,
    process_token,
):
    from app.core import config
    from app.core.config import get_settings
    from app.services import runtime_settings

    monkeypatch.setenv("MODEL_BRIDGE_ADMIN_TOKEN", process_token)
    monkeypatch.setattr(
        config,
        "_read_workspace_env",
        lambda: {
            "MODEL_BRIDGE_ENABLED": "true",
            "MODEL_BRIDGE_PORT": "8765",
            "MODEL_BRIDGE_ADMIN_TOKEN": "managed-token-must-not-revive",
        },
    )
    get_settings.cache_clear()

    try:
        configured_token = get_settings().model_bridge_admin_token
        assert configured_token == process_token
        assert configured_token != "managed-token-must-not-revive"
        with pytest.raises(RuntimeError, match="MODEL_BRIDGE_ADMIN_TOKEN"):
            runtime_settings._validated_bridge_admin_token(configured_token)
    finally:
        get_settings.cache_clear()


def test_model_bridge_admin_token_absence_never_revives_managed_value(
    monkeypatch,
):
    from app.core import config
    from app.core.config import get_settings
    from app.services import runtime_settings

    monkeypatch.delenv("MODEL_BRIDGE_ADMIN_TOKEN", raising=False)
    monkeypatch.setattr(
        config,
        "_read_workspace_env",
        lambda: {
            "MODEL_BRIDGE_ENABLED": "true",
            "MODEL_BRIDGE_PORT": "8765",
            "MODEL_BRIDGE_ADMIN_TOKEN": "managed-token-must-not-revive",
        },
    )
    get_settings.cache_clear()

    try:
        configured_token = get_settings().model_bridge_admin_token
        assert configured_token == ""
        with pytest.raises(RuntimeError, match="MODEL_BRIDGE_ADMIN_TOKEN"):
            runtime_settings._validated_bridge_admin_token(configured_token)
    finally:
        get_settings.cache_clear()


def test_update_model_settings_rejects_bridge_self_target(monkeypatch, tmp_path):
    from app.core.config import get_settings
    from app.services import runtime_settings

    env_path = tmp_path / ".env"
    example_path = tmp_path / ".env.example"
    env_path.write_text(
        "\n".join(
            [
                "MODEL_BRIDGE_ENABLED=true",
                "MODEL_BRIDGE_PORT=8765",
                "MODEL_BRIDGE_ADMIN_TOKEN=unit-token",
                "CHAT_BASE_URL=https://chat.example.test/v1",
                "EMBEDDING_BASE_URL=https://embedding.example.test/v1",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    example_path.write_text(env_path.read_text(encoding="utf-8"), encoding="utf-8")
    monkeypatch.setattr(runtime_settings, "ENV_PATH", env_path)
    monkeypatch.setattr(runtime_settings, "ENV_EXAMPLE_PATH", example_path)
    monkeypatch.setenv("MODEL_BRIDGE_ENABLED", "true")
    monkeypatch.setenv("MODEL_BRIDGE_PORT", "8765")
    effects: list[str] = []
    for name in (
        "normalize_env_file",
        "_update_env_file",
        "_apply_runtime_env",
        "publish_runtime_settings_version",
        "reload_model_bridge",
    ):
        monkeypatch.setattr(
            runtime_settings,
            name,
            lambda *args, _name=name, **kwargs: effects.append(_name),
        )
    get_settings.cache_clear()

    try:
        runtime_settings.update_model_settings({"chat_base_url": "http://host.docker.internal:8765"})
        raise AssertionError("self-target bridge URL should be rejected")
    except ValueError as exc:
        assert "CHAT_BASE_URL" in str(exc)
    assert effects == []


@pytest.mark.parametrize(
    "unsafe_target",
    [
        "http://provider.example.test/v1",
        "https://localhost/v1",
        "https://10.0.0.1/v1",
        "https://provider.example.test/v1?route=unsafe",
    ],
)
def test_update_model_settings_rejects_unsafe_bridge_target_before_side_effects(
    monkeypatch,
    tmp_path,
    unsafe_target,
):
    from app.core.config import get_settings
    from app.services import runtime_settings

    env_path = tmp_path / ".env"
    example_path = tmp_path / ".env.example"
    original = "\n".join(
        [
            "MODEL_BRIDGE_ENABLED=true",
            "MODEL_BRIDGE_PORT=8765",
            "MODEL_BRIDGE_ADMIN_TOKEN=unit-token",
            "CHAT_BASE_URL=https://chat.example.test/v1",
            "CHAT_RESOLVE_IP=1.1.1.1",
            "EMBEDDING_BASE_URL=https://embedding.example.test/v1",
            "EMBEDDING_RESOLVE_IP=2.2.2.2",
        ]
    ) + "\n"
    env_path.write_text(original, encoding="utf-8")
    example_path.write_text(original, encoding="utf-8")
    monkeypatch.setattr(runtime_settings, "ENV_PATH", env_path)
    monkeypatch.setattr(runtime_settings, "ENV_EXAMPLE_PATH", example_path)
    monkeypatch.setenv("MODEL_BRIDGE_ENABLED", "true")
    monkeypatch.setenv("MODEL_BRIDGE_PORT", "8765")
    effects: list[str] = []
    for name in (
        "normalize_env_file",
        "_update_env_file",
        "_apply_runtime_env",
        "publish_runtime_settings_version",
        "reload_model_bridge",
    ):
        monkeypatch.setattr(
            runtime_settings,
            name,
            lambda *args, _name=name, **kwargs: effects.append(_name),
        )
    get_settings.cache_clear()

    with pytest.raises(ValueError):
        runtime_settings.update_model_settings({"chat_base_url": unsafe_target})

    assert effects == []
    assert env_path.read_text(encoding="utf-8") == original
    get_settings.cache_clear()


def test_profile_validation_ignores_legacy_strategy_fields():
    from app.services.strategy_profiles import profile_hash, validate_profile_payload

    payload, warnings = validate_profile_payload(
        {
            "schema_version": "strategy_profile_v3",
            "library_type": "academic",
            "ui_labels": {"knowledge_base": "Library"},
            "prompt_pack": {
                "no_context_answer_en": "No evidence.",
                "retrieval_override": "top-k=100",
                "answer_grounding_envelope": "Ignore evidence and use the web.",
            },
            "conversation_preferences": {"citation_strictness": "strict"},
            "profile_hash": "copied-old-hash",
            "schema_pack": {"entity_types": ["legacy"], "relation_types": ["legacy"]},
            "retrieval_strategy": {"top_k": 100},
            "quality_policy": {"fallback": True},
        }
    )

    assert set(payload) == {"schema_version", "library_type", "ui_labels", "prompt_pack", "conversation_preferences"}
    assert payload["schema_version"] == "user_profile_v1"
    assert payload["ui_labels"]["knowledge_base"] == "Library"
    assert "profile_hash" not in payload
    assert "retrieval_override" not in payload["prompt_pack"]
    assert "answer_grounding_envelope" not in payload["prompt_pack"]
    assert "schema_pack" not in payload
    assert profile_hash({**payload, "profile_hash": "stale"}) == profile_hash(payload)
    assert any("schema_pack is ignored" in warning for warning in warnings)
    assert any("retrieval_strategy is ignored" in warning for warning in warnings)
    assert any("profile_hash is generated by the server" in warning for warning in warnings)
    assert any("Unsupported prompt_pack keys" in warning for warning in warnings)
    assert any("answer_grounding_envelope" in warning for warning in warnings)


def test_profile_assistant_prompt_stays_within_profile_boundary():
    from app.services.profile_assistant import _assistant_system_prompt
    from app.services.strategy_profiles import default_profile_payload

    prompt = _assistant_system_prompt()

    assert "user-profile interaction-configuration assistant" in prompt
    assert "Profiles can tune knowledge-base system prompts" in prompt
    assert "Runtime Settings" in prompt
    assert "profile_json must not include profile_hash" in prompt
    assert "chunking" in prompt
    assert "embedding" in prompt
    assert "retrieval scoring" in prompt
    assert "agent envelope" in prompt
    assert "runtime controls" in prompt
    assert "strategy-profile" not in prompt

    profile = default_profile_payload()
    profile["prompt_pack"]["profile_assistant_system"] = "Profile-specific profile assistant prompt."
    assert _assistant_system_prompt(profile) == "Profile-specific profile assistant prompt."


def test_default_profile_prompt_pack_includes_system_prompt_registry():
    from app.services.strategy_profiles import default_profile_payload

    prompt_pack = default_profile_payload()["prompt_pack"]

    for key in [
        "answer_system_template",
        "query_rewrite_system",
        "json_response_fallback_system",
        "reflection_review_system",
        "question_perception_system",
        "query_facet_extractor_system",
        "agent_planner_system",
        "agent_evidence_evaluator_system",
        "citation_entailment_judge_system",
        "mid_concept_definition_system",
        "coarse_concept_definition_system",
        "concept_i18n_system",
        "concept_edge_i18n_system",
        "profile_assistant_system",
    ]:
        assert key in prompt_pack
        assert isinstance(prompt_pack[key], str)
        assert prompt_pack[key].strip()


def test_batch_summary_matches_response_schema(db_session, sample_knowledge_base):
    from app.models import IngestionBatch
    from app.schemas import IngestionBatchSummary
    from app.services.ingestion import summarize_batch

    batch = IngestionBatch(
        knowledge_base_id=sample_knowledge_base.id,
        trigger_source="upload",
        source_root="unit",
        status="parsing",
        total_files=2,
        processed_files=1,
        success_count=1,
        stats={
            "phase": "context_graph",
            "context_graph_phase": "mid_concepts",
            "current_file": "lesson.pdf",
            "ingestion_execution_mode": "celery",
        },
    )
    db_session.add(batch)
    db_session.commit()

    payload = summarize_batch(batch)
    validated = IngestionBatchSummary.model_validate(payload)

    assert validated.knowledge_base_id == sample_knowledge_base.id
    assert validated.current_phase == "context_graph:mid_concepts"
    assert validated.current_file == "lesson.pdf"
    assert validated.stats["context_graph_phase"] == "mid_concepts"


def test_celery_owned_batches_are_not_finalized_on_api_restart():
    from app.services.ingestion import batch_is_worker_owned

    assert batch_is_worker_owned({"ingestion_execution_mode": "celery"}) is True
    assert batch_is_worker_owned({"celery_task_id": "task-1"}) is True
    assert batch_is_worker_owned({"ingestion_execution_mode": "inline"}) is False


def test_cancel_running_celery_batch_revokes_worker_task(monkeypatch, db_session, sample_knowledge_base):
    from app.models import IngestionBatch, IngestionJob
    from app.services import ingestion

    revoke_calls: list[list[str]] = []

    def fake_revoke(task_ids: list[str]) -> dict:
        revoke_calls.append(task_ids)
        return {"attempted": True, "ok": True, "terminate": True, "signal": "SIGTERM", "task_ids": task_ids, "revoked_task_ids": task_ids, "errors": []}

    def fake_inspect(task_ids: list[str]) -> dict:
        return {"attempted": True, "ok": True, "task_ids": task_ids, "active_task_ids": ["task-1"], "reserved_task_ids": [], "scheduled_task_ids": []}

    monkeypatch.setattr(ingestion, "revoke_celery_batch_tasks", fake_revoke)
    monkeypatch.setattr(ingestion, "inspect_celery_batch_tasks", fake_inspect)
    batch = IngestionBatch(
        knowledge_base_id=sample_knowledge_base.id,
        trigger_source="upload",
        source_root="unit",
        status="extracting_graph",
        total_files=1,
        stats={"phase": "context_graph", "ingestion_execution_mode": "celery", "celery_task_id": "task-1", "batch_task_ids": ["task-1"]},
    )
    db_session.add(batch)
    db_session.flush()
    job = IngestionJob(knowledge_base_id=sample_knowledge_base.id, batch_id=batch.id, trigger_source="upload", status="processing")
    db_session.add(job)
    db_session.commit()

    payload = ingestion.request_batch_cancel_control(db_session, batch.id, sample_knowledge_base.id)
    db_session.refresh(batch)
    db_session.refresh(job)

    assert revoke_calls == [["task-1"]]
    assert payload["state"] == "cancelling"
    assert payload["cancel_requested"] is True
    assert batch.status == "cancelling"
    assert batch.stats["cancellation_status"] == "worker_terminate_requested"
    assert batch.stats["celery_revoke"]["terminate"] is True
    assert job.status == "cancel_requested"


def test_cancelling_batch_finalizes_after_worker_task_released(monkeypatch, db_session, sample_knowledge_base):
    from app.models import IngestionBatch, IngestionJob
    from app.services import ingestion

    monkeypatch.setattr(
        ingestion,
        "inspect_celery_batch_tasks",
        lambda task_ids: {"attempted": True, "ok": True, "task_ids": task_ids, "active_task_ids": [], "reserved_task_ids": [], "scheduled_task_ids": []},
    )
    batch = IngestionBatch(
        knowledge_base_id=sample_knowledge_base.id,
        trigger_source="upload",
        source_root="unit",
        status="cancelling",
        total_files=1,
        worker_id="worker-1",
        stats={"phase": "context_graph", "cancel_requested": True, "ingestion_execution_mode": "celery", "celery_task_id": "task-1", "batch_task_ids": ["task-1"]},
    )
    db_session.add(batch)
    db_session.flush()
    job = IngestionJob(knowledge_base_id=sample_knowledge_base.id, batch_id=batch.id, trigger_source="upload", status="processing")
    db_session.add(job)
    db_session.commit()

    payload = ingestion.get_batch_status(db_session, batch.id)
    db_session.refresh(batch)
    db_session.refresh(job)

    assert payload["state"] == "cancelled"
    assert payload["cancellation_status"] == "worker_released"
    assert batch.completed_at is not None
    assert batch.worker_id is None
    assert job.status == "cancelled"


def test_cancel_terminal_batch_does_not_revoke(monkeypatch, db_session, sample_knowledge_base):
    from app.models import IngestionBatch
    from app.services import ingestion

    revoke_calls: list[list[str]] = []
    monkeypatch.setattr(ingestion, "revoke_celery_batch_tasks", lambda task_ids: revoke_calls.append(task_ids))
    batch = IngestionBatch(
        knowledge_base_id=sample_knowledge_base.id,
        trigger_source="upload",
        source_root="unit",
        status="completed",
        total_files=1,
        stats={"ingestion_execution_mode": "celery", "celery_task_id": "task-1"},
    )
    db_session.add(batch)
    db_session.commit()

    payload = ingestion.request_batch_cancel_control(db_session, batch.id, sample_knowledge_base.id)

    assert payload["state"] == "completed"
    assert revoke_calls == []


def test_context_graph_heartbeat_updates_batch_stats(db_session, sample_knowledge_base):
    from app.models import IngestionBatch
    from app.services.context_graph import context_graph_batch_heartbeat
    from app.services.ingestion_logs import list_ingestion_logs

    batch = IngestionBatch(
        knowledge_base_id=sample_knowledge_base.id,
        trigger_source="upload",
        source_root="unit",
        status="extracting_graph",
        total_files=1,
    )
    db_session.add(batch)
    db_session.commit()

    context_graph_batch_heartbeat(batch.id, "mid_concepts", {"relation_edges": 3})
    db_session.refresh(batch)

    assert batch.heartbeat_at is not None
    assert batch.stats["phase"] == "context_graph"
    assert batch.stats["context_graph_phase"] == "mid_concepts"
    assert batch.stats["context_graph_metrics"]["relation_edges"] == 3
    logs = list_ingestion_logs(batch.id)
    assert logs[-1]["event"] == "batch_graph_progress"
    assert logs[-1]["message"] == "中粒度概念：进度更新"
    assert logs[-1]["phase"] == "context_graph:mid_concepts"
    assert logs[-1]["relation_edges"] == 3
    context_graph_batch_heartbeat(batch.id, "coarse_concepts", {"translation_phase": "edge_i18n", "translation_items": 12})
    db_session.refresh(batch)
    logs = list_ingestion_logs(batch.id)
    assert logs[-1]["message"] == "粗粒度概念：关系双语派生，12 项"
    assert logs[-1]["phase"] == "context_graph:coarse_concepts"
    assert logs[-1]["translation_phase"] == "edge_i18n"
    assert logs[-1]["translation_items"] == 12
    context_graph_batch_heartbeat(
        batch.id,
        "coarse_concepts",
        {"translation_phase": "concept_i18n", "translation_items": 4, "translation_enabled": False, "translation_status": "disabled"},
    )
    logs = list_ingestion_logs(batch.id)
    assert logs[-1]["message"] == "粗粒度概念：节点双语派生已关闭，跳过 4 项"
    assert logs[-1]["translation_enabled"] is False
    assert logs[-1]["translation_status"] == "disabled"


def test_cleanup_stale_data_deletes_inactive_chunk_versions_only_with_explicit_flag(db_session, sample_knowledge_base):
    from sqlalchemy import func, select

    from app.models import (
        Chunk,
        ChunkContextText,
        ChunkCoordinate,
        ChunkSpan,
        ChunkStructureMapping,
        ChunkStructureNode,
        ChunkVersion,
        Document,
        DocumentVersion,
    )
    from app.services.chunking import text_hash
    from app.services.maintenance import cleanup_stale_data

    sample_knowledge_base.current_chunk_version = 2
    document = Document(
        knowledge_base_id=sample_knowledge_base.id,
        title="Versioned document",
        source_path="versioned.md",
        source_type="markdown",
        checksum="v2",
        is_active=True,
    )
    db_session.add(document)
    db_session.flush()
    old_version = DocumentVersion(document_id=document.id, version=1, checksum="v1", storage_path="versioned.md", is_active=False)
    active_version = DocumentVersion(document_id=document.id, version=2, checksum="v2", storage_path="versioned.md", is_active=True)
    db_session.add_all([old_version, active_version])
    db_session.flush()
    old_chunk = Chunk(
        knowledge_base_id=sample_knowledge_base.id,
        document_id=document.id,
        document_version_id=old_version.id,
        chunk_version=1,
        chunk_index=0,
        token_start=0,
        token_end=4,
        char_start=0,
        char_end=12,
        text="old content",
        text_hash=text_hash("old content"),
        state="inactive",
    )
    active_chunk = Chunk(
        knowledge_base_id=sample_knowledge_base.id,
        document_id=document.id,
        document_version_id=active_version.id,
        chunk_version=2,
        chunk_index=0,
        token_start=0,
        token_end=4,
        char_start=0,
        char_end=12,
        text="new content",
        text_hash=text_hash("new content"),
        state="active",
    )
    db_session.add_all(
        [
            ChunkVersion(knowledge_base_id=sample_knowledge_base.id, chunk_version=1, state_hash="old", state="active"),
            ChunkVersion(knowledge_base_id=sample_knowledge_base.id, chunk_version=2, state_hash="new", state="active"),
            old_chunk,
            active_chunk,
        ]
    )
    db_session.flush()
    old_node = ChunkStructureNode(
        knowledge_base_id=sample_knowledge_base.id,
        document_id=document.id,
        document_version_id=old_version.id,
        node_type="section",
        title="Old",
    )
    db_session.add(old_node)
    db_session.flush()
    db_session.add_all(
        [
            ChunkSpan(chunk_id=old_chunk.id, document_version_id=old_version.id, char_start=0, char_end=12, token_start=0, token_end=4),
            ChunkCoordinate(chunk_id=old_chunk.id, document_version_id=old_version.id, page_number=1),
            ChunkContextText(chunk_id=old_chunk.id, raw_text="old content", contextual_text="old content", context_hash="old"),
            ChunkStructureMapping(chunk_id=old_chunk.id, structure_node_id=old_node.id, document_version_id=old_version.id),
        ]
    )
    old_chunk_id = old_chunk.id
    active_chunk_id = active_chunk.id
    old_version_id = old_version.id
    db_session.commit()

    dry_run = cleanup_stale_data(
        db_session,
        sample_knowledge_base.id,
        sample_knowledge_base.name,
        dry_run=True,
        delete_inactive_chunks=True,
    )
    assert dry_run["inactive_chunks"] == 1
    assert dry_run["inactive_document_versions"] == 1
    assert db_session.get(Chunk, old_chunk.id) is not None

    applied = cleanup_stale_data(
        db_session,
        sample_knowledge_base.id,
        sample_knowledge_base.name,
        dry_run=False,
        delete_inactive_chunks=True,
    )

    assert applied["deleted_inactive"]["chunks"] == 1
    assert applied["deleted_inactive"]["document_versions"] == 1
    assert applied["deleted_inactive"]["chunk_versions"] == 1
    assert db_session.scalar(select(func.count(Chunk.id)).where(Chunk.id == old_chunk_id)) == 0
    assert db_session.scalar(select(func.count(DocumentVersion.id)).where(DocumentVersion.id == old_version_id)) == 0
    assert db_session.scalar(select(func.count(Chunk.id)).where(Chunk.id == active_chunk_id)) == 1
    current_chunk_version = db_session.scalar(select(ChunkVersion).where(ChunkVersion.knowledge_base_id == sample_knowledge_base.id))
    assert current_chunk_version is not None
    assert current_chunk_version.chunk_version == 2
    assert current_chunk_version.state == "active"


def test_target_chunk_version_uses_active_max_after_interrupted_batch():
    from app.services.ingestion import target_chunk_version

    assert target_chunk_version(current_version=0, active_max_version=1, full_reparse=True) == 2
    assert target_chunk_version(current_version=0, active_max_version=1, full_reparse=False) == 1
    assert target_chunk_version(current_version=3, active_max_version=1, full_reparse=True) == 4


def test_collect_source_documents_deduplicates_same_named_copies(tmp_path):
    from app.services.ingestion import collect_source_documents

    original = tmp_path / "20260611" / "lesson.pdf"
    copied = tmp_path / "20260613" / "lesson.pdf"
    same_content_other_name = tmp_path / "20260613" / "lesson-copy.pdf"
    original.parent.mkdir()
    copied.parent.mkdir()
    pdf_bytes = b"%PDF-1.7\n1 0 obj\n<<>>\nendobj\n%%EOF\n"
    original.write_bytes(pdf_bytes)
    copied.write_bytes(pdf_bytes)
    same_content_other_name.write_bytes(pdf_bytes)

    paths = collect_source_documents(tmp_path)

    assert original in paths
    assert copied not in paths
    assert same_content_other_name in paths


def test_operations_script_matrix_matches_context_graph_todo():
    repo_root = Path(__file__).resolve().parents[3]
    scripts_root = repo_root / "scripts"
    required_scripts = {
        "check_runtime_settings_contract.py",
        "destroy_legacy_derived_data.py",
        "rebuild_chunks.py",
        "rebuild_structure_graph.py",
        "rebuild_chunk_relation_graph.py",
        "rebuild_rq_membership_graph.py",
        "rebuild_mid_concept_graph.py",
        "rebuild_coarse_concept_graph.py",
        "rebuild_context_graph_all.py",
        "reconcile_vector_records.py",
        "reconcile_ingestion_batch_recoveries.py",
        "reconcile_scoped_rebuild_cache_invalidations.py",
        "diagnose_context_graph.py",
        "evaluate_layered_retrieval.py",
        "evaluate_agent_trace.py",
        "check_context_package_quality.py",
        "runtime_hot_reload_probe.py",
    }

    missing = [name for name in sorted(required_scripts) if not (scripts_root / name).exists()]
    assert missing == []

    write_scripts = [
        "cleanup_stale_data.py",
        "destroy_legacy_derived_data.py",
        "docker_smoke.py",
        "evaluate_layered_retrieval.py",
        "rebuild_chunks.py",
        "rebuild_structure_graph.py",
        "rebuild_chunk_relation_graph.py",
        "rebuild_rq_membership_graph.py",
        "rebuild_mid_concept_graph.py",
        "rebuild_coarse_concept_graph.py",
        "rebuild_context_graph_all.py",
        "runtime_hot_reload_probe.py",
        "reconcile_ingestion_batch_recoveries.py",
        "reconcile_scoped_rebuild_cache_invalidations.py",
    ]
    for name in write_scripts:
        text = (scripts_root / name).read_text(encoding="utf-8")
        assert "--execute" in text

    destructive_text = (scripts_root / "destroy_legacy_derived_data.py").read_text(encoding="utf-8")
    assert "--confirm-destroy-legacy" in destructive_text
    assert "--confirm-knowledge-base-id" in destructive_text
    assert "--confirm-inventory-hash" in destructive_text

    stale_cleanup_text = (scripts_root / "cleanup_stale_data.py").read_text(
        encoding="utf-8"
    )
    assert "--confirm-knowledge-base-id" in stale_cleanup_text
    assert "--confirm-inventory-hash" in stale_cleanup_text

    recovery_text = (scripts_root / "reconcile_ingestion_batch_recoveries.py").read_text(
        encoding="utf-8"
    )
    assert "--all-active-batches" in recovery_text
    assert "finalize_interrupted_batches" in recovery_text
