from __future__ import annotations

from pathlib import Path
import sys
import types

import pytest


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
    for layer in ("chunk-structure", "chunk-relation", "mid-concepts", "coarse-concepts"):
        payload = graph_layer_payload(db_session, kb.id, layer, limit=20)
        GraphResponse.model_validate(payload)
        assert payload["knowledge_base_id"] == kb.id
        assert payload["graph_type"] == layer
        assert payload["counts"]
        assert "freshness" in payload
        assert "grounding" in payload


@pytest.mark.asyncio
async def test_knowledge_base_summary_includes_active_graph_state(db_session, populated_context_graph):
    from app.schemas import KnowledgeBaseSummary
    from app.services.ingestion import summarize_knowledge_base

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


def test_uploaded_staging_job_is_listed_as_pending_until_batch_starts(db_session, sample_knowledge_base):
    from app.core.config import get_settings
    from app.services.ingestion import create_job, create_uploaded_files_batch, list_knowledge_base_files, register_uploaded_file

    storage_root = get_settings().knowledge_base_paths_for_name(sample_knowledge_base.name)["storage_root"]
    storage_root.mkdir(parents=True, exist_ok=True)
    source_path = storage_root / "complex-network.md"
    source_path.write_text("# Complex network\n\nA source waiting for parse.\n", encoding="utf-8")

    document, upload_job = register_uploaded_file(db_session, sample_knowledge_base, source_path)
    assert upload_job.status == "queued"
    assert upload_job.batch_id is None

    files = list_knowledge_base_files(db_session, sample_knowledge_base.id)
    uploaded_item = next(item for item in files if item["source_path"] == str(source_path))
    assert uploaded_item["document_id"] == document.id
    assert uploaded_item["status"] == "pending"
    assert uploaded_item["batch_id"] is None

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

    assert db_session.get(RetrievalTrace, trace_id) is not None
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
    monkeypatch.setattr(runtime_settings, "_check_model_bridge", lambda: None)

    payload = runtime_settings.runtime_check_payload()
    validated = RuntimeCheckResponse.model_validate(payload)

    assert validated.env_sync["synced"] is True
    assert validated.infrastructure["postgres"] is True
    assert validated.blocking_issues == []


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


def test_model_settings_payload_uses_fixed_chunk_and_context_budget(monkeypatch):
    from app.core.config import get_settings, model_bridge_client_base_url
    from app.services import runtime_settings

    get_settings.cache_clear()
    monkeypatch.setenv("FIXED_CHUNK_SIZE_TOKENS", "512")
    monkeypatch.setenv("FIXED_CHUNK_OVERLAP_TOKENS", "80")
    monkeypatch.setenv("CONTEXT_PACKAGE_TOKEN_BUDGET", "2400")
    monkeypatch.setattr(runtime_settings, "current_runtime_settings_version", lambda: "unit-version")
    monkeypatch.setattr(
        runtime_settings,
        "_env_entries",
        lambda _path: {
            "FIXED_CHUNK_SIZE_TOKENS": "512",
            "FIXED_CHUNK_OVERLAP_TOKENS": "80",
            "CONTEXT_PACKAGE_TOKEN_BUDGET": "2400",
        },
    )

    payload = runtime_settings.model_settings_payload()

    assert payload["fixed_chunk_size_tokens"] == 512
    assert payload["fixed_chunk_overlap_tokens"] == 80
    assert payload["context_package_token_budget"] == 2400
    assert payload["concept_i18n_enabled"] is False
    assert payload["query_facet_bilingual_enabled"] is False
    assert payload["agent_coarse_total_budget"] > 0
    assert payload["agent_chunk_top_k"] > 0
    assert "lifecycle" in payload
    assert "chunk_token_budget" not in payload
    assert "semantic_chunking_enabled" not in payload
    assert "agent_coarse_entry_budget" not in payload


def test_model_settings_payload_keeps_bridge_targets_editable(monkeypatch):
    from app.core.config import get_settings, model_bridge_client_base_url
    from app.services import runtime_settings

    get_settings.cache_clear()
    monkeypatch.setenv("MODEL_BRIDGE_ENABLED", "true")
    monkeypatch.setenv("MODEL_BRIDGE_PORT", "8765")
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
            "CHAT_BASE_URL": "https://chat.example.test/v1",
            "EMBEDDING_BASE_URL": "https://embedding.example.test/v1",
            "CHAT_RESOLVE_IP": "1.1.1.1",
            "EMBEDDING_RESOLVE_IP": "2.2.2.2",
        },
    )
    monkeypatch.setattr(runtime_settings, "current_runtime_settings_version", lambda: "unit-version")
    monkeypatch.setattr(runtime_settings, "model_bridge_status_payload", lambda settings=None, env_entries=None: {"enabled": True, "config_matches": True})

    payload = runtime_settings.model_settings_payload()

    assert payload["chat_base_url"] == "https://chat.example.test/v1"
    assert payload["embedding_base_url"] == "https://embedding.example.test/v1"
    assert payload["effective_chat_base_url"] == model_bridge_client_base_url(8765)
    assert payload["effective_embedding_base_url"] == model_bridge_client_base_url(8765)
    assert payload["chat_resolve_ip"] == "1.1.1.1"
    assert payload["embedding_resolve_ip"] == "2.2.2.2"


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
    monkeypatch.setattr(runtime_settings, "publish_runtime_settings_version", lambda changed_keys, source="api": {"changed_keys": changed_keys, "source": source})
    monkeypatch.setattr(runtime_settings, "model_bridge_status_payload", lambda settings=None, env_entries=None: {"enabled": True, "config_matches": True})
    reload_calls: list[dict] = []

    def fake_reload_model_bridge(settings=None, env_entries=None):
        reload_calls.append(dict(env_entries or {}))
        return {"attempted": True, "ok": True, "config_version": "bridge-version"}

    monkeypatch.setattr(runtime_settings, "reload_model_bridge", fake_reload_model_bridge)
    get_settings.cache_clear()

    result = runtime_settings.update_model_settings(
        {
            "embedding_base_url": "https://embedding.example.test/v2",
            "concept_i18n_enabled": True,
            "query_facet_bilingual_enabled": True,
        }
    )

    assert reload_calls
    assert reload_calls[-1]["EMBEDDING_BASE_URL"] == "https://embedding.example.test/v2"
    assert reload_calls[-1]["CONCEPT_I18N_ENABLED"] == "true"
    assert reload_calls[-1]["QUERY_FACET_BILINGUAL_ENABLED"] == "true"
    assert result["concept_i18n_enabled"] is True
    assert result["query_facet_bilingual_enabled"] is True
    assert result["model_bridge_status"]["last_reload"]["ok"] is True


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


def test_model_settings_update_rejects_legacy_generic_api_key():
    from pydantic import ValidationError

    from app.schemas import ModelSettingsUpdate

    assert ModelSettingsUpdate(clear_embedding_api_key=False).clear_embedding_api_key is False
    with pytest.raises(ValidationError):
        ModelSettingsUpdate(api_key="legacy-secret")


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
    get_settings.cache_clear()

    try:
        runtime_settings.update_model_settings({"embedding_base_url": "http://host.docker.internal:8765"})
        raise AssertionError("self-target bridge URL should be rejected")
    except ValueError as exc:
        assert "EMBEDDING_BASE_URL" in str(exc)


def test_profile_validation_ignores_legacy_strategy_fields():
    from app.services.strategy_profiles import validate_profile_payload

    payload, warnings = validate_profile_payload(
        {
            "schema_version": "strategy_profile_v3",
            "library_type": "academic",
            "ui_labels": {"knowledge_base": "Library"},
            "prompt_pack": {"no_context_answer_en": "No evidence."},
            "conversation_preferences": {"citation_strictness": "strict"},
            "schema_pack": {"entity_types": ["legacy"], "relation_types": ["legacy"]},
            "retrieval_strategy": {"top_k": 100},
            "quality_policy": {"fallback": True},
        }
    )

    assert set(payload) == {"schema_version", "library_type", "ui_labels", "prompt_pack", "conversation_preferences", "profile_hash"}
    assert payload["schema_version"] == "user_profile_v1"
    assert payload["ui_labels"]["knowledge_base"] == "Library"
    assert "schema_pack" not in payload
    assert any("schema_pack is ignored" in warning for warning in warnings)
    assert any("retrieval_strategy is ignored" in warning for warning in warnings)


def test_profile_assistant_prompt_stays_within_profile_boundary():
    from app.services.profile_assistant import _assistant_system_prompt

    prompt = _assistant_system_prompt()

    assert "user-profile interaction-configuration assistant" in prompt
    assert "Profiles only affect interaction wording" in prompt
    assert "Runtime Settings" in prompt
    assert "chunking" in prompt
    assert "embedding" in prompt
    assert "retrieval scoring" in prompt
    assert "agent envelope" in prompt
    assert "runtime controls" in prompt
    assert "strategy-profile" not in prompt


def test_default_profile_prompt_pack_excludes_engineering_prompts():
    from app.services.strategy_profiles import default_profile_payload

    prompt_pack = default_profile_payload()["prompt_pack"]

    assert "community_summary_system" not in prompt_pack
    assert "quality_judge_system" not in prompt_pack
    assert "query_translation_domain" not in prompt_pack
    assert "retry_query_suffix" not in prompt_pack


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
        text_hash="old",
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
        text_hash="new",
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
    original.write_bytes(b"same pdf bytes")
    copied.write_bytes(b"same pdf bytes")
    same_content_other_name.write_bytes(b"same pdf bytes")

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
        "diagnose_context_graph.py",
        "evaluate_layered_retrieval.py",
        "evaluate_agent_trace.py",
        "check_context_package_quality.py",
        "runtime_hot_reload_probe.py",
    }

    missing = [name for name in sorted(required_scripts) if not (scripts_root / name).exists()]
    assert missing == []

    write_scripts = [
        "destroy_legacy_derived_data.py",
        "rebuild_chunks.py",
        "rebuild_structure_graph.py",
        "rebuild_chunk_relation_graph.py",
        "rebuild_rq_membership_graph.py",
        "rebuild_mid_concept_graph.py",
        "rebuild_coarse_concept_graph.py",
        "rebuild_context_graph_all.py",
        "runtime_hot_reload_probe.py",
    ]
    for name in write_scripts:
        text = (scripts_root / name).read_text(encoding="utf-8")
        assert "--execute" in text

    destructive_text = (scripts_root / "destroy_legacy_derived_data.py").read_text(encoding="utf-8")
    assert "--confirm-destroy-legacy" in destructive_text
