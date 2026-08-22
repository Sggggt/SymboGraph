from __future__ import annotations

from sqlalchemy import func, select

import pytest


def _force_traversal_only_protocol_drift(monkeypatch) -> None:
    from app.services import context_graph

    monkeypatch.setattr(
        context_graph,
        "_traversal_protocol_hash_from_effective_envelope",
        lambda _envelope, **_kwargs: "b" * 64,
    )


def test_context_protocol_identity_refresh_plan_is_read_only_and_bounded(
    monkeypatch,
    db_session,
    populated_context_graph,
) -> None:
    from app.models import ContextGraphState, IngestionCompensationLog
    from app.services import scoped_context_graph_rebuild

    knowledge_base = populated_context_graph["knowledge_base"]
    state = populated_context_graph["state"]
    _force_traversal_only_protocol_drift(monkeypatch)
    before = {
        "contexts": db_session.scalar(
            select(func.count(ContextGraphState.id)).where(
                ContextGraphState.knowledge_base_id == knowledge_base.id
            )
        ),
        "intents": db_session.scalar(
            select(func.count(IngestionCompensationLog.id))
        ),
    }

    plan = (
        scoped_context_graph_rebuild.plan_context_protocol_identity_refresh(
            db_session,
            knowledge_base.id,
        )
    )

    assert plan["protocol_version"] == "context_protocol_identity_refresh_v1"
    assert plan["eligible"] is True
    assert plan["reason"] == "traversal_protocol_wrapper_refresh_required"
    assert plan["active_context_graph_state_id"] == state.id
    assert plan["admission_reasons"] == [
        "context_graph_canonical_protocol_identities_stale",
        "context_graph_canonical_traversal_envelope_mismatch",
    ]
    assert plan["identity_difference_keys"] == ["traversal_protocol_hash"]
    assert plan["model_call_count"] == 0
    assert plan["qdrant_write_count"] == 0
    assert plan["gray_zone_rule_inputs_modified"] is False
    assert plan["gray_zone_rule_protocol_modified"] is False
    after = {
        "contexts": db_session.scalar(
            select(func.count(ContextGraphState.id)).where(
                ContextGraphState.knowledge_base_id == knowledge_base.id
            )
        ),
        "intents": db_session.scalar(
            select(func.count(IngestionCompensationLog.id))
        ),
    }
    assert after == before


def test_context_protocol_identity_refresh_refuses_non_wrapper_staleness(
    monkeypatch,
    db_session,
    populated_context_graph,
) -> None:
    from app.models import ContextGraphFreshness
    from app.services import scoped_context_graph_rebuild

    knowledge_base = populated_context_graph["knowledge_base"]
    state = populated_context_graph["state"]
    _force_traversal_only_protocol_drift(monkeypatch)
    row = db_session.scalar(
        select(ContextGraphFreshness).where(
            ContextGraphFreshness.context_graph_state_id == state.id,
            ContextGraphFreshness.layer == "chunk_structure",
        )
    )
    assert row is not None
    row.state_hash = "f" * 64
    db_session.flush()

    plan = (
        scoped_context_graph_rebuild.plan_context_protocol_identity_refresh(
            db_session,
            knowledge_base.id,
        )
    )

    assert plan["eligible"] is False
    assert plan["reason"] == "active_context_graph_has_non_wrapper_staleness"
    assert any(
        reason.startswith("context_graph_freshness_")
        for reason in plan["admission_reasons"]
    )


@pytest.mark.asyncio
async def test_context_protocol_identity_refresh_reuses_all_graph_facts(
    monkeypatch,
    db_session,
    populated_context_graph,
) -> None:
    from app.models import (
        ChunkRelationGraphState,
        CoarseConceptState,
        ContextGraphState,
        IngestionCompensationLog,
        KnowledgeBaseVectorRuntimeState,
        MidConceptState,
    )
    from app.services import context_graph, scoped_context_graph_rebuild

    knowledge_base = populated_context_graph["knowledge_base"]
    previous = populated_context_graph["state"]
    previous_ids = {
        "relation": str(previous.chunk_relation_graph_state_id),
        "mid": str(previous.mid_concept_state_id),
        "coarse": str(previous.coarse_concept_state_id),
        "context": str(previous.id),
    }
    _force_traversal_only_protocol_drift(monkeypatch)
    monkeypatch.setattr(
        context_graph,
        "ChatProvider",
        lambda *_args, **_kwargs: pytest.fail("refresh must not construct chat"),
    )
    monkeypatch.setattr(
        context_graph,
        "EmbeddingProvider",
        lambda *_args, **_kwargs: pytest.fail(
            "refresh must not construct embedding"
        ),
    )
    invalidations: list[tuple[str, bool]] = []
    monkeypatch.setattr(
        context_graph,
        "invalidate_context_graph_cache_after_commit",
        lambda knowledge_base_id, *, strict=False: (
            invalidations.append((knowledge_base_id, strict)) or True
        ),
    )
    counts_before = {
        "relation": db_session.scalar(
            select(func.count(ChunkRelationGraphState.id))
        ),
        "mid": db_session.scalar(select(func.count(MidConceptState.id))),
        "coarse": db_session.scalar(
            select(func.count(CoarseConceptState.id))
        ),
    }

    refreshed = (
        await scoped_context_graph_rebuild.refresh_context_protocol_identity(
            db_session,
            knowledge_base.id,
        )
    )

    assert str(refreshed.id) != previous_ids["context"]
    assert str(refreshed.chunk_relation_graph_state_id) == previous_ids["relation"]
    assert str(refreshed.mid_concept_state_id) == previous_ids["mid"]
    assert str(refreshed.coarse_concept_state_id) == previous_ids["coarse"]
    assert db_session.get(ContextGraphState, previous.id).state == "inactive"
    assert db_session.get(ChunkRelationGraphState, previous_ids["relation"]).state == "active"
    assert db_session.get(MidConceptState, previous_ids["mid"]).state == "active"
    assert db_session.get(CoarseConceptState, previous_ids["coarse"]).state == "active"
    counts_after = {
        "relation": db_session.scalar(
            select(func.count(ChunkRelationGraphState.id))
        ),
        "mid": db_session.scalar(select(func.count(MidConceptState.id))),
        "coarse": db_session.scalar(
            select(func.count(CoarseConceptState.id))
        ),
    }
    assert counts_after == counts_before
    audit = (refreshed.diagnostics_json or {})[
        "context_protocol_identity_refresh_audit"
    ]
    assert audit["identity_difference_keys"] == ["traversal_protocol_hash"]
    assert audit["model_call_count"] == 0
    assert audit["qdrant_write_count"] == 0
    assert audit["provider_response_persisted"] is False
    assert audit["gray_zone_model_call_count"] == 0
    assert invalidations == []
    pointer = db_session.scalar(
        select(KnowledgeBaseVectorRuntimeState).where(
            KnowledgeBaseVectorRuntimeState.knowledge_base_id
            == knowledge_base.id
        )
    )
    assert pointer is not None
    assert str(pointer.active_context_graph_state_id) == str(refreshed.id)
    assert str(pointer.active_chunk_relation_graph_state_id) == previous_ids[
        "relation"
    ]
    intent = db_session.get(
        IngestionCompensationLog,
        audit["cache_invalidation_intent_id"],
    )
    assert intent is not None
    assert intent.status == "cache_invalidation_pending"
    assert context_graph.active_graph_admission_gate(
        db_session,
        knowledge_base.id,
    ).id == refreshed.id

    db_session.commit()
    assert invalidations == [(knowledge_base.id, True)]
    db_session.expire_all()
    assert db_session.get(
        IngestionCompensationLog,
        audit["cache_invalidation_intent_id"],
    ).status == "completed"
