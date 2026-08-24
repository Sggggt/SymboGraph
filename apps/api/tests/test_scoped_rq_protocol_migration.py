from types import SimpleNamespace

import pytest


def _lineage(knowledge_base_id: str):
    relation = SimpleNamespace(
        id="relation-state",
        state="active",
        knowledge_base_id=knowledge_base_id,
    )
    mid = SimpleNamespace(
        id="mid-state",
        state="active",
        knowledge_base_id=knowledge_base_id,
    )
    coarse = SimpleNamespace(
        id="coarse-state",
        state="active",
        knowledge_base_id=knowledge_base_id,
    )
    context = SimpleNamespace(
        id="context-state",
        state="active",
        knowledge_base_id=knowledge_base_id,
        chunk_relation_graph_state_id=relation.id,
        mid_concept_state_id=mid.id,
        coarse_concept_state_id=coarse.id,
    )
    return context, relation, mid, coarse


class _FakeSession:
    def __init__(self, rows):
        self._rows = {row.id: row for row in rows}

    def get(self, _model, row_id):
        return self._rows.get(row_id)


def test_rq_protocol_migration_accepts_only_replaced_layer_staleness(
    monkeypatch,
):
    from app.services import context_graph, scoped_context_graph_rebuild

    knowledge_base_id = "kb-test"
    context, relation, mid, coarse = _lineage(knowledge_base_id)
    allowed_reasons = (
        scoped_context_graph_rebuild
        .RQ_PROTOCOL_MIGRATION_ALLOWED_STALE_REASONS
    )
    reasons = sorted(allowed_reasons)

    def reject_old_protocol(*_args, **_kwargs):
        raise context_graph.ActiveContextGraphAdmissionError(
            "old RQ protocol is stale",
            reasons=reasons,
            context_graph_state_id=context.id,
        )

    monkeypatch.setattr(
        scoped_context_graph_rebuild.context_graph,
        "active_graph_admission_gate",
        reject_old_protocol,
    )
    monkeypatch.setattr(
        scoped_context_graph_rebuild.context_graph,
        "latest_context_graph_state",
        lambda *_args, **_kwargs: context,
    )

    bound = scoped_context_graph_rebuild._bound_active_lineage(
        _FakeSession([relation, mid, coarse]),
        knowledge_base_id,
        chunks=[SimpleNamespace(id="chunk")],
        requested_scope="rq_membership",
    )

    assert bound[:4] == (context, relation, mid, coarse)
    assert bound[4]["accepted"] is True
    assert bound[4]["model_call_count"] == 0
    assert bound[4]["unrelated_stale_reasons"] == []


def test_rq_protocol_migration_rejects_unrelated_staleness(monkeypatch):
    from app.services import context_graph, scoped_context_graph_rebuild

    context, relation, mid, coarse = _lineage("kb-test")

    def reject_bad_scope(*_args, **_kwargs):
        raise context_graph.ActiveContextGraphAdmissionError(
            "chunk scope is stale",
            reasons=[
                "rq_membership_protocol_hash_mismatch",
                "active_chunk_scope_hash_mismatch",
            ],
            context_graph_state_id=context.id,
        )

    monkeypatch.setattr(
        scoped_context_graph_rebuild.context_graph,
        "active_graph_admission_gate",
        reject_bad_scope,
    )

    with pytest.raises(context_graph.ActiveContextGraphAdmissionError):
        scoped_context_graph_rebuild._bound_active_lineage(
            _FakeSession([relation, mid, coarse]),
            "kb-test",
            chunks=[SimpleNamespace(id="chunk")],
            requested_scope="rq_membership",
        )


def test_non_rq_scoped_rebuild_cannot_bypass_stale_admission(monkeypatch):
    from app.services import context_graph, scoped_context_graph_rebuild

    context, relation, mid, coarse = _lineage("kb-test")

    def reject_old_protocol(*_args, **_kwargs):
        raise context_graph.ActiveContextGraphAdmissionError(
            "old RQ protocol is stale",
            reasons=["rq_membership_protocol_hash_mismatch"],
            context_graph_state_id=context.id,
        )

    monkeypatch.setattr(
        scoped_context_graph_rebuild.context_graph,
        "active_graph_admission_gate",
        reject_old_protocol,
    )

    with pytest.raises(context_graph.ActiveContextGraphAdmissionError):
        scoped_context_graph_rebuild._bound_active_lineage(
            _FakeSession([relation, mid, coarse]),
            "kb-test",
            chunks=[SimpleNamespace(id="chunk")],
            requested_scope="mid_concept",
        )
