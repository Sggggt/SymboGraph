import pytest

from app.models import AgentObservation, AgentRun
from app.services.agent_reflection import ReflectionContractError
from app.services.reflection_run import load_reflection_run_ledger


def test_run_budget_does_not_reset_at_a_durable_route_handoff(db_session, sample_knowledge_base):
    run = AgentRun(knowledge_base_id=sample_knowledge_base.id, question="A synthetic question", status="running")
    db_session.add(run)
    db_session.flush()
    ledger, events = load_reflection_run_ledger(db_session, run=run, requested_limit=2, runtime_hash="a" * 64)
    assert ledger["hard_limit"] == 2 and events == []
    event = {"protocol_version": "agent_answer_reflection_v1", "sequence_index": 0,
             "stage": "retrieval_handoff", "status": "completed", "round_budget": 2}
    db_session.add(AgentObservation(run_id=run.id, observation_type="answer_reflection", observation_json=event))
    db_session.commit()
    restored, previous = load_reflection_run_ledger(db_session, run=run, requested_limit=9, runtime_hash="a" * 64)
    assert restored == ledger and previous == [event]
    previous[0]["round_budget"] = 9
    assert db_session.query(AgentObservation).one().observation_json["round_budget"] == 2


@pytest.mark.parametrize("attack", ["runtime", "question", "sequence", "not_handoff"])
def test_run_ledger_rejects_changed_identity_or_an_unfinished_phase(db_session, sample_knowledge_base, attack):
    run = AgentRun(knowledge_base_id=sample_knowledge_base.id, question="A synthetic question", status="running")
    db_session.add(run)
    db_session.flush()
    load_reflection_run_ledger(db_session, run=run, requested_limit=2, runtime_hash="a" * 64)
    event = {"protocol_version": "agent_answer_reflection_v1", "sequence_index": 0,
             "stage": "retrieval_handoff", "status": "completed", "round_budget": 2}
    if attack == "sequence":
        event["sequence_index"] = 3
    if attack == "not_handoff":
        event["stage"] = "answer_generation"
    db_session.add(AgentObservation(run_id=run.id, observation_type="answer_reflection", observation_json=event))
    if attack == "question":
        run.question = "A different question"
    db_session.flush()
    with pytest.raises(ReflectionContractError, match="identity_changed|not_at_handoff|sequence_invalid"):
        load_reflection_run_ledger(db_session, run=run, requested_limit=2, runtime_hash=("b" if attack == "runtime" else "a") * 64)


@pytest.mark.parametrize("timestamps_equal", [False, True])
def test_handoff_replay_uses_sequence_despite_clock_rollback(db_session, sample_knowledge_base, timestamps_equal):
    from datetime import datetime, timedelta
    run = AgentRun(knowledge_base_id=sample_knowledge_base.id, question="Synthetic clock rollback", status="running")
    db_session.add(run)
    db_session.flush()
    load_reflection_run_ledger(db_session, run=run, requested_limit=2, runtime_hash="a" * 64)
    events = [{"protocol_version": "agent_answer_reflection_v1", "sequence_index": index, "stage": stage,
               "status": "completed", "round_budget": 2} for index, stage in enumerate(("answer_generation", "reflection_gate", "retrieval_handoff"))]
    for event in events:
        db_session.add(AgentObservation(run_id=run.id, observation_type="answer_reflection", observation_json=event,
            created_at=datetime(2026, 1, 1) - timedelta(seconds=0 if timestamps_equal else event["sequence_index"])))
    db_session.commit()
    assert load_reflection_run_ledger(db_session, run=run, requested_limit=2, runtime_hash="a" * 64)[1] == events


@pytest.mark.parametrize("indexes", [[0, 0], [0, 2], [-1], [True], [0.0], ["0"], list(range(129))])
def test_ordered_observations_reject_noncanonical_or_unbounded_sequences(db_session, sample_knowledge_base, indexes):
    from app.services.reflection_run import ordered_reflection_events
    run = AgentRun(knowledge_base_id=sample_knowledge_base.id, question="Synthetic sequence guard", status="running")
    db_session.add(run)
    db_session.flush()
    for index in indexes:
        db_session.add(AgentObservation(run_id=run.id, observation_type="answer_reflection", observation_json={
            "protocol_version": "agent_answer_reflection_v1", "sequence_index": index}))
    db_session.flush()
    with pytest.raises(ReflectionContractError, match="sequence_invalid"):
        ordered_reflection_events(db_session, run_id=run.id)
