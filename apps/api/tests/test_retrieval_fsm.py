import pytest

from app.services.retrieval_fsm import RetrievalControlState, next_control_state


def initial():
    return RetrievalControlState(runtime_hash="a" * 64, repair_limit=1)


def test_source_location_is_once_before_the_first_retrieval():
    state=next_control_state(initial(),'task_ready',task_hash='b'*64,strategy_hash='c'*64)
    assert 'scope_resolution_started' not in state.model_dump(mode='json')
    state=next_control_state(state,'scope_resolving')
    state=next_control_state(state,'task_ready')
    assert state.scope_resolution_started and state.repairs_used==0
    with pytest.raises(ValueError,match='call_budget_exhausted'):
        next_control_state(state,'scope_resolving')
    state=next_control_state(state,'searching')
    with pytest.raises(ValueError,match='not_allowed'):
        next_control_state(state,'scope_resolving')


def ready():
    value = next_control_state(initial(), "task_ready", task_hash="b" * 64, strategy_hash="c" * 64)
    for phase in ("searching", "packing", "diagnosing", "ready"):
        value = next_control_state(value, phase)
    return value


def test_generation_can_only_bind_or_terminate_never_return_to_retrieval():
    value = next_control_state(ready(), "generating")
    assert value.generation_started
    for forbidden in ("searching", "patching", "diagnosing", "generating"):
        with pytest.raises(ValueError, match="not_allowed"):
            next_control_state(value, forbidden)
    completed = next_control_state(next_control_state(value, "binding"), "completed")
    with pytest.raises(ValueError, match="not_allowed"):
        next_control_state(completed, "searching")


def test_repair_budget_and_strategy_execution_survive_phase_transitions():
    value = next_control_state(initial(), "task_ready", task_hash="b" * 64, strategy_hash="c" * 64)
    for phase in ("searching", "packing", "diagnosing", "discovering", "patching"):
        value = next_control_state(value, phase)
    with pytest.raises(ValueError, match="already_executed"):
        next_control_state(value, "searching")
    value = next_control_state(value, "searching", strategy_hash="d" * 64)
    assert value.repairs_used == 1 and len(value.executed_strategy_hashes) == 2
    for phase in ("packing", "diagnosing", "discovering", "patching"):
        value = next_control_state(value, phase)
    with pytest.raises(ValueError, match="repair_budget"):
        next_control_state(value, "searching", strategy_hash="e" * 64)


def test_task_identity_cannot_change_and_capability_is_explicit():
    value = next_control_state(initial(), "task_ready", task_hash="b" * 64, strategy_hash="c" * 64)
    with pytest.raises(ValueError, match="task_identity_changed"):
        next_control_state(value, "searching", task_hash="d" * 64)
    with pytest.raises(ValueError):
        next_control_state(initial(), "completed")
    assert next_control_state(initial(), "completed", system_capability=True).state == "completed"
