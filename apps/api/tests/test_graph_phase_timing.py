import pytest
from app.services.qa_performance import QAPerformance, qa_sync_timed


def test_sync_graph_phase_retains_result_and_safe_failure_metadata():
    recorder = QAPerformance()
    @qa_sync_timed('graph_entry_selection')
    def select_entry(value):
        return value + 1
    @qa_sync_timed('graph_trace_write')
    def write_trace():
        raise ValueError('unit-test-private-provider-text')
    with recorder.activate():
        assert select_entry(2) == 3
        with pytest.raises(ValueError):
            write_trace()
    result = recorder.snapshot()
    assert result.stages['graph_entry_selection'].success_count == 1
    assert result.stages['graph_trace_write'].error_count == 1
    assert 'unit-test-private-provider-text' not in result.model_dump_json()
