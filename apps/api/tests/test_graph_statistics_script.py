import importlib.util
from pathlib import Path

import pytest


def load_script():
    spec = importlib.util.spec_from_file_location('graph_statistics_script',
        Path(__file__).resolve().parents[3] / 'scripts' / 'maintain_graph_statistics.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_statistics_plan_is_read_only_and_table_scope_is_closed(monkeypatch, capsys):
    script = load_script()
    monkeypatch.setattr('sys.argv', ['maintain_graph_statistics.py', '--table', 'chunk_structure_nodes'])
    assert script.main() == 0
    output = capsys.readouterr().out
    assert '"execute": false' in output and 'vacuum_analyze' in output
    monkeypatch.setattr('sys.argv', ['maintain_graph_statistics.py', '--table', 'not_an_allowed_table'])
    with pytest.raises(SystemExit):
        script.main()


def test_maintenance_serial_mode_and_identifier_validation():
    script = load_script()
    assert script.vacuum_statement('chunk_structure_mappings', 0) == 'VACUUM (ANALYZE, PARALLEL 0) "chunk_structure_mappings"'
    for table, workers in [('unexpected_table', 0), ('chunks', -1), ('chunks', True), ('chunks', 3)]:
        with pytest.raises(ValueError):
            script.vacuum_statement(table, workers)
