from __future__ import annotations

from pathlib import Path


def test_alembic_upgrade_creates_four_layer_schema(tmp_path, monkeypatch):
    from alembic import command
    from alembic.config import Config
    from sqlalchemy import create_engine, inspect
    from app.core.config import get_settings

    database_path = tmp_path / "migration.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{database_path.as_posix()}")
    monkeypatch.setenv("DATA_ROOT", str(tmp_path / "data"))
    get_settings.cache_clear()
    engine = create_engine(f"sqlite:///{database_path.as_posix()}", future=True)
    api_root = Path(__file__).resolve().parents[1]
    config = Config(str(api_root / "alembic.ini"))
    config.set_main_option("script_location", str(api_root / "migrations"))
    config.set_main_option("sqlalchemy.url", f"sqlite:///{database_path.as_posix()}")
    command.upgrade(config, "head")
    tables = set(inspect(engine).get_table_names())
    columns = {column["name"] for column in inspect(engine).get_columns("graph_retrieval_steps")}
    trace_columns = {column["name"] for column in inspect(engine).get_columns("retrieval_traces")}
    relation_columns = {column["name"] for column in inspect(engine).get_columns("chunk_relation_edges")}
    relation_state_columns = {column["name"] for column in inspect(engine).get_columns("chunk_relation_graph_states")}
    assert "chunks" in tables
    assert "chunk_structure_nodes" in tables
    assert "chunk_relation_edges" in tables
    assert "rq_prefixes" in tables
    assert "mid_concepts" in tables
    assert "coarse_concepts" in tables
    assert "context_packages" in tables
    assert "evidence_atoms" not in tables
    assert "active_chunks" not in tables
    assert "bm25_records" not in tables
    assert "cycle_distance_reward" in columns
    assert "gray_zone_path_decisions_json" in columns
    assert "action_type" in columns
    assert "candidate_pool_ids_json" in columns
    assert "selected_topk_ids_json" in columns
    assert "cycle_reward" not in columns
    assert "ambiguous_edge_decisions_json" not in columns
    assert {"stage_queues_json", "candidate_pools_json", "topk_selection_json"}.issubset(trace_columns)
    assert {"normalization_stats_json", "edge_distance_protocol_hash", "is_cross_document", "is_cross_language"}.issubset(relation_columns)
    assert {"graph_operating_point_hash", "edge_type_calibration_protocol_hash"}.issubset(relation_state_columns)
