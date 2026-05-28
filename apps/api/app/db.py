from collections.abc import Generator
import contextvars

from pathlib import Path

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.sql.compiler import IdentifierPreparer
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.core.config import get_settings


class Base(DeclarativeBase):
    pass


settings = get_settings()


def candidate_sqlite_paths() -> list[Path]:
    apps_root = Path(__file__).resolve().parents[2]
    candidates = [
        apps_root / "course_kg.db",
        apps_root / "knowledge_base.db",
    ]
    seen: set[Path] = set()
    ordered: list[Path] = []
    for path in candidates:
        if path in seen:
            continue
        seen.add(path)
        ordered.append(path)
    return ordered


def try_connect(engine) -> bool:
    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
        return True
    except Exception:
        return False


def has_materialized_course_data(engine) -> bool:
    try:
        with engine.connect() as connection:
            document_count = connection.execute(text("SELECT COUNT(*) FROM documents")).scalar() or 0
            concept_count = connection.execute(text("SELECT COUNT(*) FROM concepts")).scalar() or 0
        return bool(document_count or concept_count)
    except Exception:
        return False


def build_engine():
    database_url = settings.database_url
    connect_args = {"connect_timeout": 5} if database_url.startswith("postgresql") else {}
    primary = create_engine(database_url, future=True, echo=False, connect_args=connect_args)
    if try_connect(primary):
        if settings.enable_database_fallback:
            for sqlite_path in candidate_sqlite_paths():
                if not sqlite_path.exists():
                    continue
                fallback = create_engine(f"sqlite:///{sqlite_path.as_posix()}", future=True, echo=False)
                if try_connect(fallback) and has_materialized_course_data(fallback) and not has_materialized_course_data(primary):
                    return fallback
        return primary

    if not settings.enable_database_fallback:
        raise RuntimeError("Primary database is unavailable and ENABLE_DATABASE_FALLBACK is false")

    for sqlite_path in candidate_sqlite_paths():
        fallback = create_engine(f"sqlite:///{sqlite_path.as_posix()}", future=True, echo=False)
        if try_connect(fallback):
            return fallback

    raise RuntimeError("No available database engine could be initialized")


engine = build_engine()
_original_SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, expire_on_commit=False, future=True)
_db_context_var = contextvars.ContextVar("db_session", default=None)
_active_sessions = contextvars.ContextVar("active_sessions", default=None)

class ContextSessionWrapper:
    def __init__(self, session):
        object.__setattr__(self, "_session", session)
        
    def __getattr__(self, name):
        return getattr(self._session, name)
        
    def __setattr__(self, name, value):
        setattr(self._session, name, value)
        
    def close(self):
        pass
        
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        return False

class ContextSessionLocalProxy:
    def __init__(self, original_sessionmaker):
        self.original_sessionmaker = original_sessionmaker
        
    def __getattr__(self, name):
        return getattr(self.original_sessionmaker, name)
        
    def __call__(self, *args, **kwargs):
        ctx_db = _db_context_var.get(None)
        if ctx_db is not None:
            return ContextSessionWrapper(ctx_db)
            
        new_db = self.original_sessionmaker(*args, **kwargs)
        sessions_list = _active_sessions.get(None)
        if sessions_list is not None:
            sessions_list.append(new_db)
        return new_db

SessionLocal = ContextSessionLocalProxy(_original_SessionLocal)


SCHEMA_PATCHES: dict[str, dict[str, str]] = {
    "chunks": {
        "parent_chunk_id": "VARCHAR(36)",
        "summary": "TEXT",
        "keywords": "JSON DEFAULT '[]'",
        "embedding_text_version": "VARCHAR(32) DEFAULT 'metadata_enriched_v1'",
    },
    "concepts": {
        "normalized_name": "TEXT",
        "concept_type": "VARCHAR(64) DEFAULT 'concept'",
        "importance_score": "FLOAT DEFAULT 0",
        "evidence_count": "INTEGER DEFAULT 0",
        "community_louvain": "INTEGER",
        "community_spectral": "INTEGER",
        "component_id": "INTEGER",
        "centrality_json": "JSON DEFAULT '{}'",
        "graph_rank_score": "FLOAT DEFAULT 0",
        "source_document_ids": "JSON DEFAULT '[]'",
        "quality_json": "JSON DEFAULT '{}'",
    },
    "concept_relations": {
        "confidence": "FLOAT DEFAULT 0.55",
        "extraction_method": "VARCHAR(64) DEFAULT 'heuristic'",
        "is_validated": "BOOLEAN DEFAULT false",
        "weight": "FLOAT DEFAULT 0",
        "semantic_similarity": "FLOAT DEFAULT 0",
        "support_count": "INTEGER DEFAULT 1",
        "relation_source": "VARCHAR(64) DEFAULT 'llm'",
        "is_inferred": "BOOLEAN DEFAULT false",
        "metadata_json": "JSON DEFAULT '{}'",
        "source_document_ids": "JSON DEFAULT '[]'",
    },
    "ingestion_jobs": {
        "batch_id": "VARCHAR(36)",
        "source_path": "TEXT",
    },
    "qa_sessions": {
        "title": "VARCHAR(255)",
        "last_question": "TEXT",
        "last_answer": "TEXT",
        "transcript": "JSON DEFAULT '[]'",
    },
    "agent_runs": {
        "session_id": "VARCHAR(36)",
        "route": "VARCHAR(64)",
        "current_node": "VARCHAR(64)",
        "retry_count": "INTEGER DEFAULT 0",
        "final_answer": "TEXT",
        "error_message": "TEXT",
        "metadata_json": "JSON DEFAULT '{}'",
        "started_at": "DATETIME",
        "completed_at": "DATETIME",
    },
    "agent_trace_events": {
        "document_ids": "JSON DEFAULT '[]'",
        "scores": "JSON DEFAULT '{}'",
        "duration_ms": "INTEGER DEFAULT 0",
        "error_message": "TEXT",
    },
    "quality_profiles": {
        "sample_chunk_ids": "JSON DEFAULT '[]'",
        "is_active": "BOOLEAN DEFAULT true",
    },
    "graph_relation_candidates": {
        "decision_json": "JSON DEFAULT '{}'",
        "metadata_json": "JSON DEFAULT '{}'",
        "source_document_ids": "JSON DEFAULT '[]'",
    },
    "graph_community_summaries": {
        "key_concepts_json": "JSON DEFAULT '[]'",
        "representative_chunk_ids": "JSON DEFAULT '[]'",
        "source_document_ids": "JSON DEFAULT '[]'",
        "quality_json": "JSON DEFAULT '{}'",
        "is_active": "BOOLEAN DEFAULT true",
    },
    "graph_extraction_runs": {
        "coverage_json": "JSON DEFAULT '{}'",
        "budget_json": "JSON DEFAULT '{}'",
        "stats_json": "JSON DEFAULT '{}'",
        "error_message": "TEXT",
        "started_at": "DATETIME",
        "completed_at": "DATETIME",
    },
    "graph_extraction_chunk_tasks": {
        "selected_reason": "JSON DEFAULT '{}'",
        "payload_json": "JSON",
        "error_message": "TEXT",
        "token_estimate": "INTEGER DEFAULT 0",
    },
    "course_model_hyperparameters": {
        "llm_model_name": "VARCHAR(128) DEFAULT 'unknown'",
        "embedding_model_name": "VARCHAR(128) DEFAULT 'unknown'",
        "embedding_text_version": "VARCHAR(32) DEFAULT 'metadata_enriched_v1'",
        "model_name": "VARCHAR(384)",
        "graph_version": "VARCHAR(128) DEFAULT 'active'",
        "min_relation_confidence": "FLOAT DEFAULT 0.72",
        "min_accepted_relation_weight": "FLOAT DEFAULT 0.62",
        "dijkstra_semantic_threshold": "FLOAT DEFAULT 0.78",
        "w_degree": "FLOAT DEFAULT 0.25",
        "w_weighted_degree": "FLOAT DEFAULT 0.25",
        "w_pagerank": "FLOAT DEFAULT 0.20",
        "w_betweenness": "FLOAT DEFAULT 0.20",
        "w_closeness": "FLOAT DEFAULT 0.10",
        "w_centrality": "FLOAT DEFAULT 0.50",
        "w_llm_importance": "FLOAT DEFAULT 0.25",
        "w_evidence": "FLOAT DEFAULT 0.25",
        "hpo_status": "VARCHAR(32) DEFAULT 'pending'",
        "last_optimized_at": "DATETIME",
        "optuna_history": "JSON DEFAULT '{}'",
    },
    "graph_hpo_judge_samples": {
        "reasons": "JSON DEFAULT '[]'",
        "safety_flags": "JSON DEFAULT '[]'",
        "raw_response": "JSON DEFAULT '{}'",
    },
    "graph_hpo_objective_models": {
        "training_audit": "JSON DEFAULT '{}'",
        "status": "VARCHAR(32) DEFAULT 'completed'",
    },
}


def _migrate_course_model_hyperparameters(connection) -> None:
    if engine.dialect.name != "postgresql":
        return
    from app.services.chunking import CURRENT_EMBEDDING_TEXT_VERSION

    default_llm = settings.chat_model
    default_embedding = settings.embedding_model
    default_text_version = CURRENT_EMBEDDING_TEXT_VERSION
    connection.execute(
        text(
            """
            UPDATE course_model_hyperparameters
            SET llm_model_name = CASE
                    WHEN model_name LIKE 'llm:%|embedding:%|text:%'
                        THEN regexp_replace(split_part(model_name, '|', 1), '^llm:', '')
                    ELSE COALESCE(NULLIF(model_name, ''), :default_llm)
                END
            WHERE llm_model_name IS NULL OR llm_model_name = '' OR llm_model_name = 'unknown'
            """
        ),
        {"default_llm": default_llm},
    )
    connection.execute(
        text(
            """
            UPDATE course_model_hyperparameters
            SET embedding_model_name = CASE
                    WHEN model_name LIKE 'llm:%|embedding:%|text:%'
                        THEN regexp_replace(split_part(model_name, '|', 2), '^embedding:', '')
                    ELSE :default_embedding
                END
            WHERE embedding_model_name IS NULL OR embedding_model_name = '' OR embedding_model_name = 'unknown'
            """
        ),
        {"default_embedding": default_embedding},
    )
    connection.execute(
        text(
            """
            UPDATE course_model_hyperparameters
            SET embedding_text_version = CASE
                    WHEN model_name LIKE 'llm:%|embedding:%|text:%'
                        THEN regexp_replace(split_part(model_name, '|', 3), '^text:', '')
                    ELSE :default_text_version
                END
            WHERE embedding_text_version IS NULL
               OR embedding_text_version = ''
               OR embedding_text_version = 'metadata_enriched_v1'
            """
        ),
        {"default_text_version": default_text_version},
    )
    connection.execute(
        text(
            """
            UPDATE course_model_hyperparameters
            SET model_name = concat(
                'llm:', llm_model_name,
                '|embedding:', embedding_model_name,
                '|text:', embedding_text_version
            )
            WHERE model_name IS NULL OR model_name = ''
            """
        )
    )
    connection.execute(
        text(
            """
            DELETE FROM course_model_hyperparameters AS old
            USING (
                SELECT ctid,
                       row_number() OVER (
                           PARTITION BY course_id, llm_model_name, embedding_model_name, embedding_text_version
                           ORDER BY updated_at DESC NULLS LAST, created_at DESC NULLS LAST, ctid DESC
                       ) AS rn
                FROM course_model_hyperparameters
            ) AS ranked
            WHERE old.ctid = ranked.ctid AND ranked.rn > 1
            """
        )
    )
    for column_name in ("llm_model_name", "embedding_model_name", "embedding_text_version"):
        connection.execute(text(f"ALTER TABLE course_model_hyperparameters ALTER COLUMN {column_name} SET NOT NULL"))
    pk_columns = [
        row[0]
        for row in connection.execute(
            text(
                """
                SELECT a.attname
                FROM pg_constraint c
                JOIN pg_class t ON t.oid = c.conrelid
                JOIN unnest(c.conkey) WITH ORDINALITY AS cols(attnum, ord) ON true
                JOIN pg_attribute a ON a.attrelid = t.oid AND a.attnum = cols.attnum
                WHERE t.relname = 'course_model_hyperparameters' AND c.contype = 'p'
                ORDER BY cols.ord
                """
            )
        )
    ]
    target_columns = ["course_id", "llm_model_name", "embedding_model_name", "embedding_text_version"]
    if pk_columns != target_columns:
        pk_name = connection.execute(
            text(
                """
                SELECT c.conname
                FROM pg_constraint c
                JOIN pg_class t ON t.oid = c.conrelid
                WHERE t.relname = 'course_model_hyperparameters' AND c.contype = 'p'
                """
            )
        ).scalar()
        if pk_name:
            connection.execute(text(f'ALTER TABLE course_model_hyperparameters DROP CONSTRAINT "{pk_name}"'))
        connection.execute(
            text(
                """
                ALTER TABLE course_model_hyperparameters
                ADD CONSTRAINT course_model_hyperparameters_pkey
                PRIMARY KEY (course_id, llm_model_name, embedding_model_name, embedding_text_version)
                """
            )
        )
    connection.execute(text("ALTER TABLE course_model_hyperparameters ALTER COLUMN model_name DROP NOT NULL"))


def ensure_schema() -> None:
    Base.metadata.create_all(bind=engine)
    inspector = inspect(engine)
    preparer = IdentifierPreparer(engine.dialect)
    with engine.begin() as connection:
        for table_name, patch_columns in SCHEMA_PATCHES.items():
            if table_name not in inspector.get_table_names():
                continue
            existing = {column["name"] for column in inspector.get_columns(table_name)}
            for column_name, column_sql in patch_columns.items():
                if column_name in existing:
                    continue
                table_sql = preparer.quote(table_name)
                column_name_sql = preparer.quote(column_name)
                connection.execute(text(" ".join(["ALTER TABLE", table_sql, "ADD COLUMN", column_name_sql, column_sql])))
        if "course_model_hyperparameters" in inspector.get_table_names():
            _migrate_course_model_hyperparameters(connection)


def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
