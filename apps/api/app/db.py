from collections.abc import Generator
import contextvars
import os
import sys

from pathlib import Path

from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, text
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.core.config import get_settings


class Base(DeclarativeBase):
    pass


settings = get_settings()


def candidate_sqlite_paths() -> list[Path]:
    apps_root = Path(__file__).resolve().parents[2]
    candidates = [
        apps_root / "symbograph.db",
        apps_root / "knowledge_base.db",
        apps_root / "course_kg.db",
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


def has_materialized_knowledge_base_data(engine) -> bool:
    try:
        with engine.connect() as connection:
            document_count = connection.execute(text("SELECT COUNT(*) FROM documents")).scalar() or 0
            chunk_count = connection.execute(text("SELECT COUNT(*) FROM chunks")).scalar() or 0
        return bool(document_count or chunk_count)
    except Exception:
        return False


def build_engine():
    database_url = settings.database_url
    connect_args = {"connect_timeout": 5} if database_url.startswith("postgresql") else {}
    engine_kwargs = {
        "future": True,
        "echo": False,
        "connect_args": connect_args,
    }
    if database_url.startswith("postgresql"):
        engine_kwargs.update(
            {
                "pool_pre_ping": True,
                "pool_recycle": 1800,
            }
        )
    primary = create_engine(database_url, **engine_kwargs)
    if try_connect(primary):
        if settings.enable_database_fallback:
            for sqlite_path in candidate_sqlite_paths():
                if not sqlite_path.exists():
                    continue
                fallback = create_engine(f"sqlite:///{sqlite_path.as_posix()}", future=True, echo=False)
                if try_connect(fallback) and has_materialized_knowledge_base_data(fallback) and not has_materialized_knowledge_base_data(primary):
                    return fallback
        return primary

    if not settings.enable_database_fallback:
        if os.getenv("PYTEST_CURRENT_TEST") or "pytest" in sys.modules or any("pytest" in Path(arg).name.lower() for arg in sys.argv):
            return create_engine("sqlite:///:memory:", future=True, echo=False)
        raise RuntimeError("Primary database is unavailable and ENABLE_DATABASE_FALLBACK is false")

    for sqlite_path in candidate_sqlite_paths():
        fallback = create_engine(f"sqlite:///{sqlite_path.as_posix()}", future=True, echo=False)
        if try_connect(fallback):
            return fallback

    raise RuntimeError("No available database engine could be initialized")


engine = build_engine()
_engine_process_id = os.getpid()
_original_SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, expire_on_commit=False, future=True)
_db_context_var = contextvars.ContextVar("db_session", default=None)
_active_sessions = contextvars.ContextVar("active_sessions", default=None)


def reset_database_engine_after_fork(*, force: bool = False) -> bool:
    """Detach inherited DBAPI connections before a child opens a Session.

    A SQLAlchemy pool created in the Celery parent may contain a live psycopg
    connection from the startup availability probe.  Reusing that socket in
    prefork children corrupts connection-local prepared-statement state.  The
    child must replace the pool without asking it to close the parent's file
    descriptors; all subsequent ``SessionLocal`` calls still bind to the same
    Engine object and therefore use its fresh, child-owned pool.
    """

    global _engine_process_id
    current_process_id = os.getpid()
    if not force and current_process_id == _engine_process_id:
        return False
    engine.dispose(close=False)
    _engine_process_id = current_process_id
    return True


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=reset_database_engine_after_fork)

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


def _running_under_pytest() -> bool:
    return bool(os.getenv("PYTEST_CURRENT_TEST") or "pytest" in sys.modules or any("pytest" in Path(arg).name.lower() for arg in sys.argv))


def _is_memory_sqlite() -> bool:
    return engine.url.drivername.startswith("sqlite") and (engine.url.database in {None, "", ":memory:"})


def run_alembic_upgrade() -> None:
    alembic_ini = Path(__file__).resolve().parents[1] / "alembic.ini"
    if not alembic_ini.exists():
        raise RuntimeError(f"Alembic configuration not found: {alembic_ini}")
    config = Config(str(alembic_ini))
    config.set_main_option("script_location", str(alembic_ini.parent / "migrations"))
    config.set_main_option("sqlalchemy.url", engine.url.render_as_string(hide_password=False))
    command.upgrade(config, "head")


def ensure_schema() -> None:
    import app.models  # noqa: F401

    if _running_under_pytest() or _is_memory_sqlite():
        Base.metadata.create_all(bind=engine)
    else:
        run_alembic_upgrade()
    from app.services.strategy_profiles import ensure_knowledge_bases_have_profiles

    with SessionLocal() as session:
        ensure_knowledge_bases_have_profiles(session)
        session.commit()


def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
