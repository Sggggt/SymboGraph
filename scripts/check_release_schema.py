"""Plan schema convergence; --execute applies reviewed Alembic revisions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    api = root / 'apps' / 'api'
    sys.path.insert(0, str(api))
    from app.services.runtime_settings import initialize_runtime_configuration_from_root_files
    initialize_runtime_configuration_from_root_files()
    from app.core.config import get_settings
    from sqlalchemy import create_engine, text
    from alembic import command
    from alembic.config import Config
    from alembic.script import ScriptDirectory
    from alembic.migration import MigrationContext
    from alembic.autogenerate import produce_migrations
    from app.db import Base
    import app.models
    config = Config(str(api / 'alembic.ini'))
    config.set_main_option('script_location', str(api / 'migrations'))
    scripts = ScriptDirectory.from_config(config)
    engine = create_engine(get_settings().database_url)
    with engine.connect() as connection:
        context = MigrationContext.configure(connection, opts={'compare_type': True})
        current = context.get_current_revision()
        target = scripts.get_current_head()
        pending = [revision.revision for revision in scripts.iterate_revisions(target, current)]
        active_runs = connection.scalar(text("SELECT count(*) FROM agent_runs WHERE status IN ('pending','running')"))
        active_batches = connection.scalar(text("SELECT count(*) FROM ingestion_batches WHERE status IN ('pending','running','processing')"))
        diff = produce_migrations(context, Base.metadata)
        drift = sum(len(group.ops) if hasattr(group,'ops') else 1 for group in diff.upgrade_ops.ops)
    print(json.dumps({'mode':'execute' if args.execute else 'plan','current_revision':current,'target_revision':target,
        'pending_revisions':pending,'schema_difference_count':drift,'active_runs':active_runs,'active_batches':active_batches,
        'impact':'schema constraints and indexes; no graph rebuild, row deletion or source-file changes'}))
    if args.execute:
        if active_runs or active_batches:
            raise RuntimeError('release_schema_requires_idle_service_boundary')
        command.upgrade(config,'head')
        command.check(config)
        print(json.dumps({'schema_convergence_passed':True,'target_revision':target}))


if __name__ == '__main__':
    main()
