"""Plan or run bounded PostgreSQL VACUUM ANALYZE for graph tables in Docker."""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
TABLES = ('chunk_structure_nodes', 'chunk_structure_edges', 'chunk_structure_mappings',
          'chunks', 'vector_records', 'chunk_context_texts')


def vacuum_statement(table, parallel_workers):
    if table not in TABLES or type(parallel_workers) is not int or not 0 <= parallel_workers <= 2:
        raise ValueError('maintenance_operation_outside_allowlist')
    return f'VACUUM (ANALYZE, PARALLEL {parallel_workers}) "{table}"'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--table', action='append', choices=TABLES)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--deadline-seconds', type=int, default=60)
    parser.add_argument('--parallel-workers', type=int, choices=(0, 1, 2), default=0)
    parser.add_argument('--output', type=Path, default=ROOT / 'output' / 'graph-statistics-maintenance.json')
    args = parser.parse_args()
    tables = list(dict.fromkeys(args.table or TABLES))
    if not 1 <= args.deadline_seconds <= 600:
        raise ValueError('maintenance_deadline_invalid')
    if not args.output.resolve().is_relative_to((ROOT / 'output').resolve()):
        raise ValueError('maintenance_output_outside_workspace_output')
    print(json.dumps({'operation': 'vacuum_analyze', 'execute': args.execute, 'tables': tables,
        'deadline_seconds': args.deadline_seconds, 'parallel_workers': args.parallel_workers,
        'impact': 'Refresh PostgreSQL visibility/statistics; preserve logical rows and graph identities'}), flush=True)
    if not args.execute:
        return 0
    if not Path('/.dockerenv').exists() or args.output.exists():
        raise ValueError('maintenance_requires_docker_and_new_output')
    sys.path.insert(0, str(ROOT / 'apps' / 'api'))
    from sqlalchemy import text
    from app.services.runtime_settings import initialize_runtime_configuration_from_root_files, refresh_runtime_settings_if_needed
    initialize_runtime_configuration_from_root_files()
    refresh_runtime_settings_if_needed(force=True, sync_bridge=False)
    from app.db import engine
    if engine.dialect.name != 'postgresql':
        raise ValueError('maintenance_requires_postgresql')
    started, results = time.monotonic(), []
    with engine.connect().execution_options(isolation_level='AUTOCOMMIT') as connection:
        for table in tables:
            remaining = args.deadline_seconds - (time.monotonic() - started)
            if remaining <= 0:
                results.append({'table': table, 'status': 'not_run', 'reason': 'deadline'})
                break
            connection.execute(text("SELECT set_config('statement_timeout', :timeout, false)"),
                               {'timeout': str(max(1, int(remaining * 1000)))})
            step = time.monotonic()
            try:
                connection.exec_driver_sql(vacuum_statement(table, args.parallel_workers))
                row = {'table': table, 'status': 'completed', 'seconds': time.monotonic()-step}
            except Exception as exc:
                row = {'table': table, 'status': 'failed', 'seconds': time.monotonic()-step, 'error_type': type(exc).__name__,
                       'sqlstate': getattr(getattr(exc, 'orig', None), 'sqlstate', None)}
            results.append(row)
            print(json.dumps(row), flush=True)
            if row['status'] != 'completed':
                break
    completed = {row['table'] for row in results}
    results.extend({'table': table, 'status': 'not_run', 'reason': 'prior_failure_or_deadline'}
                   for table in tables if table not in completed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({'operation': 'vacuum_analyze', 'results': results,
        'elapsed_seconds': time.monotonic()-started}, indent=2), encoding='utf-8')
    return int(any(row['status'] != 'completed' for row in results))


if __name__ == '__main__':
    raise SystemExit(main())
