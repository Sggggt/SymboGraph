"""Export the proposed source tree through an isolated Git index, without committing."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATHS = ('apps/api', 'apps/worker', 'apps/web', 'packages', 'scripts', 'infra', 'docs', '.env.example')


def git(*arguments, env=None):
    result = subprocess.run(['git',*arguments],cwd=ROOT,env=env,capture_output=True,text=True,encoding='utf-8')
    if result.returncode:
        raise RuntimeError('release_snapshot_git_operation_failed:' + arguments[0])
    return result.stdout


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'output'/'release-source')
    parser.add_argument('--execute',action='store_true')
    args = parser.parse_args()
    destination = args.output.resolve()
    if not destination.is_relative_to((ROOT/'output').resolve()) or destination == (ROOT/'output').resolve():
        raise ValueError('release_snapshot_requires_dedicated_output_directory')
    untracked = git('ls-files','--others','--exclude-standard','--',*SOURCE_PATHS).splitlines()
    print(json.dumps({'mode':'execute' if args.execute else 'plan','destination':str(destination),
        'proposed_new_source_files':len(untracked),'user_git_index_changed':False,'creates_commit':False,
        'source_paths':SOURCE_PATHS,'includes_runtime_env':False}))
    if not args.execute:
        return
    if destination.exists():
        raise ValueError('release_snapshot_output_must_not_exist')
    destination.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='release-index-',dir=destination.parent) as temporary:
        environment = {**os.environ,'GIT_INDEX_FILE':str(Path(temporary)/'index')}
        git('read-tree','HEAD',env=environment)
        git('add','--update',env=environment)
        git('add','--',*SOURCE_PATHS,env=environment)
        files = git('ls-files','--cached',env=environment).splitlines()
        if any(name == '.env' or name.startswith(('output/','data/','node_modules/','local_light_tests/')) for name in files):
            raise ValueError('release_snapshot_contains_runtime_or_private_files')
        destination.mkdir()
        git('checkout-index','--all','--prefix='+destination.as_posix()+'/',env=environment)
        print(json.dumps({'snapshot_created':True,'source_file_count':len(files),'user_git_index_changed':False,
            'creates_commit':False,'runtime_files_included':False}))


if __name__ == '__main__':
    main()
