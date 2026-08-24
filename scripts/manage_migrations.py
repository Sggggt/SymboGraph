from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
API_WORKDIR = "/app/apps/api"
COMPOSE_FILE = REPO_ROOT / "infra" / "docker-compose.yml"
ENV_FILE = REPO_ROOT / ".env"


def run_command(command: list[str], *, dry_run: bool = False) -> int:
    printable = " ".join(command)
    print(printable)
    if dry_run:
        return 0
    completed = subprocess.run(command, cwd=REPO_ROOT)
    return int(completed.returncode)


def docker_api_command(args: argparse.Namespace, api_args: list[str]) -> list[str]:
    if not args.compose_run:
        return ["docker", "exec", "-w", API_WORKDIR, args.container, *api_args]
    command = ["docker", "compose"]
    if ENV_FILE.exists():
        command.extend(["--env-file", str(ENV_FILE)])
    command.extend(["--project-name", args.compose_project_name])
    command.extend(["-f", str(COMPOSE_FILE), "run", "--rm", "--no-deps", "api", *api_args])
    return command


def alembic_args(operation: str, revision: str, *, allow_destructive: bool = False) -> list[str]:
    args = ["alembic"]
    if allow_destructive:
        args.extend(["-x", "allow_destructive=true"])
    args.extend([operation, revision])
    return args


def preflight_args(revision: str, *, allow_destructive: bool = False) -> list[str]:
    args = ["python", "-m", "app.core.migration_safety", "preflight", "--target-revision", revision]
    if allow_destructive:
        args.append("--allow-destructive")
    return args


def build_commands(args: argparse.Namespace) -> list[list[str]]:
    if args.command == "current":
        return [docker_api_command(args, ["alembic", "current"])]
    if args.command == "heads":
        return [docker_api_command(args, ["alembic", "heads"])]
    if args.command == "preflight":
        return [docker_api_command(args, preflight_args(args.revision, allow_destructive=args.allow_destructive))]
    if args.command == "upgrade":
        return [
            docker_api_command(args, preflight_args(args.revision, allow_destructive=args.allow_destructive)),
            docker_api_command(args, alembic_args("upgrade", args.revision, allow_destructive=args.allow_destructive)),
        ]
    if args.command == "downgrade":
        return [docker_api_command(args, alembic_args("downgrade", args.revision, allow_destructive=args.allow_destructive))]
    if args.command == "check":
        return [docker_api_command(args, ["alembic", "check"])]
    if args.command == "revision":
        revision_args = ["revision", "--autogenerate", "-m", args.message]
        if args.rev_id:
            revision_args.extend(["--rev-id", args.rev_id])
        return [docker_api_command(args, ["alembic", *revision_args])]
    raise ValueError(f"Unsupported command: {args.command}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Alembic safely inside the API Docker container.")
    parser.add_argument("--container", default="course-kg-api", help="API container name.")
    parser.add_argument(
        "--compose-project-name",
        default="knowledgegraph-dev-20260820",
        help="Compose project used by start-app.ps1.",
    )
    parser.add_argument(
        "--api-image",
        default="course-kg-api:local",
        help="Runtime API image override; prevents stale digest values in .env from becoming build tags.",
    )
    parser.add_argument(
        "--web-image",
        default="course-kg-web:local",
        help="Runtime web image override used for Compose interpolation parity.",
    )
    parser.add_argument(
        "--compose-run",
        action="store_true",
        help="Use a one-shot API Compose container; use this when the API cannot start because a migration is blocked.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print the docker command without executing it.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("current", help="Show the current database revision.")
    subparsers.add_parser("heads", help="Show migration heads in code.")
    subparsers.add_parser("check", help="Run alembic check inside the API container.")

    preflight = subparsers.add_parser("preflight", help="Read-only inspection of pending destructive migration targets.")
    preflight.add_argument("revision", nargs="?", default="head")
    preflight.add_argument("--allow-destructive", action="store_true", help="Preview the same report with explicit authorization; does not mutate data.")

    upgrade = subparsers.add_parser("upgrade", help="Upgrade the database revision.")
    upgrade.add_argument("revision", nargs="?", default="head")
    upgrade.add_argument(
        "--allow-destructive",
        action="store_true",
        help="Authorize all destructive targets printed by the mandatory preflight for this one upgrade invocation.",
    )

    downgrade = subparsers.add_parser("downgrade", help="Downgrade the database revision.")
    downgrade.add_argument("revision", help="Target revision, for example -1.")
    downgrade.add_argument("--allow-destructive", action="store_true", help="Required for an executing downgrade.")

    revision = subparsers.add_parser("revision", help="Create an autogenerated migration revision.")
    revision.add_argument("-m", "--message", required=True)
    revision.add_argument("--rev-id", default="")

    args = parser.parse_args(argv)
    if args.compose_run:
        os.environ["API_IMAGE"] = args.api_image
        os.environ["WEB_IMAGE"] = args.web_image
        os.environ["COMPOSE_PROJECT_NAME"] = args.compose_project_name
    if args.command == "downgrade" and not args.dry_run and not args.allow_destructive:
        parser.error("executing a downgrade requires --allow-destructive; use --dry-run to print the command only")
    commands = build_commands(args)
    for command in commands:
        return_code = run_command(command, dry_run=args.dry_run)
        if return_code != 0:
            return return_code
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
