from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]


def test_api_image_installs_the_exact_uv_lock_into_its_runtime_environment() -> None:
    dockerfile = (REPO_ROOT / "apps/api/Dockerfile").read_text(encoding="utf-8")

    lock_copy = "COPY apps/api/uv.lock ./uv.lock"
    locked_sync = (
        "uv sync --locked --no-install-project --extra dev --extra ocr"
    )
    locked_check = (
        "uv sync --locked --check --no-install-project --extra dev --extra ocr"
    )
    assert dockerfile.index(lock_copy) < dockerfile.index(locked_sync)
    assert dockerfile.startswith(
        "FROM python:3.13-slim@sha256:"
        "a0779d7c12fc20be6ec6b4ddc901a4fd7657b8a6bc9def9d3fde89ed5efe0a3d"
    )
    assert "pip install --no-cache-dir uv==0.12.3" in dockerfile
    assert "ca-certificates=20250419" in dockerfile
    assert "curl=8.14.1-2+deb13u4" in dockerfile
    assert "tesseract-ocr=5.5.0-1+b1" in dockerfile
    assert "--mount=type=cache,target=/var/cache/apt,sharing=locked" in dockerfile
    assert "--mount=type=cache,target=/root/.cache/uv,sharing=locked" in dockerfile
    assert "for attempt in 1 2 3" in dockerfile
    assert 'if [ "$attempt" = 3 ]' in dockerfile
    assert locked_check in dockerfile
    assert 'VIRTUAL_ENV=/app/apps/api/.venv' in dockerfile
    assert 'PATH="/app/apps/api/.venv/bin:${PATH}"' in dockerfile
    assert "uv pip install" not in dockerfile


def test_runtime_source_mounts_use_the_single_root_env_without_a_config_volume() -> None:
    compose = (REPO_ROOT / "infra/docker-compose.yml").read_text(encoding="utf-8")

    assert "../.env.example:/app/.env.example" not in compose
    assert "COPY .env.example /app/.env.example" not in (
        REPO_ROOT / "apps/api/Dockerfile"
    ).read_text(encoding="utf-8")
    assert compose.count("../apps/api/app:/app/apps/api/app:ro") == 3
    for mount in (
        "../apps/api/migrations:/app/apps/api/migrations:ro",
        "../apps/worker/worker_app:/app/apps/worker/worker_app:ro",
        "../packages/shared/src:/app/packages/shared/src:ro",
        "../scripts:/app/scripts:ro",
    ):
        assert compose.count(mount) == 3
    assert compose.count("../apps/api/tests:/app/apps/api/tests:ro") == 2
    assert compose.count("../apps/web/src:/app/apps/web/src:ro") == 1
    assert "runtime-bootstrap" not in compose
    assert "RUNTIME_DESIRED_ENV_FILE" not in compose
    assert "symbograph-runtime-config" not in compose
    assert compose.count("RUNTIME_ENV_FILE: /workspace/.env") == 3
    assert compose.count("source: ..\n        target: /workspace") == 3


def test_api_image_contains_repository_contract_sources() -> None:
    dockerfile = (REPO_ROOT / "apps/api/Dockerfile").read_text(
        encoding="utf-8"
    )

    for source_copy in (
        "COPY apps/web/src /app/apps/web/src",
        "COPY packages/shared/src /app/packages/shared/src",
        "COPY package.json /app/package.json",
        "COPY docs/todo.md /app/docs/todo.md",
        "COPY infra/sample-import/.gitkeep /app/infra/sample-import/.gitkeep",
        "COPY start-app.bat /app/start-app.bat",
        "COPY rebuild-images.ps1 /app/rebuild-images.ps1",
        "COPY rebuild-images.bat /app/rebuild-images.bat",
    ):
        assert source_copy in dockerfile


def test_single_root_env_contract_has_no_second_file_or_product_dual_state() -> None:
    compose = (REPO_ROOT / "infra/docker-compose.yml").read_text(encoding="utf-8")
    launcher = (REPO_ROOT / "start-app.ps1").read_text(encoding="utf-8")
    runtime_service = (
        REPO_ROOT / "apps/api/app/services/runtime_settings.py"
    ).read_text(encoding="utf-8")
    schemas = (REPO_ROOT / "apps/api/app/schemas.py").read_text(encoding="utf-8")
    shared = (REPO_ROOT / "packages/shared/src/index.ts").read_text(encoding="utf-8")
    settings_ui = (
        REPO_ROOT / "apps/web/src/components/settings-workspace.tsx"
    ).read_text(encoding="utf-8")

    combined_runtime = "\n".join((compose, launcher, runtime_service))
    for forbidden in (
        "desired.env",
        "RUNTIME_DESIRED_ENV_FILE",
        "symbograph-runtime-config",
        "save_desired_model_settings",
        "desired_env_path",
    ):
        assert forbidden not in combined_runtime
    for contract in (schemas, shared):
        for forbidden in (
            "desired_values",
            "desired_env_synced",
            "active_settings_version",
        ):
            assert forbidden not in contract
    for forbidden in (
        "desired 已保存",
        "共享 desired",
        "共享 active env",
        "聊天目标 hash",
        "向量目标 hash",
        "JSON.stringify(metrics)",
    ):
        assert forbidden not in settings_ui


def test_windows_launcher_uses_the_current_source_mounted_runtime_contract() -> None:
    launcher = (REPO_ROOT / "start-app.ps1").read_text(encoding="utf-8")
    wrapper = (REPO_ROOT / "start-app.bat").read_text(encoding="utf-8")
    rebuild = (REPO_ROOT / "rebuild-images.ps1").read_text(encoding="utf-8")
    rebuild_wrapper = (REPO_ROOT / "rebuild-images.bat").read_text(encoding="utf-8")

    assert '[string]$ComposeProjectName = "knowledgegraph-dev-20260820"' in launcher
    assert '[string]$ApiImage = "course-kg-api:dev"' in launcher
    assert '[string]$WebImage = "course-kg-web:dev"' in launcher
    assert 'RUNTIME_CONFIG_VOLUME_NAME' not in launcher
    assert '$env:SAMPLE_IMPORT_PATH = $sampleImportPath' in launcher
    assert 'Configured SAMPLE_IMPORT_PATH is absent' in launcher
    assert '"--project-name", $ComposeProjectName' in launcher
    assert '"--profile", "model-bridge"' in launcher
    assert '"config", "--quiet"' in launcher
    assert '& $RebuildImagesScript' in launcher
    assert '-ApiBuildTag $ApiImage' in launcher
    assert '-WebBuildTag $WebImage' in launcher
    assert "digest-qualified and cannot be used as a Docker build output tag" in launcher
    assert 'Re-run without -SkipBuild' in launcher
    assert '-f infra/docker-compose.yml --profile model-bridge down' in launcher
    assert '/api/ready' in launcher
    assert 'Wait-ContainerHealthy -ContainerName "course-kg-redis"' in launcher
    assert 'Wait-Url -Url "http://127.0.0.1:6333/readyz"' in launcher
    assert '"runtime-bootstrap"' not in launcher
    assert 'Runtime settings file: $EnvFile' in launcher
    assert '"app.core.migration_safety", "preflight"' in launcher
    assert 'docker stop $containerName' not in launcher
    assert "migrate_initial_runtime_provider_config.py" not in launcher
    assert 'ENABLE_MODEL_FALLBACK=false and ENABLE_DATABASE_FALLBACK=false' in launcher
    assert '[Security.Cryptography.RandomNumberGenerator]::Create()' not in launcher
    assert 'generated ephemerally for this launcher process' not in launcher
    assert 'Get-DotEnvValue -Key "MODEL_BRIDGE_ADMIN_TOKEN"' in launcher
    assert 'set "START_APP_EXIT_CODE=%ERRORLEVEL%"' in wrapper
    assert 'exit /b %START_APP_EXIT_CODE%' in wrapper
    assert '[string]$ApiBuildTag = "course-kg-api:dev"' in rebuild
    assert '[string]$WebBuildTag = "course-kg-web:dev"' in rebuild
    assert '$env:API_IMAGE = $ApiBuildTag' in rebuild
    assert '$env:WEB_IMAGE = $WebBuildTag' in rebuild
    assert '$buildArgs += @("api", "web")' in rebuild
    assert '$buildArgs += @("api", "worker", "web")' not in rebuild
    assert "worker uses the exact API image/tag" in rebuild
    assert "Docker build outputs require a mutable image tag" in rebuild
    assert 'set "REBUILD_IMAGES_EXIT_CODE=%ERRORLEVEL%"' in rebuild_wrapper
    assert 'exit /b %REBUILD_IMAGES_EXIT_CODE%' in rebuild_wrapper
