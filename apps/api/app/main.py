from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.core.config import get_settings
from app.services.storage import (
    StorageDurabilityCapabilityError,
    ensure_storage_durability_ready,
    fail_closed_native_windows_production_before_settings,
)

# Settings loading is pure. The complete, actual DATA_ROOT gate (not merely
# the native-Windows shortcut) must finish before router/database imports can
# construct an engine, connect PostgreSQL, import a provider, or create a
# fallback file.
fail_closed_native_windows_production_before_settings()
from app.services.runtime_settings import initialize_runtime_env_from_root_file

_EARLY_RUNTIME_ENV_INITIALIZATION = initialize_runtime_env_from_root_file()
_EARLY_SETTINGS = get_settings()
_EARLY_STORAGE_DURABILITY_CAPABILITY = ensure_storage_durability_ready(
    settings=_EARLY_SETTINGS,
    force_probe=True,
)

from app.api import router
from app.db import ensure_schema
from app.services.ingestion import finalize_interrupted_batches
from app.services.runtime_settings import refresh_runtime_settings_if_needed, sync_model_bridge_runtime_config
from app.services.strategy_profiles import (
    reconcile_builtin_default_profile_startup,
    reconcile_profile_lifecycle_events_startup,
)
from app.services.storage_maintenance import (
    reconcile_pending_storage_maintenance_startup,
)
from app.services.upload_replacement import reconcile_pending_upload_replacements_startup


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.startup_ready = False
    try:
        settings = get_settings()
        storage_root_before_refresh = str(Path(settings.data_root).absolute())
        if settings.app_env.lower() == "production" and not settings.api_key_list:
            raise RuntimeError("API_KEYS must be configured when APP_ENV=production")
        app.state.storage_durability_capability = ensure_storage_durability_ready(settings=settings)
        ensure_schema()
        # Reload the root env without touching the bridge.  Only a second,
        # current-process DB/Redis/source admission may authorize the external
        # bridge configuration side effect.
        refresh_runtime_settings_if_needed(force=True, sync_bridge=False)
        settings = get_settings()
        if str(Path(settings.data_root).absolute()) != storage_root_before_refresh:
            app.state.storage_durability_capability = ensure_storage_durability_ready(
                settings=settings,
                force_probe=True,
            )
        sync_model_bridge_runtime_config(settings=settings)
        reconcile_builtin_default_profile_startup()
        profile_lifecycle_reconcile = reconcile_profile_lifecycle_events_startup()
        if profile_lifecycle_reconcile.get("failed"):
            raise RuntimeError(
                "Pending Profile lifecycle cache/version side effects could not be "
                "reconciled at API startup"
            )
        app.state.upload_replacement_recovery = (
            await reconcile_pending_upload_replacements_startup()
        )
        app.state.storage_maintenance_recovery = (
            await reconcile_pending_storage_maintenance_startup()
        )
        finalize_interrupted_batches()
        app.state.startup_ready = True
        yield
    finally:
        app.state.startup_ready = False


app = FastAPI(title="KnowledgeBase Knowledge Base API", version="0.2.0", lifespan=lifespan)
app.state.startup_ready = False
app.state.storage_durability_capability = _EARLY_STORAGE_DURABILITY_CAPABILITY
settings = get_settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(router, prefix="/api")


_SAFE_STORAGE_DURABILITY_REASONS = frozenset(
    {
        "native_windows_namespace_barrier_unproven",
        "test_adapter_forbidden_in_production",
        "filesystem_durability_probe_failed",
        "mount_identity_mismatch",
        "mount_identity_truncated",
        "mount_identity_unavailable",
        "path_identity_check_failed",
        "posix_dirfd_capability_unavailable",
        "probe_root_not_trusted",
        "storage_capability_inspection_failed",
        "storage_root_outside_data_root",
    }
)
_STORAGE_DURABILITY_ACTION = (
    "Provision a validated Linux managed volume for DATA_ROOT, then restart API and workers."
)


async def storage_durability_capability_error_response(
    _request: Request,
    exc: StorageDurabilityCapabilityError,
) -> JSONResponse:
    """Return a fixed, non-sensitive runtime failure contract."""

    raw_reason = exc.diagnostics.get("reason")
    reason = (
        raw_reason
        if isinstance(raw_reason, str) and raw_reason in _SAFE_STORAGE_DURABILITY_REASONS
        else "storage_durability_capability_unavailable"
    )
    return JSONResponse(
        status_code=503,
        content={
            "detail": {
                "code": StorageDurabilityCapabilityError.code,
                "title": "Storage durability capability unavailable",
                "message": "Storage mutation is disabled because its durability contract is not proven.",
                "reason": reason,
                "action": _STORAGE_DURABILITY_ACTION,
                "issues": [],
                "fix_commands": [],
                "retryable": False,
            }
        },
    )


app.add_exception_handler(
    StorageDurabilityCapabilityError,
    storage_durability_capability_error_response,
)


@app.middleware("http")
async def api_key_auth(request: Request, call_next):
    refresh_runtime_settings_if_needed()
    allowed_keys = get_settings().api_key_list
    if not allowed_keys:
        return await call_next(request)
    path = request.url.path
    if path in {"/api/health", "/docs", "/openapi.json", "/redoc"}:
        return await call_next(request)
    if request.method == "GET" and path.startswith("/api/ingestion/batches/") and path.endswith("/logs"):
        if request.query_params.get("token"):
            return await call_next(request)
    provided = request.headers.get("x-api-key")
    if not provided:
        authorization = request.headers.get("authorization", "")
        if authorization.lower().startswith("bearer "):
            provided = authorization[7:].strip()
    if provided not in allowed_keys:
        return JSONResponse({"detail": "Invalid or missing API key"}, status_code=401)
    return await call_next(request)
