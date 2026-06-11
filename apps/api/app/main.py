from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api import router
from app.db import ensure_schema
from app.core.config import get_settings
from app.services.ingestion import finalize_interrupted_batches
from app.services.runtime_settings import refresh_runtime_settings_if_needed


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    if settings.app_env.lower() == "production" and not settings.api_key_list:
        raise RuntimeError("API_KEYS must be configured when APP_ENV=production")
    ensure_schema()
    finalize_interrupted_batches()
    # 启动时预加载 reranker 模型（避免首次请求时阻塞事件循环导致健康检查失败）
    if settings.reranker_enabled:
        try:
            from app.services.reranker import _load_reranker
            _load_reranker()
        except Exception:
            pass  # 预加载失败不影响启动，首次请求时会重试
    yield


app = FastAPI(title="KnowledgeBase Knowledge Base API", version="0.2.0", lifespan=lifespan)
settings = get_settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(router, prefix="/api")


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
