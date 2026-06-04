from pathlib import Path
import os
import re
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

APP_DIR = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = APP_DIR.parents[1]
INVALID_COURSE_DIR_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
HOT_RELOAD_SETTINGS = {
    "openai_api_key",
    "chat_base_url",
    "chat_resolve_ip",
    "embedding_base_url",
    "embedding_resolve_ip",
    "embedding_api_key",
    "model_bridge_enabled",
    "model_bridge_port",
    "embedding_model",
    "chat_model",
    "embedding_dimensions",
    "embedding_batch_size",
    "graph_extraction_strategy",
    "graph_extraction_soft_start_budget",
    "graph_extraction_max_input_tokens_per_run",
    "graph_extraction_max_model_calls_per_run",
    "graph_extraction_min_marginal_gain",
    "graph_extraction_stall_rounds",
    "graph_extraction_concurrency",
    "worker_concurrency",
    "ingestion_file_concurrency",
    "model_request_concurrency",
    "model_request_timeout_seconds",
    "hpo_concurrency",
    "graph_extraction_resume_batch_size",
    "enable_model_fallback",
    "retrieval_recall_k_default",
    "retrieval_recall_k_formula",
    "retrieval_layer_enabled",
    "retrieval_cache_ttl_seconds",
    "enable_agentic_reflection",
    "enable_post_generation_reflection",
    "citation_verification_sample_max",
    "reflection_max_retries",
    "reranker_enabled",
    "reranker_model",
    "reranker_max_length",
    "semantic_chunking_enabled",
    "semantic_chunking_min_length",
    "enable_auto_hpo",
    "enable_graph_community_summaries",
    "hpo_objective_mode",
    "hpo_judge_max_candidates",
    "hpo_judge_max_pairs",
    "hpo_judge_min_labels",
    "hpo_judge_max_tokens_per_pair",
    "hpo_judge_concurrency",
}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(WORKSPACE_ROOT / ".env", APP_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "Course Knowledge Base API"
    app_env: str = "development"
    app_port: int = 8000

    database_url: str = "sqlite:///./knowledge_base.db"
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "knowledge_chunks"
    redis_url: str = "redis://localhost:6379/0"
    ingestion_execution_mode: Literal["inline", "celery"] = "inline"
    ingestion_task_queue: str = "ingestion"
    enable_database_fallback: bool = False
    cors_origins: str = "http://localhost:3000,http://127.0.0.1:3000"
    api_keys: str = ""

    course_name: str = "Sample Course"
    data_root: Path = Field(default=WORKSPACE_ROOT / "data")
    storage_root: Path | None = None
    ingestion_root: Path | None = None

    openai_api_key: str | None = None
    chat_base_url: str = "https://api.openai.com/v1"
    chat_resolve_ip: str | None = None
    embedding_base_url: str = ""
    embedding_resolve_ip: str | None = None
    embedding_api_key: str | None = None
    model_bridge_enabled: bool = False
    model_bridge_port: int = 8765
    embedding_model: str = "text-embedding-v4"
    chat_model: str = "qwen-plus"
    embedding_dimensions: int = 1024
    embedding_batch_size: int = Field(default=10, ge=1, le=10)
    graph_extraction_strategy: str = "adaptive_best_first"
    graph_extraction_soft_start_budget: int | None = Field(default=24, ge=1)
    graph_extraction_max_input_tokens_per_run: int | None = Field(default=None, ge=1)
    graph_extraction_max_model_calls_per_run: int | None = Field(default=None, ge=1)
    graph_extraction_min_marginal_gain: float = Field(default=0.03, ge=0.0, le=1.0)
    graph_extraction_stall_rounds: int = Field(default=2, ge=1, le=20)
    graph_extraction_concurrency: int = Field(default=2, ge=1, le=8)
    worker_concurrency: int = Field(default=3, ge=1, le=32)
    ingestion_file_concurrency: int = Field(default=3, ge=1, le=8)
    model_request_concurrency: int = Field(default=3, ge=1, le=16)
    model_request_timeout_seconds: int = Field(default=240, ge=5, le=600)
    hpo_concurrency: int = Field(default=1, ge=1, le=8)
    graph_extraction_resume_batch_size: int = Field(default=6, ge=1, le=100)
    enable_model_fallback: bool = False
    retrieval_recall_k_default: int = Field(default=64, ge=1, le=200)
    retrieval_recall_k_formula: int = Field(default=80, ge=1, le=200)
    reranker_enabled: bool = False
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    reranker_max_length: int = Field(default=512, ge=64, le=2048)
    semantic_chunking_enabled: bool = False
    semantic_chunking_min_length: int = Field(default=2000, ge=500, le=5000)
    enable_auto_hpo: bool = Field(default=False)
    enable_graph_community_summaries: bool = Field(default=True)
    hpo_objective_mode: str = "judge_learned"
    hpo_judge_max_candidates: int = Field(default=10, ge=2, le=50)
    hpo_judge_max_pairs: int = Field(default=12, ge=1, le=100)
    hpo_judge_min_labels: int = Field(default=6, ge=1, le=100)
    hpo_judge_max_tokens_per_pair: int = Field(default=6000, ge=1000, le=50000)
    hpo_judge_concurrency: int = Field(default=1, ge=1, le=4)
    model_cache_root: Path = Field(default=WORKSPACE_ROOT / "models" / "huggingface")

    # Retrieval Layering & Agentic RAG
    retrieval_layer_enabled: bool = True
    retrieval_cache_ttl_seconds: int = 120
    enable_agentic_reflection: bool = True
    citation_verification_sample_max: int = 3
    reflection_max_retries: int = 2
    enable_post_generation_reflection: bool = False

    @property
    def cors_origin_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]

    @property
    def api_key_list(self) -> list[str]:
        return [key.strip() for key in self.api_keys.split(",") if key.strip()]

    def sanitize_course_dir_name(self, course_name: str) -> str:
        value = INVALID_COURSE_DIR_CHARS.sub("-", course_name).strip()
        value = re.sub(r"\s+", " ", value).rstrip(".")
        return value or "Course"

    def course_paths_for_name(self, course_name: str) -> dict[str, Path]:
        course_root = self.data_root / self.sanitize_course_dir_name(course_name)
        return {
            "course_root": course_root,
            "storage_root": course_root / "storage",
            "ingestion_root": course_root / "ingestion",
        }

    @property
    def course_data_root_path(self) -> Path:
        return self.course_paths_for_name(self.course_name)["course_root"]

    @property
    def storage_root_path(self) -> Path:
        return Path(self.storage_root) if self.storage_root else self.course_paths_for_name(self.course_name)["storage_root"]

    @property
    def ingestion_root_path(self) -> Path:
        return Path(self.ingestion_root) if self.ingestion_root else self.course_paths_for_name(self.course_name)["ingestion_root"]


_SETTINGS_CACHE: Settings | None = None
_SETTINGS_CACHE_TOKEN: tuple[tuple[int | None, int | None], ...] | None = None


def _settings_cache_token() -> tuple[tuple[int | None, int | None], ...]:
    token: list[tuple[int | None, int | None]] = []
    for path in (WORKSPACE_ROOT / ".env", APP_DIR / ".env"):
        try:
            stat = path.stat()
        except FileNotFoundError:
            token.append((None, None))
        else:
            token.append((stat.st_mtime_ns, stat.st_size))
    return tuple(token)


def _read_workspace_env() -> dict[str, str]:
    env_entries: dict[str, str] = {}
    env_path = WORKSPACE_ROOT / ".env"
    if env_path.exists():
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            stripped = raw_line.strip()
            if not stripped or stripped.startswith("#") or "=" not in raw_line:
                continue
            key, value = raw_line.split("=", 1)
            env_entries[key.strip().lstrip("\ufeff").upper()] = value
    return env_entries


def _apply_hot_reload_env(settings: Settings, env_entries: dict[str, str]) -> None:
    bool_fields = {
        "model_bridge_enabled",
        "enable_model_fallback",
        "reranker_enabled",
        "semantic_chunking_enabled",
        "retrieval_layer_enabled",
        "enable_agentic_reflection",
        "enable_post_generation_reflection",
        "enable_auto_hpo",
        "enable_graph_community_summaries",
    }
    int_fields = {
        "model_bridge_port",
        "embedding_dimensions",
        "embedding_batch_size",
        "graph_extraction_soft_start_budget",
        "graph_extraction_max_input_tokens_per_run",
        "graph_extraction_max_model_calls_per_run",
        "graph_extraction_stall_rounds",
        "graph_extraction_concurrency",
        "worker_concurrency",
        "ingestion_file_concurrency",
        "model_request_concurrency",
        "model_request_timeout_seconds",
        "hpo_concurrency",
        "graph_extraction_resume_batch_size",
        "retrieval_recall_k_default",
        "retrieval_recall_k_formula",
        "retrieval_cache_ttl_seconds",
        "citation_verification_sample_max",
        "reflection_max_retries",
        "reranker_max_length",
        "semantic_chunking_min_length",
        "hpo_judge_max_candidates",
        "hpo_judge_max_pairs",
        "hpo_judge_min_labels",
        "hpo_judge_max_tokens_per_pair",
        "hpo_judge_concurrency",
    }
    float_fields = {"graph_extraction_min_marginal_gain"}
    nullable_fields = {
        "chat_resolve_ip",
        "embedding_resolve_ip",
        "graph_extraction_soft_start_budget",
        "graph_extraction_max_input_tokens_per_run",
        "graph_extraction_max_model_calls_per_run",
    }
    aliases = {
        "GRAPH_EXTRACTION_CONCURRENCY": "graph_extraction_concurrency",
        "HPO_CONCURRENCY": "hpo_concurrency",
        "HPO_JUDGE_CONCURRENCY": "hpo_judge_concurrency",
        "INGESTION_FILE_CONCURRENCY": "ingestion_file_concurrency",
        "MODEL_REQUEST_CONCURRENCY": "model_request_concurrency",
        "MODEL_REQUEST_TIMEOUT_SECONDS": "model_request_timeout_seconds",
        "ENABLE_GRAPH_COMMUNITY_SUMMARIES": "enable_graph_community_summaries",
        "WORKER_CONCURRENCY": "worker_concurrency",
    }
    for env_key, value in env_entries.items():
        attr = aliases.get(env_key, env_key.lower())
        if attr not in HOT_RELOAD_SETTINGS:
            continue
        if value == "" and attr in nullable_fields:
            setattr(settings, attr, None)
            continue
        try:
            if attr in bool_fields:
                setattr(settings, attr, value.lower() in {"true", "1", "yes", "on"})
            elif attr in int_fields:
                if value != "":
                    setattr(settings, attr, int(value))
                    if attr == "hpo_concurrency":
                        settings.hpo_judge_concurrency = int(value)
            elif attr in float_fields:
                if value != "":
                    setattr(settings, attr, float(value))
            else:
                setattr(settings, attr, value)
        except ValueError:
            continue


def _build_settings() -> Settings:
    env_entries = _read_workspace_env()
    settings = Settings()
    _apply_hot_reload_env(settings, env_entries)

    api_chat_base_url = os.getenv("API_CHAT_BASE_URL")
    api_chat_resolve_ip = os.getenv("API_CHAT_RESOLVE_IP")
    model_bridge_enabled = str(env_entries.get("MODEL_BRIDGE_ENABLED") or os.getenv("MODEL_BRIDGE_ENABLED") or "").lower() in {
        "true",
        "1",
        "yes",
        "on",
    }
    model_bridge_port = env_entries.get("MODEL_BRIDGE_PORT") or os.getenv("MODEL_BRIDGE_PORT")
    if model_bridge_port:
        try:
            settings.model_bridge_port = int(model_bridge_port)
        except ValueError:
            pass
    settings.model_bridge_enabled = model_bridge_enabled
    if api_chat_base_url:
        settings.chat_base_url = api_chat_base_url
    elif model_bridge_enabled:
        settings.chat_base_url = f"http://host.docker.internal:{settings.model_bridge_port}"
        settings.chat_resolve_ip = "__none__"
    elif env_entries.get("CHAT_BASE_URL"):
        settings.chat_base_url = env_entries["CHAT_BASE_URL"]
    elif "CHAT_BASE_URL" in os.environ:
        settings.chat_base_url = os.getenv("CHAT_BASE_URL", "")
    if api_chat_resolve_ip is not None:
        settings.chat_resolve_ip = api_chat_resolve_ip
    elif model_bridge_enabled:
        settings.chat_resolve_ip = "__none__"
    elif os.getenv("CHAT_RESOLVE_IP"):
        settings.chat_resolve_ip = os.getenv("CHAT_RESOLVE_IP")
    elif env_entries.get("CHAT_RESOLVE_IP") is not None:
        settings.chat_resolve_ip = env_entries.get("CHAT_RESOLVE_IP")
    elif "CHAT_RESOLVE_IP" in os.environ:
        settings.chat_resolve_ip = os.getenv("CHAT_RESOLVE_IP")

    # Embedding-specific overrides (no fallback to chat model settings)
    embedding_base_url = env_entries.get("EMBEDDING_BASE_URL") or os.getenv("EMBEDDING_BASE_URL")
    if model_bridge_enabled:
        settings.embedding_base_url = f"http://host.docker.internal:{settings.model_bridge_port}"
        settings.embedding_resolve_ip = "__none__"
    elif embedding_base_url:
        settings.embedding_base_url = embedding_base_url
    elif env_entries.get("EMBEDDING_BASE_URL"):
        settings.embedding_base_url = env_entries["EMBEDDING_BASE_URL"]
    elif "EMBEDDING_BASE_URL" in os.environ:
        settings.embedding_base_url = ""

    embedding_resolve_ip = env_entries.get("EMBEDDING_RESOLVE_IP") or os.getenv("EMBEDDING_RESOLVE_IP")
    if model_bridge_enabled:
        settings.embedding_resolve_ip = "__none__"
    elif embedding_resolve_ip:
        settings.embedding_resolve_ip = embedding_resolve_ip
    elif env_entries.get("EMBEDDING_RESOLVE_IP") is not None:
        settings.embedding_resolve_ip = env_entries.get("EMBEDDING_RESOLVE_IP")
    elif "EMBEDDING_RESOLVE_IP" in os.environ:
        settings.embedding_resolve_ip = ""

    embedding_api_key = env_entries.get("EMBEDDING_API_KEY") or os.getenv("EMBEDDING_API_KEY")
    if embedding_api_key:
        settings.embedding_api_key = embedding_api_key
    elif env_entries.get("EMBEDDING_API_KEY"):
        settings.embedding_api_key = env_entries["EMBEDDING_API_KEY"]

    enable_auto_hpo = env_entries.get("ENABLE_AUTO_HPO") or os.getenv("ENABLE_AUTO_HPO")
    if enable_auto_hpo is not None:
        settings.enable_auto_hpo = str(enable_auto_hpo).lower() in {"true", "1", "yes", "on"}
    hpo_objective_mode = env_entries.get("HPO_OBJECTIVE_MODE") or os.getenv("HPO_OBJECTIVE_MODE")
    if hpo_objective_mode:
        settings.hpo_objective_mode = hpo_objective_mode
    for env_key, attr in {
        "HPO_JUDGE_MAX_CANDIDATES": "hpo_judge_max_candidates",
        "HPO_JUDGE_MAX_PAIRS": "hpo_judge_max_pairs",
        "HPO_JUDGE_MIN_LABELS": "hpo_judge_min_labels",
        "HPO_JUDGE_MAX_TOKENS_PER_PAIR": "hpo_judge_max_tokens_per_pair",
        "HPO_JUDGE_CONCURRENCY": "hpo_judge_concurrency",
        "HPO_CONCURRENCY": "hpo_concurrency",
        "WORKER_CONCURRENCY": "worker_concurrency",
        "INGESTION_FILE_CONCURRENCY": "ingestion_file_concurrency",
        "MODEL_REQUEST_CONCURRENCY": "model_request_concurrency",
        "MODEL_REQUEST_TIMEOUT_SECONDS": "model_request_timeout_seconds",
    }.items():
        raw_value = env_entries.get(env_key) or os.getenv(env_key)
        if raw_value is None:
            continue
        try:
            setattr(settings, attr, int(raw_value))
            if attr == "hpo_concurrency":
                settings.hpo_judge_concurrency = int(raw_value)
        except ValueError:
            pass

    settings.data_root.mkdir(parents=True, exist_ok=True)
    settings.course_data_root_path.mkdir(parents=True, exist_ok=True)
    settings.storage_root_path.mkdir(parents=True, exist_ok=True)
    settings.ingestion_root_path.mkdir(parents=True, exist_ok=True)
    return settings


def _clear_settings_cache() -> None:
    global _SETTINGS_CACHE, _SETTINGS_CACHE_TOKEN
    _SETTINGS_CACHE = None
    _SETTINGS_CACHE_TOKEN = None


def get_settings() -> Settings:
    global _SETTINGS_CACHE, _SETTINGS_CACHE_TOKEN
    token = _settings_cache_token()
    if _SETTINGS_CACHE is not None and _SETTINGS_CACHE_TOKEN == token:
        return _SETTINGS_CACHE
    settings = _build_settings()
    _SETTINGS_CACHE = settings
    _SETTINGS_CACHE_TOKEN = token
    return settings


get_settings.cache_clear = _clear_settings_cache  # type: ignore[attr-defined]
