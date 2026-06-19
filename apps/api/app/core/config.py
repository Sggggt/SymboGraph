from pathlib import Path
import os
import re
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

APP_DIR = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = APP_DIR.parents[1]
INVALID_KNOWLEDGE_BASE_DIR_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
HOT_RELOAD_SETTINGS = {
    "openai_api_key",
    "chat_base_url",
    "chat_resolve_ip",
    "embedding_base_url",
    "embedding_resolve_ip",
    "embedding_api_key",
    "model_bridge_enabled",
    "model_bridge_port",
    "model_bridge_admin_token",
    "embedding_model",
    "chat_model",
    "embedding_dimensions",
    "embedding_batch_size",
    "model_request_concurrency",
    "model_request_timeout_seconds",
    "fixed_chunk_size_tokens",
    "fixed_chunk_overlap_tokens",
    "context_package_token_budget",
    "enable_model_fallback",
    "mid_concept_extraction_max_model_batches",
    "mid_concept_extraction_max_candidates_per_batch",
    "mid_concept_extraction_max_tokens_per_batch",
    "mid_concept_candidate_keep_threshold",
    "rq_kmeans_levels",
    "rq_kmeans_max_k",
    "rq_residual_tau",
    "dense_knn_k_min",
    "dense_knn_k_max",
    "dense_reverse_b_min_base",
    "dense_reverse_b_max_base",
    "dense_reverse_b_min_doc",
    "dense_reverse_b_max_doc",
    "dense_reverse_b_min_lang",
    "dense_reverse_b_max_lang",
    "dense_min_cosine",
    "dense_strong_cosine",
    "cross_doc_out_quota_min",
    "cross_doc_out_quota_max",
    "cross_doc_min_cosine",
    "cross_language_out_quota_min",
    "cross_language_out_quota_max",
    "cross_language_min_cosine",
    "agent_coarse_total_budget",
    "agent_mid_per_coarse_budget",
    "agent_mid_top_k",
    "agent_chunk_per_mid_budget",
    "agent_chunk_top_k",
    "agent_max_depth_per_layer",
    "agent_max_labels_per_node",
    "agent_max_edge_reuse",
    "agent_max_cycle_reward_per_path",
    "agent_cycle_reward_distance_threshold",
    "agent_path_distance_green_threshold",
    "agent_path_distance_gray_threshold",
    "agent_path_distance_hard_threshold",
    "candidate_pool_dedupe_budget",
    "agent_structure_restore_budget",
    "context_path_summary_budget",
    "agent_planning_round_budget",
    "agent_max_typed_actions_per_round",
    "agent_repair_round_budget",
    "agent_verification_budget",
}


def running_in_container() -> bool:
    return Path("/.dockerenv").exists() or os.getenv("RUNNING_IN_DOCKER", "").strip().lower() in {"1", "true", "yes", "on"}


def model_bridge_client_base_url(port: int) -> str:
    host = "host.docker.internal" if running_in_container() else "127.0.0.1"
    return f"http://{host}:{int(port)}"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(WORKSPACE_ROOT / ".env", APP_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "KnowledgeBase Knowledge Base API"
    app_env: str = "development"
    app_port: int = 8000

    database_url: str = "sqlite:///./symbograph.db"
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "knowledge_chunks"
    redis_url: str = "redis://localhost:6379/0"
    ingestion_execution_mode: Literal["inline", "celery"] = "inline"
    ingestion_task_queue: str = "ingestion"
    enable_database_fallback: bool = False
    cors_origins: str = "http://localhost:3000,http://127.0.0.1:3000"
    api_keys: str = ""

    knowledge_base_name: str = "Sample KnowledgeBase"
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
    model_bridge_admin_token: str = "local-model-bridge-admin"
    embedding_model: str = "text-embedding-v4"
    chat_model: str = "qwen-plus"
    embedding_dimensions: int = 1024
    embedding_batch_size: int = Field(default=10, ge=1, le=10)
    worker_concurrency: int = Field(default=3, ge=1, le=32)
    model_request_concurrency: int = Field(default=3, ge=1, le=16)
    model_request_timeout_seconds: int = Field(default=240, ge=5, le=600)
    fixed_chunk_size_tokens: int = Field(default=512, ge=128, le=4096)
    fixed_chunk_overlap_tokens: int = Field(default=80, ge=0, le=1024)
    context_package_token_budget: int = Field(default=2400, ge=256, le=20000)
    enable_model_fallback: bool = False
    mid_concept_extraction_max_model_batches: int = Field(default=4, ge=0, le=64)
    mid_concept_extraction_max_candidates_per_batch: int = Field(default=8, ge=1, le=500)
    mid_concept_extraction_max_tokens_per_batch: int = Field(default=2400, ge=500, le=50000)
    mid_concept_candidate_keep_threshold: float = Field(default=0.62, ge=0.0, le=1.0)
    rq_kmeans_levels: int = Field(default=3, ge=1, le=8)
    rq_kmeans_max_k: int = Field(default=6, ge=1, le=64)
    rq_residual_tau: float = Field(default=0.65, gt=0.0, le=10.0)
    dense_knn_k_min: int = Field(default=5, ge=1, le=200)
    dense_knn_k_max: int = Field(default=16, ge=1, le=500)
    dense_reverse_b_min_base: int = Field(default=2, ge=1, le=200)
    dense_reverse_b_max_base: int = Field(default=12, ge=1, le=500)
    dense_reverse_b_min_doc: int = Field(default=1, ge=0, le=200)
    dense_reverse_b_max_doc: int = Field(default=8, ge=1, le=500)
    dense_reverse_b_min_lang: int = Field(default=1, ge=0, le=200)
    dense_reverse_b_max_lang: int = Field(default=8, ge=1, le=500)
    dense_min_cosine: float = Field(default=0.30, ge=-1.0, le=1.0)
    dense_strong_cosine: float = Field(default=0.72, ge=-1.0, le=1.0)
    cross_doc_out_quota_min: int = Field(default=1, ge=0, le=200)
    cross_doc_out_quota_max: int = Field(default=4, ge=1, le=500)
    cross_doc_min_cosine: float = Field(default=0.36, ge=-1.0, le=1.0)
    cross_language_out_quota_min: int = Field(default=1, ge=0, le=200)
    cross_language_out_quota_max: int = Field(default=4, ge=1, le=500)
    cross_language_min_cosine: float = Field(default=0.34, ge=-1.0, le=1.0)
    agent_coarse_total_budget: int = Field(default=8, ge=1, le=200)
    agent_mid_per_coarse_budget: int = Field(default=8, ge=1, le=500)
    agent_mid_top_k: int = Field(default=16, ge=1, le=500)
    agent_chunk_per_mid_budget: int = Field(default=8, ge=1, le=1000)
    agent_chunk_top_k: int = Field(default=80, ge=1, le=2000)
    agent_max_depth_per_layer: int = Field(default=3, ge=1, le=12)
    agent_max_labels_per_node: int = Field(default=3, ge=1, le=20)
    agent_max_edge_reuse: int = Field(default=2, ge=1, le=20)
    agent_max_cycle_reward_per_path: float = Field(default=0.18, ge=0.0, le=2.0)
    agent_cycle_reward_distance_threshold: float = Field(default=1.2, ge=0.0, le=20.0)
    agent_path_distance_green_threshold: float = Field(default=0.45, ge=0.0, le=20.0)
    agent_path_distance_gray_threshold: float = Field(default=1.35, ge=0.0, le=20.0)
    agent_path_distance_hard_threshold: float = Field(default=2.4, ge=0.0, le=40.0)
    candidate_pool_dedupe_budget: int = Field(default=1000, ge=1, le=20000)
    agent_structure_restore_budget: int = Field(default=16, ge=1, le=200)
    context_path_summary_budget: int = Field(default=32, ge=1, le=500)
    agent_planning_round_budget: int = Field(default=2, ge=1, le=10)
    agent_max_typed_actions_per_round: int = Field(default=8, ge=1, le=50)
    agent_repair_round_budget: int = Field(default=2, ge=0, le=10)
    agent_verification_budget: int = Field(default=8, ge=1, le=100)

    @property
    def cors_origin_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]

    @property
    def api_key_list(self) -> list[str]:
        return [key.strip() for key in self.api_keys.split(",") if key.strip()]

    def sanitize_knowledge_base_dir_name(self, knowledge_base_name: str) -> str:
        value = INVALID_KNOWLEDGE_BASE_DIR_CHARS.sub("-", knowledge_base_name).strip()
        value = re.sub(r"\s+", " ", value).rstrip(".")
        return value or "KnowledgeBase"

    def knowledge_base_paths_for_name(self, knowledge_base_name: str) -> dict[str, Path]:
        knowledge_base_root = self.data_root / self.sanitize_knowledge_base_dir_name(knowledge_base_name)
        return {
            "knowledge_base_root": knowledge_base_root,
            "storage_root": knowledge_base_root / "storage",
            "ingestion_root": knowledge_base_root / "ingestion",
        }

    @property
    def knowledge_base_data_root_path(self) -> Path:
        return self.knowledge_base_paths_for_name(self.knowledge_base_name)["knowledge_base_root"]

    @property
    def storage_root_path(self) -> Path:
        return Path(self.storage_root) if self.storage_root else self.knowledge_base_paths_for_name(self.knowledge_base_name)["storage_root"]

    @property
    def ingestion_root_path(self) -> Path:
        return Path(self.ingestion_root) if self.ingestion_root else self.knowledge_base_paths_for_name(self.knowledge_base_name)["ingestion_root"]


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
    }
    int_fields = {
        "model_bridge_port",
        "embedding_dimensions",
        "embedding_batch_size",
        "model_request_concurrency",
        "model_request_timeout_seconds",
        "fixed_chunk_size_tokens",
        "fixed_chunk_overlap_tokens",
        "context_package_token_budget",
        "mid_concept_extraction_max_model_batches",
        "mid_concept_extraction_max_candidates_per_batch",
        "mid_concept_extraction_max_tokens_per_batch",
        "rq_kmeans_levels",
        "rq_kmeans_max_k",
        "dense_knn_k_min",
        "dense_knn_k_max",
        "dense_reverse_b_min_base",
        "dense_reverse_b_max_base",
        "dense_reverse_b_min_doc",
        "dense_reverse_b_max_doc",
        "dense_reverse_b_min_lang",
        "dense_reverse_b_max_lang",
        "cross_doc_out_quota_min",
        "cross_doc_out_quota_max",
        "cross_language_out_quota_min",
        "cross_language_out_quota_max",
        "agent_coarse_total_budget",
        "agent_mid_per_coarse_budget",
        "agent_mid_top_k",
        "agent_chunk_per_mid_budget",
        "agent_chunk_top_k",
        "agent_max_depth_per_layer",
        "agent_max_labels_per_node",
        "agent_max_edge_reuse",
        "candidate_pool_dedupe_budget",
        "agent_structure_restore_budget",
        "context_path_summary_budget",
        "agent_planning_round_budget",
        "agent_max_typed_actions_per_round",
        "agent_repair_round_budget",
        "agent_verification_budget",
    }
    float_fields: set[str] = {
        "mid_concept_candidate_keep_threshold",
        "rq_residual_tau",
        "dense_min_cosine",
        "dense_strong_cosine",
        "cross_doc_min_cosine",
        "cross_language_min_cosine",
        "agent_max_cycle_reward_per_path",
        "agent_cycle_reward_distance_threshold",
        "agent_path_distance_green_threshold",
        "agent_path_distance_gray_threshold",
        "agent_path_distance_hard_threshold",
    }
    nullable_fields = {
        "chat_resolve_ip",
        "embedding_resolve_ip",
    }
    aliases = {
        "MODEL_REQUEST_CONCURRENCY": "model_request_concurrency",
        "MODEL_REQUEST_TIMEOUT_SECONDS": "model_request_timeout_seconds",
        "FIXED_CHUNK_SIZE_TOKENS": "fixed_chunk_size_tokens",
        "FIXED_CHUNK_OVERLAP_TOKENS": "fixed_chunk_overlap_tokens",
        "CONTEXT_PACKAGE_TOKEN_BUDGET": "context_package_token_budget",
        "MID_CONCEPT_EXTRACTION_MAX_MODEL_BATCHES": "mid_concept_extraction_max_model_batches",
        "MID_CONCEPT_EXTRACTION_MAX_CANDIDATES_PER_BATCH": "mid_concept_extraction_max_candidates_per_batch",
        "MID_CONCEPT_EXTRACTION_MAX_TOKENS_PER_BATCH": "mid_concept_extraction_max_tokens_per_batch",
        "MID_CONCEPT_CANDIDATE_KEEP_THRESHOLD": "mid_concept_candidate_keep_threshold",
        "RQ_KMEANS_LEVELS": "rq_kmeans_levels",
        "RQ_KMEANS_MAX_K": "rq_kmeans_max_k",
        "RQ_RESIDUAL_TAU": "rq_residual_tau",
        "DENSE_KNN_K_MIN": "dense_knn_k_min",
        "DENSE_KNN_K_MAX": "dense_knn_k_max",
        "DENSE_REVERSE_B_MIN_BASE": "dense_reverse_b_min_base",
        "DENSE_REVERSE_B_MAX_BASE": "dense_reverse_b_max_base",
        "DENSE_REVERSE_B_MIN_DOC": "dense_reverse_b_min_doc",
        "DENSE_REVERSE_B_MAX_DOC": "dense_reverse_b_max_doc",
        "DENSE_REVERSE_B_MIN_LANG": "dense_reverse_b_min_lang",
        "DENSE_REVERSE_B_MAX_LANG": "dense_reverse_b_max_lang",
        "DENSE_MIN_COSINE": "dense_min_cosine",
        "DENSE_STRONG_COSINE": "dense_strong_cosine",
        "CROSS_DOC_OUT_QUOTA_MIN": "cross_doc_out_quota_min",
        "CROSS_DOC_OUT_QUOTA_MAX": "cross_doc_out_quota_max",
        "CROSS_DOC_MIN_COSINE": "cross_doc_min_cosine",
        "CROSS_LANGUAGE_OUT_QUOTA_MIN": "cross_language_out_quota_min",
        "CROSS_LANGUAGE_OUT_QUOTA_MAX": "cross_language_out_quota_max",
        "CROSS_LANGUAGE_MIN_COSINE": "cross_language_min_cosine",
        "AGENT_COARSE_TOTAL_BUDGET": "agent_coarse_total_budget",
        "AGENT_MID_PER_COARSE_BUDGET": "agent_mid_per_coarse_budget",
        "AGENT_MID_TOP_K": "agent_mid_top_k",
        "AGENT_CHUNK_PER_MID_BUDGET": "agent_chunk_per_mid_budget",
        "AGENT_CHUNK_TOP_K": "agent_chunk_top_k",
        "AGENT_MAX_DEPTH_PER_LAYER": "agent_max_depth_per_layer",
        "AGENT_MAX_LABELS_PER_NODE": "agent_max_labels_per_node",
        "AGENT_MAX_EDGE_REUSE": "agent_max_edge_reuse",
        "AGENT_MAX_CYCLE_REWARD_PER_PATH": "agent_max_cycle_reward_per_path",
        "AGENT_CYCLE_REWARD_DISTANCE_THRESHOLD": "agent_cycle_reward_distance_threshold",
        "AGENT_PATH_DISTANCE_GREEN_THRESHOLD": "agent_path_distance_green_threshold",
        "AGENT_PATH_DISTANCE_GRAY_THRESHOLD": "agent_path_distance_gray_threshold",
        "AGENT_PATH_DISTANCE_HARD_THRESHOLD": "agent_path_distance_hard_threshold",
        "CANDIDATE_POOL_DEDUPE_BUDGET": "candidate_pool_dedupe_budget",
        "AGENT_STRUCTURE_RESTORE_BUDGET": "agent_structure_restore_budget",
        "CONTEXT_PATH_SUMMARY_BUDGET": "context_path_summary_budget",
        "AGENT_PLANNING_ROUND_BUDGET": "agent_planning_round_budget",
        "AGENT_MAX_TYPED_ACTIONS_PER_ROUND": "agent_max_typed_actions_per_round",
        "AGENT_REPAIR_ROUND_BUDGET": "agent_repair_round_budget",
        "AGENT_VERIFICATION_BUDGET": "agent_verification_budget",
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
    model_bridge_admin_token = env_entries.get("MODEL_BRIDGE_ADMIN_TOKEN") or os.getenv("MODEL_BRIDGE_ADMIN_TOKEN")
    if model_bridge_admin_token:
        settings.model_bridge_admin_token = model_bridge_admin_token
    settings.model_bridge_enabled = model_bridge_enabled
    if api_chat_base_url:
        settings.chat_base_url = api_chat_base_url
    elif model_bridge_enabled:
        settings.chat_base_url = model_bridge_client_base_url(settings.model_bridge_port)
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
        settings.embedding_base_url = model_bridge_client_base_url(settings.model_bridge_port)
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

    for env_key, attr in {
        "WORKER_CONCURRENCY": "worker_concurrency",
        "MODEL_REQUEST_CONCURRENCY": "model_request_concurrency",
        "MODEL_REQUEST_TIMEOUT_SECONDS": "model_request_timeout_seconds",
        "FIXED_CHUNK_SIZE_TOKENS": "fixed_chunk_size_tokens",
        "FIXED_CHUNK_OVERLAP_TOKENS": "fixed_chunk_overlap_tokens",
        "CONTEXT_PACKAGE_TOKEN_BUDGET": "context_package_token_budget",
    }.items():
        raw_value = env_entries.get(env_key) or os.getenv(env_key)
        if raw_value is None:
            continue
        try:
            setattr(settings, attr, int(raw_value))
        except ValueError:
            pass

    settings.data_root.mkdir(parents=True, exist_ok=True)
    settings.knowledge_base_data_root_path.mkdir(parents=True, exist_ok=True)
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
