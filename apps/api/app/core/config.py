from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
import hashlib
import os
import re
from typing import Literal
from uuid import UUID

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

APP_DIR = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = APP_DIR.parents[1]
INVALID_KNOWLEDGE_BASE_DIR_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
KNOWLEDGE_BASE_STORAGE_IDENTITY_PROTOCOL = "knowledge_base_uuid_root_v1"
ACTIVE_RQ_KMEANS_LEVELS = 3
EDGE_DISTANCE_PROTOCOL_DEFAULT = "edge_distance_log_calibrated_strength_v2"
EDGE_DISTANCE_PROTOCOL_ALLOWLIST = frozenset({EDGE_DISTANCE_PROTOCOL_DEFAULT})
RQ_MEMBERSHIP_PROTOCOL_DEFAULT = "rq_primary_chain_v1"
RQ_MEMBERSHIP_PROTOCOL_ALLOWLIST = frozenset({RQ_MEMBERSHIP_PROTOCOL_DEFAULT})
EDGE_PROJECTION_PROTOCOL_DEFAULT = "membership_q15_layer_type_calibrated_v3"
EDGE_PROJECTION_PROTOCOL_ALLOWLIST = frozenset({EDGE_PROJECTION_PROTOCOL_DEFAULT})
EDGE_TYPE_CALIBRATION_PROTOCOL_DEFAULT = "type_local_winsorized_minmax_v1"
EDGE_TYPE_CALIBRATION_PROTOCOL_ALLOWLIST = frozenset(
    {EDGE_TYPE_CALIBRATION_PROTOCOL_DEFAULT}
)
GRAY_ZONE_RULE_PROTOCOL_DEFAULT = "deterministic_support_progress_v1"
GRAY_ZONE_RULE_PROTOCOL_ALLOWLIST = frozenset({GRAY_ZONE_RULE_PROTOCOL_DEFAULT})
GRAY_ZONE_OBSERVATION_CADENCE_MAX = 16
TRAVERSAL_OBSERVATION_BUDGET_MAX = 20_000
QUERY_FACET_POSTERIOR_OBSERVATION_BUDGET_MAX = 1_024
QUERY_FACET_POSTERIOR_ROUND_BUDGET_MAX = 2
MODEL_API_PROTOCOL_ALLOWLIST = frozenset({"openai", "anthropic"})
EMBEDDING_API_PROTOCOL_ALLOWLIST = frozenset({"openai"})
HOT_RELOAD_SETTINGS = {
    "chat_api_key",
    "chat_api_protocol",
    "chat_base_url",
    "chat_resolve_ip",
    "chat_model",
    "graph_api_key",
    "embedding_api_key",
    "embedding_batch_size",
    "model_request_concurrency",
    "model_request_timeout_seconds",
    "chat_json_max_tokens",
    "agent_request_concurrency",
    "source_io_concurrency",
    "agent_request_queue_limit",
    "agent_request_queue_timeout_seconds",
    "agent_request_lease_ttl_seconds",
    "context_package_token_budget",
    "upload_max_bytes",
    "enable_model_fallback",
    "concept_i18n_enabled",
    "query_facet_bilingual_enabled",
    "query_facet_posterior_enabled",
    "query_facet_posterior_observation_budget",
    "query_facet_posterior_round_budget",
    "query_facet_posterior_convergence_epsilon",
    "enable_auto_tpe",
    "tpe_trial_budget",
    "tpe_startup_random_trials",
    "tpe_good_quantile_gamma",
    "tpe_probe_query_budget",
    "tpe_trial_timeout_seconds",
    "tpe_candidate_pool_size",
    "operating_point_hard_gate_max_edge_density",
    "operating_point_hard_gate_max_isolated_ratio",
    "operating_point_hard_gate_max_hubness_ratio",
    "operating_point_hard_gate_min_structure_recovery_rate",
    "operating_point_hard_gate_max_candidate_latency_p95_ms",
    "retrieval_result_top_k_default",
    "agent_coarse_initial_budget",
    "agent_coarse_total_budget",
    "agent_coarse_top_k",
    "agent_mid_per_coarse_budget",
    "agent_coarse_drilldown_mid_initial_budget",
    "agent_mid_initial_budget",
    "agent_mid_top_k",
    "agent_chunk_per_mid_budget",
    "agent_chunk_initial_budget",
    "agent_chunk_top_k",
    "agent_max_depth_per_layer",
    "agent_max_labels_per_node",
    "agent_max_edge_reuse",
    "agent_max_cycle_reward_per_path",
    "agent_cycle_reward_distance_threshold",
    "agent_path_distance_green_threshold",
    "agent_path_distance_gray_threshold",
    "agent_path_distance_hard_threshold",
    "gray_zone_rule_protocol",
    "gray_zone_observation_cadence",
    "traversal_observation_budget",
    "candidate_pool_dedupe_budget",
    "agent_structure_restore_per_chunk_budget",
    "agent_structure_restore_budget",
    "context_path_summary_budget",
    "agent_planning_round_budget",
    "agent_max_typed_actions_per_round",
    "agent_repair_round_budget",
    "agent_verification_budget",
}

REBUILD_REQUIRED_SETTINGS = {
    "fixed_chunk_size_tokens",
    "fixed_chunk_overlap_tokens",
    "embedding_base_url",
    "embedding_api_protocol",
    "embedding_resolve_ip",
    "embedding_model",
    "embedding_dimensions",
    "graph_base_url",
    "graph_api_protocol",
    "graph_resolve_ip",
    "graph_model",
    "edge_distance_protocol",
    "rq_membership_protocol",
    "edge_projection_protocol",
    "edge_type_calibration_protocol",
    "rq_kmeans_max_k",
    "rq_residual_tau",
    "rq_membership_temperature",
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
    "mid_concept_extraction_max_model_batches",
    "mid_concept_extraction_max_candidates_per_batch",
    "mid_concept_extraction_max_tokens_per_batch",
    "mid_concept_candidate_keep_threshold",
}

SERVICE_RECREATE_REQUIRED_SETTINGS = {
    "worker_concurrency",
    "model_bridge_enabled",
    "model_bridge_port",
    "model_bridge_admin_token",
}

PROCESS_ONLY_ENV_KEYS = frozenset({"MODEL_BRIDGE_ADMIN_TOKEN"})

RUNTIME_ENV_SETTINGS = (
    HOT_RELOAD_SETTINGS
    | REBUILD_REQUIRED_SETTINGS
    | SERVICE_RECREATE_REQUIRED_SETTINGS
)


def validate_path_distance_thresholds(
    green: float,
    gray: float,
    hard: float,
) -> tuple[float, float, float]:
    normalized = (float(green), float(gray), float(hard))
    if not (
        0.0 <= normalized[0] <= 20.0
        and 0.0 <= normalized[1] <= 20.0
        and 0.0 <= normalized[2] <= 40.0
    ):
        raise ValueError(
            "path distance thresholds must stay within green/gray 0..20 and hard 0..40"
        )
    if not normalized[0] <= normalized[1] <= normalized[2]:
        raise ValueError(
            "path distance thresholds must satisfy "
            "agent_path_distance_green_threshold <= "
            "agent_path_distance_gray_threshold <= "
            "agent_path_distance_hard_threshold"
        )
    return normalized


def running_in_container() -> bool:
    return Path("/.dockerenv").exists() or os.getenv("RUNNING_IN_DOCKER", "").strip().lower() in {"1", "true", "yes", "on"}


def model_bridge_client_base_url(port: int) -> str:
    default_host = "host.docker.internal" if running_in_container() else "127.0.0.1"
    host = str(os.getenv("MODEL_BRIDGE_CLIENT_HOST") or default_host).strip().casefold()
    allowed_hosts = {"127.0.0.1", "localhost", "host.docker.internal"}
    if running_in_container():
        # ``model-bridge`` is the fixed private Compose service name.  This
        # process-only wiring value is deliberately not a mutable Runtime
        # Setting: changing service topology requires container recreation.
        allowed_hosts.add("model-bridge")
    if host not in allowed_hosts:
        raise ValueError("MODEL_BRIDGE_CLIENT_HOST is not an allowlisted local bridge host")
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
    upload_max_bytes: int = Field(default=100 * 1024 * 1024, ge=1, le=10 * 1024 * 1024 * 1024)

    chat_api_key: str | None = None
    chat_api_protocol: Literal["openai", "anthropic"] = "openai"
    chat_base_url: str = ""
    chat_resolve_ip: str | None = None
    graph_api_key: str | None = None
    graph_api_protocol: Literal["openai", "anthropic"] = "openai"
    graph_base_url: str = ""
    graph_resolve_ip: str | None = None
    embedding_api_protocol: Literal["openai"] = "openai"
    embedding_base_url: str = ""
    embedding_resolve_ip: str | None = None
    embedding_api_key: str | None = None
    model_bridge_enabled: bool = False
    model_bridge_port: int = 8765
    model_bridge_admin_token: str = ""
    embedding_model: str = "fixture-embedding-model"
    chat_model: str = "fixture-chat-model"
    graph_model: str = "fixture-graph-model"
    embedding_dimensions: int = 1024
    embedding_batch_size: int = Field(default=10, ge=1, le=10)
    worker_concurrency: int = Field(default=3, ge=1, le=32)
    model_request_concurrency: int = Field(default=3, ge=1, le=16)
    model_request_timeout_seconds: int = Field(default=240, ge=5, le=600)
    chat_json_max_tokens: int = Field(default=12000, ge=256, le=32768)
    agent_request_concurrency: int = Field(default=4, ge=1, le=128)
    source_io_concurrency: int = Field(default=4, ge=1, le=64)
    agent_request_queue_limit: int = Field(default=8, ge=0, le=1000)
    agent_request_queue_timeout_seconds: int = Field(default=30, ge=1, le=3600)
    agent_request_lease_ttl_seconds: int = Field(default=300, ge=5, le=7200)
    fixed_chunk_size_tokens: int = Field(default=512, ge=128, le=4096)
    fixed_chunk_overlap_tokens: int = Field(default=80, ge=0, le=1024)
    context_package_token_budget: int = Field(default=2400, ge=256, le=20000)
    retrieval_result_top_k_default: int = Field(default=8, ge=1, le=50)
    enable_model_fallback: bool = False
    concept_i18n_enabled: bool = False
    query_facet_bilingual_enabled: bool = False
    query_facet_posterior_enabled: bool = True
    query_facet_posterior_observation_budget: int = Field(
        default=64,
        ge=1,
        le=QUERY_FACET_POSTERIOR_OBSERVATION_BUDGET_MAX,
    )
    query_facet_posterior_round_budget: int = Field(
        default=2,
        ge=1,
        le=QUERY_FACET_POSTERIOR_ROUND_BUDGET_MAX,
    )
    query_facet_posterior_convergence_epsilon: float = Field(
        default=0.02,
        ge=0.0,
        le=1.0,
    )
    mid_concept_extraction_max_model_batches: int = Field(default=4, ge=0, le=64)
    mid_concept_extraction_max_candidates_per_batch: int = Field(default=8, ge=1, le=500)
    mid_concept_extraction_max_tokens_per_batch: int = Field(default=2400, ge=500, le=50000)
    mid_concept_candidate_keep_threshold: float = Field(default=0.62, ge=0.0, le=1.0)
    rq_kmeans_levels: int = ACTIVE_RQ_KMEANS_LEVELS
    edge_distance_protocol: Literal[
        "edge_distance_log_calibrated_strength_v2"
    ] = EDGE_DISTANCE_PROTOCOL_DEFAULT
    rq_membership_protocol: Literal[
        "rq_primary_chain_v1"
    ] = RQ_MEMBERSHIP_PROTOCOL_DEFAULT
    edge_projection_protocol: Literal[
        "membership_q15_layer_type_calibrated_v3"
    ] = EDGE_PROJECTION_PROTOCOL_DEFAULT
    edge_type_calibration_protocol: Literal[
        "type_local_winsorized_minmax_v1"
    ] = EDGE_TYPE_CALIBRATION_PROTOCOL_DEFAULT
    rq_kmeans_max_k: int = Field(default=6, ge=1, le=6)
    rq_residual_tau: float = Field(default=0.65, gt=0.0, le=10.0)
    rq_membership_temperature: float = Field(default=0.35, gt=0.0, le=10.0)
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
    enable_auto_tpe: bool = False
    tpe_trial_budget: int = Field(default=6, ge=1, le=200)
    tpe_startup_random_trials: int = Field(default=3, ge=1, le=100)
    tpe_good_quantile_gamma: float = Field(default=0.25, gt=0.0, lt=1.0)
    tpe_probe_query_budget: int = Field(default=6, ge=1, le=200)
    tpe_trial_timeout_seconds: int = Field(default=30, ge=1, le=3600)
    tpe_candidate_pool_size: int = Field(default=24, ge=1, le=500)
    operating_point_hard_gate_max_edge_density: float = Field(
        default=0.45,
        gt=0.0,
        le=1.0,
    )
    operating_point_hard_gate_max_isolated_ratio: float = Field(default=0.35, ge=0.0, le=1.0)
    operating_point_hard_gate_max_hubness_ratio: float = Field(default=12.0, ge=1.0, le=1000.0)
    operating_point_hard_gate_min_structure_recovery_rate: float = Field(default=0.25, ge=0.0, le=1.0)
    operating_point_hard_gate_max_candidate_latency_p95_ms: int = Field(default=30000, ge=10, le=600000)
    agent_coarse_initial_budget: int | None = Field(default=None, ge=1, le=200)
    agent_coarse_total_budget: int = Field(default=8, ge=1, le=200)
    agent_coarse_top_k: int | None = Field(default=None, ge=1, le=200)
    agent_mid_per_coarse_budget: int = Field(default=8, ge=1, le=500)
    agent_coarse_drilldown_mid_initial_budget: int | None = Field(default=None, ge=1, le=500)
    agent_mid_initial_budget: int | None = Field(default=None, ge=1, le=500)
    agent_mid_top_k: int = Field(default=16, ge=1, le=500)
    agent_chunk_per_mid_budget: int = Field(default=8, ge=1, le=1000)
    agent_chunk_initial_budget: int | None = Field(default=None, ge=1, le=2000)
    agent_chunk_top_k: int = Field(default=80, ge=1, le=2000)
    agent_max_depth_per_layer: int = Field(default=3, ge=1, le=12)
    agent_max_labels_per_node: int = Field(default=3, ge=1, le=20)
    agent_max_edge_reuse: int = Field(default=2, ge=1, le=20)
    agent_max_cycle_reward_per_path: float = Field(default=0.18, ge=0.0, le=2.0)
    agent_cycle_reward_distance_threshold: float = Field(default=1.2, ge=0.0, le=20.0)
    agent_path_distance_green_threshold: float = Field(default=0.45, ge=0.0, le=20.0)
    agent_path_distance_gray_threshold: float = Field(default=1.35, ge=0.0, le=20.0)
    agent_path_distance_hard_threshold: float = Field(default=2.4, ge=0.0, le=40.0)
    gray_zone_rule_protocol: Literal["deterministic_support_progress_v1"] = GRAY_ZONE_RULE_PROTOCOL_DEFAULT
    gray_zone_observation_cadence: int = Field(
        default=1,
        ge=1,
        le=GRAY_ZONE_OBSERVATION_CADENCE_MAX,
    )
    traversal_observation_budget: int = Field(
        default=64,
        ge=1,
        le=TRAVERSAL_OBSERVATION_BUDGET_MAX,
    )
    candidate_pool_dedupe_budget: int = Field(default=1000, ge=1, le=20000)
    agent_structure_restore_per_chunk_budget: int | None = Field(default=None, ge=1, le=200)
    agent_structure_restore_budget: int = Field(default=16, ge=1, le=200)
    context_path_summary_budget: int = Field(default=32, ge=1, le=500)
    agent_planning_round_budget: int = Field(default=2, ge=1, le=10)
    agent_max_typed_actions_per_round: int = Field(default=8, ge=1, le=50)
    agent_repair_round_budget: int = Field(default=2, ge=0, le=10)
    agent_verification_budget: int = Field(default=8, ge=1, le=100)

    @field_validator("rq_kmeans_levels")
    @classmethod
    def validate_fixed_rq_kmeans_levels(cls, value: int) -> int:
        if value != ACTIVE_RQ_KMEANS_LEVELS:
            raise ValueError("RQ_KMEANS_LEVELS must be the fixed active protocol depth 3")
        return ACTIVE_RQ_KMEANS_LEVELS

    @model_validator(mode="after")
    def validate_path_distance_threshold_order(self) -> "Settings":
        validate_path_distance_thresholds(
            self.agent_path_distance_green_threshold,
            self.agent_path_distance_gray_threshold,
            self.agent_path_distance_hard_threshold,
        )
        return self

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
        """Return the legacy name-derived path used only by singleton defaults.

        Active knowledge-base operations must use ``knowledge_base_paths_for_id``.
        A sanitized display name is not an ownership identity because distinct
        names can collapse onto the same filesystem spelling.
        """

        knowledge_base_root = self.data_root / self.sanitize_knowledge_base_dir_name(knowledge_base_name)
        return {
            "knowledge_base_root": knowledge_base_root,
            "storage_root": knowledge_base_root / "storage",
            "ingestion_root": knowledge_base_root / "ingestion",
        }

    def knowledge_base_storage_key(self, knowledge_base_id: str) -> str:
        try:
            identity = UUID(str(knowledge_base_id))
        except (TypeError, ValueError, AttributeError) as exc:
            raise ValueError("knowledge_base_id must be a canonical UUID") from exc
        return f"kb_{identity.hex}"

    def knowledge_base_paths_for_id(self, knowledge_base_id: str) -> dict[str, Path]:
        knowledge_base_root = (
            self.data_root
            / "knowledge_bases"
            / self.knowledge_base_storage_key(knowledge_base_id)
        )
        return {
            "knowledge_base_root": knowledge_base_root,
            "storage_root": knowledge_base_root / "storage",
            "ingestion_root": knowledge_base_root / "ingestion",
        }

    def knowledge_base_paths_for_source_root(
        self,
        source_root: str | Path,
    ) -> dict[str, Path]:
        """Resolve the persisted, DB-owned storage root for an existing KB."""

        storage_root = Path(source_root)
        if not storage_root.is_absolute():
            storage_root = self.data_root / storage_root
        lexical_data_root = Path(os.path.abspath(self.data_root))
        lexical_storage_root = Path(os.path.abspath(storage_root))
        try:
            lexical_storage_root.relative_to(lexical_data_root)
        except ValueError as exc:
            raise ValueError(
                "Persisted knowledge-base source_root is outside DATA_ROOT"
            ) from exc
        knowledge_base_root = lexical_storage_root.parent
        return {
            "knowledge_base_root": knowledge_base_root,
            "storage_root": lexical_storage_root,
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
_SETTINGS_CACHE_TOKEN: tuple[tuple[object, ...], ...] | None = None
_SETTINGS_CACHE_CONTENT_HASH_KEY = os.urandom(32)
_SETTINGS_OVERRIDE: ContextVar[Settings | None] = ContextVar(
    "runtime_settings_candidate_override",
    default=None,
)


def _active_runtime_env_path() -> Path:
    configured = os.environ.get("RUNTIME_ENV_FILE", "").strip()
    return Path(configured) if configured else WORKSPACE_ROOT / ".env"


def _settings_file_cache_identity(path: Path) -> tuple[object, ...]:
    """Return a secret-free strong identity for one settings source."""

    normalized_path = str(Path(os.path.abspath(path)))
    try:
        with path.open("rb") as handle:
            opened = os.fstat(handle.fileno())
            digest = hashlib.blake2b(
                key=_SETTINGS_CACHE_CONTENT_HASH_KEY,
                digest_size=32,
            )
            for block in iter(lambda: handle.read(64 * 1024), b""):
                digest.update(block)
            after_read = os.fstat(handle.fileno())
    except FileNotFoundError:
        return (normalized_path, "missing")
    return (
        normalized_path,
        "file",
        int(opened.st_dev),
        int(opened.st_ino),
        int(opened.st_size),
        int(opened.st_mtime_ns),
        int(opened.st_ctime_ns),
        int(after_read.st_size),
        int(after_read.st_mtime_ns),
        int(after_read.st_ctime_ns),
        digest.hexdigest(),
    )


def _settings_cache_token() -> tuple[tuple[object, ...], ...]:
    token: list[tuple[object, ...]] = []
    configured_sources = Settings.model_config.get("env_file") or ()
    if isinstance(configured_sources, (str, os.PathLike)):
        configured_sources = (configured_sources,)
    seen_paths: set[str] = set()
    for raw_path in (_active_runtime_env_path(), *configured_sources):
        path = Path(raw_path)
        normalized_path = str(Path(os.path.abspath(path)))
        if normalized_path in seen_paths:
            continue
        seen_paths.add(normalized_path)
        token.append(_settings_file_cache_identity(path))
    return tuple(token)


def _deserialize_workspace_env_value(value: str) -> str:
    """Decode the exact quoted form emitted by the managed env writer."""

    text = str(value)
    if (
        len(text) < 2
        or text[0] not in {'"', "'"}
        or text[-1] != text[0]
    ):
        return text
    body = text[1:-1]
    if text[0] == "'":
        return body
    decoded: list[str] = []
    escaped = False
    for character in body:
        if escaped:
            decoded.append(character)
            escaped = False
        elif character == "\\":
            escaped = True
        else:
            decoded.append(character)
    if escaped:
        decoded.append("\\")
    return "".join(decoded)


def _read_workspace_env() -> dict[str, str]:
    env_entries: dict[str, str] = {}
    env_path = _active_runtime_env_path()
    if env_path.exists():
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            stripped = raw_line.strip()
            if not stripped or stripped.startswith("#") or "=" not in raw_line:
                continue
            key, value = raw_line.split("=", 1)
            env_entries[key.strip().lstrip("\ufeff").upper()] = (
                _deserialize_workspace_env_value(value)
            )
    return env_entries


def _validate_fixed_protocol_env(env_entries: dict[str, str]) -> None:
    for env_key in ("CHAT_API_PROTOCOL", "GRAPH_API_PROTOCOL"):
        configured_model_protocol = env_entries.get(env_key)
        if (
            configured_model_protocol is not None
            and configured_model_protocol not in MODEL_API_PROTOCOL_ALLOWLIST
        ):
            raise ValueError(
                f"{env_key} must be an allowlisted model API protocol: "
                + ", ".join(sorted(MODEL_API_PROTOCOL_ALLOWLIST))
            )
    configured_embedding_protocol = env_entries.get("EMBEDDING_API_PROTOCOL")
    if (
        configured_embedding_protocol is not None
        and configured_embedding_protocol not in EMBEDDING_API_PROTOCOL_ALLOWLIST
    ):
        raise ValueError(
            "EMBEDDING_API_PROTOCOL must be an allowlisted embedding API protocol: "
            + ", ".join(sorted(EMBEDDING_API_PROTOCOL_ALLOWLIST))
        )
    graph_protocol_allowlists = {
        "EDGE_DISTANCE_PROTOCOL": EDGE_DISTANCE_PROTOCOL_ALLOWLIST,
        "RQ_MEMBERSHIP_PROTOCOL": RQ_MEMBERSHIP_PROTOCOL_ALLOWLIST,
        "EDGE_PROJECTION_PROTOCOL": EDGE_PROJECTION_PROTOCOL_ALLOWLIST,
        "EDGE_TYPE_CALIBRATION_PROTOCOL": EDGE_TYPE_CALIBRATION_PROTOCOL_ALLOWLIST,
    }
    for env_key, allowlist in graph_protocol_allowlists.items():
        configured_protocol = env_entries.get(env_key)
        if configured_protocol is not None and configured_protocol not in allowlist:
            raise ValueError(
                f"{env_key} must be a locally allowlisted graph protocol: "
                + ", ".join(sorted(allowlist))
            )
    configured_gray_zone_protocol = env_entries.get("GRAY_ZONE_RULE_PROTOCOL")
    if (
        configured_gray_zone_protocol is not None
        and configured_gray_zone_protocol not in GRAY_ZONE_RULE_PROTOCOL_ALLOWLIST
    ):
        raise ValueError(
            "GRAY_ZONE_RULE_PROTOCOL must be an allowlisted deterministic local protocol: "
            + ", ".join(sorted(GRAY_ZONE_RULE_PROTOCOL_ALLOWLIST))
        )
    configured_observation_cadence = env_entries.get("GRAY_ZONE_OBSERVATION_CADENCE")
    if configured_observation_cadence is not None:
        try:
            parsed_observation_cadence = int(configured_observation_cadence)
        except ValueError as exc:
            raise ValueError(
                "GRAY_ZONE_OBSERVATION_CADENCE must be an integer between 1 and "
                f"{GRAY_ZONE_OBSERVATION_CADENCE_MAX}"
            ) from exc
        if not 1 <= parsed_observation_cadence <= GRAY_ZONE_OBSERVATION_CADENCE_MAX:
            raise ValueError(
                "GRAY_ZONE_OBSERVATION_CADENCE must be an integer between 1 and "
                f"{GRAY_ZONE_OBSERVATION_CADENCE_MAX}"
            )
    configured_observation_budget = env_entries.get("TRAVERSAL_OBSERVATION_BUDGET")
    if configured_observation_budget is not None:
        try:
            parsed_observation_budget = int(configured_observation_budget)
        except ValueError as exc:
            raise ValueError(
                "TRAVERSAL_OBSERVATION_BUDGET must be an integer between 1 and "
                f"{TRAVERSAL_OBSERVATION_BUDGET_MAX}"
            ) from exc
        if not 1 <= parsed_observation_budget <= TRAVERSAL_OBSERVATION_BUDGET_MAX:
            raise ValueError(
                "TRAVERSAL_OBSERVATION_BUDGET must be an integer between 1 and "
                f"{TRAVERSAL_OBSERVATION_BUDGET_MAX}"
            )
    configured_rq_levels = env_entries.get("RQ_KMEANS_LEVELS")
    if configured_rq_levels is None:
        return
    try:
        parsed_rq_levels = int(configured_rq_levels)
    except ValueError as exc:
        raise ValueError("RQ_KMEANS_LEVELS must be the fixed active protocol depth 3") from exc
    if parsed_rq_levels != ACTIVE_RQ_KMEANS_LEVELS:
        raise ValueError("RQ_KMEANS_LEVELS must be the fixed active protocol depth 3")


def _apply_hot_reload_env(settings: Settings, env_entries: dict[str, str]) -> None:
    _validate_fixed_protocol_env(env_entries)
    bool_fields = {
        "model_bridge_enabled",
        "enable_model_fallback",
        "concept_i18n_enabled",
        "query_facet_bilingual_enabled",
        "query_facet_posterior_enabled",
        "enable_auto_tpe",
    }
    int_fields = {
        "model_bridge_port",
        "embedding_dimensions",
        "embedding_batch_size",
        "model_request_concurrency",
        "model_request_timeout_seconds",
        "chat_json_max_tokens",
        "agent_request_concurrency",
        "source_io_concurrency",
        "agent_request_queue_limit",
        "agent_request_queue_timeout_seconds",
        "agent_request_lease_ttl_seconds",
        "fixed_chunk_size_tokens",
        "fixed_chunk_overlap_tokens",
        "context_package_token_budget",
        "upload_max_bytes",
        "mid_concept_extraction_max_model_batches",
        "mid_concept_extraction_max_candidates_per_batch",
        "mid_concept_extraction_max_tokens_per_batch",
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
        "tpe_trial_budget",
        "tpe_startup_random_trials",
        "tpe_probe_query_budget",
        "tpe_trial_timeout_seconds",
        "tpe_candidate_pool_size",
        "operating_point_hard_gate_max_candidate_latency_p95_ms",
        "retrieval_result_top_k_default",
        "agent_coarse_initial_budget",
        "agent_coarse_total_budget",
        "agent_coarse_top_k",
        "agent_mid_per_coarse_budget",
        "agent_coarse_drilldown_mid_initial_budget",
        "agent_mid_initial_budget",
        "agent_mid_top_k",
        "agent_chunk_per_mid_budget",
        "agent_chunk_initial_budget",
        "agent_chunk_top_k",
        "agent_max_depth_per_layer",
        "agent_max_labels_per_node",
        "agent_max_edge_reuse",
        "candidate_pool_dedupe_budget",
        "agent_structure_restore_per_chunk_budget",
        "agent_structure_restore_budget",
        "context_path_summary_budget",
        "agent_planning_round_budget",
        "agent_max_typed_actions_per_round",
        "agent_repair_round_budget",
        "agent_verification_budget",
        "gray_zone_observation_cadence",
        "traversal_observation_budget",
        "query_facet_posterior_observation_budget",
        "query_facet_posterior_round_budget",
    }
    float_fields: set[str] = {
        "mid_concept_candidate_keep_threshold",
        "rq_residual_tau",
        "rq_membership_temperature",
        "dense_min_cosine",
        "dense_strong_cosine",
        "cross_doc_min_cosine",
        "cross_language_min_cosine",
        "tpe_good_quantile_gamma",
        "operating_point_hard_gate_max_edge_density",
        "operating_point_hard_gate_max_isolated_ratio",
        "operating_point_hard_gate_max_hubness_ratio",
        "operating_point_hard_gate_min_structure_recovery_rate",
        "agent_max_cycle_reward_per_path",
        "agent_cycle_reward_distance_threshold",
        "agent_path_distance_green_threshold",
        "agent_path_distance_gray_threshold",
        "agent_path_distance_hard_threshold",
        "query_facet_posterior_convergence_epsilon",
    }
    nullable_fields = {
        "chat_resolve_ip",
        "graph_resolve_ip",
        "embedding_resolve_ip",
    }
    aliases = {
        "MODEL_REQUEST_CONCURRENCY": "model_request_concurrency",
        "MODEL_REQUEST_TIMEOUT_SECONDS": "model_request_timeout_seconds",
        "CHAT_JSON_MAX_TOKENS": "chat_json_max_tokens",
        "AGENT_REQUEST_CONCURRENCY": "agent_request_concurrency",
        "SOURCE_IO_CONCURRENCY": "source_io_concurrency",
        "AGENT_REQUEST_QUEUE_LIMIT": "agent_request_queue_limit",
        "AGENT_REQUEST_QUEUE_TIMEOUT_SECONDS": "agent_request_queue_timeout_seconds",
        "AGENT_REQUEST_LEASE_TTL_SECONDS": "agent_request_lease_ttl_seconds",
        "FIXED_CHUNK_SIZE_TOKENS": "fixed_chunk_size_tokens",
        "FIXED_CHUNK_OVERLAP_TOKENS": "fixed_chunk_overlap_tokens",
        "CONTEXT_PACKAGE_TOKEN_BUDGET": "context_package_token_budget",
        "UPLOAD_MAX_BYTES": "upload_max_bytes",
        "CONCEPT_I18N_ENABLED": "concept_i18n_enabled",
        "QUERY_FACET_BILINGUAL_ENABLED": "query_facet_bilingual_enabled",
        "QUERY_FACET_POSTERIOR_ENABLED": "query_facet_posterior_enabled",
        "QUERY_FACET_POSTERIOR_OBSERVATION_BUDGET": (
            "query_facet_posterior_observation_budget"
        ),
        "QUERY_FACET_POSTERIOR_ROUND_BUDGET": (
            "query_facet_posterior_round_budget"
        ),
        "QUERY_FACET_POSTERIOR_CONVERGENCE_EPSILON": (
            "query_facet_posterior_convergence_epsilon"
        ),
        "MID_CONCEPT_EXTRACTION_MAX_MODEL_BATCHES": "mid_concept_extraction_max_model_batches",
        "MID_CONCEPT_EXTRACTION_MAX_CANDIDATES_PER_BATCH": "mid_concept_extraction_max_candidates_per_batch",
        "MID_CONCEPT_EXTRACTION_MAX_TOKENS_PER_BATCH": "mid_concept_extraction_max_tokens_per_batch",
        "MID_CONCEPT_CANDIDATE_KEEP_THRESHOLD": "mid_concept_candidate_keep_threshold",
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
        "ENABLE_AUTO_TPE": "enable_auto_tpe",
        "TPE_TRIAL_BUDGET": "tpe_trial_budget",
        "TPE_STARTUP_RANDOM_TRIALS": "tpe_startup_random_trials",
        "TPE_GOOD_QUANTILE_GAMMA": "tpe_good_quantile_gamma",
        "TPE_PROBE_QUERY_BUDGET": "tpe_probe_query_budget",
        "TPE_TRIAL_TIMEOUT_SECONDS": "tpe_trial_timeout_seconds",
        "TPE_CANDIDATE_POOL_SIZE": "tpe_candidate_pool_size",
        "OPERATING_POINT_HARD_GATE_MAX_EDGE_DENSITY": "operating_point_hard_gate_max_edge_density",
        "OPERATING_POINT_HARD_GATE_MAX_ISOLATED_RATIO": "operating_point_hard_gate_max_isolated_ratio",
        "OPERATING_POINT_HARD_GATE_MAX_HUBNESS_RATIO": "operating_point_hard_gate_max_hubness_ratio",
        "OPERATING_POINT_HARD_GATE_MIN_STRUCTURE_RECOVERY_RATE": "operating_point_hard_gate_min_structure_recovery_rate",
        "OPERATING_POINT_HARD_GATE_MAX_CANDIDATE_LATENCY_P95_MS": "operating_point_hard_gate_max_candidate_latency_p95_ms",
        "RETRIEVAL_RESULT_TOP_K_DEFAULT": "retrieval_result_top_k_default",
        "AGENT_COARSE_INITIAL_BUDGET": "agent_coarse_initial_budget",
        "AGENT_COARSE_TOTAL_BUDGET": "agent_coarse_total_budget",
        "AGENT_COARSE_TOP_K": "agent_coarse_top_k",
        "AGENT_MID_PER_COARSE_BUDGET": "agent_mid_per_coarse_budget",
        "AGENT_COARSE_DRILLDOWN_MID_INITIAL_BUDGET": "agent_coarse_drilldown_mid_initial_budget",
        "AGENT_MID_INITIAL_BUDGET": "agent_mid_initial_budget",
        "AGENT_MID_TOP_K": "agent_mid_top_k",
        "AGENT_CHUNK_PER_MID_BUDGET": "agent_chunk_per_mid_budget",
        "AGENT_CHUNK_INITIAL_BUDGET": "agent_chunk_initial_budget",
        "AGENT_CHUNK_TOP_K": "agent_chunk_top_k",
        "AGENT_MAX_DEPTH_PER_LAYER": "agent_max_depth_per_layer",
        "AGENT_MAX_LABELS_PER_NODE": "agent_max_labels_per_node",
        "AGENT_MAX_EDGE_REUSE": "agent_max_edge_reuse",
        "AGENT_MAX_CYCLE_REWARD_PER_PATH": "agent_max_cycle_reward_per_path",
        "AGENT_CYCLE_REWARD_DISTANCE_THRESHOLD": "agent_cycle_reward_distance_threshold",
        "AGENT_PATH_DISTANCE_GREEN_THRESHOLD": "agent_path_distance_green_threshold",
        "AGENT_PATH_DISTANCE_GRAY_THRESHOLD": "agent_path_distance_gray_threshold",
        "AGENT_PATH_DISTANCE_HARD_THRESHOLD": "agent_path_distance_hard_threshold",
        "GRAY_ZONE_RULE_PROTOCOL": "gray_zone_rule_protocol",
        "GRAY_ZONE_OBSERVATION_CADENCE": "gray_zone_observation_cadence",
        "TRAVERSAL_OBSERVATION_BUDGET": "traversal_observation_budget",
        "CANDIDATE_POOL_DEDUPE_BUDGET": "candidate_pool_dedupe_budget",
        "AGENT_STRUCTURE_RESTORE_PER_CHUNK_BUDGET": "agent_structure_restore_per_chunk_budget",
        "AGENT_STRUCTURE_RESTORE_BUDGET": "agent_structure_restore_budget",
        "CONTEXT_PATH_SUMMARY_BUDGET": "context_path_summary_budget",
        "AGENT_PLANNING_ROUND_BUDGET": "agent_planning_round_budget",
        "AGENT_MAX_TYPED_ACTIONS_PER_ROUND": "agent_max_typed_actions_per_round",
        "AGENT_REPAIR_ROUND_BUDGET": "agent_repair_round_budget",
        "AGENT_VERIFICATION_BUDGET": "agent_verification_budget",
    }
    for env_key, value in env_entries.items():
        if env_key in PROCESS_ONLY_ENV_KEYS:
            continue
        attr = aliases.get(env_key, env_key.lower())
        if attr not in RUNTIME_ENV_SETTINGS:
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
    validate_path_distance_thresholds(
        settings.agent_path_distance_green_threshold,
        settings.agent_path_distance_gray_threshold,
        settings.agent_path_distance_hard_threshold,
    )


def _build_settings() -> Settings:
    # The root .env is the sole persisted authority, but lifecycle activation
    # is represented by the current process environment.  Startup copies all
    # runtime keys into the process; later broadcasts copy only the lifecycle-
    # allowed subset.  Reading the file again here would make rebuild/service
    # edits effective merely because the file identity changed.
    env_entries = {
        key.upper(): value
        for key, value in os.environ.items()
        if key.lower() in RUNTIME_ENV_SETTINGS
        or key.upper() in {"MODEL_BRIDGE_PORT", "MODEL_BRIDGE_ADMIN_TOKEN"}
    }
    settings = Settings()
    _apply_hot_reload_env(settings, env_entries)

    api_chat_base_url = os.getenv("API_CHAT_BASE_URL")
    api_chat_resolve_ip = os.getenv("API_CHAT_RESOLVE_IP")
    api_graph_base_url = os.getenv("API_GRAPH_BASE_URL")
    api_graph_resolve_ip = os.getenv("API_GRAPH_RESOLVE_IP")
    model_bridge_enabled = str(os.getenv("MODEL_BRIDGE_ENABLED") or env_entries.get("MODEL_BRIDGE_ENABLED") or "").lower() in {
        "true",
        "1",
        "yes",
        "on",
    }
    model_bridge_port = os.getenv("MODEL_BRIDGE_PORT") or env_entries.get("MODEL_BRIDGE_PORT")
    if model_bridge_port:
        try:
            settings.model_bridge_port = int(model_bridge_port)
        except ValueError:
            pass
    # The bridge admin credential is a process-only service-recreate secret.
    # Its override is authoritative by presence, not truthiness: preserve even
    # an empty, padded, or control-bearing value so lifecycle validation fails
    # closed instead of reviving a stale managed token.
    settings.model_bridge_admin_token = (
        os.environ["MODEL_BRIDGE_ADMIN_TOKEN"]
        if "MODEL_BRIDGE_ADMIN_TOKEN" in os.environ
        else ""
    )
    settings.model_bridge_enabled = model_bridge_enabled
    if model_bridge_enabled:
        settings.chat_base_url = model_bridge_client_base_url(settings.model_bridge_port)
        settings.chat_resolve_ip = "__none__"
    elif api_chat_base_url:
        settings.chat_base_url = api_chat_base_url
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

    if api_graph_base_url:
        settings.graph_base_url = api_graph_base_url
    elif env_entries.get("GRAPH_BASE_URL"):
        settings.graph_base_url = env_entries["GRAPH_BASE_URL"]
    elif "GRAPH_BASE_URL" in os.environ:
        settings.graph_base_url = os.getenv("GRAPH_BASE_URL", "")
    if api_graph_resolve_ip is not None:
        settings.graph_resolve_ip = api_graph_resolve_ip
    elif os.getenv("GRAPH_RESOLVE_IP"):
        settings.graph_resolve_ip = os.getenv("GRAPH_RESOLVE_IP")
    elif env_entries.get("GRAPH_RESOLVE_IP") is not None:
        settings.graph_resolve_ip = env_entries.get("GRAPH_RESOLVE_IP")
    elif "GRAPH_RESOLVE_IP" in os.environ:
        settings.graph_resolve_ip = os.getenv("GRAPH_RESOLVE_IP")

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
        "CHAT_JSON_MAX_TOKENS": "chat_json_max_tokens",
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

    # Configuration loading is a pure read. Storage roots are materialized
    # only after the storage capability gate has proved the deployment-
    # provisioned DATA_ROOT filesystem contract.
    return settings


def _clear_settings_cache() -> None:
    global _SETTINGS_CACHE, _SETTINGS_CACHE_TOKEN
    _SETTINGS_CACHE = None
    _SETTINGS_CACHE_TOKEN = None


def get_settings() -> Settings:
    global _SETTINGS_CACHE, _SETTINGS_CACHE_TOKEN
    override = _SETTINGS_OVERRIDE.get()
    if override is not None:
        return override
    token = _settings_cache_token()
    if _SETTINGS_CACHE is not None and _SETTINGS_CACHE_TOKEN == token:
        return _SETTINGS_CACHE
    settings = _build_settings()
    _SETTINGS_CACHE = settings
    _SETTINGS_CACHE_TOKEN = token
    return settings


get_settings.cache_clear = _clear_settings_cache  # type: ignore[attr-defined]


def runtime_settings_override_active() -> bool:
    return _SETTINGS_OVERRIDE.get() is not None


@contextmanager
def use_runtime_settings_override(overrides: dict[str, object]):
    """Use validated candidate settings in one async/task context only.

    No shared environment, active cache or Redis version is changed while a
    shadow build consumes this override.
    """

    unknown = sorted(set(overrides).difference(Settings.model_fields))
    if unknown:
        raise ValueError(
            "Unknown runtime settings override keys: " + ", ".join(unknown)
        )
    active = get_settings()
    candidate = Settings.model_validate(
        {**active.model_dump(mode="python"), **dict(overrides)}
    )
    token = _SETTINGS_OVERRIDE.set(candidate)
    try:
        yield candidate
    finally:
        _SETTINGS_OVERRIDE.reset(token)
