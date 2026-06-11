from __future__ import annotations

from app.core.config import get_settings
from app.services.runtime_settings import read_env_bool, read_env_int, read_env_str


_reranker_instance: CrossEncoderReranker | None = None
_reranker_error: Exception | None = None


class CrossEncoderReranker:
    """Cross-Encoder 精排器。使用 sentence-transformers 的 CrossEncoder。

    依赖 sentence-transformers（已在 pyproject.toml [rerank] optional 中声明）。
    未安装时实例化会抛出 ImportError。
    """

    def __init__(self, model_name: str | None = None, max_length: int = 512):
        from sentence_transformers import CrossEncoder

        settings = get_settings()
        self.model_name = model_name or settings.reranker_model
        self.max_length = max_length or settings.reranker_max_length
        self.device = read_env_str("RERANKER_DEVICE", settings.reranker_device)
        self.model = CrossEncoder(self.model_name, max_length=self.max_length, device=self.device)

    def rerank(self, query: str, candidates: list[dict], top_k: int) -> list[dict]:
        if not candidates:
            return []

        pairs = []
        for candidate in candidates:
            text = " ".join(
                [
                    candidate.get("document_title", ""),
                    candidate.get("snippet", ""),
                    candidate.get("content", "")[:400],
                ]
            ).strip()
            pairs.append((query, text))

        scores = self.model.predict(pairs)
        scored = []
        for candidate, score in zip(candidates, scores):
            candidate = dict(candidate)
            candidate["score"] = float(score)
            scores = candidate.setdefault("metadata", {}).setdefault("scores", {})
            scores["cross_encoder"] = round(float(score), 4)
            scores["rerank"] = round(float(score), 4)
            scored.append(candidate)

        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]


class RerankerError(RuntimeError):
    pass


def _load_reranker() -> CrossEncoderReranker:
    """加载并缓存 CrossEncoder 实例（只加载一次）。"""
    global _reranker_instance, _reranker_error
    if _reranker_instance is not None:
        return _reranker_instance
    if _reranker_error is not None:
        raise _reranker_error
    try:
        _reranker_instance = CrossEncoderReranker()
        return _reranker_instance
    except Exception as exc:
        _reranker_error = exc
        raise


def get_reranker() -> CrossEncoderReranker:
    """获取 Cross-Encoder 精排器实例（单例缓存）。

    如果 sentence-transformers 未安装或 RERANKER_ENABLED=false，则抛出错误（不 fallback）。
    """
    if not read_env_bool("RERANKER_ENABLED", False):
        raise RerankerError("RERANKER_ENABLED is false. Set RERANKER_ENABLED=true to enable Cross-Encoder reranking.")
    try:
        return _load_reranker()
    except ImportError as exc:
        raise RerankerError(
            "sentence-transformers is required for Cross-Encoder reranking. "
            "Install it with: pip install 'KnowledgeBase-kg-api[rerank]'"
        ) from exc


def get_reranker_status() -> dict:
    """获取 CrossEncoder 的运行时状态（不重新加载模型，查缓存）。"""
    reranker_enabled = read_env_bool("RERANKER_ENABLED", False)
    reranker_model = read_env_str("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    reranker_device = read_env_str("RERANKER_DEVICE", "cpu")
    if not reranker_enabled:
        return {
            "enabled": False,
            "device": reranker_device,
            "model": reranker_model,
            "url": "",
            "reachable": False,
            "healthy": False,
            "reported_model": None,
            "reported_device": None,
            "model_matches": None,
            "device_matches": None,
        }

    # 已缓存且成功
    if _reranker_instance is not None:
        return {
            "enabled": True,
            "device": str(_reranker_instance.model.device),
            "model": _reranker_instance.model_name,
            "url": "",
            "reachable": True,
            "healthy": True,
            "reported_model": _reranker_instance.model_name,
            "reported_device": str(_reranker_instance.model.device),
            "model_matches": _reranker_instance.model_name == reranker_model,
            "device_matches": str(_reranker_instance.model.device) == reranker_device,
        }

    # 缓存了错误
    if _reranker_error is not None:
        return {
            "enabled": True,
            "device": reranker_device,
            "model": reranker_model,
            "url": "",
            "reachable": False,
            "healthy": False,
            "reported_model": None,
            "reported_device": None,
            "model_matches": None,
            "device_matches": None,
        }

    # 从未尝试过加载，尝试一次（可能较慢，但只此一次）
    try:
        instance = _load_reranker()
        return {
            "enabled": True,
            "device": str(instance.model.device),
            "model": instance.model_name,
            "url": "",
            "reachable": True,
            "healthy": True,
            "reported_model": instance.model_name,
            "reported_device": str(instance.model.device),
            "model_matches": instance.model_name == reranker_model,
            "device_matches": str(instance.model.device) == reranker_device,
        }
    except Exception:
        return {
            "enabled": True,
            "device": reranker_device,
            "model": reranker_model,
            "url": "",
            "reachable": False,
            "healthy": False,
            "reported_model": None,
            "reported_device": None,
            "model_matches": None,
            "device_matches": None,
        }


def clear_reranker_cache() -> None:
    """清除 reranker 单例缓存（用于切换模型后重新加载）。"""
    global _reranker_instance, _reranker_error
    _reranker_instance = None
    _reranker_error = None
