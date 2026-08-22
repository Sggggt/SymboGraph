from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

import pytest
from pydantic import ValidationError


def _edge() -> SimpleNamespace:
    return SimpleNamespace(
        id="edge-i18n-1",
        explanation="原始关系解释",
    )


def test_edge_i18n_normalizer_emits_closed_public_contract() -> None:
    from app.schemas import GraphEdgeI18n
    from app.services.context_graph import _normalize_edge_i18n_item

    payload = _normalize_edge_i18n_item(
        {
            "status": "ok",
            "relation_label_i18n": {"zh": "依赖", "en": "depends on"},
            "explanation_i18n": {
                "zh": "原始关系解释",
                "en": "original relation explanation",
            },
            "summary_i18n": {"zh": "依赖关系", "en": "dependency"},
            "search_terms_i18n": {
                "zh": ["依赖"],
                "en": ["dependency"],
            },
        },
        _edge(),
        "源概念",
        "目标概念",
        "mid",
    )

    validated = GraphEdgeI18n.model_validate(payload)
    assert validated.status == "ok"
    assert validated.fallback_source is None
    assert GraphEdgeI18n.model_validate_json(validated.model_dump_json()) == validated


def test_edge_i18n_untrusted_status_is_deterministic_original_fallback() -> None:
    from app.schemas import GraphEdgeI18n
    from app.services.context_graph import _normalize_edge_i18n_item

    payload = _normalize_edge_i18n_item(
        {
            "status": "provider_decides",
            "relation_label_i18n": {"zh": "伪造", "en": "forged"},
        },
        _edge(),
        "源概念",
        "目标概念",
        "coarse",
    )

    validated = GraphEdgeI18n.model_validate(payload)
    assert validated.status == "original_text_fallback"
    assert validated.fallback_source == "original_text_fallback"
    assert validated.relation_label_i18n.zh == "源概念 -> 目标概念"
    assert validated.relation_label_i18n.en == "源概念 -> 目标概念"


def test_edge_i18n_fallback_helper_matches_closed_contract() -> None:
    from app.schemas import GraphEdgeI18n
    from app.services.context_graph import _edge_i18n_fallback

    payload = _edge_i18n_fallback(
        _edge(),
        "源概念",
        "目标概念",
        "mid",
    )
    validated = GraphEdgeI18n.model_validate(payload)

    assert validated.status == "original_text_fallback"
    assert validated.fallback_source == "original_text_fallback"


@pytest.mark.parametrize(
    "mutation",
    [
        lambda payload: payload.update({"provider_response": "forbidden"}),
        lambda payload: payload.update({"protocol_version": "legacy_i18n_v0"}),
        lambda payload: payload.update({"status": "provider_decides"}),
        lambda payload: payload.update({"layer": "chunk"}),
        lambda payload: payload["relation_label_i18n"].pop("en"),
        lambda payload: payload["relation_label_i18n"].update({"fr": "dépend"}),
        lambda payload: payload.update(
            {
                "status": "original_text_fallback",
                "fallback_source": None,
            }
        ),
        lambda payload: payload.update(
            {
                "status": "ok",
                "fallback_source": "original_text_fallback",
            }
        ),
    ],
)
def test_edge_i18n_contract_rejects_open_or_inconsistent_payloads(
    mutation,
) -> None:
    from app.schemas import GraphEdgeI18n

    payload = {
        "id": "edge-i18n-1",
        "layer": "mid",
        "protocol_version": "concept_i18n_bilingual_v1",
        "status": "ok",
        "relation_label_i18n": {"zh": "依赖", "en": "depends on"},
        "explanation_i18n": {"zh": "解释", "en": "explanation"},
        "summary_i18n": {"zh": "摘要", "en": "summary"},
        "search_terms_i18n": {"zh": ["依赖"], "en": ["dependency"]},
        "fallback_source": None,
    }
    poisoned = deepcopy(payload)
    mutation(poisoned)

    with pytest.raises(ValidationError):
        GraphEdgeI18n.model_validate(poisoned)
