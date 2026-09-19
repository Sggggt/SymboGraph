from __future__ import annotations

import json
from types import SimpleNamespace

import pytest


def test_grounded_markdown_accumulator_keeps_provider_text_append_only() -> None:
    from app.services.answer_stream import GroundedMarkdownAccumulator

    raw = (
        "## 定义\n\n状态为 $x_t$。⟦cite:src_1⟧\n\n"
        "未知引用照常显示：⟦cite:src_9⟧\n"
        "未闭合引用也照常显示：⟦cite:src_2"
    )
    accumulator = GroundedMarkdownAccumulator(
        allowed_handles=frozenset({"src_1", "src_2"})
    )
    visible = ""
    for start in range(0, len(raw), 3):
        delta = accumulator.feed(raw[start : start + 3])
        visible += delta
    result = accumulator.finalize(
        provider_stream_used=True,
        source_handle_count=3,
    )
    visible += result.final_delta

    assert visible == result.answer
    assert result.answer == (
        "## 定义\n\n状态为 $x_t$。[1](#source-1)\n\n"
        "未知引用照常显示：⟦cite:src_9⟧\n"
        "未闭合引用也照常显示：⟦cite:src_2"
    )
    assert result.audit["replace_count"] == 0
    assert result.audit["source_handle_count"] == 3
    assert result.audit["citation_marker_count"] == 1
    assert result.audit["invalid_marker_count"] == 2


@pytest.mark.parametrize(
    ("raw", "code"),
    [
        ("", "answer_stream_empty"),
        ("   \n", "answer_stream_empty"),
        ("正文\x00尾部", "answer_stream_contains_nul"),
    ],
)
def test_grounded_markdown_accumulator_rejects_invalid_transport_data(
    raw: str,
    code: str,
) -> None:
    from app.services.answer_stream import (
        GroundedMarkdownAccumulator,
        GroundedMarkdownStreamError,
    )

    accumulator = GroundedMarkdownAccumulator(
        allowed_handles=frozenset({"src_1"})
    )
    with pytest.raises(GroundedMarkdownStreamError) as caught:
        accumulator.feed(raw)
        accumulator.finalize(
            provider_stream_used=True,
            source_handle_count=1,
        )
    assert caught.value.code == code


def test_grounded_answer_delta_projector_decodes_partial_json_strings() -> None:
    from app.services.answer_stream import GroundedAnswerDeltaProjector

    raw = json.dumps(
        {
            "protocol_version": "grounded_answer_units_v2",
            "answer_units": [
                {
                    "kind": "factual",
                    "text": "## 定义\n\n状态为 $x_t$。",
                    "source_handles": ["src_1"],
                },
                {
                    "kind": "factual",
                    "text": "- 第一步\n- 第二步",
                    "source_handles": ["src_2"],
                },
            ],
        },
        ensure_ascii=True,
        separators=(",", ":"),
    )
    projector = GroundedAnswerDeltaProjector()
    updates = []
    for start in range(0, len(raw), 7):
        update = projector.feed(raw[start : start + 7])
        if update is not None:
            updates.append(update)

    rendered = "## 定义\n\n状态为 $x_t$。\n\n- 第一步\n- 第二步"
    replay = ""
    for kind, text in updates:
        replay = replay + text if kind == "delta" else text
    assert replay == rendered
    assert projector.finalize(rendered) is None
    assert projector.audit(provider_stream_used=True)["delta_count"] > 2


def test_grounded_answer_delta_projector_can_replace_a_divergent_final_value() -> None:
    from app.services.answer_stream import GroundedAnswerDeltaProjector

    projector = GroundedAnswerDeltaProjector()
    projector.rendered_text = "partial"
    assert projector.finalize("authoritative final") == (
        "replace",
        "authoritative final",
    )
    assert projector.replace_count == 1


def test_grounded_answer_delta_projector_never_projects_trailing_text_fields() -> None:
    from app.services.answer_stream import partial_grounded_answer_text

    raw = json.dumps(
        {
            "protocol_version": "grounded_answer_units_v2",
            "answer_units": [
                {
                    "kind": "factual",
                    "text": "Visible answer",
                    "source_handles": ["src_1"],
                }
            ],
            "text": "must-not-stream",
        },
        separators=(",", ":"),
    )

    assert partial_grounded_answer_text(raw) == "Visible answer"


@pytest.mark.asyncio
async def test_retrieval_generation_streams_final_markdown_and_collects_sources() -> None:
    from app.services.answer_stream import use_answer_stream_sink
    from app.services.retrieval_models import RetrievalModels

    encoded = "## Result\n\n$R(n)=0.9$⟦cite:src_1,src_2⟧"

    class Provider:
        api_protocol = "unit-test"
        model = "unit-test-stream-model"

        async def complete_text_streaming(
            self,
            system_prompt,
            user_prompt,
            *,
            max_tokens,
            on_text_delta,
        ):
            assert system_prompt == "system"
            assert user_prompt == "{}"
            assert max_tokens == 512
            for start in range(0, len(encoded), 5):
                await on_text_delta(encoded[start : start + 5])
            return encoded

        def provider_call_audit(self):
            return {"protocol_version": "unit-test-provider-call"}

    updates: list[dict[str, str]] = []

    async def sink(update: dict[str, str]) -> None:
        updates.append(update)

    with use_answer_stream_sink(sink):
        parsed, audit = await RetrievalModels(Provider)._call_grounded_markdown(
            system="system",
            packet={},
            evidence=SimpleNamespace(
                by_handle=lambda: {"src_1": {}, "src_2": {}}
            ),
            unit_limit=4,
            timeout_seconds=5,
            max_tokens=512,
        )

    replay = ""
    for update in updates:
        replay = (
            replay + update["text"]
            if update["type"] == "delta"
            else update["text"]
        )
    assert replay == parsed.answer_units[0].text
    assert replay == "## Result\n\n$R(n)=0.9$[1](#source-1) [2](#source-2)"
    assert parsed.answer_units[0].source_handles == ("src_1", "src_2")
    assert audit["answer_stream"]["provider_stream_used"] is True
    assert audit["answer_stream"]["replace_count"] == 0
    assert [update["type"] for update in updates] and all(
        update["type"] == "delta" for update in updates
    )
