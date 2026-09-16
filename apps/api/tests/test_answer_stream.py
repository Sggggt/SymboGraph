from __future__ import annotations

import json

import pytest


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
async def test_retrieval_generation_streams_only_answer_unit_text_and_keeps_final_schema() -> None:
    from app.retrieval_control_contracts import GroundedAnswerDraft
    from app.services.answer_stream import use_answer_stream_sink
    from app.services.retrieval_models import RetrievalModels

    raw = {
        "protocol_version": "grounded_answer_units_v2",
        "answer_units": [
            {
                "kind": "factual",
                "text": "Result $R(n)=0.9$",
                "source_handles": ["src_1"],
            }
        ],
    }
    encoded = json.dumps(raw, separators=(",", ":"))

    class Provider:
        api_protocol = "unit-test"
        model = "unit-test-stream-model"

        async def classify_json_streaming(
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
            return raw

        def provider_call_audit(self):
            return {"protocol_version": "unit-test-provider-call"}

    updates: list[dict[str, str]] = []

    async def sink(update: dict[str, str]) -> None:
        updates.append(update)

    with use_answer_stream_sink(sink):
        parsed, audit = await RetrievalModels(Provider)._call(
            stage="generation",
            system="system",
            packet={},
            output_type=GroundedAnswerDraft,
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
    assert replay == "- Result $R(n)=0.9$"
    assert "src_1" not in replay
    assert audit["answer_stream"]["provider_stream_used"] is True
    # Raw provider prose remains append-only while streaming. The one final
    # replacement adds a presentation-only GFM marker after schema validation.
    assert audit["answer_stream"]["replace_count"] == 1
    assert audit["gfm_projection"]["projected_unit_count"] == 1
