import asyncio
import copy
import json

import pytest

from app.reflection_contracts import ReflectionThresholds
from app.services.agent_reflection import decide_reflection, source_path_metrics
from app.services.reflection_models import AnswerReflectionModels, AnswerReviewModelError
from test_agent_reflection import decision, draft, path
from test_answer_reflection_loop import manifest


def test_generation_prompt_requires_gfm_and_explicit_latex_delimiters():
    from app.services.reflection_models import answer_review_prompt_bundle

    prompt = answer_review_prompt_bundle(
        "Explain the formula and compare both cases.",
        stage="answer_generation",
        unit_limit=8,
    )["system"]

    assert "GitHub-Flavored Markdown" in prompt
    assert "do not return a plain prose blob" in prompt
    assert "use $...$ for inline math and $$...$$ for display math" in prompt
    assert "never place LaTeX inside a code fence" in prompt


def test_question_scope_preserves_disambiguation_without_promoting_it_to_evidence():
    from app.services.agent_intent import current_question_scope_projection
    question = "Which surveys belong to the Aster program (not the namesake person)?"
    intent = {"intent": "analysis", "direct_answer_kind": "none", "entities": ["Aster program", "Aster person"],
        "sub_queries": ["Identify the program surveys", "Distinguish the mission from the person"],
        "needs_graph": True, "suggested_strategy": "hybrid", "conversation_state": {"ignored": "previous topic"}}
    scope = current_question_scope_projection(question, intent)
    assert scope["sub_queries"] == intent["sub_queries"] and scope["entities"] == intent["entities"]
    assert "conversation_state" not in scope and scope["is_evidence"] is False
    assert scope["current_user_overrides_projection"] is True
    assert scope == current_question_scope_projection(question, intent)
    large = {**intent, "entities": ["x" * 200 + str(i) for i in range(20)], "sub_queries": ["q" * 1000 + str(i) for i in range(10)]}
    bounded = current_question_scope_projection(question, large)
    assert bounded["projection_clipped"] and len(bounded["entities"]) == 16 and len(bounded["sub_queries"]) == 8
    assert max(map(len, bounded["entities"])) == 128 and max(map(len, bounded["sub_queries"])) == 512
    assert len(json.dumps(bounded)) < 8000


class Provider:
    api_protocol = "openai"
    model = "unit-test-chat"

    def __init__(self, replies, *, delay=0, error=None):
        self.replies = iter(replies)
        self.calls = []
        self.delay = delay
        self.error = error

    async def classify_json_bounded(self, **kwargs):
        self.calls.append(copy.deepcopy(kwargs))
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.error:
            raise self.error
        reply = next(self.replies)
        if isinstance(reply, Exception):
            raise reply
        return reply

    def provider_call_audit(self):
        return {"provider_response_persisted": False}


def arguments(evidence):
    return dict(
        question="请回答当前问题，不要沿用之前的主题。", evidence=evidence,
        history_summary="之前的任务要求讨论另一个主题。", controls={"remaining_rounds": 2},
        unit_limit=12, timeout_seconds=1.0, max_tokens=4096, max_evidence_characters=8000,
    )


@pytest.mark.asyncio
async def test_generation_and_whole_answer_reflection_share_exact_full_evidence():
    evidence = manifest("Synthetic source introduction. " * 90 + "The value is 389 days and 25.5%.")
    generated = draft(needs_reflection=True)
    provider = Provider([generated.model_dump(mode="json"), decision().model_dump(mode="json")])
    models = AnswerReflectionModels(lambda: provider)
    result = await models.generate(**arguments(evidence))
    gate = decide_reflection(
        result.draft, source_path_metrics(["src_1"], {"src_1": [path()]}),
        source_binding_valid=True, evidence_manifest_hash=evidence.manifest_hash, thresholds=ReflectionThresholds(),
    )
    reviewed = await models.reflect(**arguments(evidence), draft=result.draft, gate=gate)
    assert reviewed.decision.action == "accept"
    assert len(provider.calls) == 2
    packets = [json.loads(call["user_prompt"]) for call in provider.calls]
    assert packets[0]["evidence"] == packets[1]["evidence"]
    assert packets[1]["evidence"][0]["text"].endswith("389 days and 25.5%.")
    assert list(packets[0])[-1] == "history_summary"
    assert all(call["fallback"] is None for call in provider.calls)
    assert all("current user question second" in call["system_prompt"] for call in provider.calls)
    assert result.model_audit["citation_judge_model_call_count"] == 0
    assert reviewed.model_audit["evidence_manifest_hash"] == result.model_audit["evidence_manifest_hash"]
    assert "one compact whole-answer decision" in provider.calls[1]["system_prompt"]
    for call in provider.calls:
        assert "including parenthetical definitions" in call["system_prompt"]
        assert "Do not substitute a value derived from a rounded percentage" in call["system_prompt"]
        assert "User-requested calculations remain allowed" in call["system_prompt"]


@pytest.mark.asyncio
@pytest.mark.parametrize("configured_cap,expected", [(4096, 4096), (12000, 12000), (16000, 12000)])
async def test_reflection_component_uses_available_cap_without_extending_time_or_rounds(configured_cap, expected):
    from types import SimpleNamespace
    from app.services.reflection_agent import ReflectionAgentExecutor
    calls = []
    async def review(**kwargs):
        calls.append(kwargs)
        return "unit-test-review"
    executor = ReflectionAgentExecutor.__new__(ReflectionAgentExecutor)
    executor.settings = SimpleNamespace(agent_reflection_timeout_seconds=90)
    executor.phase_reflection_calls = 0
    executor.models = SimpleNamespace(reflect=review)
    executor.model_arguments = lambda evidence: {"evidence": evidence, "max_tokens": configured_cap, "timeout_seconds": 240, "controls": {}}
    evidence = manifest()
    result = await executor.reflect(evidence, draft(), None, 2)
    assert result == "unit-test-review" and len(calls) == 1
    assert calls[0]["max_tokens"] == expected and calls[0]["timeout_seconds"] == 90
    assert calls[0]["controls"]["remaining_reflection_rounds"] == 2
    assert calls[0]["evidence"] is evidence and executor.phase_reflection_calls == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("reply", [
    {"answer": "old unstructured output"},
    {**draft().model_dump(mode="json"), "unexpected_control": "unit-test secret"},
    draft(question_relevance=0.9).model_dump(mode="json") | {"self_assessment": {"question_relevance": "high"}},
])
async def test_model_schema_failures_are_safe_and_not_filled_with_high_scores(reply):
    provider = Provider([reply, reply])
    with pytest.raises(AnswerReviewModelError, match="answer_generation_schema_invalid") as caught:
        await AnswerReflectionModels(lambda: provider).generate(**arguments(manifest()))
    assert "unit-test secret" not in str(caught.value)
    assert len(provider.calls) == 2
    assert "unit-test secret" not in str(caught.value.schema_repair)


@pytest.mark.asyncio
async def test_model_timeout_and_provider_error_remain_technical_failures():
    provider = Provider([], delay=0.2)
    kwargs = {**arguments(manifest()), "timeout_seconds": 0.01}
    with pytest.raises(AnswerReviewModelError, match="answer_generation_timeout"):
        await AnswerReflectionModels(lambda: provider).generate(**kwargs)
    provider = Provider([], error=RuntimeError("private unit-test upstream response"))
    with pytest.raises(AnswerReviewModelError, match="answer_generation_unavailable") as caught:
        await AnswerReflectionModels(lambda: provider).generate(**arguments(manifest()))
    assert "upstream response" not in str(caught.value)
    assert "insufficient" not in str(caught.value)


@pytest.mark.asyncio
@pytest.mark.parametrize("cause_code,expected", [
    ("incomplete_max_tokens", "output_limit"), ("provider_refusal", "provider_refusal"),
    ("invalid_stop_reason", "provider_completion_invalid"), ("apitimeouterror", "timeout"),
    ("internalservererror", "unavailable"),
])
async def test_completion_and_transport_failures_keep_safe_classification(cause_code, expected):
    from app.services.error_sanitizer import ExternalServiceError
    cause = ExternalServiceError(service="anthropic", phase="sdk_messages_completion",
        status_code=503 if expected == "unavailable" else None, error_code=cause_code, retryable=False)
    wrapped = RuntimeError("unit-test-private-provider-body")
    wrapped.__cause__ = cause
    provider = Provider([], error=wrapped)
    with pytest.raises(AnswerReviewModelError) as caught:
        await AnswerReflectionModels(lambda: provider).generate(**arguments(manifest()))
    assert caught.value.code == expected
    assert caught.value.external_failure["error_code"] == cause_code
    assert caught.value.external_failure["phase"] == "sdk_messages_completion"
    assert caught.value.status_code == (503 if expected == "unavailable" else None)
    assert "unit-test-private" not in str(caught.value)
    assert "unit-test-private" not in str(caught.value.external_failure)


@pytest.mark.asyncio
async def test_reflection_rejects_stale_draft_identity_before_provider_call():
    evidence = manifest()
    old = draft()
    gate = decide_reflection(
        old, source_path_metrics(["src_1"], {"src_1": [path()]}), source_binding_valid=True,
        evidence_manifest_hash=evidence.manifest_hash, thresholds=ReflectionThresholds(),
    )
    provider = Provider([])
    with pytest.raises(AnswerReviewModelError, match="input_identity_mismatch"):
        await AnswerReflectionModels(lambda: provider).reflect(
            **arguments(evidence), draft=draft(needs_reflection=True), gate=gate,
        )
    assert provider.calls == []


def shape_error(code="json_decode_error"):
    from app.services.embeddings import ProviderJSONShapeError
    return ProviderJSONShapeError({"error_code": code, "field_path": "$", "contains_code_fence": True,
        "sha256": "unit-test-private-fingerprint", "body": "unit-test-private-provider-body"})


@pytest.mark.asyncio
async def test_generation_repairs_json_shape_once_with_identical_evidence():
    evidence = manifest("Complete evidence including its final fact.")
    provider = Provider([shape_error(), draft().model_dump(mode="json")])
    result = await AnswerReflectionModels(lambda: provider).generate(**arguments(evidence))
    assert result.model_audit["model_call_count"] == 2
    assert result.model_audit["schema_repair"]["attempted"] is True
    packets = [json.loads(call["user_prompt"]) for call in provider.calls]
    for key in ("current_user", "evidence", "history_summary"):
        assert packets[0][key] == packets[1][key]
    assert all(call["fallback"] is None and call["max_tokens"] == 4096 for call in provider.calls)
    assert "unit-test-private" not in str(provider.calls)
    assert "unit-test-private" not in str(result.model_audit)


@pytest.mark.asyncio
async def test_generation_repairs_field_shape_but_not_unknown_source_membership():
    malformed = {**draft().model_dump(mode="json"), "private-extra-field": "unit-test-secret"}
    provider = Provider([malformed, draft().model_dump(mode="json")])
    repaired = await AnswerReflectionModels(lambda: provider).generate(**arguments(manifest()))
    assert repaired.model_audit["model_call_count"] == 2
    failure = repaired.model_audit["schema_repair"]["failures"][0]
    assert failure["error_code"] == "output_schema_invalid"
    assert "private-extra-field" not in str(failure) and "unit-test-secret" not in str(provider.calls)
    unknown = draft().model_dump(mode="json")
    next(unit for unit in unknown["answer_units"] if unit["kind"] == "factual")["source_handles"] = ["src_999"]
    provider = Provider([unknown, draft().model_dump(mode="json")])
    with pytest.raises(AnswerReviewModelError, match="schema_invalid"):
        await AnswerReflectionModels(lambda: provider).generate(**arguments(manifest()))
    assert len(provider.calls) == 1


@pytest.mark.asyncio
async def test_factual_introduction_repair_identifies_exact_source_constraint():
    invalid = draft().model_dump(mode="json")
    unit = next(unit for unit in invalid["answer_units"] if unit["kind"] == "factual")
    unit["kind"] = "framing"
    provider = Provider([invalid, draft().model_dump(mode="json")])
    result = await AnswerReflectionModels(lambda: provider).generate(**arguments(manifest()))
    failure = result.model_audit["schema_repair"]["failures"][0]
    assert failure["field_errors"][0]["type"] == "nonfactual_source_handles_forbidden"
    packets = [json.loads(call["user_prompt"]) for call in provider.calls]
    assert packets[0]["evidence"] == packets[1]["evidence"]
    assert next(unit for unit in result.draft.answer_units if unit.kind == "factual").source_handles
    from app.reflection_contracts import AnswerUnit
    schema = AnswerUnit.model_json_schema()
    assert schema["properties"]["source_handles"]["uniqueItems"] is True
    assert schema["allOf"][0]["then"]["properties"]["source_handles"]["minItems"] == 1
    assert schema["allOf"][0]["else"]["properties"]["source_handles"]["maxItems"] == 0


@pytest.mark.asyncio
async def test_generation_format_repair_is_bounded_and_not_used_for_protocol_errors():
    provider = Provider([shape_error(), shape_error(), draft().model_dump(mode="json")])
    with pytest.raises(AnswerReviewModelError) as caught:
        await AnswerReflectionModels(lambda: provider).generate(**arguments(manifest()))
    assert caught.value.code == "schema_invalid" and caught.value.model_call_count == 2
    assert len(provider.calls) == 2 and len(caught.value.schema_repair["failures"]) == 2
    provider = Provider([shape_error("text_block_required"), draft().model_dump(mode="json")])
    with pytest.raises(AnswerReviewModelError):
        await AnswerReflectionModels(lambda: provider).generate(**arguments(manifest()))
    assert len(provider.calls) == 1


@pytest.mark.asyncio
async def test_format_repair_shares_original_timeout_and_reflection_gets_no_extra_call():
    provider = Provider([shape_error(), draft().model_dump(mode="json")], delay=0.03)
    with pytest.raises(AnswerReviewModelError, match="timeout") as caught:
        await AnswerReflectionModels(lambda: provider).generate(**{**arguments(manifest()), "timeout_seconds": 0.045})
    assert caught.value.model_call_count == 2
    evidence = manifest()
    candidate = draft()
    gate = decide_reflection(candidate, source_path_metrics(["src_1"], {"src_1": [path()]}),
        source_binding_valid=True, evidence_manifest_hash=evidence.manifest_hash, thresholds=ReflectionThresholds())
    provider = Provider([shape_error(), decision().model_dump(mode="json")])
    with pytest.raises(AnswerReviewModelError, match="schema_invalid"):
        await AnswerReflectionModels(lambda: provider).reflect(**arguments(evidence), draft=candidate, gate=gate)
    assert len(provider.calls) == 1
