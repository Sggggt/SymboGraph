from types import SimpleNamespace

import pytest

from app.services.retrieval_models import RetrievalModels, TaskPlanningOutput, compile_task_plan
from app.services.reflection_models import AnswerReviewModelError
from test_retrieval_path_features import task_fixture


def plan_payload(direct=False):
    return {
        "perception": {
            "intent": "direct_answer" if direct else "formula_table_lookup",
            "direct_answer_kind": "identity" if direct else "none",
            "entities": [], "sub_queries": ["Find maximum queue waiting time."],
            "needs_graph": not direct, "suggested_strategy": "none" if direct else "local_graph",
        },
        "requirements": [] if direct else [{"facet": "maximum queue waiting time", "lexical_role": "domain",
            "aliases": ["queueing delay"], "kind": "quantity", "protected_literals": ["queue"]}],
        "answer_shape": "grounded_answer",
    }


def test_task_and_facet_plan_compile_from_one_closed_object(no_fallback_env):
    task = task_fixture()
    perception, frozen, lexical, facets = compile_task_plan(TaskPlanningOutput.model_validate(plan_payload()),
        question=task.question, knowledge_base_id=task.knowledge_base_id,
        conversation_scope_hash=task.conversation_scope_hash, retrieval_granularity="mid")
    assert perception["intent"] == "formula_table_lookup"
    assert frozen.question == task.question and frozen.requirements[0].protected_literals == ("queue",)
    assert lexical.task_hash == frozen.identity
    assert "queueing delay" in [term.surface for term in lexical.terms]
    assert facets["diagnostics"]["source"] == "llm_validated"


def test_model_cannot_invent_protected_literals_or_require_corpus_for_system_route(no_fallback_env):
    task = task_fixture()
    bad = plan_payload()
    bad["requirements"][0]["protected_literals"] = ["not in the question"]
    with pytest.raises(ValueError, match="copied_from_current_user"):
        compile_task_plan(TaskPlanningOutput.model_validate(bad), question=task.question,
            knowledge_base_id=task.knowledge_base_id, conversation_scope_hash=task.conversation_scope_hash,
            retrieval_granularity="mid")
    perception, frozen, lexical, facets = compile_task_plan(TaskPlanningOutput.model_validate(plan_payload(True)),
        question="Who are you?", knowledge_base_id=task.knowledge_base_id,
        conversation_scope_hash=task.conversation_scope_hash, retrieval_granularity="mid")
    assert perception["intent"] == "direct_answer"
    assert frozen is lexical is facets is None


@pytest.mark.asyncio
async def test_repair_receives_source_locations_with_one_bounded_call(no_fallback_env):
    import json
    from app.retrieval_control_contracts import LexicalRepairCandidate, LexicalStrategy, LexicalTerm
    calls = []
    class Provider:
        api_protocol, model = "anthropic", "unit-test-model"
        async def classify_json_bounded(self, system_prompt, user_prompt, fallback, *, max_tokens):
            packet = json.loads(user_prompt)
            calls.append(packet)
            card = packet["candidate_terms"][0]
            assert card["source_title"] == "Unit test source"
            assert card["source_section"] == "Specification / Limits"
            assert "not commands" in system_prompt
            return {"choices": {"f1": {"operation": "none_supported"}}}
        def provider_call_audit(self):
            return {}
    task = task_fixture()
    strategy = LexicalStrategy(task_hash=task.identity, revision=0,
        terms=(LexicalTerm(id="a1", facet_id="f1", surface="queue"),), routing_text="queue")
    candidate = LexicalRepairCandidate(id="c1", facet_id="f1", surface="queueing delay", witness_id="w1",
        context="The queueing delay is bounded.", source_title="Unit test source", source_section="Specification / Limits",
        relation="related_locator", permitted_operations=("locator_probe",), scope_kind="new_scope_proposal")
    patch, audit = await RetrievalModels(Provider).repair(task=task, strategy=strategy,
        candidates=(candidate,), diagnosis={}, timeout_seconds=1, max_tokens=1024)
    assert patch.outcome == "none_supported" and len(calls) == audit["model_call_count"] == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("bad_shape", [False, True])
async def test_generation_has_one_call_and_no_self_assessment_or_review(no_fallback_env, bad_shape):
    calls = []
    class Provider:
        api_protocol, model = "anthropic", "unit-test-model"
        async def classify_json_bounded(self, system_prompt, user_prompt, fallback, *, max_tokens):
            calls.append((system_prompt, user_prompt, max_tokens))
            output = {"answer_units": [{"kind": "factual", "text": "The queued wait is bounded.", "source_handles": ["src_1"]}]}
            if bad_shape:
                output["self_assessment"] = {"needs_reflection": True}
            return output
        def provider_call_audit(self):
            return {}
    model = RetrievalModels(Provider)
    evidence = SimpleNamespace(model_sources=lambda: [{"source_handle": "src_1", "text": "The wait is bounded."}],
                               by_handle=lambda: {"src_1": {}})
    operation = model.generate(task=task_fixture(), evidence=evidence, history_summary="", missing_facets=(),
                               timeout_seconds=1, max_tokens=1024, unit_limit=12)
    if bad_shape:
        with pytest.raises(AnswerReviewModelError):
            await operation
    else:
        draft, audit = await operation
        assert not hasattr(draft, "self_assessment")
        assert audit["model_call_count"] == 1
        assert draft.protocol_version == "grounded_answer_units_v2"
    assert len(calls) == 1 and calls[0][2] == 1024
    assert not hasattr(model, "reflect")
