import pytest

from app.retrieval_control_contracts import LexicalPatch, LexicalPatchItem, LexicalRepairCandidate
from app.services.lexical_patch import compile_lexical_patch
from test_retrieval_path_features import strategy, task_fixture


def repair_candidate(surface="queueing delay", relation="proposed_equivalence", operations=("replace_surface",)):
    return LexicalRepairCandidate(id="c1", facet_id="f1", surface=surface,
        witness_id="unit-test-raw-span", context="Waiting before processing starts.",
        relation=relation, permitted_operations=operations, scope_kind="new_scope_proposal")


def patch(operation="replace_surface", remove=("old",)):
    return LexicalPatch(outcome="patch", patches=(
        LexicalPatchItem(facet_id="f1", operation=operation, remove_term_ids=remove, candidate_ids=("c1",)),))


def test_attested_patch_changes_route_and_keeps_fixed_requirement():
    task = task_fixture()
    before = strategy(task, ("old",))
    after = compile_lexical_patch(task=task, before=before, patch=patch(), candidates=(repair_candidate(),))
    assert after.task_hash == before.task_hash == task.identity
    assert after.revision == 1
    assert task.requirements[0].text in after.routing_text
    assert "queueing delay" in after.routing_text
    assert after.terms[0].witness_id == "unit-test-raw-span"
    assert before.terms[0].surface == "old"


def test_related_locator_cannot_be_relabelled_as_an_equivalent_surface():
    task = task_fixture()
    before = strategy(task, ("old",))
    candidate = repair_candidate(relation="related_locator", operations=("replace_surface", "locator_probe"))
    with pytest.raises(ValueError, match="not_equivalent"):
        compile_lexical_patch(task=task, before=before, patch=patch(), candidates=(candidate,))
    after = compile_lexical_patch(task=task, before=before,
        patch=patch("locator_probe", ()), candidates=(candidate,))
    assert after.terms == before.terms
    assert after.locator_ids == ("unit-test-raw-span",)


def test_same_surface_with_new_ids_is_no_progress():
    task = task_fixture()
    before = strategy(task, ("old",))
    with pytest.raises(ValueError, match="no_new_information"):
        compile_lexical_patch(task=task, before=before, patch=patch(),
                              candidates=(repair_candidate(surface="  OLD "),))


def test_patch_cannot_use_unprovided_candidate_or_reset_its_budget():
    task = task_fixture()
    before = strategy(task, ("old",))
    with pytest.raises(ValueError, match="outside_scope"):
        compile_lexical_patch(task=task, before=before, patch=patch(), candidates=())
    with pytest.raises(ValueError, match="budget_exhausted"):
        compile_lexical_patch(task=task, before=before.model_copy(update={"revision": 2}),
            patch=patch(), candidates=(repair_candidate(),))
    assert compile_lexical_patch(task=task, before=before,
        patch=LexicalPatch(outcome="none_supported"), candidates=()) is None
