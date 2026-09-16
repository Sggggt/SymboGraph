"""Compile model-selected, attested lexical changes without changing the task."""
from __future__ import annotations

import unicodedata

from app.retrieval_control_contracts import (
    LexicalPatch, LexicalRepairCandidate, LexicalStrategy, LexicalTerm, TaskContract, control_hash,
)


def _normalized(text):
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


class LexicalPatchNoProgress(ValueError):
    pass


def strategy_semantic_signature(strategy: LexicalStrategy):
    return control_hash({"task_hash": strategy.task_hash,
        "terms": sorted({(term.facet_id, _normalized(term.surface)) for term in strategy.terms}),
        "locator_ids": sorted(set(strategy.locator_ids))})


def compile_lexical_patch(*, task: TaskContract, before: LexicalStrategy, patch: LexicalPatch,
                          candidates: tuple[LexicalRepairCandidate, ...]) -> LexicalStrategy | None:
    before.validate_task(task)
    if patch.outcome != "patch":
        return None
    if before.revision >= 2:
        raise ValueError("lexical_revision_budget_exhausted")
    by_candidate = {item.id: item for item in candidates}
    if len(by_candidate) != len(candidates) or len(candidates) > 6:
        raise ValueError("lexical_candidate_scope_invalid")
    requirements = {item.id: item for item in task.requirements}
    terms = {term.id: term for term in before.terms}
    locators = list(before.locator_ids)
    for operation in patch.patches:
        if operation.facet_id not in requirements:
            raise ValueError("lexical_patch_outside_fixed_task")
        if any(term_id not in terms or terms[term_id].facet_id != operation.facet_id
               for term_id in operation.remove_term_ids):
            raise ValueError("lexical_patch_remove_scope_invalid")
        if operation.operation != "replace_surface" and operation.remove_term_ids:
            raise ValueError("lexical_patch_operation_cannot_remove_terms")
        additions = []
        for candidate_id in operation.candidate_ids:
            candidate = by_candidate.get(candidate_id)
            if candidate is None or candidate.facet_id != operation.facet_id:
                raise ValueError("lexical_patch_candidate_outside_scope")
            if operation.operation not in candidate.permitted_operations:
                raise ValueError("lexical_patch_operation_not_permitted")
            if candidate.relation == "related_locator" and operation.operation != "locator_probe":
                raise ValueError("related_locator_is_not_equivalent_surface")
            if operation.operation == "locator_probe":
                if candidate.witness_id not in locators:
                    locators.append(candidate.witness_id)
                continue
            term_id = "a_" + control_hash({"facet": candidate.facet_id, "surface": _normalized(candidate.surface)})[:24]
            additions.append(LexicalTerm(id=term_id, facet_id=candidate.facet_id, surface=candidate.surface,
                                        source="attested", witness_id=candidate.witness_id))
        for term_id in operation.remove_term_ids:
            terms.pop(term_id)
        for term in additions:
            terms[term.id] = term
    active_terms = tuple(sorted(terms.values(), key=lambda item: (item.facet_id, item.id)))
    # Every fixed requirement remains in the route even if one of its old
    # lexical variants was removed. The model does not author routing_text.
    routing_parts = [facet.text for facet in task.requirements]
    routing_parts.extend(term.surface for term in active_terms if term.source == "attested")
    routing_text = "; ".join(dict.fromkeys(routing_parts))
    after = LexicalStrategy(task_hash=before.task_hash, revision=before.revision + 1, terms=active_terms,
                            routing_text=routing_text, locator_ids=tuple(locators))
    if strategy_semantic_signature(before) == strategy_semantic_signature(after):
        raise LexicalPatchNoProgress("lexical_patch_no_new_information")
    return after
