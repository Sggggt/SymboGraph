"""Compile a bounded, request-specific grammar for attested lexical choices."""
from __future__ import annotations

from typing import Annotated, Literal, Union

from pydantic import Field, create_model

from app.retrieval_control_contracts import (
    ControlContract, LexicalPatch, LexicalPatchItem, control_hash,
)


PROTOCOL = 'lexical_patch_selection_v2'
OPERATIONS = ('replace_surface', 'add_attested_alias', 'qualify', 'locator_probe')


class StopChoice(ControlContract):
    operation: Literal['none_supported', 'need_scope_clarification']


def selection_model(*, task, strategy, candidates):
    strategy.validate_task(task)
    by_id = {item.id: item for item in candidates}
    facet_ids = {item.facet_id for item in candidates}
    if (not 1 <= len(candidates) <= 6 or len(by_id) != len(candidates)
            or not 1 <= len(facet_ids) <= 2 or not facet_ids <= {item.id for item in task.requirements}):
        raise ValueError('lexical_choice_candidate_scope_invalid')
    fields = {}
    for index, facet in enumerate(item for item in task.requirements if item.id in facet_ids):
        variants = [StopChoice]
        removable = tuple(term.id for term in strategy.terms if term.facet_id == facet.id)
        for operation in OPERATIONS:
            ids = tuple(sorted(item.id for item in candidates if item.facet_id == facet.id
                and operation in item.permitted_operations
                and (item.relation != 'related_locator' or operation == 'locator_probe')))
            if not ids:
                continue
            shape = {'operation': (Literal[operation], ...),
                'candidate_ids': (tuple[Literal[ids], ...], Field(min_length=1,max_length=min(3,len(ids)),
                    json_schema_extra={'uniqueItems':True}))}
            if operation == 'replace_surface':
                shape['remove_term_ids'] = ((tuple[Literal[removable], ...] if removable else tuple[()]),
                    Field(default=(),max_length=min(4,len(removable)),json_schema_extra={'uniqueItems':True}))
            variants.append(create_model(f'Facet{index}_{operation}',__base__=ControlContract,**shape))
        choice = Annotated[Union[tuple(variants)], Field(discriminator='operation')] if len(variants)>1 else StopChoice
        fields[f'facet_{index}'] = (choice,Field(alias=facet.id))
    choices = create_model('LexicalFacetChoices',__base__=ControlContract,**fields)
    return create_model('LexicalRepairSelection',__base__=ControlContract,
        protocol_version=(Literal['lexical_patch_selection_v2'],PROTOCOL),choices=(choices,...))


def project_selection(selection):
    payload = selection.model_dump(mode='json',by_alias=True)
    patches=[]
    clarification=False
    for facet_id,choice in payload['choices'].items():
        operation=choice['operation']
        if operation=='need_scope_clarification':
            clarification=True
        elif operation!='none_supported':
            candidate_ids=choice['candidate_ids']
            remove_ids=choice.get('remove_term_ids',[])
            if len(set(candidate_ids))!=len(candidate_ids) or len(set(remove_ids))!=len(remove_ids):
                raise ValueError('lexical_choice_duplicate_reference')
            patches.append(LexicalPatchItem(facet_id=facet_id,operation=operation,
                candidate_ids=tuple(candidate_ids),remove_term_ids=tuple(remove_ids)))
    patch=LexicalPatch(outcome='need_scope_clarification' if clarification else 'patch' if patches else 'none_supported',
        patches=tuple(patches) if not clarification else ())
    audit={'protocol_version':'lexical_choice_projection_v1','selection':payload,
        'selection_hash':control_hash(payload),'canonical_patch_hash':control_hash(patch.model_dump(mode='json')),
        'clarification_precedence':clarification,'additional_model_calls':0}
    audit['audit_hash']=control_hash(audit)
    return patch,audit


def replay_choice_projection(audit,*,task,strategy,candidates):
    output_type=selection_model(task=task,strategy=strategy,candidates=candidates)
    patch,replayed=project_selection(output_type.model_validate(audit['selection']))
    if replayed != audit:
        raise ValueError('lexical_choice_projection_changed')
    return patch
