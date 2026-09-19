"""Closed contracts for retrieval-end diagnosis; model plans are not evidence."""
from __future__ import annotations

import hashlib
import json
import math
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator, field_validator, model_serializer
from app.reflection_contracts import AnswerUnit, SourceHandle


UnitScore = Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]
Positive = Annotated[float, Field(gt=0, allow_inf_nan=False)]
NonNegative = Annotated[float, Field(ge=0, allow_inf_nan=False)]
Identifier = Annotated[str, Field(min_length=1, max_length=160)]

RUN_OBSERVATION_PROTOCOLS = {
    'retrieval_state_transition': ('retrieval_fsm_transition_v1',),
    'retrieval_gate': ('retrieval_gate_observation_v1',),
    'retrieval_scope_targets': ('source_scope_target_plan_v1',),
    'retrieval_packing_repair': ('retrieval_packing_repair_v1', 'scope_interval_repacking_v1'),
    'retrieval_generation_packing': ('feature_preserving_generation_packing_v1', 'scope_preserving_generation_packing_v1'),
    'retrieval_repair_request': ('retrieval_repair_request_v1',),
    'retrieval_lexical_patch': ('retrieval_patch_observation_v1',),
    'retrieval_sufficiency': ('retrieval_sufficiency_call_v1', 'retrieval_sufficiency_call_v2'),
    'retrieval_scope_resolution': ('source_location_call_v1',),
}


def control_hash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


class ControlContract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class EvidenceInterval(ControlContract):
    knowledge_base_id: Identifier
    document_version_id: Identifier
    start: int = Field(ge=0, strict=True)
    end: int = Field(gt=0, strict=True)

    @model_validator(mode='after')
    def positive_extent(self):
        if self.start >= self.end:
            raise ValueError('evidence_interval_empty_or_reversed')
        return self


class EvidenceScopeFact(ControlContract):
    """Resolver-owned location facts; never accepted as model evidence."""
    id: Identifier
    knowledge_base_id: Identifier
    kind: Literal['document', 'section', 'text', 'table', 'formula', 'code', 'figure', 'caption']
    resolution: Literal['verified', 'unresolved', 'unsupported']
    extent_complete: bool = False
    intervals: tuple[EvidenceInterval, ...] = Field(default=(), max_length=32768)
    witness_ids: tuple[Identifier, ...] = Field(default=(), max_length=32768)

    @model_validator(mode='after')
    def resolver_evidence(self):
        if self.resolution == 'verified':
            if not self.witness_ids or (not self.intervals and not self.extent_complete):
                raise ValueError('verified_scope_requires_ranges_and_witnesses')
        elif self.intervals or self.extent_complete:
            raise ValueError('unresolved_scope_cannot_assert_ranges')
        if any(item.knowledge_base_id != self.knowledge_base_id for item in self.intervals):
            raise ValueError('evidence_scope_cross_knowledge_base')
        return self


class EvidenceScopeExpression(ControlContract):
    op: Literal['scope', 'union', 'intersection']
    scope_id: Identifier | None = None
    children: tuple['EvidenceScopeExpression', ...] = Field(default=(), max_length=8)

    @model_validator(mode='after')
    def bounded_expression(self):
        if self.op == 'scope':
            if not self.scope_id or self.children:
                raise ValueError('scope_expression_leaf_invalid')
        elif self.scope_id is not None or len(self.children) < 2:
            raise ValueError('scope_expression_composition_invalid')
        pending, count = [(self, 1)], 0
        while pending:
            item, depth = pending.pop()
            count += 1
            if count > 64 or depth > 8:
                raise ValueError('scope_expression_capacity_exceeded')
            pending.extend((child, depth + 1) for child in item.children)
        return self


class EvidenceScopeCoverage(ControlContract):
    protocol_version: Literal['evidence_scope_algebra_v1'] = 'evidence_scope_algebra_v1'
    state: Literal['satisfied', 'unsatisfied', 'unknown']
    mode: Literal['overlap', 'complete']
    input_hash: str = Field(pattern=r'^[a-f0-9]{64}$')
    usable_intervals: tuple[EvidenceInterval, ...]
    missing_intervals: tuple[EvidenceInterval, ...]
    scope_extent_known: bool
    reason: Literal['covered', 'required_range_missing', 'no_scope_overlap', 'empty_resolved_scope', 'scope_not_fully_resolved']
    semantic_sufficiency_claimed: Literal[False] = False


class EvidenceIntervalCandidate(ControlContract):
    id: Identifier
    interval: EvidenceInterval
    cost: int = Field(ge=0, strict=True)
    witness_ids: tuple[Identifier, ...] = Field(min_length=1, max_length=32)


class EvidenceCompletionPlan(ControlContract):
    protocol_version: Literal['interval_scope_completion_v1'] = 'interval_scope_completion_v1'
    status: Literal['ready', 'already_covered', 'scope_unresolved', 'not_coverable', 'over_budget']
    input_hash: str = Field(pattern=r'^[a-f0-9]{64}$')
    selected_ids: tuple[Identifier, ...]
    total_cost: int | None = Field(default=None, ge=0)
    uncovered_intervals: tuple[EvidenceInterval, ...] = ()
    optimality_scope: Literal['single_interval_additive_nonnegative_cost'] = 'single_interval_additive_nonnegative_cost'
    executor_validation_required: Literal[True] = True
    semantic_sufficiency_claimed: Literal[False] = False


class SourceScopeSelector(ControlContract):
    """A user location declaration, not a resolver fact or a search command."""
    kind: Literal['document', 'section', 'text', 'table', 'formula', 'code', 'figure', 'caption']
    reference: str = Field(min_length=1, max_length=256)
    match: Literal['title', 'label', 'kind', 'role'] = 'title'
    role: Literal['summary','detail'] | None = None

    @model_validator(mode='after')
    def role_shape(self):
        if (self.match=='role') != (self.role is not None) or (self.match=='role' and self.kind!='section'):
            raise ValueError('source_scope_role_shape_invalid')
        return self

    @model_serializer(mode='wrap')
    def preserve_title_scope_identity(self,handler):
        payload=handler(self)
        if self.role is None:
            payload.pop('role',None)
        return payload


class SourceLocationCandidate(ControlContract):
    id: Identifier
    node_id: Identifier
    document_version_id: Identifier
    kind: Literal['document','section','text','table','formula','code','figure','caption']
    document_title: str = Field(max_length=160)
    title: str = Field(max_length=160)
    excerpt: str = Field(max_length=160)
    excerpt_span: tuple[int,int] | None = None
    navigation_protocol: Literal['structure_location_labels_v1'] | None = None
    source_order: int | None = Field(default=None,ge=0)
    parent_titles: tuple[Annotated[str,Field(max_length=80)],...] | None = Field(default=None,max_length=3)

    @model_serializer(mode='wrap')
    def preserve_earlier_locator_packet(self,handler):
        payload=handler(self)
        if self.navigation_protocol is None:
            for key in ('navigation_protocol','source_order','parent_titles'):
                payload.pop(key,None)
        return payload

    @model_validator(mode='after')
    def excerpt_identity(self):
        if (self.navigation_protocol is None and (self.source_order is not None or self.parent_titles is not None)
            or self.navigation_protocol is not None and (self.source_order is None or self.parent_titles is None)):
            raise ValueError('source_location_navigation_identity_invalid')
        if self.excerpt_span is None:
            if self.excerpt:
                raise ValueError('source_location_excerpt_span_missing')
        elif not 0 <= self.excerpt_span[0] < self.excerpt_span[1] or self.excerpt_span[1]-self.excerpt_span[0]!=len(self.excerpt):
            raise ValueError('source_location_excerpt_span_invalid')
        return self


class SourceLocationSelector(ControlContract):
    id: Identifier
    selector: SourceScopeSelector
    selector_hash: str = Field(pattern=r'^[a-f0-9]{64}$')
    request_hash: str = Field(pattern=r'^[a-f0-9]{64}$')
    path: tuple[int,...] = Field(max_length=8)
    candidates: tuple[SourceLocationCandidate,...] = Field(max_length=4)
    eligible_candidate_count: int = Field(ge=0)


class SourceLocationRequest(ControlContract):
    protocol_version: Literal['source_location_request_v1'] = 'source_location_request_v1'
    task_hash: str = Field(pattern=r'^[a-f0-9]{64}$')
    source_index_hash: str = Field(pattern=r'^[a-f0-9]{64}$')
    source_scope_hash: str = Field(pattern=r'^[a-f0-9]{64}$')
    source_chunk_ids: tuple[Identifier,...] = Field(min_length=1,max_length=32768)
    candidate_protocol: Literal['whole_query_location_candidates_v1','fixed_facet_location_candidates_v1'] = 'whole_query_location_candidates_v1'
    candidate_input_hash: str | None = Field(default=None,pattern=r'^[a-f0-9]{64}$')
    reference_alignment_protocol: Literal['source_reference_alignment_v1'] | None = None
    selectors: tuple[SourceLocationSelector,...] = Field(min_length=1,max_length=8)

    @model_serializer(mode='wrap')
    def preserve_earlier_candidate_request(self,handler):
        payload=handler(self)
        if self.candidate_protocol=='whole_query_location_candidates_v1':
            payload.pop('candidate_protocol',None)
            payload.pop('candidate_input_hash',None)
        if self.reference_alignment_protocol is None:
            payload.pop('reference_alignment_protocol', None)
        return payload

    @model_validator(mode='after')
    def unique_choices(self):
        if (self.candidate_protocol!='whole_query_location_candidates_v1') != (self.candidate_input_hash is not None):
            raise ValueError('source_location_candidate_input_identity_invalid')
        if len(set(self.source_chunk_ids))!=len(self.source_chunk_ids):
            raise ValueError('source_location_duplicate_source')
        if len({item.id for item in self.selectors})!=len(self.selectors) or len({item.selector_hash for item in self.selectors})!=len(self.selectors):
            raise ValueError('source_location_duplicate_selector')
        for item in self.selectors:
            if (not item.candidates or len({card.id for card in item.candidates})!=len(item.candidates)
                or len({card.node_id for card in item.candidates})!=len(item.candidates)
                or any(card.kind!=item.selector.kind for card in item.candidates)):
                raise ValueError('source_location_candidate_scope_invalid')
        return self


class SemanticScopeItem(ControlContract):
    selector_hash: str = Field(pattern=r'^[a-f0-9]{64}$')
    node_id: Identifier


class SemanticScopeSelection(ControlContract):
    protocol_version: Literal['semantic_source_location_v1'] = 'semantic_source_location_v1'
    observation_id: Identifier
    run_id: Identifier
    ledger_hash: str = Field(pattern=r'^[a-f0-9]{64}$')
    task_hash: str = Field(pattern=r'^[a-f0-9]{64}$')
    source_index_hash: str = Field(pattern=r'^[a-f0-9]{64}$')
    items: tuple[SemanticScopeItem,...] = Field(default=(),max_length=8)
    semantic_identity_proven: Literal[False] = False

    @model_validator(mode='after')
    def unique_selector(self):
        if len({item.selector_hash for item in self.items})!=len(self.items):
            raise ValueError('semantic_scope_duplicate_selector')
        return self


class SourceScopeRequest(ControlContract):
    op: Literal['scope', 'union', 'intersection'] = 'scope'
    selector: SourceScopeSelector | None = None
    children: tuple['SourceScopeRequest', ...] = Field(default=(), max_length=8)

    @model_validator(mode='after')
    def expression_shape(self):
        if (self.op == 'scope' and (self.selector is None or self.children)
                or self.op != 'scope' and (self.selector is not None or len(self.children) < 2)):
            raise ValueError('source_scope_request_shape_invalid')
        stack, count = [(self, 0)], 0
        while stack:
            node, depth = stack.pop()
            count += 1
            if depth > 8 or count > 64:
                raise ValueError('source_scope_request_capacity_exceeded')
            stack.extend((child, depth + 1) for child in node.children)
        return self


class SourceScopeObligation(ControlContract):
    op: Literal['coverage', 'all', 'any'] = 'coverage'
    scope: SourceScopeRequest | None = None
    mode: Literal['overlap', 'complete'] = 'overlap'
    children: tuple['SourceScopeObligation', ...] = Field(default=(), max_length=8)

    @model_validator(mode='after')
    def obligation_shape(self):
        if (self.op == 'coverage' and (self.scope is None or self.children)
                or self.op != 'coverage' and (self.scope is not None or len(self.children) < 2)):
            raise ValueError('source_scope_obligation_shape_invalid')
        stack, count = [(self, 0)], 0
        while stack:
            node, depth = stack.pop()
            count += 1
            if depth > 8 or count > 16:
                raise ValueError('source_scope_obligation_capacity_exceeded')
            stack.extend((child, depth + 1) for child in node.children)
        return self


class SourceScopeBinding(ControlContract):
    request_hash: str = Field(pattern=r'^[a-f0-9]{64}$')
    fact: EvidenceScopeFact
    node_ids: tuple[Identifier, ...] = Field(default=(), max_length=32768)
    source_identity_hash: str = Field(pattern=r'^[a-f0-9]{64}$')
    reason: Literal['resolved', 'no_verified_match', 'ambiguous', 'representation_incomplete', 'unsupported_representation']


class FacetScopeInput(ControlContract):
    facet_id: Identifier
    bindings: tuple[SourceScopeBinding, ...] = Field(min_length=1, max_length=16)


class FacetScopeStatus(ControlContract):
    facet_id: Identifier
    state: Literal['satisfied', 'unsatisfied', 'unknown']
    input_hash: str = Field(pattern=r'^[a-f0-9]{64}$')
    coverage: tuple[EvidenceScopeCoverage, ...] = Field(max_length=16)
    reason_codes: tuple[str, ...] = ()


class Requirement(ControlContract):
    id: Identifier
    text: str = Field(min_length=1, max_length=256)
    weight: Positive
    role: Literal["topic", "definition", "procedure", "quantity", "comparison", "source_role"] = "topic"
    protected_literals: tuple[str, ...] = ()
    source_roles: tuple[Literal["summary", "detail", "table", "formula", "code"], ...] = ()
    source_scope: SourceScopeObligation | None = None

    @model_serializer(mode='wrap')
    def preserve_legacy_scope(self, handler):
        payload = handler(self)
        if self.source_scope is None:
            payload.pop('source_scope', None)
        return payload


class ResponseConstraint(ControlContract):
    kind: Literal['no_speculation', 'insufficiency_notice', 'output_format', 'language', 'brevity']
    text: str = Field(min_length=1, max_length=1024)
    char_span: tuple[int, int]


class TaskContract(ControlContract):
    protocol_version: Literal["retrieval_task_v1"] = "retrieval_task_v1"
    knowledge_base_id: Identifier
    conversation_scope_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    question: str = Field(min_length=1, max_length=12000)
    requirements: tuple[Requirement, ...] = Field(min_length=1, max_length=8)
    allow_partial: bool = False
    retrieval_granularity: Literal["mid", "coarse"] = "mid"
    response_constraints: tuple[ResponseConstraint, ...] = Field(default=(), max_length=12)
    scope_protocol_version: Literal['task_source_scope_v1'] = 'task_source_scope_v1'
    source_reference_roles_hash: str | None = Field(default=None,pattern=r'^[a-f0-9]{64}$')

    @model_serializer(mode='wrap')
    def preserve_legacy_response_constraints(self, handler):
        payload = handler(self)
        if not self.response_constraints:
            payload.pop('response_constraints', None)
        if not any(facet.source_scope is not None for facet in self.requirements):
            payload.pop('scope_protocol_version', None)
        if self.source_reference_roles_hash is None:
            payload.pop('source_reference_roles_hash', None)
        return payload

    @model_validator(mode="after")
    def fixed_scope(self):
        if len({item.id for item in self.requirements}) != len(self.requirements):
            raise ValueError("duplicate_required_facet")
        if not math.isclose(sum(item.weight for item in self.requirements), 1.0, abs_tol=1e-6):
            raise ValueError("task_weights_must_sum_to_one")
        if "\x00" in self.question:
            raise ValueError("task_text_control_character")
        for constraint in self.response_constraints:
            start, end = constraint.char_span
            if not 0 <= start < end <= len(self.question) or self.question[start:end] != constraint.text:
                raise ValueError('task_response_constraint_span_invalid')
        for requirement in self.requirements:
            if requirement.source_scope is None:
                continue
            nodes = [requirement.source_scope]
            declarations = 0
            while nodes:
                node = nodes.pop()
                nodes.extend(node.children)
                if isinstance(node, SourceScopeObligation):
                    if node.scope is not None:
                        nodes.append(node.scope)
                elif node.selector is not None:
                    reference = node.selector.reference
                    if not reference.strip() or reference not in self.question or '\x00' in reference:
                        raise ValueError('source_scope_must_quote_current_question')
                    declarations += 1
                    if declarations > 32:
                        raise ValueError('task_source_scope_capacity_exceeded')
        return self

    @property
    def identity(self):
        return control_hash(self.model_dump(mode="json"))


class ScopedGenerationSource(ControlContract):
    source_handle: Identifier
    text_char_spans: tuple[tuple[int, int], ...] = Field(min_length=1, max_length=256)

    @model_validator(mode='after')
    def ranges(self):
        if any(not 0 <= start < end for start,end in self.text_char_spans):
            raise ValueError('generation_scope_range_invalid')
        return self


class ScopedGenerationRequirement(ControlContract):
    requirement_id: Identifier
    sources: tuple[ScopedGenerationSource, ...] = Field(max_length=256)


class GenerationSourceScopeGuidance(ControlContract):
    protocol_version: Literal['generation_source_scope_guidance_v1'] = 'generation_source_scope_guidance_v1'
    requirements: tuple[ScopedGenerationRequirement, ...] = Field(max_length=8)
    offset_unit: Literal['unicode_characters'] = 'unicode_characters'
    location_is_not_semantic_proof: Literal[True] = True


class LexicalTerm(ControlContract):
    id: Identifier
    facet_id: Identifier
    surface: str = Field(min_length=1, max_length=96)
    source: Literal["original", "attested", "hypothesis"] = "original"
    witness_id: Identifier | None = None

    @field_validator("surface")
    @classmethod
    def safe_surface(cls, value):
        if not value.strip() or "\x00" in value:
            raise ValueError("lexical_surface_invalid")
        return value.strip()

    @model_validator(mode="after")
    def attestation(self):
        if self.source == "attested" and not self.witness_id:
            raise ValueError("attested_term_missing_witness")
        return self


class LexicalStrategy(ControlContract):
    protocol_version: Literal["retrieval_lexical_strategy_v1"] = "retrieval_lexical_strategy_v1"
    task_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    revision: int = Field(ge=0, le=2)
    terms: tuple[LexicalTerm, ...] = Field(max_length=72)
    routing_text: str = Field(min_length=1, max_length=4096)
    locator_ids: tuple[Identifier, ...] = Field(default=(), max_length=6)

    @model_validator(mode="after")
    def term_ids(self):
        if len({item.id for item in self.terms}) != len(self.terms):
            raise ValueError("duplicate_lexical_term_id")
        return self

    def validate_task(self, task: TaskContract):
        if self.task_hash != task.identity:
            raise ValueError("lexical_task_identity_changed")
        known = {item.id for item in task.requirements}
        if any(term.facet_id not in known for term in self.terms):
            raise ValueError("lexical_term_outside_fixed_task")

    @property
    def identity(self):
        return control_hash(self.model_dump(mode="json"))


class ScoreBounds(ControlContract):
    lower: UnitScore
    upper: UnitScore

    @model_validator(mode="after")
    def ordered(self):
        if self.upper < self.lower:
            raise ValueError("score_bounds_reversed")
        return self


class GainBounds(ControlContract):
    lower: float = Field(ge=-1, le=1, allow_inf_nan=False)
    upper: float = Field(ge=-1, le=1, allow_inf_nan=False)

    @model_validator(mode="after")
    def ordered(self):
        if self.upper < self.lower:
            raise ValueError("gain_bounds_reversed")
        return self


class FacetOpportunity(ControlContract):
    facet_id: Identifier
    value: ScoreBounds


class PathFeatureCandidate(ControlContract):
    id: Identifier
    source_id: Identifier
    topic_group: Identifier
    source_valid: bool
    path_observed: bool = True
    opportunities: tuple[FacetOpportunity, ...] = Field(max_length=8)
    canonical_entry_distance: NonNegative
    # Physical coarse, mid, chunk distances. Membership is not a physical edge.
    physical_distances: tuple[NonNegative, NonNegative, NonNegative] = (0, 0, 0)
    matched_term_ids: tuple[Identifier, ...] = Field(default=(), max_length=72)
    routing_cost: float = Field(allow_inf_nan=False)
    depth: int = Field(default=0, ge=0)
    role_rank: int = Field(default=0, ge=0)
    mandatory: bool = False

    @model_validator(mode="after")
    def unique_facets(self):
        if len({item.facet_id for item in self.opportunities}) != len(self.opportunities):
            raise ValueError("duplicate_candidate_facet")
        return self


class DecisionPanel(ControlContract):
    id: Identifier
    candidates: tuple[PathFeatureCandidate, ...] = Field(max_length=256)
    selected_ids: tuple[Identifier, ...] = Field(max_length=256)
    limit: int = Field(ge=0, le=256)
    scope_kind: Literal["observed_eligible_candidates"] = "observed_eligible_candidates"
    routing_facet_weights: dict[Identifier, UnitScore] = Field(default_factory=dict)

    @model_serializer(mode='wrap')
    def preserve_legacy_serialization(self, handler):
        payload = handler(self)
        if not self.routing_facet_weights:
            payload.pop('routing_facet_weights', None)
        return payload

    @model_validator(mode="after")
    def selected_scope(self):
        ids = {item.id for item in self.candidates}
        if len(ids) != len(self.candidates) or len(set(self.selected_ids)) != len(self.selected_ids):
            raise ValueError("decision_panel_duplicate_id")
        if not set(self.selected_ids).issubset(ids) or len(self.selected_ids) > self.limit:
            raise ValueError("decision_panel_selection_invalid")
        return self


class PathEvaluationIdentity(ControlContract):
    graph_scope_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    vector_runtime_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    source_scope_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    canonical_vectors_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    match_protocol_hash: str = Field(pattern=r"^[a-f0-9]{64}$")


class PathEvaluationParameters(ControlContract):
    protocol_version: Literal["canonical_task_path_quality_v1", "canonical_task_path_quality_v2", "canonical_task_path_quality_v4"] = "canonical_task_path_quality_v1"
    identity: PathEvaluationIdentity
    root_scale: Positive = 1
    physical_scales: tuple[Positive, Positive, Positive] = (1, 1, 1)
    source_use_protocol: Literal['metadata_dominant_source_use_v1','metadata_dominant_source_use_v2'] = 'metadata_dominant_source_use_v1'
    scope_index_protocol: Literal['complete_structure_index_v1','task_structure_index_v1'] = 'complete_structure_index_v1'
    scope_inputs: tuple[FacetScopeInput, ...] = Field(default=(), max_length=8)
    packed_scope_intervals: tuple[EvidenceInterval, ...] = Field(default=(), max_length=256)
    scope_source_chunk_ids: tuple[Identifier, ...] = Field(default=(), max_length=32768)
    scope_selection: SemanticScopeSelection | None = None

    @model_serializer(mode='wrap')
    def preserve_legacy_scope(self, handler):
        payload = handler(self)
        if self.source_use_protocol == 'metadata_dominant_source_use_v1':
            payload.pop('source_use_protocol', None)
        if self.scope_index_protocol == 'complete_structure_index_v1':
            payload.pop('scope_index_protocol', None)
        if self.scope_selection is None:
            payload.pop('scope_selection',None)
        if self.protocol_version != 'canonical_task_path_quality_v4':
            payload.pop('scope_inputs', None)
            payload.pop('packed_scope_intervals', None)
            payload.pop('scope_source_chunk_ids', None)
        return payload

    @model_validator(mode='after')
    def scope_protocol(self):
        if self.protocol_version != 'canonical_task_path_quality_v4' and (self.scope_inputs or self.packed_scope_intervals or self.scope_source_chunk_ids or self.scope_selection):
            raise ValueError('scope_input_requires_new_evaluation_protocol')
        if self.scope_selection is not None and (not self.scope_inputs or self.scope_index_protocol!='task_structure_index_v1'):
            raise ValueError('semantic_scope_requires_task_structure_identity')
        return self


class FacetCoverage(ControlContract):
    facet_id: Identifier
    coverage: ScoreBounds
    path_quality: ScoreBounds
    best_source_id: Identifier | None = None


class TermPathFeature(ControlContract):
    term_id: Identifier
    facet_id: Identifier
    observed_source_hits: int = Field(ge=0)
    observed_sources: int = Field(ge=0)
    occurrence_scope: Literal["observed_panel"] = "observed_panel"
    leak: ScoreBounds | None
    topic_entropy: UnitScore | None
    routing_damage: GainBounds
    pivotal_panel_count: int = Field(ge=0)
    panel_count: int = Field(ge=0)
    diagnosis: Literal["routing_harm", "routing_help", "no_observed_routing_effect",
                       "not_observed", "uncertain"]
    corpus_absence_proven: Literal[False] = False


class TermInteraction(ControlContract):
    facet_id: Identifier
    term_ids: tuple[Identifier, ...]
    routing_damage: GainBounds
    selected_sets_changed: bool


class StrategyGainEstimate(ControlContract):
    protocol_version: Literal["lexical_local_gain_v1"] = "lexical_local_gain_v1"
    task_hash: str
    before_strategy_hash: str
    after_strategy_hash: str
    scope: Literal["observed_panel"] = "observed_panel"
    gain: GainBounds
    changed_panel_count: int
    model_call_count: Literal[0] = 0
    actual_retrieval_gain: Literal[False] = False


class PathFeatureSummary(ControlContract):
    protocol_version: Literal["retrieval_path_features_v1"] = "retrieval_path_features_v1"
    task_hash: str
    strategy_hash: str
    evaluation_protocol_hash: str
    input_hash: str
    facets: tuple[FacetCoverage, ...]
    utility: ScoreBounds
    terms: tuple[TermPathFeature, ...]
    interactions: tuple[TermInteraction, ...]
    observed_panel_count: int
    candidate_evaluation_count: int
    counterfactual_selection_count: int
    invalid_packaged_source_count: int = Field(ge=0)
    model_call_count: Literal[0] = 0
    is_answer_correctness_probability: Literal[False] = False
    scope_statuses: tuple[FacetScopeStatus, ...] = Field(default=(), max_length=8)

    @model_serializer(mode='wrap')
    def preserve_legacy_scope(self, handler):
        payload = handler(self)
        if not self.scope_statuses:
            payload.pop('scope_statuses', None)
        return payload


class GateThresholds(ControlContract):
    coverage: UnitScore
    path_quality: UnitScore
    calibration_id: Identifier


class RetrievalGateDecision(ControlContract):
    protocol_version: Literal["retrieval_ready_gate_v1"] = "retrieval_ready_gate_v1"
    outcome: Literal["ready_full", "ready_partial", "repairable", "scoped_not_found",
                     "scope_ambiguous", "source_unresolved", "representation_incomplete", "source_incomplete", "technical_failure", "budget_exhausted"]
    missing_facet_ids: tuple[Identifier, ...]
    reason_codes: tuple[str, ...]
    feature_hash: str
    threshold_hash: str
    proposed_action_ids: tuple[Identifier, ...] = ()
    corpus_absence_proven: Literal[False] = False
    post_generation_review: Literal[False] = False


class LexicalRepairCandidate(ControlContract):
    id: Identifier
    facet_id: Identifier
    surface: str = Field(min_length=1, max_length=96)
    witness_id: Identifier
    context: str = Field(min_length=1, max_length=240)
    source_title: str = Field(default="", max_length=160)
    source_section: str | None = Field(default=None, max_length=160)
    relation: Literal["format_variant", "explicit_alias", "proposed_equivalence", "related_locator"]
    permitted_operations: tuple[Literal["replace_surface", "add_attested_alias", "qualify", "locator_probe"], ...]
    scope_kind: Literal["observed_panel", "new_scope_proposal"]


class LexicalPatchItem(ControlContract):
    facet_id: Identifier
    operation: Literal["replace_surface", "add_attested_alias", "qualify", "locator_probe"]
    remove_term_ids: tuple[Identifier, ...] = Field(default=(), max_length=4)
    candidate_ids: tuple[Identifier, ...] = Field(min_length=1, max_length=3)


class LexicalPatch(ControlContract):
    protocol_version: Literal["lexical_patch_v1"] = "lexical_patch_v1"
    outcome: Literal["patch", "none_supported", "need_scope_clarification"]
    patches: tuple[LexicalPatchItem, ...] = Field(default=(), max_length=2)

    @model_validator(mode="after")
    def branch_shape(self):
        if (self.outcome == "patch") != bool(self.patches):
            raise ValueError("lexical_patch_outcome_mismatch")
        if len({item.facet_id for item in self.patches}) != len(self.patches):
            raise ValueError("lexical_patch_duplicate_facet")
        return self


class GroundedAnswerDraft(ControlContract):
    protocol_version: Literal["grounded_answer_units_v2"] = "grounded_answer_units_v2"
    answer_units: tuple[AnswerUnit, ...] = Field(min_length=1, max_length=32)

    @model_validator(mode="after")
    def unique_units(self):
        if len({unit.text.strip() for unit in self.answer_units}) != len(self.answer_units):
            raise ValueError("duplicate_answer_unit")
        return self


class GroundedMarkdownAnswerUnit(ControlContract):
    kind: Literal["factual"] = "factual"
    text: str = Field(min_length=1, max_length=262_144)
    source_handles: tuple[SourceHandle, ...] = Field(
        min_length=1,
        max_length=64,
    )

    @field_validator("text")
    @classmethod
    def valid_text(cls, value: str):
        if not value.strip() or "\x00" in value:
            raise ValueError("grounded_markdown_answer_text_invalid")
        return value

    @field_validator("source_handles")
    @classmethod
    def unique_sources(cls, value: tuple[str, ...]):
        if len(set(value)) != len(value):
            raise ValueError("grounded_markdown_sources_duplicate")
        return value


class GroundedMarkdownAnswerDraft(ControlContract):
    protocol_version: Literal["grounded_markdown_inline_citations_v1"] = (
        "grounded_markdown_inline_citations_v1"
    )
    answer_units: tuple[GroundedMarkdownAnswerUnit, ...] = Field(
        min_length=1,
        max_length=1,
    )


class SourceAddressedFacet(ControlContract):
    facet_id: Identifier
    coverage: ScoreBounds
    path_quality: ScoreBounds
    complete_coverage: tuple[EvidenceScopeCoverage, ...] = Field(min_length=1, max_length=16)
    selected_request_hashes: tuple[str, ...] = Field(min_length=1, max_length=16)
    usable_intervals: tuple[EvidenceInterval, ...] = Field(min_length=1, max_length=4096)
    path_source_ids: tuple[Identifier, ...] = Field(min_length=1, max_length=256)
    sources: tuple[ScopedGenerationSource, ...] = Field(min_length=1, max_length=256)


class SourceAddressedAssessment(ControlContract):
    protocol_version: Literal['source_addressed_assessment_v1'] = 'source_addressed_assessment_v1'
    task_hash: str = Field(pattern=r'^[a-f0-9]{64}$')
    strategy_hash: str = Field(pattern=r'^[a-f0-9]{64}$')
    feature_input_hash: str = Field(pattern=r'^[a-f0-9]{64}$')
    feature_hash: str = Field(pattern=r'^[a-f0-9]{64}$')
    thresholds: GateThresholds
    context_package_id: Identifier
    retrieval_trace_id: Identifier
    evidence_manifest_hash: str = Field(pattern=r'^[a-f0-9]{64}$')
    provenance_session_hash: str = Field(pattern=r'^[a-f0-9]{64}$')
    facets: tuple[SourceAddressedFacet, ...] = Field(min_length=1, max_length=8)
    relevance_uncertain: Literal[True] = True
    generation_authorized: Literal[False] = False
    model_call_count: Literal[0] = 0

    @property
    def identity(self):
        return control_hash(self.model_dump(mode='json'))


class SourceGateAdmission(ControlContract):
    protocol_version: Literal["retrieval_source_admission_v1"] = "retrieval_source_admission_v1"
    run_id: Identifier
    knowledge_base_id: Identifier
    context_package_id: Identifier
    retrieval_trace_id: Identifier
    task_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    strategy_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    feature_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    outcome: Literal["ready_full", "ready_partial"]
    evidence_manifest_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    provenance_session_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    source_chunk_ids: tuple[Identifier, ...] = Field(min_length=1, max_length=256)
    post_generation_model_review_count: Literal[0] = 0
    evidence_sufficiency_hash: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    resolved_scope_filter_hash: str | None = Field(default=None,pattern=r'^[a-f0-9]{64}$')
    source_addressed_assessment_hash: str | None = Field(default=None, pattern=r'^[a-f0-9]{64}$')

    @model_serializer(mode='wrap')
    def preserve_historical_identity(self, handler):
        payload = handler(self)
        if self.evidence_sufficiency_hash is None:
            payload.pop('evidence_sufficiency_hash', None)
        if self.resolved_scope_filter_hash is None:
            payload.pop('resolved_scope_filter_hash',None)
        if self.source_addressed_assessment_hash is None:
            payload.pop('source_addressed_assessment_hash', None)
        return payload

    @property
    def identity(self):
        return control_hash(self.model_dump(mode="json"))


class RetrievalAnswerSummary(ControlContract):
    protocol_version: Literal["retrieval_answer_v1"] = "retrieval_answer_v1"
    gate_outcome: str
    repairs_used: int = Field(ge=0, le=2)
    generation_call_count: int = Field(ge=0, le=1)
    post_generation_review_count: Literal[0] = 0
    source_binding_count: int = Field(ge=0)
    source_binding_pass_rate: UnitScore
    feature_utility: ScoreBounds | None = None
    audit_hash: str
