"""Versioned task intent and execution plans; no retrieval-result feedback."""
from __future__ import annotations

import math
import re
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.retrieval_control_contracts import ResponseConstraint, SourceScopeObligation, control_hash

PROTOCOL = "intent_execution_retrieval_v1"
Layer = Literal["coarse", "mid", "chunk"]
Channel = Literal["dense", "rq", "bm25"]
Intent = Literal["summarize", "overview", "define", "fact_lookup", "enumerate", "compare",
                 "explain", "procedure", "analyze", "relationship", "source_lookup",
                 "system_capability", "clarify"]
RequirementId = Literal["f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8"]
Identity = Annotated[str, Field(pattern=r"^[a-f0-9]{64}$")]


class ClosedPlan(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    @property
    def identity(self) -> str:
        return control_hash(self.model_dump(mode="json"))


class IntentContract(ClosedPlan):
    primary: Intent
    secondary: tuple[Intent, ...] = Field(default=(), max_length=3)

    @model_validator(mode="after")
    def distinct(self):
        if self.primary in self.secondary or len(set(self.secondary)) != len(self.secondary):
            raise ValueError("intent_duplicate")
        if self.primary in {"system_capability", "clarify"} and self.secondary:
            raise ValueError("intent_direct_cannot_add_corpus_task")
        if set(self.secondary) & {"system_capability", "clarify"}:
            raise ValueError("intent_direct_must_be_primary")
        return self


class ChannelWeights(ClosedPlan):
    dense: float = Field(ge=0, le=1, allow_inf_nan=False)
    rq: float = Field(ge=0, le=1, allow_inf_nan=False)
    bm25: float = Field(ge=0, le=1, allow_inf_nan=False)

    @field_validator("dense", "rq", "bm25", mode="before")
    @classmethod
    def numeric_only(cls, value):
        if type(value) not in (int, float):
            raise ValueError("strategy_weight_not_numeric")
        return value

    @model_validator(mode="after")
    def unit_sum(self):
        if abs(math.fsum((self.dense, self.rq, self.bm25)) - 1.0) > 1e-6:
            raise ValueError("strategy_weights_must_sum_to_one")
        return self

    def effective(self) -> dict[str, float]:
        total = math.fsum((self.dense, self.rq, self.bm25))
        return {name: getattr(self, name) / total for name in ("dense", "rq", "bm25")}


class LayerWeights(ClosedPlan):
    coarse: ChannelWeights | None = None
    mid: ChannelWeights | None = None
    chunk: ChannelWeights | None = None

    def enabled_layers(self) -> tuple[Layer, ...]:
        return tuple(layer for layer in ("coarse", "mid", "chunk") if getattr(self, layer) is not None)

    def for_layer(self, layer: Layer) -> ChannelWeights:
        value = getattr(self, layer)
        if value is None:
            raise ValueError("strategy_layer_weights_missing")
        return value


class LexicalQuerySurface(ClosedPlan):
    text: str = Field(min_length=1, max_length=160, pattern=r"\S")
    language: Literal["zh", "en", "neutral"]
    provenance: Literal["user_text", "model_query"]

    @model_validator(mode="after")
    def distinct(self):
        if "\x00" in self.text:
            raise ValueError("strategy_lexical_surface_invalid")
        has_cjk = re.search(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]", self.text) is not None
        has_latin = re.search(r"[A-Za-z]", self.text) is not None
        if self.language == "zh" and not has_cjk:
            raise ValueError("strategy_lexical_surface_zh_script_missing")
        if self.language == "en" and (not has_latin or has_cjk):
            raise ValueError("strategy_lexical_surface_en_script_invalid")
        return self


class LexicalQueryGroup(ClosedPlan):
    group_id: str = Field(pattern=r"^l(?:[1-9]|1[0-2])$")
    requirement_ids: tuple[RequirementId, ...] = Field(min_length=1, max_length=8)
    kind: Literal["concept", "identifier", "number_unit", "quoted_literal"]
    surfaces: tuple[LexicalQuerySurface, ...] = Field(min_length=1, max_length=4)

    @model_validator(mode="after")
    def valid_group(self):
        if len(set(self.requirement_ids)) != len(self.requirement_ids):
            raise ValueError("strategy_lexical_group_requirement_duplicate")
        normalized = [surface.text.casefold().strip() for surface in self.surfaces]
        if len(set(normalized)) != len(normalized):
            raise ValueError("strategy_lexical_group_surface_duplicate")
        if self.kind == "identifier" and any(
            surface.language != "neutral" for surface in self.surfaces
        ):
            raise ValueError("strategy_identifier_surface_must_be_neutral")
        return self


class LexicalQueryTerm(ClosedPlan):
    """Executor-only flattened view of one versioned lexical surface."""

    text: str
    requirement_ids: tuple[RequirementId, ...]
    provenance: Literal["user_text", "model_query"]
    language: Literal["zh", "en", "neutral"]
    group_id: str
    kind: Literal["concept", "identifier", "number_unit", "quoted_literal"]


class StrategyBudgetRequest(ClosedPlan):
    dense_candidates: int | None = Field(default=None, ge=1, strict=True)
    rq_candidates: int | None = Field(default=None, ge=1, strict=True)
    bm25_candidates: int | None = Field(default=None, ge=1, strict=True)
    root_entries: int | None = Field(default=None, ge=1, strict=True)
    per_parent_entries: int | None = Field(default=None, ge=1, strict=True)
    layer_entries: int | None = Field(default=None, ge=1, strict=True)
    max_depth: int | None = Field(default=None, ge=0, strict=True)
    restore_per_hit: int | None = Field(default=None, ge=0, strict=True)


class ExecutionBudget(ClosedPlan):
    dense_candidates: int = Field(ge=1, le=4096, strict=True)
    rq_candidates: int = Field(ge=1, le=4096, strict=True)
    bm25_candidates: int = Field(ge=1, le=4096, strict=True)
    root_entries: int = Field(ge=1, le=256, strict=True)
    per_parent_entries: int = Field(ge=1, le=256, strict=True)
    layer_entries: int = Field(ge=1, le=1024, strict=True)
    max_depth: int = Field(ge=0, le=64, strict=True)
    restore_per_hit: int = Field(ge=0, le=64, strict=True)

    def constrain(self, request: StrategyBudgetRequest) -> ExecutionBudget:
        values = self.model_dump()
        for key, value in request.model_dump(exclude_none=True).items():
            if value > values[key]:
                raise ValueError("strategy_budget_exceeds_capability")
            values[key] = value
        return ExecutionBudget.model_validate(values)


class ExecutionStrategy(ClosedPlan):
    protocol_version: Literal["intent_execution_strategy_v2"] = "intent_execution_strategy_v2"
    route: Literal["retrieve", "verified_context_reuse", "system_capability", "clarify"]
    entry_layer: Layer | None
    semantic_query: str = Field(default="", max_length=4000)
    generate_lexical: bool = Field(strict=True)
    lexical_groups: tuple[LexicalQueryGroup, ...] = Field(default=(), max_length=12)
    hybrid: bool = Field(strict=True)
    layer_weights: LayerWeights = Field(default_factory=LayerWeights)
    selection_scope: Literal["focused", "broad"] = "focused"
    budget_request: StrategyBudgetRequest = Field(default_factory=StrategyBudgetRequest)
    reason_code: Literal["broad_scope", "precise_terms", "semantic_paraphrase", "mixed_signal",
                         "source_locality", "existing_evidence", "system_request", "ambiguous_request"]

    @property
    def lexical_terms(self) -> tuple[LexicalQueryTerm, ...]:
        return tuple(
            LexicalQueryTerm(
                text=surface.text,
                requirement_ids=group.requirement_ids,
                provenance=surface.provenance,
                language=surface.language,
                group_id=group.group_id,
                kind=group.kind,
            )
            for group in self.lexical_groups
            for surface in group.surfaces
        )

    @model_validator(mode="after")
    def executable_combination(self):
        if "\x00" in self.semantic_query:
            raise ValueError("strategy_query_control_character")
        if self.route in {"system_capability", "clarify"}:
            if (self.entry_layer is not None or self.semantic_query or self.generate_lexical
                    or self.lexical_groups or self.hybrid or self.layer_weights.enabled_layers()
                    or self.budget_request.model_dump(exclude_none=True)):
                raise ValueError("strategy_direct_has_retrieval_fields")
            return self
        if self.entry_layer is None or not self.semantic_query.strip():
            raise ValueError("strategy_retrieval_needs_entry_and_query")
        layers = {"coarse": ("coarse", "mid", "chunk"), "mid": ("mid", "chunk"), "chunk": ("chunk",)}
        if self.layer_weights.enabled_layers() != layers[self.entry_layer]:
            raise ValueError("strategy_layer_scope_invalid")
        if self.generate_lexical != bool(self.lexical_groups):
            raise ValueError("strategy_lexical_flag_conflict")
        if len({group.group_id for group in self.lexical_groups}) != len(self.lexical_groups):
            raise ValueError("strategy_duplicate_lexical_group")
        if len(self.lexical_terms) > 24:
            raise ValueError("strategy_lexical_surface_budget_exceeded")
        if len({t.text.casefold().strip() for t in self.lexical_terms}) != len(self.lexical_terms):
            raise ValueError("strategy_duplicate_lexical_term")
        if self.hybrid and not self.generate_lexical:
            raise ValueError("strategy_hybrid_requires_lexical")
        for layer in self.layer_weights.enabled_layers():
            weights = self.layer_weights.for_layer(layer)
            if self.hybrid:
                if weights.dense <= 0 or weights.bm25 <= 0:
                    raise ValueError("strategy_hybrid_requires_dense_and_bm25")
            elif (weights.dense, weights.rq, weights.bm25) != (1.0, 0.0, 0.0):
                raise ValueError("strategy_dense_only_weights_invalid")
        return self


class TaskRequirement(ClosedPlan):
    id: RequirementId
    text: str = Field(min_length=1, max_length=256, pattern=r"\S")
    role: Literal["topic", "definition", "procedure", "quantity", "comparison", "relationship", "source_role"] = "topic"
    protected_literals: tuple[str, ...] = Field(default=(), max_length=8)
    source_roles: tuple[Literal["summary", "detail", "table", "formula", "code"], ...] = Field(default=(), max_length=5)
    source_scope: SourceScopeObligation | None = None


class IntentPlanningOutput(ClosedPlan):
    protocol_version: Literal["intent_execution_retrieval_v1"] = PROTOCOL
    intent: IntentContract
    requirements: tuple[TaskRequirement, ...] = Field(default=(), max_length=8)
    entities: tuple[str, ...] = Field(default=(), max_length=16)
    response_constraints: tuple[ResponseConstraint, ...] = Field(default=(), max_length=12)
    execution_strategy: ExecutionStrategy

    @model_validator(mode="after")
    def references(self):
        ids = [r.id for r in self.requirements]
        if len(set(ids)) != len(ids):
            raise ValueError("intent_duplicate_requirement")
        route = self.execution_strategy.route
        if route in {"system_capability", "clarify"}:
            if self.requirements or self.intent.primary != route:
                raise ValueError("intent_direct_route_conflict")
        elif not self.requirements or self.intent.primary in {"system_capability", "clarify"}:
            raise ValueError("intent_retrieval_task_missing")
        if any(not set(group.requirement_ids) <= set(ids) for group in self.execution_strategy.lexical_groups):
            raise ValueError("strategy_term_outside_task")
        if any(not text.strip() or len(text) > 400 or "\x00" in text for text in self.entities):
            raise ValueError("intent_entity_invalid")
        return self


class TaskContract(ClosedPlan):
    protocol_version: Literal["intent_task_v1"] = "intent_task_v1"
    knowledge_base_id: str = Field(min_length=1, max_length=160)
    conversation_identity_hash: Identity
    conversation_scope_hash: Identity
    filter_scope_hash: Identity
    question: str = Field(min_length=1, max_length=12000)
    requirements: tuple[TaskRequirement, ...] = Field(default=(), max_length=8)
    entities: tuple[str, ...] = Field(default=(), max_length=16)
    response_constraints: tuple[ResponseConstraint, ...] = Field(default=(), max_length=12)
    allow_partial: Literal[True] = True

    @model_validator(mode="after")
    def user_authority(self):
        if not self.question.strip() or "\x00" in self.question:
            raise ValueError("intent_question_invalid")
        if len({r.id for r in self.requirements}) != len(self.requirements):
            raise ValueError("intent_duplicate_requirement")
        for item in self.response_constraints:
            left, right = item.char_span
            if not 0 <= left < right <= len(self.question) or self.question[left:right] != item.text:
                raise ValueError("intent_response_constraint_not_quoted")
        for requirement in self.requirements:
            if any(not literal.strip() or literal not in self.question for literal in requirement.protected_literals):
                raise ValueError("intent_protected_literal_not_quoted")
            nodes = [requirement.source_scope] if requirement.source_scope else []
            count = 0
            while nodes:
                node = nodes.pop()
                count += 1
                if count > 128:
                    raise ValueError("intent_source_scope_too_large")
                nodes.extend(node.children)
                scope = getattr(node, "scope", None)
                if scope is not None:
                    nodes.append(scope)
                selector = getattr(node, "selector", None)
                if selector is not None and (not selector.reference.strip() or selector.reference not in self.question):
                    raise ValueError("intent_source_scope_not_quoted")
        return self


class CapabilityManifest(ClosedPlan):
    protocol_version: Literal["retrieval_capabilities_v2"] = "retrieval_capabilities_v2"
    knowledge_base_id: str = Field(min_length=1, max_length=160)
    available_layers: tuple[Layer, ...] = Field(max_length=3)
    available_channels: tuple[Channel, ...] = Field(max_length=3)
    graph_identity: Identity | None
    lexical_identity: Identity | None
    bilingual_lexical_enabled: bool = Field(strict=True)
    budget_limits: ExecutionBudget

    @model_validator(mode="after")
    def unique_capabilities(self):
        if (len(set(self.available_layers)) != len(self.available_layers)
                or len(set(self.available_channels)) != len(self.available_channels)):
            raise ValueError("strategy_capability_duplicate")
        if bool(self.available_layers) != bool(self.graph_identity):
            raise ValueError("strategy_capability_graph_identity_missing")
        if ("bm25" in self.available_channels) != bool(self.lexical_identity):
            raise ValueError("strategy_capability_lexical_identity_missing")
        return self


class AcceptedPlan(ClosedPlan):
    protocol_version: Literal["accepted_intent_plan_v2"] = "accepted_intent_plan_v2"
    task: TaskContract
    intent: IntentContract
    strategy: ExecutionStrategy
    capability_hash: Identity
    proposal_hash: Identity
    effective_budget: ExecutionBudget

    def effective_weights(self, layer: Layer) -> dict[str, float]:
        return self.strategy.layer_weights.for_layer(layer).effective()


def accept_plan(proposal: IntentPlanningOutput, *, question: str, conversation_scope_hash: str,
                filter_scope_hash: str, capabilities: CapabilityManifest,
                conversation_identity_hash: str | None = None) -> AcceptedPlan:
    strategy = proposal.execution_strategy
    from app.services.task_constraints import response_constraints

    def quoted_scope(scope) -> bool:
        if scope is None:
            return True
        pending = [scope]
        seen = 0
        while pending:
            node = pending.pop()
            seen += 1
            if seen > 128:
                raise ValueError("intent_source_scope_too_large")
            pending.extend(node.children)
            child_scope = getattr(node, "scope", None)
            if child_scope is not None:
                pending.append(child_scope)
            selector = getattr(node, "selector", None)
            if selector is not None and selector.reference not in question:
                return False
        return True

    def normalize_scope_coverage(scope):
        if scope is None:
            return None

        children = tuple(
            normalize_scope_coverage(child) for child in scope.children
        )
        value = scope.model_copy(update={"children": children})
        full_scope_requested = re.search(
            r"全文|整份|整篇|整个(?:报告|文档|文件|章节|部分)|全部内容|完整(?:报告|文档|文件|章节|部分|摘要|说明)|\b(?:entire|whole|complete)\s+(?:report|document|file|section|chapter|summary)\b|\ball\s+of\s+(?:the\s+)?(?:report|document|file|section|chapter|summary)\b",
            question,
            re.I,
        ) is not None
        if (
            value.op == "coverage"
            and value.mode == "complete"
            and not full_scope_requested
        ):
            value = value.model_copy(update={"mode": "overlap"})
        return value

    def sanitize_source_scope_authority(scope):
        if scope is None:
            return None
        from app.services.structure_roles import (
            DETAIL_TITLES,
            SUMMARY_TITLES,
            normalized_role_title,
        )

        def sanitize_request(node):
            if node.op != "scope":
                children = tuple(sanitize_request(child) for child in node.children)
                if node.op == "intersection" and any(child is None for child in children):
                    return None
                children = tuple(child for child in children if child is not None)
                if not children:
                    return None
                if len(children) == 1:
                    return children[0]
                return node.model_copy(update={"children": children})
            selector = node.selector
            if selector.kind == "text" and selector.match == "title":
                return None
            if selector.kind != "section" or selector.match != "title":
                return node
            reference = selector.reference.strip()
            normalized_reference = normalized_role_title(reference)
            if normalized_reference in SUMMARY_TITLES | DETAIL_TITLES:
                return node
            explicit_section = re.search(
                rf"(?:section|chapter)\s+{re.escape(reference)}\b|"
                rf"\b{re.escape(reference)}\s+(?:section|chapter)\b|"
                rf"第?{re.escape(reference)}(?:章|节|小节)|"
                rf"(?:章节|章|节|小节)[《\"“']?{re.escape(reference)}",
                question,
                re.I,
            )
            return node if explicit_section is not None else None

        if scope.op == "coverage":
            request = sanitize_request(scope.scope)
            return (
                scope.model_copy(update={"scope": request})
                if request is not None
                else None
            )
        children = tuple(
            sanitize_source_scope_authority(child) for child in scope.children
        )
        if scope.op == "all" and any(child is None for child in children):
            return None
        children = tuple(child for child in children if child is not None)
        if not children:
            return None
        if len(children) == 1:
            return children[0]
        return scope.model_copy(update={"children": children})

    requirements = tuple(
        item.model_copy(
            update={
                "protected_literals": tuple(
                    literal
                    for literal in item.protected_literals
                    if literal in question
                ),
                "source_scope": (
                    authorized_scope
                    if (
                        (normalized_scope := normalize_scope_coverage(item.source_scope))
                        is not None
                        and (
                            authorized_scope := sanitize_source_scope_authority(
                                normalized_scope
                            )
                        )
                        is not None
                        and quoted_scope(authorized_scope)
                    )
                    else None
                ),
            }
        )
        for item in proposal.requirements
    )
    if strategy.lexical_groups:
        strategy = strategy.model_copy(
            update={
                "lexical_groups": tuple(
                    group.model_copy(
                        update={
                            "surfaces": tuple(
                                surface
                                if surface.provenance != "user_text"
                                or surface.text in question
                                else surface.model_copy(
                                    update={"provenance": "model_query"}
                                )
                                for surface in group.surfaces
                            )
                        }
                    )
                    for group in strategy.lexical_groups
                )
            }
        )
    if capabilities.bilingual_lexical_enabled:
        for group in strategy.lexical_groups:
            if group.kind != "concept":
                continue
            languages = {surface.language for surface in group.surfaces}
            if not {"zh", "en"} <= languages:
                raise ValueError("strategy_bilingual_concept_surfaces_missing")
    for group in strategy.lexical_groups:
        if group.kind == "quoted_literal" and any(
            surface.provenance != "user_text" or surface.text not in question
            for surface in group.surfaces
        ):
            raise ValueError("strategy_quoted_literal_not_user_text")
    task = TaskContract(knowledge_base_id=capabilities.knowledge_base_id,
        conversation_identity_hash=(conversation_identity_hash or control_hash(
            {"legacy_conversation_scope_hash": conversation_scope_hash}
        )), conversation_scope_hash=conversation_scope_hash, filter_scope_hash=filter_scope_hash,
        question=question, requirements=requirements, entities=proposal.entities,
        # Response instructions are user-text facts. The model may classify
        # them, but it cannot invent their text or offsets.
        response_constraints=response_constraints(question))
    if strategy.generate_lexical:
        from app.services.lexical_index import query_terms
        query_terms(tuple(term.text for term in strategy.lexical_terms))
    if strategy.route in {"retrieve", "verified_context_reuse"}:
        if not set(strategy.layer_weights.enabled_layers()) <= set(capabilities.available_layers):
            raise ValueError("strategy_entry_unavailable")
        required = {name for layer in strategy.layer_weights.enabled_layers()
                    for name, weight in strategy.layer_weights.for_layer(layer).effective().items() if weight > 0}
        if not required <= set(capabilities.available_channels):
            raise ValueError("strategy_index_unavailable")
    return AcceptedPlan(task=task, intent=proposal.intent, strategy=strategy,
        capability_hash=capabilities.identity, proposal_hash=proposal.identity,
        effective_budget=capabilities.budget_limits.constrain(strategy.budget_request))
