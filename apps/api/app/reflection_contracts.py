from __future__ import annotations

from typing import Annotated, Literal, Self, get_args
from statistics import median

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, field_validator, model_validator
from pydantic_core import PydanticCustomError


ANSWER_REFLECTION_PROTOCOL = "agent_answer_reflection_v1"
ANSWER_DRAFT_PROTOCOL = "structured_answer_self_assessment_v1"
PROMPT_PRIORITY_PROTOCOL = "agent_prompt_priority_v1"
PATH_SUPPORT_PROTOCOL = "answer_path_support_score_v2"
SOURCE_BINDING_PROTOCOL = "answer_source_binding_v1"
REFLECTION_REWARD_PROTOCOL = "answer_reflection_reward_v1"
ANSWER_JSON_REPAIR_PROTOCOL = "answer_json_shape_repair_v1"

Score = Annotated[float, Field(ge=0.0, le=1.0, allow_inf_nan=False)]
SourceHandle = Annotated[str, StringConstraints(pattern=r"^src_[1-9][0-9]*$")]
Digest = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
IssueType = Literal[
    "missing_evidence", "ambiguous_question", "context_conflict", "answer_incomplete",
    "answer_off_topic", "source_binding", "format",
]
ReflectionAction = Literal[
    "accept", "revise_answer", "restore_context", "replan_retrieval",
    "clarify_user", "insufficient_evidence",
]
DIRECT_REFLECTION_ACTIONS = frozenset({"review_answer", *get_args(ReflectionAction)})


def is_direct_reflection_action(action_type: str, plan_id: str | None, validation: object) -> bool:
    """Recognize locally validated review actions that do not execute retrieval."""
    return (
        plan_id is None
        and action_type in DIRECT_REFLECTION_ACTIONS
        and isinstance(validation, dict)
        and validation.get("protocol_version") == ANSWER_REFLECTION_PROTOCOL
        and validation.get("direct_context_reuse") is True
        and validation.get("valid") is True
    )


class ReflectionContract(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    @field_validator("*", mode="after")
    @classmethod
    def no_nul_text(cls, value):
        def check(item):
            if isinstance(item, str) and "\x00" in item:
                raise ValueError("reflection text must be NUL-free")
            if isinstance(item, list):
                for child in item:
                    check(child)
        check(value)
        return value


class AnswerUnit(ReflectionContract):
    model_config = ConfigDict(json_schema_extra={"allOf": [{
        "if": {"properties": {"kind": {"const": "factual"}}},
        "then": {"properties": {"source_handles": {"minItems": 1}}},
        "else": {"properties": {"source_handles": {"maxItems": 0}}},
    }]})
    kind: Literal["factual", "framing", "clarification"]
    text: str = Field(
        min_length=1,
        max_length=6000,
        description=(
            "Renderable GFM for this grounded unit. Use Markdown structure when useful and "
            "$...$ or $$...$$ for every mathematical expression; never emit raw LaTeX."
        ),
    )
    source_handles: list[SourceHandle] = Field(max_length=8, json_schema_extra={"uniqueItems": True})

    @field_validator("text")
    @classmethod
    def valid_text(cls, value: str) -> str:
        if not value.strip() or "\x00" in value:
            raise ValueError("answer unit text must be nonblank and NUL-free")
        return value

    @model_validator(mode="after")
    def valid_sources(self) -> Self:
        if len(set(self.source_handles)) != len(self.source_handles):
            raise PydanticCustomError("duplicate_source_handles", "answer unit has duplicate source handles")
        if self.kind == "factual" and not self.source_handles:
            raise PydanticCustomError("factual_source_handles_required", "factual answer unit requires a source handle")
        if self.kind != "factual" and self.source_handles:
            raise PydanticCustomError("nonfactual_source_handles_forbidden", "nonfactual answer unit must not create source bindings")
        return self


class AnswerSelfAssessment(ReflectionContract):
    question_relevance: Score
    context_relevance: Score
    coverage: Score
    needs_reflection: bool
    issue_types: list[IssueType] = Field(max_length=7)
    summary: str = Field(max_length=600)

    @field_validator("issue_types")
    @classmethod
    def unique_issues(cls, value: list[IssueType]) -> list[IssueType]:
        if len(set(value)) != len(value):
            raise ValueError("duplicate self-assessment issue")
        return value


class AnswerDraft(ReflectionContract):
    protocol_version: Literal["structured_answer_self_assessment_v1"]
    answer_units: list[AnswerUnit] = Field(min_length=1, max_length=32)
    self_assessment: AnswerSelfAssessment

    @model_validator(mode="after")
    def unique_units(self) -> Self:
        texts = [unit.text.strip() for unit in self.answer_units]
        if len(set(texts)) != len(texts):
            raise ValueError("duplicate answer unit")
        return self


class ReflectionDecision(ReflectionContract):
    protocol_version: Literal["agent_answer_reflection_v1"]
    action: ReflectionAction
    issue_types: list[IssueType] = Field(max_length=7)
    target_unit_indexes: list[Annotated[int, Field(ge=0, le=31)]] = Field(max_length=32)
    source_handles: list[SourceHandle] = Field(max_length=64)
    missing_facets: list[Annotated[str, StringConstraints(min_length=1, max_length=180)]] = Field(max_length=8)
    correction_instructions: str = Field(max_length=1500)
    clarification_question: str | None = Field(max_length=400)

    @model_validator(mode="after")
    def valid_action_fields(self) -> Self:
        for values in (self.issue_types, self.target_unit_indexes, self.source_handles, self.missing_facets):
            if len(set(values)) != len(values):
                raise ValueError("reflection decision has duplicate targets or issues")
        if self.action == "accept" and (
            self.issue_types or self.target_unit_indexes or self.source_handles
            or self.missing_facets or self.clarification_question
        ):
            raise ValueError("accept cannot carry unresolved issues or executable targets")
        if self.action == "clarify_user":
            if not (self.clarification_question or "").strip():
                raise ValueError("clarify_user requires a question")
        elif self.clarification_question is not None:
            raise ValueError("only clarify_user may carry a clarification question")
        if self.action == "restore_context" and not self.source_handles:
            raise ValueError("restore_context requires existing source targets")
        if self.action in {"revise_answer", "replan_retrieval"} and not self.correction_instructions.strip():
            raise ValueError("reflection backtrack requires bounded correction instructions")
        if self.action == "insufficient_evidence" and not self.missing_facets:
            raise ValueError("insufficient_evidence requires explicit missing facets")
        return self


class ReflectionThresholds(ReflectionContract):
    path_support: Score = 0.5
    question_relevance: Score = 0.8
    context_relevance: Score = 0.8


class SourcePathScore(ReflectionContract):
    source_handle: SourceHandle
    score: Score | None
    effective_distance: Annotated[float, Field(ge=0.0, allow_inf_nan=False)] | None

    @model_validator(mode="after")
    def score_matches_distance(self) -> Self:
        if (self.score is None) != (self.effective_distance is None):
            raise ValueError("missing path must not receive a score")
        if self.score is not None and self.effective_distance is not None:
            if abs(self.score - 1.0 / (1.0 + self.effective_distance)) > 1e-12:
                raise ValueError("source path score does not match distance")
        return self


class PathSupportMetrics(ReflectionContract):
    protocol_version: Literal["answer_path_support_score_v1", "answer_path_support_score_v2"] = PATH_SUPPORT_PROTOCOL
    path_score: Score | None
    coverage: Score
    weakest_path_score: Score | None
    source_count: int = Field(ge=0, le=256)
    scored_source_count: int = Field(ge=0, le=256)
    sources: list[SourcePathScore] = Field(max_length=256)

    @model_validator(mode="after")
    def aggregates_match_sources(self) -> Self:
        scores = [source.score for source in self.sources if source.score is not None]
        expected_coverage = len(scores) / len(self.sources) if self.sources else 0.0
        if (
            len({source.source_handle for source in self.sources}) != len(self.sources)
            or self.source_count != len(self.sources)
            or self.scored_source_count != len(scores)
            or abs(self.coverage - expected_coverage) > 1e-12
            or self.path_score != (median(scores) if scores else None)
            or self.weakest_path_score != (min(scores) if scores else None)
        ):
            raise ValueError("path metric aggregates do not match source-distinct scores")
        return self


class ReflectionGate(ReflectionContract):
    protocol_version: Literal["agent_answer_reflection_v1"] = ANSWER_REFLECTION_PROTOCOL
    decision: Literal["skip_reflection", "reflect", "source_integrity_failed"]
    reasons: list[str]
    path_metrics: PathSupportMetrics
    self_assessment: AnswerSelfAssessment
    thresholds: ReflectionThresholds
    source_binding_valid: bool
    draft_hash: Digest
    evidence_manifest_hash: Digest
    decision_hash: Digest


class ReflectionTransition(ReflectionContract):
    protocol_version: Literal["agent_answer_reflection_v1"] = ANSWER_REFLECTION_PROTOCOL
    action: ReflectionAction
    destination: Literal["commit", "answer_generation", "context_restoration", "planner", "waiting_user", "evidence_gap"]
    round_index: int = Field(ge=0)
    remaining_rounds: int = Field(ge=0)
    input_hash: Digest
    decision_hash: Digest


class AnswerReflectionSummary(ReflectionContract):
    protocol_version: Literal["agent_answer_reflection_v1"] = ANSWER_REFLECTION_PROTOCOL
    outcome: Literal["accepted_without_reflection", "accepted_after_reflection", "clarify_user", "insufficient_evidence"]
    reflection_rounds_used: int = Field(ge=0, le=10)
    generation_model_call_count: int = Field(ge=0)
    reflection_model_call_count: int = Field(ge=0, le=10)
    citation_judge_model_call_count: Literal[0] = 0
    source_binding_count: int = Field(ge=0, le=256)
    source_binding_pass_rate: Score
    self_assessment: AnswerSelfAssessment
    path_score: Score | None
    path_coverage: Score
    reflection_audit_hash: Digest
    self_assessment_is_reward_label: Literal[False] = False
