"""One-shot intent and execution planning for intent_execution_retrieval_v1."""
from __future__ import annotations

import copy
import json
from typing import Any, Callable

from sqlalchemy import func, select
from sqlalchemy.orm.attributes import flag_modified

from app.core.config import get_settings
from app.intent_contracts import (
    AcceptedPlan,
    CapabilityManifest,
    ExecutionBudget,
    IntentPlanningOutput,
    PROTOCOL,
    accept_plan,
)
from app.models import (
    AgentObservation,
    AgentRun,
    CoarseConcept,
    ContextGraphState,
    LexicalIndexState,
    MidConcept,
)
from app.retrieval_control_contracts import control_hash
from app.services.embeddings import (
    ChatProvider,
    ProviderJSONShapeError,
    classify_json_with_budget,
)


PLANNING_CALL_PROTOCOL = "intent_execution_planning_call_v1"
PLANNING_MAX_TOKENS = 8192


def normalize_planning_output(raw):
    """Remove only redundant or non-user-authored planning annotations."""

    audit = {
        "protocol_version": "intent_plan_local_normalization_v1",
        "removed_selector_roles": 0,
        "role_aliases_normalized": 0,
        "section_role_selectors_normalized": 0,
        "scope_shape_noise_removed": 0,
        "direct_route_noise_removed": 0,
        "misplaced_requirement_scope_modes_removed": 0,
        "zero_budget_hints_removed": 0,
        "facts_invented": False,
    }
    if not isinstance(raw, dict):
        return raw, audit
    value = copy.deepcopy(raw)
    role_aliases = {
        "fact": "topic",
        "fact_lookup": "topic",
        "enumerate": "topic",
        "summary": "topic",
        "overview": "topic",
        "explain": "topic",
        "analyze": "topic",
        "list": "topic",
    }
    requirements = value.get("requirements")
    if isinstance(requirements, list):
        for requirement in requirements[:8]:
            if not isinstance(requirement, dict):
                continue
            if requirement.get("mode") in {"complete", "overlap"}:
                requirement.pop("mode", None)
                audit["misplaced_requirement_scope_modes_removed"] += 1
            role = requirement.get("role")
            if role in role_aliases:
                requirement["role"] = role_aliases[role]
                audit["role_aliases_normalized"] += 1
            source_roles = requirement.get("source_roles")
            source_role = (
                source_roles[0]
                if isinstance(source_roles, list) and len(source_roles) == 1
                else None
            )
            pending = [requirement.get("source_scope")]
            seen = 0
            while pending:
                node = pending.pop()
                seen += 1
                if seen > 128:
                    raise ValueError("intent_plan_normalization_scope_too_large")
                if not isinstance(node, dict):
                    continue
                if node.get("op") == "coverage":
                    scope_payload = node.get("scope")
                    if (
                        isinstance(scope_payload, dict)
                        and "op" not in scope_payload
                        and {"kind", "reference", "match"}
                        <= set(scope_payload)
                    ):
                        node["scope"] = {
                            "op": "scope",
                            "selector": scope_payload,
                            "children": [],
                        }
                        audit["scope_shape_noise_removed"] += 1
                    raw_children = node.get("children")
                    if (
                        node.get("scope") is None
                        and isinstance(raw_children, list)
                        and raw_children
                    ):
                        request_children = []
                        for child in raw_children:
                            if not isinstance(child, dict):
                                request_children = []
                                break
                            if (
                                "op" not in child
                                and {"kind", "reference", "match"}
                                <= set(child)
                            ):
                                request_children.append(
                                    {
                                        "op": "scope",
                                        "selector": child,
                                        "children": [],
                                    }
                                )
                            elif (
                                child.get("op")
                                in {"scope", "union", "intersection"}
                                and "mode" not in child
                            ):
                                request_children.append(child)
                            else:
                                request_children = []
                                break
                        if request_children:
                            node["scope"] = (
                                request_children[0]
                                if len(request_children) == 1
                                else {
                                    "op": "union",
                                    "selector": None,
                                    "children": request_children,
                                }
                            )
                            node["children"] = []
                            audit["scope_shape_noise_removed"] += 1
                if (
                    node.get("op") == "coverage"
                    and isinstance(node.get("scope"), dict)
                    and node.get("children")
                ):
                    node["children"] = []
                    audit["scope_shape_noise_removed"] += 1
                elif (
                    node.get("op") in {"all", "any"}
                    and isinstance(node.get("children"), list)
                    and len(node["children"]) >= 2
                    and node.get("scope") is not None
                ):
                    node["scope"] = None
                    audit["scope_shape_noise_removed"] += 1
                selector = node.get("selector")
                if node.get("op") == "scope" and isinstance(selector, dict):
                    if node.get("children"):
                        node["children"] = []
                        audit["scope_shape_noise_removed"] += 1
                elif node.get("op") in {"union", "intersection"} and selector is not None:
                    node["selector"] = None
                    selector = None
                    audit["scope_shape_noise_removed"] += 1
                if (
                    isinstance(selector, dict)
                    and selector.get("kind") == "section"
                    and selector.get("match") == "title"
                ):
                    from app.services.structure_roles import (
                        DETAIL_TITLES,
                        SUMMARY_TITLES,
                        normalized_role_title,
                    )

                    normalized_reference = normalized_role_title(
                        selector.get("reference")
                    )
                    inferred_role = (
                        "summary"
                        if normalized_reference in SUMMARY_TITLES
                        else "detail"
                        if normalized_reference in DETAIL_TITLES
                        else None
                    )
                    if (
                        inferred_role is not None
                        and source_role in {None, inferred_role}
                    ):
                        selector["match"] = "role"
                        selector["role"] = inferred_role
                        audit["section_role_selectors_normalized"] += 1
                if (
                    isinstance(selector, dict)
                    and selector.get("match") != "role"
                    and "role" in selector
                ):
                    selector.pop("role", None)
                    audit["removed_selector_roles"] += 1
                children = node.get("children")
                if isinstance(children, list):
                    pending.extend(children)
                pending.append(node.get("scope"))
    strategy = value.get("execution_strategy")
    if isinstance(strategy, dict):
        route = strategy.get("route")
        if route in {"system_capability", "clarify"}:
            requirements = value.get("requirements")
            if requirements:
                value["requirements"] = []
                audit["direct_route_noise_removed"] += 1
            intent = value.get("intent")
            if (
                isinstance(intent, dict)
                and intent.get("primary") == route
                and intent.get("secondary")
            ):
                intent["secondary"] = []
                audit["direct_route_noise_removed"] += 1
            direct_defaults = {
                "entry_layer": None,
                "semantic_query": "",
                "generate_lexical": False,
                "lexical_groups": [],
                "hybrid": False,
                "layer_weights": {},
                "budget_request": {},
            }
            for key, expected in direct_defaults.items():
                if strategy.get(key) != expected:
                    strategy[key] = expected
                    audit["direct_route_noise_removed"] += 1
        budgets = strategy.get("budget_request")
        if isinstance(budgets, dict):
            for key in list(budgets):
                if budgets.get(key) == 0:
                    budgets.pop(key)
                    audit["zero_budget_hints_removed"] += 1
    return value, audit


def _execution_budget() -> ExecutionBudget:
    settings = get_settings()
    return ExecutionBudget(
        dense_candidates=settings.retrieval_v1_dense_candidate_budget,
        rq_candidates=settings.retrieval_v1_rq_candidate_budget,
        bm25_candidates=settings.retrieval_v1_bm25_candidate_budget,
        root_entries=settings.retrieval_v1_root_entry_budget,
        per_parent_entries=settings.retrieval_v1_per_parent_entry_budget,
        layer_entries=settings.retrieval_v1_layer_entry_budget,
        max_depth=settings.retrieval_v1_max_depth,
        restore_per_hit=settings.retrieval_v1_restore_per_hit,
    )


def retrieval_capability_snapshot(
    db,
    knowledge_base_id: str,
    *,
    admit_graph: bool = True,
) -> tuple[CapabilityManifest, ContextGraphState | None]:
    """Read a closed capability snapshot, optionally with full graph admission."""

    from app.services.context_graph import (
        ActiveContextGraphAdmissionError,
        active_graph_online_admission_gate,
    )
    from app.services.lexical_storage import load_lexical_snapshot

    if admit_graph:
        try:
            context_state = active_graph_online_admission_gate(db, knowledge_base_id)
        except ActiveContextGraphAdmissionError:
            context_state = None
    else:
        context_state = db.scalar(
            select(ContextGraphState).where(
                ContextGraphState.knowledge_base_id == knowledge_base_id,
                ContextGraphState.state == "active",
            )
        )
    available_layers: list[str] = []
    available_channels: list[str] = []
    graph_identity = None
    if context_state is not None:
        graph_identity = str(context_state.context_graph_hash)
        available_layers.append("chunk")
        if context_state.mid_concept_state_id and db.scalar(
            select(func.count()).select_from(MidConcept).where(
                MidConcept.concept_state_id == context_state.mid_concept_state_id,
                MidConcept.state == "active",
            )
        ):
            available_layers.insert(0, "mid")
        if context_state.coarse_concept_state_id and db.scalar(
            select(func.count()).select_from(CoarseConcept).where(
                CoarseConcept.coarse_state_id == context_state.coarse_concept_state_id,
                CoarseConcept.state == "active",
            )
        ):
            available_layers.insert(0, "coarse")
        available_channels.extend(("dense", "rq"))
    lexical = db.scalar(
        select(LexicalIndexState).where(
            LexicalIndexState.knowledge_base_id == knowledge_base_id,
            LexicalIndexState.state == "active",
        )
    )
    lexical_identity = None
    if lexical is not None:
        if not admit_graph:
            lexical_identity = lexical.state_hash
            available_channels.append("bm25")
        else:
            try:
                load_lexical_snapshot(db, lexical.id, verify_sources=True)
            except ValueError:
                lexical = None
            else:
                lexical_identity = lexical.state_hash
                available_channels.append("bm25")
    manifest = CapabilityManifest(
        knowledge_base_id=knowledge_base_id,
        available_layers=tuple(available_layers),
        available_channels=tuple(available_channels),
        graph_identity=graph_identity,
        lexical_identity=lexical_identity,
        bilingual_lexical_enabled=bool(
            get_settings().query_facet_bilingual_enabled
        ),
        budget_limits=_execution_budget(),
    )
    return manifest, context_state


def retrieval_capability_manifest(db, knowledge_base_id: str) -> CapabilityManifest:
    """Compatibility projection for callers that only need the manifest."""

    return retrieval_capability_snapshot(db, knowledge_base_id)[0]


def _planning_system_prompt() -> str:
    schema = IntentPlanningOutput.model_json_schema()
    return "\n".join(
        [
            "INTENT EXECUTION RETRIEVAL V1. Return exactly one closed JSON object matching the schema.",
            "Plan the current user request once. Do not answer it and do not call or propose tools.",
            "Separate user requirements from retrieval expressions. Preserve entities, quantities, units, time, negation, comparison sides, source responsibilities, and response constraints.",
            "Choose only layers and channels listed in capabilities. The route may be retrieve, verified_context_reuse, system_capability, or clarify.",
            "system_capability is only for this service's identity, abilities, evidence policy, or usage. clarify is only for a real unresolved ambiguity.",
            "For corpus questions use retrieve, or verified_context_reuse only when verified_context_reuse_available=true and the current follow-up asks about the same named subject represented in the bounded history summary. When both conditions hold, route MUST be verified_context_reuse; the executor will replay the actual package and fall back to retrieval if validation fails. The history summary is routing context rather than answer evidence, so it never supplies answer facts.",
            "verified_context_reuse remains a retrieval-capable frozen plan: it MUST include a valid entry_layer, semantic_query, lexical decision, per-layer weights, selection_scope, and legal budget_request exactly like retrieve, so the executor can use that same plan if package replay fails.",
            "generate_lexical=false with lexical_groups=[] is valid. It requires hybrid=false and Dense=1/RQ=0/BM25=0 at every used layer.",
            "Lexical groups bind one retrieval concept, identifier, number/unit, or quoted literal to requirements. Each surface declares zh, en, or neutral plus user_text/model_query provenance. Identifiers use neutral. Do not treat a translated retrieval surface as a fact or corpus alias.",
            "When capabilities.bilingual_lexical_enabled=true and you choose generate_lexical=true, every kind=concept group MUST contain both a Chinese zh surface and an English en surface in the same group. Do not translate identifiers, codes, numbers, units, or quoted literals merely to satisfy this rule. This is part of the same planning response; do not propose another model step.",
            "Hybrid plans require non-empty lexical groups and positive Dense and BM25 weights. Weights at each used layer must be finite, nonnegative, and sum to one.",
            "Keep the JSON compact. Use plain text values without LaTeX backslashes or unescaped double quotes. Omit optional detail instead of adding prose. Shape example only: "
            + json.dumps(
                {
                    "protocol_version": PROTOCOL,
                    "intent": {"primary": "fact_lookup", "secondary": []},
                    "requirements": [
                        {
                            "id": "f1",
                            "text": "requested reliability fact",
                            "role": "topic",
                            "protected_literals": [],
                            "source_roles": [],
                            "source_scope": None,
                        }
                    ],
                    "entities": [],
                    "response_constraints": [],
                    "execution_strategy": {
                        "protocol_version": "intent_execution_strategy_v2",
                        "route": "retrieve",
                        "entry_layer": "chunk",
                        "semantic_query": "reliability fact",
                        "generate_lexical": True,
                        "lexical_groups": [
                            {
                                "group_id": "l1",
                                "requirement_ids": ["f1"],
                                "kind": "concept",
                                "surfaces": [
                                    {
                                        "text": "可靠性",
                                        "language": "zh",
                                        "provenance": "model_query",
                                    },
                                    {
                                        "text": "reliability",
                                        "language": "en",
                                        "provenance": "model_query",
                                    },
                                ],
                            }
                        ],
                        "hybrid": True,
                        "layer_weights": {
                            "chunk": {"dense": 0.5, "rq": 0.0, "bm25": 0.5}
                        },
                        "selection_scope": "focused",
                        "budget_request": {},
                        "reason_code": "mixed_signal",
                    },
                },
                ensure_ascii=False,
                separators=(",", ":"),
            ),
            "A coarse entry must provide coarse, mid, and chunk weights; mid provides mid and chunk; chunk provides chunk only.",
            "Choose the entry layer from the request and capability manifest rather than from a fixed intent mapping. If you set generate_lexical=true for a request whose answer depends on precise terms, labels, numbers, or local source facts, entry_layer MUST be chunk, because the capability manifest does not claim that higher-layer concepts cover every raw chunk. Use mid or coarse for theme-level entry where exact raw-chunk coverage is not required. Never rely on a higher layer to admit an otherwise unprojected lexical chunk.",
            "Do not infer facts from conversation summaries or capability identities. Do not emit private reasoning.",
            "Schema: " + json.dumps(schema, ensure_ascii=False, separators=(",", ":")),
        ]
    )


async def plan_intent_execution(
    db,
    *,
    run: AgentRun,
    question: str,
    conversation_scope_hash: str,
    filter_scope_hash: str,
    history_summary: str,
    capabilities: CapabilityManifest,
    verified_context_reuse_available: bool = False,
    provider_factory: Callable[[], Any] = ChatProvider,
) -> tuple[AcceptedPlan, dict]:
    """Persist prepared/completed call facts around one schema-validated model call."""

    settings = get_settings()
    system_prompt = _planning_system_prompt()
    packet = {
        "protocol_version": PLANNING_CALL_PROTOCOL,
        "question": question,
        "history_summary": {
            "text": history_summary,
            "is_evidence": False,
            "current_user_overrides": True,
        },
        "capabilities": capabilities.model_dump(mode="json"),
        "verified_context_reuse_available": bool(
            verified_context_reuse_available
        ),
        "filter_scope_hash": filter_scope_hash,
    }
    input_hash = control_hash(packet)
    prepared = {
        "protocol_version": PLANNING_CALL_PROTOCOL,
        "status": "prepared",
        "input_hash": input_hash,
        "capability_hash": capabilities.identity,
        "verified_context_reuse_available": bool(
            verified_context_reuse_available
        ),
        "timeout_seconds": min(
            float(settings.model_request_timeout_seconds),
            float(settings.retrieval_planning_timeout_seconds),
        ),
        "max_tokens": min(
            int(settings.chat_json_max_tokens),
            int(settings.retrieval_planning_max_tokens),
            PLANNING_MAX_TOKENS,
        ),
        "provider_response_persisted": False,
    }
    row = AgentObservation(
        run_id=run.id,
        observation_type="intent_execution_plan",
        verdict="prepared",
        observation_json=prepared,
    )
    db.add(row)
    db.commit()
    try:
        raw = await classify_json_with_budget(
            provider_factory(),
            system_prompt=system_prompt,
            user_prompt=json.dumps(packet, ensure_ascii=False, separators=(",", ":")),
            fallback=None,
            max_tokens=prepared["max_tokens"],
        )
    except ProviderJSONShapeError as exc:
        failed = {
            **prepared,
            "status": "failed",
            "failure_class": "provider_json_shape",
            "provider_shape": exc.diagnostics,
            "model_call_count": 1,
        }
        failed["audit_hash"] = control_hash(failed)
        row.verdict = "failed"
        row.observation_json = failed
        run.metadata_json = {
            **dict(run.metadata_json or {}),
            "intent_execution_plan": {
                "protocol_version": PLANNING_CALL_PROTOCOL,
                "observation_id": row.id,
                "status": "failed",
                "failure_class": "provider_json_shape",
                "provider_shape": exc.diagnostics,
            },
        }
        flag_modified(run, "metadata_json")
        db.commit()
        raise
    normalized_raw, normalization_audit = normalize_planning_output(raw)
    proposal = IntentPlanningOutput.model_validate(normalized_raw)
    accepted = accept_plan(
        proposal,
        question=question,
        conversation_scope_hash=conversation_scope_hash,
        conversation_identity_hash=control_hash({"qa_session_id": run.session_id}),
        filter_scope_hash=filter_scope_hash,
        capabilities=capabilities,
    )
    completed = {
        **prepared,
        "status": "completed",
        "observation_id": row.id,
        "proposal": proposal.model_dump(mode="json"),
        "proposal_hash": proposal.identity,
        "local_normalization": normalization_audit,
        "accepted_plan": accepted.model_dump(mode="json"),
        "accepted_plan_hash": accepted.identity,
        "model_call_count": 1,
    }
    completed["audit_hash"] = control_hash(completed)
    row.verdict = "completed"
    row.observation_json = completed
    run.metadata_json = {
        **dict(run.metadata_json or {}),
        "intent_execution_plan": {
            "protocol_version": accepted.protocol_version,
            "observation_id": row.id,
            "accepted_plan_hash": accepted.identity,
            "capability_hash": capabilities.identity,
        },
    }
    flag_modified(run, "metadata_json")
    db.commit()
    return accepted, completed
