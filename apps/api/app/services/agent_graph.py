from __future__ import annotations

import asyncio
import os
import re
import time
from collections.abc import AsyncGenerator
from datetime import datetime
from typing import Any, Literal, TypedDict

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel
from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from app.db import SessionLocal
from app.models import AgentRun, AgentTraceEvent, QASession
from app.schemas import AgentRequest, SearchFilters
from app.core.config import get_settings
from app.services.embeddings import ChatProvider, EmbeddingProvider, FallbackDisabledError, is_degraded_mode, prefers_chinese_answer
from app.services.ingestion import resolve_course
from app.services.retrieval import (
    assemble_evidence_documents,
    controlled_graph_enhancement,
    cosine_similarity,
    hybrid_search_chunks,
    plan_evidence_chains,
    select_evidence_anchors,
)
from app.services.strategy_profiles import active_profile_json, get_active_profile_record, profile_prompt, use_strategy_profile


AgentRoute = Literal["direct_answer", "retrieve_notes", "retrieve_exercises", "retrieve_both", "clarify", "multi_hop_research"]
NOTE_SOURCE_TYPES = {"pdf", "ppt", "pptx", "docx", "markdown", "text", "html", "image"}
EXERCISE_MARKERS = ("exercise", "homework", "problem", "assignment", "quiz", "exam")
_TRACE_SUBSCRIBERS: dict[str, set[asyncio.Queue[dict]]] = {}


def _profile_route_terms(kind: str, defaults: tuple[str, ...]) -> tuple[str, ...]:
    retrieval_strategy = (active_profile_json().get("retrieval_strategy") or {})
    route_terms = retrieval_strategy.get("route_terms") if isinstance(retrieval_strategy, dict) else None
    values = route_terms.get(kind) if isinstance(route_terms, dict) else None
    if isinstance(values, list):
        normalized = tuple(str(item).lower() for item in values if str(item).strip())
        return normalized or defaults
    return defaults


def _runtime_bool_env(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"true", "1", "yes", "on"}


def _runtime_int_env(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    try:
        return int(value)
    except ValueError:
        return default


class AgentState(TypedDict, total=False):
    run_id: str
    session_id: str
    course_id: str
    question: str
    rewritten_question: str
    history: list[dict]
    filters: SearchFilters
    top_k: int
    route: AgentRoute
    sub_queries: list[str]
    documents: list[dict]
    graded_documents: list[dict]
    context: str
    answer: str
    answer_model_audit: dict
    citations: list[dict]
    retry_count: int
    degraded_mode: bool
    low_confidence_docs: list[dict]
    reflection_result: dict
    unverified_citations: list[int]
    # New architecture fields
    perception_result: dict
    retrieval_strategy: str
    retrieval_params: dict
    evidence_evaluation: dict
    low_evidence: bool
    base_documents: list[dict]
    evidence_anchors: list[dict]
    evidence_chain_plan: list[dict]
    graph_enhanced_documents: list[dict]
    evidence_assembly: dict
    graph_navigation_audit: dict


class QueryAnalysis(BaseModel):
    normalized_question: str
    token_count: int
    likely_course_query: bool
    needs_clarification: bool
    is_multi_hop: bool


def _terms(text: str) -> set[str]:
    from app.services.chinese_text import extract_terms

    return extract_terms(text)


def _summarize(text: str, limit: int = 280) -> str:
    compact = re.sub(r"\s+", " ", text).strip()
    return compact[:limit]


def _direct_answer_text(question: str) -> str:
    profile = active_profile_json()
    if prefers_chinese_answer(question):
        return profile_prompt(profile, "agent_direct_answer_zh", "我可以回答已索引资料中的问题，提供引用，并说明检索智能体如何得到答案。")
    return profile_prompt(profile, "agent_direct_answer_en", "I can answer questions about the indexed course materials, show citations, and explain how the retrieval agent reached its answer.")
    if prefers_chinese_answer(question):
        return "我可以回答已索引课程材料中的问题，提供引用，并说明检索智能体如何得到答案。"
    return "I can answer questions about the indexed course materials, show citations, and explain how the retrieval agent reached its answer."


def _clarify_answer_text(question: str) -> str:
    profile = active_profile_json()
    if prefers_chinese_answer(question):
        return profile_prompt(profile, "agent_clarify_answer_zh", "请进一步说明你要检索的实体、分区、任务或比较问题。")
    return profile_prompt(profile, "agent_clarify_answer_en", "Please clarify the concept, section, task, or comparison you want me to retrieve.")
    if prefers_chinese_answer(question):
        return "请进一步说明你要检索的课程概念、章节、习题或比较问题。"
    return "Please clarify the course concept, chapter, exercise, or comparison you want me to retrieve."


def _correction_no_context_answer_text(question: str) -> str:
    profile = active_profile_json()
    if prefers_chinese_answer(question):
        return profile_prompt(
            profile,
            "agent_no_context_answer_zh",
            "资料中没有找到足够相关内容来回答这个问题。如果你希望我基于已检索到的有限资料尝试回答（可能包含推测），请告诉我。",
        )
    return profile_prompt(
        profile,
        "agent_no_context_answer_en",
        "I could not find enough relevant course material to answer this question. If you want me to try answering from the limited retrieved material, which may involve inference, please tell me.",
    )
    if prefers_chinese_answer(question):
        return (
            "课程材料中没有找到足够相关内容来回答这个问题。"
            "如果你希望我基于已经检索到的有限材料尝试回答（可能包含推测），请告诉我。"
        )
    return (
        "I could not find enough relevant course material to answer this question. "
        "If you want me to try answering from the limited retrieved material, which may involve inference, please tell me."
    )


def _set_run_state_sync(
    db: Session,
    run_id: str,
    state: str,
    *,
    current_node: str | None = None,
    route: str | None = None,
    retry_count: int | None = None,
    answer: str | None = None,
    error: str | None = None,
) -> None:
    run = db.get(AgentRun, run_id)
    if run is None:
        return
    run.status = state
    run.current_node = current_node
    if state == "running" and run.started_at is None:
        run.started_at = datetime.utcnow()
    if state in {"completed", "failed", "needs_clarification"}:
        run.completed_at = datetime.utcnow()
    if route is not None:
        run.route = route
    if retry_count is not None:
        run.retry_count = retry_count
    if answer is not None:
        run.final_answer = answer
    if error is not None:
        run.error_message = error
    db.commit()


async def _set_run_state(
    run_id: str,
    state: str,
    *,
    current_node: str | None = None,
    route: str | None = None,
    retry_count: int | None = None,
    answer: str | None = None,
    error: str | None = None,
) -> None:
    """Async variant that creates its own Session to avoid blocking the event loop and leaking Session across LangGraph nodes."""

    def _sync() -> None:
        with SessionLocal() as db:
            _set_run_state_sync(db, run_id, state, current_node=current_node, route=route, retry_count=retry_count, answer=answer, error=error)

    await asyncio.to_thread(_sync)


def _trace_sync(
    db: Session,
    run_id: str,
    node: str,
    *,
    input_summary: str | None = None,
    output_summary: str | None = None,
    document_ids: list[str] | None = None,
    scores: dict | None = None,
    duration_ms: int = 0,
    status: str = "completed",
    error: str | None = None,
) -> dict:
    event = AgentTraceEvent(
        run_id=run_id,
        node=node,
        status=status,
        input_summary=input_summary,
        output_summary=output_summary,
        document_ids=document_ids or [],
        scores=scores or {},
        duration_ms=duration_ms,
        error_message=error,
    )
    db.add(event)
    db.commit()
    db.refresh(event)
    payload = trace_event_to_payload(event)
    _publish_trace_event(run_id, payload)
    return payload


async def _trace(
    run_id: str,
    node: str,
    *,
    input_summary: str | None = None,
    output_summary: str | None = None,
    document_ids: list[str] | None = None,
    scores: dict | None = None,
    duration_ms: int = 0,
    status: str = "completed",
    error: str | None = None,
) -> dict:
    """Async variant that creates its own Session to avoid blocking the event loop and leaking Session across LangGraph nodes."""

    def _sync() -> dict:
        with SessionLocal() as db:
            return _trace_sync(db, run_id, node, input_summary=input_summary, output_summary=output_summary, document_ids=document_ids, scores=scores, duration_ms=duration_ms, status=status, error=error)

    return await asyncio.to_thread(_sync)


def _subscribe_trace(run_id: str) -> asyncio.Queue[dict]:
    queue: asyncio.Queue[dict] = asyncio.Queue()
    _TRACE_SUBSCRIBERS.setdefault(run_id, set()).add(queue)
    return queue


def _unsubscribe_trace(run_id: str, queue: asyncio.Queue[dict]) -> None:
    subscribers = _TRACE_SUBSCRIBERS.get(run_id)
    if not subscribers:
        return
    subscribers.discard(queue)
    if not subscribers:
        _TRACE_SUBSCRIBERS.pop(run_id, None)


def _publish_trace_event(run_id: str, payload: dict) -> None:
    for queue in list(_TRACE_SUBSCRIBERS.get(run_id, ())):
        queue.put_nowait(payload)


def trace_event_to_payload(event: AgentTraceEvent) -> dict:
    return {
        "id": event.id,
        "run_id": event.run_id,
        "node": event.node,
        "status": event.status,
        "input_summary": event.input_summary,
        "output_summary": event.output_summary,
        "document_ids": event.document_ids or [],
        "scores": event.scores or {},
        "duration_ms": event.duration_ms,
        "error": event.error_message,
        "created_at": event.created_at,
    }


class Perception:
    """Perception layer: understand intent, extract entities, query graph for context."""

    async def __call__(self, state: AgentState) -> dict:
        start = time.perf_counter()
        db = SessionLocal()
        question = state["question"].strip()
        course_id = state["course_id"]

        # Fast-path: detect clarify / direct_answer before expensive LLM call
        question_lower = question.lower()
        tokens = _terms(question)
        is_greeting = question_lower in {
            "hello", "hi", "hey", "who are you", "help", "what can you do",
            "你好", "您好", "嗨", "你是谁", "帮助", "你能做什么",
        } or bool(tokens.intersection({"hello", "hey"}) or (tokens == {"hi"}))
        needs_clarification = len(tokens) == 0 or question_lower in {"it", "this", "that", "explain it", "why", "这个", "那个", "解释一下", "为什么"}

        if is_greeting:
            route: AgentRoute = "direct_answer"
        elif needs_clarification:
            route = "clarify"
        elif any(term in question_lower for term in ("compare", "relationship", "related to", "relation between", "difference between", "connect", "derive", "prove", "比较", "关系", "区别", "联系", "推导", "证明")):
            route = "multi_hop_research"
        elif any(term in question_lower for term in _profile_route_terms("exercise", EXERCISE_MARKERS)):
            route = "retrieve_exercises"
        elif any(term in question_lower for term in _profile_route_terms("notes", ("note", "slide", "definition", "concept", "chapter"))):
            route = "retrieve_notes"
        else:
            route = "retrieve_both"

        # Skip expensive LLM + graph lookup for simple routes
        if route in {"direct_answer", "clarify"}:
            await _set_run_state( state["run_id"], "running", current_node="perception", route=route)
            await _trace(
                                state["run_id"],
                "perception",
                input_summary=question,
                output_summary=f"fast_path route={route}",
                duration_ms=int((time.perf_counter() - start) * 1000),
            )
            return {"perception_result": {"intent": "unknown", "entities": [], "sub_queries": [question], "needs_graph": False, "suggested_strategy": "base_retrieval"}, "route": route}

        # 1. LLM perception
        llm_perception: dict[str, Any]
        if is_degraded_mode():
            settings = get_settings()
            if not settings.enable_model_fallback:
                raise FallbackDisabledError("Perception requires configured model credentials because ENABLE_MODEL_FALLBACK is false")
            llm_perception = {
                "intent": "unknown",
                "entities": [],
                "sub_queries": [question],
                "needs_graph": False,
                "suggested_strategy": "base_retrieval",
            }
        else:
            try:
                llm_perception = await ChatProvider().perceive_question(question, state.get("history", []))
            except Exception:
                settings = get_settings()
                if not settings.enable_model_fallback:
                    raise
                llm_perception = {
                    "intent": "unknown",
                    "entities": [],
                    "sub_queries": [question],
                    "needs_graph": False,
                    "suggested_strategy": "base_retrieval",
                }

        # 2. Graph entity matching
        matched_concepts: list[dict] = []
        seed_concept_ids: set[str] = set()
        perceived_communities: set[int | None] = set()
        perceived_neighbors: list[dict] = []

        entities = llm_perception.get("entities", [])
        if entities:
            from app.models import Concept, ConceptAlias, ConceptRelation
            from app.services.concept_graph import normalize_concept_name

            normalized_entities = [normalize_concept_name(e) for e in entities if isinstance(e, str)]
            if normalized_entities:
                # Match by normalized_name or alias
                alias_matches = db.scalars(
                    select(ConceptAlias).where(
                        ConceptAlias.normalized_alias.in_(normalized_entities),
                    )
                ).all()
                matched_ids = {a.concept_id for a in alias_matches}

                name_matches = db.scalars(
                    select(Concept).where(
                        Concept.course_id == course_id,
                        Concept.normalized_name.in_(normalized_entities),
                    )
                ).all()
                matched_ids |= {c.id for c in name_matches}

                if matched_ids:
                    concepts = db.scalars(
                        select(Concept).where(Concept.id.in_(list(matched_ids)), Concept.course_id == course_id)
                    ).all()
                    for concept in concepts:
                        matched_concepts.append({
                            "id": concept.id,
                            "name": concept.canonical_name,
                            "community": concept.community_louvain,
                        })
                        seed_concept_ids.add(concept.id)
                        if concept.community_louvain is not None:
                            perceived_communities.add(concept.community_louvain)

                    # Fetch 1-hop neighbors
                    if seed_concept_ids:
                        neighbor_relations = db.scalars(
                            select(ConceptRelation).where(
                                ConceptRelation.course_id == course_id,
                                or_(
                                    ConceptRelation.source_concept_id.in_(seed_concept_ids),
                                    ConceptRelation.target_concept_id.in_(seed_concept_ids),
                                ),
                            )
                        ).all()
                        for rel in neighbor_relations:
                            neighbor_id = rel.target_concept_id if rel.source_concept_id in seed_concept_ids else rel.source_concept_id
                            if neighbor_id and neighbor_id not in seed_concept_ids:
                                perceived_neighbors.append({
                                    "concept_id": neighbor_id,
                                    "relation_type": rel.relation_type,
                                    "via": rel.source_concept_id if rel.source_concept_id in seed_concept_ids else rel.target_concept_id,
                                })

        perception_result = {
            "intent": llm_perception.get("intent", "unknown"),
            "entities": entities,
            "matched_concepts": matched_concepts,
            "perceived_communities": sorted(perceived_communities) if perceived_communities else [],
            "perceived_neighbors": perceived_neighbors,
            "sub_queries": llm_perception.get("sub_queries", [question]),
            "needs_graph": llm_perception.get("needs_graph", False),
            "suggested_strategy": llm_perception.get("suggested_strategy", "base_retrieval"),
        }

        await _set_run_state( state["run_id"], "running", current_node="perception", route=route)
        await _trace(
                        state["run_id"],
            "perception",
            input_summary=question,
            output_summary=(
                f"intent={perception_result['intent']} "
                f"entities={len(perception_result['entities'])} "
                f"matched={len(matched_concepts)} "
                f"strategy={perception_result['suggested_strategy']}"
            ),
            duration_ms=int((time.perf_counter() - start) * 1000),
        )
        return {"perception_result": perception_result, "route": route}


async def _llm_translate_query(question: str, target_lang: str) -> str:
    """Minimal LLM query translator for cross-lingual retrieval."""
    chat = ChatProvider()
    search_domain = profile_prompt(active_profile_json(), "query_translation_domain", "academic course search")
    payload = {
        "model": chat.settings.chat_model,
        "messages": [
            {
                "role": "system",
                "content": (
                    f"You are a query translator for {search_domain}. "
                    f"Translate the user query to {target_lang} accurately, "
                    f"preserving all technical terminology. Output ONLY the translated query, no explanation."
                ),
            },
            {"role": "user", "content": question},
        ],
        "temperature": 0.0,
    }
    try:
        return await chat._post_chat_text(payload)
    except Exception:
        return question


class RetrievalPlanner:
    """Planning layer: choose retrieval strategy based on perception."""

    async def __call__(self, state: AgentState) -> dict:
        start = time.perf_counter()
        db = SessionLocal()
        perception = state.get("perception_result", {})
        intent = perception.get("intent", "unknown")
        needs_graph = perception.get("needs_graph", False)
        matched_concepts = perception.get("matched_concepts", [])

        strategy = "evidence_first"
        use_graph = bool(needs_graph or intent in {"comparison", "procedure"} or state.get("route") == "multi_hop_research")
        use_community = bool(intent == "analysis" or perception.get("suggested_strategy") == "community")

        top_k = state["top_k"]
        # Increase recall for broader intents
        if intent in {"comparison", "analysis"}:
            top_k = max(top_k, 8)

        seed_concept_ids = [c["id"] for c in matched_concepts]
        original_sub_queries = perception.get("sub_queries", [state["question"]])

        # Cross-lingual retrieval: translate the query so BM25 can match
        # documents in the other language as well.
        from app.services.chinese_text import contains_chinese
        has_zh = contains_chinese(state["question"])
        if has_zh:
            translated = await _llm_translate_query(state["question"], "English")
        else:
            translated = await _llm_translate_query(state["question"], "Chinese")
        sub_queries = list(dict.fromkeys([state["question"], translated] + original_sub_queries))

        retrieval_params = {
            "top_k": top_k,
            "graph_seed_concept_ids": seed_concept_ids if use_graph else [],
            "community_ids": perception.get("perceived_communities", []),
            "sub_queries": sub_queries,
            "use_graph": use_graph,
            "use_community": use_community,
        }

        await _set_run_state( state["run_id"], "running", current_node="retrieval_planner")
        await _trace(
                        state["run_id"],
            "retrieval_planner",
            input_summary=f"intent={intent} needs_graph={needs_graph}",
            output_summary=f"strategy={strategy} top_k={top_k} queries={len(sub_queries)} graph={use_graph} community={use_community}",
            duration_ms=int((time.perf_counter() - start) * 1000),
        )
        return {
            "retrieval_strategy": strategy,
            "retrieval_params": retrieval_params,
            "rewritten_question": translated,
        }


class QueryAnalyzer:
    async def __call__(self, state: AgentState) -> dict:
        start = time.perf_counter()
        db = SessionLocal()
        question = state["question"].strip()
        tokens = _terms(question)
        lower = question.lower()
        analysis = QueryAnalysis(
            normalized_question=question,
            token_count=len(tokens),
            likely_course_query=not any(term in lower for term in ("weather", "stock price", "joke", "movie ticket")),
            needs_clarification=len(tokens) == 0 or lower in {"it", "this", "that", "explain it", "why", "这个", "那个", "解释一下", "为什么"},
            is_multi_hop=any(term in lower for term in ("compare", "relationship", "difference between", "connect", "derive", "prove")),
        )
        await _set_run_state( state["run_id"], "running", current_node="query_analyzer")
        await _trace(
                        state["run_id"],
            "query_analyzer",
            input_summary=question,
            output_summary=analysis.model_dump_json(),
            duration_ms=int((time.perf_counter() - start) * 1000),
        )
        return {"rewritten_question": analysis.normalized_question, "sub_queries": [], "degraded_mode": is_degraded_mode()}


class Router:
    async def __call__(self, state: AgentState) -> dict:
        start = time.perf_counter()
        db = SessionLocal()
        question = state["question"]
        lower = question.lower()
        terms = _terms(question)
        is_greeting = lower in {
            "hello", "hi", "hey", "who are you", "help", "what can you do",
            "你好", "您好", "嗨", "你是谁", "帮助", "你能做什么",
        } or bool(
            terms.intersection({"hello", "hey"}) or (terms == {"hi"})
        )
        if is_greeting:
            route: AgentRoute = "direct_answer"
        elif len(terms) == 0 or lower in {"it", "this", "that", "explain it", "why", "这个", "那个", "解释一下", "为什么"}:
            route = "clarify"
        elif any(marker in lower for marker in _profile_route_terms("exercise", EXERCISE_MARKERS)):
            route = "retrieve_exercises"
        elif any(term in lower for term in ("compare", "relationship", "related to", "relation between", "difference between", "connect", "derive", "prove", "比较", "关系", "区别", "联系", "推导", "证明")):
            route = "multi_hop_research"
        elif any(term in lower for term in _profile_route_terms("notes", ("note", "slide", "definition", "concept", "chapter"))):
            route = "retrieve_notes"
        else:
            route = "retrieve_both"
        await _set_run_state( state["run_id"], "running", current_node="router", route=route)
        await _trace(
                        state["run_id"],
            "router",
            input_summary=question,
            output_summary=f"route={route}",
            duration_ms=int((time.perf_counter() - start) * 1000),
        )
        return {"route": route}


class QueryRewriter:
    async def __call__(self, state: AgentState) -> dict:
        start = time.perf_counter()
        db = SessionLocal()
        chat = ChatProvider()
        rewritten = await chat.rewrite_question(state["question"], state.get("history", []))
        if state.get("route") == "multi_hop_research":
            sub_queries = split_multi_hop_query(rewritten)
        else:
            sub_queries = [rewritten]
        await _set_run_state( state["run_id"], "running", current_node="query_rewriter")
        await _trace(
                        state["run_id"],
            "query_rewriter",
            input_summary=state["question"],
            output_summary=" | ".join(sub_queries),
            duration_ms=int((time.perf_counter() - start) * 1000),
        )
        return {"rewritten_question": rewritten, "sub_queries": sub_queries}


class RetrievalDecision:
    async def __call__(self, state: AgentState) -> dict:
        start = time.perf_counter()
        db = SessionLocal()
        settings = get_settings()
        skip = False
        reason = "default"
        if not settings.enable_agentic_reflection or is_degraded_mode():
            skip = False
            reason = "disabled_or_degraded"
        else:
            history = state.get("history", [])
            if not history:
                skip = False
                reason = "no_history"
            else:
                question = state["question"].strip().lower()
                pronoun_hints = {"it", "this", "that", "the above", "the former", "the latter", "这个", "那个", "上述", "前者", "后者"}
                import re
                if any(re.search(rf"\b{re.escape(hint)}\b", question) for hint in pronoun_hints) and len(history) >= 2:
                    skip = True
                    reason = "pronoun_reference"
                else:
                    skip = False
                    reason = "not_reference"
        await _set_run_state( state["run_id"], "running", current_node="retrieval_decision")
        await _trace(
                        state["run_id"],
            "retrieval_decision",
            input_summary=state["question"],
            output_summary=f"skip_retrieval={skip} reason={reason}",
            duration_ms=int((time.perf_counter() - start) * 1000),
        )
        return {"skip_retrieval": skip}


class BaseRetrieval:
    """Run base evidence recall without graph expansion."""

    async def __call__(self, state: AgentState) -> dict:
        start = time.perf_counter()
        db = SessionLocal()
        params = state.get("retrieval_params", {})
        course_id = state["course_id"]
        filters = state["filters"]
        top_k = params.get("top_k", state["top_k"])
        queries = params.get("sub_queries", [state.get("rewritten_question") or state["question"]])
        route = state.get("route", "retrieve_both")

        all_results: dict[str, dict] = {}
        for query in queries:
            for flt in expand_route_filters(route, filters):
                results = await hybrid_search_chunks(db, course_id, query, flt, max(top_k * 3, top_k))
                for result in results:
                    result.setdefault("metadata", {})["retrieval_stage"] = "base_retrieval"
                    result.setdefault("metadata", {})["evidence_role"] = "base_candidate"
                    result.setdefault("metadata", {})["graph_verified"] = False
                    current = all_results.get(result["chunk_id"])
                    if current is None or result["score"] > current["score"]:
                        all_results[result["chunk_id"]] = result

        documents = sorted(all_results.values(), key=lambda item: item["score"], reverse=True)[: max(top_k * 2, top_k)]
        await _set_run_state( state["run_id"], "running", current_node="base_retrieval")
        await _trace(
                        state["run_id"],
            "base_retrieval",
            input_summary=f"route={route} | " + " | ".join(queries),
            output_summary=f"base_candidates={len(documents)}",
            document_ids=[item["chunk_id"] for item in documents],
            scores={item["chunk_id"]: item.get("metadata", {}).get("scores", {}) for item in documents},
            duration_ms=int((time.perf_counter() - start) * 1000),
        )
        return {"base_documents": documents, "documents": documents}


class EvidenceAnchorSelector:
    async def __call__(self, state: AgentState) -> dict:
        start = time.perf_counter()
        db = SessionLocal()
        anchors, audit = select_evidence_anchors(db, state["course_id"], state.get("base_documents", []))
        await _set_run_state( state["run_id"], "running", current_node="evidence_anchor_selector")
        await _trace(
                        state["run_id"],
            "evidence_anchor_selector",
            input_summary=f"{len(state.get('base_documents', []))} base candidates",
            output_summary=(
                f"anchors={audit.get('anchor_count', 0)} "
                f"verified_anchor_relations={audit.get('verified_anchor_relations', 0)}"
            ),
            document_ids=[item["chunk_id"] for item in anchors],
            scores={item["chunk_id"]: {"anchor_score": item.get("metadata", {}).get("anchor_score")} for item in anchors},
            duration_ms=int((time.perf_counter() - start) * 1000),
        )
        return {"evidence_anchors": anchors, "graph_navigation_audit": {"anchors": audit}}


class EvidenceChainPlanner:
    async def __call__(self, state: AgentState) -> dict:
        start = time.perf_counter()
        db = SessionLocal()
        params = state.get("retrieval_params", {})
        perception = state.get("perception_result", {})
        query_type = perception.get("intent") or "default"
        use_graph = bool(params.get("use_graph") or state.get("route") == "multi_hop_research")
        use_community = bool(params.get("use_community"))
        paths: list[dict] = []
        audit = {"planned_paths": 0, "verified_edges": 0, "skipped_reason": "simple_query"}
        if use_graph or use_community:
            paths, audit = plan_evidence_chains(
                db,
                state["course_id"],
                state.get("evidence_anchors", []),
                query_type=query_type,
                community_ids=params.get("community_ids") if use_community else [],
            )
        previous_audit = dict(state.get("graph_navigation_audit", {}))
        previous_audit["paths"] = audit
        await _set_run_state( state["run_id"], "running", current_node="evidence_chain_planner")
        await _trace(
                        state["run_id"],
            "evidence_chain_planner",
            input_summary=f"use_graph={use_graph} use_community={use_community}",
            output_summary=(
                f"planned_paths={audit.get('planned_paths', 0)} "
                f"verified_edges={audit.get('verified_edges', 0)} "
                f"community_summaries={audit.get('community_summaries', 0)}"
            ),
            scores={"audit": audit},
            duration_ms=int((time.perf_counter() - start) * 1000),
        )
        return {"evidence_chain_plan": paths, "graph_navigation_audit": previous_audit}


class ControlledGraphEnhancer:
    async def __call__(self, state: AgentState) -> dict:
        start = time.perf_counter()
        db = SessionLocal()
        params = state.get("retrieval_params", {})
        top_query = (params.get("sub_queries") or [state.get("rewritten_question") or state["question"]])[0]
        docs, audit = controlled_graph_enhancement(
            db,
            state["course_id"],
            top_query,
            state["filters"],
            {str(item["chunk_id"]) for item in state.get("base_documents", [])},
            state.get("evidence_chain_plan", []),
        )
        previous_audit = dict(state.get("graph_navigation_audit", {}))
        previous_audit["graph"] = audit
        await _set_run_state( state["run_id"], "running", current_node="controlled_graph_enhancer")
        await _trace(
                        state["run_id"],
            "controlled_graph_enhancer",
            input_summary=f"{len(state.get('evidence_chain_plan', []))} planned paths",
            output_summary=(
                f"graph_enhanced_chunks={audit.get('graph_enhanced_chunks', 0)} "
                f"path_evidence_chunks={audit.get('path_evidence_chunks', 0)}"
            ),
            document_ids=[item["chunk_id"] for item in docs],
            scores={"audit": audit},
            duration_ms=int((time.perf_counter() - start) * 1000),
        )
        return {"graph_enhanced_documents": docs, "graph_navigation_audit": previous_audit}


class EvidenceAssembler:
    async def __call__(self, state: AgentState) -> dict:
        start = time.perf_counter()
        db = SessionLocal()
        docs, audit = assemble_evidence_documents(
            state.get("base_documents", []),
            state.get("evidence_anchors", []),
            state.get("graph_enhanced_documents", []),
            state["top_k"],
        )
        await _set_run_state( state["run_id"], "running", current_node="evidence_assembler")
        await _trace(
                        state["run_id"],
            "evidence_assembler",
            input_summary=f"base={audit.get('base_documents', 0)} graph={audit.get('graph_documents', 0)}",
            output_summary=f"assembled_documents={audit.get('assembled_documents', 0)} anchor_documents={audit.get('anchor_documents', 0)}",
            document_ids=[item["chunk_id"] for item in docs],
            scores={"audit": audit},
            duration_ms=int((time.perf_counter() - start) * 1000),
        )
        return {"documents": docs, "evidence_assembly": audit}


class DocumentGrader:
    async def __call__(self, state: AgentState) -> dict:
        start = time.perf_counter()
        question = state.get("rewritten_question") or state["question"]

        query_variants: list[tuple[str, str, set[str]]] = []
        seen_queries: set[str] = set()

        def add_query_variant(label: str, value: str | None) -> None:
            normalized = (value or "").strip()
            if not normalized or normalized in seen_queries:
                return
            seen_queries.add(normalized)
            query_variants.append((label, normalized, _terms(normalized)))

        add_query_variant("original_question", state.get("question"))
        add_query_variant("rewritten_question", state.get("rewritten_question"))
        for sub_query in state.get("retrieval_params", {}).get("sub_queries", []):
            add_query_variant("sub_query", sub_query)
        if not query_variants:
            add_query_variant("question", question)

        graded = []
        low_confidence = []

        # Compute query embeddings once for similarity scoring.  All query
        # variants are considered so cross-language rewrites cannot mask a
        # high-quality same-language match.
        query_vectors: list[list[float]] = []
        settings = get_settings()
        if settings.retrieval_layer_enabled and not is_degraded_mode():
            try:
                embedder = EmbeddingProvider()
                query_vectors = await embedder.embed_texts([item[1] for item in query_variants], text_type="query")
            except Exception:
                query_vectors = []

        for document in state.get("documents", []):
            haystack = f"{document.get('document_title', '')} {document.get('snippet', '')} {document.get('content', '')}"
            haystack_terms = _terms(haystack)
            max_overlap = 0
            overlap_ratio = 0.0
            overlap_source = "none"
            for label, _variant, terms in query_variants:
                overlap = len(terms.intersection(haystack_terms))
                current_ratio = overlap / max(len(terms), 1)
                if current_ratio > overlap_ratio or (current_ratio == overlap_ratio and overlap > max_overlap):
                    max_overlap = overlap
                    overlap_ratio = current_ratio
                    overlap_source = label

            metadata = document.setdefault("metadata", {})
            scores = metadata.setdefault("scores", {})
            scores["grader_overlap"] = max_overlap
            scores["term_overlap_ratio"] = round(overlap_ratio, 4)
            metadata["grader_overlap_source"] = overlap_source
            metadata["grader_query_variant_count"] = len(query_variants)

            # Embedding similarity boost
            embedding_sim = 0.0
            if query_vectors and document.get("metadata", {}).get("embedding_vector"):
                try:
                    embedding_sim = max(
                        cosine_similarity(query_vector, document["metadata"]["embedding_vector"])
                        for query_vector in query_vectors
                    )
                except Exception:
                    embedding_sim = 0.0
            if embedding_sim == 0.0:
                # Fallback: the dense score from retrieval is already the
                # query-vs-chunk embedding similarity computed by the vector
                # store.  build_search_payload does not carry the raw vector,
                # but it does carry the dense score in metadata.scores.dense.
                dense_score = float(
                    document.get("metadata", {}).get("scores", {}).get("dense") or 0.0
                )
                if dense_score > 0:
                    embedding_sim = dense_score
            scores["grader_embedding_sim"] = round(embedding_sim, 4)

            grade_score = 0.4 * overlap_ratio + 0.6 * embedding_sim
            scores["grade_score"] = round(grade_score, 4)

            # Primary: composite grade score is high enough
            primary_pass = grade_score >= 0.35
            # Cross-language bridge: if embedding similarity is strong, the chunk is
            # semantically related even when term overlap is zero (e.g. Chinese query
            # vs. English course material).  This prevents multilingual dense recall
            # from being killed by monolingual term-matching.
            cross_lang_pass = embedding_sim >= 0.45
            # Secondary: reasonable term overlap AND original retrieval score is not noise
            secondary_pass = overlap_ratio >= 0.25 and document["score"] >= 0.3

            if primary_pass or cross_lang_pass or secondary_pass:
                graded.append(document)
            else:
                low_confidence.append(document)

        retry_count = state.get("retry_count", 0)
        await _set_run_state( state["run_id"], "running", current_node="document_grader", retry_count=retry_count)
        await _trace(
                        state["run_id"],
            "document_grader",
            input_summary=f"{len(state.get('documents', []))} candidates",
            output_summary=f"{len(graded)} accepted, {len(low_confidence)} low-confidence",
            document_ids=[item["chunk_id"] for item in graded],
            scores={item["chunk_id"]: item.get("metadata", {}).get("scores", {}) for item in graded},
            duration_ms=int((time.perf_counter() - start) * 1000),
        )
        return {"graded_documents": graded[: state["top_k"]], "retry_count": retry_count, "low_confidence_docs": low_confidence}


class EvidenceEvaluator:
    """Pre-generation evidence sufficiency evaluator.

    Uses heuristics (document count, grade-score distribution, intent-based
    thresholds) to decide whether retrieved evidence is sufficient to answer
    the question. If insufficient and retry budget remains, routes back to the
    retrieval planner with expanded parameters. Otherwise sets ``low_evidence``
    so the generator can produce a cautious / graceful answer.
    """

    async def __call__(self, state: AgentState) -> dict:
        start = time.perf_counter()
        docs = state.get("graded_documents", [])
        perception = state.get("perception_result", {})
        retry_count = state.get("retry_count", 0)
        top_k = state["top_k"]
        intent = perception.get("intent", "analysis")

        if not docs:
            sufficient = False
            score = 0.0
            avg_score = max_score = avg_overlap = max_overlap = 0.0
            reason = "no_documents"
        else:
            grade_scores = [
                float(d.get("metadata", {}).get("scores", {}).get("grade_score", 0))
                for d in docs
            ]
            overlap_scores = [
                float(d.get("metadata", {}).get("scores", {}).get("term_overlap_ratio", 0))
                for d in docs
            ]
            embedding_scores = [
                float(d.get("metadata", {}).get("scores", {}).get("grader_embedding_sim", 0))
                for d in docs
            ]
            avg_score = sum(grade_scores) / len(grade_scores)
            max_score = max(grade_scores)
            avg_overlap = sum(overlap_scores) / len(overlap_scores)
            max_overlap = max(overlap_scores)
            max_embedding = max(embedding_scores)

            # Intent-based thresholds
            if intent in ("definition", "procedure"):
                min_docs_needed = 1
                min_avg_score = 0.25
            elif intent in ("comparison", "analysis"):
                min_docs_needed = 2
                min_avg_score = 0.20
            else:
                min_docs_needed = 1
                min_avg_score = 0.20

            has_strong_evidence = max_score >= 0.35 or max_embedding >= 0.45 or max_overlap >= 0.5
            doc_count_ok = len(docs) >= min_docs_needed
            score_ok = avg_score >= min_avg_score or avg_overlap >= 0.35
            reason_metrics = (
                f"avg_score={avg_score:.2f} max_score={max_score:.2f} "
                f"avg_overlap={avg_overlap:.2f} max_overlap={max_overlap:.2f} "
                f"docs={len(docs)} needed={min_docs_needed}"
            )

            if has_strong_evidence and doc_count_ok and score_ok:
                sufficient = True
                score = avg_score
                reason = f"sufficient {reason_metrics}"
            else:
                sufficient = False
                score = avg_score
                reason = reason_metrics

        evaluation = {
            "sufficient": sufficient,
            "score": round(score, 4),
            "reason": reason,
            "avg_score": round(avg_score, 4),
            "max_score": round(max_score, 4),
            "avg_overlap": round(avg_overlap, 4),
            "max_overlap": round(max_overlap, 4),
            "retry_count": retry_count,
            "anchor_count": len(state.get("evidence_anchors", [])),
            "planned_paths": len(state.get("evidence_chain_plan", [])),
            "graph_enhanced_chunks": len(state.get("graph_enhanced_documents", [])),
        }

        output: dict[str, Any] = {"evidence_evaluation": evaluation}

        if not sufficient and retry_count < 2:
            # Expand recall and retry through the planning layer.
            output["retry_count"] = retry_count + 1
            output["top_k"] = top_k * 2
        elif not sufficient and retry_count >= 2:
            output["low_evidence"] = True

        await _set_run_state( state["run_id"], "running", current_node="evidence_evaluator")
        await _trace(
                        state["run_id"],
            "evidence_evaluator",
            input_summary=f"intent={intent} docs={len(docs)} retry={retry_count}",
            output_summary=f"sufficient={sufficient} score={evaluation['score']} reason={reason}",
            document_ids=[item["chunk_id"] for item in docs],
            scores={"audit": evaluation},
            duration_ms=int((time.perf_counter() - start) * 1000),
        )
        return output


class RetryPlanner:
    async def __call__(self, state: AgentState) -> dict:
        start = time.perf_counter()
        db = SessionLocal()
        retry_count = state.get("retry_count", 0) + 1
        retry_suffix = profile_prompt(active_profile_json(), "retry_query_suffix", "course lecture notes examples")
        next_question = f"{state.get('rewritten_question') or state['question']} {retry_suffix}"
        await _set_run_state( state["run_id"], "running", current_node="retry_planner", retry_count=retry_count)
        await _trace(
                        state["run_id"],
            "retry_planner",
            input_summary=f"retry={retry_count}",
            output_summary=next_question,
            duration_ms=int((time.perf_counter() - start) * 1000),
        )
        return {"retry_count": retry_count, "rewritten_question": next_question, "sub_queries": [next_question]}


class ContextSynthesizer:
    async def __call__(self, state: AgentState) -> dict:
        start = time.perf_counter()
        db = SessionLocal()
        docs = state.get("graded_documents", [])
        if not docs:
            context = ""
        else:
            # Token budget allocation by rerank score
            total_budget = 6000  # approximate token budget for context
            scores = [float(item.get("score", 0.0)) for item in docs]
            min_score = min(scores) if scores else 0
            max_score = max(scores) if scores else 1
            score_range = max_score - min_score if max_score != min_score else 1

            parts = []
            for idx, item in enumerate(docs):
                normalized_score = (float(item.get("score", 0.0)) - min_score) / score_range
                # Budget: base 800 + up to 1200 extra for highest score
                char_budget = int(800 + normalized_score * 1200)
                content = item.get("content", "")[:char_budget]
                parts.append(f"[{idx + 1}] {item['document_title']} / {item.get('chapter') or 'General'}\n{content}")
            context = "\n\n".join(parts)

        await _set_run_state( state["run_id"], "running", current_node="context_synthesizer")
        await _trace(
                        state["run_id"],
            "context_synthesizer",
            input_summary=f"{len(docs)} graded chunks",
            output_summary=_summarize(context),
            document_ids=[item["chunk_id"] for item in docs],
            duration_ms=int((time.perf_counter() - start) * 1000),
        )
        return {"context": context}


class AnswerGenerator:
    async def __call__(self, state: AgentState) -> dict:
        start = time.perf_counter()
        route = state.get("route", "retrieve_both")
        if route == "direct_answer":
            answer = _direct_answer_text(state["question"])
            citations: list[dict] = []
            used_chunks: list[dict] = []
            state["answer_model_audit"] = {
                "provider": "none",
                "model": get_settings().chat_model,
                "external_called": False,
                "fallback_reason": None,
                "skipped_reason": "direct_answer_route",
            }
        elif route == "clarify":
            answer = _clarify_answer_text(state["question"])
            citations = []
            used_chunks = []
            state["answer_model_audit"] = {
                "provider": "none",
                "model": get_settings().chat_model,
                "external_called": False,
                "fallback_reason": None,
                "skipped_reason": "clarify_route",
            }
        else:
            used_chunks = state.get("graded_documents", [])
            retry_count = state.get("retry_count", 0)
            # During reflection-correction retry, if no documents remain after correction,
            # produce a graceful response instead of the generic no_contexts fallback.
            if not used_chunks and retry_count > 0:
                answer = _correction_no_context_answer_text(state["question"])
                state["answer_model_audit"] = {
                    "provider": "none",
                    "model": get_settings().chat_model,
                    "external_called": False,
                    "fallback_reason": "correction_no_contexts",
                    "skipped_reason": None,
                }
                citations = []
            else:
                low_evidence = state.get("low_evidence", False)
                chat_result = await ChatProvider().answer_question_with_meta(
                    state["question"],
                    used_chunks,
                    state.get("history", []),
                    evidence_quality="low" if low_evidence else "normal",
                )
                answer = chat_result.answer
                state["answer_model_audit"] = {
                    "provider": chat_result.provider,
                    "model": chat_result.model,
                    "external_called": chat_result.external_called,
                    "fallback_reason": chat_result.fallback_reason,
                    "skipped_reason": None,
                }
                # When evidence is marked low, do not force citations from potentially irrelevant chunks.
                if low_evidence:
                    citations = []
                else:
                    citations = [citation for item in used_chunks for citation in item["citations"]]
        audit = state.get("answer_model_audit", {})
        await _set_run_state( state["run_id"], "running", current_node="answer_generator")
        await _trace(
                        state["run_id"],
            "answer_generator",
            input_summary=state["question"],
            output_summary=(
                f"model={audit.get('model')} provider={audit.get('provider')} "
                f"external_called={audit.get('external_called')} fallback={audit.get('fallback_reason')}\n"
                f"{_summarize(answer)}"
            ),
            document_ids=[item["chunk_id"] for item in state.get("graded_documents", [])],
            duration_ms=int((time.perf_counter() - start) * 1000),
        )
        return {"answer": answer, "citations": citations, "graded_documents": used_chunks, "answer_model_audit": state.get("answer_model_audit") or {}}


class CitationChecker:
    async def __call__(self, state: AgentState) -> dict:
        start = time.perf_counter()
        db = SessionLocal()
        allowed = {item["chunk_id"] for item in state.get("graded_documents", [])}
        citations = [citation for citation in state.get("citations", []) if citation.get("chunk_id") in allowed]
        if state.get("route", "retrieve_both") in {"direct_answer", "clarify"}:
            citations = []
        await _set_run_state( state["run_id"], "running", current_node="citation_checker")
        await _trace(
                        state["run_id"],
            "citation_checker",
            input_summary=f"{len(state.get('citations', []))} citations",
            output_summary=f"{len(citations)} verified citations",
            document_ids=[item["chunk_id"] for item in citations],
            duration_ms=int((time.perf_counter() - start) * 1000),
        )
        return {"citations": citations}


class CitationVerifier:
    async def __call__(self, state: AgentState) -> dict:
        start = time.perf_counter()
        db = SessionLocal()
        settings = get_settings()
        citations = state.get("citations", [])
        answer = state.get("answer", "")
        graded = state.get("graded_documents", [])
        unverified: list[int] = []

        if settings.enable_agentic_reflection and answer and citations and graded and not is_degraded_mode():
            try:
                result = await ChatProvider().verify_citations(answer, citations, graded)
                unverified = [int(i) for i in result.get("unverified_indices", []) if isinstance(i, (int, str))]
            except Exception:
                pass

        await _set_run_state( state["run_id"], "running", current_node="citation_verifier")
        await _trace(
                        state["run_id"],
            "citation_verifier",
            input_summary=f"{len(citations)} citations",
            output_summary=f"{len(unverified)} unverified",
            duration_ms=int((time.perf_counter() - start) * 1000),
        )
        return {"unverified_citations": unverified}


class Reflection:
    async def __call__(self, state: AgentState) -> dict:
        start = time.perf_counter()
        db = SessionLocal()
        settings = get_settings()
        reflection_result = {"has_issue": False, "issue_type": "none", "suggestion": ""}

        if settings.enable_agentic_reflection and state.get("answer") and state.get("graded_documents") and not is_degraded_mode():
            try:
                result = await ChatProvider().reflect_answer(
                    state["question"],
                    state["answer"],
                    state["graded_documents"],
                )
                reflection_result = result
            except Exception:
                pass

        await _set_run_state( state["run_id"], "running", current_node="reflection")
        await _trace(
                        state["run_id"],
            "reflection",
            input_summary=state.get("answer", "")[:200],
            output_summary=f"has_issue={reflection_result.get('has_issue')} type={reflection_result.get('issue_type')}",
            duration_ms=int((time.perf_counter() - start) * 1000),
        )
        return {"reflection_result": reflection_result}


class AnswerCorrector:
    async def __call__(self, state: AgentState) -> dict:
        start = time.perf_counter()
        db = SessionLocal()
        reflection = state.get("reflection_result", {})
        issue_type = reflection.get("issue_type", "none")
        retry_count = state.get("retry_count", 0)

        correction: dict = {}
        if issue_type == "insufficient_coverage":
            correction = {
                "retry_count": retry_count + 1,
                "top_k": state["top_k"] * 2,
                "sub_queries": state.get("sub_queries") or [state.get("rewritten_question") or state["question"]],
            }
        elif issue_type == "hallucination":
            # Hallucination is usually a generation problem, not a document quality problem.
            # Removing documents often makes it worse (leads to no_contexts).
            # Strategy: re-retrieve with expanded top_k to get more diverse evidence,
            # but keep existing documents as fallback.
            docs = state.get("graded_documents", [])
            # If all docs have very low grade_score (< 0.2), treat as insufficient_coverage
            if docs and all(float(d.get("metadata", {}).get("scores", {}).get("grade_score", 0)) < 0.2 for d in docs):
                correction = {
                    "retry_count": retry_count + 1,
                    "top_k": state["top_k"] * 2,
                    "sub_queries": state.get("sub_queries") or [state.get("rewritten_question") or state["question"]],
                }
            else:
                correction = {
                    "retry_count": retry_count + 1,
                    "graded_documents": docs[: state["top_k"]],
                }
        elif issue_type == "contradiction":
            rewritten = f"{state.get('rewritten_question') or state['question']} (consistent explanation)"
            correction = {
                "retry_count": retry_count + 1,
                "rewritten_question": rewritten,
                "sub_queries": [rewritten],
            }
        else:
            correction = {"retry_count": retry_count}

        await _set_run_state( state["run_id"], "running", current_node="answer_corrector", retry_count=correction.get("retry_count", retry_count))
        await _trace(
                        state["run_id"],
            "answer_corrector",
            input_summary=f"issue={issue_type}",
            output_summary=f"retry={correction.get('retry_count', retry_count)}",
            duration_ms=int((time.perf_counter() - start) * 1000),
        )
        return correction


class SelfCheck:
    async def __call__(self, state: AgentState) -> dict:
        start = time.perf_counter()
        db = SessionLocal()
        if state.get("route", "retrieve_both") == "clarify":
            status = "needs_clarification"
        else:
            status = "completed"
        await _set_run_state( state["run_id"], status, current_node=None, answer=state.get("answer"))
        await _trace(
                        state["run_id"],
            "self_check",
            input_summary=state.get("route", "retrieve_both"),
            output_summary=status,
            duration_ms=int((time.perf_counter() - start) * 1000),
        )
        return {}


class Clarify:
    async def __call__(self, state: AgentState) -> dict:
        return await AnswerGenerator().__call__(state)


def split_multi_hop_query(question: str) -> list[str]:
    from app.services.chinese_text import split_multi_hop_query as _cn_split

    return _cn_split(question)


def expand_route_filters(route: AgentRoute, filters: SearchFilters) -> list[SearchFilters]:
    if filters.source_type:
        return [filters]
    if route == "retrieve_exercises":
        next_filters = filters.model_copy()
        next_filters.source_type = "notebook"
        return [next_filters]
    if route == "retrieve_notes":
        return [filters.model_copy(update={"source_type": source_type}) for source_type in sorted(NOTE_SOURCE_TYPES)]
    return [filters]


def route_after_perception(state: AgentState) -> str:
    """Route after perception: check if question needs clarification or is a greeting."""
    question = state["question"].strip().lower()
    tokens = _terms(question)
    is_greeting = question in {
        "hello", "hi", "hey", "who are you", "help", "what can you do",
        "你好", "您好", "嗨", "你是谁", "帮助", "你能做什么",
    } or bool(tokens.intersection({"hello", "hey"}) or (tokens == {"hi"}))
    needs_clarification = len(tokens) == 0 or question in {"it", "this", "that", "explain it", "why", "这个", "那个", "解释一下", "为什么"}
    if is_greeting:
        return "direct_answer"
    if needs_clarification:
        return "clarify"
    return "retrieval_planner"


def route_after_router(state: AgentState) -> str:
    route = state.get("route", "retrieve_both")
    if route in {"direct_answer", "clarify"}:
        return "answer_generator"
    return "query_rewriter"


def route_after_retrieval_decision(state: AgentState) -> str:
    if state.get("skip_retrieval"):
        return "answer_generator"
    return "base_retrieval"


def route_after_grader(state: AgentState) -> str:
    # Always route to evidence evaluator for pre-generation sufficiency check.
    # Evidence evaluator handles retry logic based on quality, not just presence.
    return "evidence_evaluator"


def route_after_reflection(state: AgentState) -> str:
    reflection = state.get("reflection_result", {})
    retry_count = state.get("retry_count", 0)
    settings = get_settings()
    enable_agentic_reflection = _runtime_bool_env("ENABLE_AGENTIC_REFLECTION", settings.enable_agentic_reflection)
    enable_post_generation_reflection = _runtime_bool_env("ENABLE_POST_GENERATION_REFLECTION", settings.enable_post_generation_reflection)
    reflection_max_retries = _runtime_int_env("REFLECTION_MAX_RETRIES", settings.reflection_max_retries)
    if not enable_agentic_reflection:
        return "self_check"
    if not enable_post_generation_reflection:
        return "self_check"
    if reflection.get("has_issue") and retry_count < reflection_max_retries:
        return "answer_corrector"
    return "self_check"


def route_after_evidence_evaluator(state: AgentState) -> str:
    evaluation = state.get("evidence_evaluation", {})
    if evaluation.get("sufficient"):
        return "context_synthesizer"
    if state.get("retry_count", 0) < 2:
        return "retrieval_planner"
    return "context_synthesizer"


def route_after_corrector(state: AgentState) -> str:
    reflection = state.get("reflection_result", {})
    issue_type = reflection.get("issue_type", "none")
    # For hallucination, we already have updated graded_documents; regenerate answer directly.
    # For insufficient_coverage or contradiction, we need to re-retrieve with updated params.
    if issue_type == "hallucination":
        return "context_synthesizer"
    return "base_retrieval"


def build_agent_graph():
    workflow = StateGraph(AgentState)
    # New architecture nodes
    workflow.add_node("perception", Perception())
    workflow.add_node("retrieval_planner", RetrievalPlanner())
    workflow.add_node("base_retrieval", BaseRetrieval())
    workflow.add_node("evidence_anchor_selector", EvidenceAnchorSelector())
    workflow.add_node("evidence_chain_planner", EvidenceChainPlanner())
    workflow.add_node("controlled_graph_enhancer", ControlledGraphEnhancer())
    workflow.add_node("evidence_assembler", EvidenceAssembler())
    # Legacy nodes (kept for compatibility / phased migration)
    workflow.add_node("query_analyzer", QueryAnalyzer())
    workflow.add_node("router", Router())
    workflow.add_node("query_rewriter", QueryRewriter())
    workflow.add_node("retrieval_decision", RetrievalDecision())
    workflow.add_node("document_grader", DocumentGrader())
    workflow.add_node("evidence_evaluator", EvidenceEvaluator())
    workflow.add_node("retry_planner", RetryPlanner())
    workflow.add_node("context_synthesizer", ContextSynthesizer())
    workflow.add_node("answer_generator", AnswerGenerator())
    workflow.add_node("citation_checker", CitationChecker())
    workflow.add_node("citation_verifier", CitationVerifier())
    workflow.add_node("reflection", Reflection())
    workflow.add_node("answer_corrector", AnswerCorrector())
    workflow.add_node("self_check", SelfCheck())

    # New architecture wiring: Perception -> [direct_answer/clarify?] -> Planning -> Retrieval -> Grader -> Context -> Generation
    workflow.add_edge(START, "perception")
    workflow.add_conditional_edges(
        "perception",
        route_after_perception,
        {"direct_answer": "answer_generator", "clarify": "answer_generator", "retrieval_planner": "retrieval_planner"},
    )
    workflow.add_edge("retrieval_planner", "base_retrieval")
    workflow.add_edge("base_retrieval", "evidence_anchor_selector")
    workflow.add_edge("evidence_anchor_selector", "evidence_chain_planner")
    workflow.add_edge("evidence_chain_planner", "controlled_graph_enhancer")
    workflow.add_edge("controlled_graph_enhancer", "evidence_assembler")
    workflow.add_edge("evidence_assembler", "document_grader")
    workflow.add_conditional_edges(
        "document_grader",
        route_after_grader,
        {"evidence_evaluator": "evidence_evaluator"},
    )
    workflow.add_conditional_edges(
        "evidence_evaluator",
        route_after_evidence_evaluator,
        {"retrieval_planner": "retrieval_planner", "context_synthesizer": "context_synthesizer"},
    )
    workflow.add_edge("retry_planner", "base_retrieval")
    workflow.add_edge("context_synthesizer", "answer_generator")
    workflow.add_edge("answer_generator", "citation_checker")
    workflow.add_edge("citation_checker", "citation_verifier")
    workflow.add_edge("citation_verifier", "reflection")
    workflow.add_conditional_edges("reflection", route_after_reflection, {"answer_corrector": "answer_corrector", "self_check": "self_check"})
    workflow.add_conditional_edges(
        "answer_corrector",
        route_after_corrector,
        {"context_synthesizer": "context_synthesizer", "base_retrieval": "base_retrieval"},
    )
    workflow.add_edge("self_check", END)
    return workflow.compile()


AGENT_GRAPH = build_agent_graph()


def create_or_get_session(db: Session, course_id: str, session_id: str | None, question: str) -> QASession:
    session = db.get(QASession, session_id) if session_id else None
    if session is not None and session.course_id != course_id:
        session = None
    if session is None:
        session = QASession(course_id=course_id, title=_summarize(question, 80), transcript=[])
        db.add(session)
        db.commit()
        db.refresh(session)
    return session


def append_session_turn(db: Session, session_id: str, question: str, answer: str, run_id: str, citations: list[dict]) -> None:
    session = db.get(QASession, session_id)
    if session is None:
        return
    transcript = list(session.transcript or [])
    transcript.append({"role": "user", "content": question, "run_id": run_id})
    transcript.append({"role": "assistant", "content": answer, "run_id": run_id, "citations": citations})
    session.transcript = transcript
    session.last_question = question
    session.last_answer = answer
    session.updated_at = datetime.utcnow()
    db.commit()


def run_to_task_status(run: AgentRun) -> dict:
    return {
        "run_id": run.id,
        "state": run.status,
        "current_node": run.current_node,
        "retry_count": run.retry_count,
        "route": run.route,
        "error": run.error_message,
        "started_at": run.started_at,
        "completed_at": run.completed_at,
    }


def create_agent_run_context(db: Session, request: AgentRequest) -> tuple[QASession, AgentRun, AgentState]:
    course = resolve_course(db, request.course_id)
    session = create_or_get_session(db, course.id, request.session_id, request.question)
    run = AgentRun(
        course_id=course.id,
        session_id=session.id,
        question=request.question,
        status="queued",
        metadata_json={"top_k": request.top_k, "filters": request.filters.model_dump()},
    )
    db.add(run)
    db.commit()
    db.refresh(run)
    history = [item.model_dump() for item in request.history] or list(session.transcript or [])[-8:]
    initial: AgentState = {
        "run_id": run.id,
        "session_id": session.id,
        "course_id": course.id,
        "question": request.question,
        "history": history,
        "filters": request.filters,
        "top_k": request.top_k,
        "retry_count": 0,
    }
    return session, run, initial


async def execute_agent_run(db: Session, request: AgentRequest, session: QASession, run: AgentRun, initial: AgentState) -> dict:
    from app.db import _db_context_var, _active_sessions
    token = _db_context_var.set(db)
    sessions_token = _active_sessions.set([])
    strategy_profile = get_active_profile_record(db, run.course_id)
    try:
        with use_strategy_profile(strategy_profile.profile_json):
            final_state = await AGENT_GRAPH.ainvoke(initial)
        trace_events = db.scalars(select(AgentTraceEvent).where(AgentTraceEvent.run_id == run.id).order_by(AgentTraceEvent.created_at.asc())).all()
        used_chunks = final_state.get("graded_documents", [])
        answer = final_state.get("answer", "")
        citations = final_state.get("citations", [])
        append_session_turn(db, session.id, request.question, answer, run.id, citations)
        return {
            "run_id": run.id,
            "session_id": session.id,
            "answer": answer,
            "citations": citations,
            "used_chunks": used_chunks,
            "route": final_state.get("route") or "retrieve_both",
            "trace": [trace_event_to_payload(event) for event in trace_events],
            "degraded_mode": is_degraded_mode(),
            "answer_model_audit": final_state.get("answer_model_audit") or {},
        }
    except Exception as exc:
        _set_run_state_sync(db, run.id, "failed", current_node=None, error=str(exc))
        _trace_sync(db, run.id, "error", status="failed", output_summary=str(exc), error=str(exc))
        raise
    finally:
        _db_context_var.reset(token)
        sessions = _active_sessions.get([])
        for sess in sessions:
            try:
                sess.close()
            except Exception:
                pass
        _active_sessions.reset(sessions_token)


async def run_agent(db: Session, request: AgentRequest) -> dict:
    session, run, initial = create_agent_run_context(db, request)
    return await execute_agent_run(db, request, session, run, initial)


async def stream_agent_events(db: Session, request: AgentRequest) -> AsyncGenerator[dict, None]:
    session, run, initial = create_agent_run_context(db, request)
    trace_queue = _subscribe_trace(run.id)
    task = asyncio.create_task(execute_agent_run(db, request, session, run, initial))
    response: dict | None = None
    yielded_trace_ids: set[str] = set()
    try:
        yield {"type": "meta", "run_id": run.id, "session_id": session.id}
        while not task.done():
            try:
                event = await asyncio.wait_for(trace_queue.get(), timeout=0.25)
            except asyncio.TimeoutError:
                continue
            if request.stream_trace:
                yielded_trace_ids.add(event["id"])
                yield {"type": "trace", "trace": event}
        while not trace_queue.empty():
            event = trace_queue.get_nowait()
            if request.stream_trace:
                yielded_trace_ids.add(event["id"])
                yield {"type": "trace", "trace": event}
        response = await task
    except Exception as exc:
        yield {"type": "error", "error": str(exc)}
        return
    finally:
        _unsubscribe_trace(run.id, trace_queue)
        if not task.done():
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass

    if request.stream_trace:
        # Defensive replay for any trace rows missed by the in-memory queue.
        for event in response["trace"]:
            if event["id"] in yielded_trace_ids:
                continue
            yield {"type": "trace", "trace": event}
    answer = response["answer"] or ""
    for start in range(0, len(answer), 8):
        yield {"type": "token", "token": answer[start : start + 8]}
        await asyncio.sleep(0.012)
    yield {"type": "citations", "citations": response["citations"], "degraded_mode": response["degraded_mode"]}
    yield {"type": "final", "response": response}
