from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from typing import Any, Iterable


TYPED_REPAIR_PROTOCOL_VERSION = "typed_repair_loop_v1"
CLAIM_ID_PROTOCOL_VERSION = "exact_answer_claim_id_v1"
CLAIM_GROUNDED_GATE_PROTOCOL_VERSION = "claim_level_grounded_gate_v1"
REPAIR_PROGRESS_PROTOCOL_VERSION = "repair_semantic_progress_v1"
REPAIR_GATE_SEMANTIC_PROTOCOL_VERSION = "repair_gate_semantic_card_v1"

REPAIR_ACTION_TYPES = (
    "repair_missing_citation",
    "repair_concept_gap",
    "repair_bridge_gap",
    "repair_structure_context",
)

REPAIR_EXECUTOR_MECHANISMS = {
    "repair_missing_citation": "current_package_claim_span_rebind_then_next_window_expansion_v1",
    "repair_concept_gap": "mid_rq_chunk_per_parent_expansion_v1",
    "repair_bridge_gap": "support_backed_bridge_seed_then_layered_traversal_v1",
    "repair_structure_context": "supported_chunk_structure_closure_v1",
}

_STRUCTURE_FAILURE_TYPES = {
    "formula_context_missing",
    "formula_table_context_missing",
    "structure_context_missing",
    "table_context_missing",
    "caption_context_missing",
}
_CONCEPT_FAILURE_TYPES = {
    "concept_gap",
    "missing_concept_support",
    "missing_required_facet",
}
_BRIDGE_FAILURE_TYPES = {
    "bridge_gap",
    "cross_document_gap",
    "missing_bridge_support",
}


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def canonical_repair_hash(protocol_version: str, payload: Any) -> str:
    return hashlib.sha256(
        f"{protocol_version}\n{_canonical_json(payload)}".encode("utf-8")
    ).hexdigest()


def exact_answer_hash(answer: str) -> str:
    return hashlib.sha256((answer or "").encode("utf-8")).hexdigest()


def split_answer_claims(answer: str) -> list[str]:
    """Split exact answer text into bounded factual-claim candidates.

    The function deliberately does not use an LLM.  Blank headings and list
    markers are removed, while the remaining text is kept verbatim enough for
    exact answer/claim binding and deterministic replay.
    """

    raw_parts = re.split(
        r"(?:\n+|(?<=[.!?])\s+|(?<=[\u3002\uff01\uff1f]))",
        answer or "",
    )
    claims: list[str] = []
    for item in raw_parts:
        cleaned = re.sub(r"^\s*(?:[-*#]+|\d+[.)])\s*", "", item).strip()
        if cleaned and re.search(r"[A-Za-z0-9\u3400-\u9fff]", cleaned):
            claims.append(cleaned)
    return claims or ([answer.strip()] if answer.strip() else [])


def claim_rows(answer: str) -> list[dict[str, Any]]:
    answer_digest = exact_answer_hash(answer)
    rows: list[dict[str, Any]] = []
    for index, text in enumerate(split_answer_claims(answer)):
        claim_id = canonical_repair_hash(
            CLAIM_ID_PROTOCOL_VERSION,
            {
                "answer_hash": answer_digest,
                "claim_index": index,
                "claim_text": text,
            },
        )
        rows.append(
            {
                "claim_id": claim_id,
                "claim_index": index,
                "claim_text": text,
                "answer_hash": answer_digest,
                "claim_id_protocol_version": CLAIM_ID_PROTOCOL_VERSION,
            }
        )
    return rows


def _valid_source_span(result: dict[str, Any]) -> bool:
    source_span = result.get("source_span") or {}
    if not isinstance(source_span, dict):
        return False
    char_span = source_span.get("char_span")
    return bool(
        result.get("chunk_id")
        and source_span.get("document_version_id")
        and isinstance(char_span, (list, tuple))
        and len(char_span) == 2
        and all(isinstance(value, int) for value in char_span)
        and int(char_span[0]) >= 0
        and int(char_span[1]) >= int(char_span[0])
        and source_span.get("raw_span_text_hash")
    )


def _result_claim_id(
    result: dict[str, Any],
    *,
    claims: list[dict[str, Any]],
) -> str | None:
    diagnostics = result.get("diagnostics") or {}
    explicit = result.get("claim_id") or diagnostics.get("claim_id")
    claim_index = result.get("claim_index")
    if claim_index is None:
        claim_index = diagnostics.get("claim_index")
    claim_text = str(result.get("claim_text") or "").strip()
    if explicit:
        explicit_matches = [
            row for row in claims if str(row["claim_id"]) == str(explicit)
        ]
        if len(explicit_matches) != 1:
            return None
        matched = explicit_matches[0]
        if claim_index is not None and claim_index != matched["claim_index"]:
            return None
        if claim_text and claim_text != matched["claim_text"]:
            return None
        return str(matched["claim_id"])
    if claim_index is not None:
        if not isinstance(claim_index, int) or not 0 <= claim_index < len(claims):
            return None
        matched = claims[claim_index]
        if claim_text and claim_text != matched["claim_text"]:
            return None
        return str(matched["claim_id"])
    exact_matches = [row for row in claims if row["claim_text"] == claim_text]
    if len(exact_matches) == 1:
        return str(exact_matches[0]["claim_id"])
    return None


def claim_grounding_gate(
    answer: str,
    verification_results: Iterable[dict[str, Any]],
    *,
    require_persistence_replay: bool = False,
) -> dict[str, Any]:
    """Return the conservative, claim-level grounded gate audit.

    One supported citation never promotes sibling claims.  Every exact claim
    needs at least one supported, provenance-valid raw span.  An unbound
    verification is retained in diagnostics but cannot support a claim.
    """

    claims = claim_rows(answer)
    expected_answer_hash = exact_answer_hash(answer)
    results = [dict(item) for item in verification_results]
    results_by_claim: dict[str, list[dict[str, Any]]] = defaultdict(list)
    unbound_results: list[dict[str, Any]] = []
    for result in results:
        diagnostics = result.get("diagnostics") or {}
        result_answer_hash = str(result.get("answer_hash") or "")
        diagnostic_answer_hash = str(diagnostics.get("answer_hash") or "")
        if (
            result_answer_hash != expected_answer_hash
            or (
                diagnostic_answer_hash
                and diagnostic_answer_hash != expected_answer_hash
            )
        ):
            unbound_results.append(result)
            continue
        claim_id = _result_claim_id(result, claims=claims)
        if claim_id is None:
            unbound_results.append(result)
            continue
        results_by_claim[claim_id].append(result)

    claim_audits: list[dict[str, Any]] = []
    supported_claim_ids: list[str] = []
    for claim in claims:
        candidates = results_by_claim.get(str(claim["claim_id"]), [])
        supported_candidates: list[dict[str, Any]] = []
        for result in candidates:
            diagnostics = result.get("diagnostics") or {}
            provenance_valid = bool(
                diagnostics.get("citation_provenance_valid")
            )
            persistence_valid = bool(
                diagnostics.get("citation_provenance_persistence_gate_passed")
            )
            if (
                result.get("verdict") == "supported"
                and provenance_valid
                and _valid_source_span(result)
                and (not require_persistence_replay or persistence_valid)
            ):
                supported_candidates.append(result)
        supported = bool(supported_candidates)
        if supported:
            supported_claim_ids.append(str(claim["claim_id"]))
        claim_audits.append(
            {
                **claim,
                "supported": supported,
                "candidate_verification_count": len(candidates),
                "supported_verification_count": len(supported_candidates),
                "supported_citation_indexes": sorted(
                    {
                        int(item.get("citation_index") or 0)
                        for item in supported_candidates
                    }
                ),
                "supported_chunk_ids": sorted(
                    {
                        str(item.get("chunk_id"))
                        for item in supported_candidates
                        if item.get("chunk_id")
                    }
                ),
                "failure_types": sorted(
                    {
                        str(item.get("failure_type") or "unsupported_claim")
                        for item in candidates
                        if item.get("verdict") != "supported"
                    }
                ),
            }
        )

    claim_count = len(claims)
    supported_count = len(supported_claim_ids)
    unsupported_claims = [row for row in claim_audits if not row["supported"]]
    audit = {
        "protocol_version": CLAIM_GROUNDED_GATE_PROTOCOL_VERSION,
        "answer_hash": expected_answer_hash,
        "claim_id_protocol_version": CLAIM_ID_PROTOCOL_VERSION,
        "claim_count": claim_count,
        "supported_claim_count": supported_count,
        "unsupported_claim_count": len(unsupported_claims),
        "claim_pass_rate": round(supported_count / max(claim_count, 1), 6),
        "all_claims_supported": bool(claim_count and supported_count == claim_count),
        "supported_claim_ids": supported_claim_ids,
        "unsupported_claim_ids": [row["claim_id"] for row in unsupported_claims],
        "claims": claim_audits,
        "unbound_verification_count": len(unbound_results),
        "unbound_verification_hash": canonical_repair_hash(
            CLAIM_GROUNDED_GATE_PROTOCOL_VERSION,
            unbound_results,
        ),
        "require_persistence_replay": require_persistence_replay,
    }
    audit["gate_hash"] = canonical_repair_hash(
        CLAIM_GROUNDED_GATE_PROTOCOL_VERSION,
        audit,
    )
    return audit


def repair_gate_semantic_card(gate: dict[str, Any]) -> dict[str, Any]:
    """Project a gate across pre/post persistence provenance enrichment."""

    card = {
        "protocol_version": REPAIR_GATE_SEMANTIC_PROTOCOL_VERSION,
        "answer_hash": str(gate.get("answer_hash") or ""),
        "claim_id_protocol_version": str(
            gate.get("claim_id_protocol_version") or ""
        ),
        "claim_count": int(gate.get("claim_count") or 0),
        "supported_claim_count": int(
            gate.get("supported_claim_count") or 0
        ),
        "unsupported_claim_count": int(
            gate.get("unsupported_claim_count") or 0
        ),
        "all_claims_supported": bool(gate.get("all_claims_supported")),
        "supported_claim_ids": list(gate.get("supported_claim_ids") or []),
        "unsupported_claim_ids": list(
            gate.get("unsupported_claim_ids") or []
        ),
        "claims": [
            {
                "claim_id": str(item.get("claim_id") or ""),
                "claim_index": item.get("claim_index"),
                "claim_text": str(item.get("claim_text") or ""),
                "answer_hash": str(item.get("answer_hash") or ""),
                "supported": bool(item.get("supported")),
            }
            for item in gate.get("claims") or []
            if isinstance(item, dict)
        ],
    }
    card["gate_semantic_hash"] = canonical_repair_hash(
        REPAIR_GATE_SEMANTIC_PROTOCOL_VERSION,
        card,
    )
    return card


def supported_partial_answer(answer: str, gate: dict[str, Any]) -> dict[str, Any]:
    supported_ids = set(gate.get("supported_claim_ids") or [])
    kept = [
        row
        for row in claim_rows(answer)
        if row["claim_id"] in supported_ids
    ]
    dropped = [
        row
        for row in claim_rows(answer)
        if row["claim_id"] not in supported_ids
    ]
    partial = "\n\n".join(str(row["claim_text"]) for row in kept)
    return {
        "answer": partial,
        "answer_hash": exact_answer_hash(partial),
        "kept_claim_ids": [row["claim_id"] for row in kept],
        "dropped_claim_ids": [row["claim_id"] for row in dropped],
        "dropped_claim_texts": [row["claim_text"] for row in dropped],
        "evidence_gap": {
            "kind": "unsupported_claims_removed",
            "dropped_claim_count": len(dropped),
            "dropped_claim_ids": [row["claim_id"] for row in dropped],
        },
    }


def canonical_failure_cards(
    *,
    answer: str,
    verification_results: Iterable[dict[str, Any]],
    repair_round_index: int,
    remaining_repair_budget: int,
    context_package_id: str,
    retrieval_trace_id: str | None,
    structure_closure_status: dict[str, Any] | None,
    covered_facets: Iterable[str],
    missing_evidence_roles: Iterable[str],
    prior_repair_action_output_hashes: Iterable[str],
) -> list[dict[str, Any]]:
    claims = claim_rows(answer)
    expected_answer_hash = exact_answer_hash(answer)
    verification_rows = [dict(item) for item in verification_results]
    gate = claim_grounding_gate(answer, verification_rows)
    supported_claim_ids = set(gate.get("supported_claim_ids") or [])
    canonical_covered_facets = sorted(
        {str(item) for item in covered_facets}
    )
    canonical_missing_evidence_roles = sorted(
        {str(item) for item in missing_evidence_roles}
    )
    canonical_prior_output_hashes = list(
        prior_repair_action_output_hashes
    )
    cards: list[dict[str, Any]] = []

    def append_card(
        card: dict[str, Any],
        source_span: dict[str, Any],
    ) -> None:
        card["failure_card_hash"] = canonical_repair_hash(
            TYPED_REPAIR_PROTOCOL_VERSION,
            card,
        )
        stable_source_span = {
            key: source_span.get(key)
            for key in (
                "document_id",
                "document_version_id",
                "chunk_id",
                "char_span",
                "raw_chunk_char_span",
                "page_range",
                "section_path",
                "structure_node_ids",
                "bbox",
                "source_checksum",
                "chunk_text_hash",
                "raw_span_text_hash",
                "raw_span_text_hash_protocol_version",
            )
            if source_span.get(key) is not None
        }
        card["semantic_failure_hash"] = canonical_repair_hash(
            TYPED_REPAIR_PROTOCOL_VERSION,
            {
                "answer_hash": card["answer_hash"],
                "claim_id": card["claim_id"],
                "claim_text": card["claim_text"],
                "verdict": card["verdict"],
                "failure_type": card["failure_type"],
                "chunk_id": card["chunk_id"],
                # Package, trace and verification row UUIDs are audit
                # addresses, not semantic evidence identity.  Excluding them
                # prevents identical repair inputs from bypassing no-repeat
                # solely because a new ContextPackage row was persisted.
                "source_span": stable_source_span,
                "structure_closure_status": card[
                    "structure_closure_status"
                ],
                "covered_facets": card["covered_facets"],
                "missing_evidence_roles": card[
                    "missing_evidence_roles"
                ],
            },
        )
        cards.append(card)

    for result in verification_rows:
        claim_id = _result_claim_id(result, claims=claims)
        result_answer_hash = str(result.get("answer_hash") or "")
        diagnostics = result.get("diagnostics") or {}
        diagnostic_answer_hash = str(diagnostics.get("answer_hash") or "")
        exact_answer_bound = bool(
            result_answer_hash == expected_answer_hash
            and (
                not diagnostic_answer_hash
                or diagnostic_answer_hash == expected_answer_hash
            )
        )
        if claim_id in supported_claim_ids and exact_answer_bound:
            continue
        source_span = result.get("source_span") or {}
        if not exact_answer_bound or claim_id is None:
            effective_verdict = "unsupported"
            effective_failure_type = "claim_binding_invalid"
        elif not bool(diagnostics.get("citation_provenance_valid")):
            effective_verdict = "unsupported"
            effective_failure_type = "citation_provenance_invalid"
        elif not _valid_source_span(result):
            effective_verdict = "unsupported"
            effective_failure_type = "source_span_invalid"
        else:
            effective_verdict = str(result.get("verdict") or "unsupported")
            effective_failure_type = str(
                result.get("failure_type") or "unsupported_claim"
            )
            if effective_verdict == "supported":
                effective_verdict = "unsupported"
                effective_failure_type = "claim_grounding_gate_rejected"
        card = {
            "repair_round_index": int(repair_round_index),
            "remaining_repair_budget": int(remaining_repair_budget),
            "answer_hash": expected_answer_hash,
            "context_package_id": context_package_id,
            "retrieval_trace_id": retrieval_trace_id,
            "claim_id": claim_id,
            "claim_text": str(result.get("claim_text") or ""),
            "claim_index": (result.get("diagnostics") or {}).get("claim_index"),
            "citation_index": int(result.get("citation_index") or 0),
            "verdict": effective_verdict,
            "failure_type": effective_failure_type,
            "chunk_id": result.get("chunk_id"),
            "source_span": source_span,
            "structure_closure_status": structure_closure_status or {},
            "covered_facets": canonical_covered_facets,
            "missing_evidence_roles": canonical_missing_evidence_roles,
            "prior_repair_action_output_hashes": canonical_prior_output_hashes,
        }
        append_card(card, source_span)

    represented_claim_ids = {
        str(item.get("claim_id"))
        for item in cards
        if item.get("claim_id")
    }
    for claim in gate.get("claims") or []:
        claim_id = str(claim.get("claim_id") or "")
        if bool(claim.get("supported")) or claim_id in represented_claim_ids:
            continue
        card = {
            "repair_round_index": int(repair_round_index),
            "remaining_repair_budget": int(remaining_repair_budget),
            "answer_hash": expected_answer_hash,
            "context_package_id": context_package_id,
            "retrieval_trace_id": retrieval_trace_id,
            "claim_id": claim_id,
            "claim_text": str(claim.get("claim_text") or ""),
            "claim_index": claim.get("claim_index"),
            "citation_index": 0,
            "verdict": "missing_citation",
            "failure_type": "citation_missing",
            "chunk_id": None,
            "source_span": {},
            "structure_closure_status": structure_closure_status or {},
            "covered_facets": canonical_covered_facets,
            "missing_evidence_roles": canonical_missing_evidence_roles,
            "prior_repair_action_output_hashes": canonical_prior_output_hashes,
        }
        append_card(card, {})
    cards.sort(
        key=lambda item: (
            str(item.get("claim_id") or ""),
            int(item.get("citation_index") or 0),
            str(item.get("failure_type") or ""),
        )
    )
    return cards


def repair_direction_candidates(
    failure_cards: Iterable[dict[str, Any]],
) -> list[str]:
    failures = {
        str(item.get("failure_type") or "unsupported_claim")
        for item in failure_cards
    }
    candidates: list[str] = []
    if failures.intersection(_STRUCTURE_FAILURE_TYPES):
        candidates.append("repair_structure_context")
    if failures.intersection(_CONCEPT_FAILURE_TYPES):
        candidates.append("repair_concept_gap")
    if failures.intersection(_BRIDGE_FAILURE_TYPES):
        candidates.append("repair_bridge_gap")
    # Missing-citation repair is the conservative fallback for every observed
    # failure.  Bridge, concept and structure actions are never invented when
    # the corresponding evidence-evaluator observation is absent.
    candidates.append("repair_missing_citation")
    return candidates


def select_repair_direction(
    failure_cards: Iterable[dict[str, Any]],
    *,
    attempted_input_hashes_by_action: dict[str, set[str]] | None = None,
    exhausted_action_types: set[str] | None = None,
) -> dict[str, Any] | None:
    cards = list(failure_cards)
    failure_set_hash = canonical_repair_hash(
        TYPED_REPAIR_PROTOCOL_VERSION,
        [item.get("semantic_failure_hash") for item in cards],
    )
    attempted = attempted_input_hashes_by_action or {}
    exhausted = {str(value) for value in (exhausted_action_types or set())}
    for action_type in repair_direction_candidates(cards):
        if action_type in exhausted:
            continue
        input_hash = canonical_repair_hash(
            TYPED_REPAIR_PROTOCOL_VERSION,
            {
                "action_type": action_type,
                "failure_set_hash": failure_set_hash,
                "failure_card_hashes": [
                    item.get("semantic_failure_hash") for item in cards
                ],
            },
        )
        if input_hash in attempted.get(action_type, set()):
            continue
        return {
            "action_type": action_type,
            "executor_mechanism": REPAIR_EXECUTOR_MECHANISMS[action_type],
            "input_hash": input_hash,
            "failure_set_hash": failure_set_hash,
        }
    return None


def repair_semantic_progress_signature(
    *,
    result_chunk_ids: Iterable[str],
    package_chunk_spans: Iterable[dict[str, Any]],
    covered_facets: Iterable[str],
    evidence_roles: Iterable[str],
    graph_path_ids: Iterable[str],
    supported_claim_ids: Iterable[str],
    unsupported_claim_ids: Iterable[str],
) -> dict[str, Any]:
    canonical_payload = {
        "result_chunk_ids": sorted({str(item) for item in result_chunk_ids}),
        "package_chunk_spans": sorted(
            [
                {
                    "chunk_id": item.get("chunk_id"),
                    "document_version_id": item.get("document_version_id"),
                    "char_span": list(item.get("char_span") or []),
                    "raw_span_text_hash": item.get("raw_span_text_hash"),
                }
                for item in package_chunk_spans
            ],
            key=lambda item: (
                str(item.get("chunk_id") or ""),
                _canonical_json(item.get("char_span") or []),
            ),
        ),
        "covered_facets": sorted({str(item) for item in covered_facets}),
        "evidence_roles": sorted({str(item) for item in evidence_roles}),
        "graph_path_ids": sorted({str(item) for item in graph_path_ids}),
        "supported_claim_ids": sorted(
            {str(item) for item in supported_claim_ids}
        ),
        "unsupported_claim_ids": sorted(
            {str(item) for item in unsupported_claim_ids}
        ),
    }
    return {
        "protocol_version": REPAIR_PROGRESS_PROTOCOL_VERSION,
        "payload": canonical_payload,
        "progress_hash": canonical_repair_hash(
            REPAIR_PROGRESS_PROTOCOL_VERSION,
            canonical_payload,
        ),
    }


def repair_made_progress(
    before: dict[str, Any],
    after: dict[str, Any],
) -> bool:
    before_payload = before.get("payload") or {}
    after_payload = after.get("payload") or {}
    if before.get("progress_hash") == after.get("progress_hash"):
        return False
    before_supported = set(before_payload.get("supported_claim_ids") or [])
    after_supported = set(after_payload.get("supported_claim_ids") or [])
    before_unsupported = set(before_payload.get("unsupported_claim_ids") or [])
    after_unsupported = set(after_payload.get("unsupported_claim_ids") or [])
    # Adding evidence for one claim is not progress if the same round loses a
    # claim that was already grounded.  This prevents repair oscillation from
    # being rewarded as forward movement.
    if not before_supported.issubset(after_supported):
        return False
    return bool(
        after_supported - before_supported
        or before_unsupported - after_unsupported
        or set(after_payload.get("result_chunk_ids") or [])
        - set(before_payload.get("result_chunk_ids") or [])
        or {
            _canonical_json(item)
            for item in after_payload.get("package_chunk_spans") or []
        }
        - {
            _canonical_json(item)
            for item in before_payload.get("package_chunk_spans") or []
        }
        or set(after_payload.get("covered_facets") or [])
        - set(before_payload.get("covered_facets") or [])
        or set(after_payload.get("evidence_roles") or [])
        - set(before_payload.get("evidence_roles") or [])
        or set(after_payload.get("graph_path_ids") or [])
        - set(before_payload.get("graph_path_ids") or [])
    )
