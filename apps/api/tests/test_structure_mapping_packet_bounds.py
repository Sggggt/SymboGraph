from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from types import SimpleNamespace

import pytest


def _packet(packet_id: str = "packet-a", *, excerpt_text: str = "grounded text") -> dict:
    from app.services.context_graph import (
        COARSE_CONCEPT_PACKET_PROTOCOL_VERSION,
        MID_CONCEPT_PACKET_GROUNDING_PROTOCOL_VERSION,
        STRUCTURE_MAPPING_FACT_SET_PROTOCOL_VERSION,
    )

    representative_ids = ["chunk-1", "chunk-2"]
    return {
        "packet_id": packet_id,
        "grounding_protocol_version": (
            MID_CONCEPT_PACKET_GROUNDING_PROTOCOL_VERSION
        ),
        "packet_protocol_version": COARSE_CONCEPT_PACKET_PROTOCOL_VERSION,
        "grounding_hash": "1" * 64,
        "packet_business_hash": "2" * 64,
        "candidate_labels": ["Grounded concept"],
        "representative_chunk_ids": representative_ids,
        "support_chunk_count": 3,
        "support_chunk_ids": ["chunk-1", "chunk-2", "chunk-3"],
        "canonical_membership_fact_hashes": ["a" * 64, "b" * 64, "c" * 64],
        "canonical_membership_facts_hash": "3" * 64,
        "support_chunk_edge_ids": ["edge-1", "edge-2"],
        "support_chunk_edge_count": 2,
        "support_chunk_edge_ids_hash": "4" * 64,
        "support_chunk_edge_business_facts_hash": "5" * 64,
        "support_edge_type_distribution": {"dense_semantic": 2},
        "membership_role_distribution": {"primary_member": 3},
        "structure_paths": [
            {"mapping_id": f"mapping-{index}", "path": f"p/{index}"}
            for index in range(3)
        ],
        "structure_path_count": 3,
        "structure_path_trace_sample_count": 3,
        "structure_path_trace_sample_limit": 64,
        "structure_paths_complete": True,
        "structure_paths_hash": "6" * 64,
        "structure_mapping_address_facts_hash": "7" * 64,
        "structure_mapping_chunk_scope_business_hash": "8" * 64,
        "structure_mapping_fact_set_protocol_version": (
            STRUCTURE_MAPPING_FACT_SET_PROTOCOL_VERSION
        ),
        "structure_mapping_coverage": 1.0,
        "source_spans": [
            {
                "chunk_id": chunk_id,
                "char_span": [
                    index * 1000,
                    index * 1000 + len(f"{excerpt_text} {index}"),
                ],
                "page_range": [1, 1],
                "section_path": ["Section"],
            }
            for index, chunk_id in enumerate(representative_ids)
        ],
        "source_spans_business_facts_hash": "9" * 64,
        "chunk_excerpts": [
            {
                "chunk_id": chunk_id,
                "section_path": "Section",
                "page_range": [1, 1],
                "text": f"{excerpt_text} {index}",
                "full_text_hash": str(index + 1) * 64,
                "full_text_length_chars": len(f"{excerpt_text} {index}"),
            }
            for index, chunk_id in enumerate(representative_ids)
        ],
        "child_mid_summaries": [
            {
                "label": "Child",
                "summary": "Grounded child summary",
                "definition": "Grounded child definition",
                "grounding_hash": "d" * 64,
            }
        ],
        "coarse_membership_cards": [
            {"mid_concept_id": "mid-1", "final_membership_score": 0.8}
        ],
        "support_mid_edge_ids": ["mid-edge-1"],
    }


def test_provider_projection_binds_complete_identity_counts_hashes_and_raw_spans():
    from app.services.context_graph import concept_provider_projection

    packet = _packet()
    projection = concept_provider_projection(packet, layer="mid")

    identity = projection["identity_card"]
    assert identity["structure_mapping_count"] == 3
    assert identity["structure_paths_hash"] == "6" * 64
    assert identity["structure_mapping_address_facts_hash"] == "7" * 64
    assert identity["structure_paths_complete"] is True
    assert identity["canonical_membership_fact_count"] == 3
    assert identity["support_chunk_edge_count"] == 2
    assert projection["full_packet_business_hash"] == "2" * 64
    assert projection["full_packet_address_hash"]
    assert "structure_paths" not in projection
    assert "support_chunk_ids" not in projection
    assert projection["provider_authority"] == {
        "definition_and_labels_only": True,
        "support_ids": False,
        "membership": False,
        "edge_identity": False,
        "structure_identity": False,
        "node_weight": False,
    }
    assert [
        row["representative_chunk_id"]
        for row in projection["representative_excerpts"]
    ] == packet["representative_chunk_ids"]
    for row in projection["representative_excerpts"]:
        assert row["source_span"]["char_span"]
        assert len(row["full_text_hash"]) == 64
        assert len(row["projected_text_hash"]) == 64
        assert row["text"]
        assert len(row["binding_hash"]) == 64


def test_provider_projection_tamper_fails_replay():
    from app.services.context_graph import (
        concept_provider_projection,
        validate_concept_provider_projection,
    )

    packet = _packet()
    projection = concept_provider_projection(packet, layer="mid")
    tampered = deepcopy(projection)
    tampered["identity_card"]["structure_mapping_count"] += 1

    with pytest.raises(RuntimeError, match="not replay-equivalent"):
        validate_concept_provider_projection(
            packet,
            tampered,
            layer="mid",
        )


def test_provider_projection_rejects_excerpt_order_not_declared_by_full_packet():
    from app.services.context_graph import concept_provider_projection

    packet = _packet()
    packet["chunk_excerpts"] = list(reversed(packet["chunk_excerpts"]))
    with pytest.raises(RuntimeError, match="declared deterministic"):
        concept_provider_projection(packet, layer="mid")


def test_provider_projection_rejects_tampered_excerpt_binding():
    from app.services.context_graph import (
        concept_provider_projection,
        validate_concept_provider_projection,
    )

    packet = _packet(excerpt_text="grounded source text")
    projection = concept_provider_projection(packet, layer="mid")
    tampered = deepcopy(projection)
    tampered["representative_excerpts"][0]["text"] = "tampered"

    with pytest.raises(RuntimeError, match="not replay-equivalent"):
        validate_concept_provider_projection(packet, tampered, layer="mid")


def test_six_worst_case_cjk_excerpts_fit_default_single_packet_preflight():
    from app.services import context_graph

    packet = _packet(excerpt_text="汉" * 480)
    packet["representative_chunk_ids"] = [f"chunk-{index}" for index in range(6)]
    packet["source_spans"] = [
        {
            "chunk_id": chunk_id,
            "char_span": [index * 1000, index * 1000 + 480],
            "page_range": [index + 1, index + 1],
            "section_path": ["Section"],
        }
        for index, chunk_id in enumerate(packet["representative_chunk_ids"])
    ]
    packet["chunk_excerpts"] = [
        {
            "chunk_id": chunk_id,
            "section_path": "Section",
            "page_range": [index + 1, index + 1],
            "text": "汉" * 480,
            "full_text_hash": f"{index + 1:064x}",
            "full_text_length_chars": 480,
        }
        for index, chunk_id in enumerate(packet["representative_chunk_ids"])
    ]
    projection = context_graph.concept_provider_projection(packet, layer="mid")
    _, card = context_graph.concept_provider_preflight(
        {"concept_packets": [projection]},
        packet_ids=[packet["packet_id"]],
        identity_count=packet["support_chunk_count"],
        max_tokens=2400,
    )

    assert card["decision"] == "allow"
    assert card["rough_tokens"] <= 2400
    assert all(
        len(row["text"])
        == context_graph.CONCEPT_PROVIDER_EXCERPT_TEXT_CHAR_LIMIT
        for row in projection["representative_excerpts"]
    )
    assert projection["sample_selection"]["representative_selected_count"] >= 1
    assert (
        projection["sample_selection"]["representative_selected_count"]
        + projection["sample_selection"]["representative_omitted_count"]
        == 6
    )


def test_six_worst_case_coarse_child_summaries_fit_single_packet_preflight():
    from app.services import context_graph

    packet = _packet()
    packet["representative_chunk_ids"] = []
    packet["source_spans"] = []
    packet["chunk_excerpts"] = []
    packet["child_mid_summaries"] = [
        {
            "label": f"Child {index}",
            "summary": "汉" * 480,
            "definition": "汉" * 480,
            "grounding_hash": f"{index + 1:064x}",
        }
        for index in range(6)
    ]
    projection = context_graph.concept_provider_projection(
        packet,
        layer="coarse",
    )
    _, card = context_graph.concept_provider_preflight(
        {"concept_packets": [projection]},
        packet_ids=[packet["packet_id"]],
        identity_count=len(packet["child_mid_summaries"]),
        max_tokens=2400,
    )

    assert card["decision"] == "allow"
    assert card["rough_tokens"] <= 2400
    assert projection["sample_selection"]["child_mid_selected_count"] >= 1
    assert (
        projection["sample_selection"]["child_mid_selected_count"]
        + projection["sample_selection"]["child_mid_omitted_count"]
        == 6
    )


def test_coarse_pack_skips_oversize_child_and_keeps_later_representative(
    monkeypatch,
):
    from app.services import context_graph

    packet = _packet()
    packet["child_mid_summaries"] = [
        {
            "label": "Oversize child",
            "summary": "child summary",
            "definition": "child definition",
            "grounding_hash": "a" * 64,
        }
    ]

    def fake_rough_token_count(serialized: str) -> int:
        wrapper = json.loads(serialized)
        projection = wrapper["concept_packets"][0]
        if projection.get("child_mid_summary_excerpts"):
            return 2500
        if projection.get("representative_excerpts"):
            return 2300
        return 1900

    monkeypatch.setattr(
        context_graph,
        "rough_token_count",
        fake_rough_token_count,
    )

    projection = context_graph.concept_provider_projection(
        packet,
        layer="coarse",
    )

    selection = projection["sample_selection"]
    assert selection["child_mid_candidate_count"] == 1
    assert selection["child_mid_selected_count"] == 0
    assert selection["child_mid_omitted_count"] == 1
    assert selection["representative_selected_count"] >= 1
    assert projection["child_mid_summary_excerpts"] == []
    assert projection["representative_excerpts"]
    assert selection["selected_evidence_bindings_hash"] != context_graph.stable_hash([])
    assert selection["omitted_evidence_bindings_hash"] != context_graph.stable_hash([])


def test_provider_projection_rejects_missing_representative_source_span():
    from app.services.context_graph import concept_provider_projection

    packet = _packet()
    packet["source_spans"] = packet["source_spans"][1:]
    with pytest.raises(RuntimeError, match="exactly one source span"):
        concept_provider_projection(packet, layer="mid")


@pytest.mark.asyncio
async def test_oversize_mid_projection_makes_zero_provider_calls(monkeypatch):
    from app.services import context_graph

    calls = 0

    class ProviderMustNotRun:
        def __init__(self, *args, **kwargs):
            nonlocal calls
            calls += 1

    monkeypatch.setattr(context_graph, "ChatProvider", ProviderMustNotRun)
    monkeypatch.setattr(
        context_graph,
        "get_settings",
        lambda: SimpleNamespace(
            mid_concept_extraction_max_tokens_per_batch=10,
            enable_model_fallback=False,
        ),
    )

    with pytest.raises(context_graph.ConceptPacketPreflightError) as exc_info:
        await context_graph.define_mid_concepts_batch([_packet()])
    assert calls == 0
    assert exc_info.value.diagnostics["network_call_count"] == 0
    assert exc_info.value.diagnostics["packet_ids"] == ["packet-a"]
    assert exc_info.value.diagnostics["identity_count"] == 3
    assert "grounded text" not in str(exc_info.value)


@pytest.mark.asyncio
async def test_mid_provider_receives_one_strict_json_projection_not_python_repr(
    monkeypatch,
):
    from app.services import context_graph

    prompts: list[str] = []

    class RecordingProvider:
        def __init__(self, *args, **kwargs):
            pass

        async def classify_json(self, *, system_prompt, user_prompt, fallback):
            prompts.append(user_prompt)
            decoded = json.loads(user_prompt)
            return {
                "concepts": [
                    {
                        **fallback["concepts"][index],
                        "canonical_label": "Grounded",
                    }
                    for index, _row in enumerate(decoded["concept_packets"])
                ]
            }

    monkeypatch.setattr(context_graph, "ChatProvider", RecordingProvider)
    monkeypatch.setattr(context_graph, "active_profile_json", lambda: {})
    monkeypatch.setattr(
        context_graph,
        "profile_prompt",
        lambda *_args, **_kwargs: "system",
    )
    monkeypatch.setattr(
        context_graph,
        "get_settings",
        lambda: SimpleNamespace(
            mid_concept_extraction_max_tokens_per_batch=100_000,
            enable_model_fallback=False,
        ),
    )

    result = await context_graph.define_mid_concepts_batch([_packet()])
    assert result[0]["packet_id"] == "packet-a"
    assert len(prompts) == 1
    decoded = json.loads(prompts[0])
    assert decoded["concept_packets"][0]["protocol_version"] == (
        context_graph.CONCEPT_PROVIDER_PROJECTION_PROTOCOL_VERSION
    )
    assert "structure_paths" not in decoded["concept_packets"][0]
    assert prompts[0] == context_graph._strict_json(decoded)


@pytest.mark.asyncio
async def test_mid_provider_retries_incomplete_completion_once_with_same_projection(
    monkeypatch,
):
    from app.services import context_graph
    from app.services.error_sanitizer import ExternalServiceError

    prompts: list[tuple[str, str]] = []

    class RepairingProvider:
        def __init__(self, *args, **kwargs):
            pass

        async def classify_json(self, *, system_prompt, user_prompt, fallback):
            prompts.append((system_prompt, user_prompt))
            if len(prompts) == 1:
                raise ExternalServiceError(
                    service="anthropic",
                    phase="sdk_messages_completion",
                    error_code="incomplete_max_tokens",
                    retryable=False,
                )
            return fallback

    monkeypatch.setattr(context_graph, "ChatProvider", RepairingProvider)
    monkeypatch.setattr(context_graph, "active_profile_json", lambda: {})
    monkeypatch.setattr(
        context_graph,
        "profile_prompt",
        lambda *_args, **_kwargs: "editable definition prompt",
    )
    monkeypatch.setattr(
        context_graph,
        "get_settings",
        lambda: SimpleNamespace(
            mid_concept_extraction_max_tokens_per_batch=100_000,
            enable_model_fallback=False,
        ),
    )

    result = await context_graph.define_mid_concepts_batch([_packet()])

    assert len(prompts) == 2
    assert prompts[0][1] == prompts[1][1]
    assert context_graph.CONCEPT_PROVIDER_SCHEMA_REPAIR_SYSTEM_SUFFIX not in (
        prompts[0][0]
    )
    assert context_graph.CONCEPT_PROVIDER_SCHEMA_REPAIR_SYSTEM_SUFFIX in (
        prompts[1][0]
    )
    audit = result[0]["_provider_output_audit"]
    assert {
        key: audit[key]
        for key in (
            "protocol_version",
            "attempt_count",
            "repair_used",
            "first_attempt_failure",
            "max_attempts",
            "provider_response_persisted",
        )
    } == {
        "protocol_version": (
            context_graph.CONCEPT_PROVIDER_SCHEMA_REPAIR_PROTOCOL_VERSION
        ),
        "attempt_count": 2,
        "repair_used": True,
        "first_attempt_failure": "incomplete_max_tokens",
        "max_attempts": 2,
        "provider_response_persisted": False,
    }
    assert audit["provider_called"] is True
    assert audit["provider_request_count"] == 2
    assert audit["semantic_reuse_hit"] is False
    assert audit["effective_system_prompt_sha256"] == hashlib.sha256(
        (
            "editable definition prompt\n\n"
            + context_graph.MID_CONCEPT_PROVIDER_OUTPUT_CONTRACT
        ).encode("utf-8")
    ).hexdigest()
    assert audit["provider_call"]["provider_response_persisted"] is False


@pytest.mark.asyncio
async def test_mid_provider_schema_retry_targets_rejected_field_without_content(
    monkeypatch,
):
    from app.services import context_graph

    prompts: list[tuple[str, str]] = []
    private_rejected_text = "private-rejected-summary-sentinel"

    class FieldRepairingProvider:
        def __init__(self, *args, **kwargs):
            pass

        async def classify_json(self, *, system_prompt, user_prompt, fallback):
            prompts.append((system_prompt, user_prompt))
            if len(prompts) == 1:
                oversized = dict(fallback["concepts"][0])
                oversized["summary"] = private_rejected_text * 20
                oversized["definition"] = private_rejected_text * 40
                return {"concepts": [oversized]}
            return fallback

    monkeypatch.setattr(context_graph, "ChatProvider", FieldRepairingProvider)
    monkeypatch.setattr(context_graph, "active_profile_json", lambda: {})
    monkeypatch.setattr(
        context_graph,
        "profile_prompt",
        lambda *_args, **_kwargs: "editable definition prompt",
    )
    monkeypatch.setattr(
        context_graph,
        "get_settings",
        lambda: SimpleNamespace(
            mid_concept_extraction_max_tokens_per_batch=100_000,
            enable_model_fallback=False,
        ),
    )

    result = await context_graph.define_mid_concepts_batch([_packet()])

    assert len(prompts) == 2
    assert prompts[0][1] == prompts[1][1]
    assert private_rejected_text not in prompts[1][0]
    assert '"error_code":"string_length_exceeded"' in prompts[1][0]
    assert '"field_path":"summary"' in prompts[1][0]
    assert "exact field limits" in prompts[1][0]
    assert "summary <= 160" in prompts[1][0]
    assert "definition <= 320" in prompts[1][0]
    assert "Use at most 4 ids" in prompts[1][0]
    audit = result[0]["_provider_output_audit"]
    assert {
        key: audit[key]
        for key in (
            "protocol_version",
            "attempt_count",
            "repair_used",
            "first_attempt_failure",
            "max_attempts",
            "provider_response_persisted",
        )
    } == {
        "protocol_version": (
            context_graph.CONCEPT_PROVIDER_SCHEMA_REPAIR_PROTOCOL_VERSION
        ),
        "attempt_count": 2,
        "repair_used": True,
        "first_attempt_failure": "output_schema_rejected",
        "max_attempts": 2,
        "provider_response_persisted": False,
    }
    assert audit["provider_called"] is True
    assert audit["provider_request_count"] == 2
    assert audit["semantic_reuse_hit"] is False
    assert audit["provider_call"]["provider_response_persisted"] is False


@pytest.mark.asyncio
async def test_mid_provider_final_schema_failure_records_two_attempts_without_content(
    monkeypatch,
):
    from app.services import context_graph

    calls = 0
    private_rejected_text = "private-final-schema-sentinel"

    class NeverRepairingProvider:
        def __init__(self, *args, **kwargs):
            pass

        async def classify_json(self, *, system_prompt, user_prompt, fallback):
            nonlocal calls
            calls += 1
            oversized = dict(fallback["concepts"][0])
            oversized["summary"] = private_rejected_text * 20
            return {"concepts": [oversized]}

    monkeypatch.setattr(context_graph, "ChatProvider", NeverRepairingProvider)
    monkeypatch.setattr(context_graph, "active_profile_json", lambda: {})
    monkeypatch.setattr(
        context_graph,
        "profile_prompt",
        lambda *_args, **_kwargs: "editable definition prompt",
    )
    monkeypatch.setattr(
        context_graph,
        "get_settings",
        lambda: SimpleNamespace(
            mid_concept_extraction_max_tokens_per_batch=100_000,
            enable_model_fallback=False,
        ),
    )

    with pytest.raises(
        context_graph.ConceptProviderOutputSchemaError
    ) as exc_info:
        await context_graph.define_mid_concepts_batch([_packet()])

    assert calls == 2
    failure_card = context_graph.concept_provider_output_failure_card(
        exc_info.value
    )
    assert failure_card["attempt_count"] == 2
    assert failure_card["max_attempts"] == 2
    assert failure_card["first_attempt_failure"] == "output_schema_rejected"
    assert private_rejected_text not in json.dumps(failure_card)


def test_concept_semantic_input_binds_exact_effective_system_prompt_bytes(
    monkeypatch,
):
    from app.services import context_graph

    monkeypatch.setattr(
        context_graph,
        "get_settings",
        lambda: SimpleNamespace(
            graph_api_protocol="anthropic",
            graph_model="unit-graph-model",
            graph_base_url="https://provider.invalid/ai",
            graph_resolve_ip=None,
            model_request_timeout_seconds=240,
        ),
    )
    first_profile = {
        "prompt_pack": {"mid_concept_definition_system": "Stable prompt."}
    }
    second_profile = {
        "prompt_pack": {"mid_concept_definition_system": "Stable prompt!"}
    }

    first = context_graph.concept_definition_semantic_input_card(
        _packet(),
        layer="mid",
        profile_json=first_profile,
    )
    second = context_graph.concept_definition_semantic_input_card(
        _packet(),
        layer="mid",
        profile_json=second_profile,
    )

    assert first["full_packet_business_hash"] == second[
        "full_packet_business_hash"
    ]
    assert first["effective_system_prompt_sha256"] != second[
        "effective_system_prompt_sha256"
    ]
    assert first["semantic_input_hash"] != second["semantic_input_hash"]
    assert first["effective_system_prompt_sha256"] == hashlib.sha256(
        (
            "Stable prompt.\n\n"
            + context_graph.MID_CONCEPT_PROVIDER_OUTPUT_CONTRACT
        ).encode("utf-8")
    ).hexdigest()


def test_mid_semantic_reuse_uses_definition_edge_hash_without_weakening_graph_identity(
    monkeypatch,
):
    from app.services import context_graph

    monkeypatch.setattr(
        context_graph,
        "get_settings",
        lambda: SimpleNamespace(
            graph_api_protocol="anthropic",
            graph_model="unit-graph-model",
            graph_base_url="https://provider.invalid/ai",
            graph_resolve_ip=None,
            model_request_timeout_seconds=240,
        ),
    )
    profile = {
        "prompt_pack": {"mid_concept_definition_system": "Stable prompt."}
    }
    first_packet = _packet()
    first_packet.update(
        {
            "definition_support_chunk_edge_business_facts_hash": (
                "d" * 64
            ),
            "definition_support_chunk_edge_facts_protocol_version": (
                context_graph.CONCEPT_DEFINITION_BOTTOM_EDGE_FACT_PROTOCOL_VERSION
            ),
            "definition_membership_facts_hash": "e" * 64,
            "definition_membership_facts_protocol_version": (
                context_graph.CONCEPT_DEFINITION_RQ_MEMBERSHIP_FACT_PROTOCOL_VERSION
            ),
        }
    )
    second_packet = deepcopy(first_packet)
    second_packet["support_chunk_edge_business_facts_hash"] = "f" * 64

    first_projection = context_graph.concept_provider_projection(
        first_packet,
        layer="mid",
    )
    second_projection = context_graph.concept_provider_projection(
        second_packet,
        layer="mid",
    )
    first_card = context_graph.concept_definition_semantic_input_card(
        first_packet,
        layer="mid",
        profile_json=profile,
    )
    second_card = context_graph.concept_definition_semantic_input_card(
        second_packet,
        layer="mid",
        profile_json=profile,
    )

    assert first_projection["identity_card"][
        "support_chunk_edge_business_facts_hash"
    ] == "5" * 64
    assert second_projection["identity_card"][
        "support_chunk_edge_business_facts_hash"
    ] == "f" * 64
    assert "identity_card_hash" not in first_projection
    assert "business_identity_card_hash" not in first_projection
    assert context_graph.stable_hash(
        first_projection["identity_card"]
    ) != context_graph.stable_hash(second_projection["identity_card"])
    assert first_projection["business_identity_card"][
        "support_chunk_edge_business_facts_hash"
    ] == "d" * 64
    assert second_projection["business_identity_card"][
        "support_chunk_edge_business_facts_hash"
    ] == "d" * 64
    assert (
        "definition_support_chunk_edge_facts_protocol_version"
        not in first_projection["business_identity_card"]
    )
    assert (
        "definition_membership_facts_protocol_version"
        not in first_projection["business_identity_card"]
    )
    assert first_card["semantic_input_hash"] == second_card[
        "semantic_input_hash"
    ]


def test_reused_definition_reconstructs_provider_display_slots_exactly():
    from app.services import context_graph

    mid_display = [
        "RQ label",
        "term-1",
        "term-2",
        "term-3",
        "term-4",
        "term-5",
        "term-6",
        "term-7",
        "term-8",
        "alias-a",
        "alias-b",
    ]
    mid = SimpleNamespace(
        internal_state_json={
            "provider_definition_explanation": {"source": "provider"},
            "why_this_concept_exists": "Grounded reason.",
        },
        llm_audit_json={"packet": {"packet_id": "packet-a"}},
        canonical_label="Canonical label",
        aliases_json=["alias-a", "alias-b"],
        display_terms_json=mid_display,
        summary="Grounded summary.",
        definition="Grounded definition.",
        scope_note="Grounded scope.",
        inclusion_criteria_json=["Included."],
        exclusion_criteria_json=["Excluded."],
        confidence=0.9,
    )
    mid_output = context_graph._reconstruct_mid_semantic_output(mid)
    mid_replayed = list(
        dict.fromkeys(
            [
                "RQ label",
                *mid_output["display_terms"],
                *mid_output["aliases"],
            ]
        )
    )[:12]
    assert mid_replayed == mid_display
    assert mid_output["display_terms"] == mid_display[1:9]

    coarse_display = [
        "Coarse label",
        "term-1",
        "term-2",
        "term-3",
        "term-4",
        "term-5",
        "term-6",
        "term-7",
        "term-8",
        "alias-a",
        "alias-b",
    ]
    coarse = SimpleNamespace(
        internal_state_json={
            "source": "provider",
            "membership_model_call_count": 0,
        },
        canonical_label="Coarse label",
        aliases_json=["alias-a", "alias-b"],
        display_terms_json=coarse_display,
        summary="Grounded summary.",
        definition="Grounded definition.",
        scope_note="Grounded scope.",
        inclusion_criteria_json=["Included."],
        exclusion_criteria_json=["Excluded."],
        confidence=0.9,
    )
    coarse_output = context_graph._reconstruct_coarse_semantic_output(coarse)
    coarse_replayed = list(
        dict.fromkeys(
            [
                coarse_output["coarse_label"],
                *coarse_output["display_terms"],
                *coarse_output["aliases"],
            ]
        )
    )
    assert coarse_replayed == coarse_display
    assert coarse_output["display_terms"] == coarse_display[1:9]


def test_coarse_definition_semantic_key_ignores_opaque_lineage_digest_only(
    monkeypatch,
):
    from app.services import context_graph

    monkeypatch.setattr(
        context_graph,
        "get_settings",
        lambda: SimpleNamespace(
            graph_api_protocol="anthropic",
            graph_model="unit-graph-model",
            graph_base_url="https://provider.invalid/ai",
            graph_resolve_ip=None,
            model_request_timeout_seconds=240,
        ),
    )
    profile = {
        "prompt_pack": {
            "coarse_concept_definition_system": "Stable coarse prompt."
        }
    }
    first_packet = _packet()
    first_packet.update(
        {
            "packet_protocol_version": (
                context_graph.COARSE_CONCEPT_PACKET_PROTOCOL_VERSION
            ),
            "packet_business_hash": "lineage-hash-a",
            "child_mid_summaries": [
                {
                    "label": "Child concept",
                    "summary": "Exact child summary.",
                    "definition": "Exact child definition.",
                    "grounding_hash": "opaque-grounding-a",
                }
            ],
            "packet_business_card": {
                "coarse_membership_facts": [],
            },
            "support_mid_edge_business_facts": [],
            "definition_support_chunk_edge_business_facts_hash": (
                "e" * 64
            ),
            "definition_support_chunk_edge_facts_protocol_version": (
                context_graph.CONCEPT_DEFINITION_BOTTOM_EDGE_FACT_PROTOCOL_VERSION
            ),
        }
    )
    second_packet = deepcopy(first_packet)
    second_packet["packet_business_hash"] = "lineage-hash-b"
    second_packet["support_chunk_edge_business_facts_hash"] = "f" * 64
    second_packet["child_mid_summaries"][0][
        "grounding_hash"
    ] = "opaque-grounding-b"

    first = context_graph.concept_definition_semantic_input_card(
        first_packet,
        layer="coarse",
        profile_json=profile,
    )
    second = context_graph.concept_definition_semantic_input_card(
        second_packet,
        layer="coarse",
        profile_json=profile,
    )

    assert first["full_packet_business_hash"] != second[
        "full_packet_business_hash"
    ]
    assert first["business_identity_card"][
        "support_chunk_edge_business_facts_hash"
    ] == "e" * 64
    assert second["business_identity_card"][
        "support_chunk_edge_business_facts_hash"
    ] == "e" * 64
    assert (
        "definition_support_chunk_edge_facts_protocol_version"
        not in first["business_identity_card"]
    )
    assert first["semantic_input_hash"] == second["semantic_input_hash"]

    def repacked_rough_token_count(serialized: str) -> int:
        wrapper = json.loads(serialized)
        projection = wrapper["concept_packets"][0]
        if projection.get("child_mid_summary_excerpts"):
            return 2500
        if projection.get("representative_excerpts"):
            return 2300
        return 1900

    monkeypatch.setattr(
        context_graph,
        "rough_token_count",
        repacked_rough_token_count,
    )
    repacked = context_graph.concept_definition_semantic_input_card(
        first_packet,
        layer="coarse",
        profile_json=profile,
    )
    assert repacked["semantic_input_hash"] == first["semantic_input_hash"]

    changed_text_packet = deepcopy(second_packet)
    changed_text_packet["child_mid_summaries"][0][
        "summary"
    ] = "Changed child summary."
    changed_text = context_graph.concept_definition_semantic_input_card(
        changed_text_packet,
        layer="coarse",
        profile_json=profile,
    )
    assert changed_text["semantic_input_hash"] != first[
        "semantic_input_hash"
    ]


@pytest.mark.asyncio
async def test_mid_semantic_reuse_hit_skips_provider_and_keeps_closed_audit(
    monkeypatch,
):
    from app.services import context_graph

    monkeypatch.setattr(
        context_graph,
        "get_settings",
        lambda: SimpleNamespace(
            graph_api_protocol="anthropic",
            graph_model="unit-graph-model",
            graph_base_url="https://provider.invalid/ai",
            graph_resolve_ip=None,
            model_request_timeout_seconds=240,
        ),
    )
    profile = {
        "prompt_pack": {"mid_concept_definition_system": "Stable prompt."}
    }
    packet = _packet()
    card = context_graph.concept_definition_semantic_input_card(
        packet,
        layer="mid",
        profile_json=profile,
    )
    output = context_graph.validate_mid_concept_provider_output(
        {"concepts": [context_graph.mid_concept_fallback(packet)]},
        [packet],
    )[0]
    entry = context_graph.ConceptDefinitionSemanticReuseEntry(
        semantic_input_hash=card["semantic_input_hash"],
        output=output,
        source_state_id="source-state",
        source_state_hash="source-state-hash",
        source_concept_id="source-concept",
        source_provider_output_hash="source-output-hash",
        source_prompt_evidence="exact_effective_system_prompt_sha256",
    )
    index = context_graph.ConceptDefinitionSemanticReuseIndex(
        layer="mid",
        entries={card["semantic_input_hash"]: entry},
        source_state_id="source-state",
        admitted_source=True,
        candidate_count=1,
        valid_entry_count=1,
    )

    async def forbidden_provider(_packets):
        raise AssertionError("exact semantic reuse hit must not call provider")

    monkeypatch.setattr(
        context_graph,
        "define_mid_concepts_batch",
        forbidden_provider,
    )
    outputs, metrics = (
        await context_graph.define_mid_concepts_batch_with_semantic_reuse(
            [packet],
            index,
            profile_json=profile,
        )
    )

    assert metrics == {
        "hit_count": 1,
        "miss_count": 0,
        "provider_request_count": 0,
    }
    audit = outputs[0]["_provider_output_audit"]
    assert audit["semantic_reuse_hit"] is True
    assert audit["provider_called"] is False
    assert audit["provider_request_count"] == 0
    assert audit["effective_system_prompt_sha256"] == card[
        "effective_system_prompt_sha256"
    ]
    assert audit["provider_response_persisted"] is False


@pytest.mark.asyncio
async def test_mid_semantic_reuse_prompt_change_forces_provider_miss(
    monkeypatch,
):
    from app.services import context_graph

    monkeypatch.setattr(
        context_graph,
        "get_settings",
        lambda: SimpleNamespace(
            graph_api_protocol="anthropic",
            graph_model="unit-graph-model",
            graph_base_url="https://provider.invalid/ai",
            graph_resolve_ip=None,
            model_request_timeout_seconds=240,
        ),
    )
    packet = _packet()
    old_profile = {
        "prompt_pack": {"mid_concept_definition_system": "Stable prompt."}
    }
    changed_profile = {
        "prompt_pack": {"mid_concept_definition_system": "Stable prompt!"}
    }
    old_card = context_graph.concept_definition_semantic_input_card(
        packet,
        layer="mid",
        profile_json=old_profile,
    )
    old_output = context_graph.validate_mid_concept_provider_output(
        {"concepts": [context_graph.mid_concept_fallback(packet)]},
        [packet],
    )[0]
    index = context_graph.ConceptDefinitionSemanticReuseIndex(
        layer="mid",
        entries={
            old_card["semantic_input_hash"]: (
                context_graph.ConceptDefinitionSemanticReuseEntry(
                    semantic_input_hash=old_card["semantic_input_hash"],
                    output=old_output,
                    source_state_id="source-state",
                    source_state_hash="source-state-hash",
                    source_concept_id="source-concept",
                    source_provider_output_hash="source-output-hash",
                    source_prompt_evidence=(
                        "exact_effective_system_prompt_sha256"
                    ),
                )
            )
        },
        admitted_source=True,
    )
    calls = 0

    async def fake_provider(packets):
        nonlocal calls
        calls += 1
        output = context_graph.mid_concept_fallback(packets[0])
        output["_provider_output_audit"] = {
            "provider_request_count": 1,
            "provider_called": True,
            "provider_response_persisted": False,
        }
        return [output]

    monkeypatch.setattr(
        context_graph,
        "define_mid_concepts_batch",
        fake_provider,
    )
    outputs, metrics = (
        await context_graph.define_mid_concepts_batch_with_semantic_reuse(
            [packet],
            index,
            profile_json=changed_profile,
        )
    )

    assert calls == 1
    assert metrics == {
        "hit_count": 0,
        "miss_count": 1,
        "provider_request_count": 1,
    }
    assert outputs[0]["_provider_output_audit"]["semantic_input_hash"] != (
        old_card["semantic_input_hash"]
    )


@pytest.mark.asyncio
async def test_required_mid_semantic_reuse_miss_blocks_before_provider(
    monkeypatch,
):
    from app.services import context_graph

    monkeypatch.setattr(
        context_graph,
        "get_settings",
        lambda: SimpleNamespace(
            graph_api_protocol="anthropic",
            graph_model="unit-graph-model",
            graph_base_url="https://provider.invalid/ai",
            graph_resolve_ip=None,
            model_request_timeout_seconds=240,
        ),
    )
    provider_calls = 0

    async def forbidden_provider(_packets):
        nonlocal provider_calls
        provider_calls += 1
        raise AssertionError("required reuse miss must fail before provider")

    monkeypatch.setattr(
        context_graph,
        "define_mid_concepts_batch",
        forbidden_provider,
    )
    with pytest.raises(RuntimeError, match="blocked before network I/O"):
        await context_graph.define_mid_concepts_batch_with_semantic_reuse(
            [_packet()],
            context_graph.ConceptDefinitionSemanticReuseIndex(layer="mid"),
            profile_json={
                "prompt_pack": {
                    "mid_concept_definition_system": "Stable prompt."
                }
            },
            require_reuse_hit=True,
        )

    assert provider_calls == 0


@pytest.mark.asyncio
async def test_shared_provider_budget_blocks_second_miss_before_network(
    monkeypatch,
):
    from app.services import context_graph

    monkeypatch.setattr(
        context_graph,
        "get_settings",
        lambda: SimpleNamespace(
            graph_api_protocol="anthropic",
            graph_model="unit-graph-model",
            graph_base_url="https://provider.invalid/ai",
            graph_resolve_ip=None,
            model_request_timeout_seconds=240,
        ),
    )
    provider_calls = 0

    async def fake_provider(packets):
        nonlocal provider_calls
        provider_calls += 1
        output = context_graph.mid_concept_fallback(packets[0])
        output["_provider_output_audit"] = {
            "provider_request_count": 1,
            "provider_called": True,
            "provider_response_persisted": False,
        }
        return [output]

    monkeypatch.setattr(
        context_graph,
        "define_mid_concepts_batch",
        fake_provider,
    )
    budget = context_graph.ConceptProviderRequestBudget(max_requests=2)
    empty_index = context_graph.ConceptDefinitionSemanticReuseIndex(
        layer="mid"
    )
    profile = {
        "prompt_pack": {
            "mid_concept_definition_system": "Stable prompt."
        }
    }

    _, metrics = (
        await context_graph.define_mid_concepts_batch_with_semantic_reuse(
            [_packet()],
            empty_index,
            profile_json=profile,
            provider_request_budget=budget,
        )
    )
    assert metrics["provider_request_count"] == 1

    with pytest.raises(
        context_graph.ConceptProviderRequestBudgetExceeded,
        match="before network I/O",
    ) as exc_info:
        await context_graph.define_mid_concepts_batch_with_semantic_reuse(
            [_packet()],
            empty_index,
            profile_json=profile,
            provider_request_budget=budget,
        )

    assert provider_calls == 1
    assert exc_info.value.diagnostics["max_requests"] == 2
    assert exc_info.value.diagnostics["reserved_requests"] == 2
    assert exc_info.value.diagnostics["observed_requests"] == 1
    assert budget.diagnostics()["remaining_worst_case_requests"] == 0


@pytest.mark.asyncio
async def test_mid_provider_does_not_schema_retry_refusal(monkeypatch):
    from app.services import context_graph
    from app.services.error_sanitizer import ExternalServiceError

    calls = 0

    class RefusingProvider:
        def __init__(self, *args, **kwargs):
            pass

        async def classify_json(self, *, system_prompt, user_prompt, fallback):
            nonlocal calls
            calls += 1
            raise ExternalServiceError(
                service="anthropic",
                phase="sdk_messages_completion",
                error_code="provider_refusal",
                retryable=False,
            )

    monkeypatch.setattr(context_graph, "ChatProvider", RefusingProvider)
    monkeypatch.setattr(context_graph, "active_profile_json", lambda: {})
    monkeypatch.setattr(
        context_graph,
        "profile_prompt",
        lambda *_args, **_kwargs: "editable definition prompt",
    )
    monkeypatch.setattr(
        context_graph,
        "get_settings",
        lambda: SimpleNamespace(
            mid_concept_extraction_max_tokens_per_batch=100_000,
            enable_model_fallback=False,
        ),
    )

    with pytest.raises(ExternalServiceError, match="provider_refusal"):
        await context_graph.define_mid_concepts_batch([_packet()])
    assert calls == 1


def test_mid_batch_grouping_accounts_for_exact_wrapper_overhead(monkeypatch):
    from app.services import context_graph

    packets = {
        "cluster-a": _packet("packet-a", excerpt_text="alpha " * 12),
        "cluster-b": _packet("packet-b", excerpt_text="beta " * 12),
    }
    clusters = [
        SimpleNamespace(
            id=cluster_id,
            rq_path_prefix=[index],
            rq_prefix_key=cluster_id,
            label=cluster_id,
        )
        for index, cluster_id in enumerate(reversed(sorted(packets)), start=1)
    ]
    projections = [
        context_graph.concept_provider_projection(packets[key], layer="mid")
        for key in sorted(packets)
    ]
    single_tokens = max(
        context_graph.concept_provider_preflight(
            {"concept_packets": [projection]},
            packet_ids=[projection["packet_id"]],
            identity_count=3,
            max_tokens=100_000,
        )[1]["rough_tokens"]
        for projection in projections
    )
    double_tokens = context_graph.concept_provider_preflight(
        {"concept_packets": projections},
        packet_ids=[projection["packet_id"] for projection in projections],
        identity_count=6,
        max_tokens=100_000,
    )[1]["rough_tokens"]
    assert double_tokens > single_tokens
    token_limit = double_tokens - 1
    assert token_limit >= single_tokens
    monkeypatch.setattr(
        context_graph,
        "concept_packet_for_cluster",
        lambda _db, cluster, **_kwargs: packets[cluster.id],
    )
    settings = SimpleNamespace(
        mid_concept_extraction_max_candidates_per_batch=8,
        mid_concept_extraction_max_tokens_per_batch=token_limit,
    )

    batches = list(
        context_graph.iter_mid_concept_packet_batches(
            object(),
            clusters,
            settings,
            build_context=object(),
        )
    )
    observed = [
        [packet["packet_id"] for _cluster, packet in batch]
        for batch in batches
    ]
    assert observed == [["packet-b"], ["packet-a"]]
    replayed = list(
        context_graph.iter_mid_concept_packet_batches(
                object(),
                list(reversed(clusters)),
                settings,
                build_context=object(),
            )
        )
    assert [
        [packet["packet_id"] for _cluster, packet in batch]
        for batch in replayed
    ] == observed


def test_bounded_construction_never_materializes_more_than_one_window():
    from app.services.context_graph import bounded_concept_packet_windows

    constructed = 0

    def items():
        nonlocal constructed
        for index in range(7):
            constructed += 1
            yield index

    windows = bounded_concept_packet_windows(items(), window_size=3)
    first = next(windows)
    assert first == [0, 1, 2]
    assert constructed == 3
    remaining = list(windows)
    assert remaining == [[3, 4, 5], [6]]
    assert constructed == 7


def test_structure_business_multiset_hash_is_order_independent_and_keeps_duplicates():
    from app.services.context_graph import (
        STRUCTURE_MAPPING_BUSINESS_FACT_DIGEST_PROTOCOL_VERSION,
        STRUCTURE_MAPPING_FACT_SET_PROTOCOL_VERSION,
    )
    from app.services.graph_state_hashes import (
        streaming_canonical_fact_digest_multiset_hash,
    )

    facts = [
        {"chunk_business_key": "chunk-a", "path": "A", "weight": 0.5},
        {"chunk_business_key": "chunk-b", "path": "B", "weight": 0.7},
    ]

    def digest(rows):
        return streaming_canonical_fact_digest_multiset_hash(
            protocol_version=STRUCTURE_MAPPING_FACT_SET_PROTOCOL_VERSION,
            fact_protocol_version=(
                STRUCTURE_MAPPING_BUSINESS_FACT_DIGEST_PROTOCOL_VERSION
            ),
            facts=rows,
            sort_run_size=1,
            merge_fan_in=2,
        )

    forward = digest(facts)
    reversed_result = digest(reversed(facts))
    duplicate = digest([*facts, facts[0]])
    assert forward.state_hash == reversed_result.state_hash
    assert forward.fact_count == reversed_result.fact_count == 2
    assert duplicate.state_hash != forward.state_hash
    assert duplicate.fact_count == 3


def test_structure_packet_streams_complete_hash_with_bounded_trace_without_orm_rows(
    db_session,
    sample_knowledge_base,
):
    from app.models import (
        Chunk,
        ChunkStructureMapping,
        ChunkStructureNode,
        Document,
        DocumentVersion,
    )
    from app.services.chunking import text_hash
    from app.services.context_graph import (
        STRUCTURE_PATH_TRACE_SAMPLE_LIMIT,
        _structure_path_evidence_for_chunks,
    )

    document = Document(
        knowledge_base_id=sample_knowledge_base.id,
        title="Bounded packet",
        source_path="bounded-packet.pdf",
        source_type="pdf",
        checksum="bounded-packet",
        tags=[],
        is_active=True,
    )
    db_session.add(document)
    db_session.flush()
    version = DocumentVersion(
        document_id=document.id,
        version=1,
        checksum="bounded-packet",
        storage_path="bounded-packet.pdf",
        is_active=True,
    )
    db_session.add(version)
    db_session.flush()
    chunk = Chunk(
        knowledge_base_id=sample_knowledge_base.id,
        document_id=document.id,
        document_version_id=version.id,
        chunk_version=1,
        chunk_index=0,
        token_start=0,
        token_end=20,
        char_start=0,
        char_end=40,
        text="bounded packet evidence",
        text_hash=text_hash("bounded packet evidence"),
        section_path="Bounded",
        page_start=1,
        page_end=1,
        metadata_json={},
        state="active",
    )
    db_session.add(chunk)
    db_session.flush()
    for index in range(STRUCTURE_PATH_TRACE_SAMPLE_LIMIT + 16):
        node = ChunkStructureNode(
            knowledge_base_id=sample_knowledge_base.id,
            document_id=document.id,
            document_version_id=version.id,
            node_type="paragraph",
            depth=3,
            title=f"Node {index}",
            char_start=0,
            char_end=40,
            page_number=1,
            path=f"Bounded / {index}",
            bbox_json={},
            layout_json={},
        )
        db_session.add(node)
        db_session.flush()
        db_session.add(
            ChunkStructureMapping(
                chunk_id=chunk.id,
                structure_node_id=node.id,
                document_version_id=version.id,
                overlap_chars=40,
                overlap_tokens=20,
                coverage_ratio=1.0,
                span_overlap=1.0,
                mapping_weight=1.0,
                mapping_role="paragraph",
            )
        )
    db_session.flush()
    chunk_id = str(chunk.id)
    db_session.expunge_all()

    first = _structure_path_evidence_for_chunks(db_session, [chunk_id])
    second = _structure_path_evidence_for_chunks(db_session, [chunk_id])

    assert first["mapping_count"] == STRUCTURE_PATH_TRACE_SAMPLE_LIMIT + 16
    assert first["path_trace_sample_count"] == STRUCTURE_PATH_TRACE_SAMPLE_LIMIT
    assert len(first["paths"]) == STRUCTURE_PATH_TRACE_SAMPLE_LIMIT
    assert first["paths_complete"] is False
    assert first["business_facts_hash_card"]["count"] == first["mapping_count"]
    assert first["business_facts_hash"] == second["business_facts_hash"]
    assert first["address_facts_hash"] == second["address_facts_hash"]
    assert first["paths"] == second["paths"]
    assert not any(
        isinstance(value, (ChunkStructureMapping, ChunkStructureNode))
        for value in db_session.identity_map.values()
    )

    from sqlalchemy import select

    malformed = db_session.scalars(
        select(ChunkStructureMapping).where(
            ChunkStructureMapping.chunk_id == chunk_id
        )
    ).first()
    assert malformed is not None
    malformed.mapping_protocol_version = "   "
    db_session.flush()
    db_session.expunge_all()
    with pytest.raises(RuntimeError, match="empty mapping protocol version"):
        _structure_path_evidence_for_chunks(db_session, [chunk_id])
