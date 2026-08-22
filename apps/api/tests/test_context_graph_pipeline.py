from __future__ import annotations

import asyncio
from collections import Counter
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


def test_structure_edges_are_rejected_from_active_relation_graph():
    from app.services.context_graph import relation_edge_source_algorithm

    with pytest.raises(RuntimeError, match="Structure-derived relation edges are not allowed"):
        relation_edge_source_algorithm("same_page_region")


def test_relation_overview_rq_node_lookup_does_not_hydrate_heavy_membership_json():
    from app.services.context_graph import _rq_membership_node_metadata_by_chunk

    captured_sql: list[str] = []

    class _Rows:
        @staticmethod
        def all():
            return [
                SimpleNamespace(
                    chunk_id="chunk-1",
                    rq_path=[1, 2, 3],
                    residual_norm=0.25,
                    membership_reason="rq_leaf",
                )
            ]

    class _Database:
        @staticmethod
        def execute(statement):
            captured_sql.append(str(statement))
            return _Rows()

    result = _rq_membership_node_metadata_by_chunk(
        _Database(),
        relation_state_id="relation-state-1",
        chunk_ids=["chunk-1"],
    )

    assert result == {
        "chunk-1": {"rq_path": [1, 2, 3], "residual_norm": 0.25}
    }
    assert len(captured_sql) == 1
    assert "support_chunk_edge_ids_json" not in captured_sql[0]
    assert "diagnostics_json" not in captured_sql[0]
    assert "top_alternative_prefix_ids_json" not in captured_sql[0]


def test_postgresql_projection_admission_uses_narrow_lateral_jsonb_fences():
    from sqlalchemy.dialects import postgresql

    from app.models import CoarseConceptEdge, MidConceptEdge
    from app.services.context_graph import _projection_edge_admission_mismatch_count

    captured_sql: list[str] = []

    class _Database:
        bind = SimpleNamespace(dialect=SimpleNamespace(name="postgresql"))

        @staticmethod
        def scalar(statement):
            captured_sql.append(
                str(
                    statement.compile(
                        dialect=postgresql.dialect(),
                        compile_kwargs={"literal_binds": True},
                    )
                )
            )
            return 0

    for edge_model, state_id_column, layer in (
        (MidConceptEdge, MidConceptEdge.concept_state_id, "mid"),
        (CoarseConceptEdge, CoarseConceptEdge.coarse_state_id, "coarse"),
    ):
        assert (
            _projection_edge_admission_mismatch_count(
                _Database(),
                edge_model=edge_model,
                state_id_column=state_id_column,
                state_id=f"{layer}-state-1",
                state_hash="a" * 64,
                layer=layer,
            )
            == 0
        )

    assert len(captured_sql) == 2
    for sql in captured_sql:
        assert sql.count("JOIN LATERAL") == 3
        assert sql.count("OFFSET 0") == 3
        assert "MATERIALIZED" not in sql
        assert " AS JSONB)" in sql
    assert "CAST(mid_concept_edges.raw_strength_summary_json AS JSONB)" in captured_sql[0]
    assert "CAST(mid_concept_edges.diagnostics_json AS JSONB)" in captured_sql[0]
    assert "CAST(coarse_concept_edges.raw_strength_summary_json AS JSONB)" in captured_sql[1]
    assert "CAST(coarse_concept_edges.diagnostics_json AS JSONB)" in captured_sql[1]


def test_concept_provider_output_shape_card_is_content_free_and_stable():
    from app.services.context_graph import concept_provider_output_shape_card

    first = concept_provider_output_shape_card(
        {
            "packet_id": "private-packet-value",
            "definition": "private provider content",
            "aliases": ["private alias"],
        }
    )
    replay = concept_provider_output_shape_card(
        {
            "aliases": ["private alias"],
            "definition": "private provider content",
            "packet_id": "private-packet-value",
        }
    )

    assert first == replay
    assert first["top_level_keys"] == ["aliases", "definition", "packet_id"]
    assert first["value_types"] == {
        "aliases": "list",
        "definition": "str",
        "packet_id": "str",
    }
    assert first["canonical_bytes"] > 0
    assert len(first["canonical_sha256"]) == 64
    serialized = json.dumps(first, sort_keys=True)
    assert "private-packet-value" not in serialized
    assert "private provider content" not in serialized
    assert "private alias" not in serialized


def _valid_mid_provider_item(packet_id: str = "packet-1") -> dict:
    return {
        "packet_id": packet_id,
        "canonical_label": "Bayesian network inference",
        "aliases": ["Grounded alias"],
        "display_terms": ["Bayesian network inference"],
        "summary": "A bounded grounded summary.",
        "definition": "A bounded grounded definition.",
        "scope_note": "Only the supplied packet.",
        "inclusion_criteria": ["Supported by the packet."],
        "exclusion_criteria": ["Unsupported claims."],
        "internal_state": {"definition_source": "provider"},
        "representative_chunk_ids": ["chunk-1"],
        "support_chunk_ids": ["chunk-1"],
        "confidence": 0.8,
        "why_this_concept_exists": "The packet contains shared grounded evidence.",
    }


def test_mid_provider_output_schema_is_exact_bounded_and_content_free():
    from app.services import context_graph

    packets = [{"packet_id": "packet-1"}]
    valid = {"concepts": [_valid_mid_provider_item()]}
    assert context_graph.validate_mid_concept_provider_output(valid, packets) == valid[
        "concepts"
    ]

    unexpected = _valid_mid_provider_item()
    unexpected["private-provider-sentinel"] = "must-not-leak"
    with pytest.raises(
        context_graph.ConceptProviderOutputSchemaError
    ) as extra_error:
        context_graph.validate_mid_concept_provider_output(
            {"concepts": [unexpected]}, packets
        )
    assert "item_keys_not_exact" in str(extra_error.value)
    assert "private-provider-sentinel" not in str(extra_error.value)
    assert "must-not-leak" not in str(extra_error.value)

    oversized = _valid_mid_provider_item()
    oversized["definition"] = "private-definition-sentinel" * 100
    with pytest.raises(
        context_graph.ConceptProviderOutputSchemaError
    ) as length_error:
        context_graph.validate_mid_concept_provider_output(
            {"concepts": [oversized]}, packets
        )
    assert "string_length_exceeded" in str(length_error.value)
    assert "private-definition-sentinel" not in str(length_error.value)

    wrapped = RuntimeError("private-wrapper-message-must-not-persist")
    wrapped.__cause__ = length_error.value
    failure_card = context_graph.concept_provider_output_failure_card(wrapped)
    assert failure_card["classified"] is True
    assert failure_card["cause_depth"] == 1
    assert failure_card["error_code"] == "string_length_exceeded"
    assert failure_card["field_path"] == "definition"
    assert failure_card["provider_response_persisted"] is False
    assert "private-definition-sentinel" not in json.dumps(failure_card)
    assert "private-wrapper-message-must-not-persist" not in json.dumps(
        failure_card
    )

    batch_error = context_graph.ConceptProviderBatchError(
        layer="mid",
        batch_index=21,
        packet_ids=["8ea8f8d11106cf12", "6b9adcd7b56840f2"],
    )
    assert batch_error.packet_ids == [
        "8ea8f8d11106cf12",
        "6b9adcd7b56840f2",
    ]

    shape_error = context_graph.ProviderJSONShapeError(
        {
            "protocol_version": "provider_json_text_shape_v1",
            "error_code": "json_decode_error",
            "field_path": "$",
            "utf8_bytes": 17,
            "sha256": "a" * 64,
            "starts_with_object": True,
            "ends_with_object": False,
            "contains_code_fence": False,
            "decode_error_position": 16,
            "layer": "mid",
            "attempt_count": 2,
            "max_attempts": 2,
            "first_attempt_failure": "provider_json_shape_rejected",
            "private_value": "private-provider-text-must-not-persist",
        }
    )
    shape_batch = context_graph.ConceptProviderBatchError(
        layer="mid",
        batch_index=44,
        packet_ids=["7f05ad4924a6e7cd"],
    )
    shape_batch.__cause__ = shape_error
    shape_failure_card = context_graph.concept_provider_output_failure_card(
        shape_batch
    )
    assert shape_failure_card["classified"] is True
    assert shape_failure_card["error_code"] == "json_decode_error"
    assert shape_failure_card["attempt_count"] == 2
    assert shape_failure_card["batch"]["packet_ids"] == [
        "7f05ad4924a6e7cd"
    ]
    assert "private-provider-text-must-not-persist" not in json.dumps(
        shape_failure_card
    )

    preflight_error = context_graph.ConceptPacketPreflightError(
        {
            "protocol_version": "concept_provider_ordered_admissible_pack_v2",
            "layer": "coarse",
            "packet_ids": ["8f9f86507a2cdb40"],
            "packet_count": 1,
            "decision": "reject_projection_not_admissible",
            "network_call_count": 0,
            "candidate_count": 10,
            "candidate_bindings_hash": "b" * 64,
            "base_serialized_bytes": 24000,
            "base_rough_tokens": 2300,
            "smallest_candidate": {
                "candidate_kind": "child",
                "serialized_bytes": 27000,
                "rough_tokens": 2500,
                "candidate_binding_hash": "c" * 64,
            },
            "max_serialized_bytes": 28800,
            "max_rough_tokens": 2400,
            "private_value": "private-packet-text-must-not-persist",
        }
    )
    preflight_card = context_graph.concept_provider_output_failure_card(
        preflight_error
    )
    assert preflight_card["classified"] is True
    assert preflight_card["layer"] == "coarse"
    assert preflight_card["attempt_count"] == 0
    assert preflight_card["shape_card"]["network_call_count"] == 0
    assert preflight_card["shape_card"]["smallest_candidate"][
        "rough_tokens"
    ] == 2500
    assert "private-packet-text-must-not-persist" not in json.dumps(
        preflight_card
    )

    unknown_batch = context_graph.ConceptProviderBatchError(
        layer="mid",
        batch_index=75,
        packet_ids=["7986aac1f2747143"],
    )
    unknown_batch.__cause__ = RuntimeError(
        "private-unknown-provider-body-must-not-persist"
    )
    unknown_card = context_graph.concept_provider_output_failure_card(
        unknown_batch
    )
    assert unknown_card["classified"] is False
    assert unknown_card["exception_chain_types"] == [
        "ConceptProviderBatchError",
        "RuntimeError",
    ]
    assert "private-unknown-provider-body-must-not-persist" not in json.dumps(
        unknown_card
    )


def test_mid_provider_output_schema_rejects_duplicate_or_extra_packet_ids():
    from app.services import context_graph

    packets = [{"packet_id": "packet-1"}, {"packet_id": "packet-2"}]
    duplicate = {
        "concepts": [
            _valid_mid_provider_item("packet-1"),
            _valid_mid_provider_item("packet-1"),
        ]
    }
    with pytest.raises(
        context_graph.ConceptProviderOutputSchemaError,
        match="duplicate_packet_id",
    ):
        context_graph.validate_mid_concept_provider_output(duplicate, packets)

    extra = {"concepts": [_valid_mid_provider_item("packet-outside")]}
    with pytest.raises(
        context_graph.ConceptProviderOutputSchemaError,
        match="unexpected_packet_id",
    ):
        context_graph.validate_mid_concept_provider_output(
            extra, [{"packet_id": "packet-1"}]
        )


@pytest.mark.parametrize(
    "label,reason",
    [
        ("未命名概念", "placeholder_label"),
        ("RQ L3 3/4/1", "address_label"),
        ("Chunk 17", "address_label"),
        ("a" * 64, "storage_identity_label"),
        ("12345", "numeric_or_symbol_label"),
    ],
)
def test_mid_provider_output_schema_rejects_nonsemantic_labels(label, reason):
    from app.services import context_graph

    item = _valid_mid_provider_item()
    item["canonical_label"] = label

    with pytest.raises(
        context_graph.ConceptProviderOutputSchemaError,
        match=f"natural_label_{reason}",
    ):
        context_graph.validate_mid_concept_provider_output(
            {"concepts": [item]}, [{"packet_id": "packet-1"}]
        )


def test_coarse_provider_output_schema_rejects_rq_address_label():
    from app.services import context_graph

    output = {
        "coarse_label": "RQ L2 5/4",
        "display_terms": ["三阶段质控模型"],
        "aliases": [],
        "definition": "事前、事中和事后质控形成连续质量控制。",
        "summary": "三阶段质控模型。",
        "scope_note": "仅限给定子概念。",
        "inclusion_criteria": ["有子概念支撑。"],
        "exclusion_criteria": ["无支撑主题。"],
        "internal_state": {},
        "included_mid_concepts": [],
        "boundary_concepts": [],
        "bridge_concepts": [],
        "outlier_concepts": [],
        "low_confidence_concepts": [],
        "cross_community_weak_ties": [],
        "confidence": 0.8,
    }

    with pytest.raises(
        context_graph.ConceptProviderOutputSchemaError,
        match="natural_label_address_label",
    ):
        context_graph.validate_coarse_concept_provider_output(output)


def test_coarse_provider_output_schema_rejects_nonfinite_confidence_without_content():
    from app.services import context_graph

    output = {
        "coarse_label": "Grounded area",
        "display_terms": ["Grounded area"],
        "aliases": [],
        "definition": "Grounded definition.",
        "summary": "Grounded summary.",
        "scope_note": "Supplied packet only.",
        "inclusion_criteria": ["Supported concepts."],
        "exclusion_criteria": ["Unsupported concepts."],
        "internal_state": {"definition_source": "provider"},
        "included_mid_concepts": [],
        "boundary_concepts": [],
        "bridge_concepts": [],
        "outlier_concepts": [],
        "low_confidence_concepts": [],
        "cross_community_weak_ties": [],
        "confidence": float("nan"),
    }
    with pytest.raises(
        context_graph.ConceptProviderOutputSchemaError
    ) as error:
        context_graph.validate_coarse_concept_provider_output(output)
    assert "item_json_not_canonical" in str(error.value)
    assert "Grounded definition" not in str(error.value)


@pytest.mark.parametrize("invalid_depth", [1, 2, 4])
def test_rq_graph_builder_fails_fast_for_non_protocol_depth(monkeypatch, invalid_depth):
    from app.services import context_graph

    monkeypatch.setattr(
        context_graph,
        "get_settings",
        lambda: SimpleNamespace(rq_kmeans_levels=invalid_depth, rq_kmeans_max_k=6, rq_residual_tau=0.65),
    )

    with pytest.raises(RuntimeError, match="RQ depth must be exactly 3"):
        context_graph.rq_runtime_config()


def test_entry_seed_calibration_prevents_zero_distance_route_seeds():
    from app.services.context_graph import calibrated_entry_seed_strength, distance_from_strength

    assert calibrated_entry_seed_strength(1.0, "dense_entry") == pytest.approx(0.97)
    assert calibrated_entry_seed_strength(1.0, "mid_drilldown_entry") == pytest.approx(0.82)
    assert calibrated_entry_seed_strength(1.0, "coarse_to_mid_drilldown_entry") == pytest.approx(0.72)
    assert distance_from_strength(calibrated_entry_seed_strength(1.0, "mid_drilldown_entry")) > 0.0


@pytest.mark.asyncio
async def test_gather_bounded_propagates_worker_exception_without_deadlock():
    from app.services.context_graph import gather_bounded

    async def run_item(item: str) -> str:
        await asyncio.sleep(0)
        if item == "bad":
            raise RuntimeError("unit bounded worker failure")
        return item

    with pytest.raises(RuntimeError, match="unit bounded worker failure"):
        await asyncio.wait_for(gather_bounded(["ok-1", "bad", "ok-2", "ok-3"], 2, run_item), timeout=1.0)


def test_result_top_k_resolves_hot_reload_default_and_cache_key(monkeypatch):
    from app.core.config import get_settings
    from app.schemas import SearchFilters
    from app.services import context_graph

    monkeypatch.setattr(context_graph, "get_settings", lambda: SimpleNamespace(retrieval_result_top_k_default=7))

    assert context_graph.resolve_result_top_k(None) == 7
    assert context_graph.resolve_result_top_k(3) == 3
    with pytest.raises(ValueError, match="between 1 and 50"):
        context_graph.resolve_result_top_k(51)

    monkeypatch.setattr(context_graph, "get_settings", get_settings)
    components = context_graph.context_graph_cache_key_components(
        knowledge_base_id="kb-1",
        query="graph retrieval",
        filters=SearchFilters(),
        context_state=None,
        retrieval_mode="layered_context_graph",
        conversation_state_scope_hash="a" * 64,
        result_top_k=7,
    )
    assert components["result_top_k"] == 7
    assert components["retrieval_granularity"] == "mid"


def test_retrieval_granularity_changes_cache_key_component():
    from app.schemas import SearchFilters
    from app.services import context_graph

    base = {
        "knowledge_base_id": "kb-1",
        "query": "graph retrieval",
        "filters": SearchFilters(),
        "context_state": None,
        "retrieval_mode": "layered_context_graph",
        "conversation_state_scope_hash": "a" * 64,
        "result_top_k": 7,
    }

    mid_key = context_graph.context_graph_cache_key(**base, retrieval_granularity="mid")
    coarse_key = context_graph.context_graph_cache_key(**base, retrieval_granularity="coarse")

    assert mid_key != coarse_key


@pytest.mark.asyncio
async def test_layered_search_writes_default_result_top_k_for_empty_kb(monkeypatch, db_session, sample_knowledge_base):
    from app.schemas import SearchFilters
    from app.services import context_graph

    settings = context_graph.get_settings().model_copy(
        update={"retrieval_result_top_k_default": 6}
    )
    monkeypatch.setattr(context_graph, "get_settings", lambda: settings)

    result = await context_graph.layered_search(
        db_session,
        sample_knowledge_base.id,
        "empty knowledge base",
        SearchFilters(),
        None,
    )

    assert result.results == []
    assert result.audit["result_top_k"] == 6
    assert result.audit["retrieval_granularity"] == "mid"
    assert result.trace.diagnostics_json["result_top_k"] == 6
    assert result.trace.diagnostics_json["retrieval_granularity"] == "mid"


def test_path_distance_zone_has_red_and_hard_boundaries():
    from app.services.context_graph import _path_distance_zone

    envelope = {
        "path_distance_green_threshold": 0.45,
        "path_distance_gray_threshold": 1.35,
        "path_distance_hard_threshold": 2.4,
    }

    assert _path_distance_zone(0.45, envelope) == "green"
    assert _path_distance_zone(1.35, envelope) == "gray"
    assert _path_distance_zone(2.4, envelope) == "red"
    assert _path_distance_zone(2.4001, envelope) == "hard_stop"


def test_query_facet_packet_drops_fillers_and_matches_aliases():
    from app.services.context_graph import matched_required_facets_for_text, query_facets_for_search

    facets = query_facets_for_search(
        "\u7ed9\u6211\u914d\u7f6e\u6a21\u578b\u7684\u5177\u4f53\u7b97\u6cd5\u6b65\u9aa4",
        {
            "facet_groups": [
                {"facet": "\u914d\u7f6e\u6a21\u578b", "role": "domain", "aliases": ["configuration model"]},
                {
                    "facet": "\u7b97\u6cd5\u6b65\u9aa4",
                    "role": "procedure",
                    "aliases": ["\u6807\u51c6\u6784\u9020", "\u534a\u8fb9", "stub", "\u968f\u673a\u5339\u914d", "\u4e24\u4e24\u5339\u914d"],
                }
            ],
            "drop_terms": ["\u7ed9", "\u6211", "\u7684", "\u5177\u4f53"],
            "answer_shape": "step_by_step_algorithm",
        },
        {"intent": "procedure"},
    )

    assert facets["protocol_version"] == "query_facet_packet_v2"
    assert facets["intent"] == "procedure"
    assert "\u7ed9" not in facets["required_facets"]
    assert "\u6211" not in facets["required_facets"]
    assert "\u7684" not in facets["required_facets"]
    assert "\u914d\u7f6e\u6a21\u578b" in facets["required_facets"]
    assert "\u7b97\u6cd5\u6b65\u9aa4" in facets["required_facets"]
    assert "\u7ed9" not in facets["terms"]
    assert "\u6211" not in facets["terms"]

    text = "\u914d\u7f6e\u6a21\u578b\u7684\u6807\u51c6\u6784\u9020\uff1a\u4e3a\u6bcf\u4e2a\u8282\u70b9\u653e\u7f6e k_i \u4e2a\u534a\u8fb9 stub\uff0c\u7136\u540e\u5c06\u534a\u8fb9\u968f\u673a\u5339\u914d\u3002"
    matched = set(matched_required_facets_for_text(text, facets))
    assert {"\u914d\u7f6e\u6a21\u578b", "\u7b97\u6cd5\u6b65\u9aa4"}.issubset(matched)


def test_query_facet_packet_uses_unified_facet_groups_and_aliases():
    from app.services.context_graph import query_facets_for_search

    facets = query_facets_for_search(
        "alpha relation placeholder",
        {
            "facet_groups": [
                {"facet": "alpha concept", "role": "domain", "aliases": ["alpha topic", "alpha alias"]},
                {"facet": "beta procedure", "role": "procedure", "aliases": ["beta step", "beta alias"]},
            ],
            "drop_terms": ["placeholder"],
            "answer_shape": "comparison",
        },
        {"intent": "comparison"},
    )

    groups = {group["facet"]: group for group in facets["facet_groups"]}
    assert groups["alpha concept"]["aliases"] == ["alpha topic", "alpha alias"]
    assert groups["beta procedure"]["aliases"] == ["beta step", "beta alias"]
    assert {"alpha", "concept", "topic", "alias", "beta", "procedure", "step"}.issubset(set(facets["terms"]))
    assert facets["diagnostics"]["schema_validation"] == "canonical_facet_groups_only"


@pytest.mark.parametrize("legacy_field", ["domain_facets", "procedure_facets", "alias_facets", "required_facets"])
def test_query_facet_packet_rejects_every_legacy_split_field(legacy_field):
    from app.services.context_graph import QueryFacetValidationError, query_facets_for_search

    with pytest.raises(QueryFacetValidationError, match="legacy_fields_forbidden") as exc_info:
        query_facets_for_search("legacy concept", {legacy_field: ["legacy concept"]}, {"intent": "definition"})

    assert exc_info.value.fields == [legacy_field]


def test_query_facet_packet_rejects_role_outside_the_fixed_allowlist():
    from app.services.context_graph import QueryFacetValidationError, query_facets_for_search

    with pytest.raises(QueryFacetValidationError, match="facet_role_not_allowed"):
        query_facets_for_search(
            "alpha concept",
            {
                "facet_groups": [{"facet": "alpha concept", "role": "required", "aliases": []}],
                "drop_terms": [],
                "answer_shape": "definition",
            },
            {"intent": "definition"},
        )


def test_query_facet_packet_rejects_cross_query_or_stale_protocol_reuse():
    from app.services.context_graph import QueryFacetValidationError, query_facets_for_search

    packet = query_facets_for_search("alpha concept", None, {"intent": "definition"})

    with pytest.raises(QueryFacetValidationError, match="packet_query_mismatch"):
        query_facets_for_search("beta concept", packet, {"intent": "definition"})

    packet["protocol_version"] = "query_facet_packet_v1"
    with pytest.raises(QueryFacetValidationError, match="packet_protocol_version_mismatch"):
        query_facets_for_search("alpha concept", packet, {"intent": "definition"})


def _source_snapshot_fixture(sample_knowledge_base, filename: str, content: str) -> tuple[Path, Path, str]:
    from app.core.config import get_settings
    from app.services.storage import snapshot_source_file

    storage_root = get_settings().knowledge_base_paths_for_name(sample_knowledge_base.name)["storage_root"]
    storage_root.mkdir(parents=True, exist_ok=True)
    logical_source = storage_root / filename
    logical_source.write_text(content, encoding="utf-8")
    frozen_snapshot = snapshot_source_file(logical_source, sample_knowledge_base.name)
    return logical_source, frozen_snapshot.canonical_path, frozen_snapshot.checksum


def _build_token_budget_context_package(db_session, sample_knowledge_base, *, texts: list[str], token_budget: int):
    from app.models import Chunk, Document, DocumentVersion, RetrievalTrace
    from app.services.chunking import rough_token_count, text_hash
    from app.services.context_graph import build_context_package

    logical_source, snapshot_path, checksum = _source_snapshot_fixture(
        sample_knowledge_base,
        "token-budget-evidence.md",
        "\n".join(texts),
    )
    document = Document(
        knowledge_base_id=sample_knowledge_base.id,
        title="Token budget evidence",
        source_path=str(logical_source),
        source_type="markdown",
        tags=["budget"],
        checksum=checksum,
        is_active=True,
    )
    db_session.add(document)
    db_session.flush()
    version = DocumentVersion(
        document_id=document.id,
        version=1,
        checksum=checksum,
        storage_path=str(snapshot_path),
        is_active=True,
    )
    db_session.add(version)
    db_session.flush()

    chunks = []
    char_cursor = 0
    token_cursor = 0
    for index, text in enumerate(texts):
        chunk_token_count = rough_token_count(text)
        chunk = Chunk(
            knowledge_base_id=sample_knowledge_base.id,
            document_id=document.id,
            document_version_id=version.id,
            chunk_version=1,
            chunk_index=index,
            token_start=token_cursor,
            token_end=token_cursor + chunk_token_count,
            char_start=char_cursor,
            char_end=char_cursor + len(text),
            text=text,
            text_hash=text_hash(text),
            section_path="Token budget",
            page_start=1,
            page_end=1,
            state="active",
        )
        chunks.append(chunk)
        char_cursor += len(text) + 1
        token_cursor += chunk_token_count
    db_session.add_all(chunks)
    db_session.flush()

    trace = RetrievalTrace(
        knowledge_base_id=sample_knowledge_base.id,
        query="token budget",
        filters_json={},
        result_chunk_ids_json=[chunk.id for chunk in chunks],
        query_facets_json={},
        path_labels_json=[],
        convergence_json={
            "gray_zone_decision_count": 0,
            "gray_zone_rule_evaluation_count": 0,
            "red_zone_pruned_count": 0,
            "hard_stop_pruned_count": 0,
            "gray_zone_model_call_count": 0,
        },
    )
    db_session.add(trace)
    db_session.flush()
    package = build_context_package(
        db_session,
        knowledge_base_id=sample_knowledge_base.id,
        query=trace.query,
        trace=trace,
        results=[{"chunk_id": chunk.id, "metadata": {"traversal": {}}} for chunk in chunks],
        token_budget=token_budget,
    )
    return package, chunks


def test_context_package_clips_first_oversized_chunk_without_forging_raw_span(db_session, sample_knowledge_base):
    from app.services.chunking import rough_token_count
    from app.services.context_graph import context_package_to_contexts

    text = "alpha beta gamma delta epsilon"
    package, chunks = _build_token_budget_context_package(
        db_session,
        sample_knowledge_base,
        texts=[text],
        token_budget=3,
    )

    packed = (package.package_json or {})["chunks"]
    assert len(packed) == 1
    assert packed[0]["content"] == "alpha beta gamma"
    assert packed[0]["content_clipped"] is True
    assert packed[0]["content_token_count"] == 3
    assert packed[0]["original_token_count"] == 5
    assert packed[0]["raw_chunk_char_span"] == [0, len(text)]
    assert packed[0]["char_span"] == [0, len("alpha beta gamma")]
    assert packed[0]["source_span"]["char_span"] == packed[0]["char_span"]
    assert package.citation_spans_json[0]["char_span"] == packed[0]["char_span"]
    assert package.token_count == rough_token_count(packed[0]["content"]) == package.token_budget == 3
    assert package.diagnostics_json["token_budget_audit"]["clipped_chunk_ids"] == [chunks[0].id]
    assert package.diagnostics_json["token_budget_audit"]["within_budget"] is True

    context = context_package_to_contexts(package)[0]
    assert context["content"] == packed[0]["content"]
    assert context["metadata"]["source_span"]["char_span"] == packed[0]["char_span"]
    assert context["metadata"]["content_clipped"] is True


def test_context_package_keeps_full_chunk_when_it_fits_budget(db_session, sample_knowledge_base):
    from app.services.chunking import rough_token_count
    from app.services.context_graph import context_package_to_contexts

    text = "alpha beta gamma"
    package, _chunks = _build_token_budget_context_package(
        db_session,
        sample_knowledge_base,
        texts=[text],
        token_budget=10,
    )

    packed = (package.package_json or {})["chunks"]
    assert len(packed) == 1
    assert packed[0]["content"] == text
    assert packed[0]["content_clipped"] is False
    assert packed[0]["char_span"] == packed[0]["raw_chunk_char_span"] == [0, len(text)]
    assert package.token_count == rough_token_count(text) == 3
    assert context_package_to_contexts(package)[0]["content"] == text


def test_context_package_token_count_is_audited_and_later_oversized_chunk_is_skipped(db_session, sample_knowledge_base):
    from app.services.chunking import rough_token_count
    from app.services.context_graph import context_package_to_contexts

    package, chunks = _build_token_budget_context_package(
        db_session,
        sample_knowledge_base,
        texts=["one two three", "four five six seven"],
        token_budget=5,
    )

    packed = (package.package_json or {})["chunks"]
    audited_count = sum(rough_token_count(item["content"]) for item in packed)
    assert [item["chunk_id"] for item in packed] == [chunks[0].id]
    assert package.token_count == audited_count == 3
    assert package.token_count <= package.token_budget
    assert package.diagnostics_json["token_budget_audit"]["skipped_chunk_ids"] == [chunks[1].id]

    package.token_count += 1
    with pytest.raises(RuntimeError, match="token count audit mismatch"):
        context_package_to_contexts(package)


def test_context_package_restores_graph_path_chunks(db_session, sample_knowledge_base):
    from app.models import Chunk, Document, DocumentVersion, RetrievalTrace
    from app.services.chunking import text_hash
    from app.services.context_graph import build_context_package, context_package_to_contexts

    logical_source, snapshot_path, checksum = _source_snapshot_fixture(
        sample_knowledge_base,
        "configuration-model.md",
        "Configuration model uses stubs and random matching.\n"
        "The final answer should cite grounded algorithm steps.",
    )
    document = Document(
        knowledge_base_id=sample_knowledge_base.id,
        title="Configuration model",
        source_path=str(logical_source),
        source_type="markdown",
        tags=["network"],
        checksum=checksum,
        is_active=True,
    )
    db_session.add(document)
    db_session.flush()
    version = DocumentVersion(
        document_id=document.id,
        version=1,
        checksum=checksum,
        storage_path=str(snapshot_path),
        is_active=True,
    )
    db_session.add(version)
    db_session.flush()
    path_chunk = Chunk(
        knowledge_base_id=sample_knowledge_base.id,
        document_id=document.id,
        document_version_id=version.id,
        chunk_version=1,
        chunk_index=0,
        token_start=0,
        token_end=12,
        char_start=0,
        char_end=80,
        text="Configuration model uses stubs and random matching.",
        text_hash=text_hash("Configuration model uses stubs and random matching."),
        section_path="Configuration model",
        page_start=1,
        page_end=1,
        state="active",
    )
    hit_chunk = Chunk(
        knowledge_base_id=sample_knowledge_base.id,
        document_id=document.id,
        document_version_id=version.id,
        chunk_version=1,
        chunk_index=1,
        token_start=12,
        token_end=24,
        char_start=81,
        char_end=160,
        text="The final answer should cite grounded algorithm steps.",
        text_hash=text_hash("The final answer should cite grounded algorithm steps."),
        section_path="Configuration model",
        page_start=1,
        page_end=1,
        state="active",
    )
    db_session.add_all([path_chunk, hit_chunk])
    db_session.flush()
    trace = RetrievalTrace(
        knowledge_base_id=sample_knowledge_base.id,
        query="\u914d\u7f6e\u6a21\u578b\u7b97\u6cd5",
        filters_json={},
        result_chunk_ids_json=[hit_chunk.id],
        query_facets_json={"required_facets": ["\u914d\u7f6e\u6a21\u578b"]},
        path_labels_json=[{"path": [path_chunk.id, hit_chunk.id], "path_edge_ids": ["edge-1"]}],
        convergence_json={
            "gray_zone_decision_count": 0,
            "gray_zone_rule_evaluation_count": 0,
            "red_zone_pruned_count": 0,
            "hard_stop_pruned_count": 0,
            "gray_zone_model_call_count": 0,
        },
    )
    db_session.add(trace)
    db_session.flush()

    package = build_context_package(
        db_session,
        knowledge_base_id=sample_knowledge_base.id,
        query=trace.query,
        trace=trace,
        results=[
            {
                "chunk_id": hit_chunk.id,
                "metadata": {
                    "traversal": {
                        "path": [path_chunk.id, hit_chunk.id],
                        "path_edge_ids": ["edge-1"],
                        "covered_facets": ["\u914d\u7f6e\u6a21\u578b"],
                        "evidence_roles": ["dense_semantic"],
                        "why_selected": "accepted_by_priority_queue_graph_traversal",
                    }
                },
            }
        ],
        token_budget=1000,
    )

    chunks = (package.package_json or {}).get("chunks") or []
    contexts = context_package_to_contexts(package)
    assert path_chunk.id in package.restored_chunk_ids_json
    assert any(item.get("chunk_id") == path_chunk.id and item.get("role") == "graph_path" for item in chunks)
    hit_context = next(item for item in contexts if item.get("chunk_id") == hit_chunk.id)
    assert hit_context["source_path"] == str(snapshot_path)
    assert hit_context["metadata"]["source_path"] == str(snapshot_path)
    assert hit_context["metadata"]["logical_source_path"] == str(logical_source)
    assert hit_context["metadata"]["page_range"] == [1, 1]
    assert hit_context["metadata"]["char_span"] == [81, 160]
    assert package.why_selected_json[path_chunk.id]["reason"] == "restored_from_selected_graph_path"


def test_concept_searchable_text_uses_successful_i18n_only():
    from app.services.context_graph import concept_searchable_text

    concept = SimpleNamespace(
        canonical_label="Bayesian Regression",
        definition="Regression with Bayesian posterior inference.",
        summary="Bayesian regression summary.",
        scope_note="Course-level model concept.",
        aliases_json=["Bayesian linear model"],
        display_terms_json=[],
        llm_audit_json={
            "concept_i18n": {
                "status": "ok",
                "label_i18n": {"zh": "贝叶斯回归", "en": "Bayesian Regression"},
                "definition_i18n": {"zh": "使用后验推断的回归模型。", "en": "Regression with posterior inference."},
                "summary_i18n": {"zh": "贝叶斯回归摘要。", "en": "Bayesian regression summary."},
                "scope_note_i18n": {"zh": "课程模型概念。", "en": "Course model concept."},
                "aliases_i18n": {"zh": ["贝叶斯线性模型"], "en": ["Bayesian linear model"]},
                "search_terms_i18n": {"zh": ["后验", "回归"], "en": ["posterior", "regression"]},
            }
        },
    )

    disabled_searchable = concept_searchable_text(concept, include_i18n=False)
    assert "贝叶斯回归" not in disabled_searchable
    assert "Bayesian Regression" in disabled_searchable

    searchable = concept_searchable_text(concept, include_i18n=True)
    assert "贝叶斯回归" in searchable
    assert "后验" in searchable

    concept.llm_audit_json["concept_i18n"]["status"] = "original_text_fallback"
    fallback_searchable = concept_searchable_text(concept, include_i18n=True)
    assert "贝叶斯回归" not in fallback_searchable
    assert "Bayesian Regression" in fallback_searchable


@pytest.mark.asyncio
async def test_mid_concept_definition_reads_system_prompt_from_profile(monkeypatch):
    from app.services import context_graph
    from app.services.strategy_profiles import default_profile_payload, use_strategy_profile

    captured: dict[str, str] = {}

    class ProfileGraphChatProvider:
        def __init__(self, purpose: str | None = None):
            self.purpose = purpose

        async def classify_json(self, system_prompt: str, user_prompt: str, fallback: dict | None = None) -> dict:
            captured["system_prompt"] = system_prompt
            return fallback or {"concepts": []}

    profile = default_profile_payload()
    profile["prompt_pack"]["mid_concept_definition_system"] = "Profile-specific mid concept prompt."
    packet = {
        "packet_id": "packet-1",
        "candidate_label": "Factorization",
        "candidate_labels": ["Factorization"],
        "representative_chunk_ids": ["chunk-1"],
        "support_chunk_ids": ["chunk-1"],
        "support_chunk_count": 1,
        "canonical_membership_fact_hashes": ["2" * 64],
        "canonical_membership_facts_hash": "3" * 64,
        "structure_paths": [],
        "structure_path_count": 0,
        "structure_path_trace_sample_count": 0,
        "structure_path_trace_sample_limit": 64,
        "structure_paths_complete": True,
        "structure_paths_hash": "4" * 64,
        "structure_mapping_address_facts_hash": "5" * 64,
        "structure_mapping_chunk_scope_business_hash": "6" * 64,
        "structure_mapping_fact_set_protocol_version": (
            context_graph.STRUCTURE_MAPPING_FACT_SET_PROTOCOL_VERSION
        ),
        "source_spans": [
            {
                "chunk_id": "chunk-1",
                "char_span": [0, 19],
                "page_range": [1, 1],
                "section_path": "Factorization",
            }
        ],
        "source_spans_business_facts_hash": "7" * 64,
        "chunk_excerpts": [
            {
                "chunk_id": "chunk-1",
                "section_path": "Factorization",
                "page_range": [1, 1],
                "text": "factorized evidence",
                "full_text_hash": "8" * 64,
                "full_text_length_chars": 19,
            }
        ],
    }

    monkeypatch.setattr(context_graph, "ChatProvider", ProfileGraphChatProvider)

    with use_strategy_profile(profile):
        concepts = await context_graph.define_mid_concepts_batch([packet])

    assert captured["system_prompt"].startswith(
        "Profile-specific mid concept prompt.\n\n"
    )
    assert context_graph.MID_CONCEPT_PROVIDER_OUTPUT_CONTRACT in captured[
        "system_prompt"
    ]
    assert concepts
    assert concepts[0]["packet_id"] == "packet-1"


@pytest.mark.asyncio
async def test_rebuild_context_graph_uses_bound_profile_for_concept_prompts(
    monkeypatch,
    db_session,
    populated_context_graph,
    fake_profile_lifecycle_side_effects,
):
    from app.models import MidConceptState
    from app.services import context_graph
    from app.services.strategy_profiles import bind_profile_to_knowledge_base, create_profile, default_profile_payload
    from sqlalchemy import select

    kb = populated_context_graph["knowledge_base"]
    captured: list[str] = []

    class ProfileGraphChatProvider:
        def __init__(self, purpose: str | None = None):
            self.purpose = purpose

        async def classify_json(self, system_prompt: str, user_prompt: str, fallback: dict | None = None) -> dict:
            captured.append(system_prompt)
            return fallback or {"concepts": []}

    profile_payload = default_profile_payload()
    profile_payload["prompt_pack"]["mid_concept_definition_system"] = "Profile-bound rebuild mid prompt."
    profile, warnings = create_profile(
        db_session,
        name="Profile-bound rebuild prompt",
        library_type="custom",
        profile_json=profile_payload,
    )
    assert warnings == []
    bind_profile_to_knowledge_base(db_session, knowledge_base_id=kb.id, profile_id=profile.id)
    monkeypatch.setattr(context_graph, "ChatProvider", ProfileGraphChatProvider)

    await context_graph.rebuild_context_graph(db_session, kb.id, emit_heartbeats=False)

    assert any(
        system_prompt.startswith("Profile-bound rebuild mid prompt.\n\n")
        and context_graph.MID_CONCEPT_PROVIDER_OUTPUT_CONTRACT
        in system_prompt
        for system_prompt in captured
    )
    mid_state = db_session.scalar(
        select(MidConceptState)
        .where(MidConceptState.knowledge_base_id == kb.id, MidConceptState.state == "active")
        .order_by(MidConceptState.created_at.desc())
    )
    assert mid_state is not None
    assert (mid_state.diagnostics_json or {}).get(
        "profile_hash"
    ) == context_graph.canonical_active_profile_state_hash(db_session, kb.id)


def test_graph_overview_reuses_request_scoped_contextual_hash_replay(
    db_session,
    populated_context_graph,
    monkeypatch: pytest.MonkeyPatch,
):
    from app.services import context_graph

    kb = populated_context_graph["knowledge_base"]
    original = context_graph.contextual_index_state_hashes
    calls: list[str] = []

    def counted(*args, **kwargs):
        calls.append(str(args[1]))
        return original(*args, **kwargs)

    monkeypatch.setattr(
        context_graph,
        "contextual_index_state_hashes",
        counted,
    )

    payload = context_graph.graph_layer_payload(
        db_session,
        kb.id,
        "mid-concepts",
        limit=2,
    )

    assert payload["freshness"]["is_admissible"] is True
    assert calls == [str(kb.id)]


def test_active_admission_rejects_nonactive_concept_child(
    db_session,
    populated_context_graph,
):
    from sqlalchemy import select

    from app.models import MidConcept
    from app.services import context_graph

    kb = populated_context_graph["knowledge_base"]
    mid = db_session.scalar(
        select(MidConcept).where(
            MidConcept.knowledge_base_id == kb.id,
            MidConcept.state == "active",
        )
    )
    assert mid is not None
    mid.state = "shadow"
    db_session.flush()

    with pytest.raises(
        context_graph.ActiveContextGraphAdmissionError
    ) as captured:
        context_graph.active_graph_admission_gate(db_session, kb.id)

    assert any(
        reason.startswith("active_mid_concept_state_invalid:")
        for reason in captured.value.reasons
    )


@pytest.mark.asyncio
async def test_context_graph_pipeline_builds_all_layers(db_session, populated_context_graph):
    from sqlalchemy import func, select

    from app.models import (
        Chunk,
        ChunkContextText,
        ChunkRelationEdge,
        ChunkRelationGraphState,
        ChunkStructureMapping,
        ChunkStructureNode,
        CoarseConcept,
        CoarseConceptState,
        ContextGraphState,
        RQPrefix,
        RQPrefixMembership,
        MidConcept,
        MidConceptState,
        VectorRecord,
    )
    from app.services.context_graph import graph_layer_payload

    kb = populated_context_graph["knowledge_base"]
    assert db_session.scalar(select(func.count(Chunk.id)).where(Chunk.knowledge_base_id == kb.id, Chunk.state == "active")) >= 3
    assert db_session.scalar(select(func.count(ChunkStructureNode.id)).where(ChunkStructureNode.knowledge_base_id == kb.id)) >= 3
    assert db_session.scalar(select(func.count(ChunkStructureMapping.id))) >= 3
    assert db_session.scalar(select(func.count(ChunkContextText.id))) >= 3
    assert db_session.scalar(select(func.count(VectorRecord.id)).where(VectorRecord.knowledge_base_id == kb.id, VectorRecord.vector_status == "ready")) >= 3
    assert db_session.scalar(select(func.count(ChunkRelationEdge.id)).where(ChunkRelationEdge.knowledge_base_id == kb.id)) >= 2
    assert db_session.scalar(select(func.count(RQPrefix.id)).where(RQPrefix.knowledge_base_id == kb.id)) >= 1
    chunks_with_rq = db_session.scalars(select(Chunk).where(Chunk.knowledge_base_id == kb.id, Chunk.rq_residual_norm.is_not(None))).all()
    assert chunks_with_rq
    assert all(chunk.rq_path for chunk in chunks_with_rq)
    rq_prefixes = db_session.scalars(select(RQPrefix).where(RQPrefix.knowledge_base_id == kb.id, RQPrefix.rq_level.is_not(None))).all()
    assert rq_prefixes
    rq_prefix_memberships = [
        row
        for row in db_session.scalars(select(RQPrefixMembership).where(RQPrefixMembership.residual_norm.is_not(None))).all()
        if row.rq_path
    ]
    assert rq_prefix_memberships
    assert all(row.rq_path for row in rq_prefix_memberships)
    relation_state = db_session.scalar(select(ChunkRelationGraphState).where(ChunkRelationGraphState.knowledge_base_id == kb.id, ChunkRelationGraphState.state == "active"))
    assert relation_state is not None
    assert relation_state.graph_operating_point_hash
    assert relation_state.edge_type_calibration_protocol_hash
    assert (relation_state.diagnostics_json or {}).get("rq_pair_edges_active") is False
    relation_edges = db_session.scalars(select(ChunkRelationEdge).where(ChunkRelationEdge.knowledge_base_id == kb.id)).all()
    assert relation_edges
    assert {edge.edge_type for edge in relation_edges}.issubset({"dense_semantic", "dense_cross_document_bridge", "dense_cross_language_bridge"})
    assert (relation_state.diagnostics_json or {}).get("accepted_edge_types") == dict(Counter(edge.edge_type for edge in relation_edges))
    assert not any(edge.edge_type.startswith("rq_") for edge in relation_edges)
    assert all(edge.source_algorithm == "dense_embedding" for edge in relation_edges)
    assert all(edge.protocol_version and edge.graph_state_hash and edge.edge_distance_protocol_hash for edge in relation_edges)
    assert all(edge.distance is not None and edge.raw_strength is not None for edge in relation_edges)
    assert all(edge.is_cross_document is not None and edge.is_cross_language is not None for edge in relation_edges)
    assert all(prefix.parent_rq_prefix_id or int(prefix.rq_level or 0) == 1 for prefix in rq_prefixes)
    graph_payload = graph_layer_payload(db_session, kb.id, "chunk-relation")
    assert graph_payload["full_counts"]["nodes"] >= graph_payload["sampled_counts"]["nodes"]
    assert graph_payload["full_counts"]["edges"] >= graph_payload["sampled_counts"]["edges"]
    assert graph_payload["edge_counts"]["full"] == graph_payload["full_counts"]["edges"]
    # mid_grounded_rate is a real claim-level SummaryGrounded(m)
    # audit, not "a concept exists with non-empty support".  The provider's
    # ungrounded generic candidate is retained in the audit, while the
    # persisted fallback is an exact deterministic support span and is
    # re-audited before write, so every active Mid summary is grounded.
    assert graph_payload["grounding"]["mid_grounded_rate"] == 1.0
    assert graph_payload["grounding"]["coarse_grounded_rate"] == 1.0
    assert any(node.get("category") == "rq_prefix" for node in graph_payload["nodes"])
    assert any(edge.get("category") == "rq_membership" for edge in graph_payload["edges"])
    rq_non_membership_edges = [
        edge
        for edge in graph_payload["edges"]
        if str(edge.get("category", "")).startswith("rq_")
        and edge.get("category") != "rq_membership"
    ]
    assert rq_non_membership_edges
    assert all(
        edge.get("type") == "rq_prefix_pair_diagnostic"
        and (edge.get("metadata") or {}).get("diagnostic_only") is True
        and (edge.get("metadata") or {}).get("active_relation_edge") is False
        for edge in rq_non_membership_edges
    )
    l3_prefix_count = db_session.scalar(
        select(func.count(RQPrefix.id)).where(RQPrefix.knowledge_base_id == kb.id, RQPrefix.rq_level == 3, RQPrefix.state == "active")
    )
    l2_prefix_count = db_session.scalar(
        select(func.count(RQPrefix.id)).where(RQPrefix.knowledge_base_id == kb.id, RQPrefix.rq_level == 2, RQPrefix.state == "active")
    )
    mid_count = db_session.scalar(select(func.count(MidConcept.id)).where(MidConcept.knowledge_base_id == kb.id, MidConcept.state == "active"))
    coarse_count = db_session.scalar(select(func.count(CoarseConcept.id)).where(CoarseConcept.knowledge_base_id == kb.id, CoarseConcept.state == "active"))
    active_chunk_count = db_session.scalar(
        select(func.count(Chunk.id)).where(
            Chunk.knowledge_base_id == kb.id,
            Chunk.state == "active",
        )
    )
    assert 0 < mid_count <= active_chunk_count
    assert 0 < coarse_count <= mid_count
    if active_chunk_count > 1:
        assert mid_count < active_chunk_count
    if mid_count > 1:
        assert coarse_count < mid_count
    mid_state = db_session.scalar(select(MidConceptState).where(MidConceptState.knowledge_base_id == kb.id, MidConceptState.state == "active"))
    assert mid_state is not None
    assert (mid_state.stats_json or {})["projected_rq_l3_prefixes"] == mid_count
    assert (mid_state.stats_json or {})["rq_leaf_prefix_candidates"] == l3_prefix_count
    assert (mid_state.stats_json or {})[
        "semantic_compression_cardinality_passed"
    ] is True
    assert (mid_state.stats_json or {})["concept_node_eligibility"][
        "model_call_count"
    ] == 0
    mid_projection = (mid_state.stats_json or {})["projection_diagnostics"]
    assert mid_projection["full_edge_count"] == (mid_state.stats_json or {})[
        "mid_edge_count"
    ]
    assert mid_projection["protocol_hash_consistent"] is True
    assert mid_projection["protocol_hash_coverage"] == 1.0
    assert sum(
        item["full_edge_count"] for item in mid_projection["by_edge_type"].values()
    ) == mid_projection["full_edge_count"]
    coarse_state = db_session.scalar(select(CoarseConceptState).where(CoarseConceptState.knowledge_base_id == kb.id, CoarseConceptState.state == "active"))
    assert coarse_state is not None
    assert (coarse_state.stats_json or {})["projected_rq_l2_prefixes"] == coarse_count
    assert (coarse_state.stats_json or {})[
        "rq_l2_prefix_candidates"
    ] == l2_prefix_count
    assert (coarse_state.stats_json or {})[
        "semantic_compression_cardinality_passed"
    ] is True
    assert (coarse_state.stats_json or {})["concept_node_eligibility"][
        "model_call_count"
    ] == 0
    coarse_projection = (coarse_state.stats_json or {})["projection_diagnostics"]
    assert coarse_projection["full_edge_count"] == (coarse_state.stats_json or {})[
        "coarse_edge_count"
    ]
    assert coarse_projection["protocol_hash_consistent"] is True
    assert coarse_projection["protocol_hash_coverage"] == 1.0
    assert sum(
        item["full_edge_count"]
        for item in coarse_projection["by_edge_type"].values()
    ) == coarse_projection["full_edge_count"]
    mid_audit = [concept.llm_audit_json or {} for concept in db_session.scalars(select(MidConcept).where(MidConcept.knowledge_base_id == kb.id)).all()]
    coarse_audit = [concept.llm_audit_json or {} for concept in db_session.scalars(select(CoarseConcept).where(CoarseConcept.knowledge_base_id == kb.id)).all()]
    assert "rq_path" not in json.dumps(mid_audit + coarse_audit, sort_keys=True)
    state = db_session.scalar(select(ContextGraphState).where(ContextGraphState.knowledge_base_id == kb.id, ContextGraphState.state == "active"))
    assert state is not None
    assert state.chunk_relation_graph_hash
    assert state.mid_concept_hash
    assert state.coarse_concept_hash


def test_graph_build_materializes_gray_predicates_from_relation_and_rq_evidence(
    db_session,
    populated_context_graph,
):
    from sqlalchemy import select

    from app.models import ChunkRelationEdge, CoarseConceptEdge, MidConceptEdge
    from app.services import context_graph

    kb = populated_context_graph["knowledge_base"]
    relation_state = context_graph.latest_relation_state(db_session, kb.id)
    assert relation_state is not None
    edges = list(
        db_session.scalars(
            select(ChunkRelationEdge).where(
                ChunkRelationEdge.graph_state_id == relation_state.id
            )
        ).all()
    )
    assert edges
    for edge in edges:
        features = edge.features_json or {}
        boundary = features.get("rq_boundary_audit") or {}
        assert isinstance(features.get("semantic_uncertain"), bool)
        assert features["semantic_uncertainty_protocol_hash"] == context_graph.edge_semantic_uncertainty_protocol_hash()
        assert isinstance(features.get("crossing_rq_boundary"), bool)
        assert features["rq_boundary_protocol_hash"] == context_graph.rq_boundary_protocol_hash()
        assert boundary["model_call_count"] == 0
        assert boundary["crossing_rq_boundary"] == (
            boundary["source_rq_path"] != boundary["target_rq_path"]
        )
    assert any(
        (edge.features_json or {}).get("semantic_uncertain")
        for edge in edges
    )
    assert any(
        (edge.features_json or {}).get("crossing_rq_boundary")
        for edge in edges
    )
    gray_stats = (relation_state.diagnostics_json or {})["gray_predicates"]
    assert gray_stats["edge_count"] == len(edges)
    assert gray_stats["model_call_count"] == 0

    projected_edges = [
        *db_session.scalars(select(MidConceptEdge)).all(),
        *db_session.scalars(select(CoarseConceptEdge)).all(),
    ]
    assert projected_edges
    for edge in projected_edges:
        diagnostics = edge.diagnostics_json or {}
        projected = diagnostics.get("gray_predicates") or {}
        assert isinstance(diagnostics.get("semantic_uncertain"), bool)
        assert isinstance(diagnostics.get("crossing_rq_boundary"), bool)
        assert projected["protocol_hash"] == context_graph.projected_gray_predicate_protocol_hash()
        assert projected["model_call_count"] == 0


def test_chunk_walk_routes_built_green_uncertain_and_crossing_rq_edges_through_local_rule(
    db_session,
    populated_context_graph,
    monkeypatch,
):
    from sqlalchemy import select

    from app.models import Chunk, ChunkRelationEdge, MidConcept
    from app.schemas import SearchFilters
    from app.services import context_graph

    class ForbiddenChatProvider:
        def __init__(self, *args, **kwargs):
            raise AssertionError("chunk gray-zone traversal must not construct a model provider")

    monkeypatch.setattr(context_graph, "ChatProvider", ForbiddenChatProvider)
    kb = populated_context_graph["knowledge_base"]
    chunks = list(
        db_session.scalars(
            select(Chunk).where(
                Chunk.knowledge_base_id == kb.id,
                Chunk.state == "active",
            )
        ).all()
    )
    relation_state = context_graph.latest_relation_state(db_session, kb.id)
    assert relation_state is not None
    relation_edges = list(
        db_session.scalars(
            select(ChunkRelationEdge).where(ChunkRelationEdge.graph_state_id == relation_state.id)
        ).all()
    )
    mid = db_session.scalar(
        select(MidConcept).where(
            MidConcept.knowledge_base_id == kb.id,
            MidConcept.state == "active",
        )
    )
    assert mid is not None
    mid.support_rq_prefix_ids_json = []

    for feature_key, expected_reason in (
        ("semantic_uncertain", "semantic_uncertain"),
        ("crossing_rq_boundary", "crossing_rq_boundary"),
    ):
        edge = next(
            (
                candidate
                for candidate in relation_edges
                if bool((candidate.features_json or {}).get(feature_key))
            ),
            None,
        )
        assert edge is not None, f"production graph did not materialize {feature_key}"
        seed_id = edge.source_chunk_id
        mid.support_chunk_ids_json = [seed_id]
        green_threshold = min(20.0, float(edge.distance or 0.0) + 1.0)
        envelope = {
            **context_graph.agent_operating_envelope(),
            "agent_mid_initial_budget": 1,
            "agent_mid_top_k": 1,
            "agent_chunk_initial_budget": 1,
            "agent_chunk_top_k": len(chunks) + 5,
            "path_distance_green_threshold": green_threshold,
            "path_distance_gray_threshold": green_threshold,
            "path_distance_hard_threshold": max(green_threshold, 20.0),
        }
        traversal = context_graph.execute_priority_queue_traversal(
            db_session,
            knowledge_base_id=kb.id,
            query="gray-zone deterministic producer regression",
            chunks=chunks,
            filters=SearchFilters(),
            query_facets={"required_facets": []},
            coarse_entries={},
            mid_entries={mid.id: 1.0},
            rq_membership_entries={},
            dense_entries={seed_id: 1.0},
            query_rq=None,
            top_k=len(chunks) + 5,
            retrieval_granularity="mid",
            envelope=envelope,
        )
        records = [
            record
            for record in traversal["gray_zone_path_decisions"]
            if record.get("layer") == "chunk"
            and record.get("edge_id") == edge.id
            and record.get("decision_source") == "deterministic_local_rule"
        ]
        assert records
        assert records[0]["distance_zone"] == "green"
        assert expected_reason in records[0]["gray_candidate_reasons"]
        assert records[0]["model_call_count"] == 0
        assert records[0]["matched_rule"]
        assert records[0]["protocol_hash"] == context_graph.gray_zone_rule_protocol_hash()
        assert traversal["convergence"]["gray_zone_model_call_count"] == 0


def test_traversal_producer_ignores_external_llm_facets_for_every_gray_decision(
    db_session,
    populated_context_graph,
):
    """The real staged producer must never feed its routing packet to gray rules."""

    from sqlalchemy import select

    from app.models import Chunk, ChunkRelationEdge, MidConcept
    from app.schemas import SearchFilters
    from app.services import context_graph

    kb = populated_context_graph["knowledge_base"]
    chunks = list(
        db_session.scalars(
            select(Chunk).where(
                Chunk.knowledge_base_id == kb.id,
                Chunk.state == "active",
            )
        ).all()
    )
    relation_state = context_graph.latest_relation_state(db_session, kb.id)
    assert relation_state is not None
    relation_edges = list(
        db_session.scalars(
            select(ChunkRelationEdge).where(
                ChunkRelationEdge.graph_state_id == relation_state.id
            )
        ).all()
    )
    assert relation_edges
    edge = relation_edges[0]
    target_chunk = next(
        chunk for chunk in chunks if chunk.id == edge.target_chunk_id
    )
    matching_facet = next(
        term
        for term in context_graph.tokenize_for_search_terms(target_chunk.text)
        if len(term.strip()) >= 3
    )
    mid = db_session.scalar(
        select(MidConcept).where(
            MidConcept.knowledge_base_id == kb.id,
            MidConcept.state == "active",
        )
    )
    assert mid is not None
    mid.support_rq_prefix_ids_json = []
    mid.support_chunk_ids_json = [edge.source_chunk_id]

    query = "alpha concept"
    external_packets = [
        context_graph.query_facets_for_search(
            query,
            {
                "facet_groups": [
                    {"facet": facet, "role": "domain", "aliases": []}
                ],
                "answer_shape": "grounded answer",
                "drop_terms": [],
            },
        )
        for facet in (matching_facet, "unrelated-provider-facet")
    ]
    assert external_packets[0]["required_facets"] != external_packets[1][
        "required_facets"
    ]

    envelope = {
        **context_graph.agent_operating_envelope(),
        "agent_mid_initial_budget": 1,
        "agent_mid_top_k": 1,
        "agent_chunk_initial_budget": 1,
        "agent_chunk_top_k": len(chunks) + 5,
        "path_distance_green_threshold": 0.0,
        "path_distance_gray_threshold": 20.0,
        "path_distance_hard_threshold": 20.0,
    }

    def run(external_packet):
        return context_graph.execute_priority_queue_traversal(
            db_session,
            knowledge_base_id=kb.id,
            query=query,
            chunks=chunks,
            filters=SearchFilters(),
            query_facets=external_packet,
            coarse_entries={},
            mid_entries={mid.id: 1.0},
            rq_membership_entries={},
            dense_entries={edge.source_chunk_id: 1.0},
            query_rq=None,
            top_k=len(chunks) + 5,
            retrieval_granularity="mid",
            envelope=envelope,
        )

    traversals = [run(packet) for packet in external_packets]
    records = []
    for traversal in traversals:
        assert traversal["query_facets"] in external_packets
        authority = traversal["gray_zone_authority_audit"]
        assert authority["external_routing_packet_used"] is False
        assert authority["request_scoped_budget_used_by_gray_identity"] is False
        assert authority["gray_zone_model_call_count"] == 0
        records.append(
            [
                {
                    key: record.get(key)
                    for key in (
                        "layer",
                        "edge_id",
                        "from_node_id",
                        "to_node_id",
                        "input_hash",
                        "matched_rule",
                        "decision",
                        "decision_hash",
                        "model_call_count",
                    )
                }
                for record in traversal["gray_zone_path_decisions"]
            ]
        )

    assert records[0]
    decision_maps = [
        {
            record["input_hash"]: {
                "matched_rule": record["matched_rule"],
                "decision": record["decision"],
                "decision_hash": record["decision_hash"],
                "model_call_count": record["model_call_count"],
            }
            for record in run_records
        }
        for run_records in records
    ]
    shared_inputs = set(decision_maps[0]).intersection(decision_maps[1])
    assert shared_inputs
    assert {
        input_hash: decision_maps[0][input_hash]
        for input_hash in sorted(shared_inputs)
    } == {
        input_hash: decision_maps[1][input_hash]
        for input_hash in sorted(shared_inputs)
    }
    assert all(
        record["model_call_count"] == 0
        for run_records in records
        for record in run_records
    )
    expected_gray_hash = context_graph.stable_hash(
        context_graph.deterministic_gray_query_facets_for_search(query)
    )
    assert {
        traversal["gray_zone_authority_audit"]["gray_zone_query_facet_hash"]
        for traversal in traversals
    } == {expected_gray_hash}


@pytest.mark.asyncio
async def test_layered_retrieval_writes_trace_and_context_package(
    monkeypatch,
    db_session,
    populated_context_graph,
    fake_profile_lifecycle_side_effects,
):
    from sqlalchemy import func, select

    from app.models import Chunk, ContextGraphState, ContextPackage, GraphRetrievalStep, RetrievalTrace
    from app.schemas import SearchFilters
    from app.services.context_graph import build_context_package, layered_search
    from app.services.retrieval import get_context_package
    from app.services.strategy_profiles import bind_profile_to_knowledge_base, create_profile, default_profile_payload

    kb = populated_context_graph["knowledge_base"]
    state_before = db_session.scalar(select(ContextGraphState.id).where(ContextGraphState.knowledge_base_id == kb.id, ContextGraphState.state == "active"))
    chunk_count_before = db_session.scalar(select(func.count(Chunk.id)).where(Chunk.knowledge_base_id == kb.id, Chunk.state == "active"))
    profile_payload = default_profile_payload()
    profile_payload["prompt_pack"]["answer_system_prefix"] = "Context package audit profile."
    profile, warnings = create_profile(
        db_session,
        name="Context package audit profile",
        library_type="custom",
        profile_json=profile_payload,
    )
    assert warnings == []
    bind_profile_to_knowledge_base(db_session, knowledge_base_id=kb.id, profile_id=profile.id)
    result = await layered_search(db_session, kb.id, "How does a Markov blanket affect conditional independence?", SearchFilters(), 4)
    assert result.results
    assert result.audit["retrieval_pipeline"] == "layered_context_graph"
    assert "query_rq_path" in result.audit
    assert result.audit["stage_queue_count"] >= 0
    assert result.audit["mid_topk_selected"] >= 0
    assert result.audit["chunk_topk_selected"] >= 0
    assert result.audit["dominance_pruned_count"] >= 0
    assert result.audit["hard_stop_pruned_count"] >= 0
    assert result.audit["gray_zone_decision_count"] == result.trace.convergence_json.get(
        "gray_zone_rule_evaluation_count", 0
    )
    assert result.trace.convergence_json.get("red_zone_pruned_count", 0) >= 0
    assert result.trace.convergence_json.get("hard_stop_pruned_count", 0) >= 0
    assert any((item.get("metadata") or {}).get("rq") for item in result.results)
    for item in result.results:
        traversal = (item.get("metadata") or {}).get("traversal") or {}
        assert traversal.get("distance_so_far") is not None
        assert float(traversal["distance_so_far"]) >= 0.0
        assert item["score"] <= 1.0
        if not traversal.get("path_edge_ids"):
            assert float(traversal["distance_so_far"]) > 0.0
            assert (item.get("metadata") or {}).get("entry_strengths")
    assert result.trace.id
    assert (result.trace.diagnostics_json or {}).get("active_profile_hash") == profile.profile_hash
    assert ((result.trace.diagnostics_json or {}).get("cache_key_components") or {}).get("profile_hash") == profile.profile_hash
    assert result.trace.stage_queues_json
    assert result.trace.candidate_pools_json
    assert result.trace.topk_selection_json
    assert db_session.scalar(select(func.count(GraphRetrievalStep.id)).where(GraphRetrievalStep.retrieval_trace_id == result.trace.id)) >= 4
    assert (
        db_session.scalar(select(func.count(GraphRetrievalStep.id)).where(GraphRetrievalStep.retrieval_trace_id == result.trace.id, GraphRetrievalStep.layer == "fine"))
        == 0
    )
    seed_step = db_session.scalar(
        select(GraphRetrievalStep).where(GraphRetrievalStep.retrieval_trace_id == result.trace.id, GraphRetrievalStep.layer == "fine")
    )
    assert seed_step is None
    seed_step = db_session.scalar(
        select(GraphRetrievalStep).where(
            GraphRetrievalStep.retrieval_trace_id == result.trace.id,
            GraphRetrievalStep.layer == "chunk",
            GraphRetrievalStep.action == "select_seeds_from_mid_rq_membership",
        )
    )
    assert seed_step is not None
    assert seed_step.action_type == "select_seeds_from_mid_rq_membership"
    assert seed_step.selected_topk_ids_json
    assert "query_rq_path" in (seed_step.input_json or {})
    assert (seed_step.output_json or {}).get("candidate_count") is not None
    assert result.snapshot_verifier is not None
    verification_count_after_search = result.snapshot_verifier.verification_count
    package = build_context_package(
        db_session,
        knowledge_base_id=kb.id,
        query="Markov blanket",
        trace=result.trace,
        results=result.results,
        snapshot_verifier=result.snapshot_verifier,
    )
    assert result.snapshot_verifier.verification_count == verification_count_after_search
    db_session.commit()
    structure_step = db_session.scalar(
        select(GraphRetrievalStep).where(GraphRetrievalStep.retrieval_trace_id == result.trace.id, GraphRetrievalStep.layer == "structure")
    )
    assert structure_step is not None
    assert structure_step.action_type == "restore_context_package"
    assert (structure_step.output_json or {}).get("context_package_id") == package.id
    assert db_session.get(RetrievalTrace, result.trace.id) is not None
    assert db_session.get(ContextPackage, package.id) is not None
    assert package.hit_chunk_ids_json
    assert package.restored_chunk_ids_json
    assert package.parent_structure_node_ids_json
    assert package.citation_spans_json
    assert package.profile_hash == profile.profile_hash
    assert (package.diagnostics_json or {}).get("profile_hash") == profile.profile_hash
    assert db_session.scalar(select(ContextGraphState.id).where(ContextGraphState.knowledge_base_id == kb.id, ContextGraphState.state == "active")) == state_before
    assert db_session.scalar(select(func.count(Chunk.id)).where(Chunk.knowledge_base_id == kb.id, Chunk.state == "active")) == chunk_count_before
    chunks = (package.package_json or {}).get("chunks") or []
    assert any(item.get("structure_path") for item in chunks)
    assert any(item.get("structure_nodes") for item in chunks)
    assert any(item.get("structure_path") for item in package.citation_spans_json)
    assert package.token_budget > 0
    package_payload = get_context_package(db_session, package.id)
    assert package_payload is not None
    assert package_payload["package_hash"]
    assert package_payload["contexts"]
    assert package_payload["citation_spans"]
    assert package_payload["graph_expansion_paths"]


@pytest.mark.asyncio
async def test_layered_retrieval_granularity_mid_direct_and_coarse_regression(db_session, populated_context_graph):
    from sqlalchemy import select

    from app.models import GraphRetrievalStep
    from app.schemas import SearchFilters
    from app.services import agent_graph
    from app.services.citation_provenance import audit_citation_provenance
    from app.services.context_graph import (
        build_context_package,
        context_package_to_contexts,
        layered_search,
    )

    kb = populated_context_graph["knowledge_base"]

    mid_result = await layered_search(
        db_session,
        kb.id,
        "What is a Markov blanket?",
        SearchFilters(),
        4,
        retrieval_granularity="mid",
    )

    assert mid_result.results
    assert mid_result.audit["retrieval_granularity"] == "mid"
    assert mid_result.audit["coarse_entries"] == 0
    assert mid_result.trace.diagnostics_json["retrieval_granularity"] == "mid"
    assert mid_result.trace.diagnostics_json["cache_key_components"]["retrieval_granularity"] == "mid"
    assert mid_result.trace.stage_queues_json["coarse"]["skipped_by_granularity"] == "mid"
    assert mid_result.trace.candidate_pools_json["mid_direct_entries"]["selected_ids"]
    assert mid_result.trace.topk_selection_json["mid"]["entry_mode"] == "mid"
    mid_step_actions = {
        (step.layer, step.action_type)
        for step in db_session.scalars(select(GraphRetrievalStep).where(GraphRetrievalStep.retrieval_trace_id == mid_result.trace.id)).all()
    }
    assert ("coarse", "select_entry_nodes") in mid_step_actions
    assert ("coarse", "staged_priority_queue_walk") in mid_step_actions
    assert ("mid", "drill_down_each_coarse_or_direct_mid_entry") in mid_step_actions

    mid_package = build_context_package(
        db_session,
        knowledge_base_id=kb.id,
        query="What is a Markov blanket?",
        trace=mid_result.trace,
        results=mid_result.results,
    )
    assert (mid_package.diagnostics_json or {})["retrieval_granularity"] == "mid"

    coarse_result = await layered_search(
        db_session,
        kb.id,
        "What is a Markov blanket?",
        SearchFilters(),
        4,
        retrieval_granularity="coarse",
    )

    assert coarse_result.results
    assert coarse_result.audit["retrieval_granularity"] == "coarse"
    assert coarse_result.audit["coarse_entries"] >= 0
    assert coarse_result.trace.diagnostics_json["retrieval_granularity"] == "coarse"
    assert coarse_result.trace.diagnostics_json["cache_key_components"]["retrieval_granularity"] == "coarse"
    assert "skipped_by_granularity" not in (coarse_result.trace.stage_queues_json.get("coarse") or {})
    assert coarse_result.trace.topk_selection_json["mid"]["entry_mode"] == "coarse"
    coarse_step_actions = {
        (step.layer, step.action_type)
        for step in db_session.scalars(select(GraphRetrievalStep).where(GraphRetrievalStep.retrieval_trace_id == coarse_result.trace.id)).all()
    }
    assert ("coarse", "select_entry_nodes") in coarse_step_actions
    assert ("coarse", "staged_priority_queue_walk") in coarse_step_actions
    assert ("mid", "drill_down_each_coarse_or_direct_mid_entry") in coarse_step_actions
    assert mid_result.trace.diagnostics_json["cache_key"] != coarse_result.trace.diagnostics_json["cache_key"]
    coarse_steps = list(
        db_session.scalars(
            select(GraphRetrievalStep)
            .where(
                GraphRetrievalStep.retrieval_trace_id
                == coarse_result.trace.id
            )
            .order_by(GraphRetrievalStep.step_index.asc())
        ).all()
    )
    coarse_entry_step = next(
        step
        for step in coarse_steps
        if step.layer == "coarse"
        and step.action_type == "select_entry_nodes"
    )
    assert (coarse_entry_step.diagnostics_json or {}).get("path_labels") == []
    executor_path_labels = [
        label
        for step in coarse_steps
        if step.layer != "structure"
        for label in (step.diagnostics_json or {}).get("path_labels") or []
    ]
    assert executor_path_labels == list(coarse_result.trace.path_labels_json or [])

    coarse_package = build_context_package(
        db_session,
        knowledge_base_id=kb.id,
        query="What is a Markov blanket?",
        trace=coarse_result.trace,
        results=coarse_result.results,
        snapshot_verifier=coarse_result.snapshot_verifier,
    )
    coarse_contexts = context_package_to_contexts(coarse_package)
    exact_answer = str(coarse_contexts[0]["content"])
    coarse_citations = agent_graph.citation_payloads_from_package(
        coarse_package,
        retrieval_trace_id=coarse_package.retrieval_trace_id,
        answer=exact_answer,
        question="What is a Markov blanket?",
    )
    provenance = audit_citation_provenance(
        db_session,
        knowledge_base_id=kb.id,
        package=coarse_package,
        citations=coarse_citations,
        contexts=coarse_contexts,
    )
    assert provenance["all_valid"] is True, [
        item["reasons"] for item in provenance["audits"]
    ]


@pytest.mark.asyncio
async def test_mid_initial_budget_is_mode_specific(monkeypatch, db_session, populated_context_graph):
    from app.schemas import SearchFilters
    from app.services import context_graph

    kb = populated_context_graph["knowledge_base"]
    envelope = dict(context_graph.agent_operating_envelope())
    envelope.update(
        {
            "agent_coarse_initial_budget": 2,
            "agent_coarse_top_k": 1,
            "agent_mid_per_coarse_budget": 2,
            "agent_coarse_drilldown_mid_initial_budget": 3,
            "agent_mid_initial_budget": 1,
            "agent_mid_top_k": 4,
            "agent_chunk_per_mid_budget": 2,
            "agent_chunk_initial_budget": 3,
            "agent_chunk_top_k": 2,
            "structure_restore_per_chunk_budget": 1,
            "structure_restore_budget": 1,
        }
    )
    monkeypatch.setattr(context_graph, "agent_operating_envelope", lambda settings=None: dict(envelope))

    mid_result = await context_graph.layered_search(
        db_session,
        kb.id,
        "What is a Markov blanket budget check?",
        SearchFilters(),
        4,
        retrieval_granularity="mid",
    )

    assert mid_result.results
    assert mid_result.trace.candidate_pools_json["mid_direct_entries"]["top_k"] == 1
    assert len(mid_result.trace.stage_queues_json["mid"]["entry_ids"]) <= 1
    assert mid_result.trace.topk_selection_json["mid"]["top_k"] == 4
    assert len(mid_result.trace.topk_selection_json["mid"]["selected_ids"]) <= 4
    assert mid_result.trace.stage_queues_json["mid"]["top_k"] == 4
    assert mid_result.trace.candidate_pools_json["chunk_initial_entries"]["top_k"] == 3
    assert mid_result.trace.stage_queues_json["chunk"]["initial_top_k"] == 3
    assert mid_result.trace.topk_selection_json["chunk"]["top_k"] == 2
    assert len(mid_result.results) <= 2

    coarse_result = await context_graph.layered_search(
        db_session,
        kb.id,
        "What is a Markov blanket budget check?",
        SearchFilters(),
        4,
        retrieval_granularity="coarse",
    )

    assert coarse_result.results
    assert len(coarse_result.trace.stage_queues_json["coarse"]["entry_ids"]) <= 2
    assert coarse_result.trace.topk_selection_json["coarse"]["top_k"] == 1
    assert len(coarse_result.trace.topk_selection_json["coarse"]["selected_ids"]) <= 1
    assert "mid_direct_entries" not in coarse_result.trace.candidate_pools_json
    assert len(coarse_result.trace.candidate_pools_json["mid_by_coarse"]) <= 1
    assert all(
        pool["per_parent_budget_status"]["budget"] == 2
        for pool in coarse_result.trace.candidate_pools_json["mid_by_coarse"]
    )
    assert coarse_result.trace.candidate_pools_json["mid_initial_entries"]["top_k"] == 3
    assert len(coarse_result.trace.stage_queues_json["mid"]["entry_ids"]) <= 3
    assert coarse_result.trace.stage_queues_json["mid"]["initial_top_k"] == 3
    assert coarse_result.trace.topk_selection_json["mid"]["entry_mode"] == "coarse"
    assert coarse_result.trace.topk_selection_json["mid"]["top_k"] == 4
    assert coarse_result.trace.stage_queues_json["mid"]["top_k"] == 4
    assert coarse_result.trace.candidate_pools_json["chunk_initial_entries"]["top_k"] == 3
    assert coarse_result.trace.topk_selection_json["chunk"]["top_k"] == 2
    assert len(coarse_result.results) <= 2


@pytest.mark.asyncio
async def test_staged_candidate_pools_enforce_dedupe_hard_budget(monkeypatch, db_session, populated_context_graph):
    from app.schemas import SearchFilters
    from app.services import context_graph

    kb = populated_context_graph["knowledge_base"]
    envelope = dict(context_graph.agent_operating_envelope())
    envelope.update(
        {
            "candidate_pool_dedupe_budget": 1,
            "agent_mid_initial_budget": 8,
            "agent_mid_top_k": 8,
            "agent_chunk_per_mid_budget": 8,
            "agent_chunk_initial_budget": 8,
            "agent_chunk_top_k": 8,
        }
    )
    monkeypatch.setattr(context_graph, "agent_operating_envelope", lambda settings=None: dict(envelope))

    result = await context_graph.layered_search(
        db_session,
        kb.id,
        "How does a Markov blanket support conditional independence?",
        SearchFilters(),
        8,
        retrieval_granularity="mid",
    )

    audit = result.trace.candidate_pools_json["candidate_dedupe_budget"]
    assert audit["protocol_version"] == "candidate_pool_dedupe_hard_interrupt_v1"
    assert audit["limit_per_pool"] == 1
    assert audit["pool_count"] >= 3
    assert audit["hard_interrupt_count"] > 0
    assert all(pool["unique_admitted_count"] <= 1 for pool in audit["audits"])
    assert result.trace.candidate_pools_json["mid_direct_entries"]["candidate_count"] <= 1
    assert result.trace.candidate_pools_json["chunk_initial_entries"]["candidate_count"] <= 1
    assert result.trace.convergence_json["candidate_pool_dedupe_budget"]["limit_per_pool"] == 1
