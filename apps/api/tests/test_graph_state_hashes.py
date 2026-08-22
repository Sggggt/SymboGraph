from __future__ import annotations

import copy
import hashlib
import re
import uuid
from pathlib import Path

import pytest
from sqlalchemy import delete, select, update


def _context_card_args() -> dict:
    return {
        "chunk_business_scope_hash": "scope-a",
        "contextual_index_hash": "contextual-a",
        "structure_state_hash": "structure-a",
        "relation_state_hash": "relation-a",
        "rq_state_hash": "rq-a",
        "rq_pair_aggregate_hash": "pairs-a",
        "mid_state_hash": "mid-a",
        "coarse_state_hash": "coarse-a",
        "runtime_settings_hash": "runtime-a",
        "profile_hash": "profile-a",
        "policy_state_hash": "policy-a",
        "prompt_protocol_hash": "prompt-a",
        "agent_operating_envelope_hash": "agent-a",
        "edge_distance_protocol_hash": "distance-a",
        "edge_projection_protocol_hash": "projection-a",
        "traversal_protocol_hash": "traversal-a",
        "graph_protocol_runtime_identity_hash": "protocols-a",
        "vector_identity": {
            "embedding_model": "unit-embedding",
            "embedding_dimension": 8,
            "created_at": "2026-01-01T00:00:00",
            "provider_status": "healthy",
        },
    }


def test_canonical_fact_sets_ignore_row_order_and_database_uuid() -> None:
    from app.services.graph_state_hashes import canonical_fact_set_hash

    facts_a = [
        {
            "id": str(uuid.uuid4()),
            "created_at": "2026-01-01T00:00:00",
            "source": "chunk:a",
            "target": "chunk:b",
            "weight": 0.4,
        },
        {
            "id": str(uuid.uuid4()),
            "source": "chunk:b",
            "target": "chunk:c",
            "weight": 0.8,
        },
    ]
    facts_b = copy.deepcopy(list(reversed(facts_a)))
    facts_b[0]["id"] = str(uuid.uuid4())
    facts_b[1]["id"] = str(uuid.uuid4())
    facts_b[1]["created_at"] = "2030-01-01T00:00:00"

    baseline = canonical_fact_set_hash("unit_business_facts_v1", facts_a)
    assert canonical_fact_set_hash("unit_business_facts_v1", facts_b) == baseline
    facts_b[0]["weight"] = 0.81
    assert canonical_fact_set_hash("unit_business_facts_v1", facts_b) != baseline


def test_canonical_fact_sets_ignore_uuid_derived_integrity_hashes() -> None:
    from app.services.graph_state_hashes import canonical_fact_set_hash

    business_fact = {
        "source": "chunk:a",
        "target": "chunk:b",
        "support_chunk_business_keys_hash": "business-support-hash",
        "support_edge_fact_hash": "business-edge-hash",
        "support_chunk_ids_sample_hash": "uuid-sample-hash-a",
        "support_chunk_edge_ids_hash": "uuid-edge-hash-a",
    }
    rebuilt_fact = {
        **business_fact,
        "support_chunk_ids_sample_hash": "uuid-sample-hash-b",
        "support_chunk_edge_ids_hash": "uuid-edge-hash-b",
    }

    baseline = canonical_fact_set_hash("unit_business_facts_v1", [business_fact])
    assert canonical_fact_set_hash("unit_business_facts_v1", [rebuilt_fact]) == baseline

    rebuilt_fact["support_edge_fact_hash"] = "changed-business-edge-hash"
    assert canonical_fact_set_hash("unit_business_facts_v1", [rebuilt_fact]) != baseline


def _reference_digest_multiset_hash(
    facts,
    *,
    protocol_version: str,
    fact_protocol_version: str,
) -> str:
    from app.services.graph_state_hashes import canonical_graph_hash

    digests = sorted(
        canonical_graph_hash(fact_protocol_version, fact) for fact in facts
    )
    return canonical_graph_hash(
        protocol_version,
        {
            "canonical_fact_digests": digests,
            "fact_count": len(digests),
            "fact_protocol_version": fact_protocol_version,
        },
    )


def test_streaming_fact_digest_multiset_matches_reference_and_is_bounded() -> None:
    from app.services.graph_state_hashes import (
        streaming_canonical_fact_digest_multiset_hash,
    )

    protocol = "unit_fact_digest_multiset_v1"
    fact_protocol = "unit_business_fact_digest_v1"
    facts = [
        {
            "id": str(uuid.uuid4()),
            "source": f"chunk:{index % 3}",
            "weight": float(index % 4) / 10.0,
        }
        for index in range(9)
    ]
    facts.append(copy.deepcopy(facts[2]))
    expected = _reference_digest_multiset_hash(
        facts,
        protocol_version=protocol,
        fact_protocol_version=fact_protocol,
    )

    streamed = streaming_canonical_fact_digest_multiset_hash(
        protocol_version=protocol,
        fact_protocol_version=fact_protocol,
        facts=facts,
        sort_run_size=2,
        merge_fan_in=2,
    )
    reordered = copy.deepcopy(list(reversed(facts)))
    for fact in reordered:
        fact["id"] = str(uuid.uuid4())
    reordered_stream = streaming_canonical_fact_digest_multiset_hash(
        protocol_version=protocol,
        fact_protocol_version=fact_protocol,
        facts=reordered,
        sort_run_size=3,
        merge_fan_in=3,
    )

    assert streamed.state_hash == expected
    assert reordered_stream.state_hash == expected
    assert streamed.fact_count == len(facts)
    assert streamed.max_buffered_digests <= 2
    assert streamed.initial_run_count > 2
    assert streamed.max_open_runs <= 2


def test_streaming_fact_digest_multiset_retains_duplicate_multiplicity() -> None:
    from app.services.graph_state_hashes import (
        streaming_canonical_fact_digest_multiset_hash,
    )

    common = {
        "protocol_version": "unit_fact_digest_multiset_v1",
        "fact_protocol_version": "unit_business_fact_digest_v1",
        "sort_run_size": 1,
        "merge_fan_in": 2,
    }
    fact = {"source": "chunk:a", "target": "chunk:b", "weight": 0.5}
    single = streaming_canonical_fact_digest_multiset_hash(
        facts=[fact], **common
    )
    duplicate = streaming_canonical_fact_digest_multiset_hash(
        facts=[fact, copy.deepcopy(fact)], **common
    )

    assert single.fact_count == 1
    assert duplicate.fact_count == 2
    assert duplicate.state_hash != single.state_hash


def test_streaming_fact_digest_multiset_cleans_temporary_runs_on_error(
    monkeypatch,
) -> None:
    from app.services import graph_state_hashes

    real_temporary_directory = graph_state_hashes.tempfile.TemporaryDirectory
    created_paths: list[Path] = []

    def tracking_temporary_directory(*args, **kwargs):
        directory = real_temporary_directory(*args, **kwargs)
        created_paths.append(Path(directory.name))
        return directory

    monkeypatch.setattr(
        graph_state_hashes.tempfile,
        "TemporaryDirectory",
        tracking_temporary_directory,
    )
    with pytest.raises(TypeError, match="JSON primitives"):
        graph_state_hashes.streaming_canonical_fact_digest_multiset_hash(
            protocol_version="unit_fact_digest_multiset_v1",
            fact_protocol_version="unit_business_fact_digest_v1",
            facts=[{"fact": "valid"}, object()],
            sort_run_size=1,
            merge_fan_in=2,
        )

    assert created_paths
    assert all(not path.exists() for path in created_paths)


def test_legacy_structure_v1_card_remains_self_verifiable() -> None:
    from app.services.graph_state_hashes import (
        canonical_graph_hash,
        verify_state_hash_card,
    )

    protocol = "chunk_structure_state_hash_v1"
    payload = {
        "protocol_version": protocol,
        "chunk_business_scope_hash": "legacy-scope",
        "component_hashes": {
            "nodes": "legacy-nodes",
            "edges": "legacy-edges",
            "mappings": "legacy-mappings",
        },
        "counts": {"nodes": 1, "edges": 0, "mappings": 1},
    }
    card = {
        **payload,
        "state_hash": canonical_graph_hash(protocol, payload),
    }

    assert verify_state_hash_card(card)


@pytest.mark.parametrize("field", ["title", "definition", "content"])
def test_uuid_shaped_semantic_text_remains_a_business_fact(field: str) -> None:
    from app.services.graph_state_hashes import canonical_graph_hash

    first = canonical_graph_hash(
        "uuid_shaped_semantic_text_v1",
        {field: "11111111-1111-4111-8111-111111111111"},
    )
    second = canonical_graph_hash(
        "uuid_shaped_semantic_text_v1",
        {field: "22222222-2222-4222-8222-222222222222"},
    )

    assert first != second


def test_database_address_fields_are_reference_mapped_or_excluded_by_schema() -> None:
    from app.services.graph_state_hashes import _canonical_value

    mapped_address = "11111111-1111-4111-8111-111111111111"
    unresolved_address = "22222222-2222-4222-8222-222222222222"
    business_key = "chunk-business-key-a"
    canonical = _canonical_value(
        {
            "source_chunk_id": mapped_address,
            "target_chunk_id": unresolved_address,
            "title": mapped_address,
            "definition": unresolved_address,
            "content": mapped_address,
        },
        references={mapped_address: business_key},
    )

    assert canonical == {
        "content": mapped_address,
        "definition": unresolved_address,
        "source_chunk_id": business_key,
        "title": mapped_address,
    }


def test_context_composite_mutation_changes_only_composite_identity() -> None:
    from app.services.graph_state_hashes import (
        build_context_state_hash_card,
        verify_state_hash_card,
    )

    args = _context_card_args()
    baseline = build_context_state_hash_card(**args)
    assert verify_state_hash_card(baseline)

    ephemeral = copy.deepcopy(args)
    ephemeral["vector_identity"].update(
        {
            "created_at": "2035-01-01T00:00:00",
            "provider_status": "offline",
            "raw_output": "not a graph fact",
            "id": str(uuid.uuid4()),
        }
    )
    assert build_context_state_hash_card(**ephemeral) == baseline

    mutated = copy.deepcopy(args)
    mutated["relation_state_hash"] = "relation-b"
    changed = build_context_state_hash_card(**mutated)
    assert changed["state_hash"] != baseline["state_hash"]
    assert baseline["layer_hashes"]["structure"] == "structure-a"
    assert baseline["layer_hashes"]["mid"] == "mid-a"
    assert changed["layer_hashes"]["relation"] == "relation-b"


def test_policy_hash_excludes_reward_uuid_but_binds_policy_facts() -> None:
    from app.services.graph_state_hashes import canonical_policy_state_hash

    def state_hash(event_id: str, weight: float) -> str:
        return canonical_policy_state_hash(
            policy_family="context_graph_bandit",
            policy_version="context_graph_bandit_v1",
            profile_objective_hash=None,
            weights={"direct": weight},
            constraints={"fallback_disabled": True},
            exploration={"epsilon": 0.05},
            reward_summary={
                "last_reward_event_id": event_id,
                "last_reward": {"citation_pass_rate": 1.0},
                "provider_status": "healthy",
            },
        )

    baseline = state_hash(str(uuid.uuid4()), 1.0)
    assert state_hash(str(uuid.uuid4()), 1.0) == baseline
    assert state_hash(str(uuid.uuid4()), 0.9) != baseline


def _all_string_values(value):
    if isinstance(value, dict):
        for key, item in value.items():
            yield str(key)
            yield from _all_string_values(item)
    elif isinstance(value, list):
        for item in value:
            yield from _all_string_values(item)
    elif isinstance(value, str):
        yield value


def test_active_state_cards_are_valid_uuid_free_and_gray_model_free(
    db_session,
    populated_context_graph,
) -> None:
    from app.services.context_graph import active_graph_admission_gate
    from app.services.graph_state_hashes import (
        STRUCTURE_STATE_HASH_PROTOCOL_VERSION,
        verify_state_hash_card,
    )

    state = populated_context_graph["state"]
    diagnostics = dict(state.diagnostics_json or {})
    cards = [
        diagnostics["structure_state_hash_card"],
        diagnostics["relation_state_hash_card"],
        diagnostics["mid_state_hash_card"],
        diagnostics["coarse_state_hash_card"],
        diagnostics["canonical_state_hash_card"],
    ]
    assert (
        diagnostics["structure_state_hash_card"]["protocol_version"]
        == STRUCTURE_STATE_HASH_PROTOCOL_VERSION
        == "chunk_structure_state_hash_v2"
    )
    assert all(verify_state_hash_card(card) for card in cards)
    assert all(
        not re.fullmatch(
            r"[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}",
            value,
            flags=re.IGNORECASE,
        )
        for card in cards
        for value in _all_string_values(card)
    )
    assert (
        diagnostics["canonical_protocol_identities"][
            "gray_zone_model_call_count"
        ]
        == 0
    )
    assert (
        active_graph_admission_gate(
            db_session, populated_context_graph["knowledge_base"].id
        )
        is state
    )


def test_mid_layer_card_changes_when_uuid_shaped_definition_changes(
    db_session,
    populated_context_graph,
) -> None:
    from app.models import ChunkRelationGraphState, MidConcept, MidConceptState
    from app.services.context_graph import graph_state_protocol_identities
    from app.services.graph_state_hashes import build_mid_state_hash_card

    context_state = populated_context_graph["state"]
    mid_state = db_session.get(MidConceptState, context_state.mid_concept_state_id)
    relation_state = db_session.get(
        ChunkRelationGraphState,
        context_state.chunk_relation_graph_state_id,
    )
    assert mid_state is not None
    assert relation_state is not None
    concept = db_session.scalar(
        select(MidConcept).where(MidConcept.concept_state_id == mid_state.id)
    )
    assert concept is not None
    diagnostics = dict(mid_state.diagnostics_json or {})
    common = {
        "relation_state_hash": str(relation_state.state_hash),
        "profile_hash": str(diagnostics["profile_hash"]),
        "prompt_protocol_hash": str(diagnostics["prompt_protocol_hash"]),
        "protocol_identities": graph_state_protocol_identities(scope="concept"),
    }

    concept.definition = "11111111-1111-4111-8111-111111111111"
    db_session.flush()
    first = build_mid_state_hash_card(
        db_session,
        mid_state,
        relation_state,
        populated_context_graph["chunks"],
        **common,
    )
    concept.definition = "22222222-2222-4222-8222-222222222222"
    db_session.flush()
    second = build_mid_state_hash_card(
        db_session,
        mid_state,
        relation_state,
        populated_context_graph["chunks"],
        **common,
    )

    assert first["component_hashes"]["concepts"] != second["component_hashes"]["concepts"]
    assert first["state_hash"] != second["state_hash"]


def test_admission_rejects_row_count_drift_without_deep_reserialization(
    db_session,
    populated_context_graph,
) -> None:
    from app.models import MidConcept, MidConceptMembership
    from app.services.context_graph import (
        ActiveContextGraphAdmissionError,
        active_graph_admission_gate,
    )

    state = populated_context_graph["state"]
    membership = db_session.scalar(
        select(MidConceptMembership)
        .join(MidConcept)
        .where(MidConcept.concept_state_id == state.mid_concept_state_id)
    )
    assert membership is not None
    db_session.delete(membership)
    db_session.flush()
    with pytest.raises(
        ActiveContextGraphAdmissionError,
        match="mid_concepts_canonical_count_mismatch:memberships",
    ):
        active_graph_admission_gate(
            db_session, populated_context_graph["knowledge_base"].id
        )


def _reference_structure_mapping_facts(db_session, chunks):
    from app.models import ChunkStructureMapping, ChunkStructureNode
    from app.services.graph_state_hashes import (
        _canonical_value,
        _reference_value,
        canonical_graph_hash,
        chunk_business_references,
    )

    refs = chunk_business_references(db_session, chunks)
    version_ids = sorted({str(chunk.document_version_id) for chunk in chunks})
    nodes = list(
        db_session.scalars(
            select(ChunkStructureNode).where(
                ChunkStructureNode.document_version_id.in_(version_ids)
            )
        ).all()
    )
    node_keys = {}
    for node in nodes:
        document_key = refs.document_version_key_by_id[str(node.document_version_id)]
        node_keys[str(node.id)] = canonical_graph_hash(
            "chunk_structure_node_business_key_v1",
            {
                "document": document_key,
                "node_type": str(node.node_type or ""),
                "depth": int(node.depth or 0),
                "title": str(node.title or ""),
                "char_span": [node.char_start, node.char_end],
                "page_number": node.page_number,
                "bbox": node.bbox_json or {},
                "layout": node.layout_json or {},
                "path": str(node.path or ""),
            },
        )
    mappings = list(
        db_session.scalars(
            select(ChunkStructureMapping).where(
                ChunkStructureMapping.chunk_id.in_(list(refs.key_by_id))
            )
        ).all()
    )
    references = {**refs.key_by_id, **node_keys}
    return [
        {
            "chunk": _reference_value(mapping.chunk_id, refs.key_by_id),
            "structure_node": _reference_value(
                mapping.structure_node_id, node_keys
            ),
            "overlap_chars": int(mapping.overlap_chars or 0),
            "overlap_tokens": int(mapping.overlap_tokens or 0),
            "coverage_ratio": float(mapping.coverage_ratio or 0.0),
            "span_overlap": float(mapping.span_overlap or 0.0),
            "bbox_iou": mapping.bbox_iou,
            "path_match": mapping.path_match,
            "mapping_weight": float(mapping.mapping_weight or 0.0),
            "mapping_protocol_version": str(
                mapping.mapping_protocol_version or ""
            ),
            "bbox_intersection": mapping.bbox_intersection_json or {},
            "mapping_role": str(mapping.mapping_role or ""),
            "metadata": _canonical_value(
                mapping.metadata_json or {}, references=references
            ),
        }
        for mapping in mappings
    ]


@pytest.fixture
def structure_mapping_hash_fixture(db_session, sample_knowledge_base):
    from app.models import (
        Chunk,
        ChunkStructureEdge,
        ChunkStructureMapping,
        ChunkStructureNode,
        Document,
        DocumentVersion,
    )

    document = Document(
        knowledge_base_id=sample_knowledge_base.id,
        title="Structure mapping hash fixture",
        source_path="unit://structure-mapping-hash.md",
        source_type="markdown",
        checksum="fixture-document-checksum",
        is_active=True,
    )
    db_session.add(document)
    db_session.flush()
    version = DocumentVersion(
        document_id=document.id,
        version=1,
        checksum="fixture-version-checksum",
        storage_path="unit://structure-mapping-hash.snapshot",
        parse_protocol_version="unit_parser_v1",
        is_active=True,
    )
    db_session.add(version)
    db_session.flush()

    chunks = []
    for index, text in enumerate(("alpha", "beta", "gamma")):
        chunk = Chunk(
            knowledge_base_id=sample_knowledge_base.id,
            document_id=document.id,
            document_version_id=version.id,
            chunk_version=1,
            chunk_index=index,
            token_start=index * 10,
            token_end=index * 10 + 5,
            char_start=index * 20,
            char_end=index * 20 + len(text),
            text=text,
            text_hash=hashlib.sha256(text.encode("utf-8")).hexdigest(),
            section_path=f"section-{index}",
            page_start=index + 1,
            page_end=index + 1,
            metadata_json={
                "chunk_protocol_descriptor": {
                    "chunk_schema_version": "unit_chunk_schema_v1",
                    "tokenizer_version": "unit_tokenizer_v1",
                    "chunk_size": 5,
                    "chunk_overlap": 0,
                }
            },
            state="active",
        )
        db_session.add(chunk)
        chunks.append(chunk)
    db_session.flush()

    root = ChunkStructureNode(
        knowledge_base_id=sample_knowledge_base.id,
        document_id=document.id,
        document_version_id=version.id,
        node_type="section",
        depth=0,
        title="Root",
        char_start=0,
        char_end=60,
        page_number=1,
        bbox_json={},
        layout_json={},
        path="Root",
    )
    child = ChunkStructureNode(
        knowledge_base_id=sample_knowledge_base.id,
        document_id=document.id,
        document_version_id=version.id,
        node_type="paragraph",
        depth=1,
        title="Child",
        char_start=0,
        char_end=60,
        page_number=1,
        bbox_json={},
        layout_json={},
        path="Root > Child",
    )
    db_session.add_all([root, child])
    db_session.flush()
    child.parent_id = root.id
    db_session.add(
        ChunkStructureEdge(
            knowledge_base_id=sample_knowledge_base.id,
            document_version_id=version.id,
            source_node_id=root.id,
            target_node_id=child.id,
            edge_type="contains",
            weight=1.0,
            confidence=1.0,
            metadata_json={"fixture": True},
        )
    )
    for index, chunk in enumerate(chunks):
        for node, role, weight in (
            (root, "ancestor", 0.4),
            (child, "overlap", 0.9),
        ):
            db_session.add(
                ChunkStructureMapping(
                    chunk_id=chunk.id,
                    structure_node_id=node.id,
                    document_version_id=version.id,
                    overlap_chars=5 + index,
                    overlap_tokens=2 + index,
                    coverage_ratio=0.5 + index / 10.0,
                    span_overlap=0.4 + index / 10.0,
                    bbox_iou=None,
                    path_match=1.0,
                    mapping_weight=weight,
                    mapping_protocol_version="unit_structure_mapping_v1",
                    bbox_intersection_json={},
                    mapping_role=role,
                    metadata_json={"chunk_id": chunk.id, "node_id": node.id},
                )
            )
    db_session.commit()
    return {
        "chunks": chunks,
        "document": document,
        "version": version,
    }


def test_structure_mapping_stream_matches_reference_and_ignores_row_uuid(
    db_session,
    structure_mapping_hash_fixture,
) -> None:
    from app.models import Chunk, ChunkStructureMapping
    from app.services.graph_state_hashes import (
        STRUCTURE_MAPPING_FACT_DIGEST_PROTOCOL_VERSION,
        STRUCTURE_MAPPING_MULTISET_HASH_PROTOCOL_VERSION,
        build_structure_state_hash_card,
    )

    chunks = structure_mapping_hash_fixture["chunks"]
    reference_facts = _reference_structure_mapping_facts(db_session, chunks)
    expected_hash = _reference_digest_multiset_hash(
        reference_facts,
        protocol_version=STRUCTURE_MAPPING_MULTISET_HASH_PROTOCOL_VERSION,
        fact_protocol_version=STRUCTURE_MAPPING_FACT_DIGEST_PROTOCOL_VERSION,
    )
    baseline = build_structure_state_hash_card(db_session, chunks)
    assert baseline["counts"]["mappings"] == len(reference_facts)
    assert baseline["component_hashes"]["mappings"] == expected_hash

    mapping_ids = list(
        db_session.scalars(select(ChunkStructureMapping.id)).all()
    )
    for mapping_id in mapping_ids:
        db_session.execute(
            update(ChunkStructureMapping)
            .where(ChunkStructureMapping.id == mapping_id)
            .values(id=str(uuid.uuid4()))
        )
    db_session.flush()
    assert build_structure_state_hash_card(db_session, chunks) == baseline

    chunk_ids = [str(chunk.id) for chunk in chunks]
    db_session.expunge_all()
    reloaded_chunks = list(
        db_session.scalars(select(Chunk).where(Chunk.id.in_(chunk_ids))).all()
    )
    assert build_structure_state_hash_card(db_session, reloaded_chunks) == baseline
    assert not any(
        isinstance(instance, ChunkStructureMapping)
        for instance in db_session.identity_map.values()
    )


def test_structure_mapping_hash_detects_tamper_missing_and_bad_provenance(
    db_session,
    structure_mapping_hash_fixture,
) -> None:
    from app.models import ChunkStructureMapping, DocumentVersion
    from app.services.graph_state_hashes import build_structure_state_hash_card

    chunks = structure_mapping_hash_fixture["chunks"]
    baseline = build_structure_state_hash_card(db_session, chunks)
    mapping_row = db_session.execute(
        select(
            ChunkStructureMapping.id,
            ChunkStructureMapping.mapping_weight,
        ).limit(1)
    ).one()

    db_session.execute(
        update(ChunkStructureMapping)
        .where(ChunkStructureMapping.id == mapping_row.id)
        .values(mapping_weight=float(mapping_row.mapping_weight or 0.0) + 0.125)
    )
    db_session.flush()
    tampered = build_structure_state_hash_card(db_session, chunks)
    assert tampered["component_hashes"]["mappings"] != baseline["component_hashes"][
        "mappings"
    ]
    assert tampered["state_hash"] != baseline["state_hash"]

    db_session.execute(
        update(ChunkStructureMapping)
        .where(ChunkStructureMapping.id == mapping_row.id)
        .values(mapping_weight=mapping_row.mapping_weight)
    )
    db_session.flush()
    assert build_structure_state_hash_card(db_session, chunks) == baseline

    db_session.execute(
        delete(ChunkStructureMapping).where(
            ChunkStructureMapping.id == mapping_row.id
        )
    )
    db_session.flush()
    missing = build_structure_state_hash_card(db_session, chunks)
    assert missing["counts"]["mappings"] == baseline["counts"]["mappings"] - 1
    assert missing["component_hashes"]["mappings"] != baseline["component_hashes"][
        "mappings"
    ]
    assert missing["state_hash"] != baseline["state_hash"]

    db_session.rollback()
    current_mapping_id = db_session.scalar(select(ChunkStructureMapping.id).limit(1))
    assert current_mapping_id is not None
    db_session.execute(
        update(ChunkStructureMapping)
        .where(ChunkStructureMapping.id == current_mapping_id)
        .values(mapping_protocol_version="")
    )
    db_session.flush()
    with pytest.raises(RuntimeError, match="missing mapping protocol version"):
        build_structure_state_hash_card(db_session, chunks)

    db_session.rollback()
    version = structure_mapping_hash_fixture["version"]
    invalid_version = DocumentVersion(
        document_id=version.document_id,
        version=int(version.version) + 1,
        checksum="invalid-mapping-provenance",
        storage_path=str(version.storage_path),
        is_active=False,
    )
    db_session.add(invalid_version)
    db_session.flush()
    current_mapping_id = db_session.scalar(select(ChunkStructureMapping.id).limit(1))
    assert current_mapping_id is not None
    db_session.execute(
        update(ChunkStructureMapping)
        .where(ChunkStructureMapping.id == current_mapping_id)
        .values(document_version_id=invalid_version.id)
    )
    db_session.flush()
    with pytest.raises(RuntimeError, match="document-version provenance"):
        build_structure_state_hash_card(db_session, chunks)
