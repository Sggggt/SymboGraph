from __future__ import annotations

import threading

import pytest
from sqlalchemy.exc import StatementError

from test_tpe_audit_postgres import postgres_tpe_scope


def test_pdf_label_cleaning_survives_real_postgres_text_gate(postgres_tpe_scope):
    from app.db import SessionLocal
    from app.models import ChunkStructureNode
    from app.services.ingestion import exception_message
    from app.services.parsers import ParsedSection, ParsedStructureObject, _clean_section
    scope = postgres_tpe_scope
    raw = "Unit test\x00 formula"
    artifact = ParsedStructureObject(structure_id="unit-test",object_type="formula",text=raw,char_start=0,char_end=len(raw),title=raw)
    clean = _clean_section(ParsedSection(title="unit-test",text=raw,structure_objects=[artifact]), "pdf")
    def node(title):
        return ChunkStructureNode(knowledge_base_id=scope["knowledge_base_id"],document_id=scope["document_id"],document_version_id=scope["document_version_id"],node_type="formula",title=title,char_start=0,char_end=len(clean.text),depth=1)
    with SessionLocal() as db:
        with pytest.raises(StatementError) as failed:
            with db.begin_nested():
                db.add(node(raw))
                db.flush()
        assert exception_message(failed.value) == "database_text_invalid_control: NUL byte rejected"
        assert db.is_active
        cleaned = node(clean.structure_objects[0].title)
        db.add(cleaned)
        db.commit()
        assert "\x00" not in db.get(ChunkStructureNode,cleaned.id).title
        assert clean.text[clean.structure_objects[0].char_start:clean.structure_objects[0].char_end] == clean.structure_objects[0].text


def test_build_sampler_observes_durable_cancellation(postgres_tpe_scope):
    from app.db import SessionLocal
    from app.models import IngestionBatch
    from app.services.build_performance import BuildPerformance
    from app.services.cancellation import IngestionCancelled
    perf = BuildPerformance(batch_id=postgres_tpe_scope["batch_id"])
    sampler = threading.Thread(target=perf.sampler)
    sampler.start()
    try:
        with SessionLocal() as db:
            batch=db.get(IngestionBatch,postgres_tpe_scope["batch_id"])
            batch.status="cancel_requested"
            db.commit()
        assert perf.cancel.wait(2)
        with pytest.raises(IngestionCancelled): perf.check()
    finally:
        perf.stop.set()
        sampler.join(timeout=3)
    assert not sampler.is_alive()


@pytest.mark.parametrize("packed",[False,True])
def test_membership_hash_patch_preserves_full_facts_after_postgres_reload(postgres_tpe_scope,packed):
    from app.db import SessionLocal
    from app.models import ChunkRelationGraphState,RQPrefix,RQPrefixMembership
    from app.services.graph_state_hashes import build_relation_state_hash_card,freeze_constructed_row_json
    from test_relation_quota_signals import _add_document_chunks
    with SessionLocal() as db:
        kb_id=postgres_tpe_scope["knowledge_base_id"]
        _,_,chunks=_add_document_chunks(db,kb_id,suffix="unit-test-json-patch",chunk_ids=["unit-test-patch-chunk"])
        state=ChunkRelationGraphState(knowledge_base_id=kb_id,chunk_version=1,scope_hash="1"*64,state_hash="2"*64,
            embedding_text_version="unit-test",active_chunk_ids_json=[chunks[0].id])
        db.add(state);db.flush()
        prefix=RQPrefix(graph_state_id=state.id,knowledge_base_id=kb_id,rq_prefix_key="unit-test-prefix",label="unit test",rq_level=1,rq_path_prefix=[1])
        db.add(prefix);db.flush()
        original={"residual_vector":[.012345678901234567,-0.0,1e-19],"reconstructed_vector":[.25,1.,0.],"preserved":{"score":.75}}
        if packed:
            from app.services.rq_numeric_storage import pack_rq_vector,RQ_NUMERIC_STORAGE_PROTOCOL
            original={**original,"residual_vector":pack_rq_vector(original["residual_vector"]),
                "reconstructed_vector":pack_rq_vector(original["reconstructed_vector"]),"numeric_storage_protocol":RQ_NUMERIC_STORAGE_PROTOCOL}
        row=RQPrefixMembership(rq_prefix_id=prefix.id,chunk_id=chunks[0].id,diagnostics_json=original)
        db.add(row);db.flush()
        before=build_relation_state_hash_card(db,state,chunks,protocol_identities={},vector_identity={})
        stored_hash=row.diagnostics_json["canonical_membership_fact_hash"]
        db.expire(row)
        assert row.diagnostics_json["canonical_membership_fact_hash"] == stored_hash
        assert all(row.diagnostics_json[key] == value for key,value in original.items())
        after=build_relation_state_hash_card(db,state,chunks,protocol_identities={},vector_identity={})
        assert after == before
        db.flush()
        freeze_constructed_row_json(row)
        db.flush()
        cached=build_relation_state_hash_card(db,state,chunks,protocol_identities={},vector_identity={},relation_edges_override=[],memberships_override=[row])
        assert cached == before
        freeze_constructed_row_json(row)
        db.flush()
        from sqlalchemy import update
        db.execute(update(RQPrefixMembership.__table__).where(RQPrefixMembership.id == row.id).values(diagnostics_json={"tampered":True}))
        with pytest.raises(RuntimeError,match="JSON identity drifted"):
            build_relation_state_hash_card(db,state,chunks,protocol_identities={},vector_identity={},relation_edges_override=[],memberships_override=[row])
        db.rollback()


def test_signal_pool_trace_and_business_hash_survive_postgres_reload(postgres_tpe_scope):
    from app.db import SessionLocal
    from app.models import ChunkRelationGraphState,ChunkRelationEdge
    from app.services.context_graph import add_chunk_relation_edge,relation_edge_rank_trace_payload
    from app.services.graph_state_hashes import build_relation_state_hash_card
    from app.services.relation_signal_storage import compact_signal_features,pack_signal_pool,STATE_KEY
    from test_relation_quota_signals import _add_document_chunks
    with SessionLocal() as db:
        kb_id=postgres_tpe_scope["knowledge_base_id"]
        _,_,chunks=_add_document_chunks(db,kb_id,suffix="unit-test-signal-pool",chunk_ids=["unit-test-pool-a","unit-test-pool-b"])
        state=ChunkRelationGraphState(knowledge_base_id=kb_id,chunk_version=1,scope_hash="1"*64,state_hash="2"*64,
            embedding_text_version="unit-test",active_chunk_ids_json=[row.id for row in chunks])
        db.add(state);db.flush()
        card={"value":.75,"complete_audit":{"count":7}}
        features={"directed_source_chunk_id":chunks[0].id,"directed_target_chunk_id":chunks[1].id,
            "source_node_quality_card":card,"directional_contributions":[{"source_chunk_id":chunks[0].id,"target_chunk_id":chunks[1].id,"source_node_quality_card":card}]}
        pool={};compact=compact_signal_features(features,pool)
        state.diagnostics_json={STATE_KEY:pack_signal_pool(pool)}
        edges={}
        edge=add_chunk_relation_edge(db,state,chunks[0].id,chunks[1].id,"dense_semantic",.8,compact,edges)
        db.flush()
        before=build_relation_state_hash_card(db,state,chunks,protocol_identities={},vector_identity={})
        edge_id=edge.id
        db.expire_all()
        edge=db.get(ChunkRelationEdge,edge_id)
        trace=relation_edge_rank_trace_payload(edge)
        assert trace["source_node_quality_card"] == card
        assert trace["directional_contributions"][0]["source_node_quality_card"] == card
        assert build_relation_state_hash_card(db,state,chunks,protocol_identities={},vector_identity={}) == before
        db.rollback()
