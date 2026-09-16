import pytest
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError

from app.models import LexicalIndexJob, LexicalIndexState


def state(kb_id, lifecycle="candidate"):
    return LexicalIndexState(knowledge_base_id=kb_id, protocol_version="source_chunk_bm25_v1",
        tokenizer_protocol="unit-test-tokenizer", tokenizer_hash="1" * 64,
        scoring_protocol="positive_idf_bm25_v1", scoring_hash="6" * 64,
        bm25_k1=1.2, bm25_b=.75, source_scope_hash="2" * 64,
        statistics_hash="3" * 64, postings_hash="4" * 64, state_hash="5" * 64, state=lifecycle)


def test_database_allows_candidates_but_never_two_active_indexes_for_one_kb(db_session, sample_knowledge_base):
    db_session.add_all([state(sample_knowledge_base.id), state(sample_knowledge_base.id), state(sample_knowledge_base.id, "active")])
    db_session.commit()
    with pytest.raises(IntegrityError):
        with db_session.begin_nested():
            db_session.add(state(sample_knowledge_base.id, "active"))
            db_session.flush()


def test_invalid_counts_and_states_are_rejected_by_database(db_session, sample_knowledge_base):
    for field, value in (("document_count", -1), ("posting_count", -1), ("state", "ready_but_unverified")):
        row = state(sample_knowledge_base.id)
        setattr(row, field, value)
        with pytest.raises(IntegrityError):
            with db_session.begin_nested():
                db_session.add(row)
                db_session.flush()


def test_job_cannot_claim_another_knowledge_bases_index(db_session, sample_knowledge_base):
    from app.models import KnowledgeBase
    db_session.execute(text("PRAGMA foreign_keys=ON"))
    assert db_session.scalar(text("PRAGMA foreign_keys")) == 1
    other = KnowledgeBase(name="Unit Test Other KB", source_root=sample_knowledge_base.source_root + "-other")
    db_session.add(other)
    row = state(sample_knowledge_base.id)
    db_session.add(row)
    db_session.commit()
    with pytest.raises(IntegrityError):
        with db_session.begin_nested():
            db_session.add(LexicalIndexJob(knowledge_base_id=other.id, target_state_id=row.id, source_scope_hash=row.source_scope_hash))
            db_session.flush()
