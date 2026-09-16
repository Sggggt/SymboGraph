"""Real PostgreSQL source/span persistence; no model or vector service calls."""
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4

import pytest
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError

from test_answer_sources import answer_draft, new_answer
from test_citation_provenance import _build_package
from test_tpe_audit_postgres import postgres_tpe_scope


@pytest.fixture
def postgres_source_root(postgres_tpe_scope):
    from app.core.config import get_settings

    # Only this freshly allocated synthetic source scope is cleaned up.
    with TemporaryDirectory(prefix="unit-test-source-binding-", dir=get_settings().data_root) as directory:
        yield Path(directory) / "storage"


@pytest.mark.asyncio
@pytest.mark.parametrize("rollback", [False, True])
async def test_source_binding_commit_rollback_and_chunk_scope_in_postgres(postgres_tpe_scope, postgres_source_root, rollback):
    from app.db import SessionLocal
    from app.models import AnswerSession, AnswerSourceBinding, CitationVerification, KnowledgeBase, ContextPackage
    from app.services.answer_sources import build_answer_evidence_manifest, persist_answer_source_bindings, source_binding_citations

    scope = postgres_tpe_scope
    with SessionLocal() as db:
        kb = db.get(KnowledgeBase, scope["knowledge_base_id"])
        kb.source_root = str(postgres_source_root)
        db.commit()
        kb, _document, _version, _chunks, package, contexts, text, _legacy = await _build_package(db, kb, postgres_source_root)
        db.commit()
        package_id = package.id
        draft = answer_draft(text)
        manifest = build_answer_evidence_manifest(package, contexts)
        answer = new_answer(db, kb, package, draft)
        rows = persist_answer_source_bindings(db, answer_session=answer, package=package, contexts=contexts,
            draft=draft, evidence=manifest, unit_limit=12, reflection_audit_hash="a" * 64)
        answer_id, binding_ids = answer.id, [row.id for row in rows]
        assert binding_ids and db.get_bind().dialect.name == "postgresql"
        if rollback:
            db.rollback()
        else:
            db.commit()

    with SessionLocal() as db:
        answer = db.get(AnswerSession, answer_id)
        bindings = list(db.scalars(select(AnswerSourceBinding).where(AnswerSourceBinding.answer_session_id == answer_id)))
        if rollback:
            assert answer is None and bindings == []
            assert db.get(ContextPackage, package_id) is not None
            return
        assert answer is not None and [row.id for row in bindings] == binding_ids
        public = source_binding_citations(answer_session=answer, package=db.get(ContextPackage, package_id),
            rows=bindings, reflection_audit_hash="a" * 64)
        assert public[0]["source_binding"]["status"] == "source_bound"
        assert public[0]["verification"] is None
        assert db.scalar(select(func.count()).select_from(CitationVerification).where(CitationVerification.knowledge_base_id == scope["knowledge_base_id"])) == 0

        # A real second KB makes this a composite ownership check, rather than
        # merely a missing-parent foreign-key failure. All forged rows roll back.
        with pytest.raises(IntegrityError) as failed:
            with db.begin_nested():
                foreign = KnowledgeBase(name="unit-test-foreign-source-scope", source_root=str(postgres_source_root / "foreign"))
                db.add(foreign)
                db.flush()
                original = bindings[0]
                values = {column.name: getattr(original, column.name) for column in AnswerSourceBinding.__table__.columns}
                values.update(id=str(uuid4()), knowledge_base_id=foreign.id, binding_hash="b" * 64, unit_id="c" * 64)
                db.add(AnswerSourceBinding(**values))
                db.flush()
        assert failed.value.orig.diag.constraint_name == "fk_answer_source_binding_chunk_scope"
        assert db.is_active
