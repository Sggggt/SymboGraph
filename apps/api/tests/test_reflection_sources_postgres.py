"""Real PostgreSQL durability and deferred origin restrictions for retention."""
from uuid import uuid4

import pytest
from sqlalchemy import delete, func, select
from sqlalchemy.exc import IntegrityError

from test_answer_sources_postgres import postgres_source_root
from test_tpe_audit_postgres import postgres_tpe_scope
from test_reflection_sources import source_pair, audit_package


@pytest.mark.asyncio
@pytest.mark.parametrize("rollback", [False, True])
async def test_retention_commit_rollback_and_origin_foreign_keys(postgres_tpe_scope, postgres_source_root, rollback):
    from app.db import SessionLocal
    from app.models import ContextPackage, ContextPackageSourceRetention, KnowledgeBase
    from app.services.reflection_sources import retain_reflection_sources
    with SessionLocal() as db:
        kb = db.get(KnowledgeBase, postgres_tpe_scope["knowledge_base_id"])
        kb.source_root = str(postgres_source_root)
        db.commit()
        old, new = await source_pair(db, kb, postgres_source_root)
        db.commit()
        old_id, new_id = old.id, new.id
        package, _ = retain_reflection_sources(db, candidate_package=new, source_package=old,
            preserve_chunk_ids=old.hit_chunk_ids_json, token_budget=new.token_budget)
        package_id = package.id
        db.rollback() if rollback else db.commit()
    with SessionLocal() as db:
        package = db.get(ContextPackage, package_id)
        assert db.get(ContextPackage, old_id) is not None and db.get(ContextPackage, new_id) is not None
        rows = list(db.scalars(select(ContextPackageSourceRetention).where(ContextPackageSourceRetention.target_context_package_id == package_id)))
        if rollback:
            assert package is None and rows == []
            return
        assert len(rows) == 1 and audit_package(db, package)["all_valid"]
        row = rows[0]
        # The original snapshot remains an audit dependency until its target
        # is removed, including when deletion reaches transaction commit.
        db.execute(delete(ContextPackage).where(ContextPackage.id == old_id))
        with pytest.raises(IntegrityError):
            db.commit()
        db.rollback()
        assert db.get(ContextPackage, old_id) is not None
        with pytest.raises(IntegrityError):
            with db.begin_nested():
                foreign = KnowledgeBase(name="unit-test-retention-foreign", source_root=str(postgres_source_root / "foreign"))
                db.add(foreign)
                db.flush()
                values = {column.name: getattr(row, column.name) for column in ContextPackageSourceRetention.__table__.columns}
                values.update(id=str(uuid4()), knowledge_base_id=foreign.id, target_context_package_id=new_id)
                db.add(ContextPackageSourceRetention(**values))
                db.flush()
        assert db.is_active
