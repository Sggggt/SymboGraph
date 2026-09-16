import pytest
from sqlalchemy import delete, select
from sqlalchemy.exc import IntegrityError

from test_answer_sources_postgres import postgres_source_root
from test_tpe_audit_postgres import postgres_tpe_scope
from test_reflection_expansion import long_source, facets
from test_reflection_sources import audit_package


@pytest.mark.asyncio
@pytest.mark.parametrize("rollback", [False, True])
@pytest.mark.parametrize("focused", [False, True])
async def test_source_expansion_postgres_atomicity_and_origin_guard(postgres_tpe_scope, postgres_source_root, rollback, focused):
    from app.db import SessionLocal
    from app.models import KnowledgeBase, ContextPackage, ContextPackageSourceExpansion
    from app.services.reflection_context import restore_reflection_context
    with SessionLocal() as db:
        kb = db.get(KnowledgeBase, postgres_tpe_scope["knowledge_base_id"])
        kb.source_root = str(postgres_source_root)
        db.commit()
        if focused:
            from app.services.parsers import ParsedSection
            from test_citation_provenance import _build_package
            sections = [
                ParsedSection(title="4 Photon calibration", section="4 Photon calibration", page_number=4,
                    text="Photon calibration measurements require 73 seconds."),
                ParsedSection(title="8 Background", section="8 Background", page_number=8,
                    text="Unrelated routine operations background. " * 120),
                ParsedSection(title="9 Current context", section="9 Current context", page_number=9,
                    text="This current passage mentions photon calibration without its measurement."),
            ]
            _, _, _, chunks, source, _, _, _ = await _build_package(db, kb, postgres_source_root,
                source_text="\n\n".join(section.text for section in sections), parsed_sections=sections, hit_index=-1)
        else:
            _, _, _, chunks, source, _, _, _ = await long_source(db, kb, postgres_source_root)
        db.commit()
        source_id = source.id
        candidate, _ = restore_reflection_context(db, source_package=source, target_chunk_ids=[chunks[-1].id],
            preserve_chunk_ids=[chunks[-1].id], token_budget=source.token_budget, restore_per_chunk_budget=1, query_facets=facets(),
            restoration_focus=["Photon calibration in section 4"] if focused else None)
        if focused:
            assert any("73 seconds" in item["content"] for item in candidate.package_json["chunks"])
        target_id = candidate.id
        db.rollback() if rollback else db.commit()
    with SessionLocal() as db:
        target = db.get(ContextPackage, target_id)
        rows = list(db.scalars(select(ContextPackageSourceExpansion).where(ContextPackageSourceExpansion.target_context_package_id == target_id)))
        assert db.get(ContextPackage, source_id) is not None
        if rollback:
            assert target is None and not rows
            return
        assert len(rows) == 1 and audit_package(db, target)["all_valid"]
        assert rows[0].witness_json["navigation"]["rank_protocol"] == (
            "source_structure_facet_navigation_v3" if focused else "source_structure_facet_navigation_v1")
        db.execute(delete(ContextPackage).where(ContextPackage.id == source_id))
        with pytest.raises(IntegrityError):
            db.commit()
        db.rollback()
        assert db.get(ContextPackage, source_id) is not None and db.get(ContextPackage, target_id) is not None
