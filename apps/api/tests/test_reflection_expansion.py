from copy import deepcopy

import pytest
from sqlalchemy import select

from app.models import ContextPackage, ContextPackageSourceExpansion, ChunkStructureMapping
from app.services.reflection_context import restore_reflection_context
from app.services.reflection_expansion import select_source_expansions
from test_citation_provenance import _build_package
from test_reflection_sources import audit_package


@pytest.mark.parametrize("text,expected", [
    ("Find §4.4 measurements", [{"section": [4, 4], "at_or_after": False}]),
    ("正文第3节及以后", [{"section": [3], "at_or_after": True}]),
    ("section 4 details", [{"section": [4], "at_or_after": False}]),
    ("第3页之后的文字", []),
    ("§1234 and section 1.2.3.4.5.6.7", []),
])
def test_restoration_addresses_are_explicit_sections_not_incidental_numbers(text, expected):
    from app.services.reflection_expansion import restoration_section_locators
    assert restoration_section_locators([text]) == expected


@pytest.mark.parametrize("focus", [False, [1], [""], ["a\x00b"], ["x" * 181], ["same", "same"], [str(i) for i in range(9)]])
def test_restoration_focus_is_bounded_before_source_access(focus):
    from app.services.agent_reflection import ReflectionContractError
    with pytest.raises(ReflectionContractError, match="focus_invalid"):
        select_source_expansions(None, source_package=None, anchor_ids=[], query_facets={}, per_anchor_budget=0, restoration_focus=focus)


def test_locator_count_is_bounded_within_one_focus_entry():
    from app.services.agent_reflection import ReflectionContractError
    with pytest.raises(ReflectionContractError, match="locator_capacity"):
        select_source_expansions(None, source_package=None, anchor_ids=[], query_facets={}, per_anchor_budget=0,
            restoration_focus=[" ".join(f"§{i}" for i in range(9))])


@pytest.mark.asyncio
async def test_summary_continuation_reserves_an_existing_slot_for_requested_body(db_session, sample_knowledge_base, tmp_path, fake_model_stack):
    from app.services.parsers import ParsedSection
    sections = [
        ParsedSection(title="Executive Summary", section="Executive Summary", page_number=1,
            text="Photon calibration summary overview. " + "Additional photon calibration summary information. " * 100),
        ParsedSection(title="A bold introductory sentence", section="A bold introductory sentence", page_number=2,
            text="The detailed photon calibration summary specifies 73 seconds."),
        ParsedSection(title="3 Background", section="3 Background", page_number=3,
            text="Unrelated routine operations background. " * 50),
        ParsedSection(title="4 Photon calibration measurements", section="4 Photon calibration measurements", page_number=4,
            text="The body records a photon calibration measurement at 92 seconds."),
    ]
    _, _, _, chunks, source, _, _, _ = await _build_package(db_session, sample_knowledge_base, tmp_path,
        source_text="\n\n".join(section.text for section in sections), parsed_sections=sections)
    assert not any("73 seconds" in item["content"] or "92 seconds" in item["content"] for item in source.package_json["chunks"])
    anchors = [item["chunk_id"] for item in source.package_json["chunks"][:2]]
    assert len(anchors) == 2
    query_facets = {"facet_groups": [{"facet": "photon calibration", "role": "domain", "aliases": []},
        {"facet": "summary and detailed comparison", "role": "procedure", "aliases": ["summary", "detailed"]}]}
    choices = select_source_expansions(db_session, source_package=source, anchor_ids=anchors,
        query_facets=query_facets, per_anchor_budget=2,
        restoration_focus=["Complete summary details", "Body photon calibration in section 4"])
    selected = {item["chunk_id"] for item in choices}
    text = "\n".join(chunk.text for chunk in chunks if chunk.id in selected)
    assert "73 seconds" in text and "92 seconds" in text
    assert len(choices) <= 4
    assert any(item["navigation"]["summary_continuation"] is True for item in choices)
    for anchor in anchors:
        assert sum(item["anchor_chunk_id"] == anchor for item in choices) <= 2


@pytest.mark.asyncio
async def test_reflection_focus_restores_requested_section_and_replays_independently(db_session, sample_knowledge_base, tmp_path, fake_model_stack, monkeypatch):
    import importlib
    from app.services.parsers import ParsedSection
    from app.services.agent_reflection import reflection_hash
    from app.services.reflection_expansion import expansion_identity
    from test_quality_gate_scripts import SCRIPTS_ROOT, _load_quality_gate
    sections = [
        ParsedSection(title="Executive Summary", section="Executive Summary", page_number=1,
            text="Photon calibration summary uses 41 seconds. " + "Additional photon calibration summary context. " * 12),
        ParsedSection(title="3 Background", section="3 Background", page_number=2,
            text="Unrelated routine operations background. " * 12),
        ParsedSection(title="4 Photon calibration measurements", section="4 Photon calibration measurements", page_number=3,
            text="The detailed measurement records a photon calibration integration of 73 seconds."),
    ]
    _, _, _, chunks, source, _, _, _ = await _build_package(db_session, sample_knowledge_base, tmp_path,
        source_text="\n\n".join(section.text for section in sections), parsed_sections=sections)
    from app.services import agent_graph as ag, context_graph as graph
    from app.schemas import SearchFilters
    await graph.write_contextual_indexes(db_session, knowledge_base=sample_knowledge_base, chunks=chunks)
    await graph.rebuild_context_graph(db_session, sample_knowledge_base.id)
    db_session.commit()
    envelope = ag.agent_operating_envelope()
    actions = ag.fallback_typed_actions("Photon calibration summary", envelope)
    next(action for action in actions if action["action_type"] == "recall_chunks")["target_ids"] = [chunks[0].id]
    actions, valid = ag.validate_typed_actions(actions, envelope, db=db_session, knowledge_base_id=sample_knowledge_base.id, retrieval_granularity="mid")
    assert valid["valid"]
    controls = ag.compile_typed_action_execution_controls(actions, envelope, requested_result_top_k=1,
        retrieval_granularity="mid", validation_diagnostics=valid)
    result = await graph.layered_search(db_session, sample_knowledge_base.id, "Photon calibration summary", SearchFilters(), 1,
        typed_action_controls=controls, allow_cache_read=False)
    source = graph.build_context_package(db_session, knowledge_base_id=sample_knowledge_base.id, query=result.trace.query,
        trace=result.trace, results=result.results, token_budget=2400, restore_per_chunk_budget=0, reserved_token_budget=2390)
    assert not any("73 seconds" in item["content"] for item in source.package_json["chunks"])
    focus = ["Photon calibration measurements in section 4, compared with the summary"]
    restored, _ = restore_reflection_context(db_session, source_package=source, target_chunk_ids=[chunks[0].id],
        preserve_chunk_ids=[chunks[0].id], token_budget=source.token_budget, restore_per_chunk_budget=1,
        query_facets=facets(), restoration_focus=focus)
    assert any("73 seconds" in item["content"] for item in restored.package_json["chunks"])
    assert restored.hit_chunk_ids_json == source.hit_chunk_ids_json
    assert audit_package(db_session, source)["all_valid"] and audit_package(db_session, restored)["all_valid"]
    row = db_session.scalar(select(ContextPackageSourceExpansion).where(ContextPackageSourceExpansion.target_context_package_id == restored.id))
    assert row.witness_json["navigation"]["restoration_focus"] == focus
    assert row.witness_json["navigation"]["rank_protocol"] == "source_structure_facet_navigation_v3"
    historical = deepcopy(row.witness_json)
    historical["navigation"]["rank_protocol"] = "source_structure_facet_navigation_v2"
    historical["navigation"].pop("continuation_direction")
    historical["navigation"].pop("continuation_anchor_chunk_id")
    historical["navigation"].pop("summary_window")
    row.witness_json = historical
    row.witness_hash = reflection_hash(expansion_identity(row))
    db_session.flush()
    assert audit_package(db_session, restored)["all_valid"]
    monkeypatch.syspath_prepend(str(SCRIPTS_ROOT))
    checker, gate = importlib.import_module("check_context_package_quality"), _load_quality_gate()
    snapshot = checker.persisted_context_package_quality_snapshot(db_session, restored.id)
    quality = gate.audit_context_package_quality(snapshot)
    assert quality["pass"], quality["findings"]
    corrupted = deepcopy(snapshot)
    corrupted["source_expansions"][0]["current_section_match"]["node"]["title"] = "99 Different section"
    assert not gate.audit_context_package_quality(corrupted)["pass"]
    assert select_source_expansions(db_session, source_package=source, anchor_ids=[chunks[0].id], query_facets=facets(),
        per_anchor_budget=1, restoration_focus=["Missing section 99 photon calibration"]) == []
    witness = deepcopy(row.witness_json)
    witness["navigation"]["restoration_focus"] = ["Missing section 99 photon calibration"]
    row.witness_json = witness
    row.witness_hash = reflection_hash(expansion_identity(row))
    db_session.flush()
    assert not audit_package(db_session, restored)["all_valid"]


@pytest.mark.asyncio
async def test_explicit_next_fragment_precedes_distant_section_search_and_v2_replays(db_session, sample_knowledge_base, tmp_path, fake_model_stack):
    _, _, _, chunks, source, _, _, _ = await _build_package(db_session, sample_knowledge_base, tmp_path,
        source_text="Photon calibration context remains addressable across chunk boundaries. " * 100)
    by_id = {chunk.id: chunk for chunk in chunks}
    anchor = max((by_id[item["chunk_id"]] for item in source.package_json["chunks"]), key=lambda chunk: chunk.chunk_index)
    assert anchor.next_chunk_id and anchor.next_chunk_id not in {item["chunk_id"] for item in source.package_json["chunks"]}
    kwargs = dict(source_package=source, anchor_ids=[anchor.id], query_facets=facets(), per_anchor_budget=1,
        restoration_focus=["Truncated passage: read the next chunk before searching section 99"])
    selected = select_source_expansions(db_session, **kwargs)
    assert [item["chunk_id"] for item in selected] == [anchor.next_chunk_id]
    assert selected[0]["navigation"]["continuation_direction"] == "next"
    assert select_source_expansions(db_session, **kwargs, _replay_rank_protocol="source_structure_facet_navigation_v2") == []


async def long_source(db, kb, path):
    text = "# Executive Summary\nPhoton calibration requires a reference exposure of 41 seconds.\n\n"
    text += "Unrelated intermediate material describes routine operations. " * 65
    text += "\n\nThe late report section mentions photon calibration without giving its exposure."
    return await _build_package(db, kb, path, source_text=text, hit_index=-1)


def facets():
    return {"facet_groups": [{"facet": "summary calibration", "aliases": ["summary", "photon calibration"]}]}


@pytest.mark.asyncio
async def test_navigation_restores_budget_skipped_hit_using_original_trace_authority(
    db_session, populated_context_graph, monkeypatch,
):
    import importlib
    from app.models import Chunk
    from app.schemas import ContextPackageResponse
    from app.services.retrieval import get_context_package
    from test_context_packing import small_package
    from test_quality_gate_scripts import SCRIPTS_ROOT, _load_quality_gate

    result, source = await small_package(db_session, populated_context_graph)
    original_hits = list(result.trace.result_chunk_ids_json)
    original_paths = deepcopy(result.trace.path_labels_json)
    original_package = deepcopy(source.package_json)
    omitted = next(cid for cid in original_hits if cid not in source.hit_chunk_ids_json)
    chunk = db_session.get(Chunk, omitted)
    packet = {"facet_groups": [{"facet": chunk.text, "role": "domain", "aliases": []}]}
    anchors = [source.hit_chunk_ids_json[0]]
    selected = select_source_expansions(db_session, source_package=source, anchor_ids=anchors,
        query_facets=packet, per_anchor_budget=1)
    assert [item["chunk_id"] for item in selected] == [omitted]
    restored, _ = restore_reflection_context(db_session, source_package=source,
        target_chunk_ids=anchors, preserve_chunk_ids=[], token_budget=2400,
        restore_per_chunk_budget=1, query_facets=packet)
    item = next(item for item in restored.package_json["chunks"] if item["chunk_id"] == omitted)
    assert item["role"] == "hit" and item["content"] == chunk.text
    assert restored.hit_chunk_ids_json == original_hits
    assert omitted not in restored.restored_chunk_ids_json
    assert not list(db_session.scalars(select(ContextPackageSourceExpansion).where(
        ContextPackageSourceExpansion.target_context_package_id == restored.id)))
    assert "source_expansion" not in restored.diagnostics_json
    assert audit_package(db_session, restored)["all_valid"]
    assert audit_package(db_session, source)["all_valid"]
    assert result.trace.result_chunk_ids_json == original_hits
    assert result.trace.path_labels_json == original_paths and source.package_json == original_package
    ContextPackageResponse.model_validate(get_context_package(db_session, restored.id))
    monkeypatch.syspath_prepend(str(SCRIPTS_ROOT))
    checker = importlib.import_module("check_context_package_quality")
    report = _load_quality_gate().audit_context_package_quality(
        checker.persisted_context_package_quality_snapshot(db_session, restored.id))
    assert report["pass"], report["findings"]


@pytest.mark.parametrize("missing", ["package_hit", "trace_hit", "trace_knowledge_base"])
def test_expansion_writer_does_not_accept_forged_native_hit(missing):
    from types import SimpleNamespace
    from app.services.agent_reflection import ReflectionContractError
    from app.services.reflection_expansion import persist_source_expansions

    item = {"chunk_id": "unit-test-hit", "role": "hit"}
    target = SimpleNamespace(package_json={"chunks": [item]}, knowledge_base_id="unit-test-kb",
        retrieval_trace_id="unit-test-trace",
        hit_chunk_ids_json=[] if missing == "package_hit" else [item["chunk_id"]])
    trace = SimpleNamespace(knowledge_base_id="different" if missing == "trace_knowledge_base" else target.knowledge_base_id,
        result_chunk_ids_json=[] if missing == "trace_hit" else [item["chunk_id"]])
    source = SimpleNamespace(package_json={"chunks": [{"chunk_id": "unit-test-anchor"}]})
    db = SimpleNamespace(get=lambda _model, _id: trace)
    with pytest.raises(ReflectionContractError, match="native_hit_authority_missing"):
        persist_source_expansions(db, source=source, target=target,
            witnesses=[{"anchor_chunk_id": "unit-test-anchor", "chunk_id": item["chunk_id"]}])


@pytest.mark.asyncio
async def test_selected_source_restores_remote_summary_without_new_graph_hit(db_session, sample_knowledge_base, tmp_path, fake_model_stack):
    _, _, _, chunks, source, _, _, _ = await long_source(db_session, sample_knowledge_base, tmp_path)
    original = deepcopy(source.package_json)
    candidate, _ = restore_reflection_context(db_session, source_package=source,
        target_chunk_ids=[chunks[-1].id], preserve_chunk_ids=[chunks[-1].id], token_budget=source.token_budget,
        restore_per_chunk_budget=1, query_facets=facets())
    assert candidate.hit_chunk_ids_json == source.hit_chunk_ids_json
    assert source.package_json == original
    assert any("41 seconds" in item["content"] for item in candidate.package_json["chunks"])
    row = db_session.scalar(select(ContextPackageSourceExpansion).where(ContextPackageSourceExpansion.target_context_package_id == candidate.id))
    assert row is not None and row.anchor_chunk_id == chunks[-1].id
    assert row.chunk_id != chunks[-1].previous_chunk_id
    assert audit_package(db_session, candidate)["all_valid"]
    db_session.commit()
    db_session.expire_all()
    assert audit_package(db_session, db_session.get(ContextPackage, candidate.id))["all_valid"]


@pytest.mark.asyncio
async def test_expansion_public_quality_and_copied_witness_replay(db_session, populated_context_graph, monkeypatch):
    import importlib
    from test_quality_gate_scripts import SCRIPTS_ROOT, _load_quality_gate
    from app.services.context_graph import layered_search, build_context_package
    from app.services.retrieval import get_context_package
    from app.schemas import ContextPackageResponse, SearchFilters
    kb = populated_context_graph["knowledge_base"]
    query = "Bayesian network factorization"
    result = await layered_search(db_session, kb.id, query, SearchFilters(), 1)
    source = build_context_package(db_session, knowledge_base_id=kb.id, query=query, trace=result.trace,
        results=result.results, token_budget=2400, restore_per_chunk_budget=0)
    anchor = source.hit_chunk_ids_json[0]
    packet = {"facet_groups": [{"facet": "Bayesian", "aliases": ["Markov", "posterior", "table"]}]}
    candidate, _ = restore_reflection_context(db_session, source_package=source,
        target_chunk_ids=[anchor], preserve_chunk_ids=[], token_budget=source.token_budget,
        restore_per_chunk_budget=1, query_facets=packet)
    ContextPackageResponse.model_validate(get_context_package(db_session, candidate.id))
    monkeypatch.syspath_prepend(str(SCRIPTS_ROOT))
    checker, gate = importlib.import_module("check_context_package_quality"), _load_quality_gate()
    snapshot = checker.persisted_context_package_quality_snapshot(db_session, candidate.id)
    report = gate.audit_context_package_quality(snapshot)
    assert report["pass"], report
    snapshot["source_expansions"][0]["current_structure"]["target_mapping"]["mapping_weight"] += 0.25
    assert not gate.audit_context_package_quality(snapshot)["pass"]
    # A second restoration keeps the existing remote span and copies its original authority.
    again, _ = restore_reflection_context(db_session, source_package=candidate,
        target_chunk_ids=[anchor], preserve_chunk_ids=[item["chunk_id"] for item in candidate.package_json["chunks"]],
        token_budget=candidate.token_budget, restore_per_chunk_budget=0, query_facets=packet)
    assert audit_package(db_session, again)["all_valid"]
    copied = db_session.scalar(select(ContextPackageSourceExpansion).where(ContextPackageSourceExpansion.target_context_package_id == again.id))
    assert copied is not None and copied.source_context_package_id == source.id
    assert gate.audit_context_package_quality(checker.persisted_context_package_quality_snapshot(db_session, again.id))["pass"]

    # A subsequent real retrieval can reach the previously expanded source as
    # a native hit. Copying its old witness must not give it two source roles.
    from app.services import agent_graph as ag
    from app.services.reflection_expansion import copy_source_expansions
    envelope = ag.agent_operating_envelope()
    actions = ag.fallback_typed_actions(query, envelope)
    next(action for action in actions if action["action_type"] == "recall_chunks")["target_ids"] = [copied.chunk_id]
    actions, valid = ag.validate_typed_actions(actions, envelope, db=db_session,
        knowledge_base_id=kb.id, retrieval_granularity="mid")
    assert valid["valid"]
    controls = ag.compile_typed_action_execution_controls(actions, envelope, requested_result_top_k=1,
        retrieval_granularity="mid", validation_diagnostics=valid)
    reached = await layered_search(db_session, kb.id, query, SearchFilters(), 1,
        typed_action_controls=controls, allow_cache_read=False)
    assert copied.chunk_id in reached.trace.result_chunk_ids_json
    native = build_context_package(db_session, knowledge_base_id=kb.id, query=query,
        trace=reached.trace, results=reached.results, token_budget=2400, restore_per_chunk_budget=0)
    assert copied.chunk_id in native.hit_chunk_ids_json
    native.diagnostics_json = {**native.diagnostics_json,
        "source_expansion": deepcopy(candidate.diagnostics_json["source_expansion"])}
    copy_source_expansions(db_session, source=candidate, target=native)
    assert "source_expansion" not in native.diagnostics_json
    assert not list(db_session.scalars(select(ContextPackageSourceExpansion).where(
        ContextPackageSourceExpansion.target_context_package_id == native.id)))
    assert audit_package(db_session, native)["all_valid"]
    ContextPackageResponse.model_validate(get_context_package(db_session, native.id))
    native_snapshot = checker.persisted_context_package_quality_snapshot(db_session, native.id)
    assert gate.audit_context_package_quality(native_snapshot)["pass"]
    assert audit_package(db_session, candidate)["all_valid"]


@pytest.mark.asyncio
async def test_expansion_replays_selection_budget_even_when_hash_is_updated(db_session, sample_knowledge_base, tmp_path, fake_model_stack):
    from app.services.agent_reflection import reflection_hash
    from app.services.reflection_expansion import expansion_identity
    _, _, _, chunks, source, _, _, _ = await long_source(db_session, sample_knowledge_base, tmp_path)
    candidate, _ = restore_reflection_context(db_session, source_package=source,
        target_chunk_ids=[chunks[-1].id], preserve_chunk_ids=[], token_budget=source.token_budget,
        restore_per_chunk_budget=1, query_facets=facets())
    row = db_session.scalar(select(ContextPackageSourceExpansion).where(ContextPackageSourceExpansion.target_context_package_id == candidate.id))
    witness = deepcopy(row.witness_json)
    witness["navigation"]["per_anchor_budget"] = 0
    row.witness_json = witness
    row.witness_hash = reflection_hash(expansion_identity(row))
    db_session.flush()
    assert not audit_package(db_session, candidate)["all_valid"]


@pytest.mark.parametrize("changed", ["knowledge_base_id", "document_id", "document_version_id", "state"])
def test_expansion_rejects_cross_source_scope_before_reading(db_session, changed):
    from types import SimpleNamespace
    from app.services.agent_reflection import ReflectionContractError
    from app.services.reflection_expansion import structure_witness
    fields = dict(knowledge_base_id="unit-test-kb", document_id="unit-test-doc", document_version_id="unit-test-version", state="active")
    with pytest.raises(ReflectionContractError, match="structure_expansion_source_scope_invalid"):
        structure_witness(db_session, SimpleNamespace(**fields), SimpleNamespace(**{**fields, changed: "different"}))


@pytest.mark.asyncio
async def test_budget_filtered_package_has_exact_public_scope(db_session, populated_context_graph, monkeypatch):
    import importlib
    from test_quality_gate_scripts import SCRIPTS_ROOT, _load_quality_gate
    from app.services.context_graph import layered_search, build_context_package
    from app.services.retrieval import get_context_package
    from app.schemas import ContextPackageResponse, SearchFilters
    kb = populated_context_graph["knowledge_base"]
    query = "Bayes theorem prior posterior"
    result = await layered_search(db_session, kb.id, query, SearchFilters(), 3)
    original_hits = list(result.trace.result_chunk_ids_json)
    package = build_context_package(db_session, knowledge_base_id=kb.id, query=query, trace=result.trace,
        results=result.results, token_budget=5, restore_per_chunk_budget=2)
    assert package.diagnostics_json["token_budget_audit"]["skipped_chunk_ids"]
    assert result.trace.result_chunk_ids_json == original_hits
    source_audit = audit_package(db_session, package)
    assert source_audit["all_valid"], sorted({reason for row in source_audit["audits"] for reason in row["reasons"]})
    ContextPackageResponse.model_validate(get_context_package(db_session, package.id))
    monkeypatch.syspath_prepend(str(SCRIPTS_ROOT))
    checker, gate = importlib.import_module("check_context_package_quality"), _load_quality_gate()
    report = gate.audit_context_package_quality(checker.persisted_context_package_quality_snapshot(db_session, package.id))
    assert report["pass"], report


@pytest.mark.asyncio
async def test_expansion_structure_tamper_invalidates_source_proof(db_session, sample_knowledge_base, tmp_path, fake_model_stack):
    _, _, _, chunks, source, _, _, _ = await long_source(db_session, sample_knowledge_base, tmp_path)
    candidate, _ = restore_reflection_context(db_session, source_package=source,
        target_chunk_ids=[chunks[-1].id], preserve_chunk_ids=[], token_budget=source.token_budget,
        restore_per_chunk_budget=1, query_facets=facets())
    row = db_session.scalar(select(ContextPackageSourceExpansion).where(ContextPackageSourceExpansion.target_context_package_id == candidate.id))
    mapping = db_session.get(ChunkStructureMapping, row.witness_json["structure"]["target_mapping"]["id"])
    mapping.mapping_weight = mapping.mapping_weight + 0.125
    db_session.flush()
    assert not audit_package(db_session, candidate)["all_valid"]


@pytest.mark.asyncio
async def test_structure_expansion_selection_keeps_existing_per_anchor_budget(db_session, sample_knowledge_base, tmp_path, fake_model_stack):
    _, _, _, chunks, source, _, _, _ = await long_source(db_session, sample_knowledge_base, tmp_path)
    assert select_source_expansions(db_session, source_package=source, anchor_ids=[chunks[-1].id], query_facets=facets(), per_anchor_budget=0) == []
    assert len(select_source_expansions(db_session, source_package=source, anchor_ids=[chunks[-1].id], query_facets=facets(), per_anchor_budget=1)) == 1
