from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest

from app.services.graph_build_workspace import GraphBuildWorkspace, current_workspace, numeric_distances


def inputs(n=32, d=16):
    matrix = np.random.default_rng(19).normal(size=(n, d))
    chunks = [SimpleNamespace(id=f"unit-test-{i:05d}", document_id=f"doc-{i%3}") for i in range(n)]
    return chunks, {chunk.id: row.tolist() for chunk, row in zip(chunks, matrix)}


@pytest.mark.parametrize("n,d", [(1, 1), (32, 16), (256, 64), (512, 16)])
def test_exact_workspace_matches_scalar_and_prepares_once(n, d):
    from app.services.context_graph import cosine_similarity
    chunks, vectors = inputs(n, d)
    with GraphBuildWorkspace() as workspace:
        workspace.bind(chunks, vectors)
        first = workspace.scores.copy()
        workspace.bind(chunks, vectors)
        assert workspace.counts["similarity_preparations"] == 1
        for row in range(min(n, 20)):
            for column in range(min(n, 20)):
                assert first[row, column] == pytest.approx(cosine_similarity(vectors[chunks[row].id], vectors[chunks[column].id]), abs=1e-12)
        assert all(len(row) == n-1 for row in workspace.rows().values())
        with pytest.raises(RuntimeError, match="identity changed"):
            workspace.bind(list(reversed(chunks)), dict(vectors))
    assert current_workspace() is None


def test_memmap_matches_memory_and_cleans_on_failure(tmp_path):
    chunks, vectors = inputs(1800, 8)
    with GraphBuildWorkspace(memory_mb=64) as workspace:
        workspace.bind(chunks, vectors)
        expected = workspace.scores.copy()
    with pytest.raises(ValueError, match="unit-test failure"):
        with GraphBuildWorkspace(memory_mb=32, scratch_root=tmp_path) as workspace:
            workspace.bind(chunks, vectors)
            assert isinstance(workspace.scores, np.memmap)
            np.testing.assert_array_equal(expected, workspace.scores)
            raise ValueError("unit-test failure")
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("change", ["value", "shape"])
def test_workspace_rejects_in_place_vector_changes(change):
    chunks, vectors = inputs(8, 4)
    with GraphBuildWorkspace() as workspace:
        workspace.bind(chunks, vectors)
        if change == "value":
            vectors[chunks[0].id][0] += .25
        else:
            vectors[chunks[0].id].append(.25)
        with pytest.raises(RuntimeError, match="vector"):
            workspace.bind(chunks, vectors)


def test_structure_positive_cache_is_probe_bound_and_returns_independent_results(monkeypatch):
    from app.services import auto_tpe
    chunks, vectors = inputs(8, 4)
    calls = []
    def prepare(db, scope, probes, **kwargs):
        calls.append([probe.id for probe in probes])
        return {probes[0].id: {"same_page": {scope[-1].id}}}, {scope[-1].id}, {"valid": True}
    monkeypatch.setattr(auto_tpe, "_structure_positive_context_uncached", prepare)
    with GraphBuildWorkspace() as workspace:
        workspace.bind(chunks, vectors)
        first = auto_tpe._structure_positive_context(None, chunks, chunks[:1])
        first[0][chunks[0].id]["same_page"].clear()
        first[2]["valid"] = False
        repeated = auto_tpe._structure_positive_context(None, chunks, chunks[:1])
        assert repeated[0][chunks[0].id]["same_page"] == {chunks[-1].id}
        assert repeated[2]["valid"]
        assert len(calls) == 1
        changed = auto_tpe._structure_positive_context(None, chunks, chunks[1:2])
        assert chunks[1].id in changed[0]
        auto_tpe._structure_positive_context(None, chunks, chunks[:1], chunk_business_keys={chunks[0].id: "unit-test-new-key"})
        assert len(calls) == 3


def test_trial_invariant_quality_is_prepared_once_without_reusing_threshold_signals(db_session, sample_knowledge_base, monkeypatch):
    from app.services import context_graph as graph
    from test_relation_quota_signals import _add_document_chunks
    _, _, chunks = _add_document_chunks(db_session, sample_knowledge_base.id,
        suffix="unit-test-intrinsic", chunk_ids=[f"unit-test-intrinsic-{i}" for i in range(12)])
    matrix = np.random.default_rng(42).normal(size=(len(chunks), 8)) + 1.0
    vectors = {chunk.id: row.tolist() for chunk, row in zip(chunks, matrix)}
    theta = graph.dense_graph_operating_point()
    changed = {**theta, "dense_min_cosine": .65, "cross_doc_min_cosine": .65}
    expected = [graph.relation_edge_candidates(db_session, chunks, vectors, item)[0] for item in (theta, changed)]
    original = graph._availability_weighted_signal_card
    calls = []
    def observed(**kwargs):
        if kwargs["protocol_version"] == graph.CHUNK_NODE_QUALITY_PROTOCOL_VERSION:
            calls.append(1)
        return original(**kwargs)
    monkeypatch.setattr(graph, "_availability_weighted_signal_card", observed)
    with GraphBuildWorkspace():
        for config, oracle in zip((theta, changed), expected):
            actual, diagnostics = graph.relation_edge_candidates(db_session, chunks, vectors, config)
            assert set(actual) == set(oracle)
            assert all(actual[key].raw_strength == oracle[key].raw_strength for key in oracle)
            assert diagnostics["candidate_intent_count"] >= len(actual)
    assert len(calls) == len(chunks)


def test_boundary_distances_duplicates_and_empty_input():
    vectors = np.asarray([[0., 0.], [1., 1.], [1e-15, 0.]])
    centers = np.asarray([[-1., 0.], [1., 0.], [1., 0.]])
    distances = numeric_distances(vectors, centers)
    assert np.argmin(distances, axis=1).tolist() == [0, 1, 1]
    assert numeric_distances(np.empty((0, 2)), centers).shape == (0, 3)


def test_numeric_block_latency_records_real_samples_without_changing_results():
    from app.services.build_performance import BuildPerformance, _CURRENT
    chunks, vectors = inputs(256, 16)
    matrix = np.asarray(list(vectors.values()))
    centers = matrix[:6]
    expected = numeric_distances(matrix, centers)
    performance = BuildPerformance()
    token = _CURRENT.set(performance)
    try:
        with GraphBuildWorkspace() as workspace:
            workspace.bind(chunks, vectors)
            observed = numeric_distances(matrix, centers)
        np.testing.assert_array_equal(observed, expected)
    finally:
        _CURRENT.reset(token)
    for name in ("similarity_block", "rq_distance_block"):
        stage = performance.summary()["stages"][name]
        assert stage["sample_count"] == stage["success_count"] == 2
        assert stage["failure_count"] == 0
        assert 0 <= stage["p50_ms"] <= stage["p95_ms"] <= stage["p99_ms"]


@pytest.mark.parametrize("digits",[6,12])
def test_decimal_rounding_matches_scalar_bits_at_halfway_boundaries(digits):
    from app.services.graph_build_workspace import exact_decimal_round
    values=np.concatenate([np.random.default_rng(3).normal(size=20000),
        (np.arange(-10000,10000,dtype=float)+.5)/(10**digits),np.asarray([-.0,1e100,-1e100,16.055,56294995342131.5])]).reshape(1,-1)
    expected=np.asarray([[round(float(value),digits) for value in values[0]]])
    with GraphBuildWorkspace() as workspace:
        actual=exact_decimal_round(values,digits)
        assert workspace.counts["decimal_round_refinements"]>0
    np.testing.assert_array_equal(actual.view(np.uint64),expected.view(np.uint64))


@pytest.mark.parametrize("case", ["normal", "duplicates", "near_tie", "single", "empty"])
def test_rq_training_and_primary_chain_match_frozen_scalar(case):
    import sys
    from pathlib import Path
    sys.path.insert(0,str(Path(__file__).resolve().parents[3]/"scripts"))
    from _graph_scalar_reference import train_codebooks, primary_path
    from app.services.context_graph import train_rq_kmeans, encode_rq_vectors_batch
    matrix = np.random.default_rng(103).normal(size=(48,12))
    if case == "duplicates": matrix[:] = matrix[0]
    if case == "near_tie": matrix = np.asarray([[1.,0.],[-1.,0.],[0.,1e-15],[0.,-1e-15]]*12)
    if case == "single": matrix = matrix[:1]
    if case == "empty": matrix = matrix[:0]
    vectors = matrix.tolist()
    expected = train_codebooks(vectors)
    with GraphBuildWorkspace():
        model = train_rq_kmeans(vectors,levels=3,max_k=6,tau_r=.65,tau_l=.35)
        encoded,_ = encode_rq_vectors_batch([(str(i),row) for i,row in enumerate(vectors)],model)
    assert len(model["codebooks"]) == len(expected)
    for actual,reference in zip(model["codebooks"],expected):
        np.testing.assert_allclose(actual,reference,rtol=1e-12,atol=1e-12)
    for i,vector in enumerate(vectors):
        assert encoded[str(i)]["rq_path"] == primary_path(vector,expected)
        assert len(encoded[str(i)]["prefix_memberships"]) == 3


def test_support_index_preserves_order_and_limit():
    from app.services.context_graph import support_chunk_edge_index, support_chunk_edge_ids_for_chunks
    chunks, _ = inputs(24)
    keys = {c.id: f"business-{i:05d}" for i,c in enumerate(reversed(chunks))}
    edges = {(a.id,b.id,"dense_semantic"):SimpleNamespace(id=f"e-{i}-{j}", source_chunk_id=a.id,target_chunk_id=b.id,edge_type="dense_semantic")
             for i,a in enumerate(chunks) for j,b in enumerate(chunks) if i<j}
    with GraphBuildWorkspace() as workspace:
        index = support_chunk_edge_index(edges, chunk_business_keys=keys)
        for chunk in chunks:
            assert index[chunk.id] == support_chunk_edge_ids_for_chunks({chunk.id},edges,chunk_business_keys=keys)[:16]
        assert workspace.counts["support_edge_sort_passes"] == 1
        assert workspace.counts["support_edge_visits"] == len(edges)


def test_structure_address_prefilter_preserves_all_possible_admissions():
    from app.services.context_graph import StructureMappingIndex, _bbox_iou_score, _structure_mapping_admission, _is_exact_canonical_section_path, _path_match_score
    nodes = []
    bbox = {"x0":0.1,"y0":0.1,"x1":0.8,"y1":0.8,"coordinate_system":"normalized_page_v1","page_number":2}
    for i in range(100):
        nodes.append(SimpleNamespace(id=str(i),knowledge_base_id="kb",document_id="doc",document_version_id="version",node_type="paragraph",
            char_start=i*10,char_end=i*10+15,page_number=i%5,bbox_json={**bbox,"page_number":i%5},layout_json={},path="document / section"))
    for i in range(2):
        nodes.append(SimpleNamespace(id=f"section-{i}",knowledge_base_id="kb",document_id="doc",document_version_id="version",node_type="section",
            char_start=None,char_end=None,page_number=None,bbox_json={},layout_json={},path="document / section"))
    chunk=SimpleNamespace(knowledge_base_id="kb",document_id="doc",document_version_id="version",char_start=330,char_end=360,section_path="section")
    coordinates=[{"bbox":bbox,"page_number":2,"coordinate_system":"normalized_page_v1"}]
    index=StructureMappingIndex(nodes)
    candidates={node.id for node in index.candidates(chunk,coordinates)}
    exact={node.id for node in index.sections if _is_exact_canonical_section_path(chunk.section_path,node.path)}
    for node in nodes:
        overlap=max(0,min(chunk.char_end,node.char_end)-max(chunk.char_start,node.char_start)) if node.char_start is not None else 0
        iou,_=_bbox_iou_score(coordinates,node)
        path,_=_path_match_score(chunk.section_path,node.path)
        admitted,_=_structure_mapping_admission(chunk=chunk,node=node,overlap=overlap,span_available=node.char_start is not None,
            bbox_iou=iou,path_match=path,exact_section_candidate_ids=exact)
        if admitted: assert node.id in candidates
    assert len(candidates) < len(nodes)//2


@pytest.mark.asyncio
async def test_concept_packet_context_retains_isolated_ineligible_scope(db_session,populated_context_graph):
    from sqlalchemy import select
    from app.models import ChunkRelationGraphState, RQPrefix
    from app.services.context_graph import prepare_concept_packet_build_context
    from test_relation_quota_signals import _add_document_chunks
    kb = populated_context_graph["knowledge_base"]
    relation = db_session.scalar(select(ChunkRelationGraphState).where(ChunkRelationGraphState.knowledge_base_id == kb.id,ChunkRelationGraphState.state == "active"))
    _,_,extra = _add_document_chunks(db_session,kb.id,suffix="isolated-numeric",chunk_ids=["unit-test-isolated-no-concept"])
    relation.active_chunk_ids_json = [*relation.active_chunk_ids_json,extra[0].id]
    db_session.flush()
    cluster = db_session.scalar(select(RQPrefix).where(RQPrefix.graph_state_id == relation.id,RQPrefix.rq_level == 3))
    context = prepare_concept_packet_build_context(db_session,[cluster])
    assert set(context.chunk_by_id) == set(relation.active_chunk_ids_json)
    assert extra[0].id in context.chunk_business_keys
    assert extra[0].id not in context.relation_edges_by_chunk


def test_vectorized_candidates_match_scalar(db_session, sample_knowledge_base):
    from test_relation_quota_signals import _add_document_chunks
    from app.services.context_graph import relation_edge_candidates, dense_graph_operating_point
    chunks = []
    for group in range(3):
        _,_,rows = _add_document_chunks(db_session,sample_knowledge_base.id,suffix=f"numeric-{group}",chunk_ids=[f"numeric-{group}-{i}" for i in range(8)])
        chunks.extend(rows)
    matrix = np.random.default_rng(35).normal(size=(len(chunks),16)) + 1.4
    vectors = {chunk.id:row.tolist() for chunk,row in zip(chunks,matrix)}
    theta = dense_graph_operating_point()
    expected, expected_diagnostics = relation_edge_candidates(db_session,chunks,vectors,theta)
    with GraphBuildWorkspace() as workspace:
        actual, diagnostics = relation_edge_candidates(db_session,chunks,vectors,theta)
        assert set(actual) == set(expected)
        for key in expected:
            assert actual[key].raw_strength == expected[key].raw_strength
            assert actual[key].calibrated_strength == expected[key].calibrated_strength
            def compare(left, right):
                if isinstance(left, dict):
                    assert left.keys() == right.keys()
                    for name in left:
                        # The versioned numeric engine can change hashes of
                        # unrounded diagnostic floats, never discrete gates.
                        if not name.endswith("hash"):
                            compare(left[name], right[name])
                elif isinstance(left, list):
                    assert len(left) == len(right)
                    for a,b in zip(left,right): compare(a,b)
                elif isinstance(left, float):
                    assert left == pytest.approx(right, rel=1e-12, abs=1e-12)
                else:
                    assert left == right
            compare(actual[key].features_json, expected[key].features_json)
        assert diagnostics["candidate_intent_count"] == expected_diagnostics["candidate_intent_count"]


@pytest.mark.parametrize("kind", ["formula", "paragraph"])
def test_pdf_artifact_labels_remove_nul_and_preserve_spans(kind):
    from app.services.parsers import ParsedSection, ParsedStructureObject, _clean_section
    raw = "Unit\x00 test\x01 formula"
    artifact = ParsedStructureObject(structure_id="unit-test",object_type=kind,text=raw,
        char_start=0,char_end=len(raw),title=raw,path="Page\x00 / item")
    cleaned = _clean_section(ParsedSection(title="Unit test",text=raw,structure_objects=[artifact]), "pdf")
    obj = cleaned.structure_objects[0]
    assert "\x00" not in obj.title and "\x01" not in obj.title
    assert "\x00" not in obj.path
    assert cleaned.text[obj.char_start:obj.char_end] == obj.text
    assert raw[4] == "\x00"


def test_layout_contract_preserves_ocr_errors_and_rejects_unknown_fields():
    from pydantic import ValidationError
    from app.schemas import ContextStructureLayoutAudit
    payload = {"ocr_image_errors": ["unit-test image decode failure"]}
    assert ContextStructureLayoutAudit.model_validate(payload, strict=True).ocr_image_errors == payload["ocr_image_errors"]
    with pytest.raises(ValidationError):
        ContextStructureLayoutAudit.model_validate({"ocr_image_errors": [123]}, strict=True)
    with pytest.raises(ValidationError):
        ContextStructureLayoutAudit.model_validate({"unit_test_unknown_layout": True}, strict=True)


def test_metrics_nearest_rank_sample_counts_and_deadline():
    import time
    from app.services.build_performance import BuildPerformance
    from app.services.cancellation import IngestionCancelled
    perf = BuildPerformance(cold=True, deadline_seconds=1)
    for i in range(1, 7):
        perf.record("trial", time.perf_counter()-i/1000, True)
    summary = perf.summary()
    row = summary["stages"]["trial"]
    assert row["sample_count"] == 6
    assert row["p99_ms"] == row["p95_ms"]
    assert row["success_count"] == 6
    perf.started -= 2
    with pytest.raises(IngestionCancelled, match="build_deadline_exceeded"):
        perf.check()


def test_database_json_transport_keeps_float_bits_and_standard_edge_cases():
    import json
    from datetime import datetime
    from app.core.json_codec import database_json_dumps
    # Cover arbitrary exponents, subnormals and signed zero, not only vectors
    # near unit magnitude. Standard JSON parsing remains the DB read path.
    bits=np.random.default_rng(17).integers(0,2**64,size=20000,dtype=np.uint64)
    floats=bits.view(np.float64)
    floats=floats[np.isfinite(floats)]
    loaded=np.asarray(json.loads(database_json_dumps(floats.tolist())),dtype=np.float64)
    np.testing.assert_array_equal(loaded.view(np.uint64),floats.view(np.uint64))
    for value in [{"large":2**90},{2:"key","sequence":(1,2)}, {"zero":-0.0,"text":"unit test 中文"},{"surrogate":"\ud800"}]:
        assert json.loads(database_json_dumps(value)) == json.loads(json.dumps(value))
    for value in [float("nan"),float("inf"),float("-inf")]:
        with pytest.raises(ValueError): database_json_dumps({"value":value})
    with pytest.raises(TypeError): database_json_dumps({"value":datetime(2026,1,1)})


def test_frozen_database_json_digest_reuses_exact_transport_and_checks_serializer_identity(monkeypatch):
    import hashlib
    import json
    from app.core import json_codec
    from app.services.graph_state_hashes import freeze_graph_input
    original = json_codec._database_json_dumps
    calls = []
    def observed(value):
        calls.append(1)
        return original(value)
    monkeypatch.setattr(json_codec, "_database_json_dumps", observed)
    frozen = freeze_graph_input({"unit-test": [-0.0, 1e-19, "合成文本"]})
    encoded = json_codec.database_json_dumps(frozen)
    expected = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
    assert json_codec.database_json_digest(frozen, json_codec.database_json_dumps) == expected
    assert len(calls) == 1
    assert json_codec.database_json_digest(frozen, json.dumps) == hashlib.sha256(json.dumps(frozen).encode()).hexdigest()
    assert json_codec.database_json_digest(frozen, json_codec.database_json_dumps) == expected
    with pytest.raises(TypeError):
        frozen["unit-test"][0] = .5
    mutable = {"unit-test": 1}
    before = json_codec.database_json_digest(mutable, json_codec.database_json_dumps)
    mutable["unit-test"] = 2
    assert json_codec.database_json_digest(mutable, json_codec.database_json_dumps) != before


def test_rq_packed_vectors_round_trip_bits_and_reject_corruption():
    import base64,hashlib
    from app.services.rq_numeric_storage import pack_rq_vector,validate_rq_vector_blob,read_rq_vector
    values=np.random.default_rng(21).normal(size=1024)
    values[:3]=[0.,-0.,5e-324]
    blob=pack_rq_vector(values)
    np.testing.assert_array_equal(validate_rq_vector_blob(blob).view(np.uint64),values.view(np.uint64))
    np.testing.assert_array_equal(np.asarray(read_rq_vector(blob)).view(np.uint64),values.view(np.uint64))
    for invalid in [{**blob,"dimensions":1023},{**blob,"sha256":"0"*64},{**blob,"data":blob["data"][:-1]}, {**blob,"protocol":"unknown"}]:
        with pytest.raises(ValueError): validate_rq_vector_blob(invalid)
    invalid=pack_rq_vector([1.])
    data=np.asarray([float("nan")],dtype="<f8").tobytes()
    invalid.update(data=base64.b64encode(data).decode("ascii"),sha256=hashlib.sha256(data).hexdigest())
    with pytest.raises(ValueError,match="non-finite"):validate_rq_vector_blob(invalid)


def test_signal_pool_preserves_every_directional_card_and_rejects_bad_refs():
    from app.services.relation_signal_storage import compact_signal_features,pack_signal_pool,unpack_signal_pool,expand_features_from_pool
    source={"value":.6,"details":{"count":7,"complete":[1,2,3]}}
    target={"value":.8,"details":{"count":8}}
    fields={"source_out_signal_card":source,"target_in_acceptance_signal_card":target,
        "source_node_quality_card":source,"target_node_quality_card":target}
    original={**fields,"directed_source_chunk_id":"source","directed_target_chunk_id":"target",
        "directional_contributions":[{**fields,"source_chunk_id":"source","target_chunk_id":"target","raw_strength":.7}]}
    pool={}
    stored=compact_signal_features(original,pool)
    blob=pack_signal_pool(pool)
    restored=expand_features_from_pool(stored,unpack_signal_pool(blob),{"source","target"})
    restored.pop("node_signal_storage_protocol")
    assert restored == original
    with pytest.raises(ValueError,match="integrity"):
        unpack_signal_pool({**blob,"sha256":"0"*64})
    with pytest.raises(ValueError,match="outside"):
        expand_features_from_pool(stored,pool,{"source"})


def test_settings_compute_scope_reads_root_once_and_is_not_shadow(monkeypatch):
    from app.core import config
    original = config._settings_cache_token
    reads = []
    def observed():
        reads.append(True)
        return original()
    monkeypatch.setattr(config, "_settings_cache_token", observed)
    with config.runtime_settings_read_scope() as snapshot:
        for _ in range(20):
            assert config.get_settings() is snapshot
        assert not config.runtime_settings_override_active()
    assert len(reads) == 1
    config.get_settings()
    assert len(reads) == 2


def test_benchmark_execute_cannot_relax_cold_or_time_gates():
    import sys
    from pathlib import Path
    sys.path.insert(0,str(Path(__file__).resolve().parents[3]/"scripts"))
    from benchmark_build_pipeline import validate_execution
    for controls in [dict(execute=True,cold=False,full_reparse=True,deadline_seconds=1800),dict(execute=True,cold=True,full_reparse=True,deadline_seconds=3600)]:
        with pytest.raises(ValueError): validate_execution(SimpleNamespace(**controls))
    with pytest.raises(ValueError, match="rq-reference-report"):
        validate_execution(SimpleNamespace(execute=True,cold=True,full_reparse=True,deadline_seconds=1800))
    validate_execution(SimpleNamespace(execute=True,cold=True,full_reparse=True,deadline_seconds=1800,rq_reference_report="unit-test-reference.json"))


def test_periodic_queue_coalescing_preserves_user_jobs_and_chains():
    import base64,json,sys
    from pathlib import Path
    sys.path.insert(0,str(Path(__file__).resolve().parents[3]/"scripts"))
    from coalesce_maintenance_queue import plan_messages
    def message(identifier,task="reconcile_profile_lifecycle",args=None,chain=None):
        body=[args or [],{}, {"chain":chain}]
        return json.dumps({"headers":{"id":identifier,"task":task},"body":base64.b64encode(json.dumps(body).encode()).decode()})
    messages=[message("newest"),message("user-1",task="ingest_uploaded_batch"),message("with-args",args=["unit-test"]),message("with-chain",chain=[{}]),message("duplicate")]
    duplicates, invalid = plan_messages(messages)
    assert invalid == 0
    assert [item["id"] for item in duplicates] == ["duplicate"]
    assert plan_messages([b"broken json"])[1] == 1


def test_resource_stream_accepts_windows_ansi_frames():
    import sys
    from pathlib import Path
    sys.path.insert(0,str(Path(__file__).resolve().parents[3]/"scripts"))
    from monitor_build_resources import parse_stats_line, resource_row, distribution
    assert parse_stats_line('\x1b[H{"Name":"course-kg-worker"}\x1b[K\n') == {"Name":"course-kg-worker"}
    assert parse_stats_line('\x1b[J\n') is None
    source = {"Name":"course-kg-worker","CPUPerc":"250%","MemUsage":"2.5GiB / 7GiB",
              "BlockIO":"1GB / 2GB","NetIO":"2GB / 3GB","PIDs":"16"}
    row = resource_row(source,16)
    assert row["cpu_cores"] == 2.5 and row["cpu_percent_of_available"] == 15.625
    assert row["memory_bytes"] == 2.5*1024**3
    with pytest.raises(ValueError):
        resource_row({**source,"CPUPerc":"nan%"},16)
    with pytest.raises(ValueError):
        resource_row({**source,"Name":"unrelated-container"},16)
    assert distribution([1,2,3,4,5,6])["p99"] == 6


@pytest.mark.asyncio
async def test_embedding_batches_are_bounded_and_ordered(monkeypatch):
    import asyncio
    from app.services import embeddings
    from app.services.embeddings import EmbeddingProvider
    provider = EmbeddingProvider()
    monkeypatch.setattr(provider.settings, "model_request_concurrency", 3)
    monkeypatch.setattr(provider.settings, "embedding_batch_size", 10)
    monkeypatch.setattr(embeddings, "enforce_memory_budget", lambda *_args, **_kwargs: None)
    active = peak = 0
    async def batch(texts, text_type):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(.001*(4-(int(texts[0])//10)%3))
        active -= 1
        return [[float(x)] for x in texts]
    monkeypatch.setattr(provider, "_openai_compatible_embeddings_batch", batch)
    result = await provider._openai_compatible_embeddings([str(i) for i in range(71)])
    assert result == [[float(i)] for i in range(71)]
    assert peak == 3


@pytest.mark.asyncio
async def test_source_preparation_is_one_ahead_and_failed_file_does_not_poison_batch():
    import asyncio
    from app.services.source_parse_pipeline import SourcePreparationPipeline,prepared_source
    calls=[]
    async def prepare(path):
        calls.append(path)
        if path == "bad": raise ValueError("unit-test snapshot failure")
        return path
    async with SourcePreparationPipeline(["first","bad","last"],prepare):
        assert await prepared_source("first",lambda:prepare("first")) == "first"
        await asyncio.sleep(.01)
        assert calls == ["first","bad"]
        with pytest.raises(ValueError,match="snapshot failure"):
            await prepared_source("bad",lambda:prepare("bad"))
        assert await prepared_source("last",lambda:prepare("last")) == "last"
    assert calls == ["first","bad","last"]


@pytest.mark.asyncio
async def test_source_preparation_cleanup_preserves_original_failure():
    import asyncio
    from app.services.source_parse_pipeline import SourcePreparationPipeline,prepared_source,check_parse_cancellation
    started=asyncio.Event()
    stopped=[]
    async def prepare(path):
        if path == "first": return path
        started.set()
        try:
            while True:
                check_parse_cancellation()
                await asyncio.sleep(.001)
        finally:
            stopped.append(path)
    with pytest.raises(RuntimeError,match="unit-test original"):
        async with SourcePreparationPipeline(["first","future"],prepare):
            await prepared_source("first",lambda:prepare("first"))
            await started.wait()
            raise RuntimeError("unit-test original")
    assert stopped == ["future"]
