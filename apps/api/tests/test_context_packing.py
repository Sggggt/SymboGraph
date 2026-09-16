from copy import deepcopy

import pytest


def _completion_problem(required, candidates, *, available=(), version='unit-version', budget=None):
    from app.retrieval_control_contracts import EvidenceInterval, EvidenceIntervalCandidate, EvidenceScopeFact, EvidenceScopeExpression
    from app.services.source_use import evaluate_evidence_scope
    from app.services.context_packing import plan_interval_scope_completion
    def interval(pair):
        return EvidenceInterval(knowledge_base_id='unit-kb',document_version_id=version,start=pair[0],end=pair[1])
    fact = EvidenceScopeFact(id='scope',knowledge_base_id='unit-kb',kind='text',resolution='verified',
        extent_complete=True,intervals=tuple(interval(pair) for pair in required),witness_ids=('unit-scope-witness',))
    coverage = evaluate_evidence_scope(knowledge_base_id='unit-kb',expression=EvidenceScopeExpression(op='scope',scope_id='scope'),
        scopes=(fact,),packed=tuple(interval(pair) for pair in available),mode='complete')
    choices = tuple(EvidenceIntervalCandidate(id=cid,interval=interval(pair),cost=cost,witness_ids=('unit-candidate-witness',))
                    for cid,pair,cost in candidates)
    return plan_interval_scope_completion(knowledge_base_id='unit-kb',coverage=coverage,candidates=choices,budget=budget)


def test_interval_completion_minimizes_additive_cost_and_preserves_budget():
    choices = [('large',(0,30),15),('left',(0,10),3),('right',(20,30),4)]
    plan = _completion_problem([(0,10),(20,30)],choices,budget=7)
    assert plan.status=='ready' and plan.selected_ids==('left','right') and plan.total_cost==7
    assert plan.executor_validation_required and not plan.semantic_sufficiency_claimed
    constrained = _completion_problem([(0,10),(20,30)],choices,budget=6)
    assert constrained.status=='over_budget' and constrained.total_cost==7
    assert _completion_problem([(0,10),(20,30)],list(reversed(choices))).selected_ids==plan.selected_ids


def test_completion_only_fills_declared_missing_ranges_and_reports_candidate_gaps():
    plan = _completion_problem([(0,20)],[('tail',(10,20),4)],available=[(0,10)])
    assert plan.status=='ready' and plan.selected_ids==('tail',)
    missing = _completion_problem([(0,20)],[('prefix',(0,7),1),('tail',(10,20),1)])
    assert missing.status=='not_coverable' and missing.selected_ids==()
    assert [(part.start,part.end) for part in missing.uncovered_intervals]==[(7,10)]


def test_completion_matches_small_independent_subset_oracle():
    import itertools
    import random
    rng = random.Random(431)
    targets = [(0,8),(12,18)]
    wanted = {point for left,right in targets for point in range(left,right)}
    for _ in range(25):
        candidates = [(str(index),(start,min(20,start+rng.randint(1,12))),rng.randint(0,9))
                      for index,start in enumerate(rng.sample(range(18),6))]
        optimum = None
        # Exhaustive enumeration is only this tiny independent test oracle;
        # the production planner uses the acyclic progress-state algorithm.
        for count in range(len(candidates)+1):
            for subset in itertools.combinations(candidates,count):
                covered = {point for _,(left,right),_ in subset for point in range(left,right)}
                if wanted.issubset(covered):
                    value = sum(cost for _,_,cost in subset),count
                    optimum = value if optimum is None else min(optimum,value)
        plan = _completion_problem(targets,candidates)
        if optimum is None:
            assert plan.status=='not_coverable'
        else:
            assert (plan.total_cost,len(plan.selected_ids))==optimum


def test_completion_keeps_unresolved_scope_and_cooperative_cancel(monkeypatch):
    from app.retrieval_control_contracts import EvidenceScopeCoverage
    from app.services.context_packing import plan_interval_scope_completion
    unknown = EvidenceScopeCoverage(state='unknown',mode='complete',input_hash='a'*64,
        usable_intervals=(),missing_intervals=(),scope_extent_known=False,reason='scope_not_fully_resolved')
    assert plan_interval_scope_completion(knowledge_base_id='unit-kb',coverage=unknown,candidates=()).status=='scope_unresolved'
    def cancel():
        raise InterruptedError('unit cancellation')
    monkeypatch.setattr('app.services.storage.raise_if_source_io_cancelled',cancel)
    with pytest.raises(InterruptedError):
        _completion_problem([(0,20)],[('whole',(0,20),1)])


def test_completion_solves_document_versions_independently_and_rejects_foreign_candidates():
    from app.retrieval_control_contracts import EvidenceInterval, EvidenceIntervalCandidate, EvidenceScopeCoverage
    from app.services.context_packing import plan_interval_scope_completion
    spans = tuple(EvidenceInterval(knowledge_base_id='unit-kb',document_version_id=version,start=0,end=10)
                  for version in ('document-a-v1','document-a-v2'))
    coverage = EvidenceScopeCoverage(state='unsatisfied',mode='complete',input_hash='a'*64,
        usable_intervals=(),missing_intervals=spans,scope_extent_known=True,reason='required_range_missing')
    candidates = tuple(EvidenceIntervalCandidate(id=f'candidate-{index}',interval=span,cost=index+1,witness_ids=('unit-witness',))
                       for index,span in enumerate(spans))
    plan = plan_interval_scope_completion(knowledge_base_id='unit-kb',coverage=coverage,candidates=candidates)
    assert plan.status=='ready' and plan.total_cost==3 and len(plan.selected_ids)==2
    foreign = candidates[0].model_copy(update={'interval':spans[0].model_copy(update={'knowledge_base_id':'another-kb'})})
    with pytest.raises(ValueError,match='candidate_scope_invalid'):
        plan_interval_scope_completion(knowledge_base_id='unit-kb',coverage=coverage,candidates=(foreign,))

from app.schemas import SearchFilters
from app.services.context_graph import build_context_package, layered_search
from test_reflection_sources import audit_package


@pytest.mark.parametrize("reserved", [-1, 10, 11, True])
def test_reservation_cannot_exceed_total_or_use_non_integer_amount(db_session, sample_knowledge_base, reserved):
    with pytest.raises(ValueError, match="reserved_budget_invalid"):
        build_context_package(db_session, knowledge_base_id=sample_knowledge_base.id, query="Synthetic budget check",
            trace=None, results=[], token_budget=10, reserved_token_budget=reserved)


async def small_package(db, fixture):
    kb = fixture["knowledge_base"]
    query = "Bayes theorem prior posterior"
    result = await layered_search(db, kb.id, query, SearchFilters(), 3)
    package = build_context_package(db, knowledge_base_id=kb.id, query=query, trace=result.trace,
        results=result.results, token_budget=5, restore_per_chunk_budget=2)
    assert len(package.hit_chunk_ids_json) < len(result.trace.result_chunk_ids_json)
    assert audit_package(db, package)["all_valid"]
    return result, package


@pytest.mark.asyncio
@pytest.mark.parametrize("token_budget", [5, 2400])
async def test_priority_packing_preserves_original_hit_ranking(monkeypatch, db_session, populated_context_graph, token_budget):
    import importlib
    from sqlalchemy import select
    from app.models import GraphRetrievalStep
    from test_quality_gate_scripts import SCRIPTS_ROOT, _load_quality_gate
    from app.services.context_packing import audit_context_packing
    kb = populated_context_graph["knowledge_base"]
    result = await layered_search(db_session, kb.id, "Synthetic Bayesian network comparison", SearchFilters(), 3)
    original_hits = list(result.trace.result_chunk_ids_json)
    assert len(original_hits) >= 2
    priority_results = list(reversed(result.results))
    package = build_context_package(db_session, knowledge_base_id=kb.id, query=result.trace.query, trace=result.trace,
        results=priority_results, token_budget=token_budget, restore_per_chunk_budget=0)
    actual_ids = [item["chunk_id"] for item in package.package_json["chunks"]]
    assert actual_ids[0] == priority_results[0]["chunk_id"]
    assert package.hit_chunk_ids_json == [cid for cid in original_hits if cid in actual_ids]
    assert result.trace.result_chunk_ids_json == original_hits
    assert audit_package(db_session, package)["all_valid"]
    step = db_session.scalar(select(GraphRetrievalStep).where(GraphRetrievalStep.retrieval_trace_id == result.trace.id,
        GraphRetrievalStep.layer == "structure"))
    assert step.selected_topk_ids_json == package.hit_chunk_ids_json
    monkeypatch.syspath_prepend(str(SCRIPTS_ROOT))
    checker = importlib.import_module("check_context_package_quality")
    quality = _load_quality_gate().audit_context_package_quality(checker.persisted_context_package_quality_snapshot(db_session, package.id))
    assert quality["pass"], quality["findings"]
    if token_budget == 2400:
        package.hit_chunk_ids_json = list(reversed(package.hit_chunk_ids_json))
        assert "context_package_packing_selection_mismatch" in audit_context_packing(db_session, package=package, trace=result.trace)


@pytest.mark.asyncio
@pytest.mark.parametrize("mutation", ["omit_candidate", "increase_selection_budget", "hide_skip"])
async def test_partial_hit_scope_requires_exact_packing_replay(db_session, populated_context_graph, mutation):
    result, package = await small_package(db_session, populated_context_graph)
    diagnostics = deepcopy(package.diagnostics_json)
    control = diagnostics["token_budget_audit"]
    if mutation == "omit_candidate":
        dropped_hit = next(cid for cid in result.trace.result_chunk_ids_json if cid not in package.hit_chunk_ids_json)
        control["candidate_chunk_ids"].remove(dropped_hit)
    elif mutation == "increase_selection_budget":
        control["selection_token_budget"] = 100000
    else:
        control["skipped_chunk_ids"] = []
    package.diagnostics_json = diagnostics
    db_session.flush()
    assert not audit_package(db_session, package)["all_valid"]


@pytest.mark.asyncio
async def test_restoration_keeps_original_trace_hit_domain_after_budget_skip(db_session, populated_context_graph):
    from app.services.reflection_context import restore_reflection_context
    result, package = await small_package(db_session, populated_context_graph)
    trace_hits = list(result.trace.result_chunk_ids_json)
    restored, _ = restore_reflection_context(db_session, source_package=package,
        target_chunk_ids=[package.hit_chunk_ids_json[0]], preserve_chunk_ids=[],
        token_budget=2400, restore_per_chunk_budget=0)
    assert restored.hit_chunk_ids_json == trace_hits
    assert result.trace.result_chunk_ids_json == trace_hits
    assert audit_package(db_session, restored)["all_valid"]


@pytest.mark.asyncio
async def test_scope_repacking_materializes_skipped_authorized_hit_without_search(db_session,populated_context_graph):
    from app.models import RetrievalTrace
    from app.services.reflection_context import restore_reflection_context
    result, package = await small_package(db_session,populated_context_graph)
    missing = next(cid for cid in result.trace.result_chunk_ids_json if cid not in package.hit_chunk_ids_json)
    original = deepcopy(package.package_json)
    restored,_ = restore_reflection_context(db_session,source_package=package,
        target_chunk_ids=[package.package_json['chunks'][0]['chunk_id']],preserve_chunk_ids=[],
        token_budget=2400,restore_per_chunk_budget=0,packing_priority_chunk_ids=[missing])
    assert restored.package_json['chunks'][0]['chunk_id'] == missing
    assert not restored.package_json['chunks'][0]['content_clipped']
    assert audit_package(db_session,restored)['all_valid']
    assert package.package_json == original
    assert db_session.get(RetrievalTrace,restored.retrieval_trace_id).result_chunk_ids_json == result.trace.result_chunk_ids_json
    with pytest.raises(ValueError,match='outside_authorized_sources'):
        restore_reflection_context(db_session,source_package=package,
            target_chunk_ids=[package.package_json['chunks'][0]['chunk_id']],preserve_chunk_ids=[],
            token_budget=2400,restore_per_chunk_budget=0,packing_priority_chunk_ids=['unit-unobserved-source'])


@pytest.mark.asyncio
async def test_cache_replays_partial_package_and_rejects_old_envelope(db_session, populated_context_graph):
    from app.services.context_graph import build_layered_retrieval_cache_envelope, replay_layered_retrieval_cache, RetrievalCacheReplayError
    result, package = await small_package(db_session, populated_context_graph)
    envelope = build_layered_retrieval_cache_envelope(db_session, result=result, package=package)
    arguments = dict(knowledge_base_id=package.knowledge_base_id, cache_components=result.cache_components, ttl_seconds_remaining=200)
    replay = replay_layered_retrieval_cache(db_session, envelope=envelope, **arguments)
    assert replay[2].id == package.id
    assert replay[2].hit_chunk_ids_json == package.hit_chunk_ids_json
    assert len(replay[0]) == len(result.results)
    with pytest.raises(RetrievalCacheReplayError):
        replay_layered_retrieval_cache(db_session, envelope={**envelope, "protocol_version": "layered_retrieval_postgresql_replay_envelope_v1"}, **arguments)
    # A matching hash is not sufficient: replay must check the public package contract.
    from app.services.retrieval import ContextPackagePublicIntegrityError
    extra = next(cid for cid in result.trace.result_chunk_ids_json if cid not in package.hit_chunk_ids_json)
    package.why_selected_json = {**package.why_selected_json, extra: deepcopy(next(iter(package.why_selected_json.values())))}
    db_session.flush()
    altered = build_layered_retrieval_cache_envelope(db_session, result=result, package=package)
    with pytest.raises(ContextPackagePublicIntegrityError):
        replay_layered_retrieval_cache(db_session, envelope=altered, **arguments)
