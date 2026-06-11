from __future__ import annotations

import pytest


def test_quality_signals_drive_chunk_actions_without_stopword_gate():
    from app.services.quality.policies import ChunkQualityPolicy
    from app.services.quality.signals import build_quality_signals

    toc = "\n".join(["1", "2", "3", "4", "5", "6", "7", "8"])
    toc_decision = ChunkQualityPolicy().decide(build_quality_signals(target_type="chunk", text=toc, content_kind="pdf_page"))
    assert toc_decision.action == "summary_only"
    assert "toc_layout_noise" in toc_decision.reasons
    assert toc_decision.audit["retention_decision"]["retain"] is True
    assert toc_decision.audit["route_eligibility"]["evidence_graph"] is False

    definition = (
        "Residual Network is defined as the remaining capacity graph used by augmenting path algorithms. "
        "It explains how maximum flow updates admissible edges."
    )
    graph_decision = ChunkQualityPolicy().decide(build_quality_signals(target_type="chunk", text=definition, content_kind="markdown"))
    assert graph_decision.action == "graph_candidate"
    assert graph_decision.audit["signals"]["semantic_density"]["definition_score"] == 1.0
    assert graph_decision.audit["route_eligibility"]["evidence_graph"] is True


def test_chunk_policy_only_physically_discards_mechanical_noise():
    from app.services.quality.policies import ChunkQualityPolicy
    from app.services.quality.signals import build_quality_signals

    policy = ChunkQualityPolicy()
    short = policy.decide(build_quality_signals(target_type="chunk", text="short note", content_kind="markdown"))
    assert short.action == "evidence_only"
    assert short.audit["retention_decision"]["retain"] is True

    formula = policy.decide(build_quality_signals(target_type="chunk", text="Q = 1 / 2m", content_kind="formula"))
    assert formula.action == "retrieval_candidate"
    assert formula.audit["route_eligibility"]["retrieval"] is True
    assert formula.audit["route_eligibility"]["evidence_graph"] is False

    output = policy.decide(build_quality_signals(target_type="chunk", text="[Output]\n1\n2\n3", content_kind="output"))
    assert output.action == "evidence_only"
    assert output.audit["route_eligibility"]["evidence_graph"] is False

    empty = policy.decide(build_quality_signals(target_type="chunk", text="", content_kind="markdown"))
    assert empty.action == "discard"
    assert empty.audit["retention_decision"]["retain"] is False


def test_chunk_policy_routes_code_by_traceable_section_context():
    from app.services.quality.policies import ChunkQualityPolicy
    from app.services.quality.signals import build_quality_signals

    code = "def helper(value):\n    adjusted = value + 1\n    return adjusted"
    generic = ChunkQualityPolicy().decide(
        build_quality_signals(target_type="chunk", text=code, content_kind="code"),
        section_name="Code",
        section_title="Utility",
    )
    assert generic.action == "embed_only"
    assert "code_without_traceable_context" in generic.reasons
    assert generic.audit["route_eligibility"]["evidence_graph"] is False

    specific = ChunkQualityPolicy().decide(
        build_quality_signals(target_type="chunk", text=code, content_kind="code"),
        section_name="Evidence Graph Builder",
        section_title="Community Detection",
    )
    assert specific.action in {"graph_candidate", "retrieval_candidate"}
    assert "code_without_traceable_context" not in specific.reasons


@pytest.mark.asyncio
async def test_quality_judge_uses_cache(monkeypatch):
    from app.services.quality import judge as judge_module
    from app.services.quality.judge import QualityJudge

    class FakeCache:
        def __init__(self) -> None:
            self.values = {}

        def get_quality_judgment(self, key):
            return self.values.get(key)

        def set_quality_judgment(self, key, result, ttl=86400):
            self.values[key] = result

    class FakeSettings:
        chat_model = "judge-model"

    class FakeProvider:
        settings = FakeSettings()
        calls = 0

        async def classify_json(self, _system_prompt, _user_prompt, fallback=None):
            self.calls += 1
            return {"action": "accept", "score": 0.9, "reasons": ["policy_passed"]}

    cache = FakeCache()
    provider = FakeProvider()
    monkeypatch.setattr(judge_module, "get_cache_manager", lambda: cache)

    judge = QualityJudge(provider=provider)
    candidate = {"active_chunk_id": "chunk-1", "signals": {"score": 0.9}}
    first = await judge.judge(
        knowledge_base_id="knowledge-base-1",
        profile={"version": "quality_profile_v1:test"},
        target_type="chunk_candidate",
        candidate=candidate,
    )
    second = await judge.judge(
        knowledge_base_id="knowledge-base-1",
        profile={"version": "quality_profile_v1:test"},
        target_type="chunk_candidate",
        candidate=candidate,
    )

    assert first["cached"] is False
    assert second["cached"] is True
    assert provider.calls == 1
