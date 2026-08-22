from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPTS_ROOT = REPO_ROOT / "scripts"


def _load_compliance_module():
    scripts_path = str(SCRIPTS_ROOT)
    if scripts_path not in sys.path:
        sys.path.insert(0, scripts_path)
    spec = importlib.util.spec_from_file_location(
        "test_check_technical_spec_compliance_static",
        SCRIPTS_ROOT / "check_technical_spec_compliance.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _legacy_token_issues(module) -> list[dict]:
    return [
        issue
        for issue in module.static_checks()
        if issue["code"] == "legacy_active_token"
    ]


def test_legacy_scan_has_no_historical_source_exemptions(
    tmp_path: Path,
    monkeypatch,
) -> None:
    module = _load_compliance_module()
    monkeypatch.setattr(module, "REPO_ROOT", tmp_path)
    assert module.HISTORICAL_SOURCE_PREFIXES == ()
    assert module.LEGACY_TOKEN_PATH_EXEMPTIONS == {}

    active_bad = (
        tmp_path / "apps" / "api" / "app" / "services" / "active_bad.py"
    )
    active_bad.parent.mkdir(parents=True)
    active_bad.write_text(
        "BM25Record = object\n"
        'ACTIVE_TRACE_FIELD = "ambiguous_edge_decisions_json"\n',
        encoding="utf-8",
    )

    issues = _legacy_token_issues(module)
    assert len(issues) == 2
    assert all(
        issue["evidence"]["paths"]
        == ["apps/api/app/services/active_bad.py"]
        for issue in issues
    )


def test_current_tree_has_no_legacy_token_in_active_source() -> None:
    module = _load_compliance_module()

    assert _legacy_token_issues(module) == []


def test_citation_compliance_gate_tracks_current_claim_level_protocol() -> None:
    script = (SCRIPTS_ROOT / "check_technical_spec_compliance.py").read_text(
        encoding="utf-8"
    )
    technical_spec = (REPO_ROOT / "docs" / "technical-spec.md").read_text(
        encoding="utf-8"
    )

    assert "claim_structure_plus_llm_entailment_v2" in script
    assert "claim_structure_plus_llm_entailment_v2" in technical_spec
    assert '!= "structure_plus_llm_entailment_v1"' not in script


def test_cross_language_edge_identity_audit_replays_endpoint_hashes_and_fails_closed() -> None:
    module = _load_compliance_module()
    source_hash = "a" * 64
    target_hash = "b" * 64
    scope_hash = "c" * 64
    calibration_hash = "d" * 64
    identities = {
        "source": {
            "valid": True,
            "known": True,
            "language": "en",
            "detection_hash": source_hash,
            "protocol_version": "document_language_unicode_script_v1",
        },
        "target": {
            "valid": True,
            "known": True,
            "language": "zh",
            "detection_hash": target_hash,
            "protocol_version": "document_language_unicode_script_v1",
        },
    }
    edge = SimpleNamespace(
        id="edge-1",
        source_chunk_id="source",
        target_chunk_id="target",
        source_language="zh",
        target_language="en",
        is_cross_language=True,
        is_bridge=True,
        bridge_quota_reason="cross_language_dense_quota",
        features_json={
            "source_language": "zh",
            "target_language": "en",
            "source_language_detection_hash": target_hash,
            "target_language_detection_hash": source_hash,
            "source_language_detection_protocol_version": "document_language_unicode_script_v1",
            "target_language_detection_protocol_version": "document_language_unicode_script_v1",
            "language_identity_scope_hash": scope_hash,
            "candidate_channels": ["cross_language_candidates"],
            "is_cross_language": True,
            "bridge_quota_reason": "cross_language_dense_quota",
        },
        raw_strength_summary_json={
            "edge_type_calibration_stats_hash": calibration_hash,
        },
        normalization_stats_json={
            "edge_type_calibration": {
                "edge_type": "dense_cross_language_bridge",
                "stats_hash": calibration_hash,
            }
        },
        support_json={"model_call_count": 0},
    )

    accepted = module.cross_language_edge_identity_audit(
        [edge], identities, expected_scope_hash=scope_hash
    )
    assert accepted["pass"] is True
    assert accepted["valid_edge_count"] == 1
    assert accepted["model_call_count"] == 0

    edge.features_json = {
        **edge.features_json,
        "source_language_detection_hash": "f" * 64,
    }
    rejected = module.cross_language_edge_identity_audit(
        [edge], identities, expected_scope_hash=scope_hash
    )
    assert rejected["pass"] is False
    assert rejected["invalid_details"] == [
        {
            "edge_id": "edge-1",
            "errors": ["endpoint_detection_hash_mismatch"],
        }
    ]
