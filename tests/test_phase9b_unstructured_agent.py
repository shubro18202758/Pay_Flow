"""
PayFlow -- Phase 9B Tests: NLU Sub-Agent for Unstructured Data Analysis
========================================================================
Verifies heuristic pre-filters, LLM response parsing, finding deduplication,
aggregate risk scoring, consensus loop integration, and the full analysis
pipeline of the UnstructuredAnalysisAgent.

Tests:
 1. Encoding attack detection (zero-width characters)
 2. Cyrillic-Latin homoglyph detection
 3. User-agent bot detection
 4. User-agent outdated browser detection
 5. Emulator signature detection
 6. Device fingerprint transition detection
 7. VPN/Tor indicator detection
 8. Urgency & authority language detection
 9. LLM response JSON parsing
10. Finding deduplication (heuristic vs LLM merge)
11. Social engineering score aggregation
12. Device anomaly score aggregation
13. Risk modifier computation
14. No-findings risk modifier (returns -0.05)
15. Full analysis pipeline (heuristics only, no LLM)
16. Consensus loop integration (_run_nlu_consensus)
17. Consensus loop skips when no textual data
18. UnstructuredPayload.has_textual_data property
19. UnstructuredAnalysisResult.has_critical_findings property
20. Config and exports verification
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import InvestigatorAgentConfig, UnstructuredAgentConfig
from src.llm.agent import InvestigatorAgent, VerdictPayload
from src.llm.unstructured_agent import UnstructuredAnalysisAgent
from src.llm.unstructured_models import (
    FindingConfidence,
    SemanticAnomalyType,
    SemanticFinding,
    UnstructuredAgentMetrics,
    UnstructuredAnalysisResult,
    UnstructuredPayload,
)
from src.llm.unstructured_prompts import (
    UNSTRUCTURED_ANALYSIS_SCHEMA,
    UNSTRUCTURED_ANALYSIS_SYSTEM_PROMPT,
    build_consensus_injection_prompt,
    build_unstructured_analysis_prompt,
)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _safe_print(msg: str) -> None:
    try:
        print(msg)
    except UnicodeEncodeError:
        print(msg.encode("ascii", "replace").decode())


passed = 0
failed = 0


def run_test(func):
    global passed, failed
    name = func.__name__
    try:
        if asyncio.iscoroutinefunction(func):
            asyncio.get_event_loop().run_until_complete(func())
        else:
            func()
        _safe_print(f"  PASS  {name}")
        passed += 1
    except Exception as e:
        _safe_print(f"  FAIL  {name}: {e}")
        failed += 1


# ── Test Fixtures ────────────────────────────────────────────────────────────

def _make_agent() -> UnstructuredAnalysisAgent:
    """Create a sub-agent without LLM client (heuristics only)."""
    return UnstructuredAnalysisAgent(llm_client=None)


def _make_payload(**kwargs) -> UnstructuredPayload:
    """Create a test UnstructuredPayload with sensible defaults."""
    defaults = {
        "txn_id": "TXN_TEST_001",
        "sender_id": "ACC_S001",
        "receiver_id": "ACC_R001",
    }
    defaults.update(kwargs)
    return UnstructuredPayload(**defaults)


# ── Test 1: Encoding Attack (Zero-Width Characters) ─────────────────────────

def test_encoding_attack_zero_width():
    agent = _make_agent()
    # Name with zero-width space (U+200B) and bidi override (U+202E)
    name = "Ram\u200besh Ku\u202emar"
    findings = agent._check_encoding_attacks(name, "sender_name")
    assert len(findings) >= 1, f"Expected encoding findings, got {len(findings)}"
    types = {f.anomaly_type for f in findings}
    assert SemanticAnomalyType.NAME_ENCODING_ARTEFACT in types, (
        f"Expected NAME_ENCODING_ARTEFACT, got {types}"
    )
    assert findings[0].confidence == FindingConfidence.HIGH
    assert findings[0].severity_score >= 0.85


# ── Test 2: Cyrillic-Latin Homoglyph Detection ──────────────────────────────

def test_cyrillic_homoglyph():
    agent = _make_agent()
    # Mix Cyrillic 'а' (U+0430) into Latin name
    name = "R\u0430mesh Kumar"  # 'а' is Cyrillic
    findings = agent._check_encoding_attacks(name, "receiver_name")
    types = {f.anomaly_type for f in findings}
    assert SemanticAnomalyType.CHARSET_HOMOGLYPH in types, (
        f"Expected CHARSET_HOMOGLYPH, got {types}"
    )
    homoglyph_f = [f for f in findings if f.anomaly_type == SemanticAnomalyType.CHARSET_HOMOGLYPH]
    assert homoglyph_f[0].severity_score >= 0.9


# ── Test 3: User-Agent Bot Detection ────────────────────────────────────────

def test_user_agent_bot():
    agent = _make_agent()
    ua = "python-requests/2.28.0"
    findings = agent._check_user_agent(ua)
    assert len(findings) >= 1
    assert findings[0].anomaly_type == SemanticAnomalyType.USER_AGENT_SPOOFING
    assert findings[0].confidence == FindingConfidence.HIGH


# ── Test 4: User-Agent Outdated Browser ──────────────────────────────────────

def test_user_agent_outdated():
    agent = _make_agent()
    ua = "Mozilla/5.0 (Windows NT 10.0) Chrome/45.0.2454.85 Safari/537.36"
    findings = agent._check_user_agent(ua)
    assert len(findings) >= 1
    assert findings[0].anomaly_type == SemanticAnomalyType.USER_AGENT_SPOOFING
    assert findings[0].confidence == FindingConfidence.MEDIUM
    assert "45" in findings[0].description


# ── Test 5: Emulator Signature Detection ─────────────────────────────────────

def test_emulator_signature():
    agent = _make_agent()
    fp = "google/sdk_gphone_x86_64/generic_x86_64:12/SE1A.220203.002"
    findings = agent._check_emulator_signature(fp)
    assert len(findings) == 1
    assert findings[0].anomaly_type == SemanticAnomalyType.EMULATOR_SIGNATURE
    assert findings[0].confidence == FindingConfidence.HIGH


# ── Test 6: Device Fingerprint Transition ────────────────────────────────────

def test_device_transition_physical_to_emulator():
    agent = _make_agent()
    findings = agent._check_device_transition(
        current="sdk_gphone_x86_64",
        previous="Samsung Galaxy S23 Ultra",
    )
    assert len(findings) == 1
    assert findings[0].anomaly_type == SemanticAnomalyType.DEVICE_FINGERPRINT_JUMP
    assert findings[0].confidence == FindingConfidence.HIGH
    assert findings[0].severity_score >= 0.85


# ── Test 7: VPN/Tor Indicator ────────────────────────────────────────────────

def test_vpn_tor_indicator():
    agent = _make_agent()
    findings = agent._check_vpn_tor("Tor Exit Node, Frankfurt, DE")
    assert len(findings) == 1
    assert findings[0].anomaly_type == SemanticAnomalyType.VPN_TOR_INDICATOR
    assert findings[0].confidence == FindingConfidence.MEDIUM


# ── Test 8: Urgency & Authority Language ─────────────────────────────────────

def test_urgency_authority_detection():
    agent = _make_agent()
    text = "URGENT: The CEO requires immediate action on this wire transfer!"
    findings = agent._check_urgency_authority(text, "email_subject")
    types = {f.anomaly_type for f in findings}
    assert SemanticAnomalyType.URGENCY_LANGUAGE in types
    assert SemanticAnomalyType.AUTHORITY_IMPERSONATION in types
    assert len(findings) == 2


# ── Test 9: LLM Response JSON Parsing ────────────────────────────────────────

def test_llm_response_parsing():
    agent = _make_agent()
    llm_response = json.dumps({
        "findings": [
            {
                "anomaly_type": "name_transliteration_mismatch",
                "confidence": "HIGH",
                "severity_score": 0.8,
                "description": "Inconsistent romanization of Hindi name",
                "evidence_text": "Shyam vs Shyaam",
                "field_source": "sender_name",
            },
            {
                "anomaly_type": "pretexting_pattern",
                "confidence": "MEDIUM",
                "severity_score": 0.6,
                "description": "Constructed scenario for unusual transfer",
                "evidence_text": "vendor payment overdue by 3 months",
                "field_source": "email_body_snippet",
            },
        ],
        "social_engineering_score": 0.6,
        "linguistic_anomaly_score": 0.4,
        "device_anomaly_score": 0.0,
        "summary": "Two anomalies found.",
    })
    # Wrap in ```json``` block
    content = f"```json\n{llm_response}\n```"
    findings = agent._parse_llm_findings(content)
    assert len(findings) == 2
    assert findings[0].anomaly_type == SemanticAnomalyType.NAME_TRANSLITERATION_MISMATCH
    assert findings[1].anomaly_type == SemanticAnomalyType.PRETEXTING_PATTERN
    assert findings[0].confidence == FindingConfidence.HIGH


# ── Test 10: Finding Deduplication ───────────────────────────────────────────

def test_finding_deduplication():
    agent = _make_agent()
    heuristic = [
        SemanticFinding(
            anomaly_type=SemanticAnomalyType.EMULATOR_SIGNATURE,
            confidence=FindingConfidence.HIGH,
            severity_score=0.8,
            description="Heuristic: emulator detected",
            evidence_text="sdk_gphone",
            field_source="device_fingerprint",
        ),
    ]
    llm = [
        SemanticFinding(
            anomaly_type=SemanticAnomalyType.EMULATOR_SIGNATURE,
            confidence=FindingConfidence.HIGH,
            severity_score=0.9,  # higher severity from LLM
            description="LLM: emulator signature with additional context",
            evidence_text="sdk_gphone_x86_64",
            field_source="device_fingerprint",
        ),
        SemanticFinding(
            anomaly_type=SemanticAnomalyType.PRETEXTING_PATTERN,
            confidence=FindingConfidence.MEDIUM,
            severity_score=0.6,
            description="Pretexting found",
            evidence_text="vendor payment",
            field_source="email_body_snippet",
        ),
    ]
    merged = agent._merge_findings(heuristic, llm)
    # Should have 2 findings: emulator (LLM version with higher severity) + pretexting
    assert len(merged) == 2
    emulator_f = [f for f in merged if f.anomaly_type == SemanticAnomalyType.EMULATOR_SIGNATURE]
    assert emulator_f[0].severity_score == 0.9  # LLM version kept
    assert emulator_f[0].description.startswith("LLM:")


# ── Test 11: Social Engineering Score ────────────────────────────────────────

def test_social_engineering_score():
    agent = _make_agent()
    findings = [
        SemanticFinding(
            anomaly_type=SemanticAnomalyType.URGENCY_LANGUAGE,
            confidence=FindingConfidence.HIGH,
            severity_score=0.7,
            description="Urgency",
            evidence_text="urgent",
            field_source="email_subject",
        ),
        SemanticFinding(
            anomaly_type=SemanticAnomalyType.AUTHORITY_IMPERSONATION,
            confidence=FindingConfidence.MEDIUM,
            severity_score=0.65,
            description="Authority",
            evidence_text="CEO",
            field_source="email_subject",
        ),
    ]
    score = agent._compute_social_engineering_score(findings)
    assert score > 0.0
    assert score <= 1.0
    # Should be: avg(0.7, 0.65) + 0.1*2 = 0.675 + 0.2 = 0.875
    assert abs(score - 0.875) < 0.01, f"Expected ~0.875, got {score}"


# ── Test 12: Device Anomaly Score ────────────────────────────────────────────

def test_device_anomaly_score():
    agent = _make_agent()
    findings = [
        SemanticFinding(
            anomaly_type=SemanticAnomalyType.EMULATOR_SIGNATURE,
            confidence=FindingConfidence.HIGH,
            severity_score=0.8,
            description="Emulator",
            evidence_text="sdk_gphone",
            field_source="device_fingerprint",
        ),
    ]
    score = agent._compute_device_anomaly_score(findings)
    assert score > 0.0
    # avg(0.8) + 0.1*1 = 0.8 + 0.1 = 0.9
    assert abs(score - 0.9) < 0.01, f"Expected ~0.9, got {score}"


# ── Test 13: Risk Modifier Computation ───────────────────────────────────────

def test_risk_modifier_computation():
    agent = _make_agent()
    # Moderate scores across all domains
    modifier = agent._compute_risk_modifier(
        se_score=0.5, la_score=0.4, da_score=0.3,
    )
    assert modifier > 0.0
    assert modifier <= 0.30
    # weighted = 0.5*0.4 + 0.4*0.35 + 0.3*0.25 = 0.20 + 0.14 + 0.075 = 0.415
    # result = min(0.30, round(0.415 * 0.35, 4)) = min(0.30, 0.1453) = 0.1453
    expected = round(0.415 * 0.35, 4)
    assert abs(modifier - expected) < 0.001, f"Expected ~{expected}, got {modifier}"


# ── Test 14: No-Findings Risk Modifier ───────────────────────────────────────

def test_no_findings_risk_modifier():
    agent = _make_agent()
    modifier = agent._compute_risk_modifier(0.0, 0.0, 0.0)
    assert modifier == -0.05, f"Expected -0.05, got {modifier}"


# ── Test 15: Full Analysis Pipeline (Heuristics Only) ───────────────────────

async def test_full_pipeline_heuristics():
    agent = _make_agent()
    payload = _make_payload(
        sender_name="Ram\u200besh",  # zero-width space
        user_agent="curl/7.68.0",
        device_fingerprint="sdk_gphone_x86_64",
        ip_geo="VPN Datacenter, US",
    )
    result = await agent.analyze(payload)
    assert isinstance(result, UnstructuredAnalysisResult)
    assert result.txn_id == "TXN_TEST_001"
    assert len(result.findings) >= 3  # encoding, bot ua, emulator, vpn
    assert result.overall_risk_modifier > 0.0
    assert result.device_anomaly_score > 0.0
    assert result.linguistic_anomaly_score > 0.0
    assert result.analysis_duration_ms > 0.0
    assert agent.metrics.analyses_completed == 1


# ── Test 16: Consensus Loop Integration ──────────────────────────────────────

async def test_consensus_loop_integration():
    """Test _run_nlu_consensus in InvestigatorAgent."""
    # Mock the unstructured agent
    mock_result = UnstructuredAnalysisResult(
        txn_id="TXN_CONSENSUS_001",
        findings=(
            SemanticFinding(
                anomaly_type=SemanticAnomalyType.URGENCY_LANGUAGE,
                confidence=FindingConfidence.HIGH,
                severity_score=0.8,
                description="Urgency detected",
                evidence_text="urgent transfer",
                field_source="email_subject",
            ),
        ),
        overall_risk_modifier=0.15,
        social_engineering_score=0.7,
        linguistic_anomaly_score=0.0,
        device_anomaly_score=0.0,
        analysis_duration_ms=42.0,
    )

    mock_nlu_agent = MagicMock()
    mock_nlu_agent.analyze = AsyncMock(return_value=mock_result)

    agent = InvestigatorAgent(
        llm_client=None,
        unstructured_agent=mock_nlu_agent,
    )

    alert_dict = {
        "txn_id": "TXN_CONSENSUS_001",
        "sender_id": "ACC_S001",
        "receiver_id": "ACC_R001",
        "email_subject": "URGENT: Wire transfer required",
        "sender_name": "Test Sender",
    }

    result = await agent._run_nlu_consensus(alert_dict, {})
    assert result is not None
    assert result["findings_count"] == 1
    assert result["overall_risk_modifier"] == 0.15
    assert result["has_critical_findings"] is True
    assert result["social_engineering_score"] == 0.7
    mock_nlu_agent.analyze.assert_called_once()


# ── Test 17: Consensus Loop Skips When No Textual Data ──────────────────────

async def test_consensus_loop_skips_no_data():
    mock_nlu_agent = MagicMock()
    mock_nlu_agent.analyze = AsyncMock()

    agent = InvestigatorAgent(
        llm_client=None,
        unstructured_agent=mock_nlu_agent,
    )

    alert_dict = {
        "txn_id": "TXN_NODATA_001",
        "sender_id": "ACC_S002",
        "receiver_id": "ACC_R002",
        # No textual fields
    }

    result = await agent._run_nlu_consensus(alert_dict, {})
    assert result is None
    mock_nlu_agent.analyze.assert_not_called()


# ── Test 18: UnstructuredPayload.has_textual_data Property ───────────────────

def test_payload_has_textual_data():
    # No textual fields
    p1 = _make_payload()
    assert p1.has_textual_data is False

    # With sender name
    p2 = _make_payload(sender_name="Ramesh")
    assert p2.has_textual_data is True

    # With user agent only
    p3 = _make_payload(user_agent="Mozilla/5.0")
    assert p3.has_textual_data is True

    # With email
    p4 = _make_payload(email_subject="Hello")
    assert p4.has_textual_data is True


# ── Test 19: UnstructuredAnalysisResult.has_critical_findings ────────────────

def test_has_critical_findings():
    # Critical: HIGH confidence + severity >= 0.7
    r1 = UnstructuredAnalysisResult(
        txn_id="T1",
        findings=(
            SemanticFinding(
                anomaly_type=SemanticAnomalyType.CHARSET_HOMOGLYPH,
                confidence=FindingConfidence.HIGH,
                severity_score=0.95,
                description="Homoglyph",
                evidence_text="test",
                field_source="sender_name",
            ),
        ),
        overall_risk_modifier=0.2,
        social_engineering_score=0.0,
        linguistic_anomaly_score=0.9,
        device_anomaly_score=0.0,
        analysis_duration_ms=10.0,
    )
    assert r1.has_critical_findings is True

    # Not critical: MEDIUM confidence
    r2 = UnstructuredAnalysisResult(
        txn_id="T2",
        findings=(
            SemanticFinding(
                anomaly_type=SemanticAnomalyType.VPN_TOR_INDICATOR,
                confidence=FindingConfidence.MEDIUM,
                severity_score=0.9,
                description="VPN",
                evidence_text="vpn",
                field_source="ip_geo",
            ),
        ),
        overall_risk_modifier=0.1,
        social_engineering_score=0.0,
        linguistic_anomaly_score=0.0,
        device_anomaly_score=0.5,
        analysis_duration_ms=5.0,
    )
    assert r2.has_critical_findings is False

    # Not critical: HIGH confidence but severity < 0.7
    r3 = UnstructuredAnalysisResult(
        txn_id="T3",
        findings=(
            SemanticFinding(
                anomaly_type=SemanticAnomalyType.USER_AGENT_SPOOFING,
                confidence=FindingConfidence.HIGH,
                severity_score=0.5,
                description="Outdated",
                evidence_text="Chrome/45",
                field_source="user_agent",
            ),
        ),
        overall_risk_modifier=0.05,
        social_engineering_score=0.0,
        linguistic_anomaly_score=0.0,
        device_anomaly_score=0.3,
        analysis_duration_ms=5.0,
    )
    assert r3.has_critical_findings is False


# ── Test 20: Config and Exports Verification ────────────────────────────────

def test_config_and_exports():
    from config.settings import UNSTRUCTURED_AGENT_CFG, UnstructuredAgentConfig

    assert isinstance(UNSTRUCTURED_AGENT_CFG, UnstructuredAgentConfig)
    assert UNSTRUCTURED_AGENT_CFG.analysis_temperature == 0.2
    assert UNSTRUCTURED_AGENT_CFG.se_weight == 0.40
    assert UNSTRUCTURED_AGENT_CFG.la_weight == 0.35
    assert UNSTRUCTURED_AGENT_CFG.da_weight == 0.25
    assert UNSTRUCTURED_AGENT_CFG.risk_modifier_ceiling == 0.30

    # Check __init__.py exports
    from src.llm import (
        UnstructuredAnalysisAgent,
        UnstructuredPayload,
        UnstructuredAnalysisResult,
        SemanticFinding,
        SemanticAnomalyType,
        FindingConfidence,
        UnstructuredAgentMetrics,
        build_unstructured_analysis_prompt,
        build_consensus_injection_prompt,
    )
    assert UnstructuredAnalysisAgent is not None
    assert UnstructuredPayload is not None


# ── Test Runner ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    _safe_print("\n=== Phase 9B Tests: NLU Sub-Agent for Unstructured Data Analysis ===\n")

    tests = [
        test_encoding_attack_zero_width,
        test_cyrillic_homoglyph,
        test_user_agent_bot,
        test_user_agent_outdated,
        test_emulator_signature,
        test_device_transition_physical_to_emulator,
        test_vpn_tor_indicator,
        test_urgency_authority_detection,
        test_llm_response_parsing,
        test_finding_deduplication,
        test_social_engineering_score,
        test_device_anomaly_score,
        test_risk_modifier_computation,
        test_no_findings_risk_modifier,
        test_full_pipeline_heuristics,
        test_consensus_loop_integration,
        test_consensus_loop_skips_no_data,
        test_payload_has_textual_data,
        test_has_critical_findings,
        test_config_and_exports,
    ]

    for test_func in tests:
        run_test(test_func)

    _safe_print(f"\n--- Results: {passed} passed, {failed} failed, {passed + failed} total ---")
    if failed:
        sys.exit(1)
    _safe_print("All Phase 9B tests passed!")
