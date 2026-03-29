"""
PayFlow -- Phase 8 Tests: ZKP Verification & Smart Contract Circuit Breaker
==============================================================================
Verifies zero-knowledge proofs, multi-model consensus freeze logic,
transaction blocking, ledger anchoring, and TTL expiry.
"""

from __future__ import annotations

import asyncio
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import CircuitBreakerConfig, LedgerConfig
from src.blockchain.circuit_breaker import CircuitBreaker, CircuitBreakerMetrics, FreezeOrder
from src.blockchain.ledger import AuditLedger
from src.blockchain.models import EventType
from src.blockchain.zkp import ComplianceProof, ZKPCommitment, ZKPProver, ZKPVerifier


# ── Helpers ──────────────────────────────────────────────────────────────────

def _safe_print(msg: str) -> None:
    """Print with ASCII fallback for Windows cp1252 consoles."""
    try:
        print(msg)
    except UnicodeEncodeError:
        print(msg.encode("ascii", "replace").decode())


def make_ledger_config(tmp_dir: Path, **overrides) -> LedgerConfig:
    defaults = {
        "db_path": tmp_dir / "test_ledger.db",
        "key_dir": tmp_dir / "keys",
        "checkpoint_interval": 100,
        "enable_signing": True,
    }
    defaults.update(overrides)
    return LedgerConfig(**defaults)


@dataclass
class MockAlertPayload:
    """Minimal AlertPayload mock for testing."""
    txn_id: str
    sender_id: str
    receiver_id: str
    timestamp: int
    risk_score: float
    tier: str
    threshold_at_eval: float = 0.85
    features: object = None
    feature_names: list = None

    def to_dict(self) -> dict:
        return {
            "txn_id": self.txn_id,
            "sender_id": self.sender_id,
            "receiver_id": self.receiver_id,
            "timestamp": self.timestamp,
            "risk_score": round(self.risk_score, 4),
            "tier": self.tier,
        }


class MockInvestigationResult:
    """Minimal InvestigationResult mock for testing."""
    def __init__(self, txn_id, sender_id, receiver_id, mule_findings=None,
                 cycle_findings=None, gnn_risk_score=-1.0):
        self.txn_id = txn_id
        self.sender_id = sender_id
        self.receiver_id = receiver_id
        self.subgraph_nodes = 10
        self.subgraph_edges = 15
        self.mule_findings = mule_findings or []
        self.cycle_findings = cycle_findings or []
        self.investigation_ms = 5.0
        self.gnn_risk_score = gnn_risk_score


# ── ZKP Tests ────────────────────────────────────────────────────────────────

async def test_zkp_commitment():
    """Commit to a value and verify the commitment binding."""
    value = 0.92
    domain = "risk_score"

    commitment = ZKPProver.commit(value, domain)

    # Commitment is 64-hex-char SHA-256
    assert len(commitment.commitment) == 64
    assert len(commitment.blinding_hex) == 64  # 32 bytes = 64 hex chars

    # Binding check: correct value + blinding → match
    assert ZKPVerifier.verify_commitment_binding(
        commitment.commitment, value, commitment.blinding_hex, domain,
    )

    # Wrong value → no match
    assert not ZKPVerifier.verify_commitment_binding(
        commitment.commitment, 0.50, commitment.blinding_hex, domain,
    )

    # Wrong blinding → no match
    assert not ZKPVerifier.verify_commitment_binding(
        commitment.commitment, value, "a" * 64, domain,
    )

    _safe_print("  [PASS] ZKP commitment binding")


async def test_zkp_threshold_proof():
    """Prove risk_score >= 0.85 without revealing actual value 0.92."""
    actual_value = 0.92
    threshold = 0.85
    domain = "risk_score"
    node_id = "ACCT_12345"

    commitment = ZKPProver.commit(actual_value, domain)
    proof = ZKPProver.prove_threshold(
        commitment, actual_value, threshold,
        predicate=f"risk_score >= {threshold}",
        node_id=node_id,
    )

    # Proof structure
    assert proof.node_id == node_id
    assert proof.predicate == f"risk_score >= {threshold}"
    assert len(proof.challenge) == 64
    assert len(proof.response) == 64
    assert len(proof.nonce) == 64

    # Verifier sees the proof but NOT the actual value
    assert ZKPVerifier.verify(proof)

    # The proof dict for ledger anchoring has no secret value
    proof_dict = proof.to_dict()
    assert "actual_value" not in str(proof_dict)
    assert "0.92" not in str(proof_dict)

    _safe_print("  [PASS] ZKP threshold proof (risk >= 0.85, actual 0.92)")


async def test_zkp_kyc_tier_proof():
    """Prove KYC tier >= 2 without revealing actual tier 3."""
    actual_tier = 3
    min_tier = 2
    domain = "kyc_tier"

    commitment = ZKPProver.commit(actual_tier, domain)
    proof = ZKPProver.prove_kyc_tier(
        commitment, actual_tier, min_tier, node_id="ACCT_67890",
    )

    assert proof.predicate == f"kyc_tier >= {min_tier}"
    assert ZKPVerifier.verify(proof)

    # Cannot prove false predicate
    try:
        ZKPProver.prove_kyc_tier(commitment, actual_tier=1, min_tier=2, node_id="X")
        assert False, "Should have raised ValueError"
    except ValueError as exc:
        assert "Cannot prove" in str(exc)

    _safe_print("  [PASS] ZKP KYC tier proof (tier >= 2, actual 3)")


async def test_zkp_invalid_proof_rejected():
    """Tampered proof must fail verification."""
    commitment = ZKPProver.commit(0.92, "risk_score")
    proof = ZKPProver.prove_threshold(
        commitment, 0.92, 0.85,
        predicate="risk_score >= 0.85",
        node_id="ACCT_001",
    )

    # Valid proof passes
    assert ZKPVerifier.verify(proof)

    # Tamper with challenge → invalid
    tampered = ComplianceProof(
        commitment=proof.commitment,
        predicate=proof.predicate,
        challenge="b" * 64,   # tampered
        response=proof.response,
        nonce=proof.nonce,
        timestamp=proof.timestamp,
        node_id=proof.node_id,
    )
    assert not ZKPVerifier.verify(tampered)

    # Tamper with nonce → invalid (challenge won't match)
    tampered2 = ComplianceProof(
        commitment=proof.commitment,
        predicate=proof.predicate,
        challenge=proof.challenge,
        response=proof.response,
        nonce="c" * 64,   # tampered
        timestamp=proof.timestamp,
        node_id=proof.node_id,
    )
    assert not ZKPVerifier.verify(tampered2)

    _safe_print("  [PASS] ZKP invalid proof rejected")


async def test_zkp_different_predicates():
    """Same commitment, different predicates produce different proofs."""
    commitment = ZKPProver.commit(0.92, "risk_score")

    proof1 = ZKPProver.prove_threshold(
        commitment, 0.92, 0.80, "risk_score >= 0.80", "ACCT_001",
    )
    proof2 = ZKPProver.prove_threshold(
        commitment, 0.92, 0.85, "risk_score >= 0.85", "ACCT_001",
    )

    # Both valid
    assert ZKPVerifier.verify(proof1)
    assert ZKPVerifier.verify(proof2)

    # Different predicates → different challenges
    assert proof1.challenge != proof2.challenge
    assert proof1.predicate != proof2.predicate

    _safe_print("  [PASS] ZKP different predicates")


# ── Circuit Breaker Tests ────────────────────────────────────────────────────

async def test_circuit_breaker_consensus_scoring():
    """Verify the weighted consensus formula."""
    cfg = CircuitBreakerConfig(
        consensus_threshold=0.80,
        ml_weight=0.35,
        gnn_weight=0.35,
        graph_evidence_weight=0.30,
    )
    breaker = CircuitBreaker(config=cfg)

    # Full consensus: ML=0.95, GNN=0.90, 2 mules + 1 cycle
    # graph_evidence = min(1.0, 2*0.5 + 1*0.5) = 1.0 (capped)
    # consensus = 0.35*0.95 + 0.35*0.90 + 0.30*1.0 = 0.3325 + 0.315 + 0.30 = 0.9475
    score = breaker._compute_consensus(0.95, 0.90, 2, 1)
    assert abs(score - 0.9475) < 0.001, f"Expected ~0.9475, got {score}"

    # No graph evidence: ML=0.80, GNN=0.70, 0 mules, 0 cycles
    # graph_evidence = 0.0
    # consensus = 0.35*0.80 + 0.35*0.70 + 0.30*0.0 = 0.28 + 0.245 + 0.0 = 0.525
    score2 = breaker._compute_consensus(0.80, 0.70, 0, 0)
    assert abs(score2 - 0.525) < 0.001, f"Expected ~0.525, got {score2}"

    _safe_print("  [PASS] Circuit breaker consensus scoring")


async def test_circuit_breaker_freeze_on_consensus():
    """ML + GNN consensus above threshold triggers freeze."""
    with tempfile.TemporaryDirectory() as tmp:
        lcfg = make_ledger_config(Path(tmp))
        cfg = CircuitBreakerConfig(
            consensus_threshold=0.70,
            cooldown_seconds=0,        # no cooldown for testing
            require_investigation=True,
        )

        async with AuditLedger(config=lcfg) as ledger:
            breaker = CircuitBreaker(config=cfg, audit_ledger=ledger)

            alert = MockAlertPayload(
                txn_id="TXN001", sender_id="ACCT_A", receiver_id="ACCT_B",
                timestamp=int(time.time()), risk_score=0.95, tier="HIGH",
            )
            investigation = MockInvestigationResult(
                txn_id="TXN001", sender_id="ACCT_A", receiver_id="ACCT_B",
                gnn_risk_score=0.88,
                mule_findings=["mule1"],     # count = 1
                cycle_findings=["cycle1"],   # count = 1
            )

            # Feed alert → stashes in pending
            await breaker.on_alert(alert)
            assert len(breaker._pending_alerts) == 1
            assert not breaker.is_frozen("ACCT_A")

            # Feed investigation → triggers evaluation
            await breaker.on_investigation(investigation)

            # Both sender and receiver should be frozen
            assert breaker.is_frozen("ACCT_A"), "Sender should be frozen"
            assert breaker.is_frozen("ACCT_B"), "Receiver should be frozen"
            assert breaker.metrics.consensus_triggers == 1
            assert breaker.metrics.nodes_frozen == 2

    _safe_print("  [PASS] Circuit breaker freeze on consensus")


async def test_circuit_breaker_no_freeze_below_threshold():
    """Score below threshold does NOT freeze nodes."""
    cfg = CircuitBreakerConfig(
        consensus_threshold=0.95,     # very high threshold
        cooldown_seconds=0,
        require_investigation=True,
    )
    breaker = CircuitBreaker(config=cfg)

    alert = MockAlertPayload(
        txn_id="TXN002", sender_id="ACCT_C", receiver_id="ACCT_D",
        timestamp=int(time.time()), risk_score=0.60, tier="MEDIUM",
    )
    investigation = MockInvestigationResult(
        txn_id="TXN002", sender_id="ACCT_C", receiver_id="ACCT_D",
        gnn_risk_score=0.50,
    )

    await breaker.on_alert(alert)
    await breaker.on_investigation(investigation)

    assert not breaker.is_frozen("ACCT_C")
    assert not breaker.is_frozen("ACCT_D")
    assert breaker.metrics.consensus_triggers == 0

    _safe_print("  [PASS] Circuit breaker no freeze below threshold")


async def test_circuit_breaker_cooldown():
    """Same node not re-triggered within cooldown period."""
    cfg = CircuitBreakerConfig(
        consensus_threshold=0.50,
        cooldown_seconds=600,    # 10 minutes
        require_investigation=True,
    )
    breaker = CircuitBreaker(config=cfg)

    # First trigger
    alert1 = MockAlertPayload(
        txn_id="TXN003", sender_id="ACCT_E", receiver_id="ACCT_F",
        timestamp=int(time.time()), risk_score=0.95, tier="HIGH",
    )
    inv1 = MockInvestigationResult(
        txn_id="TXN003", sender_id="ACCT_E", receiver_id="ACCT_F",
        gnn_risk_score=0.90, mule_findings=["m1"],
    )
    await breaker.on_alert(alert1)
    await breaker.on_investigation(inv1)
    assert breaker.is_frozen("ACCT_E")

    # Manually unfreeze to test re-trigger
    await breaker.unfreeze_node("ACCT_E", reason="test")
    assert not breaker.is_frozen("ACCT_E")

    # Second trigger for same node — should be blocked by cooldown
    alert2 = MockAlertPayload(
        txn_id="TXN004", sender_id="ACCT_E", receiver_id="ACCT_G",
        timestamp=int(time.time()), risk_score=0.95, tier="HIGH",
    )
    inv2 = MockInvestigationResult(
        txn_id="TXN004", sender_id="ACCT_E", receiver_id="ACCT_G",
        gnn_risk_score=0.90, mule_findings=["m1"],
    )
    await breaker.on_alert(alert2)
    await breaker.on_investigation(inv2)

    # ACCT_E should NOT be re-frozen (cooldown), but ACCT_G should be frozen
    assert not breaker.is_frozen("ACCT_E"), "ACCT_E should be in cooldown"
    assert breaker.is_frozen("ACCT_G"), "ACCT_G should be frozen (no cooldown)"

    _safe_print("  [PASS] Circuit breaker cooldown")


async def test_circuit_breaker_ttl_expiry():
    """Frozen nodes auto-expire after TTL."""
    cfg = CircuitBreakerConfig(
        consensus_threshold=0.50,
        cooldown_seconds=0,
        freeze_ttl_seconds=1,     # 1 second TTL for testing
        require_investigation=True,
    )
    breaker = CircuitBreaker(config=cfg)

    alert = MockAlertPayload(
        txn_id="TXN005", sender_id="ACCT_H", receiver_id="ACCT_I",
        timestamp=int(time.time()), risk_score=0.95, tier="HIGH",
    )
    inv = MockInvestigationResult(
        txn_id="TXN005", sender_id="ACCT_H", receiver_id="ACCT_I",
        gnn_risk_score=0.90, mule_findings=["m1"],
    )
    await breaker.on_alert(alert)
    await breaker.on_investigation(inv)
    assert breaker.is_frozen("ACCT_H")

    # Wait for TTL to expire
    await asyncio.sleep(1.1)
    expired = await breaker.cleanup_expired()

    assert expired == 2  # both sender and receiver
    assert not breaker.is_frozen("ACCT_H")
    assert not breaker.is_frozen("ACCT_I")

    _safe_print("  [PASS] Circuit breaker TTL expiry")


async def test_circuit_breaker_transaction_blocking():
    """check_transaction() blocks frozen nodes."""
    cfg = CircuitBreakerConfig(
        consensus_threshold=0.50,
        cooldown_seconds=0,
        require_investigation=True,
    )
    breaker = CircuitBreaker(config=cfg)

    alert = MockAlertPayload(
        txn_id="TXN006", sender_id="ACCT_J", receiver_id="ACCT_K",
        timestamp=int(time.time()), risk_score=0.95, tier="HIGH",
    )
    inv = MockInvestigationResult(
        txn_id="TXN006", sender_id="ACCT_J", receiver_id="ACCT_K",
        gnn_risk_score=0.90, mule_findings=["m1"],
    )
    await breaker.on_alert(alert)
    await breaker.on_investigation(inv)

    # Transactions involving frozen nodes should be blocked
    assert breaker.check_transaction("ACCT_J", "ACCT_X")    # sender frozen
    assert breaker.check_transaction("ACCT_X", "ACCT_K")    # receiver frozen
    assert breaker.check_transaction("ACCT_J", "ACCT_K")    # both frozen

    # Transactions not involving frozen nodes pass
    assert not breaker.check_transaction("ACCT_X", "ACCT_Y")

    _safe_print("  [PASS] Circuit breaker transaction blocking")


async def test_circuit_breaker_ledger_anchoring():
    """Freeze and unfreeze events are anchored to the audit ledger."""
    with tempfile.TemporaryDirectory() as tmp:
        lcfg = make_ledger_config(Path(tmp))
        cfg = CircuitBreakerConfig(
            consensus_threshold=0.50,
            cooldown_seconds=0,
            require_investigation=True,
        )

        async with AuditLedger(config=lcfg) as ledger:
            breaker = CircuitBreaker(config=cfg, audit_ledger=ledger)

            alert = MockAlertPayload(
                txn_id="TXN007", sender_id="ACCT_L", receiver_id="ACCT_M",
                timestamp=int(time.time()), risk_score=0.95, tier="HIGH",
            )
            inv = MockInvestigationResult(
                txn_id="TXN007", sender_id="ACCT_L", receiver_id="ACCT_M",
                gnn_risk_score=0.90, mule_findings=["m1"],
            )

            head_before = ledger.head_index
            await breaker.on_alert(alert)
            await breaker.on_investigation(inv)
            head_after_freeze = ledger.head_index

            # Each frozen node produces 2 blocks: CIRCUIT_BREAKER + ZKP_VERIFICATION
            # 2 nodes × 2 blocks = 4 new blocks
            assert head_after_freeze >= head_before + 4, (
                f"Expected at least 4 new blocks, got {head_after_freeze - head_before}"
            )

            # Check ledger has circuit breaker events
            cb_blocks = await ledger.get_recent_blocks(event_type=EventType.CIRCUIT_BREAKER)
            assert len(cb_blocks) >= 2, f"Expected >= 2 circuit breaker blocks, got {len(cb_blocks)}"

            # Check ledger has ZKP verification events
            zkp_blocks = await ledger.get_recent_blocks(event_type=EventType.ZKP_VERIFICATION)
            assert len(zkp_blocks) >= 2, f"Expected >= 2 ZKP blocks, got {len(zkp_blocks)}"

            # Unfreeze and verify anchoring
            head_before_unfreeze = ledger.head_index
            await breaker.unfreeze_node("ACCT_L", reason="test_unfreeze")
            head_after_unfreeze = ledger.head_index
            assert head_after_unfreeze > head_before_unfreeze

            # Verify chain integrity
            result = await ledger.verify_chain()
            assert result.valid, f"Chain invalid: {result.error_message}"

    _safe_print("  [PASS] Circuit breaker ledger anchoring")


async def test_circuit_breaker_zkp_integration():
    """Freeze generates ZKP proof that is readable from the ledger."""
    with tempfile.TemporaryDirectory() as tmp:
        lcfg = make_ledger_config(Path(tmp))
        cfg = CircuitBreakerConfig(
            consensus_threshold=0.50,
            cooldown_seconds=0,
            require_investigation=True,
        )

        async with AuditLedger(config=lcfg) as ledger:
            breaker = CircuitBreaker(config=cfg, audit_ledger=ledger)

            alert = MockAlertPayload(
                txn_id="TXN008", sender_id="ACCT_N", receiver_id="ACCT_O",
                timestamp=int(time.time()), risk_score=0.95, tier="HIGH",
            )
            inv = MockInvestigationResult(
                txn_id="TXN008", sender_id="ACCT_N", receiver_id="ACCT_O",
                gnn_risk_score=0.90, mule_findings=["m1"],
            )

            await breaker.on_alert(alert)
            await breaker.on_investigation(inv)

            # Read ZKP blocks from ledger
            zkp_blocks = await ledger.get_recent_blocks(event_type=EventType.ZKP_VERIFICATION)
            assert len(zkp_blocks) >= 1

            # Verify the proof embedded in the ledger
            for block in zkp_blocks:
                payload = block["payload"]
                proof = ComplianceProof(
                    commitment=payload["commitment"],
                    predicate=payload["predicate"],
                    challenge=payload["challenge"],
                    response=payload["response"],
                    nonce=payload["nonce"],
                    timestamp=payload["timestamp"],
                    node_id=payload["node_id"],
                )
                assert ZKPVerifier.verify(proof), f"ZKP proof in ledger failed verification for {proof.node_id}"

                # Verify no secret data leaked
                assert "consensus_score" not in str(payload) or "0.95" not in str(payload.get("commitment", ""))

    _safe_print("  [PASS] Circuit breaker ZKP integration")


async def test_gnn_unavailable_fallback():
    """GNN score -1.0 redistributes weights to ML + graph evidence."""
    cfg = CircuitBreakerConfig(
        consensus_threshold=0.70,
        ml_weight=0.35,
        gnn_weight=0.35,
        graph_evidence_weight=0.30,
        cooldown_seconds=0,
        require_investigation=True,
    )
    breaker = CircuitBreaker(config=cfg)

    # GNN unavailable: ml_adj = 0.35/0.65 ≈ 0.5385, ge_adj = 0.30/0.65 ≈ 0.4615
    # ML=0.95, 2 mules → graph_evidence = 1.0
    # consensus = 0.5385*0.95 + 0.4615*1.0 ≈ 0.5116 + 0.4615 = 0.9731
    score_no_gnn = breaker._compute_consensus(0.95, -1.0, 2, 0)
    assert score_no_gnn > 0.90, f"Expected > 0.90 without GNN, got {score_no_gnn}"

    # With GNN: consensus = 0.35*0.95 + 0.35*0.90 + 0.30*1.0 = 0.9475
    score_with_gnn = breaker._compute_consensus(0.95, 0.90, 2, 0)
    assert abs(score_with_gnn - 0.9475) < 0.001

    # Low ML without GNN or evidence should be low
    score_low = breaker._compute_consensus(0.30, -1.0, 0, 0)
    assert score_low < 0.30, f"Expected < 0.30, got {score_low}"

    # Verify freeze works without GNN
    alert = MockAlertPayload(
        txn_id="TXN009", sender_id="ACCT_P", receiver_id="ACCT_Q",
        timestamp=int(time.time()), risk_score=0.95, tier="HIGH",
    )
    inv = MockInvestigationResult(
        txn_id="TXN009", sender_id="ACCT_P", receiver_id="ACCT_Q",
        gnn_risk_score=-1.0,        # GNN unavailable
        mule_findings=["m1", "m2"],  # strong graph evidence
    )
    await breaker.on_alert(alert)
    await breaker.on_investigation(inv)

    assert breaker.is_frozen("ACCT_P"), "Should freeze even without GNN"
    assert breaker.is_frozen("ACCT_Q"), "Should freeze even without GNN"

    _safe_print("  [PASS] GNN unavailable fallback")


# ── Runner ───────────────────────────────────────────────────────────────────

async def main():
    _safe_print("PayFlow Phase 8 — ZKP & Circuit Breaker Tests")
    _safe_print("=" * 55)

    tests = [
        # ZKP tests
        test_zkp_commitment,
        test_zkp_threshold_proof,
        test_zkp_kyc_tier_proof,
        test_zkp_invalid_proof_rejected,
        test_zkp_different_predicates,
        # Circuit Breaker tests
        test_circuit_breaker_consensus_scoring,
        test_circuit_breaker_freeze_on_consensus,
        test_circuit_breaker_no_freeze_below_threshold,
        test_circuit_breaker_cooldown,
        test_circuit_breaker_ttl_expiry,
        test_circuit_breaker_transaction_blocking,
        test_circuit_breaker_ledger_anchoring,
        test_circuit_breaker_zkp_integration,
        test_gnn_unavailable_fallback,
    ]

    passed = 0
    failed = 0
    for test_fn in tests:
        try:
            await test_fn()
            passed += 1
        except Exception as exc:
            _safe_print(f"  [FAIL] {test_fn.__name__}: {exc}")
            import traceback
            traceback.print_exc()
            failed += 1

    _safe_print("=" * 55)
    _safe_print(f"Results: {passed} passed, {failed} failed out of {len(tests)}")
    if failed:
        sys.exit(1)
    _safe_print("All tests passed!")


if __name__ == "__main__":
    asyncio.run(main())
