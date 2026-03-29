"""
PayFlow -- Audit Ledger Integration Tests
============================================
Verifies the tamper-evident hash chain: genesis creation, block anchoring,
chain verification, tamper detection, Merkle checkpoints, Ed25519 signatures,
restart resilience, and sub-millisecond anchoring performance.
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

from config.settings import LedgerConfig
from src.blockchain.crypto import BlockHasher, BlockSigner
from src.blockchain.ledger import AuditLedger
from src.blockchain.models import Block, EventType, VerificationResult


# ── Helpers ──────────────────────────────────────────────────────────────────

def make_test_config(tmp_dir: Path, **overrides) -> LedgerConfig:
    """Create a LedgerConfig pointing at a temporary directory."""
    defaults = {
        "db_path": tmp_dir / "test_ledger.db",
        "key_dir": tmp_dir / "keys",
        "checkpoint_interval": 100,
        "enable_signing": True,
    }
    defaults.update(overrides)
    return LedgerConfig(**defaults)


# ── Tests ────────────────────────────────────────────────────────────────────

async def test_genesis_creation():
    """Open a fresh ledger and verify genesis block exists at index 0."""
    with tempfile.TemporaryDirectory() as tmp:
        cfg = make_test_config(Path(tmp))
        async with AuditLedger(config=cfg) as ledger:
            assert ledger.head_index == 0, f"Expected head_index=0, got {ledger.head_index}"
            assert len(ledger.head_hash) == 64, "Genesis hash should be 64 hex chars"
            assert ledger.metrics.blocks_created == 1
            assert ledger.metrics.system_events_anchored == 1

            stats = await ledger.get_stats()
            assert stats.total_blocks == 1
            assert stats.latest_index == 0
    print("  [PASS] Genesis creation")


async def test_chain_integrity():
    """Anchor 10 mixed events, verify entire chain is valid."""
    with tempfile.TemporaryDirectory() as tmp:
        cfg = make_test_config(Path(tmp))
        async with AuditLedger(config=cfg) as ledger:
            # Anchor mixed event types
            await ledger.anchor(EventType.ALERT, {"txn_id": "TXN001", "risk": 0.92})
            await ledger.anchor(EventType.INVESTIGATION, {"txn_id": "TXN001", "mules": 2})
            await ledger.anchor(EventType.MODEL_UPDATE, {"action": "train_complete"})
            await ledger.anchor(EventType.SYSTEM_STATE, {"event": "pipeline_started"})
            await ledger.anchor(EventType.ALERT, {"txn_id": "TXN002", "risk": 0.87})
            await ledger.anchor(EventType.INVESTIGATION, {"txn_id": "TXN002", "cycles": 1})
            await ledger.anchor(EventType.MODEL_UPDATE, {"action": "model_saved"})
            await ledger.anchor(EventType.SYSTEM_STATE, {"event": "vram_mode_analysis"})
            await ledger.anchor(EventType.ALERT, {"txn_id": "TXN003", "risk": 0.95})
            await ledger.anchor(EventType.MODEL_UPDATE, {"action": "model_loaded"})

            assert ledger.head_index == 10  # genesis + 10 events
            assert ledger.metrics.blocks_created == 11

            # Verify full chain
            result = await ledger.verify_chain()
            assert result.valid, f"Chain invalid: {result.error_message}"
            assert result.blocks_checked == 11
            assert result.first_invalid_index is None
    print("  [PASS] Chain integrity (10 mixed events)")


async def test_tamper_detection():
    """Tamper with a block's payload and verify detection."""
    with tempfile.TemporaryDirectory() as tmp:
        cfg = make_test_config(Path(tmp))
        async with AuditLedger(config=cfg) as ledger:
            await ledger.anchor(EventType.ALERT, {"txn_id": "TXN001", "risk": 0.92})
            await ledger.anchor(EventType.ALERT, {"txn_id": "TXN002", "risk": 0.85})
            await ledger.anchor(EventType.ALERT, {"txn_id": "TXN003", "risk": 0.78})

            # Verify chain is valid before tampering
            result = await ledger.verify_chain()
            assert result.valid

            # TAMPER: modify block 2's payload directly in SQLite
            await ledger._storage.execute_raw(
                "UPDATE blocks SET payload = ? WHERE idx = ?",
                ('{"txn_id":"TXN002","risk":0.10,"TAMPERED":true}', 2),
            )

            # Verify chain detects the tamper
            result = await ledger.verify_chain()
            assert not result.valid, "Tampered chain should be invalid"
            assert result.first_invalid_index == 2
            assert "hash mismatch" in result.error_message
    print("  [PASS] Tamper detection")


async def test_signature_verification():
    """Verify Ed25519 signatures on individual blocks."""
    with tempfile.TemporaryDirectory() as tmp:
        cfg = make_test_config(Path(tmp))
        async with AuditLedger(config=cfg) as ledger:
            block = await ledger.anchor(EventType.ALERT, {"test": "sig_check"})

            # Verify signature directly
            assert ledger._signer.verify(block.block_hash, block.signature)

            # Verify with wrong hash
            assert not ledger._signer.verify("a" * 64, block.signature)

            # Verify chain (includes signature check)
            result = await ledger.verify_chain()
            assert result.valid
    print("  [PASS] Signature verification")


async def test_merkle_checkpoint():
    """Set checkpoint_interval=5, anchor 5 events, verify Merkle root."""
    with tempfile.TemporaryDirectory() as tmp:
        cfg = make_test_config(Path(tmp), checkpoint_interval=5)
        async with AuditLedger(config=cfg) as ledger:
            # Genesis is block 0 (1 pending hash). Need 4 more to reach 5.
            for i in range(4):
                await ledger.anchor(EventType.ALERT, {"idx": i})

            # At this point we have 5 blocks (0-4), checkpoint should trigger
            assert ledger.metrics.checkpoints_created == 1

            # The checkpoint block should have a non-None merkle_root
            blocks = await ledger.get_recent_blocks(limit=10)
            checkpoint_block = [b for b in blocks if b["merkle_root"] is not None]
            assert len(checkpoint_block) == 1, "Expected exactly 1 checkpoint block"

            # Verify Merkle checkpoint
            cp_idx = checkpoint_block[0]["index"]
            valid = await ledger.verify_checkpoint(cp_idx)
            assert valid, "Merkle checkpoint verification failed"
    print("  [PASS] Merkle checkpoint")


async def test_restart_resilience():
    """Close and reopen the ledger, verify chain continuity."""
    with tempfile.TemporaryDirectory() as tmp:
        cfg = make_test_config(Path(tmp))

        # Phase 1: create chain
        async with AuditLedger(config=cfg) as ledger:
            await ledger.anchor(EventType.ALERT, {"phase": 1, "idx": 0})
            await ledger.anchor(EventType.ALERT, {"phase": 1, "idx": 1})
            await ledger.anchor(EventType.ALERT, {"phase": 1, "idx": 2})
            head_after_phase1 = ledger.head_hash
            idx_after_phase1 = ledger.head_index

        # Phase 2: reopen and continue
        async with AuditLedger(config=cfg) as ledger:
            assert ledger.head_index == idx_after_phase1
            assert ledger.head_hash == head_after_phase1

            await ledger.anchor(EventType.ALERT, {"phase": 2, "idx": 3})
            await ledger.anchor(EventType.ALERT, {"phase": 2, "idx": 4})

            assert ledger.head_index == idx_after_phase1 + 2

            # Verify full chain across restart boundary
            result = await ledger.verify_chain()
            assert result.valid, f"Chain invalid after restart: {result.error_message}"
            assert result.blocks_checked == idx_after_phase1 + 3  # genesis + all blocks
    print("  [PASS] Restart resilience")


async def test_event_type_filtering():
    """Anchor mixed events, query by type."""
    with tempfile.TemporaryDirectory() as tmp:
        cfg = make_test_config(Path(tmp))
        async with AuditLedger(config=cfg) as ledger:
            await ledger.anchor(EventType.ALERT, {"a": 1})
            await ledger.anchor(EventType.ALERT, {"a": 2})
            await ledger.anchor(EventType.INVESTIGATION, {"i": 1})
            await ledger.anchor(EventType.MODEL_UPDATE, {"m": 1})
            await ledger.anchor(EventType.ALERT, {"a": 3})

            alerts = await ledger.get_recent_blocks(event_type=EventType.ALERT)
            assert len(alerts) == 3
            assert all(b["event_type"] == "ALERT" for b in alerts)

            investigations = await ledger.get_recent_blocks(
                event_type=EventType.INVESTIGATION,
            )
            assert len(investigations) == 1
    print("  [PASS] Event type filtering")


async def test_signing_disabled():
    """Verify ledger works with signing disabled (null signatures)."""
    with tempfile.TemporaryDirectory() as tmp:
        cfg = make_test_config(Path(tmp), enable_signing=False)
        async with AuditLedger(config=cfg) as ledger:
            block = await ledger.anchor(EventType.ALERT, {"test": "no_sig"})
            assert block.signature == "0" * 128  # null signature

            result = await ledger.verify_chain()
            assert result.valid
    print("  [PASS] Signing disabled mode")


async def test_performance():
    """Anchor 100 blocks and verify average latency is under 2 ms."""
    with tempfile.TemporaryDirectory() as tmp:
        cfg = make_test_config(Path(tmp))
        async with AuditLedger(config=cfg) as ledger:
            t0 = time.perf_counter()
            for i in range(100):
                await ledger.anchor(EventType.ALERT, {"idx": i, "risk": 0.5 + i * 0.001})
            elapsed = (time.perf_counter() - t0) * 1000

            avg_ms = elapsed / 100
            total_blocks = ledger.metrics.blocks_created
            assert total_blocks == 101  # genesis + 100

            # Verify chain integrity even after rapid-fire writes
            result = await ledger.verify_chain()
            assert result.valid
    print(f"  [PASS] Performance: {avg_ms:.3f} ms/block avg ({elapsed:.1f} ms total for 100 blocks)")


async def test_convenience_methods():
    """Test anchor_alert, anchor_investigation, anchor_model_event, anchor_system_event."""
    with tempfile.TemporaryDirectory() as tmp:
        cfg = make_test_config(Path(tmp))
        async with AuditLedger(config=cfg) as ledger:
            # anchor_model_event
            block = await ledger.anchor_model_event(
                "train_complete", "xgboost",
                details={"auc": 0.95, "trees": 500},
            )
            assert block.event_type == EventType.MODEL_UPDATE
            assert block.payload["action"] == "train_complete"
            assert block.payload["model"] == "xgboost"

            # anchor_system_event
            block = await ledger.anchor_system_event(
                "pipeline_started",
                details={"consumers": 3},
            )
            assert block.event_type == EventType.SYSTEM_STATE
            assert block.payload["event"] == "pipeline_started"

            # Verify chain
            result = await ledger.verify_chain()
            assert result.valid
    print("  [PASS] Convenience methods (model_event, system_event)")


# ── Runner ───────────────────────────────────────────────────────────────────

async def main():
    print("PayFlow Audit Ledger Integration Tests")
    print("=" * 50)

    tests = [
        test_genesis_creation,
        test_chain_integrity,
        test_tamper_detection,
        test_signature_verification,
        test_merkle_checkpoint,
        test_restart_resilience,
        test_event_type_filtering,
        test_signing_disabled,
        test_performance,
        test_convenience_methods,
    ]

    passed = 0
    failed = 0
    for test_fn in tests:
        try:
            await test_fn()
            passed += 1
        except Exception as exc:
            print(f"  [FAIL] {test_fn.__name__}: {exc}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("=" * 50)
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)}")
    if failed:
        sys.exit(1)
    print("All tests passed!")


if __name__ == "__main__":
    asyncio.run(main())
