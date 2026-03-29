"""
PayFlow -- Audit Ledger Orchestrator
=======================================
Tamper-evident hash chain with Ed25519 signatures and Merkle checkpoints.

Pipeline Integration::

    pipeline.add_consumer(ledger.ingest)   # EventBatch consumer protocol
    router.register_ledger_consumer(ledger.anchor_alert)  # alert anchoring

Design:
    - Async-first: all I/O through aiosqlite
    - Non-blocking: block creation (hash + sign + insert) < 1 ms
    - Single-writer: asyncio.Lock protects chain head state
    - In-memory head cache: avoids DB reads on every append

Block Structure:
    index | timestamp | event_type | payload_json | prev_hash | block_hash | signature | merkle_root
      0       ...       SYSTEM       {genesis}      000...000    abc...def    sig_hex      None
      1       ...       ALERT        {alert}        abc...def    123...456    sig_hex      None
      ...
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field

from src.blockchain.crypto import BlockHasher, BlockSigner, MerkleCheckpointer
from src.blockchain.models import Block, EventType, LedgerStats, VerificationResult
from src.blockchain.storage import LedgerStorage

logger = logging.getLogger(__name__)

_GENESIS_HASH = "0" * 64


# ── Metrics ──────────────────────────────────────────────────────────────────

@dataclass
class LedgerMetrics:
    """Runtime performance counters for the audit ledger."""
    blocks_created: int = 0
    alerts_anchored: int = 0
    investigations_anchored: int = 0
    model_updates_anchored: int = 0
    system_events_anchored: int = 0
    zkp_verifications_anchored: int = 0
    circuit_breaker_events_anchored: int = 0
    agent_verdicts_anchored: int = 0
    checkpoints_created: int = 0
    total_anchor_ms: float = 0.0
    verification_runs: int = 0
    _start_time: float = field(default_factory=time.monotonic)

    @property
    def uptime_sec(self) -> float:
        return time.monotonic() - self._start_time

    def snapshot(self) -> dict:
        avg = self.total_anchor_ms / max(self.blocks_created, 1)
        return {
            "blocks": self.blocks_created,
            "alerts": self.alerts_anchored,
            "investigations": self.investigations_anchored,
            "model_updates": self.model_updates_anchored,
            "system_events": self.system_events_anchored,
            "zkp_verifications": self.zkp_verifications_anchored,
            "circuit_breaker": self.circuit_breaker_events_anchored,
            "agent_verdicts": self.agent_verdicts_anchored,
            "checkpoints": self.checkpoints_created,
            "avg_anchor_ms": round(avg, 3),
            "verifications": self.verification_runs,
            "uptime_sec": round(self.uptime_sec, 1),
        }


# ── Audit Ledger ─────────────────────────────────────────────────────────────

class AuditLedger:
    """
    Tamper-evident audit ledger with cryptographic hash chain.

    Lifecycle::

        ledger = AuditLedger()
        await ledger.open()
        ...
        await ledger.close()

    As pipeline consumer::

        pipeline.add_consumer(ledger.ingest)

    Direct anchoring::

        await ledger.anchor(EventType.ALERT, alert.to_dict())
        await ledger.anchor(EventType.MODEL_UPDATE, {"action": "train_complete", ...})

    Verification::

        result = await ledger.verify_chain()
        assert result.valid
    """

    def __init__(self, config=None) -> None:
        from config.settings import LEDGER_CFG
        self._cfg = config or LEDGER_CFG

        self._storage = LedgerStorage(self._cfg.db_path)
        self._signer = BlockSigner(self._cfg.key_dir) if self._cfg.enable_signing else None
        self._checkpointer = MerkleCheckpointer()

        # In-memory chain head state
        self._head_index: int = -1
        self._head_hash: str = _GENESIS_HASH
        self._lock = asyncio.Lock()

        self.metrics = LedgerMetrics()

    # ── Lifecycle ────────────────────────────────────────────────────────

    async def open(self) -> None:
        """
        Open storage, initialise signing keys, load chain head,
        create genesis block if needed.
        """
        await self._storage.open()

        if self._signer is not None:
            self._signer.initialize()

        # Load chain head from database
        head_row = await self._storage.get_head()
        if head_row is None:
            await self._create_genesis()
        else:
            self._head_index = head_row[0]   # idx column
            self._head_hash = head_row[5]    # block_hash column
            await self._reseed_checkpointer()

        logger.info(
            "Audit ledger opened: head_index=%d, head_hash=%s...%s",
            self._head_index, self._head_hash[:8], self._head_hash[-8:],
        )

    async def close(self) -> None:
        """Flush any pending writes and close storage."""
        await self._storage.close()
        logger.info("Audit ledger closed. Final metrics: %s", self.metrics.snapshot())

    async def __aenter__(self):
        await self.open()
        return self

    async def __aexit__(self, *exc):
        await self.close()

    # ── Core: Block Creation ─────────────────────────────────────────────

    async def anchor(self, event_type: EventType, payload: dict) -> Block:
        """
        Create and persist a new block anchoring the given event.

        Safe to call concurrently from multiple coroutines; the asyncio.Lock
        serialises chain-head access.

        Args:
            event_type: category of the event being anchored.
            payload: JSON-serialisable dict of event data.

        Returns:
            The newly created Block.
        """
        async with self._lock:
            t0 = time.perf_counter()

            new_index = self._head_index + 1
            now = time.time()

            # Compute block hash
            block_hash = BlockHasher.compute_hash(
                index=new_index,
                timestamp=now,
                event_type=int(event_type),
                payload=payload,
                previous_hash=self._head_hash,
            )

            # Sign (or use null signature)
            if self._signer is not None:
                signature = self._signer.sign(block_hash)
            else:
                signature = BlockSigner.null_signature()

            # Merkle accumulation
            self._checkpointer.add_hash(block_hash)
            merkle_root = None
            if self._checkpointer.pending_count >= self._cfg.checkpoint_interval:
                merkle_root = self._checkpointer.compute_checkpoint()
                self.metrics.checkpoints_created += 1

            # Build block
            block = Block(
                index=new_index,
                timestamp=now,
                event_type=event_type,
                payload=payload,
                previous_hash=self._head_hash,
                block_hash=block_hash,
                signature=signature,
                merkle_root=merkle_root,
            )

            # Persist
            await self._storage.insert_block(block.to_row())

            # Update in-memory head
            self._head_index = new_index
            self._head_hash = block_hash

            # Update metrics
            elapsed = (time.perf_counter() - t0) * 1000
            self.metrics.blocks_created += 1
            self.metrics.total_anchor_ms += elapsed
            self._increment_type_counter(event_type)

            if self.metrics.blocks_created % 100 == 0:
                logger.info("Ledger metrics: %s", self.metrics.snapshot())

            return block

    # ── Pipeline Consumer Protocol ───────────────────────────────────────

    async def ingest(self, batch) -> None:
        """
        Consumer protocol: ``async def ingest(batch: EventBatch) -> None``.

        Anchors a batch-level summary as a SYSTEM_STATE event.
        Individual high-value events (alerts, investigations) are anchored
        separately via direct ``anchor()`` calls from their respective modules.
        """
        payload = {
            "event": "batch_processed",
            "batch_id": batch.batch_id,
            "batch_timestamp": batch.batch_timestamp,
            "event_count": batch.event_count,
            "txn_count": len(batch.transactions),
            "msg_count": len(batch.interbank_messages),
            "auth_count": len(batch.auth_events),
        }
        await self.anchor(EventType.SYSTEM_STATE, payload)

    # ── Convenience Anchoring Methods ────────────────────────────────────

    async def anchor_alert(self, alert_payload) -> Block:
        """
        Anchor a high-risk AlertPayload.

        Converts the AlertPayload to a sanitised dict (no numpy arrays).
        Called from AlertRouter after dispatch.
        """
        payload = alert_payload.to_dict()
        payload["_source"] = "alert_router"
        return await self.anchor(EventType.ALERT, payload)

    async def anchor_investigation(self, investigation_result) -> Block:
        """
        Anchor an InvestigationResult from TransactionGraph.

        Serialises the NamedTuple fields (complex sub-objects are summarised
        as counts for space efficiency).
        """
        payload = {
            "_source": "transaction_graph",
            "txn_id": investigation_result.txn_id,
            "sender_id": investigation_result.sender_id,
            "receiver_id": investigation_result.receiver_id,
            "subgraph_nodes": investigation_result.subgraph_nodes,
            "subgraph_edges": investigation_result.subgraph_edges,
            "mule_findings_count": len(investigation_result.mule_findings),
            "cycle_findings_count": len(investigation_result.cycle_findings),
            "investigation_ms": round(investigation_result.investigation_ms, 2),
            "gnn_risk_score": investigation_result.gnn_risk_score,
        }
        return await self.anchor(EventType.INVESTIGATION, payload)

    async def anchor_model_event(
        self, action: str, model_name: str, details: dict | None = None,
    ) -> Block:
        """
        Anchor a model lifecycle event (train, save, load).

        Args:
            action: ``"train_complete"``, ``"model_saved"``, ``"model_loaded"``
            model_name: ``"xgboost"`` or ``"gnn"``
            details: optional dict with metrics, path, etc.
        """
        payload: dict = {
            "_source": "model_lifecycle",
            "action": action,
            "model": model_name,
        }
        if details:
            payload["details"] = details
        return await self.anchor(EventType.MODEL_UPDATE, payload)

    async def anchor_system_event(
        self, event: str, details: dict | None = None,
    ) -> Block:
        """Anchor a system state change (pipeline start/stop, VRAM mode transition)."""
        payload: dict = {"event": event}
        if details:
            payload.update(details)
        return await self.anchor(EventType.SYSTEM_STATE, payload)

    async def anchor_zkp_proof(self, proof) -> Block:
        """
        Anchor a zero-knowledge proof verification.

        The proof contains only the commitment, predicate, challenge,
        response, and nonce — NEVER the secret value.
        """
        payload = proof.to_dict()
        payload["_source"] = "zkp_verifier"
        return await self.anchor(EventType.ZKP_VERIFICATION, payload)

    async def anchor_circuit_breaker(
        self, action: str, details: dict | None = None,
    ) -> Block:
        """Anchor a circuit breaker action (freeze/unfreeze)."""
        payload: dict = {"action": action}
        if details:
            payload["details"] = details
        return await self.anchor(EventType.CIRCUIT_BREAKER, payload)

    async def anchor_agent_verdict(self, verdict_payload) -> Block:
        """
        Anchor an investigator agent verdict.

        Stores the structured verdict (fraud typology, confidence, reasoning
        summary, recommended action) as a tamper-evident ledger entry.
        """
        payload = verdict_payload.to_dict()
        payload["_source"] = "investigator_agent"
        return await self.anchor(EventType.AGENT_VERDICT, payload)

    # ── Verification ─────────────────────────────────────────────────────

    async def verify_chain(
        self, start_idx: int = 0, end_idx: int | None = None,
    ) -> VerificationResult:
        """
        Walk the chain and verify hash linkage + signatures.

        Args:
            start_idx: first block to verify (default: 0 = genesis).
            end_idx: last block to verify (default: current head).

        Returns:
            VerificationResult indicating chain integrity.
        """
        t0 = time.perf_counter()
        if end_idx is None:
            end_idx = self._head_index

        rows = await self._storage.get_block_range(start_idx, end_idx)
        if not rows:
            return VerificationResult(
                valid=True, blocks_checked=0,
                first_invalid_index=None, error_message="",
                elapsed_ms=(time.perf_counter() - t0) * 1000,
            )

        prev_hash = _GENESIS_HASH if start_idx == 0 else None

        # If starting mid-chain, fetch the predecessor's hash
        if start_idx > 0:
            pred_row = await self._storage.get_block(start_idx - 1)
            if pred_row is None:
                return VerificationResult(
                    valid=False, blocks_checked=0,
                    first_invalid_index=start_idx,
                    error_message=f"Predecessor block {start_idx - 1} not found",
                    elapsed_ms=(time.perf_counter() - t0) * 1000,
                )
            prev_hash = pred_row[5]  # block_hash column

        for row in rows:
            idx, ts, etype, payload_json, stored_prev, stored_hash, sig, _mroot = row

            # 1. Verify linkage
            if prev_hash is not None and stored_prev != prev_hash:
                return VerificationResult(
                    valid=False, blocks_checked=idx - start_idx,
                    first_invalid_index=idx,
                    error_message=(
                        f"Block {idx}: prev_hash mismatch "
                        f"(expected {prev_hash[:16]}..., got {stored_prev[:16]}...)"
                    ),
                    elapsed_ms=(time.perf_counter() - t0) * 1000,
                )

            # 2. Recompute hash
            payload_dict = json.loads(payload_json)
            expected_hash = BlockHasher.compute_hash(
                idx, ts, etype, payload_dict, stored_prev,
            )
            if expected_hash != stored_hash:
                return VerificationResult(
                    valid=False, blocks_checked=idx - start_idx,
                    first_invalid_index=idx,
                    error_message=(
                        f"Block {idx}: hash mismatch "
                        f"(recomputed {expected_hash[:16]}..., stored {stored_hash[:16]}...)"
                    ),
                    elapsed_ms=(time.perf_counter() - t0) * 1000,
                )

            # 3. Verify signature
            if self._signer is not None and sig != BlockSigner.null_signature():
                if not self._signer.verify(stored_hash, sig):
                    return VerificationResult(
                        valid=False, blocks_checked=idx - start_idx,
                        first_invalid_index=idx,
                        error_message=f"Block {idx}: Ed25519 signature invalid",
                        elapsed_ms=(time.perf_counter() - t0) * 1000,
                    )

            prev_hash = stored_hash

        elapsed = (time.perf_counter() - t0) * 1000
        self.metrics.verification_runs += 1
        return VerificationResult(
            valid=True, blocks_checked=len(rows),
            first_invalid_index=None, error_message="",
            elapsed_ms=elapsed,
        )

    async def verify_block(self, idx: int) -> bool:
        """Verify a single block's hash and signature integrity."""
        result = await self.verify_chain(start_idx=idx, end_idx=idx)
        return result.valid

    async def verify_checkpoint(self, checkpoint_idx: int) -> bool:
        """
        Reconstruct the Merkle tree for a checkpoint range and compare roots.

        Finds the checkpoint block, determines its range (from the previous
        checkpoint + 1 to the checkpoint block itself), reconstructs the tree
        from stored block hashes, and compares against the stored merkle_root.
        """
        cp_row = await self._storage.get_block(checkpoint_idx)
        if cp_row is None or cp_row[7] is None:  # merkle_root column
            return False

        stored_root = cp_row[7]

        # Find range start (block after previous checkpoint, or 0)
        checkpoints = await self._storage.get_checkpoints()
        range_start = 0
        for cp in checkpoints:
            if cp[0] < checkpoint_idx:
                range_start = cp[0] + 1
            else:
                break

        rows = await self._storage.get_block_range(range_start, checkpoint_idx)

        from pymerkle import InmemoryTree

        tree = InmemoryTree()
        for row in rows:
            tree.append_entry(row[5].encode("utf-8"))  # block_hash column

        computed_root = tree.get_state().hex()
        return computed_root == stored_root

    # ── Query API ────────────────────────────────────────────────────────

    async def get_stats(self) -> LedgerStats:
        """Return summary statistics for monitoring."""
        count = await self._storage.get_block_count()
        checkpoints = await self._storage.get_checkpoints()
        db_size = await self._storage.get_db_size_bytes()
        head = await self._storage.get_head()

        return LedgerStats(
            total_blocks=count,
            latest_index=head[0] if head else -1,
            latest_hash=head[5] if head else _GENESIS_HASH,
            latest_timestamp=head[1] if head else 0.0,
            checkpoints=len(checkpoints),
            db_size_bytes=db_size,
        )

    async def get_recent_blocks(
        self, event_type: EventType | None = None, limit: int = 50,
    ) -> list[dict]:
        """Query recent blocks, optionally filtered by event type."""
        if event_type is not None:
            rows = await self._storage.get_blocks_by_event_type(int(event_type), limit)
        else:
            rows = await self._storage.get_block_range(
                max(0, self._head_index - limit + 1), self._head_index,
            )

        results = []
        for row in rows:
            results.append({
                "index": row[0],
                "timestamp": row[1],
                "event_type": EventType(row[2]).name,
                "payload": json.loads(row[3]),
                "prev_hash": row[4],
                "block_hash": row[5],
                "has_signature": row[6] != BlockSigner.null_signature(),
                "merkle_root": row[7],
            })
        return results

    # ── Diagnostics ──────────────────────────────────────────────────────

    @property
    def head_index(self) -> int:
        return self._head_index

    @property
    def head_hash(self) -> str:
        return self._head_hash

    def snapshot(self) -> dict:
        """Full ledger state for monitoring dashboards."""
        return {
            "head_index": self._head_index,
            "head_hash": self._head_hash[:16] + "...",
            "metrics": self.metrics.snapshot(),
        }

    # ── Internal ─────────────────────────────────────────────────────────

    async def _create_genesis(self) -> None:
        """Create the genesis block (index 0)."""
        now = time.time()
        genesis_payload = {
            "event": "genesis",
            "version": "1.0.0",
            "created_at": now,
        }
        block_hash = BlockHasher.compute_hash(
            index=0,
            timestamp=now,
            event_type=int(EventType.SYSTEM_STATE),
            payload=genesis_payload,
            previous_hash=_GENESIS_HASH,
        )

        signature = (
            self._signer.sign(block_hash)
            if self._signer is not None
            else BlockSigner.null_signature()
        )

        genesis = Block(
            index=0,
            timestamp=now,
            event_type=EventType.SYSTEM_STATE,
            payload=genesis_payload,
            previous_hash=_GENESIS_HASH,
            block_hash=block_hash,
            signature=signature,
            merkle_root=None,
        )

        self._checkpointer.add_hash(block_hash)
        await self._storage.insert_block(genesis.to_row())

        self._head_index = 0
        self._head_hash = block_hash
        self.metrics.blocks_created = 1
        self.metrics.system_events_anchored = 1
        logger.info("Genesis block created: %s", block_hash)

    async def _reseed_checkpointer(self) -> None:
        """
        On restart, re-seed the Merkle checkpointer with block hashes
        accumulated since the last checkpoint.
        """
        checkpoints = await self._storage.get_checkpoints()
        if checkpoints:
            last_cp_idx = checkpoints[-1][0]
            range_start = last_cp_idx + 1
        else:
            range_start = 0

        if range_start <= self._head_index:
            rows = await self._storage.get_block_range(range_start, self._head_index)
            for row in rows:
                self._checkpointer.add_hash(row[5])  # block_hash column

        logger.debug(
            "Merkle checkpointer re-seeded with %d pending hashes",
            self._checkpointer.pending_count,
        )

    def _increment_type_counter(self, event_type: EventType) -> None:
        """Update per-type metric counters."""
        if event_type == EventType.ALERT:
            self.metrics.alerts_anchored += 1
        elif event_type == EventType.INVESTIGATION:
            self.metrics.investigations_anchored += 1
        elif event_type == EventType.MODEL_UPDATE:
            self.metrics.model_updates_anchored += 1
        elif event_type == EventType.SYSTEM_STATE:
            self.metrics.system_events_anchored += 1
        elif event_type == EventType.ZKP_VERIFICATION:
            self.metrics.zkp_verifications_anchored += 1
        elif event_type == EventType.CIRCUIT_BREAKER:
            self.metrics.circuit_breaker_events_anchored += 1
        elif event_type == EventType.AGENT_VERDICT:
            self.metrics.agent_verdicts_anchored += 1
