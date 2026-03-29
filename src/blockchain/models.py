"""
PayFlow -- Audit Ledger Data Models
=====================================
Immutable data structures for the tamper-evident hash-chain ledger.

Block:
    Each block contains an event payload, cryptographic hash linking it to
    its predecessor, an Ed25519 digital signature, and an optional Merkle
    tree checkpoint root.

EventType:
    Categorises ledger entries by source subsystem so that auditors can
    filter and verify specific event streams independently.
"""

from __future__ import annotations

import enum
import json
from dataclasses import dataclass
from typing import NamedTuple


class EventType(enum.IntEnum):
    """Categorises ledger entries by source subsystem."""
    SYSTEM_STATE     = 0   # genesis, pipeline start/stop, VRAM mode transition
    ALERT            = 1   # AlertPayload from AlertRouter (HIGH/MEDIUM risk)
    INVESTIGATION    = 2   # InvestigationResult from TransactionGraph
    MODEL_UPDATE     = 3   # XGBoost/GNN train, save, load events
    ZKP_VERIFICATION = 4   # zero-knowledge proof verification
    CIRCUIT_BREAKER  = 5   # node freeze/unfreeze actions
    AGENT_VERDICT    = 6   # investigator agent fraud verdict


@dataclass(frozen=True)
class Block:
    """
    Single block in the audit hash chain.

    Frozen to prevent post-creation tampering in memory.
    All fields are JSON-serialisable for canonical hashing.
    """
    index: int
    timestamp: float
    event_type: EventType
    payload: dict
    previous_hash: str
    block_hash: str
    signature: str
    merkle_root: str | None = None

    def to_row(self) -> tuple:
        """Convert to SQLite INSERT parameter tuple."""
        return (
            self.index,
            self.timestamp,
            int(self.event_type),
            json.dumps(self.payload, sort_keys=True, ensure_ascii=True),
            self.previous_hash,
            self.block_hash,
            self.signature,
            self.merkle_root,
        )


class VerificationResult(NamedTuple):
    """Result of a chain verification operation."""
    valid: bool
    blocks_checked: int
    first_invalid_index: int | None   # None if all valid
    error_message: str                # empty string if valid
    elapsed_ms: float


class LedgerStats(NamedTuple):
    """Summary statistics for monitoring dashboards."""
    total_blocks: int
    latest_index: int
    latest_hash: str
    latest_timestamp: float
    checkpoints: int
    db_size_bytes: int
