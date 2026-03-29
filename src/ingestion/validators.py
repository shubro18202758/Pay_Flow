"""
PayFlow — Data Integrity Validators
=====================================
Validates every event at ingestion time using CRC32 checksums and
structural integrity rules. Rejects malformed events BEFORE they
enter the pipeline, preventing garbage propagation to ML/graph stages.

CRC32 chosen over SHA-256 for the hot path:
  - CRC32: ~3.5 GB/s throughput (hardware-accelerated on x86 via SSE4.2)
  - SHA-256: ~0.5 GB/s
  At 100K events/sec × ~85 bytes each, CRC32 adds <1ms overhead per batch.
  SHA-256 would add ~7ms — acceptable but unnecessary for integrity checks.
  (Cryptographic hashing is used separately in the blockchain audit trail.)
"""

from __future__ import annotations

import logging
import zlib
from dataclasses import dataclass
from typing import Union

import msgspec

from src.ingestion.schemas import (
    AccountType,
    AuthAction,
    AuthEvent,
    Channel,
    InterbankMessage,
    Transaction,
)

logger = logging.getLogger(__name__)

Event = Union[Transaction, InterbankMessage, AuthEvent]


# ── CRC32 Checksum Computation ────────────────────────────────────────────────

def compute_transaction_checksum(
    txn_id: str,
    timestamp: int,
    sender_id: str,
    receiver_id: str,
    amount_paisa: int,
    channel: int,
) -> int:
    """
    Compute CRC32 over the critical fields that MUST NOT be tampered with.
    We hash the canonical byte representation of the financial core fields.
    Non-critical fields (geo, fingerprint) are excluded to allow metadata
    enrichment without invalidating the checksum.
    """
    payload = f"{txn_id}|{timestamp}|{sender_id}|{receiver_id}|{amount_paisa}|{channel}"
    return zlib.crc32(payload.encode("utf-8")) & 0xFFFFFFFF


def compute_interbank_checksum(
    msg_id: str,
    timestamp: int,
    sender_ifsc: str,
    receiver_ifsc: str,
    amount_paisa: int,
    channel: int,
) -> int:
    payload = f"{msg_id}|{timestamp}|{sender_ifsc}|{receiver_ifsc}|{amount_paisa}|{channel}"
    return zlib.crc32(payload.encode("utf-8")) & 0xFFFFFFFF


def compute_auth_checksum(
    event_id: str,
    timestamp: int,
    account_id: str,
    action: int,
    ip_address: str,
) -> int:
    payload = f"{event_id}|{timestamp}|{account_id}|{action}|{ip_address}"
    return zlib.crc32(payload.encode("utf-8")) & 0xFFFFFFFF


# ── Validation Results ────────────────────────────────────────────────────────

@dataclass
class ValidationResult:
    valid: bool
    event_id: str
    errors: list[str]


# ── Structural + Integrity Validators ─────────────────────────────────────────

def validate_transaction(txn: Transaction) -> ValidationResult:
    """Full validation: checksum integrity + structural rules."""
    errors: list[str] = []

    # Checksum integrity
    expected_crc = compute_transaction_checksum(
        txn.txn_id, txn.timestamp, txn.sender_id,
        txn.receiver_id, txn.amount_paisa, int(txn.channel),
    )
    if txn.checksum != expected_crc:
        errors.append(
            f"checksum mismatch: got {txn.checksum}, expected {expected_crc}"
        )

    # Structural rules
    if txn.amount_paisa <= 0:
        errors.append("amount must be positive")

    if txn.sender_id == txn.receiver_id:
        errors.append("self-transfer: sender_id == receiver_id")

    if txn.timestamp <= 0:
        errors.append("timestamp must be positive unix epoch")

    if not txn.txn_id.startswith("TXN"):
        errors.append(f"invalid txn_id prefix: {txn.txn_id[:10]}")

    if len(txn.device_fingerprint) != 16:
        errors.append(f"device_fingerprint must be 16 hex chars, got {len(txn.device_fingerprint)}")

    # Geo bounds (India approximate bounding box)
    if not (6.0 <= txn.sender_geo_lat <= 37.0 and 68.0 <= txn.sender_geo_lon <= 98.0):
        # Allow (0.0, 0.0) as "location unavailable" sentinel
        if not (txn.sender_geo_lat == 0.0 and txn.sender_geo_lon == 0.0):
            errors.append(f"sender geo out of India bounds: ({txn.sender_geo_lat}, {txn.sender_geo_lon})")

    return ValidationResult(
        valid=len(errors) == 0,
        event_id=txn.txn_id,
        errors=errors,
    )


def validate_interbank_message(msg: InterbankMessage) -> ValidationResult:
    errors: list[str] = []

    expected_crc = compute_interbank_checksum(
        msg.msg_id, msg.timestamp, msg.sender_ifsc,
        msg.receiver_ifsc, msg.amount_paisa, int(msg.channel),
    )
    if msg.checksum != expected_crc:
        errors.append(f"checksum mismatch: got {msg.checksum}, expected {expected_crc}")

    if msg.amount_paisa <= 0:
        errors.append("amount must be positive")

    if len(msg.sender_ifsc) != 11:
        errors.append(f"sender_ifsc must be 11 chars, got {len(msg.sender_ifsc)}")

    if len(msg.receiver_ifsc) != 11:
        errors.append(f"receiver_ifsc must be 11 chars, got {len(msg.receiver_ifsc)}")

    if not msg.msg_id.startswith("MSG"):
        errors.append(f"invalid msg_id prefix: {msg.msg_id[:10]}")

    return ValidationResult(valid=len(errors) == 0, event_id=msg.msg_id, errors=errors)


def validate_auth_event(evt: AuthEvent) -> ValidationResult:
    errors: list[str] = []

    expected_crc = compute_auth_checksum(
        evt.event_id, evt.timestamp, evt.account_id,
        int(evt.action), evt.ip_address,
    )
    if evt.checksum != expected_crc:
        errors.append(f"checksum mismatch: got {evt.checksum}, expected {expected_crc}")

    if not evt.event_id.startswith("AUTH"):
        errors.append(f"invalid event_id prefix: {evt.event_id[:10]}")

    if evt.timestamp <= 0:
        errors.append("timestamp must be positive")

    return ValidationResult(valid=len(errors) == 0, event_id=evt.event_id, errors=errors)


# ── Dispatcher ────────────────────────────────────────────────────────────────

def validate_event(event: Event) -> ValidationResult:
    """Route any event type to its specific validator."""
    if isinstance(event, Transaction):
        return validate_transaction(event)
    if isinstance(event, InterbankMessage):
        return validate_interbank_message(event)
    if isinstance(event, AuthEvent):
        return validate_auth_event(event)
    return ValidationResult(valid=False, event_id="UNKNOWN", errors=["unrecognized event type"])
