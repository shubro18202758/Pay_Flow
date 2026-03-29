"""
PayFlow -- Ledger Cryptographic Primitives
============================================
SHA-256 block hashing, Ed25519 signing, and Merkle tree checkpoints.

Key Management:
    - Auto-generates Ed25519 keypair on first run
    - Persists raw key bytes to disk (32 bytes each)
    - Loads existing key on subsequent runs

Hash Strategy:
    Canonical JSON (sorted keys, compact separators, ensure_ascii=True)
    guarantees identical SHA-256 output across Python versions and platforms.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


# ── SHA-256 Block Hashing ────────────────────────────────────────────────────

class BlockHasher:
    """Deterministic SHA-256 hasher for block content."""

    @staticmethod
    def compute_hash(
        index: int,
        timestamp: float,
        event_type: int,
        payload: dict,
        previous_hash: str,
    ) -> str:
        """
        Compute the SHA-256 hash of a block's content fields.

        Returns:
            64-character lowercase hex digest.
        """
        canonical = json.dumps(
            {
                "index": index,
                "timestamp": timestamp,
                "event_type": event_type,
                "payload": payload,
                "previous_hash": previous_hash,
            },
            sort_keys=True,
            ensure_ascii=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# ── Ed25519 Digital Signatures ───────────────────────────────────────────────

class BlockSigner:
    """
    Ed25519 digital signatures for non-repudiation.

    Lazy-imports nacl to avoid startup cost when signing is disabled.
    """

    def __init__(self, key_dir: Path) -> None:
        self._key_dir = key_dir
        self._signing_key = None   # nacl.signing.SigningKey (lazy)
        self._verify_key = None    # nacl.signing.VerifyKey (lazy)

    def initialize(self) -> None:
        """Load or generate the Ed25519 keypair."""
        from nacl.signing import SigningKey

        sk_path = self._key_dir / "signing_key.bin"
        pk_path = self._key_dir / "signing_key.pub"

        if sk_path.exists():
            raw = sk_path.read_bytes()
            if len(raw) != 32:
                raise ValueError(
                    f"Corrupt signing key at {sk_path}: expected 32 bytes, got {len(raw)}"
                )
            self._signing_key = SigningKey(raw)
            self._verify_key = self._signing_key.verify_key
            logger.info("Ed25519 signing key loaded from %s", sk_path)
        else:
            self._key_dir.mkdir(parents=True, exist_ok=True)
            self._signing_key = SigningKey.generate()
            sk_path.write_bytes(bytes(self._signing_key))
            pk_path.write_bytes(bytes(self._signing_key.verify_key))
            self._verify_key = self._signing_key.verify_key
            logger.info("Ed25519 keypair generated and saved to %s", self._key_dir)

    def sign(self, block_hash: str) -> str:
        """
        Sign a block hash and return the hex-encoded signature.

        Args:
            block_hash: 64-char hex SHA-256 digest.

        Returns:
            Hex-encoded Ed25519 signature (128 hex chars = 64 bytes).
        """
        if self._signing_key is None:
            raise RuntimeError("Signer not initialized. Call initialize() first.")
        signed = self._signing_key.sign(block_hash.encode("utf-8"))
        return signed.signature.hex()

    def verify(self, block_hash: str, signature_hex: str) -> bool:
        """
        Verify an Ed25519 signature against a block hash.

        Returns:
            True if valid, False if tampered.
        """
        from nacl.exceptions import BadSignatureError

        if self._verify_key is None:
            raise RuntimeError("Signer not initialized. Call initialize() first.")
        try:
            sig_bytes = bytes.fromhex(signature_hex)
            self._verify_key.verify(block_hash.encode("utf-8"), sig_bytes)
            return True
        except (BadSignatureError, ValueError):
            return False

    @staticmethod
    def null_signature() -> str:
        """Placeholder signature when signing is disabled."""
        return "0" * 128


# ── Merkle Tree Checkpoints ─────────────────────────────────────────────────

class MerkleCheckpointer:
    """
    Periodic Merkle tree checkpoint builder using pymerkle.

    Collects block hashes between checkpoints and computes an
    RFC-6962-compliant Merkle root at checkpoint boundaries.
    """

    def __init__(self) -> None:
        self._pending_hashes: list[bytes] = []

    def add_hash(self, block_hash: str) -> None:
        """Buffer a block hash for the next checkpoint."""
        self._pending_hashes.append(block_hash.encode("utf-8"))

    def compute_checkpoint(self) -> str | None:
        """
        Build a Merkle tree from buffered hashes and return the hex root.

        Returns None if no hashes have been buffered.
        Clears the internal buffer after computing.
        """
        if not self._pending_hashes:
            return None

        from pymerkle import InmemoryTree

        tree = InmemoryTree()
        for h in self._pending_hashes:
            tree.append_entry(h)

        root = tree.get_state().hex()
        self._pending_hashes.clear()
        return root

    @property
    def pending_count(self) -> int:
        return len(self._pending_hashes)
