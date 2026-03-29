"""
PayFlow -- Async SQLite Ledger Storage
=========================================
Append-only block storage with WAL mode for concurrent reads.

Single-writer design (matches project convention):
    - One AuditLedger instance holds the single aiosqlite connection.
    - Reads are safe to perform concurrently from the async event loop.
    - No connection pooling needed.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS blocks (
    idx         INTEGER PRIMARY KEY,
    timestamp   REAL    NOT NULL,
    event_type  INTEGER NOT NULL,
    payload     TEXT    NOT NULL,
    prev_hash   TEXT    NOT NULL,
    block_hash  TEXT    NOT NULL UNIQUE,
    signature   TEXT    NOT NULL,
    merkle_root TEXT
);
"""

_CREATE_INDEXES_SQL = [
    "CREATE INDEX IF NOT EXISTS ix_blocks_event_type ON blocks(event_type);",
    "CREATE INDEX IF NOT EXISTS ix_blocks_timestamp ON blocks(timestamp);",
    "CREATE INDEX IF NOT EXISTS ix_blocks_hash ON blocks(block_hash);",
]

_PRAGMA_SQL = [
    "PRAGMA journal_mode = WAL;",
    "PRAGMA synchronous = NORMAL;",
    "PRAGMA cache_size = -2000;",
    "PRAGMA busy_timeout = 5000;",
]

_INSERT_BLOCK_SQL = """
INSERT INTO blocks (idx, timestamp, event_type, payload, prev_hash, block_hash, signature, merkle_root)
VALUES (?, ?, ?, ?, ?, ?, ?, ?);
"""


class LedgerStorage:
    """
    Async SQLite storage engine for the audit ledger.

    Lifecycle::

        storage = LedgerStorage(db_path)
        await storage.open()
        ...
        await storage.close()

    Or via async context manager::

        async with LedgerStorage(db_path) as storage:
            ...
    """

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        self._conn = None  # aiosqlite.Connection (lazy)

    async def open(self) -> None:
        """Open the database connection, create schema if needed."""
        import aiosqlite

        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = await aiosqlite.connect(str(self._db_path))

        for pragma in _PRAGMA_SQL:
            await self._conn.execute(pragma)

        await self._conn.execute(_CREATE_TABLE_SQL)
        for idx_sql in _CREATE_INDEXES_SQL:
            await self._conn.execute(idx_sql)
        await self._conn.commit()
        logger.info("Ledger storage opened: %s", self._db_path)

    async def close(self) -> None:
        """Flush and close the database connection."""
        if self._conn is not None:
            await self._conn.close()
            self._conn = None
            logger.info("Ledger storage closed.")

    async def __aenter__(self):
        await self.open()
        return self

    async def __aexit__(self, *exc):
        await self.close()

    # ── Writes ───────────────────────────────────────────────────────────

    async def insert_block(self, row: tuple) -> None:
        """Insert a single block row. Caller provides tuple from Block.to_row()."""
        await self._conn.execute(_INSERT_BLOCK_SQL, row)
        await self._conn.commit()

    # ── Reads ────────────────────────────────────────────────────────────

    async def get_head(self) -> tuple | None:
        """Return the latest block row, or None if the chain is empty."""
        cursor = await self._conn.execute(
            "SELECT * FROM blocks ORDER BY idx DESC LIMIT 1;"
        )
        return await cursor.fetchone()

    async def get_block(self, idx: int) -> tuple | None:
        """Fetch a single block by index."""
        cursor = await self._conn.execute(
            "SELECT * FROM blocks WHERE idx = ?;", (idx,)
        )
        return await cursor.fetchone()

    async def get_block_range(self, start: int, end: int) -> list[tuple]:
        """
        Fetch blocks in [start, end] inclusive, ordered by index.
        Used for chain verification and Merkle proof reconstruction.
        """
        cursor = await self._conn.execute(
            "SELECT * FROM blocks WHERE idx >= ? AND idx <= ? ORDER BY idx;",
            (start, end),
        )
        return await cursor.fetchall()

    async def get_block_count(self) -> int:
        """Total number of blocks in the chain."""
        cursor = await self._conn.execute("SELECT COUNT(*) FROM blocks;")
        row = await cursor.fetchone()
        return row[0] if row else 0

    async def get_blocks_by_event_type(
        self, event_type: int, limit: int = 100,
    ) -> list[tuple]:
        """Query blocks by event type, most recent first."""
        cursor = await self._conn.execute(
            "SELECT * FROM blocks WHERE event_type = ? ORDER BY idx DESC LIMIT ?;",
            (event_type, limit),
        )
        return await cursor.fetchall()

    async def get_checkpoints(self) -> list[tuple]:
        """Return all blocks that have a non-NULL merkle_root."""
        cursor = await self._conn.execute(
            "SELECT * FROM blocks WHERE merkle_root IS NOT NULL ORDER BY idx;"
        )
        return await cursor.fetchall()

    async def get_db_size_bytes(self) -> int:
        """Return the SQLite file size on disk."""
        if self._db_path.exists():
            return self._db_path.stat().st_size
        return 0

    async def execute_raw(self, sql: str, params: tuple = ()) -> None:
        """Execute raw SQL (for testing / tamper simulation only)."""
        await self._conn.execute(sql, params)
        await self._conn.commit()
