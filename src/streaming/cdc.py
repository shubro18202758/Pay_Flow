"""
PayFlow — Change Data Capture (CDC) Core
==========================================
SQLite-based CDC using INSERT triggers + WAL mode for concurrent read/write.

Architecture:
    ┌──────────────────┐   INSERT trigger   ┌──────────────┐
    │ Source Tables     │ ────────────────→  │ _cdc_log     │
    │ • transactions   │   (same txn,       │ (append-only │
    │ • interbank_msgs │    atomic)         │  event log)  │
    │ • auth_events    │                    └──────┬───────┘
    └──────────────────┘                           │
                                                   │ async poll
                                            ┌──────▼───────┐
                                            │  CDCReader    │
                                            │ (watermark   │
                                            │  cursor)     │
                                            └──────────────┘

Why SQLite + WAL (no Kafka/Redis):
- Zero infrastructure: no brokers, no JVMs, no containers
- Atomic capture: trigger fires within the same INSERT transaction
- WAL mode: concurrent readers don't block the writer
- Already have aiosqlite as established dependency
"""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path
from typing import Union

import aiosqlite
import msgspec

from config.settings import StreamingConfig
from src.ingestion.schemas import (
    AuthEvent,
    Channel,
    AccountType,
    AuthAction,
    FraudPattern,
    InterbankMessage,
    Transaction,
)

logger = logging.getLogger(__name__)

Event = Union[Transaction, InterbankMessage, AuthEvent]

# ── JSON encoders/decoders (singleton, thread-safe) ─────────────────────────

_json_encoder = msgspec.json.Encoder()
_txn_decoder = msgspec.json.Decoder(Transaction)
_msg_decoder = msgspec.json.Decoder(InterbankMessage)
_auth_decoder = msgspec.json.Decoder(AuthEvent)

_DECODER_MAP: dict[str, msgspec.json.Decoder] = {
    "transactions": _txn_decoder,
    "interbank_messages": _msg_decoder,
    "auth_events": _auth_decoder,
}


# ── Banking Database with CDC Triggers ──────────────────────────────────────

class BankingDatabase:
    """
    Source database with INSERT triggers that atomically capture
    every write into a CDC log table.
    """

    def __init__(self, config: StreamingConfig) -> None:
        self._config = config
        self._db_path = config.db_path
        self._conn: aiosqlite.Connection | None = None

    async def initialize(self) -> None:
        """Create tables, triggers, enable WAL mode."""
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = await aiosqlite.connect(str(self._db_path))

        # WAL mode for concurrent read/write
        await self._conn.execute("PRAGMA journal_mode=WAL")
        await self._conn.execute("PRAGMA synchronous=NORMAL")

        # Source tables
        await self._conn.execute("""
            CREATE TABLE IF NOT EXISTS transactions (
                txn_id TEXT PRIMARY KEY,
                timestamp INTEGER,
                payload TEXT NOT NULL,
                captured_at REAL DEFAULT (unixepoch('subsec'))
            )
        """)
        await self._conn.execute("""
            CREATE TABLE IF NOT EXISTS interbank_messages (
                msg_id TEXT PRIMARY KEY,
                timestamp INTEGER,
                payload TEXT NOT NULL,
                captured_at REAL DEFAULT (unixepoch('subsec'))
            )
        """)
        await self._conn.execute("""
            CREATE TABLE IF NOT EXISTS auth_events (
                event_id TEXT PRIMARY KEY,
                timestamp INTEGER,
                payload TEXT NOT NULL,
                captured_at REAL DEFAULT (unixepoch('subsec'))
            )
        """)

        # CDC log (append-only event stream)
        await self._conn.execute("""
            CREATE TABLE IF NOT EXISTS _cdc_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                table_name TEXT NOT NULL,
                operation TEXT NOT NULL DEFAULT 'INSERT',
                payload TEXT NOT NULL,
                captured_at REAL DEFAULT (unixepoch('subsec'))
            )
        """)

        # CDC watermark (consumer cursor persistence)
        await self._conn.execute("""
            CREATE TABLE IF NOT EXISTS _cdc_watermark (
                consumer_id TEXT PRIMARY KEY,
                watermark_id INTEGER NOT NULL DEFAULT 0
            )
        """)

        # INSERT triggers on each source table
        for table in ("transactions", "interbank_messages", "auth_events"):
            await self._conn.execute(f"""
                CREATE TRIGGER IF NOT EXISTS _cdc_trigger_{table}
                AFTER INSERT ON {table}
                BEGIN
                    INSERT INTO _cdc_log (table_name, operation, payload)
                    VALUES ('{table}', 'INSERT', NEW.payload);
                END
            """)

        await self._conn.commit()
        logger.info("BankingDatabase initialized: %s (WAL mode)", self._db_path)

    async def insert_transaction(self, txn: Transaction) -> None:
        """Insert a transaction; CDC trigger captures it atomically."""
        assert self._conn is not None
        payload = _json_encoder.encode(txn).decode("utf-8")
        await self._conn.execute(
            "INSERT OR IGNORE INTO transactions (txn_id, timestamp, payload) VALUES (?, ?, ?)",
            (txn.txn_id, txn.timestamp, payload),
        )
        await self._conn.commit()

    async def insert_interbank(self, msg: InterbankMessage) -> None:
        """Insert an interbank message; CDC trigger captures it atomically."""
        assert self._conn is not None
        payload = _json_encoder.encode(msg).decode("utf-8")
        await self._conn.execute(
            "INSERT OR IGNORE INTO interbank_messages (msg_id, timestamp, payload) VALUES (?, ?, ?)",
            (msg.msg_id, msg.timestamp, payload),
        )
        await self._conn.commit()

    async def insert_auth_event(self, auth: AuthEvent) -> None:
        """Insert an auth event; CDC trigger captures it atomically."""
        assert self._conn is not None
        payload = _json_encoder.encode(auth).decode("utf-8")
        await self._conn.execute(
            "INSERT OR IGNORE INTO auth_events (event_id, timestamp, payload) VALUES (?, ?, ?)",
            (auth.event_id, auth.timestamp, payload),
        )
        await self._conn.commit()

    async def insert_batch(self, events: list[Event]) -> None:
        """Batch insert multiple events in a single transaction."""
        assert self._conn is not None
        for event in events:
            payload = _json_encoder.encode(event).decode("utf-8")
            if isinstance(event, Transaction):
                await self._conn.execute(
                    "INSERT OR IGNORE INTO transactions (txn_id, timestamp, payload) VALUES (?, ?, ?)",
                    (event.txn_id, event.timestamp, payload),
                )
            elif isinstance(event, InterbankMessage):
                await self._conn.execute(
                    "INSERT OR IGNORE INTO interbank_messages (msg_id, timestamp, payload) VALUES (?, ?, ?)",
                    (event.msg_id, event.timestamp, payload),
                )
            elif isinstance(event, AuthEvent):
                await self._conn.execute(
                    "INSERT OR IGNORE INTO auth_events (event_id, timestamp, payload) VALUES (?, ?, ?)",
                    (event.event_id, event.timestamp, payload),
                )
        await self._conn.commit()

    async def close(self) -> None:
        """Close the database connection."""
        if self._conn:
            await self._conn.close()
            self._conn = None


# ── CDC Reader (async tail with watermark cursor) ──────────────────────────

class CDCReader:
    """
    Async CDC log reader with adaptive polling and watermark-based
    crash recovery. Reads new entries from _cdc_log since the last
    acknowledged watermark.
    """

    def __init__(self, db_path: Path, config: StreamingConfig) -> None:
        self._db_path = db_path
        self._config = config
        self._conn: aiosqlite.Connection | None = None
        self._watermark: int = 0
        self._current_sleep_ms: float = float(config.cdc_poll_interval_ms)
        self._consumer_id = "default"
        self._running = False

    async def start(self) -> None:
        """Open read connection, recover watermark from last session."""
        self._conn = await aiosqlite.connect(str(self._db_path))
        await self._conn.execute("PRAGMA journal_mode=WAL")

        # Recover watermark
        async with self._conn.execute(
            "SELECT watermark_id FROM _cdc_watermark WHERE consumer_id = ?",
            (self._consumer_id,),
        ) as cursor:
            row = await cursor.fetchone()
            if row:
                self._watermark = row[0]

        self._running = True
        logger.info("CDCReader started: watermark=%d", self._watermark)

    async def stop(self) -> None:
        """Close the read connection."""
        self._running = False
        if self._conn:
            await self._conn.close()
            self._conn = None

    async def read_changes(self) -> list[tuple[int, Event]]:
        """
        Read new CDC entries since the last watermark.

        Advances the in-memory watermark so subsequent reads never return
        duplicates.  The persistent (on-disk) watermark is only updated
        when ``acknowledge()`` is called, preserving at-least-once
        semantics on crash recovery.

        Returns:
            List of (cdc_id, Event) tuples. Empty list if no new changes.
        """
        assert self._conn is not None
        results: list[tuple[int, Event]] = []

        async with self._conn.execute(
            "SELECT id, table_name, payload FROM _cdc_log WHERE id > ? ORDER BY id LIMIT ?",
            (self._watermark, self._config.cdc_batch_size),
        ) as cursor:
            rows = await cursor.fetchall()

        for row_id, table_name, payload_str in rows:
            decoder = _DECODER_MAP.get(table_name)
            if decoder is None:
                logger.warning("Unknown CDC table_name: %s (id=%d)", table_name, row_id)
                continue
            try:
                event = decoder.decode(payload_str.encode("utf-8"))
                results.append((row_id, event))
            except Exception as exc:
                logger.error("CDC decode error (id=%d, table=%s): %s", row_id, table_name, exc)

        # Advance in-memory watermark to prevent duplicate reads on
        # subsequent polls before acknowledge() is called.
        if results:
            self._watermark = results[-1][0]

        return results

    async def acknowledge(self, up_to_id: int) -> None:
        """
        Acknowledge consumption up to the given CDC log id.
        Persists watermark and prunes old log entries.
        """
        assert self._conn is not None
        self._watermark = up_to_id

        # Upsert watermark
        await self._conn.execute(
            """INSERT INTO _cdc_watermark (consumer_id, watermark_id) VALUES (?, ?)
               ON CONFLICT(consumer_id) DO UPDATE SET watermark_id = excluded.watermark_id""",
            (self._consumer_id, up_to_id),
        )

        # Prune old entries beyond retention threshold
        if self._config.cdc_log_retention > 0:
            await self._conn.execute(
                "DELETE FROM _cdc_log WHERE id <= ?",
                (up_to_id - self._config.cdc_log_retention,),
            )

        await self._conn.commit()

    def adaptive_sleep(self, had_results: bool) -> float:
        """
        Calculate adaptive sleep interval.

        When busy (had_results=True): reset to base poll interval (10ms).
        When idle: exponentially back off up to the ceiling (100ms).

        Returns the sleep duration in seconds.
        """
        base = float(self._config.cdc_poll_interval_ms)
        ceiling = float(self._config.cdc_max_poll_interval_ms)

        if had_results:
            self._current_sleep_ms = base
        else:
            self._current_sleep_ms = min(self._current_sleep_ms * 1.5, ceiling)

        return self._current_sleep_ms / 1000.0

    @property
    def watermark(self) -> int:
        return self._watermark
