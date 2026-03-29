"""
PayFlow -- Phase 11 Tests: Real-Time CDC Event Streaming Pipeline
===================================================================
Verifies BankingDatabase CDC triggers, CDCReader watermark recovery,
StreamingConsumer batching and fan-out, BankingEndpointSimulator event
generation, and end-to-end pipeline integration.

Tests:
 1. StreamingConfig defaults validation
 2. StreamingConfig frozen immutability
 3. BankingDatabase initialization (tables, triggers, WAL)
 4. BankingDatabase insert_transaction triggers CDC log
 5. BankingDatabase insert_interbank triggers CDC log
 6. BankingDatabase insert_auth_event triggers CDC log
 7. BankingDatabase batch insert multiple event types
 8. CDCReader reads changes from CDC log
 9. CDCReader watermark recovery after restart
10. CDCReader deserialization correctness
11. CDCReader adaptive sleep (busy vs idle)
12. CDCReader acknowledge prunes old entries
13. StreamingConsumer fan-out to multiple consumers
14. StreamingConsumer batching by size threshold
15. StreamingConsumer batching by timeout
16. StreamingConsumer graceful shutdown
17. StreamingConsumer error isolation
18. StreamingConsumer metrics tracking
19. BankingEndpointSimulator generates events
20. BankingEndpointSimulator fraud injection
21. BankingEndpointSimulator TPS control
22. End-to-end CDC pipeline integration
23. Streaming metrics serialization
24. Consumer protocol compatibility with IngestionPipeline
"""

from __future__ import annotations

import asyncio
import sys
import tempfile
import time
from dataclasses import FrozenInstanceError
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import STREAMING_CFG, StreamingConfig
from src.ingestion.schemas import (
    AccountType,
    AuthAction,
    AuthEvent,
    Channel,
    EventBatch,
    FraudPattern,
    InterbankMessage,
    Transaction,
)
from src.ingestion.validators import (
    compute_auth_checksum,
    compute_interbank_checksum,
    compute_transaction_checksum,
)
from src.streaming.cdc import BankingDatabase, CDCReader
from src.streaming.consumer import StreamingConsumer, StreamingMetrics
from src.streaming.endpoints import BankingEndpointSimulator


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


def _make_test_config(tmp_dir: Path) -> StreamingConfig:
    """Create a StreamingConfig pointed at a temp directory."""
    return StreamingConfig(
        db_path=tmp_dir / "test_banking.db",
        cdc_poll_interval_ms=5,
        cdc_max_poll_interval_ms=50,
        cdc_batch_size=10,
        cdc_batch_timeout_sec=0.2,
        cdc_log_retention=100,
        endpoint_tps=100.0,
        endpoint_fraud_ratio=0.05,
        consumer_concurrency=3,
        enable_validation=True,
    )


def _make_transaction(txn_id: str = "TXN-aabbccdd0011", timestamp: int = 1700000000) -> Transaction:
    """Create a valid test transaction."""
    cs = compute_transaction_checksum(
        txn_id, timestamp, "00420000001234", "00420000005678",
        500000, Channel.UPI,
    )
    return Transaction(
        txn_id=txn_id,
        timestamp=timestamp,
        sender_id="00420000001234",
        receiver_id="00420000005678",
        amount_paisa=500000,
        channel=Channel.UPI,
        sender_branch="0042",
        receiver_branch="0042",
        sender_geo_lat=19.076,
        sender_geo_lon=72.877,
        receiver_geo_lat=28.613,
        receiver_geo_lon=77.209,
        device_fingerprint="a1b2c3d4e5f6a7b8",
        sender_account_type=AccountType.SAVINGS,
        receiver_account_type=AccountType.SAVINGS,
        checksum=cs,
        fraud_label=FraudPattern.NONE,
    )


def _make_interbank_msg(msg_id: str = "MSG-112233445566", timestamp: int = 1700000001) -> InterbankMessage:
    """Create a valid test interbank message."""
    cs = compute_interbank_checksum(
        msg_id, timestamp, "UBIN0530042", "SBIN0001234",
        10000000, Channel.RTGS,
    )
    return InterbankMessage(
        msg_id=msg_id,
        timestamp=timestamp,
        sender_ifsc="UBIN0530042",
        receiver_ifsc="SBIN0001234",
        sender_account="00420000001234",
        receiver_account="00010000005678",
        amount_paisa=10000000,
        currency_code=356,
        channel=Channel.RTGS,
        message_type="MT103",
        sender_geo_lat=19.076,
        sender_geo_lon=72.877,
        device_fingerprint="b2c3d4e5f6a7b8c9",
        priority=1,
        checksum=cs,
    )


def _make_auth_event(event_id: str = "AUTH-aabbccdd0022", timestamp: int = 1700000002) -> AuthEvent:
    """Create a valid test auth event."""
    cs = compute_auth_checksum(
        event_id, timestamp, "00420000001234",
        AuthAction.LOGIN, "192.168.1.1",
    )
    return AuthEvent(
        event_id=event_id,
        timestamp=timestamp,
        account_id="00420000001234",
        action=AuthAction.LOGIN,
        ip_address="192.168.1.1",
        geo_lat=19.076,
        geo_lon=72.877,
        device_fingerprint="c3d4e5f6a7b8c9d0",
        user_agent_hash="d4e5f6a7b8c9d0e1",
        success=True,
        checksum=cs,
    )


# ── Test 1: StreamingConfig defaults ──────────────────────────────────────────

def test_streaming_config_defaults():
    cfg = StreamingConfig()
    assert cfg.cdc_poll_interval_ms == 10, f"Expected 10, got {cfg.cdc_poll_interval_ms}"
    assert cfg.cdc_max_poll_interval_ms == 100, f"Expected 100, got {cfg.cdc_max_poll_interval_ms}"
    assert cfg.cdc_batch_size == 256, f"Expected 256, got {cfg.cdc_batch_size}"
    assert cfg.cdc_batch_timeout_sec == 0.5, f"Expected 0.5, got {cfg.cdc_batch_timeout_sec}"
    assert cfg.cdc_log_retention == 10_000, f"Expected 10000, got {cfg.cdc_log_retention}"
    assert cfg.endpoint_tps == 50.0, f"Expected 50.0, got {cfg.endpoint_tps}"
    assert cfg.endpoint_burst_multiplier == 3.0
    assert cfg.endpoint_fraud_ratio == 0.05
    assert cfg.consumer_concurrency == 3
    assert cfg.enable_validation is True
    # Singleton export
    assert STREAMING_CFG is not None
    assert STREAMING_CFG.cdc_poll_interval_ms == 10


# ── Test 2: StreamingConfig frozen immutability ─────────────────────────────

def test_streaming_config_immutability():
    cfg = StreamingConfig()
    try:
        cfg.cdc_batch_size = 999
        raise AssertionError("Expected FrozenInstanceError")
    except FrozenInstanceError:
        pass


# ── Test 3: BankingDatabase initialization ──────────────────────────────────

async def test_banking_db_initialization():
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _make_test_config(Path(tmp))
        db = BankingDatabase(cfg)
        await db.initialize()

        # Check tables exist
        import aiosqlite
        async with aiosqlite.connect(str(cfg.db_path)) as conn:
            async with conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
            ) as cursor:
                tables = [row[0] for row in await cursor.fetchall()]

        assert "_cdc_log" in tables, f"_cdc_log missing from {tables}"
        assert "_cdc_watermark" in tables, f"_cdc_watermark missing from {tables}"
        assert "transactions" in tables, f"transactions missing from {tables}"
        assert "interbank_messages" in tables, f"interbank_messages missing from {tables}"
        assert "auth_events" in tables, f"auth_events missing from {tables}"

        # Check triggers exist
        async with aiosqlite.connect(str(cfg.db_path)) as conn:
            async with conn.execute(
                "SELECT name FROM sqlite_master WHERE type='trigger'"
            ) as cursor:
                triggers = [row[0] for row in await cursor.fetchall()]

        assert len(triggers) == 3, f"Expected 3 triggers, found {len(triggers)}: {triggers}"
        await db.close()


# ── Test 4: insert_transaction triggers CDC ─────────────────────────────────

async def test_banking_db_insert_transaction():
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _make_test_config(Path(tmp))
        db = BankingDatabase(cfg)
        await db.initialize()

        txn = _make_transaction()
        await db.insert_transaction(txn)

        # Verify CDC log entry
        import aiosqlite
        async with aiosqlite.connect(str(cfg.db_path)) as conn:
            async with conn.execute("SELECT COUNT(*) FROM _cdc_log") as cursor:
                count = (await cursor.fetchone())[0]

        assert count == 1, f"Expected 1 CDC entry, got {count}"

        async with aiosqlite.connect(str(cfg.db_path)) as conn:
            async with conn.execute("SELECT table_name, operation FROM _cdc_log") as cursor:
                row = await cursor.fetchone()

        assert row[0] == "transactions", f"Expected table_name='transactions', got {row[0]}"
        assert row[1] == "INSERT", f"Expected operation='INSERT', got {row[1]}"
        await db.close()


# ── Test 5: insert_interbank triggers CDC ───────────────────────────────────

async def test_banking_db_insert_interbank():
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _make_test_config(Path(tmp))
        db = BankingDatabase(cfg)
        await db.initialize()

        msg = _make_interbank_msg()
        await db.insert_interbank(msg)

        import aiosqlite
        async with aiosqlite.connect(str(cfg.db_path)) as conn:
            async with conn.execute(
                "SELECT table_name FROM _cdc_log WHERE table_name='interbank_messages'"
            ) as cursor:
                row = await cursor.fetchone()

        assert row is not None, "No CDC entry for interbank_messages"
        assert row[0] == "interbank_messages"
        await db.close()


# ── Test 6: insert_auth_event triggers CDC ──────────────────────────────────

async def test_banking_db_insert_auth_event():
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _make_test_config(Path(tmp))
        db = BankingDatabase(cfg)
        await db.initialize()

        auth = _make_auth_event()
        await db.insert_auth_event(auth)

        import aiosqlite
        async with aiosqlite.connect(str(cfg.db_path)) as conn:
            async with conn.execute(
                "SELECT table_name FROM _cdc_log WHERE table_name='auth_events'"
            ) as cursor:
                row = await cursor.fetchone()

        assert row is not None, "No CDC entry for auth_events"
        assert row[0] == "auth_events"
        await db.close()


# ── Test 7: batch insert multiple event types ──────────────────────────────

async def test_banking_db_batch_insert():
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _make_test_config(Path(tmp))
        db = BankingDatabase(cfg)
        await db.initialize()

        events = [
            _make_transaction("TXN-batch0000001"),
            _make_transaction("TXN-batch0000002"),
            _make_interbank_msg("MSG-batch0000001"),
            _make_auth_event("AUTH-batch000001"),
        ]
        await db.insert_batch(events)

        import aiosqlite
        async with aiosqlite.connect(str(cfg.db_path)) as conn:
            async with conn.execute("SELECT COUNT(*) FROM _cdc_log") as cursor:
                count = (await cursor.fetchone())[0]

        assert count == 4, f"Expected 4 CDC entries, got {count}"
        await db.close()


# ── Test 8: CDCReader reads changes ────────────────────────────────────────

async def test_cdc_reader_reads_changes():
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _make_test_config(Path(tmp))
        db = BankingDatabase(cfg)
        await db.initialize()

        # Insert 3 events
        await db.insert_transaction(_make_transaction("TXN-read00000001"))
        await db.insert_interbank(_make_interbank_msg("MSG-read00000001"))
        await db.insert_auth_event(_make_auth_event("AUTH-read0000001"))

        reader = CDCReader(cfg.db_path, cfg)
        await reader.start()

        changes = await reader.read_changes()
        assert len(changes) == 3, f"Expected 3 changes, got {len(changes)}"

        # Verify (cdc_id, Event) tuple structure
        for cdc_id, event in changes:
            assert isinstance(cdc_id, int), f"cdc_id should be int, got {type(cdc_id)}"
            assert isinstance(event, (Transaction, InterbankMessage, AuthEvent))

        await reader.stop()
        await db.close()


# ── Test 9: CDCReader watermark recovery ───────────────────────────────────

async def test_cdc_reader_watermark_recovery():
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _make_test_config(Path(tmp))
        db = BankingDatabase(cfg)
        await db.initialize()

        # Insert 3 events
        await db.insert_transaction(_make_transaction("TXN-wm0000000001"))
        await db.insert_transaction(_make_transaction("TXN-wm0000000002"))
        await db.insert_transaction(_make_transaction("TXN-wm0000000003"))

        # First reader: read and acknowledge
        reader1 = CDCReader(cfg.db_path, cfg)
        await reader1.start()
        changes = await reader1.read_changes()
        assert len(changes) == 3
        max_id = changes[-1][0]
        await reader1.acknowledge(max_id)
        await reader1.stop()

        # Insert 2 more events
        await db.insert_transaction(_make_transaction("TXN-wm0000000004"))
        await db.insert_transaction(_make_transaction("TXN-wm0000000005"))

        # Second reader: should only see the 2 new events
        reader2 = CDCReader(cfg.db_path, cfg)
        await reader2.start()
        assert reader2.watermark == max_id, f"Watermark not recovered: {reader2.watermark} != {max_id}"
        changes2 = await reader2.read_changes()
        assert len(changes2) == 2, f"Expected 2 new changes, got {len(changes2)}"
        await reader2.stop()
        await db.close()


# ── Test 10: CDCReader deserialization correctness ─────────────────────────

async def test_cdc_reader_deserialization():
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _make_test_config(Path(tmp))
        db = BankingDatabase(cfg)
        await db.initialize()

        original_txn = _make_transaction("TXN-deser0000001")
        await db.insert_transaction(original_txn)

        reader = CDCReader(cfg.db_path, cfg)
        await reader.start()
        changes = await reader.read_changes()
        assert len(changes) == 1

        _, decoded_txn = changes[0]
        assert isinstance(decoded_txn, Transaction), f"Expected Transaction, got {type(decoded_txn)}"
        assert decoded_txn.txn_id == original_txn.txn_id
        assert decoded_txn.amount_paisa == original_txn.amount_paisa
        assert decoded_txn.sender_id == original_txn.sender_id
        assert decoded_txn.checksum == original_txn.checksum

        await reader.stop()
        await db.close()


# ── Test 11: CDCReader adaptive sleep ──────────────────────────────────────

def test_cdc_reader_adaptive_sleep():
    cfg = StreamingConfig(
        db_path=Path("/tmp/unused.db"),
        cdc_poll_interval_ms=10,
        cdc_max_poll_interval_ms=100,
    )
    reader = CDCReader(cfg.db_path, cfg)

    # Busy: should return base interval
    sleep1 = reader.adaptive_sleep(had_results=True)
    assert abs(sleep1 - 0.010) < 0.001, f"Busy sleep should be ~10ms, got {sleep1}"

    # Idle: should increase
    sleep2 = reader.adaptive_sleep(had_results=False)
    assert sleep2 > sleep1, f"Idle sleep should increase: {sleep2} <= {sleep1}"

    # Multiple idle cycles: should approach ceiling
    for _ in range(20):
        sleep_val = reader.adaptive_sleep(had_results=False)
    assert abs(sleep_val - 0.100) < 0.001, f"Should hit ceiling ~100ms, got {sleep_val}"

    # Back to busy: should reset
    sleep_reset = reader.adaptive_sleep(had_results=True)
    assert abs(sleep_reset - 0.010) < 0.001, f"Reset to base ~10ms, got {sleep_reset}"


# ── Test 12: CDCReader acknowledge and cleanup ─────────────────────────────

async def test_cdc_reader_acknowledge_cleanup():
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _make_test_config(Path(tmp))
        db = BankingDatabase(cfg)
        await db.initialize()

        # Insert 5 events
        for i in range(5):
            await db.insert_transaction(_make_transaction(f"TXN-ack{i:010d}"))

        reader = CDCReader(cfg.db_path, cfg)
        await reader.start()
        changes = await reader.read_changes()
        assert len(changes) == 5

        # Acknowledge up to id 3
        await reader.acknowledge(changes[2][0])

        # Read again: should see only events after id 3
        changes2 = await reader.read_changes()
        assert len(changes2) == 2, f"Expected 2 remaining, got {len(changes2)}"

        await reader.stop()
        await db.close()


# ── Test 13: StreamingConsumer fan-out ─────────────────────────────────────

async def test_consumer_fan_out():
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _make_test_config(Path(tmp))
        cfg = StreamingConfig(
            db_path=Path(tmp) / "fanout.db",
            cdc_poll_interval_ms=5,
            cdc_max_poll_interval_ms=20,
            cdc_batch_size=5,
            cdc_batch_timeout_sec=0.1,
            cdc_log_retention=100,
        )
        db = BankingDatabase(cfg)
        await db.initialize()

        reader = CDCReader(cfg.db_path, cfg)
        await reader.start()

        consumer = StreamingConsumer(reader, cfg)

        received_batches_1 = []
        received_batches_2 = []

        async def consumer1(batch: EventBatch) -> None:
            received_batches_1.append(batch)

        async def consumer2(batch: EventBatch) -> None:
            received_batches_2.append(batch)

        consumer.add_consumer(consumer1)
        consumer.add_consumer(consumer2)
        await consumer.start()

        # Insert events to trigger a batch
        for i in range(5):
            await db.insert_transaction(_make_transaction(f"TXN-fan{i:010d}"))

        # Wait for consumer to process
        await asyncio.sleep(0.5)
        await consumer.stop()

        assert len(received_batches_1) >= 1, f"Consumer1 got {len(received_batches_1)} batches"
        assert len(received_batches_2) >= 1, f"Consumer2 got {len(received_batches_2)} batches"
        # Both should receive the same number of batches
        assert len(received_batches_1) == len(received_batches_2)

        await reader.stop()
        await db.close()


# ── Test 14: StreamingConsumer batching by size ────────────────────────────

async def test_consumer_batching_by_size():
    tmp = tempfile.mkdtemp()
    try:
        cfg = StreamingConfig(
            db_path=Path(tmp) / "batchsize.db",
            cdc_poll_interval_ms=5,
            cdc_max_poll_interval_ms=20,
            cdc_batch_size=3,
            cdc_batch_timeout_sec=5.0,  # High timeout so size triggers first
            cdc_log_retention=100,
        )
        db = BankingDatabase(cfg)
        await db.initialize()

        reader = CDCReader(cfg.db_path, cfg)
        await reader.start()

        consumer = StreamingConsumer(reader, cfg)
        received = []

        async def collector(batch: EventBatch) -> None:
            received.append(batch)

        consumer.add_consumer(collector)
        await consumer.start()

        # Insert exactly batch_size events
        for i in range(3):
            await db.insert_transaction(_make_transaction(f"TXN-sz{i:011d}"))

        await asyncio.sleep(0.3)
        await consumer.stop()

        assert len(received) >= 1, f"Expected at least 1 batch, got {len(received)}"
        total_events = sum(b.event_count for b in received)
        assert total_events == 3, f"Expected 3 total events, got {total_events}"

        await reader.stop()
        await db.close()
    finally:
        import shutil
        await asyncio.sleep(0.05)
        shutil.rmtree(tmp, ignore_errors=True)


# ── Test 15: StreamingConsumer batching by timeout ─────────────────────────

async def test_consumer_batching_by_timeout():
    tmp = tempfile.mkdtemp()
    try:
        cfg = StreamingConfig(
            db_path=Path(tmp) / "batchtimeout.db",
            cdc_poll_interval_ms=5,
            cdc_max_poll_interval_ms=20,
            cdc_batch_size=1000,  # Very large, so timeout triggers
            cdc_batch_timeout_sec=0.15,
            cdc_log_retention=100,
        )
        db = BankingDatabase(cfg)
        await db.initialize()

        reader = CDCReader(cfg.db_path, cfg)
        await reader.start()

        consumer = StreamingConsumer(reader, cfg)
        received = []

        async def collector(batch: EventBatch) -> None:
            received.append(batch)

        consumer.add_consumer(collector)
        await consumer.start()

        # Insert 2 events (well under batch_size=1000)
        await db.insert_transaction(_make_transaction("TXN-to0000000001"))
        await db.insert_transaction(_make_transaction("TXN-to0000000002"))

        # Wait longer than timeout
        await asyncio.sleep(0.6)
        await consumer.stop()

        assert len(received) >= 1, f"Timeout should have triggered a flush, got {len(received)} batches"
        total_events = sum(b.event_count for b in received)
        assert total_events == 2, f"Expected 2 events, got {total_events}"

        await reader.stop()
        await db.close()
    finally:
        import shutil
        await asyncio.sleep(0.05)
        shutil.rmtree(tmp, ignore_errors=True)


# ── Test 16: StreamingConsumer graceful shutdown ───────────────────────────

async def test_consumer_graceful_shutdown():
    tmp = tempfile.mkdtemp()
    try:
        cfg = _make_test_config(Path(tmp))
        db = BankingDatabase(cfg)
        await db.initialize()

        reader = CDCReader(cfg.db_path, cfg)
        await reader.start()

        consumer = StreamingConsumer(reader, cfg)
        received = []

        async def collector(batch: EventBatch) -> None:
            received.append(batch)

        consumer.add_consumer(collector)
        await consumer.start()

        # Insert an event
        await db.insert_transaction(_make_transaction("TXN-shut000000001"))
        await asyncio.sleep(0.3)

        # Graceful shutdown should flush remaining
        await consumer.stop()

        total_events = sum(b.event_count for b in received)
        assert total_events == 1, f"Expected 1 event flushed on shutdown, got {total_events}"

        await reader.stop()
        await db.close()
    finally:
        import shutil
        await asyncio.sleep(0.05)
        shutil.rmtree(tmp, ignore_errors=True)


# ── Test 17: StreamingConsumer error isolation ─────────────────────────────

async def test_consumer_error_isolation():
    with tempfile.TemporaryDirectory() as tmp:
        cfg = StreamingConfig(
            db_path=Path(tmp) / "errisolation.db",
            cdc_poll_interval_ms=5,
            cdc_max_poll_interval_ms=20,
            cdc_batch_size=2,
            cdc_batch_timeout_sec=0.1,
            cdc_log_retention=100,
        )
        db = BankingDatabase(cfg)
        await db.initialize()

        reader = CDCReader(cfg.db_path, cfg)
        await reader.start()

        consumer = StreamingConsumer(reader, cfg)
        good_batches = []

        async def failing_consumer(batch: EventBatch) -> None:
            raise RuntimeError("Simulated consumer failure")

        async def good_consumer(batch: EventBatch) -> None:
            good_batches.append(batch)

        consumer.add_consumer(failing_consumer)
        consumer.add_consumer(good_consumer)
        await consumer.start()

        await db.insert_transaction(_make_transaction("TXN-err0000000001"))
        await db.insert_transaction(_make_transaction("TXN-err0000000002"))

        await asyncio.sleep(0.5)
        await consumer.stop()

        # Good consumer should still receive batches despite failing consumer
        assert len(good_batches) >= 1, "Good consumer should receive batches even if another fails"
        assert consumer.metrics.consumer_errors >= 1, "Should have recorded consumer errors"

        await reader.stop()
        await db.close()


# ── Test 18: StreamingConsumer metrics tracking ────────────────────────────

async def test_consumer_metrics():
    tmp = tempfile.mkdtemp()
    try:
        cfg = StreamingConfig(
            db_path=Path(tmp) / "metrics.db",
            cdc_poll_interval_ms=5,
            cdc_max_poll_interval_ms=20,
            cdc_batch_size=5,
            cdc_batch_timeout_sec=0.1,
            cdc_log_retention=100,
        )
        db = BankingDatabase(cfg)
        await db.initialize()

        reader = CDCReader(cfg.db_path, cfg)
        await reader.start()

        consumer = StreamingConsumer(reader, cfg)

        async def noop_consumer(batch: EventBatch) -> None:
            pass

        consumer.add_consumer(noop_consumer)
        await consumer.start()

        for i in range(5):
            await db.insert_transaction(_make_transaction(f"TXN-met{i:010d}"))

        await asyncio.sleep(0.5)
        await consumer.stop()

        m = consumer.metrics
        assert m.events_consumed == 5, f"Expected 5 consumed, got {m.events_consumed}"
        assert m.batches_dispatched >= 1, f"Expected >=1 batch, got {m.batches_dispatched}"
        assert m.uptime_sec > 0, "Uptime should be positive"
        assert m.throughput_eps > 0, "Throughput should be positive"

        await reader.stop()
        await db.close()
    finally:
        import shutil
        await asyncio.sleep(0.05)
        shutil.rmtree(tmp, ignore_errors=True)


# ── Test 19: BankingEndpointSimulator generates events ────────────────────

async def test_endpoint_generates_events():
    with tempfile.TemporaryDirectory() as tmp:
        cfg = StreamingConfig(
            db_path=Path(tmp) / "endpoint.db",
            endpoint_tps=200.0,  # Fast for testing
            endpoint_fraud_ratio=0.05,
            cdc_poll_interval_ms=5,
            cdc_max_poll_interval_ms=20,
            cdc_batch_size=256,
            cdc_batch_timeout_sec=0.5,
            cdc_log_retention=1000,
        )
        db = BankingDatabase(cfg)
        await db.initialize()

        sim = BankingEndpointSimulator(db, cfg)
        await sim.start()

        # Let it run briefly
        await asyncio.sleep(0.3)
        await sim.stop()

        assert sim.events_generated > 0, f"Expected events, got {sim.events_generated}"
        await db.close()


# ── Test 20: BankingEndpointSimulator fraud injection ─────────────────────

async def test_endpoint_fraud_injection():
    with tempfile.TemporaryDirectory() as tmp:
        cfg = StreamingConfig(
            db_path=Path(tmp) / "fraud_inject.db",
            endpoint_tps=50.0,
            cdc_poll_interval_ms=5,
            cdc_max_poll_interval_ms=20,
            cdc_batch_size=256,
            cdc_batch_timeout_sec=0.5,
            cdc_log_retention=1000,
        )
        db = BankingDatabase(cfg)
        await db.initialize()

        sim = BankingEndpointSimulator(db, cfg)
        await sim.start()

        # Inject layering fraud
        fraud_txns = await sim.inject_fraud(FraudPattern.LAYERING)
        assert len(fraud_txns) > 0, "Should inject at least 1 fraud transaction"
        assert all(isinstance(t, Transaction) for t in fraud_txns)
        assert all(t.fraud_label == FraudPattern.LAYERING for t in fraud_txns)

        # Inject round-trip fraud
        rt_txns = await sim.inject_fraud(FraudPattern.ROUND_TRIPPING)
        assert len(rt_txns) > 0

        await sim.stop()
        assert sim.events_generated >= len(fraud_txns) + len(rt_txns)
        await db.close()


# ── Test 21: BankingEndpointSimulator TPS control ────────────────────────

async def test_endpoint_tps_control():
    with tempfile.TemporaryDirectory() as tmp:
        cfg = StreamingConfig(
            db_path=Path(tmp) / "tps.db",
            endpoint_tps=20.0,  # 20 TPS
            cdc_poll_interval_ms=5,
            cdc_max_poll_interval_ms=20,
            cdc_batch_size=256,
            cdc_batch_timeout_sec=0.5,
            cdc_log_retention=1000,
        )
        db = BankingDatabase(cfg)
        await db.initialize()

        sim = BankingEndpointSimulator(db, cfg)
        await sim.start()

        # Run for ~0.5 second at 20 TPS -> expect ~10 events (with jitter tolerance)
        await asyncio.sleep(0.5)
        await sim.stop()

        # Allow generous margin for jitter and startup overhead
        assert sim.events_generated >= 2, f"Expected >=2 events at 20 TPS for 0.5s, got {sim.events_generated}"
        assert sim.events_generated <= 50, f"Expected <=50 events (TPS overshoot), got {sim.events_generated}"

        await db.close()


# ── Test 22: End-to-end CDC pipeline ──────────────────────────────────────

async def test_end_to_end_cdc_pipeline():
    with tempfile.TemporaryDirectory() as tmp:
        cfg = StreamingConfig(
            db_path=Path(tmp) / "e2e.db",
            cdc_poll_interval_ms=5,
            cdc_max_poll_interval_ms=20,
            cdc_batch_size=5,
            cdc_batch_timeout_sec=0.2,
            endpoint_tps=100.0,
            cdc_log_retention=500,
        )

        # 1. Initialize database
        db = BankingDatabase(cfg)
        await db.initialize()

        # 2. Start CDC reader
        reader = CDCReader(cfg.db_path, cfg)
        await reader.start()

        # 3. Start streaming consumer with collectors
        consumer = StreamingConsumer(reader, cfg)
        ml_batches = []
        graph_batches = []

        async def ml_consumer(batch: EventBatch) -> None:
            ml_batches.append(batch)

        async def graph_consumer(batch: EventBatch) -> None:
            graph_batches.append(batch)

        consumer.add_consumer(ml_consumer)
        consumer.add_consumer(graph_consumer)
        await consumer.start()

        # 4. Start endpoint simulator
        sim = BankingEndpointSimulator(db, cfg)
        await sim.start()

        # Let pipeline run
        await asyncio.sleep(0.6)

        # 5. Shutdown in order
        await sim.stop()
        await asyncio.sleep(0.3)  # Let consumer drain
        await consumer.stop()
        await reader.stop()
        await db.close()

        # Verify end-to-end flow
        assert sim.events_generated > 0, "Simulator should have generated events"
        total_ml_events = sum(b.event_count for b in ml_batches)
        total_graph_events = sum(b.event_count for b in graph_batches)
        assert total_ml_events > 0, "ML consumer should have received events"
        assert total_graph_events > 0, "Graph consumer should have received events"
        assert total_ml_events == total_graph_events, "Both consumers should see same events"


# ── Test 23: Streaming metrics serialization ──────────────────────────────

def test_streaming_metrics_serialization():
    m = StreamingMetrics()
    m.events_consumed = 100
    m.batches_dispatched = 10
    m.record_latency(5.0)
    m.record_latency(15.0)
    m.idle_cycles = 3
    m.consumer_errors = 1

    snap = m.snapshot()
    assert snap["events_consumed"] == 100
    assert snap["batches_dispatched"] == 10
    assert snap["avg_latency_ms"] == 10.0
    assert snap["peak_latency_ms"] == 15.0
    assert snap["consumer_errors"] == 1
    assert snap["idle_cycles"] == 3
    assert "throughput_eps" in snap
    assert "uptime_sec" in snap

    # All values should be JSON-serializable
    import json
    serialized = json.dumps(snap)
    assert isinstance(serialized, str)


# ── Test 24: Consumer protocol compatibility ──────────────────────────────

def test_consumer_protocol_compatibility():
    """
    Verify StreamingConsumer uses the same Consumer protocol
    as IngestionPipeline: Callable[[EventBatch], Coroutine[Any, Any, None]]
    """
    from src.ingestion.stream_processor import Consumer as PipelineConsumer
    from src.streaming.consumer import Consumer as StreamingConsumerProtocol

    # Both should be the same type alias
    assert PipelineConsumer == StreamingConsumerProtocol, (
        f"Protocol mismatch: Pipeline={PipelineConsumer}, Streaming={StreamingConsumerProtocol}"
    )

    # Verify StreamingConsumer accepts IngestionPipeline-style consumers
    async def mock_consumer(batch: EventBatch) -> None:
        pass

    # Should not raise
    cfg = StreamingConfig(db_path=Path("/tmp/unused.db"))
    reader_mock = MagicMock()
    sc = StreamingConsumer(reader_mock, cfg)
    sc.add_consumer(mock_consumer)
    assert len(sc._consumers) == 1

    # Verify package exports
    from src.streaming import (
        BankingDatabase,
        CDCReader,
        StreamingConsumer as SC,
        StreamingMetrics as SM,
        BankingEndpointSimulator as BES,
    )
    assert SC is not None
    assert SM is not None
    assert BES is not None


# ── Test Runner ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    _safe_print("\n=== Phase 11 Tests: Real-Time CDC Event Streaming Pipeline ===\n")

    tests = [
        test_streaming_config_defaults,
        test_streaming_config_immutability,
        test_banking_db_initialization,
        test_banking_db_insert_transaction,
        test_banking_db_insert_interbank,
        test_banking_db_insert_auth_event,
        test_banking_db_batch_insert,
        test_cdc_reader_reads_changes,
        test_cdc_reader_watermark_recovery,
        test_cdc_reader_deserialization,
        test_cdc_reader_adaptive_sleep,
        test_cdc_reader_acknowledge_cleanup,
        test_consumer_fan_out,
        test_consumer_batching_by_size,
        test_consumer_batching_by_timeout,
        test_consumer_graceful_shutdown,
        test_consumer_error_isolation,
        test_consumer_metrics,
        test_endpoint_generates_events,
        test_endpoint_fraud_injection,
        test_endpoint_tps_control,
        test_end_to_end_cdc_pipeline,
        test_streaming_metrics_serialization,
        test_consumer_protocol_compatibility,
    ]

    for test_func in tests:
        run_test(test_func)

    _safe_print(f"\n--- Results: {passed} passed, {failed} failed, {passed + failed} total ---")
    if failed:
        sys.exit(1)
    _safe_print("All Phase 11 tests passed!")
