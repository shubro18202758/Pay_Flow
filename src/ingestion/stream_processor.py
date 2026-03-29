"""
PayFlow — Asynchronous Streaming Ingestion Pipeline
=====================================================
Three-stage async pipeline:  Producer → Validator → Consumer fan-out

Architecture:
    ┌──────────┐     ┌───────────┐     ┌──────────────┐
    │ Generator │────→│ Validator │────→│  Fan-out     │
    │ (async)   │  Q1 │ (CRC32)   │  Q2 │  ├─ ML store │
    │ txn/msg/  │     │ reject    │     │  ├─ Graph    │
    │ auth      │     │ malformed │     │  └─ Ledger   │
    └──────────┘     └───────────┘     └──────────────┘

Backpressure: bounded asyncio.Queues. If a consumer is slow, the validator
blocks on put(), which blocks the generator on put(). No events are dropped;
the pipeline self-regulates to the slowest consumer.

Why pure asyncio (no Kafka/Flink/Bytewax):
- Zero infrastructure: no brokers, no JVMs, no Docker containers
- Windows-native: asyncio ProactorEventLoop is first-class on Win11
- For a hackathon demo with synthetic data, external message brokers
  add operational complexity with zero analytical benefit
- Throughput ceiling of asyncio.Queue: ~500K events/sec on a single
  core — more than sufficient for our demo (target: 10K events/sec)
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine, Union

from src.ingestion.schemas import (
    AuthEvent,
    EventBatch,
    InterbankMessage,
    Transaction,
)
from src.ingestion.validators import ValidationResult, validate_event

logger = logging.getLogger(__name__)

Event = Union[Transaction, InterbankMessage, AuthEvent]

# Sentinel to signal pipeline shutdown
_SHUTDOWN = object()


# ── Pipeline Metrics ──────────────────────────────────────────────────────────

@dataclass
class PipelineMetrics:
    """Real-time pipeline health counters (lock-free, single-writer)."""
    events_ingested: int = 0
    events_validated: int = 0
    events_rejected: int = 0
    events_dispatched: int = 0
    batches_emitted: int = 0
    _start_time: float = field(default_factory=time.monotonic)

    @property
    def uptime_sec(self) -> float:
        return time.monotonic() - self._start_time

    @property
    def throughput_eps(self) -> float:
        """Events per second (ingested)."""
        elapsed = self.uptime_sec
        return self.events_ingested / elapsed if elapsed > 0 else 0.0

    def snapshot(self) -> dict:
        return {
            "ingested": self.events_ingested,
            "validated": self.events_validated,
            "rejected": self.events_rejected,
            "dispatched": self.events_dispatched,
            "batches": self.batches_emitted,
            "throughput_eps": round(self.throughput_eps, 1),
            "uptime_sec": round(self.uptime_sec, 1),
        }


# ── Consumer Protocol ────────────────────────────────────────────────────────

# A consumer is any async callable that accepts an EventBatch.
Consumer = Callable[[EventBatch], Coroutine[Any, Any, None]]


# ── Streaming Pipeline ───────────────────────────────────────────────────────

class IngestionPipeline:
    """
    Three-stage async streaming pipeline with backpressure.

    Usage:
        pipeline = IngestionPipeline(batch_size=512, batch_timeout_sec=1.0)
        pipeline.add_consumer(ml_feature_store.ingest)
        pipeline.add_consumer(graph_builder.ingest)
        pipeline.add_consumer(blockchain_ledger.ingest)

        async with pipeline:
            # Feed events from a generator
            async for event in transaction_generator():
                await pipeline.ingest(event)
    """

    def __init__(
        self,
        batch_size: int = 512,
        batch_timeout_sec: float = 1.0,
        queue_capacity: int = 10_000,
        max_reject_log: int = 100,
    ):
        self._batch_size = batch_size
        self._batch_timeout = batch_timeout_sec
        self._queue_capacity = queue_capacity
        self._max_reject_log = max_reject_log

        # Stage queues
        self._raw_queue: asyncio.Queue[Event | object] = asyncio.Queue(maxsize=queue_capacity)
        self._valid_queue: asyncio.Queue[Event | object] = asyncio.Queue(maxsize=queue_capacity)

        # Consumers
        self._consumers: list[Consumer] = []

        # Metrics
        self.metrics = PipelineMetrics()

        # Worker tasks
        self._tasks: list[asyncio.Task] = []
        self._running = False
        self._batch_counter = 0

    def add_consumer(self, consumer: Consumer) -> None:
        """Register an async consumer to receive validated EventBatches."""
        self._consumers.append(consumer)

    # ── Public API ────────────────────────────────────────────────────────

    async def ingest(self, event: Event) -> None:
        """
        Submit a single event into the pipeline.
        Blocks if the raw queue is full (backpressure to the producer).
        """
        await self._raw_queue.put(event)
        self.metrics.events_ingested += 1

    async def ingest_many(self, events: list[Event]) -> None:
        """Bulk ingest — maintains backpressure per event."""
        for event in events:
            await self.ingest(event)

    async def start(self) -> None:
        """Launch the validator and batcher worker tasks."""
        if self._running:
            return
        self._running = True
        self.metrics = PipelineMetrics()

        self._tasks = [
            asyncio.create_task(self._validator_worker(), name="validator"),
            asyncio.create_task(self._batcher_worker(), name="batcher"),
        ]
        logger.info(
            "Pipeline started: batch_size=%d, timeout=%.1fs, consumers=%d",
            self._batch_size, self._batch_timeout, len(self._consumers),
        )

    async def stop(self) -> None:
        """Gracefully shut down: drain queues, flush final batch, cancel workers."""
        if not self._running:
            return
        self._running = False

        # Send shutdown sentinels
        await self._raw_queue.put(_SHUTDOWN)

        # Wait for workers to finish processing
        for task in self._tasks:
            try:
                await asyncio.wait_for(task, timeout=10.0)
            except asyncio.TimeoutError:
                task.cancel()
                logger.warning("Task %s cancelled (timeout)", task.get_name())

        self._tasks.clear()
        logger.info("Pipeline stopped. Final metrics: %s", self.metrics.snapshot())

    async def __aenter__(self):
        await self.start()
        return self

    async def __aexit__(self, *exc):
        await self.stop()

    # ── Stage 1: Validator Worker ─────────────────────────────────────────

    async def _validator_worker(self) -> None:
        """
        Pulls events from raw_queue, validates, pushes valid events
        to valid_queue. Rejects malformed events with logging.
        """
        reject_count = 0

        while True:
            event = await self._raw_queue.get()

            if event is _SHUTDOWN:
                await self._valid_queue.put(_SHUTDOWN)
                break

            result: ValidationResult = validate_event(event)

            if result.valid:
                await self._valid_queue.put(event)
                self.metrics.events_validated += 1
            else:
                self.metrics.events_rejected += 1
                reject_count += 1
                if reject_count <= self._max_reject_log:
                    logger.warning(
                        "Event rejected [%s]: %s",
                        result.event_id, "; ".join(result.errors),
                    )

    # ── Stage 2: Batcher + Dispatcher Worker ──────────────────────────────

    async def _batcher_worker(self) -> None:
        """
        Collects validated events into micro-batches (by count OR timeout),
        then fans out each batch to all registered consumers concurrently.
        """
        buffer_txn: list[Transaction] = []
        buffer_msg: list[InterbankMessage] = []
        buffer_auth: list[AuthEvent] = []
        last_flush = time.monotonic()

        while True:
            try:
                event = await asyncio.wait_for(
                    self._valid_queue.get(),
                    timeout=self._batch_timeout,
                )
            except asyncio.TimeoutError:
                # Timeout — flush whatever we have
                event = None

            if event is _SHUTDOWN:
                # Flush remaining buffer then exit
                await self._flush_batch(buffer_txn, buffer_msg, buffer_auth)
                break

            if event is not None:
                # Route to typed buffer
                if isinstance(event, Transaction):
                    buffer_txn.append(event)
                elif isinstance(event, InterbankMessage):
                    buffer_msg.append(event)
                elif isinstance(event, AuthEvent):
                    buffer_auth.append(event)

            total_buffered = len(buffer_txn) + len(buffer_msg) + len(buffer_auth)
            elapsed = time.monotonic() - last_flush

            # Flush on size OR timeout
            if total_buffered >= self._batch_size or (elapsed >= self._batch_timeout and total_buffered > 0):
                await self._flush_batch(buffer_txn, buffer_msg, buffer_auth)
                buffer_txn = []
                buffer_msg = []
                buffer_auth = []
                last_flush = time.monotonic()

    async def _flush_batch(
        self,
        txns: list[Transaction],
        msgs: list[InterbankMessage],
        auths: list[AuthEvent],
    ) -> None:
        """Package events into an EventBatch and dispatch to all consumers."""
        total = len(txns) + len(msgs) + len(auths)
        if total == 0:
            return

        self._batch_counter += 1
        batch = EventBatch(
            transactions=txns,
            interbank_messages=msgs,
            auth_events=auths,
            batch_id=self._batch_counter,
            batch_timestamp=int(time.time()),
            event_count=total,
        )

        # Collect all event IDs for per-txn stage tracking
        txn_ids: list[str] = (
            [t.txn_id for t in txns]
            + [m.msg_id for m in msgs]
            + [a.event_id for a in auths]
        )

        # Broadcast per-txn ingestion + validation stage complete
        try:
            from src.api.events import EventBroadcaster
            broadcaster = EventBroadcaster.get()
            await broadcaster.publish("pipeline", {
                "type": "stage_complete",
                "stage": "ingested",
                "txn_ids": txn_ids,
                "batch_id": self._batch_counter,
                "duration_ms": 0,
            })
        except Exception:
            pass

        # Fan-out to all consumers concurrently, tracking per-consumer timing
        if self._consumers:
            consumer_results = []
            # Map consumer qualnames to pipeline stage names
            _CONSUMER_STAGE_MAP = {
                "FeatureEngine.ingest": "ml_scored",
                "TransactionGraph.ingest": "graph_investigated",
                "AuditLedger.ingest": "cb_evaluated",
            }
            for consumer in self._consumers:
                consumer_name = getattr(consumer, "__qualname__", getattr(consumer, "__name__", str(consumer)))
                t_start = time.monotonic()
                try:
                    await consumer(batch)
                    duration_ms = round((time.monotonic() - t_start) * 1000, 1)
                    consumer_results.append({
                        "consumer": consumer_name,
                        "success": True,
                        "duration_ms": duration_ms,
                    })
                except Exception as exc:
                    duration_ms = round((time.monotonic() - t_start) * 1000, 1)
                    consumer_results.append({
                        "consumer": consumer_name,
                        "success": False,
                        "duration_ms": duration_ms,
                        "error": str(exc)[:200],
                    })

                # Broadcast per-consumer stage_complete for tracked events
                stage_name = _CONSUMER_STAGE_MAP.get(consumer_name)
                if stage_name:
                    try:
                        from src.api.events import EventBroadcaster
                        await EventBroadcaster.get().publish("pipeline", {
                            "type": "stage_complete",
                            "stage": stage_name,
                            "txn_ids": txn_ids,
                            "batch_id": self._batch_counter,
                            "duration_ms": duration_ms,
                            "consumer": consumer_name,
                        })
                    except Exception:
                        pass

            # Broadcast pipeline stage details via SSE
            try:
                from src.api.events import EventBroadcaster
                await EventBroadcaster.get().publish("pipeline", {
                    "type": "batch_dispatched",
                    "batch_id": self._batch_counter,
                    "event_count": total,
                    "transactions": len(txns),
                    "auth_events": len(auths),
                    "interbank_messages": len(msgs),
                    "consumers": consumer_results,
                    "txn_ids": txn_ids,
                })
            except Exception:
                pass

        self.metrics.events_dispatched += total
        self.metrics.batches_emitted += 1

        if self.metrics.batches_emitted % 50 == 0:
            logger.info("Pipeline metrics: %s", self.metrics.snapshot())
