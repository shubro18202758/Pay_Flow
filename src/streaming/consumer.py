"""
PayFlow — Streaming Consumer
==============================
Consumes CDC events from CDCReader and fans out to downstream consumers
using the same Consumer protocol as IngestionPipeline.

Architecture:
    ┌──────────┐     read_changes()    ┌───────────────────┐
    │ CDCReader │ ────────────────────→ │ StreamingConsumer  │
    │ (async   │     list[(id, Event)] │  ├─ batch by size  │
    │  poll)   │                       │  ├─ batch by time  │
    └──────────┘                       │  └─ fan-out:       │
                                       │     ├─ ML store   │
                                       │     ├─ Graph      │
                                       │     └─ Ledger     │
                                       └───────────────────┘

Consumer Protocol: Callable[[EventBatch], Coroutine[Any, Any, None]]
  - Same as IngestionPipeline — zero modifications to existing consumers.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine

from config.settings import StreamingConfig
from src.ingestion.schemas import (
    AuthEvent,
    EventBatch,
    InterbankMessage,
    Transaction,
)
from src.streaming.cdc import CDCReader

logger = logging.getLogger(__name__)

Consumer = Callable[[EventBatch], Coroutine[Any, Any, None]]


# ── Streaming Metrics ───────────────────────────────────────────────────────

@dataclass
class StreamingMetrics:
    """Real-time streaming pipeline health counters."""
    events_consumed: int = 0
    batches_dispatched: int = 0
    avg_latency_ms: float = 0.0
    peak_latency_ms: float = 0.0
    consumer_errors: int = 0
    idle_cycles: int = 0
    _start_time: float = field(default_factory=time.monotonic)
    _latency_sum: float = field(default=0.0, repr=False)
    _latency_count: int = field(default=0, repr=False)

    @property
    def uptime_sec(self) -> float:
        return time.monotonic() - self._start_time

    @property
    def throughput_eps(self) -> float:
        elapsed = self.uptime_sec
        return self.events_consumed / elapsed if elapsed > 0 else 0.0

    def record_latency(self, latency_ms: float) -> None:
        """Record a dispatch latency measurement."""
        self._latency_sum += latency_ms
        self._latency_count += 1
        self.avg_latency_ms = self._latency_sum / self._latency_count
        if latency_ms > self.peak_latency_ms:
            self.peak_latency_ms = latency_ms

    def snapshot(self) -> dict:
        return {
            "events_consumed": self.events_consumed,
            "batches_dispatched": self.batches_dispatched,
            "avg_latency_ms": round(self.avg_latency_ms, 2),
            "peak_latency_ms": round(self.peak_latency_ms, 2),
            "consumer_errors": self.consumer_errors,
            "idle_cycles": self.idle_cycles,
            "throughput_eps": round(self.throughput_eps, 1),
            "uptime_sec": round(self.uptime_sec, 1),
        }


# ── Streaming Consumer ─────────────────────────────────────────────────────

class StreamingConsumer:
    """
    Consumes CDC events and dispatches to registered consumers
    as EventBatch micro-batches.

    Batching uses dual flush triggers:
    - Size threshold: flush when buffer reaches cdc_batch_size
    - Timeout: flush after cdc_batch_timeout_sec even if under size

    Fan-out uses asyncio.gather with return_exceptions for error isolation.
    """

    def __init__(self, cdc_reader: CDCReader, config: StreamingConfig) -> None:
        self._reader = cdc_reader
        self._config = config
        self._consumers: list[Consumer] = []
        self._metrics = StreamingMetrics()
        self._shutdown = asyncio.Event()
        self._task: asyncio.Task | None = None
        self._batch_counter: int = 0

    def add_consumer(self, consumer: Consumer) -> None:
        """Register a downstream consumer."""
        self._consumers.append(consumer)

    async def start(self) -> None:
        """Launch the consume loop as a background task."""
        self._shutdown.clear()
        self._metrics = StreamingMetrics()
        self._task = asyncio.create_task(self._consume_loop(), name="streaming-consumer")
        logger.info(
            "StreamingConsumer started: consumers=%d, batch_size=%d, timeout=%.1fs",
            len(self._consumers), self._config.cdc_batch_size, self._config.cdc_batch_timeout_sec,
        )

    async def stop(self) -> None:
        """Signal shutdown and wait for consume loop to finish."""
        self._shutdown.set()
        if self._task:
            try:
                await asyncio.wait_for(self._task, timeout=10.0)
            except asyncio.TimeoutError:
                self._task.cancel()
                logger.warning("StreamingConsumer loop cancelled (timeout)")
            self._task = None
        logger.info("StreamingConsumer stopped. Metrics: %s", self._metrics.snapshot())

    @property
    def metrics(self) -> StreamingMetrics:
        return self._metrics

    # ── Internal consume loop ──────────────────────────────────────────

    async def _consume_loop(self) -> None:
        """Main consume loop: poll → batch → dispatch → acknowledge."""
        buffer_txn: list[Transaction] = []
        buffer_msg: list[InterbankMessage] = []
        buffer_auth: list[AuthEvent] = []
        last_flush = time.monotonic()
        max_cdc_id: int = 0

        while not self._shutdown.is_set():
            # 1. Read new changes from CDC log
            try:
                changes = await self._reader.read_changes()
            except Exception as exc:
                logger.error("CDC read error: %s", exc)
                self._metrics.consumer_errors += 1
                await asyncio.sleep(0.1)
                continue

            had_results = len(changes) > 0

            # 2. Accumulate into typed buffers
            for cdc_id, event in changes:
                if cdc_id > max_cdc_id:
                    max_cdc_id = cdc_id

                if isinstance(event, Transaction):
                    buffer_txn.append(event)
                elif isinstance(event, InterbankMessage):
                    buffer_msg.append(event)
                elif isinstance(event, AuthEvent):
                    buffer_auth.append(event)

            # 3. Check flush conditions
            total_buffered = len(buffer_txn) + len(buffer_msg) + len(buffer_auth)
            elapsed = time.monotonic() - last_flush

            should_flush = (
                total_buffered >= self._config.cdc_batch_size
                or (elapsed >= self._config.cdc_batch_timeout_sec and total_buffered > 0)
            )

            if should_flush:
                await self._flush_and_dispatch(buffer_txn, buffer_msg, buffer_auth)
                buffer_txn = []
                buffer_msg = []
                buffer_auth = []
                last_flush = time.monotonic()

                # 4. Acknowledge consumed entries
                if max_cdc_id > 0:
                    try:
                        await self._reader.acknowledge(max_cdc_id)
                    except Exception as exc:
                        logger.error("CDC acknowledge error: %s", exc)
                        self._metrics.consumer_errors += 1

            if not had_results:
                self._metrics.idle_cycles += 1

            # 5. Adaptive sleep
            sleep_sec = self._reader.adaptive_sleep(had_results)
            try:
                await asyncio.wait_for(self._shutdown.wait(), timeout=sleep_sec)
                break  # Shutdown signaled during sleep
            except asyncio.TimeoutError:
                pass  # Normal wakeup

        # Final flush on shutdown
        total_remaining = len(buffer_txn) + len(buffer_msg) + len(buffer_auth)
        if total_remaining > 0:
            await self._flush_and_dispatch(buffer_txn, buffer_msg, buffer_auth)
            if max_cdc_id > 0:
                try:
                    await self._reader.acknowledge(max_cdc_id)
                except Exception:
                    pass

    async def _flush_and_dispatch(
        self,
        txns: list[Transaction],
        msgs: list[InterbankMessage],
        auths: list[AuthEvent],
    ) -> None:
        """Build an EventBatch and fan out to all consumers."""
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

        # Fan-out with error isolation
        if self._consumers:
            t0 = time.monotonic()
            results = await asyncio.gather(
                *(consumer(batch) for consumer in self._consumers),
                return_exceptions=True,
            )
            latency_ms = (time.monotonic() - t0) * 1000
            self._metrics.record_latency(latency_ms)

            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    self._metrics.consumer_errors += 1
                    logger.error("Consumer[%d] error: %s", i, result)

        self._metrics.events_consumed += total
        self._metrics.batches_dispatched += 1

        if self._metrics.batches_dispatched % 50 == 0:
            logger.info("Streaming metrics: %s", self._metrics.snapshot())
