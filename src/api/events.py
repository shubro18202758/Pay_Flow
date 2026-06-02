"""
PayFlow -- Dashboard Event Broadcasting
=========================================
Lightweight in-process async pub/sub bus that streams subsystem events
to connected SSE dashboard clients.

Each channel carries a distinct event family:

    "graph"           — node/edge additions and graph mutations
    "agent"           — investigation trace steps, tool calls, verdicts
    "circuit_breaker" — freeze / unfreeze / device ban / routing pause
    "risk_scores"     — ML risk score batches and threshold changes
    "system"          — hardware telemetry, pipeline throughput (1 Hz)

Design notes:

    * Zero external dependencies — pure ``asyncio.Queue`` fan-out.
    * ``publish_sync()`` is safe to call from synchronous LangGraph
      nodes and ``TransactionGraph._add_transactions()`` because
      ``asyncio.Queue.put_nowait()`` is non-blocking and thread-loop-
      safe when invoked from the same event-loop thread.
    * Back-pressure per client: if a subscriber's queue is full the
      event is silently dropped (the dashboard is a best-effort view).
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import defaultdict
from typing import ClassVar

logger = logging.getLogger(__name__)


class EventBroadcaster:
    """
    Singleton in-process pub/sub for real-time dashboard streaming.

    Usage::

        broadcaster = EventBroadcaster.get()

        # Async subsystems
        await broadcaster.publish("agent", {"type": "verdict", ...})

        # Synchronous subsystems (LangGraph nodes, graph mutations)
        broadcaster.publish_sync("graph", {"type": "batch_update", ...})

        # SSE endpoint
        queue = await broadcaster.subscribe(["agent", "graph"])
        async for event in queue:  ...
    """

    _instance: ClassVar[EventBroadcaster | None] = None

    def __init__(self) -> None:
        # channel name -> set of subscriber queues
        self._subscribers: dict[str, set[asyncio.Queue]] = defaultdict(set)
        self._lock = asyncio.Lock()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._pending_sync_fanouts = 0
        self._max_pending_sync_fanouts = 256

    def _bind_loop(self) -> None:
        """Remember the owning asyncio loop for thread-safe sync publishers."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        if self._loop is None or self._loop.is_closed():
            self._loop = loop

    # ── Singleton ─────────────────────────────────────────────────────

    @classmethod
    def get(cls) -> EventBroadcaster:
        """Return the global singleton, creating it on first access."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Reset the singleton (for tests only)."""
        cls._instance = None

    # ── Publishing ────────────────────────────────────────────────────

    async def publish(self, channel: str, data: dict) -> None:
        """
        Async fan-out to all subscribers on *channel*.

        Drops the event for any subscriber whose queue is full
        (back-pressure: dashboard is best-effort).
        """
        event = {"channel": channel, "timestamp": time.time(), "data": data}
        self._bind_loop()
        async with self._lock:
            self._fanout(channel, event)

    def publish_sync(self, channel: str, data: dict) -> None:
        """
        Synchronous non-blocking publish — safe from sync code.

        When called from LangGraph/LLM worker threads, schedules fan-out back
        onto the owning asyncio loop so subscriber queues are not mutated
        across threads.
        """
        event = {"channel": channel, "timestamp": time.time(), "data": data}
        loop = self._loop
        if loop is not None and loop.is_running():
            try:
                running_loop = asyncio.get_running_loop()
            except RuntimeError:
                running_loop = None
            if running_loop is loop:
                self._fanout(channel, event)
                return
            if self._pending_sync_fanouts >= self._max_pending_sync_fanouts:
                return
            self._pending_sync_fanouts += 1
            loop.call_soon_threadsafe(self._fanout_scheduled, channel, event)
            return
        self._fanout(channel, event)

    def _fanout_scheduled(self, channel: str, event: dict) -> None:
        try:
            self._fanout(channel, event)
        finally:
            self._pending_sync_fanouts = max(0, self._pending_sync_fanouts - 1)

    def _fanout(self, channel: str, event: dict) -> None:
        """Deliver *event* to every subscriber queue on *channel*."""
        for queue in list(self._subscribers.get(channel, set())):
            try:
                queue.put_nowait(event)
            except asyncio.QueueFull:
                pass  # back-pressure — silently drop

    # ── Subscription ──────────────────────────────────────────────────

    async def subscribe(
        self, channels: list[str], max_size: int = 256,
    ) -> asyncio.Queue:
        """
        Create a new subscriber queue registered to *channels*.

        Returns the queue — callers ``get()`` events from it.
        """
        queue: asyncio.Queue = asyncio.Queue(maxsize=max_size)
        self._bind_loop()
        async with self._lock:
            for ch in channels:
                self._subscribers[ch].add(queue)
        return queue

    async def unsubscribe(
        self, queue: asyncio.Queue, channels: list[str],
    ) -> None:
        """Remove *queue* from all *channels*."""
        async with self._lock:
            for ch in channels:
                self._subscribers[ch].discard(queue)

    # ── Diagnostics ───────────────────────────────────────────────────

    def subscriber_count(self, channel: str) -> int:
        """Number of active subscribers on *channel*."""
        return len(self._subscribers.get(channel, set()))

    def snapshot(self) -> dict:
        """Current broadcaster state for debugging."""
        return {
            ch: len(queues) for ch, queues in self._subscribers.items()
        }
