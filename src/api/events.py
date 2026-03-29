"""
PayFlow -- Dashboard Event Broadcasting
=========================================
Lightweight in-process async pub/sub bus that streams subsystem events
to connected SSE dashboard clients.

Each channel carries a distinct event family:

    "graph"           — node/edge additions and graph mutations
    "agent"           — Chain-of-Thought steps, tool calls, verdicts
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
        async with self._lock:
            self._fanout(channel, event)

    def publish_sync(self, channel: str, data: dict) -> None:
        """
        Synchronous non-blocking publish — safe from sync code.

        Uses ``Queue.put_nowait()`` which is synchronous and safe
        when called from the same event-loop thread.
        """
        event = {"channel": channel, "timestamp": time.time(), "data": data}
        self._fanout(channel, event)

    def _fanout(self, channel: str, event: dict) -> None:
        """Deliver *event* to every subscriber queue on *channel*."""
        for queue in self._subscribers.get(channel, set()):
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
