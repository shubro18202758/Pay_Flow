"""
PayFlow — Simulated Banking Endpoint
======================================
Generates realistic banking events at configurable TPS and writes them
into the BankingDatabase, where CDC triggers capture them for streaming.

Reuses the established synthetic generators (build_world, generate_event_stream)
from src.ingestion.generators for data fidelity.
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from typing import Union

from config.settings import StreamingConfig
from src.ingestion.generators.synthetic_transactions import (
    build_world,
    generate_event_stream,
    generate_layering_chain,
    generate_round_trip,
    generate_structuring_burst,
    WorldState,
)
from src.ingestion.schemas import (
    AuthEvent,
    FraudPattern,
    InterbankMessage,
    Transaction,
)
from src.streaming.cdc import BankingDatabase

logger = logging.getLogger(__name__)

Event = Union[Transaction, InterbankMessage, AuthEvent]


class BankingEndpointSimulator:
    """
    Simulates live banking endpoints producing realistic transaction events.

    Events are generated using the proven synthetic generators and written
    into the BankingDatabase at a configurable TPS rate with realistic
    jitter (exponential inter-arrival distribution).
    """

    def __init__(self, db: BankingDatabase, config: StreamingConfig) -> None:
        self._db = db
        self._config = config
        self._world: WorldState | None = None
        self._task: asyncio.Task | None = None
        self._shutdown = asyncio.Event()
        self._events_generated: int = 0

    async def start(self) -> None:
        """Initialize the world state and launch the simulation loop."""
        self._world = build_world(num_accounts=500)
        self._shutdown.clear()
        self._events_generated = 0
        self._task = asyncio.create_task(self._simulation_loop(), name="endpoint-simulator")
        logger.info(
            "BankingEndpointSimulator started: tps=%.1f, fraud_ratio=%.2f",
            self._config.endpoint_tps, self._config.endpoint_fraud_ratio,
        )

    async def stop(self) -> None:
        """Signal shutdown and wait for the simulation loop to finish."""
        self._shutdown.set()
        if self._task:
            try:
                await asyncio.wait_for(self._task, timeout=10.0)
            except asyncio.TimeoutError:
                self._task.cancel()
                logger.warning("EndpointSimulator loop cancelled (timeout)")
            self._task = None
        logger.info("EndpointSimulator stopped. Total events: %d", self._events_generated)

    @property
    def events_generated(self) -> int:
        return self._events_generated

    async def inject_fraud(self, pattern: FraudPattern) -> list[Transaction]:
        """
        On-demand fraud injection for testing specific attack patterns.

        Returns the injected fraud transactions.
        """
        assert self._world is not None
        base_ts = int(time.time())

        if pattern == FraudPattern.LAYERING:
            txns = generate_layering_chain(self._world, base_ts)
        elif pattern == FraudPattern.ROUND_TRIPPING:
            txns = generate_round_trip(self._world, base_ts)
        elif pattern == FraudPattern.STRUCTURING:
            txns = generate_structuring_burst(self._world, base_ts)
        else:
            txns = generate_layering_chain(self._world, base_ts)

        await self._db.insert_batch(txns)
        self._events_generated += len(txns)
        logger.info("Injected %d fraud events (pattern=%s)", len(txns), pattern.name)
        return txns

    # ── Internal simulation loop ───────────────────────────────────────

    async def _simulation_loop(self) -> None:
        """Generate and insert events at the configured TPS rate."""
        assert self._world is not None

        # Pre-generate a batch of events to feed from
        batch_size = 1000
        event_iter = iter(
            generate_event_stream(
                self._world,
                num_events=batch_size,
                fraud_ratio=self._config.endpoint_fraud_ratio,
            )
        )

        target_interval = 1.0 / self._config.endpoint_tps

        while not self._shutdown.is_set():
            t0 = time.monotonic()

            # Get next event, regenerate batch if exhausted
            try:
                event = next(event_iter)
            except StopIteration:
                event_iter = iter(
                    generate_event_stream(
                        self._world,
                        num_events=batch_size,
                        fraud_ratio=self._config.endpoint_fraud_ratio,
                    )
                )
                event = next(event_iter)

            # Insert into database (CDC trigger fires automatically)
            try:
                if isinstance(event, Transaction):
                    await self._db.insert_transaction(event)
                elif isinstance(event, InterbankMessage):
                    await self._db.insert_interbank(event)
                elif isinstance(event, AuthEvent):
                    await self._db.insert_auth_event(event)
                self._events_generated += 1
            except Exception as exc:
                logger.error("Insert error: %s", exc)
                continue

            # Rate limiting with jitter (exponential distribution)
            elapsed = time.monotonic() - t0
            remaining = target_interval - elapsed
            if remaining > 0:
                jitter = random.expovariate(1.0 / remaining) if remaining > 0.001 else remaining
                sleep_time = min(jitter, remaining * 2)
                try:
                    await asyncio.wait_for(self._shutdown.wait(), timeout=sleep_time)
                    break  # Shutdown signaled
                except asyncio.TimeoutError:
                    pass
