"""
PayFlow -- Threat Simulation Engine
=====================================
Controllable engine that generates realistic fraud attack scenarios
and drip-feeds them into the IngestionPipeline. The core fraud
analyzer has zero awareness that payloads are simulated.

Usage::

    engine = ThreatSimulationEngine(pipeline, world)
    scenario_id = await engine.launch_attack("upi_mule_network")
    await engine.stop_attack(scenario_id)
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, ClassVar, Union

from config.settings import SIMULATION_CFG, SimulationConfig
from src.ingestion.schemas import (
    AuthEvent,
    InterbankMessage,
    Transaction,
)
from src.ingestion.stream_processor import IngestionPipeline
from src.ingestion.generators.synthetic_transactions import WorldState
from src.simulation.attack_generators import (
    generate_circular_laundering,
    generate_swift_heist,
    generate_upi_mule_network,
    generate_velocity_phishing,
    get_account_ids,
)

logger = logging.getLogger(__name__)

Event = Union[Transaction, InterbankMessage, AuthEvent]


# -- Scenario tracking -----------------------------------------------------

@dataclass
class AttackScenario:
    """Represents a running or completed attack simulation."""
    scenario_id: str
    attack_type: str
    attack_label: str
    status: str = "running"
    started_at: float = field(default_factory=time.time)
    stopped_at: float | None = None
    events_generated: int = 0
    events_ingested: int = 0
    account_ids: list[str] = field(default_factory=list)
    _task: asyncio.Task | None = field(default=None, repr=False)

    def to_dict(self) -> dict[str, Any]:
        elapsed = (self.stopped_at or time.time()) - self.started_at
        progress = (
            round(self.events_ingested / self.events_generated * 100, 1)
            if self.events_generated > 0 else 0.0
        )
        return {
            "scenario_id": self.scenario_id,
            "attack_type": self.attack_type,
            "attack_label": self.attack_label,
            "status": self.status,
            "events_generated": self.events_generated,
            "events_ingested": self.events_ingested,
            "progress_pct": progress,
            "accounts_involved": self.account_ids,
            "started_at": self.started_at,
            "stopped_at": self.stopped_at,
            "elapsed_sec": round(elapsed, 2),
        }


# -- Engine ----------------------------------------------------------------

class ThreatSimulationEngine:
    """
    Controllable threat simulation engine for live hackathon demos.

    Generates realistic fraud attack scenarios and drip-feeds them
    into the async IngestionPipeline at configurable pacing.
    """

    ATTACK_TYPES: ClassVar[dict[str, str]] = {
        "upi_mule_network": "Coordinated UPI Mule Network",
        "circular_laundering": "Circular Laundering via Shell Accounts",
        "velocity_phishing": "Velocity Phishing Attack",
        "swift_heist": "Bangladesh Bank SWIFT Night Heist",
    }

    def __init__(
        self,
        pipeline: IngestionPipeline,
        world: WorldState,
        config: SimulationConfig = SIMULATION_CFG,
    ) -> None:
        self._pipeline = pipeline
        self._world = world
        self._config = config
        self._rng = random.Random()
        self._scenarios: dict[str, AttackScenario] = {}
        self._broadcaster: Any = None

    # -- Public API --------------------------------------------------------

    async def launch_attack(
        self,
        attack_type: str,
        params: dict | None = None,
    ) -> str:
        """
        Launch a named attack scenario. Returns scenario_id.

        Raises ValueError for unknown attack_type.
        Raises RuntimeError if max concurrent attacks reached.
        """
        if attack_type not in self.ATTACK_TYPES:
            raise ValueError(
                f"Unknown attack type '{attack_type}'. "
                f"Available: {list(self.ATTACK_TYPES)}"
            )

        active = [s for s in self._scenarios.values() if s.status == "running"]
        if len(active) >= self._config.max_concurrent_attacks:
            raise RuntimeError(
                f"Max concurrent attacks ({self._config.max_concurrent_attacks}) reached"
            )

        scenario_id = str(uuid.uuid4())[:8]
        params = params or {}

        # Generate events
        events = self._generate_events(attack_type, params)
        account_ids = get_account_ids(events)

        scenario = AttackScenario(
            scenario_id=scenario_id,
            attack_type=attack_type,
            attack_label=self.ATTACK_TYPES[attack_type],
            events_generated=len(events),
            account_ids=account_ids,
        )
        self._scenarios[scenario_id] = scenario

        # Determine pacing
        if attack_type == "velocity_phishing":
            interval = self._config.burst_interval_sec
        else:
            interval = self._config.default_event_interval_sec

        # Spawn drip-feed task
        scenario._task = asyncio.create_task(
            self._drip_feed(scenario, events, interval),
            name=f"sim-{attack_type}-{scenario_id}",
        )

        await self._broadcast_status(scenario, "started")
        logger.info(
            "Attack launched: %s [%s] -- %d events, interval=%.2fs",
            scenario.attack_label, scenario_id, len(events), interval,
        )
        return scenario_id

    async def stop_attack(self, scenario_id: str) -> bool:
        """Stop a running attack. Returns True if found and stopped."""
        scenario = self._scenarios.get(scenario_id)
        if scenario is None or scenario.status != "running":
            return False

        scenario.status = "stopped"
        scenario.stopped_at = time.time()
        if scenario._task and not scenario._task.done():
            scenario._task.cancel()
            try:
                await scenario._task
            except asyncio.CancelledError:
                pass

        await self._broadcast_status(scenario, "stopped")
        logger.info("Attack stopped: %s [%s]", scenario.attack_label, scenario_id)
        return True

    async def stop_all(self) -> int:
        """Stop all running attacks. Returns count stopped."""
        count = 0
        for sid in list(self._scenarios):
            if self._scenarios[sid].status == "running":
                await self.stop_attack(sid)
                count += 1
        return count

    def get_status(self, scenario_id: str) -> dict | None:
        """Get status dict for a specific scenario."""
        scenario = self._scenarios.get(scenario_id)
        return scenario.to_dict() if scenario else None

    def list_active(self) -> list[dict]:
        """List all running scenarios."""
        return [
            s.to_dict() for s in self._scenarios.values()
            if s.status == "running"
        ]

    def list_all(self) -> list[dict]:
        """List all scenarios (active + completed + stopped)."""
        return [s.to_dict() for s in self._scenarios.values()]

    def available_attacks(self) -> dict[str, str]:
        """Return the attack type registry."""
        return dict(self.ATTACK_TYPES)

    def available_attacks_detailed(self) -> dict[str, dict]:
        """Return detailed attack type schemas with configurable parameters."""
        return {
            "upi_mule_network": {
                "label": "Coordinated UPI Mule Network",
                "description": "A compromised account sends UPI P2P transfers to N mule accounts. Each mule rapidly disperses funds to a single collector, creating a fan-out then fan-in pattern typical of organized fraud networks.",
                "phases": ["Compromise (OTP failures + attacker login)", "Fan-out (victim → N mules via UPI)", "Dispersal (mules → collector via IMPS)", "Settlement (collector → external via NEFT)"],
                "params": {
                    "mule_count": {"type": "int", "default": self._config.upi_mule_count, "min": 2, "max": 20, "step": 1, "label": "Mule Accounts", "description": "Number of intermediary mule accounts in the network"},
                    "total_amount_inr": {"type": "int", "default": self._config.upi_total_amount_inr, "min": 100000, "max": 50000000, "step": 100000, "label": "Total Amount (INR)", "description": "Total funds to launder through the mule network"},
                },
            },
            "circular_laundering": {
                "label": "Circular Laundering via Shell Accounts",
                "description": "Money circles through N shell/current accounts via alternating RTGS/NEFT hops. Each hop deducts a 1-3% consulting fee. The circle completes when funds return to the originating shell.",
                "phases": ["Shell Activation (account logins)", "Circular Hops (Transaction + InterbankMessage per hop)", "Fee Collection (each shell sends fee to collector)"],
                "params": {
                    "shell_count": {"type": "int", "default": self._config.circular_shell_count, "min": 3, "max": 15, "step": 1, "label": "Shell Accounts", "description": "Number of shell company accounts in the laundering ring"},
                    "hop_amount_inr": {"type": "int", "default": self._config.circular_hop_amount_inr, "min": 200000, "max": 100000000, "step": 100000, "label": "Hop Amount (INR)", "description": "Starting amount for the circular hops"},
                },
            },
            "velocity_phishing": {
                "label": "Velocity Phishing Attack",
                "description": "An attacker phishes credentials then rapidly attempts logins from multiple geolocations with mismatched device fingerprints. After gaining access, unauthorized transactions originate from geographically impossible locations.",
                "phases": ["Credential Theft (failed logins from scattered cities)", "Breach (successful login + OTP verify)", "Unauthorized Drain (impossible-travel transactions)", "Lockout (password change)"],
                "params": {
                    "credential_attempts": {"type": "int", "default": self._config.phishing_credential_attempts, "min": 3, "max": 30, "step": 1, "label": "Credential Attempts", "description": "Number of failed login attempts before breach"},
                    "unauthorized_txns": {"type": "int", "default": self._config.phishing_unauthorized_txns, "min": 1, "max": 20, "step": 1, "label": "Unauthorized Transactions", "description": "Number of fraudulent transactions after breach"},
                    "geo_spread": {"type": "int", "default": self._config.phishing_geo_spread, "min": 2, "max": 10, "step": 1, "label": "Geographic Spread", "description": "Number of distinct cities for impossible-travel pattern"},
                },
            },
            "swift_heist": {
                "label": "Bangladesh Bank SWIFT Night Heist",
                "description": "Simulates a SWIFT-based high-value international wire fraud at 2:30 AM. ₹8.1 Cr transfer to a NEW beneficiary from an unrecognized device 800 km away. Expected: risk score 90+, immediate BLOCK, FMR queued for RBI.",
                "phases": ["Reconnaissance (off-hours logins from 800km away)", "Credential Setup (login + OTP from unrecognized device)", "SWIFT Transfer (₹8.1 Cr single wire + MT103 message)", "Cover Tracks (quick logout)"],
                "params": {
                    "transfer_amount_inr": {"type": "int", "default": 8_10_00_000, "min": 1_00_00_000, "max": 50_00_00_000, "step": 10_00_000, "label": "Transfer Amount (INR)", "description": "Amount of the single SWIFT wire transfer"},
                    "recon_logins": {"type": "int", "default": 3, "min": 1, "max": 10, "step": 1, "label": "Recon Logins", "description": "Number of reconnaissance login/logout cycles before the heist"},
                },
            },
        }

    def snapshot(self) -> dict:
        """Engine state for system telemetry."""
        return {
            "active_attacks": len([
                s for s in self._scenarios.values() if s.status == "running"
            ]),
            "total_attacks": len(self._scenarios),
            "scenarios": self.list_all(),
        }

    # -- Internal ----------------------------------------------------------

    def _generate_events(
        self, attack_type: str, params: dict,
    ) -> list[Event]:
        """Dispatch to the appropriate generator."""
        base_ts = int(time.time())

        if attack_type == "upi_mule_network":
            return generate_upi_mule_network(
                self._world, self._rng, base_ts,
                mule_count=params.get("mule_count", self._config.upi_mule_count),
                total_amount_inr=params.get(
                    "total_amount_inr", self._config.upi_total_amount_inr,
                ),
            )
        elif attack_type == "circular_laundering":
            return generate_circular_laundering(
                self._world, self._rng, base_ts,
                shell_count=params.get(
                    "shell_count", self._config.circular_shell_count,
                ),
                hop_amount_inr=params.get(
                    "hop_amount_inr", self._config.circular_hop_amount_inr,
                ),
            )
        elif attack_type == "velocity_phishing":
            return generate_velocity_phishing(
                self._world, self._rng, base_ts,
                credential_attempts=params.get(
                    "credential_attempts",
                    self._config.phishing_credential_attempts,
                ),
                unauthorized_txns=params.get(
                    "unauthorized_txns",
                    self._config.phishing_unauthorized_txns,
                ),
                geo_spread=params.get(
                    "geo_spread", self._config.phishing_geo_spread,
                ),
            )
        elif attack_type == "swift_heist":
            return generate_swift_heist(
                self._world, self._rng, base_ts,
                transfer_amount_inr=params.get(
                    "transfer_amount_inr", 8_10_00_000,
                ),
                recon_logins=params.get("recon_logins", 3),
            )
        else:
            raise ValueError(f"No generator for '{attack_type}'")

    async def _drip_feed(
        self,
        scenario: AttackScenario,
        events: list[Event],
        interval_sec: float,
    ) -> None:
        """Drip-feed events into the pipeline with configurable pacing."""
        # Ensure pipeline is running
        if not self._pipeline._running:
            await self._pipeline.start()

        try:
            for i, event in enumerate(events):
                if scenario.status != "running":
                    break

                await self._pipeline.ingest(event)
                scenario.events_ingested += 1

                await self._broadcast_event(scenario, event, i, len(events))

                if i < len(events) - 1:
                    await asyncio.sleep(interval_sec)

            if scenario.status == "running":
                scenario.status = "completed"
                scenario.stopped_at = time.time()
                await self._broadcast_status(scenario, "completed")
                logger.info(
                    "Attack completed: %s [%s] -- %d/%d events ingested",
                    scenario.attack_label, scenario.scenario_id,
                    scenario.events_ingested, scenario.events_generated,
                )
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            scenario.status = "error"
            scenario.stopped_at = time.time()
            logger.error(
                "Attack error: %s [%s] -- %s",
                scenario.attack_label, scenario.scenario_id, exc,
            )

    # -- SSE Broadcasting --------------------------------------------------

    def _get_broadcaster(self):
        if self._broadcaster is None:
            try:
                from src.api.events import EventBroadcaster
                self._broadcaster = EventBroadcaster.get()
            except Exception:
                return None
        return self._broadcaster

    async def _broadcast_event(
        self, scenario: AttackScenario, event: Event,
        index: int, total: int,
    ) -> None:
        """Publish a simulation event to the 'simulation' SSE channel."""
        broadcaster = self._get_broadcaster()
        if broadcaster is None:
            return

        summary = self._event_summary(event)
        try:
            await broadcaster.publish("simulation", {
                "type": "attack_event",
                "scenario_id": scenario.scenario_id,
                "attack_type": scenario.attack_type,
                "event_index": index,
                "total_events": total,
                "progress_pct": round((index + 1) / total * 100, 1),
                "event": summary,
            })
        except Exception:
            pass

    async def _broadcast_status(
        self, scenario: AttackScenario, status_type: str,
    ) -> None:
        """Publish scenario lifecycle events."""
        broadcaster = self._get_broadcaster()
        if broadcaster is None:
            return
        try:
            await broadcaster.publish("simulation", {
                "type": f"simulation_{status_type}",
                "scenario_id": scenario.scenario_id,
                "attack_type": scenario.attack_type,
                "attack_label": scenario.attack_label,
                "events_generated": scenario.events_generated,
                "events_ingested": scenario.events_ingested,
                "accounts_involved": scenario.account_ids,
            })
        except Exception:
            pass

    @staticmethod
    def _event_summary(event: Event) -> dict:
        """Create a concise summary dict of an event for SSE."""
        if isinstance(event, Transaction):
            return {
                "type": "transaction",
                "txn_id": event.txn_id,
                "sender": event.sender_id,
                "receiver": event.receiver_id,
                "amount_paisa": event.amount_paisa,
                "channel": event.channel.name,
                "fraud_label": event.fraud_label.name,
            }
        elif isinstance(event, InterbankMessage):
            return {
                "type": "interbank",
                "msg_id": event.msg_id,
                "sender_ifsc": event.sender_ifsc,
                "receiver_ifsc": event.receiver_ifsc,
                "amount_paisa": event.amount_paisa,
                "channel": event.channel.name,
            }
        elif isinstance(event, AuthEvent):
            return {
                "type": "auth",
                "event_id": event.event_id,
                "account": event.account_id,
                "action": event.action.name,
                "success": event.success,
                "ip": event.ip_address,
            }
        return {"type": "unknown"}
