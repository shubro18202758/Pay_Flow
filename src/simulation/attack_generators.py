"""
PayFlow -- Threat Simulation Attack Generators
================================================
Three highly realistic fraud typology generators for live hackathon demos.

Each generator returns a time-ordered list of mixed event types
(Transaction, InterbankMessage, AuthEvent) with correct CRC32 checksums
and structural compliance so they pass ingestion validation unmodified.

Typologies:
  1. Coordinated UPI Mule Network  -- fan-out/fan-in through mules
  2. Circular Laundering via Shell Accounts -- RTGS/NEFT hops in a ring
  3. Velocity Phishing Attack -- credential theft + impossible-travel drain
"""

from __future__ import annotations

import hashlib
import random
import time
from typing import Union

from src.ingestion.generators.synthetic_transactions import (
    UBI_IFSC_PREFIX,
    WorldState,
    _BRANCH_CODES,
    _CITY_COORDS,
    _device_fingerprint,
    _hex_id,
)
from src.ingestion.schemas import (
    AccountType,
    AuthAction,
    AuthEvent,
    Channel,
    FraudPattern,
    InterbankMessage,
    Transaction,
)
from src.ingestion.validators import (
    compute_auth_checksum,
    compute_interbank_checksum,
    compute_transaction_checksum,
)

Event = Union[Transaction, InterbankMessage, AuthEvent]


# -- Local utilities (avoid contaminating the shared _RNG) -----------------

def _sim_get_coords(branch_code: str, rng: random.Random) -> tuple[float, float]:
    """Get approximate coordinates for a branch, with jitter. Own RNG."""
    base_key = branch_code[:6]
    if base_key in _CITY_COORDS:
        lat, lon = _CITY_COORDS[base_key]
    else:
        lat = rng.uniform(8.0, 35.0)
        lon = rng.uniform(69.0, 97.0)
    return (
        round(lat + rng.uniform(-0.05, 0.05), 6),
        round(lon + rng.uniform(-0.05, 0.05), 6),
    )


def _random_indian_ip(rng: random.Random) -> str:
    """Generate a plausible Indian IP address."""
    first_octet = rng.choice([49, 59, 103, 106, 117, 122, 157, 182, 203])
    return f"{first_octet}.{rng.randint(0,255)}.{rng.randint(0,255)}.{rng.randint(1,254)}"


def _ua_hash() -> str:
    """Generate a user-agent hash (16 hex chars)."""
    return hashlib.sha256(f"Mozilla/{random.randint(5,6)}.0".encode()).hexdigest()[:16]


def _pick_distinct_cities(rng: random.Random, count: int) -> list[str]:
    """Pick N distinct city branch codes that have known coordinates."""
    known = [b for b in _BRANCH_CODES if b[:6] in _CITY_COORDS]
    count = min(count, len(known))
    return rng.sample(known, count)


# ==========================================================================
# Attack 1: Coordinated UPI Mule Network
# ==========================================================================

def generate_upi_mule_network(
    world: WorldState,
    rng: random.Random,
    base_timestamp: int | None = None,
    mule_count: int = 6,
    total_amount_inr: int = 15_00_000,
) -> list[Event]:
    """
    Coordinated UPI Mule Network attack.

    A compromised account sends UPI P2P transfers to N mule accounts.
    Each mule rapidly disperses funds to a single collector account,
    creating a fan-out then fan-in pattern.

    Phases:
      1. Compromise -- OTP failures then attacker login
      2. Fan-out   -- victim -> N mules via UPI
      3. Dispersal -- each mule -> collector via IMPS
      4. Settlement -- collector -> external via NEFT interbank message
    """
    if base_timestamp is None:
        base_timestamp = int(time.time())

    needed = mule_count + 2  # victim + N mules + collector
    accounts = rng.sample(world.accounts, min(needed, len(world.accounts)))
    victim = accounts[0]
    mules = accounts[1 : mule_count + 1]
    collector = accounts[-1] if len(accounts) > mule_count + 1 else accounts[1]

    events: list[Event] = []
    ts = base_timestamp

    attacker_fp = _device_fingerprint()
    attacker_ip = _random_indian_ip(rng)
    victim_ip = _random_indian_ip(rng)

    # -- Phase 1: Compromise (AuthEvents) --
    # Failed OTP attempts from victim's IP (phished credentials being tested)
    for _ in range(rng.randint(2, 3)):
        ts += rng.randint(5, 20)
        eid = _hex_id("AUTH")
        cs = compute_auth_checksum(eid, ts, victim.account_id, int(AuthAction.OTP_FAIL), victim_ip)
        lat, lon = _sim_get_coords(victim.branch_code, rng)
        events.append(AuthEvent(
            event_id=eid, timestamp=ts, account_id=victim.account_id,
            action=AuthAction.OTP_FAIL, ip_address=victim_ip,
            geo_lat=lat, geo_lon=lon, device_fingerprint=_device_fingerprint(),
            user_agent_hash=_ua_hash(), success=False, checksum=cs,
        ))

    # Successful login from attacker's device/IP
    ts += rng.randint(30, 120)
    eid = _hex_id("AUTH")
    cs = compute_auth_checksum(eid, ts, victim.account_id, int(AuthAction.LOGIN), attacker_ip)
    lat, lon = _sim_get_coords(rng.choice(_BRANCH_CODES), rng)
    events.append(AuthEvent(
        event_id=eid, timestamp=ts, account_id=victim.account_id,
        action=AuthAction.LOGIN, ip_address=attacker_ip,
        geo_lat=lat, geo_lon=lon, device_fingerprint=attacker_fp,
        user_agent_hash=_ua_hash(), success=True, checksum=cs,
    ))

    # -- Phase 2: Fan-out (victim -> mules via UPI) --
    per_mule_base = total_amount_inr // mule_count
    mule_amounts: list[int] = []
    for i, mule in enumerate(mules):
        ts += rng.randint(30, 120)
        amount_inr = per_mule_base + rng.randint(-per_mule_base // 5, per_mule_base // 5)
        amount_inr = max(10_000, amount_inr)
        amount_paisa = amount_inr * 100
        mule_amounts.append(amount_paisa)

        txn_id = _hex_id("TXN")
        s_lat, s_lon = _sim_get_coords(victim.branch_code, rng)
        r_lat, r_lon = _sim_get_coords(mule.branch_code, rng)
        cs = compute_transaction_checksum(
            txn_id, ts, victim.account_id, mule.account_id,
            amount_paisa, int(Channel.UPI),
        )
        events.append(Transaction(
            txn_id=txn_id, timestamp=ts,
            sender_id=victim.account_id, receiver_id=mule.account_id,
            amount_paisa=amount_paisa, channel=Channel.UPI,
            sender_branch=victim.branch_code[:4], receiver_branch=mule.branch_code[:4],
            sender_geo_lat=s_lat, sender_geo_lon=s_lon,
            receiver_geo_lat=r_lat, receiver_geo_lon=r_lon,
            device_fingerprint=attacker_fp,
            sender_account_type=victim.account_type,
            receiver_account_type=mule.account_type,
            checksum=cs, fraud_label=FraudPattern.UPI_MULE_NETWORK,
        ))

    # -- Phase 3: Dispersal (each mule -> collector via IMPS) --
    for i, mule in enumerate(mules):
        ts += rng.randint(180, 600)  # 3-10 min after receiving
        fee_ratio = rng.uniform(0.01, 0.05)
        dispersal_paisa = int(mule_amounts[i] * (1 - fee_ratio))

        txn_id = _hex_id("TXN")
        s_lat, s_lon = _sim_get_coords(mule.branch_code, rng)
        r_lat, r_lon = _sim_get_coords(collector.branch_code, rng)
        cs = compute_transaction_checksum(
            txn_id, ts, mule.account_id, collector.account_id,
            dispersal_paisa, int(Channel.IMPS),
        )
        events.append(Transaction(
            txn_id=txn_id, timestamp=ts,
            sender_id=mule.account_id, receiver_id=collector.account_id,
            amount_paisa=dispersal_paisa, channel=Channel.IMPS,
            sender_branch=mule.branch_code[:4], receiver_branch=collector.branch_code[:4],
            sender_geo_lat=s_lat, sender_geo_lon=s_lon,
            receiver_geo_lat=r_lat, receiver_geo_lon=r_lon,
            device_fingerprint=_device_fingerprint(),
            sender_account_type=mule.account_type,
            receiver_account_type=collector.account_type,
            checksum=cs, fraud_label=FraudPattern.UPI_MULE_NETWORK,
        ))

    # -- Phase 4: Settlement (collector -> external via NEFT) --
    ts += rng.randint(600, 1800)
    total_collected = sum(
        int(mule_amounts[i] * (1 - rng.uniform(0.01, 0.05)))
        for i in range(len(mules))
    )
    ext_account = rng.choice([a for a in world.accounts if a.account_id != collector.account_id])
    msg_id = _hex_id("MSG")
    sender_ifsc = f"{UBI_IFSC_PREFIX}{collector.branch_code}"[:11].ljust(11, "0")
    receiver_ifsc = f"{UBI_IFSC_PREFIX}{ext_account.branch_code}"[:11].ljust(11, "0")
    cs = compute_interbank_checksum(
        msg_id, ts, sender_ifsc, receiver_ifsc, total_collected, int(Channel.NEFT),
    )
    s_lat, s_lon = _sim_get_coords(collector.branch_code, rng)
    events.append(InterbankMessage(
        msg_id=msg_id, timestamp=ts,
        sender_ifsc=sender_ifsc, receiver_ifsc=receiver_ifsc,
        sender_account=collector.account_id, receiver_account=ext_account.account_id,
        amount_paisa=total_collected, currency_code=356,
        channel=Channel.NEFT, message_type="N06",
        sender_geo_lat=s_lat, sender_geo_lon=s_lon,
        device_fingerprint=_device_fingerprint(), priority=0, checksum=cs,
    ))

    return events


# ==========================================================================
# Attack 2: Circular Laundering via Shell Accounts
# ==========================================================================

def generate_circular_laundering(
    world: WorldState,
    rng: random.Random,
    base_timestamp: int | None = None,
    shell_count: int = 5,
    hop_amount_inr: int = 25_00_000,
) -> list[Event]:
    """
    Circular Laundering via Shell Accounts.

    Money circles through N shell (CURRENT/INTERNAL) accounts via
    alternating RTGS/NEFT hops. Each hop deducts a 1-3% "consulting fee".
    The circle completes when funds return to the originating shell.

    Phases:
      1. Activation   -- shell accounts log in
      2. Circular hops -- Transaction + paired InterbankMessage per hop
      3. Fee collection -- each shell sends its fee to a collector
    """
    if base_timestamp is None:
        base_timestamp = int(time.time())

    # Prefer CURRENT/INTERNAL accounts for shells
    shell_candidates = [
        a for a in world.accounts
        if a.account_type in (AccountType.CURRENT, AccountType.INTERNAL)
    ]
    if len(shell_candidates) < shell_count:
        shell_candidates = world.accounts[:]
    shells = rng.sample(shell_candidates, min(shell_count, len(shell_candidates)))

    # Fee collector -- separate from shells
    fee_collector = rng.choice([a for a in world.accounts if a not in shells])

    events: list[Event] = []
    ts = base_timestamp

    # -- Phase 1: Activation (AuthEvents) --
    for shell in shells:
        ts += rng.randint(60, 300)
        ip = _random_indian_ip(rng)
        eid = _hex_id("AUTH")
        cs = compute_auth_checksum(eid, ts, shell.account_id, int(AuthAction.LOGIN), ip)
        lat, lon = _sim_get_coords(shell.branch_code, rng)
        events.append(AuthEvent(
            event_id=eid, timestamp=ts, account_id=shell.account_id,
            action=AuthAction.LOGIN, ip_address=ip,
            geo_lat=lat, geo_lon=lon, device_fingerprint=_device_fingerprint(),
            user_agent_hash=_ua_hash(), success=True, checksum=cs,
        ))

    # -- Phase 2: Circular hops --
    amount_paisa = hop_amount_inr * 100
    hop_fees: list[int] = []

    for i in range(len(shells)):
        sender = shells[i]
        receiver = shells[(i + 1) % len(shells)]
        ts += rng.randint(7200, 21600)  # 2-6 hours between hops

        fee_pct = rng.uniform(0.01, 0.03)
        fee_paisa = int(amount_paisa * fee_pct)
        hop_fees.append(fee_paisa)
        transfer_paisa = amount_paisa - fee_paisa

        channel = Channel.RTGS if i % 2 == 0 else Channel.NEFT
        # RTGS requires >= 2L INR
        if channel == Channel.RTGS and transfer_paisa < 2_00_000_00:
            channel = Channel.NEFT

        msg_type = "N01" if channel == Channel.RTGS else "N06"

        # Transaction
        txn_id = _hex_id("TXN")
        s_lat, s_lon = _sim_get_coords(sender.branch_code, rng)
        r_lat, r_lon = _sim_get_coords(receiver.branch_code, rng)
        cs = compute_transaction_checksum(
            txn_id, ts, sender.account_id, receiver.account_id,
            transfer_paisa, int(channel),
        )
        events.append(Transaction(
            txn_id=txn_id, timestamp=ts,
            sender_id=sender.account_id, receiver_id=receiver.account_id,
            amount_paisa=transfer_paisa, channel=channel,
            sender_branch=sender.branch_code[:4], receiver_branch=receiver.branch_code[:4],
            sender_geo_lat=s_lat, sender_geo_lon=s_lon,
            receiver_geo_lat=r_lat, receiver_geo_lon=r_lon,
            device_fingerprint=_device_fingerprint(),
            sender_account_type=sender.account_type,
            receiver_account_type=receiver.account_type,
            checksum=cs, fraud_label=FraudPattern.CIRCULAR_LAUNDERING,
        ))

        # Paired InterbankMessage
        msg_id = _hex_id("MSG")
        sender_ifsc = f"{UBI_IFSC_PREFIX}{sender.branch_code}"[:11].ljust(11, "0")
        receiver_ifsc = f"{UBI_IFSC_PREFIX}{receiver.branch_code}"[:11].ljust(11, "0")
        mcs = compute_interbank_checksum(
            msg_id, ts, sender_ifsc, receiver_ifsc, transfer_paisa, int(channel),
        )
        events.append(InterbankMessage(
            msg_id=msg_id, timestamp=ts,
            sender_ifsc=sender_ifsc, receiver_ifsc=receiver_ifsc,
            sender_account=sender.account_id, receiver_account=receiver.account_id,
            amount_paisa=transfer_paisa, currency_code=356,
            channel=channel, message_type=msg_type,
            sender_geo_lat=s_lat, sender_geo_lon=s_lon,
            device_fingerprint=_device_fingerprint(),
            priority=1 if channel == Channel.RTGS else 0,
            checksum=mcs,
        ))

        # Reduce amount for next hop
        amount_paisa = transfer_paisa

    # -- Phase 3: Fee collection --
    for i, shell in enumerate(shells):
        ts += rng.randint(1800, 7200)
        fee_paisa = hop_fees[i]
        if fee_paisa <= 0:
            continue

        txn_id = _hex_id("TXN")
        s_lat, s_lon = _sim_get_coords(shell.branch_code, rng)
        r_lat, r_lon = _sim_get_coords(fee_collector.branch_code, rng)
        cs = compute_transaction_checksum(
            txn_id, ts, shell.account_id, fee_collector.account_id,
            fee_paisa, int(Channel.NETBANKING),
        )
        events.append(Transaction(
            txn_id=txn_id, timestamp=ts,
            sender_id=shell.account_id, receiver_id=fee_collector.account_id,
            amount_paisa=fee_paisa, channel=Channel.NETBANKING,
            sender_branch=shell.branch_code[:4],
            receiver_branch=fee_collector.branch_code[:4],
            sender_geo_lat=s_lat, sender_geo_lon=s_lon,
            receiver_geo_lat=r_lat, receiver_geo_lon=r_lon,
            device_fingerprint=_device_fingerprint(),
            sender_account_type=shell.account_type,
            receiver_account_type=fee_collector.account_type,
            checksum=cs, fraud_label=FraudPattern.CIRCULAR_LAUNDERING,
        ))

    return events


# ==========================================================================
# Attack 3: Velocity Phishing Attack
# ==========================================================================

def generate_velocity_phishing(
    world: WorldState,
    rng: random.Random,
    base_timestamp: int | None = None,
    credential_attempts: int = 8,
    unauthorized_txns: int = 5,
    geo_spread: int = 4,
) -> list[Event]:
    """
    Velocity Phishing Attack.

    An attacker phishes credentials then rapidly attempts logins from
    multiple geolocations with mismatched device fingerprints. After
    gaining access, unauthorized transactions originate from geographically
    impossible locations (e.g., Mumbai and Kolkata within minutes).

    Phases:
      1. Credential theft -- N failed logins from scattered geolocations
      2. Breach          -- successful login + OTP from new location
      3. Unauthorized drain -- M transactions from impossible-travel locations
      4. Lockout         -- password change to lock out victim
    """
    if base_timestamp is None:
        base_timestamp = int(time.time())

    victim = rng.choice(world.accounts)
    receivers = rng.sample(
        [a for a in world.accounts if a.account_id != victim.account_id],
        min(unauthorized_txns, len(world.accounts) - 1),
    )

    # Pick distinct cities for impossible-travel
    city_branches = _pick_distinct_cities(rng, max(geo_spread, credential_attempts))

    events: list[Event] = []
    ts = base_timestamp

    # -- Phase 1: Credential theft (failed logins from scattered locations) --
    for i in range(credential_attempts):
        ts += rng.randint(5, 30)
        branch = city_branches[i % len(city_branches)]
        ip = _random_indian_ip(rng)
        eid = _hex_id("AUTH")
        lat, lon = _sim_get_coords(branch, rng)
        cs = compute_auth_checksum(
            eid, ts, victim.account_id, int(AuthAction.FAILED_LOGIN), ip,
        )
        events.append(AuthEvent(
            event_id=eid, timestamp=ts, account_id=victim.account_id,
            action=AuthAction.FAILED_LOGIN, ip_address=ip,
            geo_lat=lat, geo_lon=lon, device_fingerprint=_device_fingerprint(),
            user_agent_hash=_ua_hash(), success=False, checksum=cs,
        ))

    # -- Phase 2: Breach (success from yet another location) --
    ts += rng.randint(30, 120)
    breach_branch = rng.choice(city_branches)
    breach_ip = _random_indian_ip(rng)
    breach_fp = _device_fingerprint()

    eid = _hex_id("AUTH")
    lat, lon = _sim_get_coords(breach_branch, rng)
    cs = compute_auth_checksum(
        eid, ts, victim.account_id, int(AuthAction.LOGIN), breach_ip,
    )
    events.append(AuthEvent(
        event_id=eid, timestamp=ts, account_id=victim.account_id,
        action=AuthAction.LOGIN, ip_address=breach_ip,
        geo_lat=lat, geo_lon=lon, device_fingerprint=breach_fp,
        user_agent_hash=_ua_hash(), success=True, checksum=cs,
    ))

    # OTP verify
    ts += rng.randint(10, 30)
    eid = _hex_id("AUTH")
    cs = compute_auth_checksum(
        eid, ts, victim.account_id, int(AuthAction.OTP_VERIFY), breach_ip,
    )
    events.append(AuthEvent(
        event_id=eid, timestamp=ts, account_id=victim.account_id,
        action=AuthAction.OTP_VERIFY, ip_address=breach_ip,
        geo_lat=lat, geo_lon=lon, device_fingerprint=breach_fp,
        user_agent_hash=_ua_hash(), success=True, checksum=cs,
    ))

    # -- Phase 3: Unauthorized drain (impossible-travel transactions) --
    channels = [Channel.UPI, Channel.NETBANKING, Channel.IMPS]
    for i in range(unauthorized_txns):
        ts += rng.randint(60, 300)
        # Each from a DIFFERENT geolocation + device
        city_branch = city_branches[i % len(city_branches)]
        s_lat, s_lon = _sim_get_coords(city_branch, rng)

        rcv = receivers[i % len(receivers)]
        r_lat, r_lon = _sim_get_coords(rcv.branch_code, rng)

        amount_inr = rng.randint(20_000, 2_00_000)
        amount_paisa = amount_inr * 100
        channel = rng.choice(channels)

        txn_id = _hex_id("TXN")
        cs = compute_transaction_checksum(
            txn_id, ts, victim.account_id, rcv.account_id,
            amount_paisa, int(channel),
        )
        events.append(Transaction(
            txn_id=txn_id, timestamp=ts,
            sender_id=victim.account_id, receiver_id=rcv.account_id,
            amount_paisa=amount_paisa, channel=channel,
            sender_branch=victim.branch_code[:4],
            receiver_branch=rcv.branch_code[:4],
            sender_geo_lat=s_lat, sender_geo_lon=s_lon,
            receiver_geo_lat=r_lat, receiver_geo_lon=r_lon,
            device_fingerprint=_device_fingerprint(),
            sender_account_type=victim.account_type,
            receiver_account_type=rcv.account_type,
            checksum=cs, fraud_label=FraudPattern.VELOCITY_PHISHING,
        ))

    # -- Phase 4: Lockout (password change) --
    ts += rng.randint(30, 120)
    lockout_ip = _random_indian_ip(rng)
    eid = _hex_id("AUTH")
    lat, lon = _sim_get_coords(rng.choice(city_branches), rng)
    cs = compute_auth_checksum(
        eid, ts, victim.account_id, int(AuthAction.PASSWORD_CHANGE), lockout_ip,
    )
    events.append(AuthEvent(
        event_id=eid, timestamp=ts, account_id=victim.account_id,
        action=AuthAction.PASSWORD_CHANGE, ip_address=lockout_ip,
        geo_lat=lat, geo_lon=lon, device_fingerprint=_device_fingerprint(),
        user_agent_hash=_ua_hash(), success=True, checksum=cs,
    ))

    return events


# ==========================================================================
# Attack 4: Bangladesh Bank SWIFT Night Heist
# ==========================================================================

def generate_swift_heist(
    world: WorldState,
    rng: random.Random,
    base_timestamp: int | None = None,
    transfer_amount_inr: int = 8_10_00_000,
    recon_logins: int = 3,
) -> list[Event]:
    """
    Bangladesh Bank SWIFT Night Heist scenario.

    Simulates a SWIFT-based high-value international wire fraud:
      - 2:30 AM timing (off-hours to evade manual review)
      - ₹8.1 Cr transfer to a NEW beneficiary never seen before
      - Unrecognized device fingerprint
      - Geolocation 800 km from the account's usual branch
      - Expected outcome: risk score 90+, immediate BLOCK, FMR queued

    Phases:
      1. Reconnaissance -- off-hours logins from unusual location
      2. Credential Setup -- OTP verify from unrecognized device
      3. SWIFT Transfer -- ₹8.1 Cr single wire to new beneficiary
      4. Cover Tracks -- session timeout / quick logout
    """
    if base_timestamp is None:
        # Default: 2:30 AM today
        import datetime as _dt
        now = _dt.datetime.now()
        base_timestamp = int(
            now.replace(hour=2, minute=30, second=0, microsecond=0).timestamp()
        )

    # Pick attacker (origin) and a NEW beneficiary
    accounts = rng.sample(world.accounts, min(4, len(world.accounts)))
    origin = accounts[0]
    beneficiary = accounts[1]

    # Attacker operates from ~800 km away from origin branch
    origin_lat, origin_lon = _sim_get_coords(origin.branch_code, rng)
    # Shift ~800 km ≈ ~7.2 degrees latitude
    attacker_lat = round(origin_lat + rng.choice([-7.2, 7.2]) + rng.uniform(-0.5, 0.5), 6)
    attacker_lon = round(origin_lon + rng.uniform(-1.0, 1.0), 6)
    # Clamp to India bounds
    attacker_lat = max(8.0, min(35.0, attacker_lat))
    attacker_lon = max(69.0, min(97.0, attacker_lon))

    attacker_fp = _device_fingerprint()  # unrecognized device
    attacker_ip = _random_indian_ip(rng)

    events: list[Event] = []
    ts = base_timestamp

    # -- Phase 1: Reconnaissance (off-hours logins) --
    for _ in range(recon_logins):
        ts += rng.randint(60, 300)
        eid = _hex_id("AUTH")
        cs = compute_auth_checksum(
            eid, ts, origin.account_id, int(AuthAction.LOGIN), attacker_ip,
        )
        events.append(AuthEvent(
            event_id=eid, timestamp=ts, account_id=origin.account_id,
            action=AuthAction.LOGIN, ip_address=attacker_ip,
            geo_lat=attacker_lat, geo_lon=attacker_lon,
            device_fingerprint=attacker_fp,
            user_agent_hash=_ua_hash(), success=True, checksum=cs,
        ))
        # Quick logout after recon
        ts += rng.randint(10, 60)
        eid2 = _hex_id("AUTH")
        cs2 = compute_auth_checksum(
            eid2, ts, origin.account_id, int(AuthAction.LOGOUT), attacker_ip,
        )
        events.append(AuthEvent(
            event_id=eid2, timestamp=ts, account_id=origin.account_id,
            action=AuthAction.LOGOUT, ip_address=attacker_ip,
            geo_lat=attacker_lat, geo_lon=attacker_lon,
            device_fingerprint=attacker_fp,
            user_agent_hash=_ua_hash(), success=True, checksum=cs2,
        ))

    # -- Phase 2: Credential Setup (final login + OTP) --
    ts += rng.randint(60, 180)
    eid = _hex_id("AUTH")
    cs = compute_auth_checksum(
        eid, ts, origin.account_id, int(AuthAction.LOGIN), attacker_ip,
    )
    events.append(AuthEvent(
        event_id=eid, timestamp=ts, account_id=origin.account_id,
        action=AuthAction.LOGIN, ip_address=attacker_ip,
        geo_lat=attacker_lat, geo_lon=attacker_lon,
        device_fingerprint=attacker_fp,
        user_agent_hash=_ua_hash(), success=True, checksum=cs,
    ))

    ts += rng.randint(15, 45)
    eid = _hex_id("AUTH")
    cs = compute_auth_checksum(
        eid, ts, origin.account_id, int(AuthAction.OTP_VERIFY), attacker_ip,
    )
    events.append(AuthEvent(
        event_id=eid, timestamp=ts, account_id=origin.account_id,
        action=AuthAction.OTP_VERIFY, ip_address=attacker_ip,
        geo_lat=attacker_lat, geo_lon=attacker_lon,
        device_fingerprint=attacker_fp,
        user_agent_hash=_ua_hash(), success=True, checksum=cs,
    ))

    # -- Phase 3: SWIFT Transfer (₹8.1 Cr single wire) --
    ts += rng.randint(30, 90)
    amount_paisa = transfer_amount_inr * 100
    txn_id = _hex_id("TXN")

    r_lat, r_lon = _sim_get_coords(beneficiary.branch_code, rng)
    cs = compute_transaction_checksum(
        txn_id, ts, origin.account_id, beneficiary.account_id,
        amount_paisa, int(Channel.SWIFT),
    )
    events.append(Transaction(
        txn_id=txn_id, timestamp=ts,
        sender_id=origin.account_id,
        receiver_id=beneficiary.account_id,
        amount_paisa=amount_paisa,
        channel=Channel.SWIFT,
        sender_branch=origin.branch_code[:4],
        receiver_branch=beneficiary.branch_code[:4],
        sender_geo_lat=attacker_lat,
        sender_geo_lon=attacker_lon,
        receiver_geo_lat=r_lat,
        receiver_geo_lon=r_lon,
        device_fingerprint=attacker_fp,
        sender_account_type=origin.account_type,
        receiver_account_type=beneficiary.account_type,
        checksum=cs,
        fraud_label=FraudPattern.SWIFT_HEIST,
    ))

    # Also create an interbank SWIFT message (MT103)
    ts += rng.randint(2, 10)
    msg_id = _hex_id("MSG")
    ib_cs = compute_interbank_checksum(
        msg_id, ts, origin.account_id, beneficiary.account_id,
        amount_paisa, "MT103",
    )
    events.append(InterbankMessage(
        msg_id=msg_id, timestamp=ts,
        sender_account=origin.account_id,
        receiver_account=beneficiary.account_id,
        amount_paisa=amount_paisa,
        message_type="MT103",
        sender_ifsc=f"{UBI_IFSC_PREFIX}{origin.branch_code[:4]}",
        receiver_ifsc=f"{UBI_IFSC_PREFIX}{beneficiary.branch_code[:4]}",
        reference_id=txn_id,
        checksum=ib_cs,
    ))

    # -- Phase 4: Cover Tracks (quick logout) --
    ts += rng.randint(10, 30)
    eid = _hex_id("AUTH")
    cs = compute_auth_checksum(
        eid, ts, origin.account_id, int(AuthAction.LOGOUT), attacker_ip,
    )
    events.append(AuthEvent(
        event_id=eid, timestamp=ts, account_id=origin.account_id,
        action=AuthAction.LOGOUT, ip_address=attacker_ip,
        geo_lat=attacker_lat, geo_lon=attacker_lon,
        device_fingerprint=attacker_fp,
        user_agent_hash=_ua_hash(), success=True, checksum=cs,
    ))

    return events


# ==========================================================================
# iDEA 2.0 PS3: Fund Flow Tracking Scenarios
# ==========================================================================

PS3_SCENARIO_DETAILS: dict[str, dict[str, object]] = {
    "rapid_layering": {
        "label": "Rapid Layering Through Multiple Accounts",
        "typologies": ["LAYERING"],
        "expected_indicators": [
            "Funds move through 5+ accounts within a compressed time window",
            "Amounts decay slightly at each hop, consistent with fee skimming",
            "Mixed channels obscure a single end-to-end fund journey",
        ],
        "recommended_actions": [
            "Trace downstream beneficiaries and freeze terminal accounts",
            "Escalate as STR candidate with graph path evidence",
        ],
    },
    "round_tripping": {
        "label": "Circular Transactions / Round-Tripping",
        "typologies": ["ROUND_TRIPPING"],
        "expected_indicators": [
            "Originating account receives funds back after a closed loop",
            "Shell-like current accounts form a directed cycle",
            "Transaction purpose is inconsistent with circular movement",
        ],
        "recommended_actions": [
            "Escalate cycle participants for enhanced due diligence",
            "Attach circular graph evidence to FIU package",
        ],
    },
    "structuring": {
        "label": "Structuring Below Reporting Thresholds",
        "typologies": ["STRUCTURING"],
        "expected_indicators": [
            "Repeated transfers remain just below INR 10 lakh threshold",
            "Multiple originators converge on a single collector",
            "Activity clusters in a short window across channels",
        ],
        "recommended_actions": [
            "Aggregate linked transfers before regulatory assessment",
            "Flag collector and repeated originators for review",
        ],
    },
    "dormant_activation": {
        "label": "Dormant Account Activation for High-Value Transfer",
        "typologies": ["DORMANT_ACTIVATION"],
        "expected_indicators": [
            "Dormant account logs in after long inactivity",
            "OTP verification precedes a high-value outward transfer",
            "New device fingerprint appears immediately before transfer",
        ],
        "recommended_actions": [
            "Place account under hold pending branch verification",
            "Preserve auth trail and transaction evidence",
        ],
    },
    "profile_mismatch": {
        "label": "Declared Profile vs Actual Fund Movement Mismatch",
        "typologies": ["PROFILE_MISMATCH"],
        "expected_indicators": [
            "Savings-style account behaves like high-volume business account",
            "Large outward transfers go to current/internal accounts",
            "Observed behavior deviates from declared low-risk profile",
        ],
        "recommended_actions": [
            "Trigger customer profile refresh and relationship manager review",
            "Attach behavior mismatch narrative to case package",
        ],
    },
}


def _ps3_hex_id(prefix: str, rng: random.Random, marker: str) -> str:
    """Deterministic hex ID for PS3 replay scenarios."""
    raw = f"{prefix}-{marker}-{rng.random():.12f}-{rng.randint(0, 1_000_000)}"
    return f"{prefix}{hashlib.sha256(raw.encode()).hexdigest()[:12].upper()}"


def _ps3_device(rng: random.Random, marker: str) -> str:
    raw = f"ps3-device-{marker}-{rng.random():.12f}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _ps3_ip(rng: random.Random, marker: str) -> str:
    first_octet = rng.choice([49, 59, 103, 106, 117, 122, 157, 182, 203])
    suffix = int(hashlib.sha256(marker.encode()).hexdigest()[:6], 16)
    return f"{first_octet}.{suffix % 250}.{rng.randint(1, 250)}.{rng.randint(1, 254)}"


def _pick_ps3_accounts(
    world: WorldState,
    rng: random.Random,
    count: int,
    prefer: tuple[AccountType, ...] | None = None,
    exclude_ids: set[str] | None = None,
) -> list:
    exclude_ids = exclude_ids or set()
    candidates = [
        acct for acct in world.accounts
        if acct.account_id not in exclude_ids
        and (prefer is None or acct.account_type in prefer)
    ]
    if len(candidates) < count:
        candidates = [
            acct for acct in world.accounts
            if acct.account_id not in exclude_ids
        ]
    if len(candidates) < count:
        raise ValueError("World state does not have enough accounts for PS3 scenario")
    return rng.sample(candidates, count)


def _ps3_txn(
    rng: random.Random,
    marker: str,
    timestamp: int,
    sender,
    receiver,
    amount_inr: int,
    channel: Channel,
    pattern: FraudPattern,
    device_fingerprint: str | None = None,
) -> Transaction:
    txn_id = _ps3_hex_id("TXN", rng, marker)
    amount_paisa = amount_inr * 100
    s_lat, s_lon = _sim_get_coords(sender.branch_code, rng)
    r_lat, r_lon = _sim_get_coords(receiver.branch_code, rng)
    checksum = compute_transaction_checksum(
        txn_id,
        timestamp,
        sender.account_id,
        receiver.account_id,
        amount_paisa,
        int(channel),
    )
    return Transaction(
        txn_id=txn_id,
        timestamp=timestamp,
        sender_id=sender.account_id,
        receiver_id=receiver.account_id,
        amount_paisa=amount_paisa,
        channel=channel,
        sender_branch=sender.branch_code[:4],
        receiver_branch=receiver.branch_code[:4],
        sender_geo_lat=s_lat,
        sender_geo_lon=s_lon,
        receiver_geo_lat=r_lat,
        receiver_geo_lon=r_lon,
        device_fingerprint=device_fingerprint or _ps3_device(rng, marker),
        sender_account_type=sender.account_type,
        receiver_account_type=receiver.account_type,
        checksum=checksum,
        fraud_label=pattern,
    )


def _ps3_auth(
    rng: random.Random,
    marker: str,
    timestamp: int,
    account,
    action: AuthAction,
    success: bool = True,
    device_fingerprint: str | None = None,
) -> AuthEvent:
    event_id = _ps3_hex_id("AUTH", rng, marker)
    ip = _ps3_ip(rng, marker)
    lat, lon = _sim_get_coords(account.branch_code, rng)
    checksum = compute_auth_checksum(
        event_id,
        timestamp,
        account.account_id,
        int(action),
        ip,
    )
    return AuthEvent(
        event_id=event_id,
        timestamp=timestamp,
        account_id=account.account_id,
        action=action,
        ip_address=ip,
        geo_lat=lat,
        geo_lon=lon,
        device_fingerprint=device_fingerprint or _ps3_device(rng, marker),
        user_agent_hash=hashlib.sha256(marker.encode()).hexdigest()[:16],
        success=success,
        checksum=checksum,
    )


def generate_ps3_rapid_layering(
    world: WorldState,
    rng: random.Random,
    base_timestamp: int | None = None,
    hop_count: int = 5,
    total_amount_inr: int = 18_50_000,
) -> list[Event]:
    """PS3 rapid layering: a compressed multi-hop fund trail."""
    if base_timestamp is None:
        base_timestamp = int(time.time())

    accounts = _pick_ps3_accounts(world, rng, hop_count + 2)
    origin = accounts[0]
    path = accounts[1:]
    events: list[Event] = []
    device = _ps3_device(rng, "rapid-layering")
    amount_inr = total_amount_inr
    timestamp = base_timestamp
    channels = [Channel.UPI, Channel.IMPS, Channel.NEFT, Channel.NETBANKING, Channel.IMPS, Channel.NEFT]

    current = origin
    for idx, receiver in enumerate(path):
        timestamp += rng.randint(60, 240)
        events.append(_ps3_txn(
            rng,
            f"rapid-layering-{idx}",
            timestamp,
            current,
            receiver,
            amount_inr,
            channels[idx % len(channels)],
            FraudPattern.LAYERING,
            device,
        ))
        current = receiver
        amount_inr = max(50_000, int(amount_inr * rng.uniform(0.86, 0.94)))

    return events


def generate_ps3_round_tripping(
    world: WorldState,
    rng: random.Random,
    base_timestamp: int | None = None,
    ring_size: int = 5,
    hop_amount_inr: int = 42_00_000,
) -> list[Event]:
    """PS3 round-tripping: a closed loop returning funds to origin."""
    if base_timestamp is None:
        base_timestamp = int(time.time())

    shells = _pick_ps3_accounts(
        world,
        rng,
        ring_size,
        prefer=(AccountType.CURRENT, AccountType.INTERNAL),
    )
    events: list[Event] = []
    timestamp = base_timestamp
    amount_inr = hop_amount_inr
    channels = [Channel.RTGS, Channel.NEFT, Channel.IMPS, Channel.NETBANKING]

    for idx, sender in enumerate(shells):
        receiver = shells[(idx + 1) % len(shells)]
        timestamp += rng.randint(900, 3600)
        events.append(_ps3_txn(
            rng,
            f"round-trip-{idx}",
            timestamp,
            sender,
            receiver,
            amount_inr,
            channels[idx % len(channels)],
            FraudPattern.ROUND_TRIPPING,
        ))
        amount_inr = max(2_00_000, int(amount_inr * rng.uniform(0.97, 0.995)))

    return events


def generate_ps3_structuring(
    world: WorldState,
    rng: random.Random,
    base_timestamp: int | None = None,
    originator_count: int = 4,
    transfers_per_originator: int = 2,
) -> list[Event]:
    """PS3 structuring: repeated sub-threshold transfers to a collector."""
    if base_timestamp is None:
        base_timestamp = int(time.time())

    accounts = _pick_ps3_accounts(world, rng, originator_count + 1)
    originators = accounts[:originator_count]
    collector = accounts[-1]
    events: list[Event] = []
    timestamp = base_timestamp
    channels = [Channel.UPI, Channel.IMPS, Channel.NETBANKING]

    for i, originator in enumerate(originators):
        for j in range(transfers_per_originator):
            timestamp += rng.randint(120, 900)
            amount_inr = rng.randint(9_40_000, 9_95_000)
            events.append(_ps3_txn(
                rng,
                f"structuring-{i}-{j}",
                timestamp,
                originator,
                collector,
                amount_inr,
                channels[(i + j) % len(channels)],
                FraudPattern.STRUCTURING,
            ))

    return events


def generate_ps3_dormant_activation(
    world: WorldState,
    rng: random.Random,
    base_timestamp: int | None = None,
    transfer_amount_inr: int = 38_00_000,
) -> list[Event]:
    """PS3 dormant activation: auth trail followed by high-value transfer."""
    if base_timestamp is None:
        base_timestamp = int(time.time())

    dormant, beneficiary = _pick_ps3_accounts(world, rng, 2)
    timestamp = base_timestamp
    device = _ps3_device(rng, "dormant-activation-new-device")
    events: list[Event] = []

    timestamp += 30
    events.append(_ps3_auth(
        rng,
        "dormant-login-after-180-days",
        timestamp,
        dormant,
        AuthAction.LOGIN,
        True,
        device,
    ))
    timestamp += 45
    events.append(_ps3_auth(
        rng,
        "dormant-otp-verify-new-device",
        timestamp,
        dormant,
        AuthAction.OTP_VERIFY,
        True,
        device,
    ))
    timestamp += 90
    events.append(_ps3_txn(
        rng,
        "dormant-high-value-transfer",
        timestamp,
        dormant,
        beneficiary,
        transfer_amount_inr,
        Channel.RTGS,
        FraudPattern.DORMANT_ACTIVATION,
        device,
    ))

    return events


def generate_ps3_profile_mismatch(
    world: WorldState,
    rng: random.Random,
    base_timestamp: int | None = None,
    transfer_count: int = 6,
) -> list[Event]:
    """PS3 profile mismatch: savings profile behaves like business routing."""
    if base_timestamp is None:
        base_timestamp = int(time.time())

    origin = _pick_ps3_accounts(
        world,
        rng,
        1,
        prefer=(AccountType.SAVINGS, AccountType.RECURRING_DEPOSIT),
    )[0]
    receivers = _pick_ps3_accounts(
        world,
        rng,
        transfer_count,
        prefer=(AccountType.CURRENT, AccountType.INTERNAL),
        exclude_ids={origin.account_id},
    )
    events: list[Event] = []
    timestamp = base_timestamp
    channels = [Channel.NEFT, Channel.RTGS, Channel.NETBANKING, Channel.IMPS]

    for idx, receiver in enumerate(receivers):
        timestamp += rng.randint(600, 2400)
        amount_inr = rng.randint(1_75_000, 7_50_000)
        events.append(_ps3_txn(
            rng,
            f"profile-mismatch-{idx}",
            timestamp,
            origin,
            receiver,
            amount_inr,
            channels[idx % len(channels)],
            FraudPattern.PROFILE_MISMATCH,
        ))

    return events


def generate_ps3_scenario(
    scenario: str,
    world: WorldState,
    rng: random.Random,
    base_timestamp: int | None = None,
    intensity: str = "demo",
) -> list[Event]:
    """Dispatch a named PS3 scenario to its deterministic generator."""
    scale = intensity == "scale"
    if scenario == "rapid_layering":
        return generate_ps3_rapid_layering(
            world,
            rng,
            base_timestamp,
            hop_count=10 if scale else 5,
            total_amount_inr=45_00_000 if scale else 18_50_000,
        )
    if scenario == "round_tripping":
        return generate_ps3_round_tripping(
            world,
            rng,
            base_timestamp,
            ring_size=8 if scale else 5,
            hop_amount_inr=75_00_000 if scale else 42_00_000,
        )
    if scenario == "structuring":
        return generate_ps3_structuring(
            world,
            rng,
            base_timestamp,
            originator_count=12 if scale else 4,
            transfers_per_originator=4 if scale else 2,
        )
    if scenario == "dormant_activation":
        return generate_ps3_dormant_activation(
            world,
            rng,
            base_timestamp,
            transfer_amount_inr=95_00_000 if scale else 38_00_000,
        )
    if scenario == "profile_mismatch":
        return generate_ps3_profile_mismatch(
            world,
            rng,
            base_timestamp,
            transfer_count=18 if scale else 6,
        )
    raise ValueError(
        f"Unknown PS3 scenario '{scenario}'. Available: {list(PS3_SCENARIO_DETAILS)}"
    )


# ==========================================================================
# Helpers
# ==========================================================================

def get_account_ids(events: list[Event]) -> list[str]:
    """Extract all account IDs from a list of events."""
    ids: set[str] = set()
    for e in events:
        if isinstance(e, Transaction):
            ids.add(e.sender_id)
            ids.add(e.receiver_id)
        elif isinstance(e, InterbankMessage):
            ids.add(e.sender_account)
            ids.add(e.receiver_account)
        elif isinstance(e, AuthEvent):
            ids.add(e.account_id)
    return sorted(ids)
