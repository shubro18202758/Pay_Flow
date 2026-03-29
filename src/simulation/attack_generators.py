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
