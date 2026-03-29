"""
PayFlow — Synthetic Transaction Generator
===========================================
Generates realistic Indian banking transaction streams with embedded fraud
patterns for all 5 categories from the problem statement:

  1. Rapid Layering — money hops through N accounts within minutes
  2. Round-Tripping — circular A→B→C→...→A flows
  3. Structuring (Smurfing) — amounts clustered just below ₹10L CTR threshold
  4. Dormant Account Activation — long-idle accounts suddenly transacting
  5. Profile-Behavior Mismatch — volumes inconsistent with declared KYC profile

The generator maintains a "world state" of synthetic accounts, branches,
and customer profiles that persist across the stream, ensuring consistency.
"""

from __future__ import annotations

import hashlib
import math
import os
import random
import time
from dataclasses import dataclass, field
from typing import AsyncIterator, Iterator

from config.settings import FRAUD_THRESHOLDS
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

# ── Deterministic Seeding ─────────────────────────────────────────────────────

_RNG = random.Random(42)  # Separate RNG for reproducibility without affecting global state


# ── Indian Banking Reference Data ─────────────────────────────────────────────

# Union Bank IFSC prefix
UBI_IFSC_PREFIX = "UBIN0"

# Major branch codes (first 4 digits after UBIN0)
_BRANCH_CODES = [
    "530042", "532810", "540094", "541693", "550037",  # Andhra / Telangana
    "560003", "560018", "570011", "580024",             # Karnataka
    "600004", "600015", "610023", "620031",             # Tamil Nadu
    "400001", "400015", "411005", "440002", "500001",   # Maharashtra
    "110002", "110015", "110042",                        # Delhi NCR
    "700001", "700015", "711004",                        # West Bengal
    "380001", "380009", "395001",                        # Gujarat
    "302001", "301001",                                  # Rajasthan
    "226001", "208001", "211001",                        # UP
    "800001", "831001",                                  # Bihar / Jharkhand
    "682001", "695001",                                  # Kerala
]

# Indian city coordinates (lat, lon) mapped to branch regions
_CITY_COORDS: dict[str, tuple[float, float]] = {
    "530042": (17.6868, 83.2185),  # Visakhapatnam
    "560003": (12.9716, 77.5946),  # Bangalore
    "600004": (13.0827, 80.2707),  # Chennai
    "400001": (18.9388, 72.8354),  # Mumbai
    "110002": (28.6139, 77.2090),  # Delhi
    "700001": (22.5726, 88.3639),  # Kolkata
    "380001": (23.0225, 72.5714),  # Ahmedabad
    "302001": (26.9124, 75.7873),  # Jaipur
    "226001": (26.8467, 80.9462),  # Lucknow
    "800001": (25.6093, 85.1376),  # Patna
    "682001": (9.9312, 76.2673),   # Kochi
}

_FIRST_NAMES = [
    "Aarav", "Vivaan", "Aditya", "Sai", "Arjun", "Rahul", "Priya", "Ananya",
    "Diya", "Ishaan", "Krishna", "Meera", "Rohan", "Sneha", "Tanvi", "Vikram",
    "Neha", "Karthik", "Pooja", "Amit", "Sunita", "Rajesh", "Deepak", "Kavita",
    "Suresh", "Lakshmi", "Ramesh", "Geeta", "Mohan", "Anjali",
]

_LAST_NAMES = [
    "Sharma", "Verma", "Patel", "Singh", "Kumar", "Gupta", "Reddy", "Nair",
    "Joshi", "Iyer", "Rao", "Das", "Mukherjee", "Chatterjee", "Banerjee",
    "Desai", "Pillai", "Bhat", "Agarwal", "Mishra", "Tiwari", "Yadav",
]


# ── Account World State ──────────────────────────────────────────────────────

@dataclass
class SyntheticAccount:
    account_id: str
    branch_code: str
    account_type: AccountType
    holder_name: str
    declared_monthly_income_paisa: int  # KYC declared income
    last_active_timestamp: int          # for dormant detection
    is_dormant: bool = False
    cumulative_30d_paisa: int = 0       # rolling 30-day volume


@dataclass
class WorldState:
    """Persistent state of all synthetic accounts in the simulated bank."""
    accounts: list[SyntheticAccount] = field(default_factory=list)
    dormant_accounts: list[SyntheticAccount] = field(default_factory=list)
    _account_index: dict[str, SyntheticAccount] = field(default_factory=dict)

    def get(self, account_id: str) -> SyntheticAccount | None:
        return self._account_index.get(account_id)

    def register(self, account: SyntheticAccount) -> None:
        self.accounts.append(account)
        self._account_index[account.account_id] = account


def _generate_account_id(branch_code: str) -> str:
    """Generate a 14-digit account number: 4-digit branch + 10-digit unique."""
    suffix = "".join([str(_RNG.randint(0, 9)) for _ in range(10)])
    return f"{branch_code[:4]}{suffix}"


def _get_coords(branch_code: str) -> tuple[float, float]:
    """Get approximate coordinates for a branch, with jitter."""
    base_key = branch_code[:6]
    if base_key in _CITY_COORDS:
        lat, lon = _CITY_COORDS[base_key]
    else:
        # Pick random Indian coordinates
        lat = _RNG.uniform(8.0, 35.0)
        lon = _RNG.uniform(69.0, 97.0)
    # Add small jitter (±0.05°) for within-city variation
    return (
        round(lat + _RNG.uniform(-0.05, 0.05), 6),
        round(lon + _RNG.uniform(-0.05, 0.05), 6),
    )


def _device_fingerprint() -> str:
    """Generate a 16-char truncated SHA-256 fingerprint."""
    raw = os.urandom(16)
    return hashlib.sha256(raw).hexdigest()[:16]


def _hex_id(prefix: str) -> str:
    """Generate a prefixed 12-char hex ID (e.g., TXN-a1b2c3d4e5f6)."""
    return f"{prefix}-{os.urandom(6).hex()}"


def build_world(num_accounts: int = 2000, dormant_ratio: float = 0.08) -> WorldState:
    """
    Bootstrap the synthetic bank world with N accounts.
    ~8% are marked dormant (inactive >180 days) for pattern #4 testing.
    """
    world = WorldState()
    now = int(time.time())

    for i in range(num_accounts):
        branch = _RNG.choice(_BRANCH_CODES)
        acct_type = _RNG.choices(
            [AccountType.SAVINGS, AccountType.CURRENT, AccountType.LOAN,
             AccountType.NRE, AccountType.OVERDRAFT],
            weights=[50, 25, 10, 5, 10],
        )[0]

        # KYC-declared monthly income (₹15K – ₹50L, log-normal distribution)
        income_inr = int(math.exp(_RNG.gauss(10.5, 1.2)))  # median ~₹36K
        income_inr = max(15_000, min(income_inr, 50_00_000))

        is_dormant = i < int(num_accounts * dormant_ratio)
        last_active = now - (_RNG.randint(200, 700) * 86400 if is_dormant else _RNG.randint(0, 30) * 86400)

        first = _RNG.choice(_FIRST_NAMES)
        last = _RNG.choice(_LAST_NAMES)

        account = SyntheticAccount(
            account_id=_generate_account_id(branch),
            branch_code=branch,
            account_type=acct_type,
            holder_name=f"{first} {last}",
            declared_monthly_income_paisa=income_inr * 100,
            last_active_timestamp=last_active,
            is_dormant=is_dormant,
        )
        world.register(account)
        if is_dormant:
            world.dormant_accounts.append(account)

    return world


# ── Transaction ID Counter ────────────────────────────────────────────────────

_txn_counter = 0


def _next_txn_id() -> str:
    global _txn_counter
    _txn_counter += 1
    return _hex_id("TXN")


# ── Normal Transaction Generator ─────────────────────────────────────────────

def _generate_normal_transaction(
    world: WorldState,
    base_timestamp: int,
) -> Transaction:
    """Generate a single legitimate transaction."""
    sender = _RNG.choice(world.accounts)
    receiver = _RNG.choice(world.accounts)
    while receiver.account_id == sender.account_id:
        receiver = _RNG.choice(world.accounts)

    # Amount: log-normal, median ~₹5,000 for savings, ~₹50,000 for current
    if sender.account_type == AccountType.CURRENT:
        amount_inr = int(math.exp(_RNG.gauss(10.8, 1.5)))
    else:
        amount_inr = int(math.exp(_RNG.gauss(8.5, 1.3)))
    amount_inr = max(100, min(amount_inr, 50_00_000))
    amount_paisa = amount_inr * 100

    channel = _RNG.choices(
        [Channel.UPI, Channel.NETBANKING, Channel.MOBILE, Channel.NEFT,
         Channel.RTGS, Channel.BRANCH, Channel.ATM, Channel.IMPS],
        weights=[35, 20, 15, 10, 5, 5, 5, 5],
    )[0]

    # RTGS requires >₹2L
    if channel == Channel.RTGS and amount_inr < 2_00_000:
        amount_inr = _RNG.randint(2_00_000, 10_00_000)
        amount_paisa = amount_inr * 100

    ts = base_timestamp + _RNG.randint(0, 86400)
    s_lat, s_lon = _get_coords(sender.branch_code)
    r_lat, r_lon = _get_coords(receiver.branch_code)
    txn_id = _next_txn_id()

    checksum = compute_transaction_checksum(
        txn_id, ts, sender.account_id, receiver.account_id, amount_paisa, int(channel),
    )

    return Transaction(
        txn_id=txn_id,
        timestamp=ts,
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
        device_fingerprint=_device_fingerprint(),
        sender_account_type=sender.account_type,
        receiver_account_type=receiver.account_type,
        checksum=checksum,
        fraud_label=FraudPattern.NONE,
    )


# ── Fraud Pattern #1: Rapid Layering ─────────────────────────────────────────

def generate_layering_chain(
    world: WorldState,
    base_timestamp: int,
    chain_length: int = 5,
    amount_paisa: int | None = None,
) -> list[Transaction]:
    """
    A→B→C→D→E within minutes. Same amount (±small jitter) cascades
    through a chain of accounts with very short inter-hop delays.
    """
    accounts = _RNG.sample(world.accounts, min(chain_length + 1, len(world.accounts)))
    if amount_paisa is None:
        amount_paisa = _RNG.randint(50_000_00, 9_00_000_00)  # ₹50K – ₹9L

    txns = []
    ts = base_timestamp
    fp = _device_fingerprint()  # Same device across chain = strong indicator

    for i in range(len(accounts) - 1):
        sender = accounts[i]
        receiver = accounts[i + 1]

        # Slight amount variation (laundering "fee" skimming)
        jittered_amount = int(amount_paisa * _RNG.uniform(0.97, 1.0))
        ts += _RNG.randint(30, 300)  # 30 sec – 5 min between hops

        txn_id = _next_txn_id()
        checksum = compute_transaction_checksum(
            txn_id, ts, sender.account_id, receiver.account_id,
            jittered_amount, int(Channel.IMPS),
        )

        txns.append(Transaction(
            txn_id=txn_id, timestamp=ts,
            sender_id=sender.account_id, receiver_id=receiver.account_id,
            amount_paisa=jittered_amount, channel=Channel.IMPS,
            sender_branch=sender.branch_code[:4], receiver_branch=receiver.branch_code[:4],
            sender_geo_lat=_get_coords(sender.branch_code)[0],
            sender_geo_lon=_get_coords(sender.branch_code)[1],
            receiver_geo_lat=_get_coords(receiver.branch_code)[0],
            receiver_geo_lon=_get_coords(receiver.branch_code)[1],
            device_fingerprint=fp,
            sender_account_type=sender.account_type,
            receiver_account_type=receiver.account_type,
            checksum=checksum,
            fraud_label=FraudPattern.LAYERING,
        ))

    return txns


# ── Fraud Pattern #2: Round-Tripping (Circular) ─────────────────────────────

def generate_round_trip(
    world: WorldState,
    base_timestamp: int,
    cycle_size: int = 4,
) -> list[Transaction]:
    """
    A→B→C→D→A. Funds return to origin through a cycle.
    Creates artificial revenue or launders money through intermediaries.
    """
    accounts = _RNG.sample(world.accounts, min(cycle_size, len(world.accounts)))
    amount_paisa = _RNG.randint(1_00_000_00, 5_00_000_00)  # ₹1L – ₹5L

    txns = []
    ts = base_timestamp

    for i in range(len(accounts)):
        sender = accounts[i]
        receiver = accounts[(i + 1) % len(accounts)]  # wraps to form cycle
        ts += _RNG.randint(3600, 43200)  # 1–12 hours between hops

        jittered = int(amount_paisa * _RNG.uniform(0.95, 1.0))
        txn_id = _next_txn_id()
        checksum = compute_transaction_checksum(
            txn_id, ts, sender.account_id, receiver.account_id,
            jittered, int(Channel.NEFT),
        )

        txns.append(Transaction(
            txn_id=txn_id, timestamp=ts,
            sender_id=sender.account_id, receiver_id=receiver.account_id,
            amount_paisa=jittered, channel=Channel.NEFT,
            sender_branch=sender.branch_code[:4], receiver_branch=receiver.branch_code[:4],
            sender_geo_lat=_get_coords(sender.branch_code)[0],
            sender_geo_lon=_get_coords(sender.branch_code)[1],
            receiver_geo_lat=_get_coords(receiver.branch_code)[0],
            receiver_geo_lon=_get_coords(receiver.branch_code)[1],
            device_fingerprint=_device_fingerprint(),
            sender_account_type=sender.account_type,
            receiver_account_type=receiver.account_type,
            checksum=checksum,
            fraud_label=FraudPattern.ROUND_TRIPPING,
        ))

    return txns


# ── Fraud Pattern #3: Structuring (Smurfing) ────────────────────────────────

def generate_structuring_burst(
    world: WorldState,
    base_timestamp: int,
    num_transactions: int = 12,
) -> list[Transaction]:
    """
    Multiple transactions from the same sender, all just below ₹10L CTR
    threshold, spread across a short time window. Classic smurfing.
    """
    sender = _RNG.choice(world.accounts)
    lo, hi = FRAUD_THRESHOLDS.structuring_band_inr

    txns = []
    ts = base_timestamp

    for _ in range(num_transactions):
        receiver = _RNG.choice(world.accounts)
        while receiver.account_id == sender.account_id:
            receiver = _RNG.choice(world.accounts)

        amount_inr = _RNG.randint(lo, hi)
        amount_paisa = amount_inr * 100
        ts += _RNG.randint(600, 7200)  # 10 min – 2 hours apart

        channel = _RNG.choice([Channel.BRANCH, Channel.NETBANKING, Channel.UPI])
        txn_id = _next_txn_id()
        checksum = compute_transaction_checksum(
            txn_id, ts, sender.account_id, receiver.account_id,
            amount_paisa, int(channel),
        )

        txns.append(Transaction(
            txn_id=txn_id, timestamp=ts,
            sender_id=sender.account_id, receiver_id=receiver.account_id,
            amount_paisa=amount_paisa, channel=channel,
            sender_branch=sender.branch_code[:4], receiver_branch=receiver.branch_code[:4],
            sender_geo_lat=_get_coords(sender.branch_code)[0],
            sender_geo_lon=_get_coords(sender.branch_code)[1],
            receiver_geo_lat=_get_coords(receiver.branch_code)[0],
            receiver_geo_lon=_get_coords(receiver.branch_code)[1],
            device_fingerprint=_device_fingerprint(),
            sender_account_type=sender.account_type,
            receiver_account_type=receiver.account_type,
            checksum=checksum,
            fraud_label=FraudPattern.STRUCTURING,
        ))

    return txns


# ── Fraud Pattern #4: Dormant Account Activation ────────────────────────────

def generate_dormant_activation(
    world: WorldState,
    base_timestamp: int,
    num_transactions: int = 5,
) -> list[Transaction]:
    """
    A long-dormant account suddenly receives and sends high-value transfers.
    Classic mule account behavior.
    """
    if not world.dormant_accounts:
        return []

    dormant = _RNG.choice(world.dormant_accounts)
    txns = []
    ts = base_timestamp

    for i in range(num_transactions):
        # Alternating inflows and outflows through the mule account
        if i % 2 == 0:
            partner = _RNG.choice(world.accounts)
            sender_id, receiver_id = partner.account_id, dormant.account_id
            s_branch, r_branch = partner.branch_code, dormant.branch_code
        else:
            partner = _RNG.choice(world.accounts)
            sender_id, receiver_id = dormant.account_id, partner.account_id
            s_branch, r_branch = dormant.branch_code, partner.branch_code

        amount_paisa = _RNG.randint(5_00_000_00, 25_00_000_00)  # ₹5L – ₹25L
        ts += _RNG.randint(300, 3600)
        txn_id = _next_txn_id()
        checksum = compute_transaction_checksum(
            txn_id, ts, sender_id, receiver_id, amount_paisa, int(Channel.NETBANKING),
        )

        txns.append(Transaction(
            txn_id=txn_id, timestamp=ts,
            sender_id=sender_id, receiver_id=receiver_id,
            amount_paisa=amount_paisa, channel=Channel.NETBANKING,
            sender_branch=s_branch[:4], receiver_branch=r_branch[:4],
            sender_geo_lat=_get_coords(s_branch)[0],
            sender_geo_lon=_get_coords(s_branch)[1],
            receiver_geo_lat=_get_coords(r_branch)[0],
            receiver_geo_lon=_get_coords(r_branch)[1],
            device_fingerprint=_device_fingerprint(),
            sender_account_type=AccountType.SAVINGS,
            receiver_account_type=AccountType.SAVINGS,
            checksum=checksum,
            fraud_label=FraudPattern.DORMANT_ACTIVATION,
        ))

    return txns


# ── Fraud Pattern #5: Profile-Behavior Mismatch ─────────────────────────────

def generate_profile_mismatch(
    world: WorldState,
    base_timestamp: int,
    num_transactions: int = 6,
) -> list[Transaction]:
    """
    A low-income savings account (KYC: ₹25K/month) suddenly moves
    crores. The volume is wildly inconsistent with the declared profile.
    """
    # Find a low-income account
    low_income = [a for a in world.accounts if a.declared_monthly_income_paisa < 50_000_00]
    if not low_income:
        low_income = world.accounts[:10]
    suspect = _RNG.choice(low_income)

    txns = []
    ts = base_timestamp

    for _ in range(num_transactions):
        partner = _RNG.choice(world.accounts)
        while partner.account_id == suspect.account_id:
            partner = _RNG.choice(world.accounts)

        # Amount far exceeding declared income (10–50× monthly income)
        multiplier = _RNG.uniform(10, 50)
        amount_paisa = int(suspect.declared_monthly_income_paisa * multiplier)
        ts += _RNG.randint(1800, 14400)

        txn_id = _next_txn_id()
        checksum = compute_transaction_checksum(
            txn_id, ts, suspect.account_id, partner.account_id,
            amount_paisa, int(Channel.RTGS),
        )

        txns.append(Transaction(
            txn_id=txn_id, timestamp=ts,
            sender_id=suspect.account_id, receiver_id=partner.account_id,
            amount_paisa=amount_paisa, channel=Channel.RTGS,
            sender_branch=suspect.branch_code[:4], receiver_branch=partner.branch_code[:4],
            sender_geo_lat=_get_coords(suspect.branch_code)[0],
            sender_geo_lon=_get_coords(suspect.branch_code)[1],
            receiver_geo_lat=_get_coords(partner.branch_code)[0],
            receiver_geo_lon=_get_coords(partner.branch_code)[1],
            device_fingerprint=_device_fingerprint(),
            sender_account_type=suspect.account_type,
            receiver_account_type=partner.account_type,
            checksum=checksum,
            fraud_label=FraudPattern.PROFILE_MISMATCH,
        ))

    return txns


# ── Interbank Message Generator ──────────────────────────────────────────────

def _generate_interbank_message(
    world: WorldState,
    base_timestamp: int,
) -> InterbankMessage:
    """Generate a single RTGS/NEFT/SWIFT interbank settlement message."""
    sender = _RNG.choice(world.accounts)
    receiver = _RNG.choice(world.accounts)
    while receiver.account_id == sender.account_id:
        receiver = _RNG.choice(world.accounts)

    channel = _RNG.choice([Channel.RTGS, Channel.NEFT, Channel.SWIFT])
    msg_type = {"RTGS": "N01", "NEFT": "N06", "SWIFT": "MT103"}.get(channel.name, "N01")

    amount_paisa = _RNG.randint(2_00_000_00, 50_00_000_00)
    ts = base_timestamp + _RNG.randint(0, 86400)

    sender_ifsc = f"{UBI_IFSC_PREFIX}{sender.branch_code}"[:11].ljust(11, "0")
    receiver_ifsc = f"{UBI_IFSC_PREFIX}{receiver.branch_code}"[:11].ljust(11, "0")

    msg_id = _hex_id("MSG")
    checksum = compute_interbank_checksum(
        msg_id, ts, sender_ifsc, receiver_ifsc, amount_paisa, int(channel),
    )

    return InterbankMessage(
        msg_id=msg_id, timestamp=ts,
        sender_ifsc=sender_ifsc, receiver_ifsc=receiver_ifsc,
        sender_account=sender.account_id, receiver_account=receiver.account_id,
        amount_paisa=amount_paisa, currency_code=356,
        channel=channel, message_type=msg_type,
        sender_geo_lat=_get_coords(sender.branch_code)[0],
        sender_geo_lon=_get_coords(sender.branch_code)[1],
        device_fingerprint=_device_fingerprint(),
        priority=0 if channel != Channel.RTGS else 1,
        checksum=checksum,
    )


# ── Auth Event Generator ────────────────────────────────────────────────────

def _generate_auth_event(
    world: WorldState,
    base_timestamp: int,
) -> AuthEvent:
    """Generate a single authentication log entry."""
    account = _RNG.choice(world.accounts)
    ts = base_timestamp + _RNG.randint(0, 86400)

    action = _RNG.choices(
        [AuthAction.LOGIN, AuthAction.LOGOUT, AuthAction.FAILED_LOGIN,
         AuthAction.OTP_VERIFY, AuthAction.OTP_FAIL, AuthAction.PASSWORD_CHANGE],
        weights=[40, 25, 10, 15, 5, 5],
    )[0]

    success = action not in (AuthAction.FAILED_LOGIN, AuthAction.OTP_FAIL)

    # Generate plausible Indian IP ranges
    ip_octets = [_RNG.choice([49, 59, 103, 106, 117, 122, 157, 182, 203]),
                 _RNG.randint(0, 255), _RNG.randint(0, 255), _RNG.randint(1, 254)]
    ip = ".".join(map(str, ip_octets))

    lat, lon = _get_coords(account.branch_code)
    event_id = _hex_id("AUTH")
    ua_hash = hashlib.sha256(f"Mozilla/{_RNG.randint(5, 6)}.0".encode()).hexdigest()[:16]

    checksum = compute_auth_checksum(event_id, ts, account.account_id, int(action), ip)

    return AuthEvent(
        event_id=event_id, timestamp=ts,
        account_id=account.account_id, action=action,
        ip_address=ip, geo_lat=lat, geo_lon=lon,
        device_fingerprint=_device_fingerprint(),
        user_agent_hash=ua_hash, success=success,
        checksum=checksum,
    )


# ── Master Stream Generator ──────────────────────────────────────────────────

def generate_event_stream(
    world: WorldState,
    num_events: int = 10_000,
    fraud_ratio: float = 0.05,
    base_timestamp: int | None = None,
) -> Iterator[Transaction | InterbankMessage | AuthEvent]:
    """
    Generate a mixed stream of transactions, interbank messages, and auth logs.

    ~70% transactions, ~15% interbank, ~15% auth events.
    `fraud_ratio` controls what fraction of transactions contain embedded
    fraud patterns (distributed across all 5 types).
    """
    if base_timestamp is None:
        base_timestamp = int(time.time()) - 30 * 86400  # start 30 days ago

    num_fraud = int(num_events * 0.7 * fraud_ratio)
    num_normal_txn = int(num_events * 0.7) - num_fraud
    num_interbank = int(num_events * 0.15)
    num_auth = num_events - num_normal_txn - num_fraud - num_interbank

    events: list = []

    # Normal transactions
    for _ in range(num_normal_txn):
        events.append(_generate_normal_transaction(world, base_timestamp))

    # Fraud patterns (equal distribution across 5 types)
    fraud_per_type = max(1, num_fraud // 5)

    for _ in range(fraud_per_type):
        ts = base_timestamp + _RNG.randint(0, 25 * 86400)
        events.extend(generate_layering_chain(world, ts, chain_length=_RNG.randint(3, 7)))

    for _ in range(fraud_per_type):
        ts = base_timestamp + _RNG.randint(0, 25 * 86400)
        events.extend(generate_round_trip(world, ts, cycle_size=_RNG.randint(3, 6)))

    for _ in range(fraud_per_type):
        ts = base_timestamp + _RNG.randint(0, 25 * 86400)
        events.extend(generate_structuring_burst(world, ts, num_transactions=_RNG.randint(8, 15)))

    for _ in range(fraud_per_type):
        ts = base_timestamp + _RNG.randint(0, 25 * 86400)
        events.extend(generate_dormant_activation(world, ts))

    for _ in range(fraud_per_type):
        ts = base_timestamp + _RNG.randint(0, 25 * 86400)
        events.extend(generate_profile_mismatch(world, ts))

    # Interbank messages
    for _ in range(num_interbank):
        events.append(_generate_interbank_message(world, base_timestamp))

    # Auth events
    for _ in range(num_auth):
        events.append(_generate_auth_event(world, base_timestamp))

    # Shuffle to simulate real interleaved arrival order
    _RNG.shuffle(events)

    yield from events


async def async_event_stream(
    world: WorldState,
    num_events: int = 10_000,
    fraud_ratio: float = 0.05,
    events_per_sec: int = 1000,
) -> AsyncIterator[Transaction | InterbankMessage | AuthEvent]:
    """
    Async wrapper with throttled emission to simulate real-time arrival.
    Yields events at the specified rate with asyncio.sleep() between bursts.
    """
    import asyncio

    burst_size = max(1, events_per_sec // 10)  # 10 bursts per second
    delay = 0.1  # 100ms between bursts

    stream = generate_event_stream(world, num_events, fraud_ratio)

    burst: list = []
    for event in stream:
        burst.append(event)
        if len(burst) >= burst_size:
            for e in burst:
                yield e
            burst.clear()
            await asyncio.sleep(delay)

    # Flush remaining
    for e in burst:
        yield e
