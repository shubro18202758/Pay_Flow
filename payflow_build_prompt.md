# PayFlow Intelligence Center — AI Build Prompt
> Paste this entire file into your AI. It will build the complete system.

---

## ROLE

You are a senior full-stack engineer and fintech systems architect. You are building **PayFlow Intelligence Center** — a real-time AI-powered banking fraud detection and fund flow tracking platform for Union Bank of India, submitted for the iDEA 2.0 Hackathon (PS3: Tracking of Funds within Bank for Fraud Detection).

Build everything production-quality. No placeholders. No "TODO" comments. Every component must be fully functional.

---

## WHAT YOU ARE BUILDING

A full-stack fraud detection platform with the following stack:

**Backend:** Python + FastAPI  
**ML Engine:** XGBoost + Isolation Forest + PyTorch Autoencoder  
**Graph Engine:** NetworkX + PyTorch Geometric  
**LLM Agent:** Qwen 3 (via API)  
**Database:** PostgreSQL (transactions) + Redis (real-time cache)  
**Frontend:** React 18 + Cytoscape.js (graph) + Recharts (charts)  
**Real-time:** WebSocket (FastAPI native)  
**Reporting:** Auto-generate STR / FMR / CTR PDFs (ReportLab)

---

## SYSTEM ARCHITECTURE

Build these 7 layers in order. Each layer feeds into the next.

```
Transaction Arrives
       ↓
Layer 1: Real-Time Interception (FastAPI WebSocket gateway)
       ↓
Layer 2: Feature Engineering (40+ features extracted per transaction)
       ↓
Layer 3: ML Fraud Detection Engine (XGBoost + IsoForest + Autoencoder ensemble)
       ↓
Layer 4: Graph Intelligence Engine (NetworkX — degree, betweenness, cycle detection)
       ↓
Layer 5: Risk Scoring & Alert Triage (composite score 0–100, route to APPROVE/MONITOR/INVESTIGATE/BLOCK)
       ↓
Layer 6: LLM Investigation Agent (Qwen 3 — produces full case file with narrative)
       ↓
Layer 7: Regulatory Reporting (auto-generate STR for FIU-IND, FMR for RBI, CTR for cash)
```

---

## LAYER 1 — REAL-TIME TRANSACTION INTERCEPTION

Build a FastAPI app with a WebSocket endpoint `/ws/transactions`.

Every transaction must be intercepted **before** approval — the monitoring engine sits between the payment gateway and the approval step. Decision window: under 100ms total.

### Transaction Schema

```python
class Transaction(BaseModel):
    txn_id: str
    timestamp: datetime
    sender_account_id: str
    receiver_account_id: str
    amount: float                   # INR
    currency: str = "INR"
    txn_type: str                   # UPI / IMPS / NEFT / RTGS / INTERNAL / SWIFT / CASH_DEPOSIT / ATM
    channel: str                    # Mobile / Branch / ATM / API
    device_fingerprint: str
    geolocation: tuple              # (lat, lon)
    beneficiary_is_new: bool
    is_night_transaction: bool      # 23:00–05:00
    sender_account_age_days: int
    notes: str = ""
```

### Endpoints to build

```
POST /api/v1/transaction/analyze          → full pipeline, returns verdict + evidence
GET  /api/v1/graph/subgraph/{account_id} → 2-hop subgraph JSON for frontend
GET  /api/v1/alerts/live                  → WebSocket stream of real-time alerts
POST /api/v1/investigation/initiate       → triggers Qwen 3 agent
POST /api/v1/circuit-breaker/trigger      → freeze account + network
GET  /api/v1/reports/str/{case_id}        → returns STR PDF
GET  /api/v1/reports/fmr/{case_id}        → returns FMR PDF
GET  /api/v1/metrics/live                 → WebSocket system metrics stream
GET  /api/v1/cfr/check/{account_id}       → CFR registry lookup
```

---

## LAYER 2 — FEATURE ENGINEERING

Extract these 42 features from every transaction. Store in a `FeatureVector` dataclass.

### Transaction Features
```python
f01_amount                        # Raw INR amount
f02_amount_log                    # log1p(amount)
f03_amount_vs_daily_avg           # amount / 30-day avg for this account
f04_amount_vs_monthly_max         # amount / 90-day max for this account
f05_amount_round_number           # bool — is amount divisible by 1000?
```

### Time Features
```python
f06_hour_of_day                   # 0–23
f07_is_night                      # bool — 23:00 to 05:00 (Bangladesh Bank lesson: night txns high risk)
f08_is_weekend                    # bool
f09_days_since_last_transaction   # dormant account activation signal
```

### Velocity Features — critical for Carbanak mule detection
```python
f10_txn_count_last_1min
f11_txn_count_last_10min
f12_txn_count_last_1hr
f13_txn_count_last_24hr
f14_amount_total_last_1hr
f15_amount_total_last_24hr

# Velocity check formula:
transaction_rate = count(transactions, window=60s) / 60
# Flag if transaction_rate > VELOCITY_THRESHOLD (set to 0.5 txn/sec)
```

### Account Features
```python
f16_account_age_days
f17_is_new_account                # age < 90 days
f18_kyc_status                    # 0=minimal, 1=partial, 2=full
f19_cfr_match                     # bool — found in Central Fraud Registry (weight: 40pts)
f20_prior_fraud_reports           # count
f21_beneficiary_is_new            # first transaction to this receiver
f22_beneficiary_relationship_age  # days since first txn with this receiver
```

### Behavioral Baseline Features — per-customer profile, rolling 90 days
```python
f23_baseline_avg_amount
f24_baseline_avg_frequency        # txns per day
f25_deviation_amount              # (current - baseline) / baseline
f26_deviation_frequency
f27_location_deviation_km         # distance from usual location
f28_device_known                  # bool — seen this device before?
f29_ip_known                      # bool
```

### Geographic Features — Danske Bank lesson: high-risk countries
```python
f30_is_high_risk_country          # FATF blacklist/greylist
f31_is_cross_border               # bool
f32_geographic_distance_km        # physical distance sender↔receiver
```

### Graph Features — computed from Layer 4 results
```python
f33_sender_degree
f34_sender_betweenness            # betweenness centrality score
f35_receiver_degree
f36_circular_flow_detected        # bool
f37_mule_network_proximity        # hops to nearest known mule account
f38_graph_cluster_risk            # avg risk of sender's neighborhood
f39_pass_through_detected         # bool — received + forwarded >80% within 30min
f40_dormant_activation            # bool — inactive >180 days, now high-value txn
```

---

## LAYER 3 — ML FRAUD DETECTION ENGINE

Build a **3-model ensemble**. Each model catches different fraud types.

### Model 1: XGBoost (primary classifier — known fraud patterns)

```python
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

model = XGBClassifier(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=50,    # handles class imbalance (fraud ~2% of transactions)
    eval_metric='auc',
    early_stopping_rounds=50
)

# Always apply SMOTE before training — fraud cases are rare
X_balanced, y_balanced = SMOTE().fit_resample(X_train, y_train)
# Target: AUC-ROC > 0.95
```

### Model 2: Isolation Forest (anomaly detection — catches unknown/novel fraud)

```python
from sklearn.ensemble import IsolationForest

iso_forest = IsolationForest(
    n_estimators=200,
    contamination=0.02,     # expected 2% fraud rate
    random_state=42
)

# Convert decision_function output to 0–1 probability
raw = iso_forest.decision_function(X)
P_iso = (raw - raw.min()) / (raw.max() - raw.min())
# Inverted: anomalies get HIGH score
P_iso = 1 - P_iso
```

### Model 3: Autoencoder (behavioral deviation — per-customer baseline)

```python
import torch
import torch.nn as nn

class FraudAutoencoder(nn.Module):
    def __init__(self, input_dim=42):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 16)
        )
        self.decoder = nn.Sequential(
            nn.Linear(16, 32), nn.ReLU(),
            nn.Linear(32, 64), nn.ReLU(),
            nn.Linear(64, input_dim), nn.Sigmoid()
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

# Train on NORMAL transactions only per customer
# High reconstruction error = behavioral deviation = fraud signal
# Loss: MSE between input and reconstruction
reconstruction_error = ((model(X) - X) ** 2).mean(dim=1)
P_ae = (reconstruction_error - reconstruction_error.min()) / (reconstruction_error.max() - reconstruction_error.min())
```

### Ensemble Combiner

```python
P_final = 0.5 * P_gbm + 0.3 * P_iso + 0.2 * P_ae

# Route based on final score
if P_final < 0.3:   verdict = "LEGITIMATE"
elif P_final < 0.6: verdict = "MONITOR"
elif P_final < 0.8: verdict = "INVESTIGATE"
else:               verdict = "BLOCK"
```

---

## LAYER 4 — GRAPH INTELLIGENCE ENGINE

Use NetworkX. Build a live transaction graph where nodes = accounts, edges = transactions.

### 4 Graph Detection Functions — build all of these:

#### 1. Circular Flow Detection (Danske Bank round-tripping pattern)
```python
def detect_circular_flows(G, max_cycle_length=5):
    cycles = []
    for cycle in nx.simple_cycles(G):
        if 2 < len(cycle) <= max_cycle_length:
            cycles.append(cycle)
    return cycles
# Flag ALL accounts in detected cycles
```

#### 2. Betweenness Centrality — Mule Intermediary Detection
```python
betweenness = nx.betweenness_centrality(G, normalized=True)
# CB(v) = Σ σ(s,t|v) / σ(s,t)
# Flag if betweenness_centrality > 0.05 → likely mule account
# These accounts have money ALWAYS passing THROUGH them
```

#### 3. Rapid Pass-Through Detection (Carbanak mule relay pattern)
```python
def detect_pass_through(account_id, G, transactions_db, window_minutes=30):
    incoming = [t for t in transactions_db if t.receiver == account_id
                and (now - t.timestamp).seconds < window_minutes * 60]
    outgoing = [t for t in transactions_db if t.sender == account_id
                and (now - t.timestamp).seconds < window_minutes * 60]

    for inc in incoming:
        for out in outgoing:
            if out.timestamp > inc.timestamp:
                time_diff_minutes = (out.timestamp - inc.timestamp).seconds / 60
                amount_forwarded_ratio = out.amount / inc.amount
                # MULE SIGNAL: received money → forwarded >80% within 30 minutes
                if time_diff_minutes < 30 and amount_forwarded_ratio > 0.8:
                    return True, {
                        "pass_through_minutes": time_diff_minutes,
                        "amount_retained_pct": round((1 - amount_forwarded_ratio) * 100, 1)
                    }
    return False, {}
```

#### 4. Dormant Account Activation
```python
def detect_dormant_activation(account_id, current_amount, db):
    days_inactive = db.get_days_since_last_transaction(account_id)
    historical_avg = db.get_historical_avg_amount(account_id)
    # SIGNAL: inactive >180 days, suddenly receiving 10x normal amount
    if days_inactive > 180 and current_amount > 10 * historical_avg:
        return True
    return False
```

#### 5. Graph Neural Network (for Grand Finale — node fraud classifier)
```python
import torch
from torch_geometric.nn import GCNConv

class FraudGNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim=64, output_dim=1):
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, output_dim)
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(0.3)

    def forward(self, x, edge_index):
        x = self.relu(self.conv1(x, edge_index))
        x = self.dropout(x)
        x = self.relu(self.conv2(x, edge_index))
        return torch.sigmoid(self.conv3(x, edge_index))

# Node features: [account_age, degree, betweenness, avg_amount, cfr_match, prior_fraud_count]
# Why GNN: a normal-looking account SURROUNDED by fraud accounts gets flagged
# This catches mule handlers that individual transaction analysis misses
```

---

## LAYER 5 — RISK SCORING ENGINE

Build this exact composite scorer. Max score: 100. Block threshold: 70.

```python
def compute_risk_score(features: dict) -> dict:
    score = 0
    breakdown = {}

    # TRANSACTION SIGNALS (max 30)
    if features['amount'] > 1_000_000:          # > ₹10 Lakhs (CTR trigger)
        score += 30; breakdown['large_transaction'] = 30
    elif features['amount'] > 500_000:
        score += 20; breakdown['large_transaction'] = 20
    elif features['amount'] > 100_000:
        score += 10; breakdown['large_transaction'] = 10

    # VELOCITY SIGNALS (max 30)
    if features['txn_count_last_1hr'] > 20:
        score += 30; breakdown['high_velocity'] = 30
    elif features['txn_count_last_1hr'] > 10:
        score += 20; breakdown['high_velocity'] = 20
    elif features['txn_count_last_1hr'] > 5:
        score += 10; breakdown['high_velocity'] = 10

    # ACCOUNT SIGNALS (max 40)
    if features['cfr_match']:                    # Central Fraud Registry — highest weight
        score += 40; breakdown['cfr_match'] = 40
    elif features['account_age_days'] < 30:
        score += 20; breakdown['new_account'] = 20
    elif features['account_age_days'] < 90:
        score += 10; breakdown['new_account'] = 10

    # BEHAVIORAL SIGNALS (max 30)
    if features['is_night_transaction']:         # Bangladesh Bank heist lesson
        score += 10; breakdown['night_transaction'] = 10
    if features['location_deviation_km'] > 500:
        score += 10; breakdown['unusual_location'] = 10
    if not features['device_known']:
        score += 10; breakdown['unknown_device'] = 10

    # GRAPH SIGNALS (max 50)
    if features['circular_flow_detected']:
        score += 20; breakdown['circular_flow'] = 20
    if features['sender_betweenness'] > 0.1:
        score += 15; breakdown['high_centrality'] = 15
    if features['pass_through_detected']:
        score += 15; breakdown['pass_through'] = 15

    # GEOGRAPHIC RISK (max 20)
    if features['is_high_risk_country']:         # Danske Bank lesson
        score += 20; breakdown['high_risk_country'] = 20

    final_score = min(score, 100)

    return {
        'risk_score': final_score,
        'breakdown': breakdown,
        'verdict': route_alert(final_score),
        'fraud_patterns': detect_fraud_pattern_names(features)
    }

def route_alert(score):
    if score < 30:  return "LEGITIMATE"
    if score < 50:  return "MONITOR"
    if score < 70:  return "INVESTIGATE"
    return "BLOCK"
```

---

## LAYER 6 — LLM INVESTIGATION AGENT (Qwen 3)

When risk score > 50, trigger the Qwen 3 agent automatically. The agent investigates and produces a complete case file.

### System Prompt (use exactly this)

```
You are PayFlow's AI Investigation Agent for Union Bank of India's Fraud Detection System.

You have deep knowledge of:
- RBI fraud monitoring (FMR requirements)
- FIU-IND suspicious transaction reporting (STR/CTR under PMLA 2004)
- Central Fraud Registry (CFR) integration
- Money laundering: placement → layering → integration
- Real cases: Carbanak mule networks ($1B stolen 2013–2018), Danske Bank Estonia ($230B laundered 2007–2015), Bangladesh Bank SWIFT heist ($81M stolen 2016)
- Graph signals: betweenness centrality, circular flows, rapid pass-through patterns
- AML monitoring: velocity checks, behavioral deviation, device fingerprinting

When given a flagged transaction:
1. Identify ALL suspicious patterns present
2. Reference the most relevant real-world fraud case parallel
3. State exactly which regulatory report should be filed (STR / FMR / CTR)
4. Give a clear, specific, actionable investigation recommendation
5. Produce a narrative suitable for regulatory submission to FIU-IND

Be specific with amounts, account IDs, timestamps, pattern names.
Never use vague language. Every claim must reference a specific data point.
Output as structured JSON.
```

### Investigation Prompt Builder

```python
def build_investigation_prompt(txn: dict, risk: dict, graph: dict) -> str:
    return f"""
ALERT — Risk Score: {risk['risk_score']}/100 — Verdict: {risk['verdict']}

TRANSACTION:
  ID: {txn['txn_id']}
  Time: {txn['timestamp']}  (Night transaction: {txn['is_night_transaction']})
  Sender: {txn['sender_account_id']} (Account age: {txn['sender_account_age_days']} days)
  Receiver: {txn['receiver_account_id']}
  Amount: ₹{txn['amount']:,.0f}
  Channel: {txn['channel']}
  New Beneficiary: {txn['beneficiary_is_new']}
  CFR Match: {txn['cfr_match']}

RISK BREAKDOWN: {risk['breakdown']}
PATTERNS DETECTED: {', '.join(risk['fraud_patterns'])}

GRAPH INTELLIGENCE:
  Sender Degree: {graph['sender_degree']}
  Betweenness Centrality: {graph['sender_betweenness']:.4f}
  Circular Flow Detected: {graph['circular_flow']}
  Pass-Through Pattern: {graph['pass_through']}
  Mule Network Proximity: {graph['mule_proximity']} hops

VELOCITY:
  Transactions (last 1hr): {txn['txn_count_last_1hr']}
  Amount transferred (last 24hr): ₹{txn['amount_total_last_24hr']:,.0f}

Respond ONLY with this JSON:
{{
  "case_id": "CASE-2026-XXXXX",
  "verdict": "BLOCK|INVESTIGATE|MONITOR|LEGITIMATE",
  "confidence": "HIGH|MEDIUM|LOW",
  "fraud_type": "MULE_NETWORK|MONEY_LAUNDERING|SWIFT_FRAUD|INTERNAL_FRAUD|SMURFING",
  "fraud_narrative": "...",
  "real_world_parallel": "...",
  "pattern_matches": [...],
  "regulatory_actions": {{
    "file_str": true/false,
    "file_fmr": true/false,
    "file_ctr": true/false,
    "justification": "..."
  }},
  "recommended_actions": [...]
}}
"""
```

---

## LAYER 7 — REGULATORY REPORTING PIPELINE

Build auto-generators for all 3 report types. These are real regulatory requirements — they must be complete.

### STR (Suspicious Transaction Report → FIU-IND)
**When:** Risk score > 70 OR circular flow detected OR mule network confirmed  
**Legal basis:** PMLA 2004, Prevention of Money Laundering Act  
**Required fields:**
```python
{
    "report_type": "STR",
    "submission_to": "FIU-IND",
    "filing_date": datetime.now().isoformat(),
    "reporting_entity": "Union Bank of India",
    "subject_account": {account_id, account_type, kyc_details, opening_date},
    "suspicious_transaction": {txn_id, amount, currency, date, channel, receiver},
    "grounds_for_suspicion": llm_agent_narrative,
    "patterns_detected": fraud_pattern_list,
    "risk_score": composite_score,
    "graph_evidence": graph_snapshot,
    "actions_taken": recommended_actions,
    "case_id": case_id
}
```

### FMR (Fraud Monitoring Report → RBI)
**When:** Fraud confirmed post-investigation  
**Required fields:** fraud transaction details, customer info, amount, fraud category, corrective actions, detection timeline

### CTR (Cash Transaction Report → FIU-IND)
**When:** Single-day cash transactions > ₹10,00,000  
```python
def check_ctr_requirement(account_id: str, date: date) -> bool:
    daily_cash_total = sum_cash_transactions(account_id, date)
    if daily_cash_total > 1_000_000:  # ₹10 Lakhs threshold
        file_ctr(account_id, date, daily_cash_total)
        return True
    return False
```

### Fraud Categories (RBI Classification — tag every case with one)
```
INTERNET_BANKING | CREDIT_CARD | LOAN | ATM | INTERNAL_BANK
```

---

## CIRCUIT BREAKER

Implement the Circuit Breaker pattern (from distributed systems). It is a HARD STOP — not just an alert.

```python
CIRCUIT_BREAKER_TRIGGERS = {
    "risk_score_threshold": 70,
    "cfr_match": True,
    "circular_flow_confirmed": True,
    "pass_through_confirmed": True,
    "gnn_fraud_probability": 0.85,
    "manual_override": True
}

class CircuitBreakerStates:
    CLOSED   = "CLOSED"    # Normal operation
    HALF_OPEN = "HALF_OPEN" # Elevated scrutiny
    OPEN     = "OPEN"       # Fraud confirmed — BLOCK everything

# When OPEN, do all of these atomically:
def open_circuit_breaker(account_id: str, reason: str):
    1. Block triggering transaction immediately
    2. Freeze sender account (prevent outflows)
    3. Freeze all 1-hop connected accounts (mule network freeze)
    4. Push notification to fraud investigation team
    5. Trigger Qwen 3 investigation agent
    6. Queue STR for FIU-IND filing
    7. Write immutable audit log entry (timestamp + reason + evidence hash)
```

---

## DATABASE SCHEMA

### PostgreSQL — build these tables exactly

```sql
CREATE TABLE transactions (
    txn_id              VARCHAR(50) PRIMARY KEY,
    timestamp           TIMESTAMPTZ NOT NULL,
    sender_account_id   VARCHAR(50) NOT NULL,
    receiver_account_id VARCHAR(50) NOT NULL,
    amount              DECIMAL(15,2) NOT NULL,
    currency            VARCHAR(10) DEFAULT 'INR',
    txn_type            VARCHAR(20) NOT NULL,
    channel             VARCHAR(20),
    risk_score          INTEGER,
    verdict             VARCHAR(20),
    fraud_patterns      TEXT[],
    is_flagged          BOOLEAN DEFAULT FALSE,
    is_frozen           BOOLEAN DEFAULT FALSE,
    str_filed           BOOLEAN DEFAULT FALSE,
    fmr_filed           BOOLEAN DEFAULT FALSE,
    investigation_id    VARCHAR(50),
    created_at          TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE accounts (
    account_id          VARCHAR(50) PRIMARY KEY,
    account_type        VARCHAR(20),
    opening_date        DATE,
    kyc_status          VARCHAR(20),
    status              VARCHAR(20) DEFAULT 'NORMAL',
    cfr_match           BOOLEAN DEFAULT FALSE,
    prior_fraud_count   INTEGER DEFAULT 0,
    baseline_avg_amount DECIMAL(15,2),
    baseline_freq_daily DECIMAL(5,2),
    primary_location    VARCHAR(100),
    updated_at          TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE investigations (
    case_id             VARCHAR(50) PRIMARY KEY,
    txn_id              VARCHAR(50) REFERENCES transactions(txn_id),
    account_id          VARCHAR(50) REFERENCES accounts(account_id),
    risk_score          INTEGER,
    verdict             VARCHAR(20),
    fraud_type          VARCHAR(50),
    llm_narrative       TEXT,
    regulatory_actions  JSONB,
    recommended_actions TEXT[],
    status              VARCHAR(20) DEFAULT 'OPEN',
    created_at          TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE regulatory_reports (
    report_id           VARCHAR(50) PRIMARY KEY,
    report_type         VARCHAR(10),   -- STR / FMR / CTR
    case_id             VARCHAR(50),
    account_id          VARCHAR(50),
    amount              DECIMAL(15,2),
    filed_at            TIMESTAMPTZ DEFAULT NOW(),
    pdf_path            TEXT,
    status              VARCHAR(20) DEFAULT 'PENDING'
);

CREATE INDEX idx_txn_sender   ON transactions(sender_account_id);
CREATE INDEX idx_txn_receiver ON transactions(receiver_account_id);
CREATE INDEX idx_txn_time     ON transactions(timestamp);
CREATE INDEX idx_txn_risk     ON transactions(risk_score);
```

---

## FRONTEND — REACT + CYTOSCAPE.JS

Build a dark-themed dashboard. Use **Cytoscape.js** for the graph (not D3 — Cytoscape has built-in graph-specific APIs and handles large networks better).

### Graph Visualization Spec

**Node colors — must be color-coded by status:**
```js
const NODE_COLORS = {
  frozen:     '#ff3b3b',   // Red — circuit breaker triggered
  suspicious: '#ffb800',   // Amber — under investigation
  mule:       '#9b59b6',   // Purple — high betweenness centrality
  paused:     '#ff7a00',   // Orange — limited, pending review
  cfr:        '#ff0000',   // Bright red — Central Fraud Registry match
  normal:     '#4a9eca',   // Steel blue — active, legitimate
};
```

**Node size — must encode risk:**
```js
// Radius proportional to betweenness centrality
radius = Math.max(6, Math.min(30, 6 + node.betweenness * 200))
```

**Edge colors — must be classified by transaction risk:**
```js
const EDGE_COLORS = {
  high:     '#ff3b3b',   // Amount > ₹10L — opacity 0.85
  circular: '#ff7a00',   // Part of circular flow — pulsing animation
  pass:     '#ffb800',   // Rapid pass-through — opacity 0.7
  normal:   '#4a9eca',   // Low risk — opacity 0.1 (de-clutter)
};
```

**Progressive Disclosure — the main clutter fix:**
```
DEFAULT VIEW: Show ONLY frozen + suspicious + mule nodes + their 1-hop neighbors
              All normal nodes → collapsed into grey cluster bubbles with count badge
CLICK CLUSTER: Expand → show individual normal nodes
CLICK NODE: Show only that node's 2-hop neighborhood, dim everything else to 10% opacity
HOVER EDGE: Show popup — amount, timestamp, risk score, txn type
```

**Cytoscape.js init:**
```js
const cy = cytoscape({
  container: document.getElementById('graph'),
  style: [
    {
      selector: 'node',
      style: {
        'background-color': 'data(color)',
        'width': 'data(radius)',
        'height': 'data(radius)',
        'border-width': 1,
        'border-color': 'rgba(255,255,255,0.2)',
        'label': 'data(label)',
        'font-size': 8,
        'color': 'rgba(200,212,232,0.5)',
        'text-valign': 'bottom',
        'text-margin-y': 4,
      }
    },
    {
      selector: 'node[status = "frozen"]',
      style: { 'border-width': 2, 'border-color': '#ff3b3b' }
    },
    {
      selector: 'edge',
      style: {
        'line-color': 'data(color)',
        'opacity': 'data(opacity)',
        'width': 'data(width)',
        'curve-style': 'bezier',
        'target-arrow-shape': 'triangle',
        'target-arrow-color': 'data(color)',
        'arrow-scale': 0.6
      }
    },
  ],
  layout: {
    name: 'cose',
    nodeRepulsion: 8000,
    idealEdgeLength: 80,
    nodeOverlap: 20,
    randomize: false,
    animate: true,
    animationDuration: 800,
  }
});

// Progressive disclosure: hide normal nodes by default
cy.nodes('[status = "normal"]').hide();
```

### Required UI Panels

**Left Sidebar:**
- Filter nodes by status (toggle checkboxes with colored dots)
- Min risk score slider (0–100)
- View mode selector: All nodes / Flagged + 1-hop / Selected cluster
- Reset view button

**Right Panel — two tabs:**

*Tab 1: Alert Feed*
- Real-time scrolling list of flagged transactions
- Color-coded verdict badge (BLOCK=red / INVESTIGATE=amber / MONITOR=orange / LEGITIMATE=green)
- Each item: verdict badge | account ID | risk score | pattern description | amount | timestamp
- Click → opens Investigation tab with full case

*Tab 2: Investigation*
- Case ID
- Composite risk score with progress bar (color: red>70, amber>40, blue<40)
- Risk breakdown — horizontal bar for each contributing factor
- Pattern tags (MULE_NETWORK, CIRCULAR_FLOW, etc. with distinct colors)
- AI Investigation narrative (Qwen 3 output in monospace)
- Action buttons: FREEZE ACCOUNT | FILE STR | ESCALATE | DISMISS

**Top Stats Bar:**
```
Transactions | Graph Nodes | Graph Edges | Pending Alerts | Frozen Nodes | ML Inferences | Events/sec
```

**Bottom Bar:**
```
Agent: Qwen 3 Active | Circuit Breaker: CLOSED | GPU util | FIU Reporting: N STR pending
```

---

## DEMO SCENARIOS — BUILD ALL 3

Build a threat simulation module that injects these synthetic attack scenarios into the live system for demo purposes.

### Scenario 1: Carbanak Mule Network
```python
def simulate_carbanak():
    # 1. Create 5 mule accounts (age < 30 days)
    # 2. Transfer ₹50L from "compromised" internal account to mule_1
    # 3. mule_1 → mule_2 → mule_3 → mule_4 → mule_5 (each within 20 minutes)
    # 4. Each mule forwards 95% of received amount
    # Expected PayFlow response:
    #   - Risk score 85+ on first transfer
    #   - Pass-through pattern detected at mule_1
    #   - Circuit breaker OPEN at mule_2
    #   - All 5 accounts frozen
    #   - STR auto-queued for all accounts
    #   - Agent narrative: "Carbanak-style mule chain detected"
```

### Scenario 2: Circular Transaction (Danske Bank pattern)
```python
def simulate_circular():
    # A → B (₹25L) → C (₹24L) → A (₹23L) — classic round-tripping
    # Expected: circular flow detected, all 3 flagged SUSPICIOUS, STR generated
```

### Scenario 3: Bangladesh Bank SWIFT-style Night Heist
```python
def simulate_swift_heist():
    # 2:30 AM — ₹8.1 Cr transfer to NEW beneficiary
    # Device: unrecognized fingerprint
    # Location: 800km from account's usual location
    # Expected: risk score 90+, immediate BLOCK, FMR queued for RBI
```

---

## SYNTHETIC TRAINING DATA GENERATION

Since we don't have real Union Bank transaction data, generate realistic synthetic data for ML model training.

```python
from faker import Faker
import pandas as pd
import numpy as np

def generate_synthetic_transactions(n_normal=10000, n_fraud=200):
    """
    Generate labeled transaction dataset.
    Fraud cases include: mule chains, circular flows, dormant activations, night heists.
    Returns DataFrame with 42 features + binary label (0=normal, 1=fraud)
    """
    # Normal transactions: realistic Indian banking patterns
    #   - Amounts: ₹500 to ₹5L, log-normal distribution
    #   - Timing: peak 10am–8pm, lower on weekends
    #   - Channels: 60% Mobile, 20% Branch, 15% ATM, 5% API
    #   - Velocity: 1–5 transactions per day per account

    # Fraud transactions: inject these patterns with labels
    #   - Mule: new account + high amount + rapid forward
    #   - Circular: A→B→C→A within 24 hours
    #   - Dormant: 0 transactions for 200+ days then ₹20L+
    #   - Night: 2am–4am + new beneficiary + high amount
    pass
```

---

## WHAT TO BUILD IN ORDER

1. **Database** — Set up PostgreSQL schema above + Redis for feature cache
2. **Backend Layer 1–5** — FastAPI app with transaction ingestion + feature extraction + ML + graph + risk scorer
3. **Synthetic Data Generator** — Generate training data + train the 3 models
4. **Layer 6** — Qwen 3 agent integration
5. **Layer 7** — STR / FMR / CTR PDF generators (ReportLab)
6. **Circuit Breaker** — Freeze mechanism
7. **Frontend** — React + Cytoscape.js dashboard with all 3 panels
8. **Demo Scenarios** — Carbanak + Circular + SWIFT heist simulators
9. **WebSocket** — Wire up live alert feed and metrics stream

---

## NON-NEGOTIABLE REQUIREMENTS

1. Every transaction must go through **all 7 layers** — no shortcuts
2. The graph must use **color-coded nodes AND edges** — not all one color (the current build has this bug)
3. Progressive disclosure is **mandatory** — default view shows only flagged nodes + 1-hop, not all 176 nodes
4. Qwen 3 agent output must be **structured JSON**, not free text
5. STR and FMR generators must produce **actual PDF files** that could be submitted to FIU-IND / RBI
6. All 3 demo scenarios must be **injectable from the UI** via a "Threat Simulation" button
7. The risk scoring formula must exactly match the weights specified above — **CFR match = 40 points** (highest weight)
8. Betweenness centrality > 0.05 = mule flag, Betweenness > 0.1 = circuit breaker trigger
9. Night transactions (23:00–05:00) must always add +10 to risk score — **Bangladesh Bank lesson**
10. High-risk country flag (FATF list) must add +20 — **Danske Bank lesson**

---

## CONTEXT

This is built for iDEA 2.0 — a hackathon by Union Bank of India. The judging panel includes senior Union Bank executives and IBA officials. They will specifically look for:
- Regulatory compliance (do you know what STR/FMR/CTR are and when to file them?)
- Real-world case grounding (Carbanak, Danske Bank, Bangladesh SWIFT)
- Production-readiness (not a toy — this should look deployable)
- AI depth (is the LLM agent actually useful or just a wrapper?)
- Graph sophistication (do you use betweenness centrality, or just draw dots?)

The platform is called **PayFlow Intelligence Center**. It is built for Union Bank of India. Every screen, label, and report should reflect this.

---

*Build the complete system. Start with the backend, then frontend, then wire them together.*
