"""PS3 case tracing and evidence packaging helpers."""

from __future__ import annotations

import hashlib
import time
from html import escape
from typing import Any

from config.settings import OLLAMA_CFG
from src.ingestion.schemas import Channel, FraudPattern
from src.simulation.attack_generators import PS3_SCENARIO_DETAILS


def _now() -> float:
    return round(time.time(), 3)


def _format_inr(amount_paisa: int) -> str:
    return f"INR {amount_paisa / 100:,.2f}"


def _enum_name(enum_cls, value: Any, default: str = "UNKNOWN") -> str:
    try:
        return enum_cls(value).name
    except Exception:
        if isinstance(value, str):
            return value
        return default


def _get_ps3_case_meta(orch: Any, case_id: str) -> dict[str, Any] | None:
    engine = getattr(orch, "_threat_engine", None) if orch is not None else None
    if engine is not None and hasattr(engine, "get_ps3_case"):
        return engine.get_ps3_case(case_id)
    return None


def _get_graph(orch: Any):
    graph_obj = getattr(orch, "_graph", None) if orch is not None else None
    if graph_obj is None:
        graph_obj = getattr(orch, "graph", None) if orch is not None else None
    if graph_obj is None:
        return None
    graph = getattr(graph_obj, "_graph", None)
    if graph is not None:
        return graph
    return getattr(graph_obj, "graph", None)


def _collect_graph_transactions(orch: Any, meta: dict[str, Any] | None) -> list[dict[str, Any]]:
    graph = _get_graph(orch)
    if graph is None:
        return []

    account_ids = set((meta or {}).get("accounts_involved") or [])
    focus_txn_id = (meta or {}).get("focus_txn_id", "")
    rows: list[dict[str, Any]] = []

    for source, target, key, data in graph.edges(keys=True, data=True):
        fraud_label = int(data.get("fraud_label", 0) or 0)
        include = (
            key == focus_txn_id
            or bool(account_ids and (source in account_ids or target in account_ids))
            or (not account_ids and fraud_label > 0)
        )
        if not include:
            continue
        rows.append({
            "txn_id": str(key),
            "source": str(source),
            "target": str(target),
            "amount_paisa": int(data.get("amount_paisa", 0) or 0),
            "channel": _enum_name(Channel, data.get("channel", 0)),
            "fraud_label": fraud_label,
            "fraud_label_name": _enum_name(FraudPattern, fraud_label, "NONE"),
            "timestamp": int(data.get("timestamp", 0) or 0),
            "device_fingerprint": data.get("device_fingerprint", ""),
            "evidence_id": f"GRAPH:{key}",
        })

    rows.sort(key=lambda item: item.get("timestamp", 0))
    return rows[:120]


def _metadata_transactions(meta: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not meta:
        return []
    base_ts = int(meta.get("generated_at") or time.time())
    rows: list[dict[str, Any]] = []
    for idx, item in enumerate(meta.get("transaction_chain", []) or []):
        if item.get("type") != "transaction":
            continue
        rows.append({
            "txn_id": item.get("txn_id", f"TXN-PREVIEW-{idx}"),
            "source": item.get("sender", ""),
            "target": item.get("receiver", ""),
            "amount_paisa": int(item.get("amount_paisa", 0) or 0),
            "channel": item.get("channel", "UNKNOWN"),
            "fraud_label": -1,
            "fraud_label_name": item.get("fraud_label", "PS3_PREVIEW"),
            "timestamp": base_ts + idx,
            "device_fingerprint": "",
            "evidence_id": f"SIM:{item.get('txn_id', idx)}",
        })
    return rows


def _derive_account_roles(transactions: list[dict[str, Any]], focus_account: str = "") -> list[dict[str, str]]:
    ordered: list[str] = []
    for txn in transactions:
        for account_id in (txn.get("source", ""), txn.get("target", "")):
            if account_id and account_id not in ordered:
                ordered.append(account_id)

    roles: list[dict[str, str]] = []
    terminal = ordered[-1] if ordered else ""
    for idx, account_id in enumerate(ordered):
        if account_id == focus_account or idx == 0:
            role = "origin"
        elif account_id == terminal:
            role = "terminal_beneficiary"
        else:
            role = "intermediary"
        roles.append({
            "account_id": account_id,
            "role": role,
            "position": str(idx + 1),
        })
    return roles


def _timeline(transactions: list[dict[str, Any]], indicators: list[str]) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for idx, txn in enumerate(transactions):
        indicator = indicators[idx % len(indicators)] if indicators else "Fund-flow hop recorded"
        entries.append({
            "step": idx + 1,
            "timestamp": txn.get("timestamp", 0),
            "title": f"{txn.get('source', '')} to {txn.get('target', '')}",
            "txn_id": txn.get("txn_id", ""),
            "amount_paisa": txn.get("amount_paisa", 0),
            "amount_display": _format_inr(int(txn.get("amount_paisa", 0) or 0)),
            "channel": txn.get("channel", "UNKNOWN"),
            "indicator": indicator,
            "evidence_id": txn.get("evidence_id", ""),
        })
    return entries


def build_case_trace(orch: Any, case_id: str) -> dict[str, Any]:
    """Build a PS3 case trace from live graph data, falling back to launch metadata."""
    meta = _get_ps3_case_meta(orch, case_id) or {}
    scenario = meta.get("scenario", "ad_hoc")
    details = PS3_SCENARIO_DETAILS.get(str(scenario), {})
    transactions = _collect_graph_transactions(orch, meta)
    if not transactions:
        transactions = _metadata_transactions(meta)

    indicators = list(meta.get("expected_indicators") or details.get("expected_indicators") or [])
    typologies = list(meta.get("typologies") or details.get("typologies") or [])
    recommended_actions = list(
        meta.get("recommended_actions") or details.get("recommended_actions") or [
            "Review account behavior and attach graph evidence",
            "Escalate to FIU reporting queue if suspicion remains",
        ]
    )
    focus_account = str(meta.get("focus_account_id") or (transactions[0]["source"] if transactions else ""))
    graph_path: list[str] = []
    for txn in transactions:
        for account_id in (txn.get("source", ""), txn.get("target", "")):
            if account_id and account_id not in graph_path:
                graph_path.append(account_id)

    total_amount = sum(int(txn.get("amount_paisa", 0) or 0) for txn in transactions)
    fraud_edges = sum(1 for txn in transactions if str(txn.get("fraud_label_name", "NONE")) != "NONE")
    graph_score = round(min(1.0, fraud_edges / max(len(transactions), 1)), 4)
    case_status = meta.get("scenario_status", "ready" if transactions else "no_live_case")
    try:
        from src.intel import get_pre_fraud_intel_service

        pre_fraud_intelligence = get_pre_fraud_intel_service().evidence_context()
    except Exception:
        pre_fraud_intelligence = {
            "active_playbooks": [],
            "top_trends": [],
            "guardrail": "Pre-fraud intelligence context unavailable.",
        }

    return {
        "case_id": case_id,
        "scenario_id": meta.get("scenario_id", ""),
        "scenario": scenario,
        "scenario_label": meta.get("scenario_label") or details.get("label", "Ad-hoc PS3 Case"),
        "status": case_status,
        "focus_account_id": focus_account,
        "focus_txn_id": meta.get("focus_txn_id", transactions[0]["txn_id"] if transactions else ""),
        "ps3_typologies": typologies,
        "expected_indicators": indicators,
        "timeline": _timeline(transactions, indicators),
        "transaction_chain": transactions,
        "graph_path": graph_path,
        "account_roles": _derive_account_roles(transactions, focus_account),
        "risk_scores": {
            "graph_evidence_score": graph_score,
            "fraud_edge_count": fraud_edges,
            "transaction_count": len(transactions),
            "total_amount_paisa": total_amount,
            "total_amount_display": _format_inr(total_amount),
        },
        "evidence_references": [
            "GRAPH_PATH",
            "FRAUD_LABELS",
            "AUDIT_LEDGER",
            "QWEN_GROUNDED_SUMMARY",
        ],
        "recommended_actions": recommended_actions,
        "pre_fraud_intelligence": pre_fraud_intelligence,
        "narrative": (
            f"{len(transactions)} linked fund-flow events were assembled for "
            f"{meta.get('scenario_label') or details.get('label', 'PS3 tracing')}."
        ),
        "generated_at": _now(),
    }


def _html_package(package: dict[str, Any]) -> str:
    trace = package["case_trace"]
    pre_fraud = package.get("pre_fraud_intelligence", {}) or {}
    active_playbooks = pre_fraud.get("active_playbooks", [])
    top_trends = pre_fraud.get("top_trends", [])
    rows = []
    for item in trace["timeline"]:
        rows.append(
            "<tr>"
            f"<td>{escape(str(item['step']))}</td>"
            f"<td>{escape(str(item['txn_id']))}</td>"
            f"<td>{escape(str(item['title']))}</td>"
            f"<td>{escape(str(item['amount_display']))}</td>"
            f"<td>{escape(str(item['channel']))}</td>"
            f"<td>{escape(str(item['indicator']))}</td>"
            "</tr>"
        )
    typologies = ", ".join(trace.get("ps3_typologies") or [])
    actions = "".join(f"<li>{escape(str(action))}</li>" for action in trace.get("recommended_actions", []))
    intel_rows = "".join(
        "<tr>"
        f"<td>{escape(str(item.get('trend_id', '')))}</td>"
        f"<td>{escape(str(item.get('title', '')))}</td>"
        f"<td>{escape(str(item.get('trust_score', '')))}</td>"
        f"<td>{escape(str(item.get('india_relevance_score', '')))}</td>"
        "</tr>"
        for item in top_trends
    )
    playbook_rows = "".join(
        "<tr>"
        f"<td>{escape(str(item.get('playbook_id', '')))}</td>"
        f"<td>{escape(str(item.get('title', '')))}</td>"
        f"<td>{escape(str(item.get('audit_hash', ''))[:16])}</td>"
        "</tr>"
        for item in active_playbooks
    )
    counter_rows = "".join(
        "<tr>"
        f"<td>{escape(str(item.get('proposal_id', '')))}</td>"
        f"<td>{escape(str(item.get('action', '')))}</td>"
        f"<td>{escape(str(item.get('status', '')))}</td>"
        f"<td>{escape(', '.join(str(t) for t in item.get('targets', [])))}</td>"
        f"<td>{escape(str(item.get('audit_hash', ''))[:16])}</td>"
        "</tr>"
        for item in package.get("countermeasure_proposals", [])
    )
    return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>PayFlow FIU Evidence Package {escape(package['package_id'])}</title>
  <style>
    body {{ font-family: Arial, sans-serif; color: #111827; margin: 32px; }}
    h1 {{ font-size: 22px; margin-bottom: 4px; }}
    h2 {{ font-size: 15px; margin-top: 24px; }}
    .muted {{ color: #4b5563; font-size: 12px; }}
    .grid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; }}
    .box {{ border: 1px solid #d1d5db; padding: 10px; border-radius: 6px; }}
    table {{ border-collapse: collapse; width: 100%; margin-top: 10px; }}
    td, th {{ border: 1px solid #d1d5db; padding: 7px; font-size: 12px; text-align: left; }}
    th {{ background: #f3f4f6; }}
  </style>
</head>
<body>
  <h1>PayFlow FIU Evidence Package</h1>
  <div class="muted">Package {escape(package['package_id'])} | Case {escape(trace['case_id'])} | Model {escape(package['model_metadata']['model'])}</div>
  <h2>Case Summary</h2>
  <p>{escape(package['fiu_summary'])}</p>
  <div class="grid">
    <div class="box"><strong>Typologies</strong><br />{escape(typologies)}</div>
    <div class="box"><strong>Total Value</strong><br />{escape(trace['risk_scores']['total_amount_display'])}</div>
    <div class="box"><strong>Ledger Hash</strong><br />{escape(package['audit_hashes'].get('latest_block_hash') or 'not available')}</div>
  </div>
  <h2>Transaction Trail</h2>
  <table>
    <thead><tr><th>#</th><th>Txn</th><th>Path</th><th>Amount</th><th>Channel</th><th>Indicator</th></tr></thead>
    <tbody>{''.join(rows)}</tbody>
  </table>
  <h2>Recommended Actions</h2>
  <ul>{actions}</ul>
  <h2>Pre-Fraud Intelligence Context</h2>
  <p class="muted">External intelligence is preventive context only. PayFlow graph, ML, rules, and ledger evidence remain the decision authority.</p>
  <table>
    <thead><tr><th>Trend</th><th>Title</th><th>Trust</th><th>India Relevance</th></tr></thead>
    <tbody>{intel_rows}</tbody>
  </table>
  <h2>Active Adaptive Playbooks</h2>
  <table>
    <thead><tr><th>Playbook</th><th>Title</th><th>Audit Hash</th></tr></thead>
    <tbody>{playbook_rows}</tbody>
  </table>
  <h2>Adaptive Event Lab Countermeasures</h2>
  <p class="muted">Analyst-gated countermeasures require approval before execution. Qwen provides bounded explanations only.</p>
  <table>
    <thead><tr><th>Proposal</th><th>Action</th><th>Status</th><th>Targets</th><th>Audit Hash</th></tr></thead>
    <tbody>{counter_rows}</tbody>
  </table>
</body>
</html>"""


async def build_evidence_package(orch: Any, case_id: str) -> dict[str, Any]:
    trace = build_case_trace(orch, case_id)
    package_id = "FIU-" + hashlib.sha256(f"{case_id}-{time.time()}".encode()).hexdigest()[:10].upper()
    latest_hash = ""
    latest_index = None

    ledger = getattr(orch, "_ledger", None) if orch is not None else None
    if ledger is not None and hasattr(ledger, "get_stats"):
        try:
            stats = await ledger.get_stats()
            latest_hash = getattr(stats, "latest_hash", "") or ""
            latest_index = getattr(stats, "latest_index", None)
        except Exception:
            latest_hash = ""
            latest_index = None

    typologies = ", ".join(trace.get("ps3_typologies") or ["PS3 fund-flow anomaly"])
    summary = (
        f"Case {case_id} identifies {typologies} across "
        f"{trace['risk_scores']['transaction_count']} transaction events, total "
        f"{trace['risk_scores']['total_amount_display']}. Evidence includes graph path, "
        "transaction labels, model/runtime metadata, and audit-ledger references."
    )
    try:
        from src.intel import get_pre_fraud_intel_service

        pre_fraud_intel = get_pre_fraud_intel_service().evidence_context()
    except Exception:
        pre_fraud_intel = {
            "active_playbooks": [],
            "top_trends": [],
            "guardrail": "Pre-fraud intelligence context unavailable.",
        }

    if pre_fraud_intel.get("active_playbooks"):
        summary += (
            " Preventive intelligence was active before evidence packaging, "
            "with adaptive playbooks included as contextual watchlist support."
        )
    try:
        from src.simulation import get_event_lab_service

        event_lab_context = get_event_lab_service().evidence_context()
    except Exception:
        event_lab_context = {
            "event_lab_run_id": None,
            "countermeasure_proposals": [],
            "analyst_decisions": [],
            "executed_actions": [],
            "qwen_explanation": "",
            "pre_fraud_playbook": None,
            "countermeasure_audit_hashes": [],
        }

    if event_lab_context.get("event_lab_run_id"):
        summary += (
            " The Adaptive Event Lab captured preventive event generation, "
            "analyst-gated countermeasure decisions, and audit hashes before packaging."
        )

    package = {
        "package_id": package_id,
        "case_id": case_id,
        "fiu_summary": summary,
        "suspicious_indicators": trace.get("expected_indicators", []),
        "involved_entities": trace.get("account_roles", []),
        "transactions": trace.get("transaction_chain", []),
        "case_trace": trace,
        "audit_hashes": {
            "latest_block_hash": latest_hash,
            "latest_block_index": latest_index,
            "package_hash": hashlib.sha256(summary.encode()).hexdigest(),
        },
        "model_metadata": {
            "model": OLLAMA_CFG.model,
            "role": "grounded investigator copilot",
            "decision_authority": "graph_ml_rules_ledger_pipeline",
        },
        "pre_fraud_intelligence": pre_fraud_intel,
        "event_lab_run_id": event_lab_context.get("event_lab_run_id"),
        "countermeasure_proposals": event_lab_context.get("countermeasure_proposals", []),
        "analyst_decisions": event_lab_context.get("analyst_decisions", []),
        "executed_actions": event_lab_context.get("executed_actions", []),
        "qwen_explanation": event_lab_context.get("qwen_explanation", ""),
        "pre_fraud_playbook": event_lab_context.get("pre_fraud_playbook"),
        "countermeasure_audit_hashes": event_lab_context.get("countermeasure_audit_hashes", []),
        "json_payload": {},
        "generated_at": _now(),
    }
    package["json_payload"] = {
        key: value for key, value in package.items()
        if key not in {"json_payload", "printable_html"}
    }
    package["printable_html"] = _html_package(package)
    return package


def build_ps3_readiness(orch: Any) -> dict[str, Any]:
    snapshot: dict[str, Any] = {}
    if orch is not None and hasattr(type(orch), "full_snapshot"):
        try:
            snapshot = orch.full_snapshot()
        except Exception:
            snapshot = {}

    graph = snapshot.get("graph", {}) if isinstance(snapshot, dict) else {}
    orchestrator = snapshot.get("orchestrator", {}) if isinstance(snapshot, dict) else {}
    hardware = snapshot.get("hardware", {}) if isinstance(snapshot, dict) else {}
    simulation = snapshot.get("threat_simulation", {}) if isinstance(snapshot, dict) else {}
    try:
        from src.intel import get_pre_fraud_intel_service

        intel_status = get_pre_fraud_intel_service().tuning_status()
    except Exception:
        intel_status = {}

    requirements = [
        {
            "id": "fund_flow_visualization",
            "label": "End-to-end fund-flow visualization",
            "status": "ready",
            "evidence": "Graph topology, path timeline, and account-role map",
        },
        {
            "id": "graph_ml_detection",
            "label": "Graph analytics plus ML detection",
            "status": "ready",
            "evidence": "Existing graph, rule, ML, CFR, and circuit-breaker signals",
        },
        {
            "id": "ps3_typologies",
            "label": "PS3 typology coverage",
            "status": "ready",
            "evidence": "Layering, round-tripping, structuring, dormant activation, profile mismatch",
        },
        {
            "id": "investigator_trace",
            "label": "Investigator case tracing",
            "status": "ready",
            "evidence": "Case Workbench trace API and timeline",
        },
        {
            "id": "fiu_package",
            "label": "FIU evidence package generation",
            "status": "ready",
            "evidence": "JSON and printable HTML evidence package with audit hashes",
        },
    ]

    return {
        "title": "Union Bank iDEA 2.0 PS3 Readiness",
        "requirements": requirements,
        "runtime_health": {
            "orchestrator": orch is not None,
            "graph": bool(graph),
            "simulation": bool(simulation) or orch is not None,
            "qwen_model": OLLAMA_CFG.model,
            "single_port_app": "http://localhost:8010/app",
            "pre_fraud_intel": bool(intel_status),
            "active_intel_playbooks": intel_status.get("active_playbooks", 0),
        },
        "scale_metrics": {
            "events_ingested": orchestrator.get("events_ingested", 0),
            "events_per_sec": orchestrator.get("events_per_sec", 0),
            "graph_nodes": (graph.get("graph") or {}).get("nodes", 0) if isinstance(graph, dict) else 0,
            "graph_edges": (graph.get("graph") or {}).get("edges", 0) if isinstance(graph, dict) else 0,
            "active_scenarios": simulation.get("active_attacks", 0) if isinstance(simulation, dict) else 0,
            "gpu_vram_free_mb": hardware.get("gpu_vram_free_mb", 0),
            "llm_tokens_total": hardware.get("llm_tokens_total", 0),
            "intel_queue_depth": (intel_status.get("bounded_queue", {}) or {}).get("depth", 0),
        },
        "pilot_architecture": [
            "Pre-fraud intelligence watches official Indian sources, public news, and compliant public social signals before transaction anomalies peak",
            "CDC or streaming ingest from core banking, UPI, NEFT, RTGS, cards, and branch channels",
            "Feature and graph services partitioned by account/customer/entity ids",
            "Rules and ML remain deterministic decision layers; Qwen produces grounded case narratives",
            "Immutable evidence ledger stores case, verdict, and package hashes for audit review",
        ],
        "generated_at": _now(),
    }
