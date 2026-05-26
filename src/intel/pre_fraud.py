"""Union Bank tuned pre-fraud intelligence layer.

This module keeps external OSINT/SOCMINT signals separate from PayFlow's
authoritative fraud decision path. Signals can shape watchlists and Qwen
context, but promotion is guarded by trust, corroboration, TTL, and rollback
metadata.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError, as_completed
from dataclasses import asdict, dataclass, field
from html import unescape
from html.parser import HTMLParser
from datetime import datetime, timezone
from typing import Any
from urllib.parse import parse_qs, quote_plus, unquote, urlencode, urljoin, urlparse
from urllib.request import Request, urlopen

from config.settings import OLLAMA_CFG


SOURCE_TIER_TRUST = {
    "tier_0": 0.90,
    "tier_1": 0.74,
    "tier_2": 0.54,
    "tier_3": 0.34,
}

SOURCE_TIER_LABELS = {
    "tier_0": "Official regulator, public authority, or payment network",
    "tier_1": "Reputable finance, cyber, or business news",
    "tier_2": "Compliant public social or community signal",
    "tier_3": "Unverified raw open-web signal",
}

PROMOTION_TRUST_THRESHOLD = 0.85
PLAYBOOK_TTL_SECONDS = 7 * 24 * 60 * 60
REAL_IMAGE_ORIGINS = {"live_image", "open_graph_image", "gdelt_social_image", "bing_news_image"}
REAL_VIDEO_ORIGINS = {"real_video_embed"}
SOURCE_CARD_STATUSES = {"source_card", "publisher_logo_only"}

HTTP_USER_AGENT = (
    "PayFlow-UnionBank-PS3-PoC/1.0 "
    "(bounded public OSINT media resolver; no private scraping)"
)

PUBLIC_FRAUD_QUERIES = [
    "UPI fraud mule account India",
    "digital arrest scam India bank account mule",
    "KYC phishing APK banking India",
    "QR code fraud UPI merchant misuse India",
    "SIM swap banking fraud India UPI",
    "loan app extortion mule account India",
    "investment scam UPI deposits India",
    "dormant bank account fraud mule India",
    "account rent mule UPI India",
]

PUBLIC_FRAUD_VIDEO_QUERIES = [
    "CyberDost I4C digital arrest scam YouTube",
    "digital arrest scam Cyber Dost YouTube",
    "MHA cyber dost digital arrest video",
    "UPI fraud mule account India video",
]


TYPOLOGY_KEYWORDS: dict[str, list[str]] = {
    "UPI_MULE_NETWORK": ["upi", "mule", "cashout", "cash-out", "qr", "vpa"],
    "DIGITAL_ARREST": ["digital arrest", "courier fraud", "law enforcement impersonation"],
    "KYC_UPDATE_PHISHING": ["kyc", "phishing", "smishing", "apk", "remote access"],
    "LOAN_APP_EXTORTION": ["loan app", "extortion", "contact list", "instant loan"],
    "INVESTMENT_SCAM": ["investment", "trading", "crypto", "task scam", "pig butchering"],
    "SIM_SWAP": ["sim swap", "sim-swap", "otp", "duplicate sim"],
    "MERCHANT_QR_MISUSE": ["merchant qr", "qr code", "collect request"],
    "DORMANT_ACTIVATION": ["dormant", "reactivation", "inactive account"],
    "LAYERING": ["layering", "multi-hop", "hop", "rapid forwarding"],
    "STRUCTURING": ["structuring", "below threshold", "split transactions", "sub-threshold"],
    "ROUND_TRIPPING": ["round-tripping", "round tripping", "circular"],
    "PROFILE_MISMATCH": ["profile mismatch", "income mismatch", "behaviour mismatch", "behavior mismatch"],
}

CHANNEL_KEYWORDS: dict[str, list[str]] = {
    "UPI": ["upi", "vpa", "qr", "collect"],
    "IMPS": ["imps"],
    "NEFT": ["neft"],
    "RTGS": ["rtgs"],
    "CARDS": ["card", "debit", "credit"],
    "BRANCH": ["branch", "cash", "counter"],
    "DIGITAL_BANKING": ["mobile banking", "internet banking", "apk", "otp", "kyc"],
}

INDIA_RELEVANCE_TERMS = [
    "india",
    "indian",
    "rbi",
    "npci",
    "upi",
    "imps",
    "neft",
    "rtgs",
    "fiu-ind",
    "i4c",
    "cert-in",
    "mha",
    "kyc",
    "aadhaar",
    "pan",
    "hindi",
    "hinglish",
    "union bank",
]

INDIA_GEO_BASELINE = [
    {"label": "Delhi NCR", "lat": 28.6139, "lng": 77.2090, "channels": ["UPI", "DIGITAL_BANKING"]},
    {"label": "Mumbai", "lat": 19.0760, "lng": 72.8777, "channels": ["UPI", "IMPS", "RTGS"]},
    {"label": "Bengaluru", "lat": 12.9716, "lng": 77.5946, "channels": ["DIGITAL_BANKING", "UPI"]},
    {"label": "Hyderabad", "lat": 17.3850, "lng": 78.4867, "channels": ["UPI", "IMPS"]},
    {"label": "Kolkata", "lat": 22.5726, "lng": 88.3639, "channels": ["UPI", "NEFT"]},
    {"label": "Chennai", "lat": 13.0827, "lng": 80.2707, "channels": ["IMPS", "DIGITAL_BANKING"]},
    {"label": "Pune", "lat": 18.5204, "lng": 73.8567, "channels": ["UPI", "DIGITAL_BANKING"]},
    {"label": "Ahmedabad", "lat": 23.0225, "lng": 72.5714, "channels": ["UPI", "CARDS"]},
    {"label": "Jaipur", "lat": 26.9124, "lng": 75.7873, "channels": ["UPI", "BRANCH"]},
    {"label": "Lucknow", "lat": 26.8467, "lng": 80.9462, "channels": ["UPI", "NEFT"]},
    {"label": "Kochi", "lat": 9.9312, "lng": 76.2673, "channels": ["IMPS", "DIGITAL_BANKING"]},
    {"label": "Guwahati", "lat": 26.1445, "lng": 91.7362, "channels": ["UPI", "BRANCH"]},
    {"label": "Bhopal", "lat": 23.2599, "lng": 77.4126, "channels": ["UPI", "NEFT"]},
    {"label": "Indore", "lat": 22.7196, "lng": 75.8577, "channels": ["UPI", "DIGITAL_BANKING"]},
    {"label": "Nagpur", "lat": 21.1458, "lng": 79.0882, "channels": ["IMPS", "UPI"]},
    {"label": "Surat", "lat": 21.1702, "lng": 72.8311, "channels": ["UPI", "CARDS"]},
    {"label": "Patna", "lat": 25.5941, "lng": 85.1376, "channels": ["UPI", "BRANCH"]},
    {"label": "Bhubaneswar", "lat": 20.2961, "lng": 85.8245, "channels": ["UPI", "DIGITAL_BANKING"]},
    {"label": "Chandigarh", "lat": 30.7333, "lng": 76.7794, "channels": ["UPI", "NEFT"]},
    {"label": "Coimbatore", "lat": 11.0168, "lng": 76.9558, "channels": ["IMPS", "DIGITAL_BANKING"]},
    {"label": "Visakhapatnam", "lat": 17.6868, "lng": 83.2185, "channels": ["UPI", "RTGS"]},
    {"label": "Thiruvananthapuram", "lat": 8.5241, "lng": 76.9366, "channels": ["UPI", "DIGITAL_BANKING"]},
    {"label": "Ranchi", "lat": 23.3441, "lng": 85.3096, "channels": ["UPI", "BRANCH"]},
    {"label": "Siliguri", "lat": 26.7271, "lng": 88.3953, "channels": ["UPI", "NEFT"]},
    {"label": "Dubai-India corridor", "lat": 25.2048, "lng": 55.2708, "channels": ["UPI", "IMPS"]},
    {"label": "Singapore-India corridor", "lat": 1.3521, "lng": 103.8198, "channels": ["UPI", "RTGS"]},
]


@dataclass
class SourceConfig:
    source_id: str
    name: str
    tier: str
    category: str
    jurisdiction: str
    url: str
    poll_interval_sec: int
    enabled: bool
    terms_mode: str
    last_polled_at: float | None = None
    last_status: str = "not_polled"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class MediaPreview:
    media_id: str
    media_type: str
    source_kind: str
    title: str
    caption: str
    thumbnail_key: str
    source_url: str
    publisher: str
    language: str
    duration_sec: int | None = None
    published_at: float | None = None
    image_url: str | None = None
    source_domain: str | None = None
    media_origin: str = "generated_poster"
    media_url: str | None = None
    thumbnail_url: str | None = None
    image_status: str = "generated_fallback"
    fallback_reason: str | None = None
    license_hint: str = "public-source metadata; verify before external reuse"
    fetched_at: float | None = None
    publisher_logo_url: str | None = None
    preview_status: str = "generated_fallback"
    is_real_media: bool = False
    video_embed_url: str | None = None
    video_page_url: str | None = None
    video_provider: str | None = None
    embed_allowed: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ExternalThreatSignal:
    signal_id: str
    source_id: str
    source_name: str
    source_tier: str
    title: str
    normalized_text: str
    url: str
    observed_at: float
    region: str
    language: str
    entities: list[str]
    typologies: list[str]
    affected_channels: list[str]
    trust_score: float
    confidence: float
    corroboration_ids: list[str] = field(default_factory=list)
    sovereignty_tags: list[str] = field(default_factory=list)
    media_preview: dict[str, Any] = field(default_factory=dict)
    geo_scope: list[dict[str, Any]] = field(default_factory=list)
    public_reach_score: float = 0.0
    signal_velocity_score: float = 0.0
    audit_hash: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class FraudTrendCluster:
    trend_id: str
    title: str
    typologies: list[str]
    affected_channels: list[str]
    first_seen: float
    last_seen: float
    velocity_score: float
    reach_score: float
    india_relevance_score: float
    evidence_count: int
    source_tiers: list[str]
    evidence_ids: list[str]
    trust_score: float
    audit_hash: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AdaptivePlaybook:
    playbook_id: str
    trend_id: str
    title: str
    prompt_context: str
    rule_deltas: dict[str, float]
    risk_weight_deltas: dict[str, float]
    watchlist_terms: list[str]
    scenario_seed: str
    ttl_seconds: int
    expires_at: float
    promotion_status: str
    promotion_reason: str
    rollback_available: bool
    audit_hash: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class PreFraudIntelService:
    """In-memory prototype service for preventive fraud intelligence."""

    def __init__(self) -> None:
        self._sources = self._default_sources()
        self._signals: dict[str, ExternalThreatSignal] = {}
        self._trends: dict[str, FraudTrendCluster] = {}
        self._playbooks: dict[str, AdaptivePlaybook] = {}
        self._audit_log: list[dict[str, Any]] = []
        self._last_refresh_at: float | None = None
        self._bounded_queue_depth = 0
        self._max_queue_depth = 64
        self._pulse_seq = 0

    # -- Public state -----------------------------------------------------

    def reset(self) -> None:
        self._signals.clear()
        self._trends.clear()
        self._playbooks.clear()
        self._audit_log.clear()
        self._last_refresh_at = None
        self._bounded_queue_depth = 0
        self._pulse_seq = 0
        for source in self._sources.values():
            source.last_polled_at = None
            source.last_status = "not_polled"

    def list_sources(self) -> dict[str, Any]:
        return {
            "sources": [s.to_dict() for s in self._sources.values()],
            "trust_policy": {
                "tiers": SOURCE_TIER_LABELS,
                "base_trust": SOURCE_TIER_TRUST,
                "promotion_threshold": PROMOTION_TRUST_THRESHOLD,
                "promotion_rule": (
                    "Official Tier 0 evidence can promote directly; otherwise "
                    "signals need independent source-tier corroboration."
                ),
                "guardrails": [
                    "No private group or credentialed scraping",
                    "Low-trust social signals remain advisory unless corroborated",
                    "All playbook deltas are capped, TTL-bound, and rollback-ready",
                    "Qwen updates context and narratives; graph, ML, rules, and ledger remain authoritative",
                ],
            },
            "last_refresh_at": self._last_refresh_at,
        }

    def refresh(self, seed: int | None = None) -> dict[str, Any]:
        """Run a bounded refresh across public-source adapters with safe fallback."""
        self._bounded_queue_depth = min(self._max_queue_depth, self._bounded_queue_depth + 1)
        try:
            added: list[ExternalThreatSignal] = []
            payloads = self._curated_refresh_payloads(seed=seed)
            if seed is None:
                payloads.extend(self._live_public_payloads())

            for payload in payloads:
                source = self._sources[payload["source_id"]]
                if not source.enabled:
                    continue
                signal = self.ingest_signal(payload)
                added.append(signal)
                source.last_polled_at = _now()
                source.last_status = "ok"

            self._rebuild_intelligence()
            self._last_refresh_at = _now()
            self._audit("refresh", {"signals_added": len(added)})
            return {
                "status": "refreshed",
                "signals_added": len(added),
                "signals": [s.to_dict() for s in added],
                "trends": self.list_trends()["trends"],
                "playbooks": self.list_playbooks()["playbooks"],
                "tuning_status": self.tuning_status(),
            }
        finally:
            self._bounded_queue_depth = max(0, self._bounded_queue_depth - 1)

    def ingest_signal(self, payload: dict[str, Any]) -> ExternalThreatSignal:
        source = self._sources[payload["source_id"]]
        observed_at = float(payload.get("observed_at") or _now())
        text = _normalize_text(f"{payload.get('title', '')} {payload.get('text', '')}")
        typologies = payload.get("typologies") or classify_typologies(text)
        affected_channels = payload.get("affected_channels") or classify_channels(text)
        entities = payload.get("entities") or extract_entities(text)
        region = str(payload.get("region") or "India")
        language = str(payload.get("language") or detect_language(text))
        sovereignty_tags = sovereignty_tags_for(text, region)
        trust_score = self._base_trust_score(
            source=source,
            text=text,
            typologies=typologies,
            region=region,
            observed_at=observed_at,
        )
        signal_id = _stable_id("SIG", source.source_id, payload.get("title", ""), text)
        media_preview = payload.get("media_preview") or media_preview_for(
            source=source,
            title=str(payload.get("title") or "External fraud signal"),
            text=text,
            typologies=typologies,
            observed_at=observed_at,
            url=str(payload.get("url") or source.url),
            language=language,
            image_url=payload.get("image_url"),
            source_domain=payload.get("source_domain"),
            publisher_name=payload.get("publisher_name"),
            video_embed_url=payload.get("video_embed_url"),
            video_page_url=payload.get("video_page_url"),
            video_provider=payload.get("video_provider"),
            embed_allowed=bool(payload.get("embed_allowed")),
        )
        geo_scope = payload.get("geo_scope") or geo_scope_for(region=region, typologies=typologies)
        public_reach = payload.get("public_reach_score")
        velocity = payload.get("signal_velocity_score")

        signal = ExternalThreatSignal(
            signal_id=signal_id,
            source_id=source.source_id,
            source_name=source.name,
            source_tier=source.tier,
            title=str(payload.get("title") or "Untitled external fraud signal"),
            normalized_text=text,
            url=str(payload.get("url") or source.url),
            observed_at=observed_at,
            region=region,
            language=language,
            entities=entities[:16],
            typologies=typologies[:8],
            affected_channels=affected_channels[:8],
            trust_score=trust_score,
            confidence=round(min(0.98, trust_score + 0.04), 4),
            sovereignty_tags=sovereignty_tags,
            media_preview=media_preview,
            geo_scope=geo_scope,
            public_reach_score=round(float(public_reach) if public_reach is not None else public_reach_for(source, text), 4),
            signal_velocity_score=round(float(velocity) if velocity is not None else signal_velocity_for(text, observed_at), 4),
        )
        signal.audit_hash = _hash_json(signal.to_dict() | {"audit_hash": ""})
        self._signals[signal.signal_id] = signal
        self._audit("signal_ingested", {"signal_id": signal.signal_id, "source_id": source.source_id})
        return signal

    def simulate_signal(self, scenario: str = "digital_arrest_mule") -> dict[str, Any]:
        payload = self._simulation_payload(scenario)
        signal = self.ingest_signal(payload)
        self._rebuild_intelligence()
        self._audit("signal_simulated", {"scenario": scenario, "signal_id": signal.signal_id})
        return {
            "status": "simulated",
            "scenario": scenario,
            "signal": signal.to_dict(),
            "trends": self.list_trends()["trends"],
            "playbooks": self.list_playbooks()["playbooks"],
            "tuning_status": self.tuning_status(),
        }

    def list_signals(
        self,
        typology: str | None = None,
        region: str | None = None,
        source_tier: str | None = None,
        min_trust: float | None = None,
        since: float | None = None,
    ) -> dict[str, Any]:
        signals = list(self._signals.values())
        if typology:
            typology_upper = typology.upper()
            signals = [s for s in signals if typology_upper in s.typologies]
        if region:
            region_lower = region.lower()
            signals = [s for s in signals if region_lower in s.region.lower()]
        if source_tier:
            signals = [s for s in signals if s.source_tier == source_tier]
        if min_trust is not None:
            signals = [s for s in signals if s.trust_score >= min_trust]
        if since is not None:
            signals = [s for s in signals if s.observed_at >= since]

        signals.sort(key=lambda s: (s.trust_score, s.observed_at), reverse=True)
        return {
            "count": len(signals),
            "signals": [s.to_dict() for s in signals],
            "last_refresh_at": self._last_refresh_at,
        }

    def list_trends(self) -> dict[str, Any]:
        trends = sorted(
            self._trends.values(),
            key=lambda t: (t.trust_score, t.velocity_score, t.last_seen),
            reverse=True,
        )
        return {
            "count": len(trends),
            "trends": [t.to_dict() for t in trends],
            "generated_at": _now(),
        }

    def list_playbooks(self) -> dict[str, Any]:
        playbooks = sorted(
            self._playbooks.values(),
            key=lambda p: (p.promotion_status == "applied", p.expires_at),
            reverse=True,
        )
        return {
            "count": len(playbooks),
            "playbooks": [p.to_dict() for p in playbooks],
            "tuning_status": self.tuning_status(),
        }

    def cockpit(self) -> dict[str, Any]:
        """Return a judge-facing cockpit dataset for live OSINT/SOCMINT visualization."""
        self._pulse_seq += 1
        signals = sorted(
            self._signals.values(),
            key=lambda s: (s.trust_score, s.signal_velocity_score, s.observed_at),
            reverse=True,
        )
        trends = sorted(
            self._trends.values(),
            key=lambda t: (t.trust_score, t.velocity_score, t.last_seen),
            reverse=True,
        )
        playbooks = sorted(
            self._playbooks.values(),
            key=lambda p: (p.promotion_status == "applied", p.expires_at),
            reverse=True,
        )
        now = _now()
        source_mix = source_mix_for(signals, self._sources)
        channel_exposure = channel_exposure_for(signals, now)
        typology_matrix = typology_matrix_for(signals)
        timeline = signal_timeline_for(signals, now)
        geo_hotspots = geo_hotspots_for(signals, trends, now)
        geo_links = geo_links_for(geo_hotspots)
        media_previews = [
            signal.media_preview
            | {
                "signal_id": signal.signal_id,
                "source_id": signal.source_id,
                "source_tier": signal.source_tier,
                "trust_score": signal.trust_score,
                "typologies": signal.typologies,
                "affected_channels": signal.affected_channels,
                "region": signal.region,
            }
            for signal in signals
            if signal.media_preview
        ]
        media_previews = sorted(
            media_previews,
            key=lambda item: (
                media_preview_rank(item),
                float(item.get("trust_score") or 0.0),
                float(item.get("published_at") or 0.0),
            ),
            reverse=True,
        )[:18]
        fusion_nodes, fusion_links = fusion_graph_for(signals, trends)
        trust_index = round(max((s.trust_score for s in signals), default=0.0), 4)
        india_fit = round(max((_india_relevance(s.normalized_text, s.region) for s in signals), default=0.0), 4)
        live_mentions = int(sum(point["total"] for point in timeline[-6:]))
        freshness_sec = int(now - max((s.observed_at for s in signals), default=now))
        source_health = source_health_for(self._sources)
        live_sources = sum(1 for source in self._sources.values() if source.last_status == "ok")
        corroboration_rate = corroboration_rate_for(signals)
        source_velocity_series = source_velocity_series_for(signals, now)
        typology_velocity_series = typology_velocity_series_for(signals, now)
        channel_typology_heatmap = channel_typology_heatmap_for(signals)
        geo_layers = geo_layers_for(geo_hotspots)
        media_matrix = media_evidence_matrix_for(media_previews)
        media_summary = media_summary_for(media_previews, self._sources, now)
        playbook_impact = playbook_impact_series_for(playbooks, trends, now)
        freshness_sla = source_freshness_sla_for(self._sources, now)

        return {
            "generated_at": now,
            "cadence": "bounded_public_signal_refresh",
            "live_state": {
                "pulse_seq": self._pulse_seq,
                "last_refresh_at": self._last_refresh_at,
                "freshness_sec": max(0, freshness_sec),
                "live_sources": live_sources,
                "source_health": source_health,
                "public_mode": "live_public_api" if any(
                    source.last_status == "ok" and source.source_id == "gdelt_fincrime_news"
                    for source in self._sources.values()
                ) else "seeded_public_signal_mix",
            },
            "metrics": {
                "signal_count": len(signals),
                "trend_count": len(trends),
                "active_sources": sum(1 for source in self._sources.values() if source.enabled),
                "active_playbooks": self.tuning_status()["active_playbooks"],
                "trust_index": trust_index,
                "india_fit": india_fit,
                "media_items": len(media_previews),
                "live_media_items": media_summary["live_media"],
                "real_images": media_summary["real_images"],
                "real_videos": media_summary["real_videos"],
                "source_cards": media_summary["source_cards"],
                "publisher_logo_only": media_summary["publisher_logo_only"],
                "generated_fallbacks": media_summary["generated_fallbacks"],
                "broken": media_summary["broken"],
                "stale_sources": media_summary["stale_sources"],
                "last_successful_poll": media_summary["last_successful_poll"],
                "media_health": media_summary["health"],
                "live_mentions": live_mentions,
                "velocity_index": velocity_index_for(signals, timeline),
                "map_coverage": round(min(1.0, len(geo_hotspots) / 24), 4),
                "corroboration_rate": corroboration_rate,
                "freshness_sec": max(0, freshness_sec),
            },
            "source_mix": source_mix,
            "channel_exposure": channel_exposure,
            "typology_matrix": typology_matrix,
            "signal_timeline": timeline,
            "geo_hotspots": geo_hotspots,
            "geo_links": geo_links,
            "geo_layers": geo_layers,
            "media_previews": media_previews,
            "media_evidence_matrix": media_matrix,
            "fusion_graph": {
                "nodes": fusion_nodes,
                "links": fusion_links,
            },
            "source_velocity_series": source_velocity_series,
            "typology_velocity_series": typology_velocity_series,
            "channel_typology_heatmap": channel_typology_heatmap,
            "playbook_impact_series": playbook_impact,
            "source_freshness_sla": freshness_sla,
            "corroboration_network": corroboration_network_for(signals),
            "social_pulse": social_pulse_for(signals, now),
            "top_trends": [trend.to_dict() for trend in trends[:5]],
            "active_playbooks": [playbook.to_dict() for playbook in playbooks[:5]],
            "guardrails": self.tuning_status(),
        }

    def media(self) -> dict[str, Any]:
        cockpit = self.cockpit()
        previews = cockpit["media_previews"]
        return {
            "generated_at": cockpit["generated_at"],
            "count": len(previews),
            "summary": {
                "live_media": cockpit["metrics"].get("live_media_items", 0),
                "real_images": cockpit["metrics"].get("real_images", 0),
                "real_videos": cockpit["metrics"].get("real_videos", 0),
                "source_cards": cockpit["metrics"].get("source_cards", 0),
                "publisher_logo_only": cockpit["metrics"].get("publisher_logo_only", 0),
                "generated_fallbacks": cockpit["metrics"].get("generated_fallbacks", 0),
                "broken": cockpit["metrics"].get("broken", 0),
                "stale_sources": cockpit["metrics"].get("stale_sources", 0),
                "last_successful_poll": cockpit["metrics"].get("last_successful_poll"),
                "health": cockpit["metrics"].get("media_health", 0.0),
            },
            "media_previews": previews,
            "media_evidence_matrix": cockpit["media_evidence_matrix"],
        }

    def tuning_status(self) -> dict[str, Any]:
        now = _now()
        playbooks = list(self._playbooks.values())
        applied = [p for p in playbooks if p.promotion_status == "applied" and p.expires_at > now]
        shadow = [p for p in playbooks if p.promotion_status == "shadow" and p.expires_at > now]
        advisory = [p for p in playbooks if p.promotion_status == "advisory" and p.expires_at > now]
        return {
            "active_playbooks": len(applied),
            "shadow_changes": len(shadow),
            "advisory_changes": len(advisory),
            "applied_changes": [p.playbook_id for p in applied],
            "rollback_available": any(p.rollback_available for p in applied),
            "last_refresh_at": self._last_refresh_at,
            "bounded_queue": {
                "depth": self._bounded_queue_depth,
                "max_depth": self._max_queue_depth,
                "state": "bounded",
            },
            "qwen_model": OLLAMA_CFG.model,
            "decision_authority": "graph_ml_rules_ledger_pipeline",
        }

    def active_context_for_ai(self) -> dict[str, Any]:
        now = _now()
        active = [
            p for p in self._playbooks.values()
            if p.promotion_status == "applied" and p.expires_at > now
        ]
        top_trends = sorted(
            self._trends.values(),
            key=lambda t: (t.trust_score, t.velocity_score),
            reverse=True,
        )[:3]
        return {
            "source": "PayFlow Pre-Fraud Intel Radar",
            "role": "preventive_context_only",
            "active_playbooks": [
                {
                    "playbook_id": p.playbook_id,
                    "title": p.title,
                    "prompt_context": p.prompt_context,
                    "watchlist_terms": p.watchlist_terms[:10],
                    "expires_at": p.expires_at,
                    "audit_hash": p.audit_hash,
                }
                for p in active[:3]
            ],
            "top_trends": [
                {
                    "trend_id": t.trend_id,
                    "title": t.title,
                    "typologies": t.typologies,
                    "trust_score": t.trust_score,
                    "india_relevance_score": t.india_relevance_score,
                    "evidence_count": t.evidence_count,
                }
                for t in top_trends
            ],
            "guardrail": (
                "Use this as context for questions, hypotheses, watchlists, "
                "and narratives. Do not override internal graph, ML, rules, "
                "or ledger evidence."
            ),
        }

    def evidence_context(self) -> dict[str, Any]:
        context = self.active_context_for_ai()
        context["tuning_status"] = self.tuning_status()
        context["source_count"] = len(self._sources)
        context["signal_count"] = len(self._signals)
        context["trend_count"] = len(self._trends)
        return context

    def validate_llm_extraction(self, raw_text: str) -> dict[str, Any]:
        """Validate structured LLM extraction before it can become a signal."""
        try:
            parsed = json.loads(raw_text)
        except json.JSONDecodeError:
            result = {
                "status": "quarantined",
                "reason": "invalid_json",
                "raw_hash": _hash_text(raw_text),
            }
            self._audit("llm_extraction_quarantined", result)
            return result

        required = {"title", "text", "source_id"}
        if not isinstance(parsed, dict) or not required.issubset(parsed):
            result = {
                "status": "quarantined",
                "reason": "missing_required_fields",
                "raw_hash": _hash_text(raw_text),
            }
            self._audit("llm_extraction_quarantined", result)
            return result
        if parsed["source_id"] not in self._sources:
            result = {
                "status": "quarantined",
                "reason": "unknown_source",
                "raw_hash": _hash_text(raw_text),
            }
            self._audit("llm_extraction_quarantined", result)
            return result
        return {"status": "valid", "payload": parsed}

    # -- Internal scoring -------------------------------------------------

    def _base_trust_score(
        self,
        source: SourceConfig,
        text: str,
        typologies: list[str],
        region: str,
        observed_at: float,
    ) -> float:
        score = SOURCE_TIER_TRUST.get(source.tier, 0.25)
        if typologies:
            score += 0.04
        if _india_relevance(text, region) >= 0.6:
            score += 0.04
        age_hours = max(0.0, (_now() - observed_at) / 3600)
        if age_hours <= 48:
            score += 0.02
        if any(term in text for term in ["debunked", "false alarm", "rumour", "rumor"]):
            score -= 0.18
        return round(max(0.05, min(0.98, score)), 4)

    def _rebuild_intelligence(self) -> None:
        self._update_corroboration()
        self._rebuild_trends()
        self._rebuild_playbooks()

    def _update_corroboration(self) -> None:
        signals = list(self._signals.values())
        for signal in signals:
            related = []
            tiers = set()
            for other in signals:
                if other.signal_id == signal.signal_id:
                    continue
                overlap = set(signal.typologies) & set(other.typologies)
                if overlap:
                    related.append(other.signal_id)
                    tiers.add(other.source_tier)
            signal.corroboration_ids = sorted(related)[:12]
            if related:
                boost = 0.03 + min(0.08, 0.025 * len(tiers))
                signal.trust_score = round(min(0.98, signal.trust_score + boost), 4)
                signal.confidence = round(min(0.99, signal.trust_score + 0.03), 4)
                signal.audit_hash = _hash_json(signal.to_dict() | {"audit_hash": ""})

    def _rebuild_trends(self) -> None:
        grouped: dict[str, list[ExternalThreatSignal]] = {}
        for signal in self._signals.values():
            typology = signal.typologies[0] if signal.typologies else "UNCLASSIFIED"
            grouped.setdefault(typology, []).append(signal)

        trends: dict[str, FraudTrendCluster] = {}
        for typology, signals in grouped.items():
            ids = sorted(s.signal_id for s in signals)
            trend_id = _stable_id("TRD", typology, *ids)
            first_seen = min(s.observed_at for s in signals)
            last_seen = max(s.observed_at for s in signals)
            source_tiers = sorted({s.source_tier for s in signals})
            channels = sorted({ch for s in signals for ch in s.affected_channels}) or ["DIGITAL_BANKING"]
            trust = round(max(s.trust_score for s in signals), 4)
            india_score = round(max(_india_relevance(s.normalized_text, s.region) for s in signals), 4)
            velocity = round(min(1.0, 0.28 + 0.13 * len(signals) + (0.12 if "tier_0" in source_tiers else 0.0)), 4)
            reach = round(min(1.0, 0.22 + 0.12 * len(channels) + 0.08 * len(source_tiers)), 4)
            title = trend_title_for(typology, channels, len(signals))
            audit_hash = _hash_json({
                "trend_id": trend_id,
                "typology": typology,
                "evidence_ids": ids,
                "trust": trust,
            })
            trends[trend_id] = FraudTrendCluster(
                trend_id=trend_id,
                title=title,
                typologies=[typology],
                affected_channels=channels,
                first_seen=first_seen,
                last_seen=last_seen,
                velocity_score=velocity,
                reach_score=reach,
                india_relevance_score=india_score,
                evidence_count=len(signals),
                source_tiers=source_tiers,
                evidence_ids=ids,
                trust_score=trust,
                audit_hash=audit_hash,
            )
        self._trends = trends

    def _rebuild_playbooks(self) -> None:
        playbooks: dict[str, AdaptivePlaybook] = {}
        for trend in self._trends.values():
            signals = [self._signals[sid] for sid in trend.evidence_ids if sid in self._signals]
            has_tier0 = any(s.source_tier == "tier_0" for s in signals)
            independent_tiers = {s.source_tier for s in signals}
            if trend.trust_score >= PROMOTION_TRUST_THRESHOLD and (has_tier0 or len(independent_tiers) >= 2):
                status = "applied"
                reason = "high_trust_official_or_independent_corroboration"
            elif trend.trust_score >= 0.65:
                status = "shadow"
                reason = "moderate_trust_shadow_only"
            else:
                status = "advisory"
                reason = "low_trust_advisory_only"

            typology = trend.typologies[0] if trend.typologies else "UNCLASSIFIED"
            playbook_id = _stable_id("PBK", trend.trend_id, status)
            terms = watchlist_terms_for(typology)
            risk_deltas = capped_deltas_for(typology)
            expires_at = _now() + PLAYBOOK_TTL_SECONDS
            prompt_context = (
                f"Emerging India banking trend: {trend.title}. "
                f"Typology {typology}, affected channels {', '.join(trend.affected_channels)}. "
                f"Treat as preventive context only; cite internal PayFlow evidence before any action."
            )
            audit_hash = _hash_json({
                "playbook_id": playbook_id,
                "trend_id": trend.trend_id,
                "status": status,
                "deltas": risk_deltas,
                "terms": terms,
            })
            playbooks[playbook_id] = AdaptivePlaybook(
                playbook_id=playbook_id,
                trend_id=trend.trend_id,
                title=trend.title,
                prompt_context=prompt_context,
                rule_deltas={k: round(v / 2, 4) for k, v in risk_deltas.items()},
                risk_weight_deltas=risk_deltas,
                watchlist_terms=terms,
                scenario_seed=scenario_seed_for(typology),
                ttl_seconds=PLAYBOOK_TTL_SECONDS,
                expires_at=expires_at,
                promotion_status=status,
                promotion_reason=reason,
                rollback_available=status == "applied",
                audit_hash=audit_hash,
            )
        self._playbooks = playbooks

    # -- Fixtures / source policy ----------------------------------------

    def _default_sources(self) -> dict[str, SourceConfig]:
        sources = [
            SourceConfig(
                source_id="rbi_payment_security",
                name="Reserve Bank of India payment security publications",
                tier="tier_0",
                category="regulator",
                jurisdiction="IN",
                url="https://rbi.org.in/",
                poll_interval_sec=6 * 60 * 60,
                enabled=True,
                terms_mode="public_official_pages",
            ),
            SourceConfig(
                source_id="fiu_ind_public",
                name="FIU-IND public suspicious intelligence context",
                tier="tier_0",
                category="financial_intelligence_unit",
                jurisdiction="IN",
                url="https://fiuindia.gov.in/",
                poll_interval_sec=12 * 60 * 60,
                enabled=True,
                terms_mode="public_official_pages",
            ),
            SourceConfig(
                source_id="i4c_cybercrime",
                name="I4C / MHA cybercrime public advisories",
                tier="tier_0",
                category="cybercrime_coordination",
                jurisdiction="IN",
                url="https://i4c.mha.gov.in/",
                poll_interval_sec=6 * 60 * 60,
                enabled=True,
                terms_mode="public_official_pages",
            ),
            SourceConfig(
                source_id="npci_upi_safety",
                name="NPCI UPI product and safety signals",
                tier="tier_0",
                category="payment_network",
                jurisdiction="IN",
                url="https://www.npci.org.in/what-we-do/upi/product-statistics/",
                poll_interval_sec=6 * 60 * 60,
                enabled=True,
                terms_mode="public_official_pages",
            ),
            SourceConfig(
                source_id="certin_advisories",
                name="CERT-In cyber security advisories",
                tier="tier_0",
                category="cyber_advisory",
                jurisdiction="IN",
                url="https://www.cert-in.org.in/",
                poll_interval_sec=6 * 60 * 60,
                enabled=True,
                terms_mode="public_advisories",
            ),
            SourceConfig(
                source_id="gdelt_fincrime_news",
                name="GDELT and reputable finance news cluster",
                tier="tier_1",
                category="open_news",
                jurisdiction="global",
                url="https://www.gdeltproject.org/",
                poll_interval_sec=60 * 60,
                enabled=True,
                terms_mode="rss_or_public_api",
            ),
            SourceConfig(
                source_id="public_socmint_curated",
                name="Curated public SOCMINT signal set",
                tier="tier_2",
                category="public_social",
                jurisdiction="IN_diaspora",
                url="curated://public-social-fixtures",
                poll_interval_sec=60 * 60,
                enabled=True,
                terms_mode="compliant_public_api_or_fixture",
            ),
            SourceConfig(
                source_id="deep_open_web_curated",
                name="Curated deep open-web threat signal simulator",
                tier="tier_3",
                category="deep_open_web",
                jurisdiction="global",
                url="curated://deep-open-web-fixtures",
                poll_interval_sec=2 * 60 * 60,
                enabled=True,
                terms_mode="fixture_until_licensed_provider",
            ),
        ]
        return {source.source_id: source for source in sources}

    def _curated_refresh_payloads(self, seed: int | None = None) -> list[dict[str, Any]]:
        base = _now()
        return [
            {
                "source_id": "i4c_cybercrime",
                "title": "Digital arrest scam cash-out pattern uses Indian mule accounts",
                "text": "Public cybercrime warning references digital arrest impersonation, UPI transfers, mule accounts, and rapid cashout.",
                "typologies": ["DIGITAL_ARREST", "UPI_MULE_NETWORK", "LAYERING"],
                "affected_channels": ["UPI", "IMPS", "DIGITAL_BANKING"],
                "region": "India",
                "language": "en",
                "observed_at": base - 600,
            },
            {
                "source_id": "npci_upi_safety",
                "title": "UPI collect and QR misuse requires closer mule-network watch",
                "text": "UPI QR collect request misuse, mule VPA cashout, and rapid forwarding are rising public safety concerns.",
                "typologies": ["UPI_MULE_NETWORK", "MERCHANT_QR_MISUSE", "LAYERING"],
                "affected_channels": ["UPI", "DIGITAL_BANKING"],
                "region": "India",
                "language": "en",
                "observed_at": base - 900,
            },
            {
                "source_id": "certin_advisories",
                "title": "KYC update phishing and Android APK remote access lure",
                "text": "KYC update phishing, smishing, APK install, OTP capture, and remote access can precede account takeover.",
                "typologies": ["KYC_UPDATE_PHISHING", "SIM_SWAP"],
                "affected_channels": ["DIGITAL_BANKING", "UPI"],
                "region": "India",
                "language": "en",
                "observed_at": base - 1200,
            },
            {
                "source_id": "gdelt_fincrime_news",
                "title": "Indian finance news tracks loan app extortion and mule-account collections",
                "text": "Finance news cluster mentions instant loan app extortion, contact-list harassment, UPI repayments, and mule accounts.",
                "typologies": ["LOAN_APP_EXTORTION", "UPI_MULE_NETWORK"],
                "affected_channels": ["UPI", "DIGITAL_BANKING"],
                "region": "India",
                "language": "en",
                "observed_at": base - 1800,
            },
            {
                "source_id": "public_socmint_curated",
                "title": "Public Hinglish chatter offers student accounts for UPI cashout",
                "text": "Public social posts discuss account rent, mule account, UPI cashout, QR withdrawal, and quick commission in Hinglish.",
                "typologies": ["UPI_MULE_NETWORK", "LAYERING"],
                "affected_channels": ["UPI"],
                "region": "India diaspora",
                "language": "hinglish",
                "observed_at": base - 2400,
            },
            {
                "source_id": "deep_open_web_curated",
                "title": "Unverified open-web post claims dormant salary accounts for cashout",
                "text": "Unverified raw web signal claims dormant accounts can be reactivated for UPI and IMPS cashout; treat as advisory.",
                "typologies": ["DORMANT_ACTIVATION", "UPI_MULE_NETWORK"],
                "affected_channels": ["UPI", "IMPS"],
                "region": "India",
                "language": "en",
                "observed_at": base - 2700,
            },
        ]

    def _live_public_payloads(self) -> list[dict[str, Any]]:
        """Best-effort public-web intake; never blocks the demo if unavailable."""
        try:
            return fetch_gdelt_fincrime_payloads(max_records=24, timeout_sec=10.0)
        except Exception as exc:
            source = self._sources.get("gdelt_fincrime_news")
            if source is not None:
                source.last_polled_at = _now()
                source.last_status = f"live_fallback:{type(exc).__name__}"
            self._audit("live_source_fallback", {"source_id": "gdelt_fincrime_news", "reason": str(exc)[:160]})
            return []

    def _simulation_payload(self, scenario: str) -> dict[str, Any]:
        scenario_key = scenario.strip().lower().replace("-", "_")
        base = _now()
        scenarios = {
            "digital_arrest_mule": {
                "source_id": "i4c_cybercrime",
                "title": "High-confidence digital arrest to UPI mule cash-out burst",
                "text": (
                    "Digital arrest impersonation targeting Indian customers is followed by "
                    "UPI transfers to mule accounts, IMPS consolidation, and rapid cashout."
                ),
                "typologies": ["DIGITAL_ARREST", "UPI_MULE_NETWORK", "LAYERING"],
                "affected_channels": ["UPI", "IMPS", "DIGITAL_BANKING"],
                "region": "India",
                "language": "en",
                "observed_at": base,
            },
            "kyc_phishing": {
                "source_id": "certin_advisories",
                "title": "KYC smishing APK campaign targets mobile banking credentials",
                "text": "KYC update phishing, APK sideload, OTP theft, SIM swap, and mobile banking account takeover pattern.",
                "typologies": ["KYC_UPDATE_PHISHING", "SIM_SWAP"],
                "affected_channels": ["DIGITAL_BANKING", "UPI"],
                "region": "India",
                "language": "en",
                "observed_at": base,
            },
            "loan_app_mule": {
                "source_id": "gdelt_fincrime_news",
                "title": "Loan app extortion collections route into UPI mule networks",
                "text": "Instant loan app extortion and repayment pressure route funds through UPI mule accounts and collectors.",
                "typologies": ["LOAN_APP_EXTORTION", "UPI_MULE_NETWORK"],
                "affected_channels": ["UPI", "DIGITAL_BANKING"],
                "region": "India",
                "language": "en",
                "observed_at": base,
            },
            "investment_scam": {
                "source_id": "public_socmint_curated",
                "title": "Public posts describe fake trading task scam deposit chains",
                "text": "Fake trading investment scam, task scam, crypto return promise, UPI deposits, and collector accounts.",
                "typologies": ["INVESTMENT_SCAM", "STRUCTURING", "LAYERING"],
                "affected_channels": ["UPI", "IMPS"],
                "region": "India diaspora",
                "language": "en",
                "observed_at": base,
            },
        }
        return scenarios.get(scenario_key, scenarios["digital_arrest_mule"])

    def _audit(self, event_type: str, payload: dict[str, Any]) -> None:
        entry = {
            "event_type": event_type,
            "timestamp": _now(),
            "payload": payload,
        }
        entry["audit_hash"] = _hash_json(entry)
        self._audit_log.append(entry)
        self._audit_log = self._audit_log[-200:]


def classify_typologies(text: str) -> list[str]:
    hits = []
    for typology, keywords in TYPOLOGY_KEYWORDS.items():
        if any(keyword in text for keyword in keywords):
            hits.append(typology)
    if "mule" in text and "upi" in text and "UPI_MULE_NETWORK" not in hits:
        hits.append("UPI_MULE_NETWORK")
    return hits or ["UNCLASSIFIED"]


def classify_channels(text: str) -> list[str]:
    channels = []
    for channel, keywords in CHANNEL_KEYWORDS.items():
        if any(keyword in text for keyword in keywords):
            channels.append(channel)
    return channels or ["DIGITAL_BANKING"]


def fetch_gdelt_fincrime_payloads(max_records: int = 24, timeout_sec: float = 8.0) -> list[dict[str, Any]]:
    """Fetch a bounded multi-query Tier 1 public-news slice from GDELT Doc API."""
    bing_results = fetch_bing_news_payloads(
        PUBLIC_FRAUD_QUERIES,
        max_records=max(8, min(max_records - 4, 24)),
        timeout_sec=min(timeout_sec, 8.0),
    )
    bing_results.extend(fetch_bing_video_payloads(
        PUBLIC_FRAUD_VIDEO_QUERIES,
        max_records=min(4, max(1, max_records - len(bing_results))),
        timeout_sec=min(timeout_sec, 8.0),
    ))
    if bing_results:
        return bing_results

    per_query = max(2, min(5, max_records // max(1, len(PUBLIC_FRAUD_QUERIES) // 2)))
    budget = max(1, min(max_records, 40))
    results: list[dict[str, Any]] = []
    seen: set[str] = set()

    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [
            pool.submit(fetch_gdelt_query_payloads, query, per_query, timeout_sec / 2)
            for query in PUBLIC_FRAUD_QUERIES
        ]
        try:
            completed = as_completed(futures, timeout=max(4.0, timeout_sec + 2.0))
            for future in completed:
                try:
                    payloads = future.result()
                except Exception:
                    continue
                for payload in payloads:
                    key = _hash_text(f"{payload.get('url')}|{payload.get('title')}")[:18]
                    if key in seen:
                        continue
                    seen.add(key)
                    results.append(payload)
                    if len(results) >= budget:
                        return results
        except FuturesTimeoutError:
            pass
    if results:
        return results
    return fetch_google_news_payloads(PUBLIC_FRAUD_QUERIES, max_records=min(budget, 18), timeout_sec=timeout_sec)


def fetch_bing_news_payloads(queries: list[str], max_records: int, timeout_sec: float) -> list[dict[str, Any]]:
    """Public news RSS intake with real news thumbnail metadata when available."""
    results: list[dict[str, Any]] = []
    seen: set[str] = set()
    for query in queries:
        if len(results) >= max_records:
            break
        url = f"https://www.bing.com/news/search?{urlencode({'q': query, 'format': 'rss'})}"
        try:
            request = Request(
                url,
                headers={"User-Agent": HTTP_USER_AGENT, "Accept": "application/rss+xml,application/xml"},
            )
            with urlopen(request, timeout=max(2.0, min(6.0, timeout_sec / 2))) as response:
                raw = response.read(256_000)
            root = ET.fromstring(raw)
        except Exception:
            continue
        channel = root.find("channel")
        if channel is None:
            continue
        for item in channel.findall("item")[:4]:
            title = str(item.findtext("title") or "").strip()
            raw_link = str(item.findtext("link") or "").strip()
            description = _strip_html(str(item.findtext("description") or "")).strip()
            source_name = first_child_text_by_suffix(item, "Source") or "Bing News"
            image_url = first_child_text_by_suffix(item, "Image")
            published = parse_rss_date(str(item.findtext("pubDate") or "")) or _now()
            if not title or not raw_link:
                continue
            target_url = extract_bing_target_url(raw_link)
            key = _hash_text(f"{target_url}|{title}")[:18]
            if key in seen:
                continue
            seen.add(key)
            relevance_text = f"{title} {description} {query}"
            if not is_relevant_public_signal(relevance_text):
                continue
            direct_domain = urlparse(target_url).netloc or source_name.lower().replace(" ", "-")
            text = (
                f"{title}. {description} Public news RSS source {source_name}. Query family: {query}. "
                "Mapped as live open-news India banking fraud intelligence."
            )
            results.append({
                "source_id": "gdelt_fincrime_news",
                "title": title,
                "text": text,
                "url": target_url,
                "image_url": normalize_public_image_url(image_url),
                "source_domain": direct_domain,
                "publisher_name": source_name,
                "typologies": classify_live_article_typologies(relevance_text),
                "affected_channels": classify_channels(_normalize_text(relevance_text)),
                "region": "India",
                "language": "en",
                "observed_at": published,
                "public_reach_score": 0.84,
                "signal_velocity_score": 0.74,
            })
            if len(results) >= max_records:
                return results
    return results


def fetch_bing_video_payloads(queries: list[str], max_records: int, timeout_sec: float) -> list[dict[str, Any]]:
    """Bounded public video search metadata intake; embeds only trusted public providers."""
    results: list[dict[str, Any]] = []
    seen: set[str] = set()
    for query in queries:
        if len(results) >= max_records:
            break
        url = f"https://www.bing.com/videos/search?{urlencode({'q': query})}"
        try:
            request = Request(url, headers={"User-Agent": HTTP_USER_AGENT, "Accept": "text/html"})
            with urlopen(request, timeout=max(2.0, min(6.0, timeout_sec / 2))) as response:
                raw = response.read(384_000).decode("utf-8", errors="replace")
        except Exception:
            continue
        for item in parse_bing_video_items(raw):
            page_url = str(item.get("page_url") or "")
            label = str(item.get("label") or "")
            if not page_url or not label:
                continue
            key = _hash_text(page_url)[:18]
            if key in seen:
                continue
            seen.add(key)
            if not is_relevant_public_video_signal(label):
                continue
            embed_url, provider, embed_allowed = normalize_video_embed(page_url)
            if not provider:
                provider = urlparse(page_url).netloc.replace("www.", "")
            title = clean_video_title(label)
            text = (
                f"{title}. Public video source discovered from bounded video metadata. Query family: {query}. "
                "Mapped as advisory video evidence for Indian banking fraud intelligence."
            )
            source_domain = urlparse(page_url).netloc or provider
            results.append({
                "source_id": "gdelt_fincrime_news",
                "title": title,
                "text": text,
                "url": page_url,
                "image_url": normalize_public_image_url(str(item.get("thumbnail_url") or "")),
                "source_domain": source_domain,
                "publisher_name": publisher_from_video_label(label, provider),
                "video_page_url": page_url,
                "video_embed_url": embed_url,
                "video_provider": provider,
                "embed_allowed": embed_allowed,
                "typologies": classify_live_article_typologies(label),
                "affected_channels": classify_channels(_normalize_text(label)),
                "region": "India",
                "language": "en",
                "observed_at": _now(),
                "public_reach_score": 0.74,
                "signal_velocity_score": 0.64,
            })
            if len(results) >= max_records:
                return results
    return results


def parse_bing_video_items(html: str) -> list[dict[str, str]]:
    items: list[dict[str, str]] = []
    for match in re.finditer(r'mmeta="(?P<meta>\{.*?\})"', html):
        raw_meta = unescape(match.group("meta"))
        try:
            meta = json.loads(raw_meta)
        except Exception:
            continue
        tail = html[match.end(): match.end() + 1400]
        label_match = re.search(r'aria-label="(?P<label>[^"]+)"', tail)
        label = unescape(label_match.group("label")) if label_match else ""
        page_url = str(meta.get("murl") or meta.get("pgurl") or "").strip()
        thumb = str(meta.get("turl") or "").strip()
        if page_url:
            items.append({
                "page_url": page_url,
                "thumbnail_url": thumb,
                "label": label,
            })
    return items


def clean_video_title(label: str) -> str:
    title = label.split(" from ", 1)[0].strip()
    return re.sub(r"\s+", " ", title)[:180] or "Public fraud intelligence video"


def publisher_from_video_label(label: str, provider: str | None) -> str:
    match = re.search(r"uploaded by ([^·]+)", label, flags=re.IGNORECASE)
    if match:
        return match.group(1).strip()
    match = re.search(r"from ([^·]+)", label, flags=re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return provider or "Public video source"


def first_child_text_by_suffix(node: ET.Element, suffix: str) -> str | None:
    for child in list(node):
        tag = str(child.tag)
        if tag == suffix or tag.endswith(f"}}{suffix}"):
            text = str(child.text or "").strip()
            if text:
                return text
    return None


def extract_bing_target_url(raw_link: str) -> str:
    parsed = urlparse(raw_link)
    if parsed.netloc.lower().endswith("bing.com") and "apiclick.aspx" in parsed.path.lower():
        target = parse_qs(parsed.query).get("url", [None])[0]
        if target:
            return unquote(target)
    return raw_link


def normalize_public_image_url(image_url: str | None) -> str | None:
    if not image_url:
        return None
    candidate = image_url.strip()
    if candidate.startswith("//"):
        candidate = f"https:{candidate}"
    if candidate.startswith("http://www.bing.com/"):
        candidate = "https://" + candidate[len("http://"):]
    return candidate if candidate.startswith(("http://", "https://")) else None


def _strip_html(value: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", value)).strip()


def fetch_google_news_payloads(queries: list[str], max_records: int, timeout_sec: float) -> list[dict[str, Any]]:
    """Fallback public RSS intake when GDELT is unavailable in the local network."""
    results: list[dict[str, Any]] = []
    seen: set[str] = set()
    for query in queries:
        if len(results) >= max_records:
            break
        params = urlencode({"q": query, "hl": "en-IN", "gl": "IN", "ceid": "IN:en"})
        url = f"https://news.google.com/rss/search?{params}"
        try:
            request = Request(url, headers={"User-Agent": HTTP_USER_AGENT, "Accept": "application/rss+xml,application/xml"})
            with urlopen(request, timeout=max(2.0, min(6.0, timeout_sec / 2))) as response:
                raw = response.read(192_000)
            root = ET.fromstring(raw)
        except Exception:
            continue
        channel = root.find("channel")
        if channel is None:
            continue
        for item in channel.findall("item")[:3]:
            title = str(item.findtext("title") or "").strip()
            link = str(item.findtext("link") or "").strip()
            source_name = str(item.findtext("source") or "Google News").strip()
            source_url = None
            source_node = item.find("source")
            if source_node is not None:
                source_url = source_node.attrib.get("url")
            published = parse_rss_date(str(item.findtext("pubDate") or "")) or _now()
            if not title or not link:
                continue
            key = _hash_text(f"{title}|{source_name}")[:18]
            if key in seen:
                continue
            seen.add(key)
            text = (
                f"{title}. Public news RSS source {source_name}. Query family: {query}. "
                "Mapped as live open-news India banking fraud intelligence."
            )
            if not is_relevant_public_signal(title):
                continue
            results.append({
                "source_id": "gdelt_fincrime_news",
                "title": title,
                "text": text,
                "url": link,
                "image_url": None,
                "source_domain": urlparse(source_url or "").netloc or source_name.lower().replace(" ", "-"),
                "publisher_name": source_name,
                "typologies": classify_live_article_typologies(f"{title} {query}"),
                "affected_channels": classify_channels(_normalize_text(f"{title} {text} {query}")),
                "region": "India",
                "language": "en",
                "observed_at": published,
                "public_reach_score": 0.78,
                "signal_velocity_score": 0.68,
            })
            if len(results) >= max_records:
                return results
    return results


def fetch_gdelt_query_payloads(query: str, max_records: int, timeout_sec: float) -> list[dict[str, Any]]:
    params = urlencode({
        "query": query,
        "mode": "ArtList",
        "format": "json",
        "maxrecords": max(1, min(max_records, 10)),
        "sort": "HybridRel",
    })
    url = f"http://api.gdeltproject.org/api/v2/doc/doc?{params}"
    request = Request(url, headers={"User-Agent": HTTP_USER_AGENT, "Accept": "application/json"})
    with urlopen(request, timeout=max(2.0, timeout_sec)) as response:
        raw = response.read(256_000)
    payload = json.loads(raw.decode("utf-8", errors="replace"))
    articles = payload.get("articles", [])
    if not isinstance(articles, list):
        return []

    results: list[dict[str, Any]] = []
    for article in articles[:max_records]:
        if not isinstance(article, dict):
            continue
        title = str(article.get("title") or "").strip()
        article_url = str(article.get("url") or "").strip()
        if not title or not article_url:
            continue
        domain = str(article.get("domain") or urlparse(article_url).netloc).strip()
        country = str(article.get("sourcecountry") or "India").strip() or "India"
        language = str(article.get("language") or "English").strip().lower()
        observed_at = parse_gdelt_seen_date(str(article.get("seendate") or "")) or _now()
        text = (
            f"{title}. Query family: {query}. Public finance/cyber news source {domain} from {country}. "
            "Mapped as open-web payment fraud intelligence for UPI, mule accounts, KYC phishing, "
            "loan-app extortion, QR misuse, SIM swap, dormant-account abuse, and social-engineering watchlists."
        )
        if not is_relevant_public_signal(title):
            continue
        typologies = classify_live_article_typologies(f"{title} {query}")
        social_image = str(article.get("socialimage") or "").strip() or None
        if not social_image:
            social_image = discover_open_graph_image(article_url, timeout_sec=2.2)
        results.append({
            "source_id": "gdelt_fincrime_news",
            "title": title,
            "text": text,
            "url": article_url,
            "image_url": social_image,
            "source_domain": domain or None,
            "typologies": typologies,
            "affected_channels": classify_channels(_normalize_text(f"{title} {text} {query}")),
            "region": "India" if country.lower() == "india" else f"{country} / India relevance",
            "language": "en" if language.startswith("english") else language[:12] or "en",
            "observed_at": observed_at,
            "public_reach_score": 0.82,
            "signal_velocity_score": 0.72,
        })
    return results


class _OpenGraphParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.images: list[str] = []
        self.videos: list[str] = []
        self.players: list[str] = []
        self.json_ld_blocks: list[str] = []
        self._capture_json_ld = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag_name = tag.lower()
        if tag_name == "script":
            data = {key.lower(): value for key, value in attrs if key and value}
            script_type = str(data.get("type") or "").lower()
            self._capture_json_ld = "ld+json" in script_type
            return
        if tag_name != "meta":
            return
        data = {key.lower(): value for key, value in attrs if key and value}
        prop = (data.get("property") or data.get("name") or "").lower()
        if prop in {"og:image", "twitter:image", "twitter:image:src"} and data.get("content"):
            self.images.append(str(data["content"]).strip())
        if prop in {"og:video", "og:video:url", "og:video:secure_url", "twitter:player", "twitter:player:stream"} and data.get("content"):
            target = str(data["content"]).strip()
            if prop == "twitter:player":
                self.players.append(target)
            else:
                self.videos.append(target)

    def handle_data(self, data: str) -> None:
        if self._capture_json_ld and data.strip():
            self.json_ld_blocks.append(data.strip())

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() == "script":
            self._capture_json_ld = False


_PAGE_MEDIA_CACHE: dict[str, dict[str, Any]] = {}


def discover_open_graph_image(url: str, timeout_sec: float = 2.0) -> str | None:
    """Resolve a public page preview image without scraping private/credentialed spaces."""
    return discover_page_media(url, timeout_sec=timeout_sec).get("image_url")


def discover_page_media(url: str, timeout_sec: float = 2.0) -> dict[str, Any]:
    """Resolve public OpenGraph/Twitter/JSON-LD media metadata for a source page."""
    empty = {
        "image_url": None,
        "video_embed_url": None,
        "video_page_url": None,
        "video_provider": None,
        "embed_allowed": False,
    }
    if not url.startswith(("http://", "https://")):
        return empty
    if url in _PAGE_MEDIA_CACHE:
        return _PAGE_MEDIA_CACHE[url]
    try:
        request = Request(
            url,
            headers={
                "User-Agent": HTTP_USER_AGENT,
                "Accept": "text/html,application/xhtml+xml",
            },
        )
        with urlopen(request, timeout=timeout_sec) as response:
            content_type = str(response.headers.get("content-type") or "").lower()
            if "html" not in content_type:
                _PAGE_MEDIA_CACHE[url] = empty
                return empty
            raw = response.read(96_000)
        parser = _OpenGraphParser()
        parser.feed(raw.decode("utf-8", errors="replace"))
        image = next((urljoin(url, item) for item in parser.images if item), None)
        video_candidates = [urljoin(url, item) for item in [*parser.players, *parser.videos] if item]
        json_ld_video = extract_jsonld_video(parser.json_ld_blocks, base_url=url)
        if json_ld_video.get("thumbnail_url") and not image:
            image = json_ld_video["thumbnail_url"]
        if json_ld_video.get("video_url"):
            video_candidates.append(str(json_ld_video["video_url"]))
        video_url = next((item for item in video_candidates if item.startswith(("http://", "https://"))), None)
        embed_url, provider, embed_allowed = normalize_video_embed(video_url)
        result = {
            "image_url": image if image and image.startswith(("http://", "https://")) else None,
            "video_embed_url": embed_url,
            "video_page_url": video_url,
            "video_provider": provider,
            "embed_allowed": embed_allowed,
        }
        _PAGE_MEDIA_CACHE[url] = result
        return result
    except Exception:
        pass
    _PAGE_MEDIA_CACHE[url] = empty
    return empty


def extract_jsonld_video(blocks: list[str], base_url: str) -> dict[str, str | None]:
    def iter_nodes(value: Any):
        if isinstance(value, dict):
            yield value
            graph = value.get("@graph")
            if isinstance(graph, list):
                for item in graph:
                    yield from iter_nodes(item)
        elif isinstance(value, list):
            for item in value:
                yield from iter_nodes(item)

    for block in blocks:
        try:
            parsed = json.loads(block)
        except Exception:
            continue
        for node in iter_nodes(parsed):
            node_type = node.get("@type")
            types = node_type if isinstance(node_type, list) else [node_type]
            if "VideoObject" not in {str(item) for item in types}:
                continue
            video_url = node.get("embedUrl") or node.get("contentUrl") or node.get("url")
            thumbnail = node.get("thumbnailUrl")
            if isinstance(thumbnail, list):
                thumbnail = next((item for item in thumbnail if item), None)
            return {
                "video_url": urljoin(base_url, str(video_url)) if video_url else None,
                "thumbnail_url": urljoin(base_url, str(thumbnail)) if thumbnail else None,
            }
    return {"video_url": None, "thumbnail_url": None}


def normalize_video_embed(video_url: str | None) -> tuple[str | None, str | None, bool]:
    if not video_url:
        return None, None, False
    parsed = urlparse(video_url)
    host = parsed.netloc.lower()
    provider = host.replace("www.", "")
    if "youtube.com" in host:
        if "/embed/" in parsed.path:
            return video_url, "youtube", True
        match = re.search(r"[?&]v=([^&]+)", video_url)
        if match:
            return f"https://www.youtube.com/embed/{match.group(1)}", "youtube", True
    if "youtu.be" in host:
        video_id = parsed.path.strip("/")
        if video_id:
            return f"https://www.youtube.com/embed/{video_id}", "youtube", True
    if "vimeo.com" in host:
        match = re.search(r"/(?:video/)?([0-9]+)", parsed.path)
        if match:
            return f"https://player.vimeo.com/video/{match.group(1)}", "vimeo", True
    if any(name in host for name in ["twitter.com", "x.com"]):
        return video_url, "twitter", True
    if parsed.scheme == "https" and ("player" in host or "embed" in parsed.path):
        return video_url, provider, True
    return None, provider, False


def parse_gdelt_seen_date(value: str) -> float | None:
    try:
        return datetime.strptime(value, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc).timestamp()
    except (TypeError, ValueError):
        return None


def parse_rss_date(value: str) -> float | None:
    try:
        from email.utils import parsedate_to_datetime

        return parsedate_to_datetime(value).timestamp()
    except Exception:
        return None


def classify_live_article_typologies(title: str) -> list[str]:
    text = _normalize_text(title)
    typologies = classify_typologies(text)
    if typologies == ["UNCLASSIFIED"] and "fraud" in text:
        typologies = ["UPI_MULE_NETWORK" if "upi" in text else "PROFILE_MISMATCH"]
    if "upi" in text and "UPI_MULE_NETWORK" not in typologies:
        typologies.insert(0, "UPI_MULE_NETWORK")
    return typologies[:4]


def is_relevant_public_signal(text: str) -> bool:
    normalized = _normalize_text(text)
    strong_terms = [
        "fraud",
        "scam",
        "phishing",
        "mule",
        "kyc",
        "sim swap",
        "sim-swap",
        "loan app",
        "extortion",
        "digital arrest",
        "cyber",
        "cybercrime",
        "money mule",
        "account rent",
        "cashout",
        "cash-out",
        "otp",
        "apk",
        "qr fraud",
        "arrested",
        "complaint",
        "banking cyber fraud",
    ]
    if not any(term in normalized for term in strong_terms):
        return False
    excluded = ["betting sites", "bookmakers picks", "sports betting", "online betting", "betting operation", "odds ranked"]
    return not any(term in normalized for term in excluded)


def is_relevant_public_video_signal(text: str) -> bool:
    normalized = _normalize_text(text)
    excluded = ["betting", "bookmaker", "odds", "kpop", "elevator wins"]
    if any(term in normalized for term in excluded):
        return False
    video_terms = [
        "digital arrest",
        "digitalarrest",
        "upi",
        "mule",
        "money mule",
        "banking fraud",
        "cyber fraud",
        "cyberfraud",
        "phishing",
        "kyc",
        "sim swap",
        "loan app",
        "scam",
    ]
    return any(term in normalized for term in video_terms)


def media_preview_rank(item: dict[str, Any]) -> int:
    status = str(item.get("preview_status") or item.get("image_status") or "")
    origin = str(item.get("media_origin") or "")
    if status == "real_video_embed":
        return 5
    if status == "real_image" or origin in REAL_IMAGE_ORIGINS:
        return 4
    if status == "source_card":
        return 3
    if status == "publisher_logo_only":
        return 2
    return 1


def extract_entities(text: str) -> list[str]:
    patterns = [
        r"\b[A-Z]{4}0[A-Z0-9]{6}\b",
        r"\b[A-Z]{2,6}[_-]?[0-9]{3,}\b",
        r"\b[\w.-]+@[\w.-]+\b",
        r"\b(?:UPI|VPA|IFSC|PAN|AADHAAR)\b",
    ]
    entities: list[str] = []
    upper_text = text.upper()
    for pattern in patterns:
        for match in re.findall(pattern, upper_text):
            value = match if isinstance(match, str) else match[0]
            if value and value not in entities:
                entities.append(value)
    return entities


def detect_language(text: str) -> str:
    if any(term in text for term in ["bhai", "account rent", "paise", "jaldi", "commission"]):
        return "hinglish"
    return "en"


def sovereignty_tags_for(text: str, region: str) -> list[str]:
    tags = ["local_first_processing", "no_customer_pii_export"]
    if "india" in region.lower() or _india_relevance(text, region) >= 0.4:
        tags.extend(["india_relevant", "rbi_fiu_i4c_context"])
    if "diaspora" in region.lower():
        tags.append("indian_diaspora_relevance")
    return sorted(set(tags))


def trend_title_for(typology: str, channels: list[str], evidence_count: int) -> str:
    pretty = typology.replace("_", " ").title()
    channel_text = "/".join(channels[:3])
    return f"{pretty} pressure on {channel_text} ({evidence_count} signals)"


def watchlist_terms_for(typology: str) -> list[str]:
    terms = {
        "UPI_MULE_NETWORK": ["mule account", "account rent", "upi cashout", "vpa mule", "qr cashout"],
        "DIGITAL_ARREST": ["digital arrest", "courier fraud", "police impersonation", "aadhaar case"],
        "KYC_UPDATE_PHISHING": ["kyc update", "apk install", "remote access", "otp capture"],
        "LOAN_APP_EXTORTION": ["loan app", "contact list", "repayment threat", "instant loan"],
        "INVESTMENT_SCAM": ["task scam", "fake trading", "guaranteed return", "crypto deposit"],
        "DORMANT_ACTIVATION": ["dormant reactivation", "inactive account", "new device login"],
        "LAYERING": ["rapid forwarding", "multi-hop transfer", "amount decay", "collector account"],
        "STRUCTURING": ["below threshold", "split transfer", "sub-threshold", "collector convergence"],
        "ROUND_TRIPPING": ["circular transfer", "round tripping", "shell loop"],
        "PROFILE_MISMATCH": ["income mismatch", "profile mismatch", "unexpected business volume"],
    }
    return terms.get(typology, [typology.lower().replace("_", " ")])


def capped_deltas_for(typology: str) -> dict[str, float]:
    deltas = {
        "UPI_MULE_NETWORK": {"mule_network_weight": 0.08, "rapid_forward_weight": 0.06},
        "DIGITAL_ARREST": {"social_engineering_weight": 0.07, "upi_mule_weight": 0.05},
        "KYC_UPDATE_PHISHING": {"new_device_weight": 0.06, "otp_anomaly_weight": 0.07},
        "LOAN_APP_EXTORTION": {"collector_convergence_weight": 0.05, "upi_velocity_weight": 0.05},
        "INVESTMENT_SCAM": {"new_beneficiary_weight": 0.05, "structuring_weight": 0.04},
        "DORMANT_ACTIVATION": {"dormancy_weight": 0.08, "new_device_weight": 0.05},
        "LAYERING": {"hop_count_weight": 0.08, "amount_decay_weight": 0.05},
        "STRUCTURING": {"sub_threshold_weight": 0.08, "time_window_weight": 0.05},
        "ROUND_TRIPPING": {"cycle_detection_weight": 0.08, "shell_account_weight": 0.04},
        "PROFILE_MISMATCH": {"profile_deviation_weight": 0.08, "behavioral_drift_weight": 0.05},
    }
    selected = deltas.get(typology, {"watchlist_context_weight": 0.03})
    return {key: round(min(0.10, max(-0.10, value)), 4) for key, value in selected.items()}


def scenario_seed_for(typology: str) -> str:
    mapping = {
        "UPI_MULE_NETWORK": "rapid_layering",
        "DIGITAL_ARREST": "rapid_layering",
        "KYC_UPDATE_PHISHING": "dormant_activation",
        "LOAN_APP_EXTORTION": "structuring",
        "INVESTMENT_SCAM": "profile_mismatch",
        "DORMANT_ACTIVATION": "dormant_activation",
        "LAYERING": "rapid_layering",
        "STRUCTURING": "structuring",
        "ROUND_TRIPPING": "round_tripping",
        "PROFILE_MISMATCH": "profile_mismatch",
    }
    return mapping.get(typology, "rapid_layering")


def media_preview_for(
    source: SourceConfig,
    title: str,
    text: str,
    typologies: list[str],
    observed_at: float,
    url: str,
    language: str,
    image_url: str | None = None,
    source_domain: str | None = None,
    publisher_name: str | None = None,
    video_embed_url: str | None = None,
    video_page_url: str | None = None,
    video_provider: str | None = None,
    embed_allowed: bool = False,
) -> dict[str, Any]:
    typology = typologies[0] if typologies else "UNCLASSIFIED"
    source_kind = {
        "tier_0": "official-advisory",
        "tier_1": "news-brief",
        "tier_2": "public-social",
        "tier_3": "open-web",
    }.get(source.tier, "open-source")
    media = media_details_for(
        source=source,
        source_url=url,
        image_url=image_url,
        source_domain=source_domain,
        video_embed_url=video_embed_url,
        video_page_url=video_page_url,
        video_provider=video_provider,
        embed_allowed=embed_allowed,
    )
    media_type = "video" if media["preview_status"] in {"real_video_embed", "source_card"} and media.get("video_page_url") else "image"
    duration = 42 if media_type == "video" and "digital arrest" in text else (30 if media_type == "video" else None)
    preview = MediaPreview(
        media_id=_stable_id("MED", source.source_id, title, typology),
        media_type=media_type,
        source_kind=source_kind,
        title=title,
        caption=media_caption_for(typology, source),
        thumbnail_key=thumbnail_key_for(typology, source.source_id),
        source_url=url,
        publisher=str(publisher_name or source.name),
        language=language,
        duration_sec=duration,
        published_at=observed_at,
        image_url=media["thumbnail_url"],
        source_domain=media["source_domain"],
        media_origin=media["media_origin"],
        media_url=media["media_url"],
        thumbnail_url=media["thumbnail_url"],
        image_status=media["image_status"],
        fallback_reason=media["fallback_reason"],
        license_hint=media["license_hint"],
        fetched_at=media["fetched_at"],
        publisher_logo_url=media["publisher_logo_url"],
        preview_status=media["preview_status"],
        is_real_media=media["is_real_media"],
        video_embed_url=media["video_embed_url"],
        video_page_url=media["video_page_url"],
        video_provider=media["video_provider"],
        embed_allowed=media["embed_allowed"],
    )
    return preview.to_dict()


def media_details_for(
    source: SourceConfig,
    source_url: str,
    image_url: str | None,
    source_domain: str | None,
    video_embed_url: str | None = None,
    video_page_url: str | None = None,
    video_provider: str | None = None,
    embed_allowed: bool = False,
) -> dict[str, Any]:
    domain = source_domain or urlparse(source_url if source_url.startswith("http") else source.url).netloc
    logo = publisher_logo_for(domain or source.url)
    now = _now()
    if video_embed_url and embed_allowed:
        return {
            "media_origin": "real_video_embed",
            "media_url": video_embed_url,
            "thumbnail_url": image_url or logo,
            "image_status": "real_video_embed",
            "preview_status": "real_video_embed",
            "is_real_media": True,
            "fallback_reason": None,
            "license_hint": "Public embeddable video metadata from source page; verify publisher terms before reuse",
            "fetched_at": now,
            "publisher_logo_url": logo,
            "source_domain": domain or None,
            "video_embed_url": video_embed_url,
            "video_page_url": video_page_url or source_url,
            "video_provider": video_provider,
            "embed_allowed": True,
        }
    if video_page_url:
        return {
            "media_origin": "source_card",
            "media_url": video_page_url,
            "thumbnail_url": image_url or logo,
            "image_status": "source_card",
            "preview_status": "source_card",
            "is_real_media": False,
            "fallback_reason": "video_page_not_embeddable_public_source_card",
            "license_hint": "Public video source card; embedding not verified",
            "fetched_at": now,
            "publisher_logo_url": logo,
            "source_domain": domain or None,
            "video_embed_url": None,
            "video_page_url": video_page_url,
            "video_provider": video_provider,
            "embed_allowed": False,
        }
    if image_url and image_url.startswith(("http://", "https://")):
        if "bing.com/th?" in image_url:
            origin = "bing_news_image"
        else:
            origin = "gdelt_social_image" if source.source_id == "gdelt_fincrime_news" else "live_image"
        return {
            "media_origin": origin,
            "media_url": image_url,
            "thumbnail_url": image_url,
            "image_status": "real_image",
            "preview_status": "real_image",
            "is_real_media": True,
            "fallback_reason": None,
            "license_hint": "public-source image metadata; verify original publisher terms before external reuse",
            "fetched_at": now,
            "publisher_logo_url": logo,
            "source_domain": domain or None,
            "video_embed_url": None,
            "video_page_url": None,
            "video_provider": None,
            "embed_allowed": False,
        }

    should_discover_og = (
        source.source_id == "gdelt_fincrime_news"
        and source_url.startswith(("http://", "https://"))
        and source_url != source.url
    )
    page_media = discover_page_media(source_url, timeout_sec=1.8) if should_discover_og else {}
    og_image = page_media.get("image_url")
    if og_image:
        return {
            "media_origin": "open_graph_image",
            "media_url": og_image,
            "thumbnail_url": og_image,
            "image_status": "real_image",
            "preview_status": "real_image",
            "is_real_media": True,
            "fallback_reason": None,
            "license_hint": "OpenGraph/Twitter card image from public page; verify publisher terms before reuse",
            "fetched_at": now,
            "publisher_logo_url": logo,
            "source_domain": domain or None,
            "video_embed_url": page_media.get("video_embed_url"),
            "video_page_url": page_media.get("video_page_url"),
            "video_provider": page_media.get("video_provider"),
            "embed_allowed": bool(page_media.get("embed_allowed")),
        }

    if page_media.get("video_embed_url") and page_media.get("embed_allowed"):
        return {
            "media_origin": "real_video_embed",
            "media_url": page_media.get("video_embed_url"),
            "thumbnail_url": logo,
            "image_status": "real_video_embed",
            "preview_status": "real_video_embed",
            "is_real_media": True,
            "fallback_reason": None,
            "license_hint": "Public embeddable video metadata from source page; verify publisher terms before reuse",
            "fetched_at": now,
            "publisher_logo_url": logo,
            "source_domain": domain or None,
            "video_embed_url": page_media.get("video_embed_url"),
            "video_page_url": page_media.get("video_page_url"),
            "video_provider": page_media.get("video_provider"),
            "embed_allowed": True,
        }

    status = "publisher_logo_only" if logo else "generated_fallback"
    return {
        "media_origin": "publisher_logo" if logo else "generated_poster",
        "media_url": logo,
        "thumbnail_url": logo,
        "image_status": status,
        "preview_status": status,
        "is_real_media": False,
        "fallback_reason": "publisher_logo_used_no_article_image" if logo else "no_live_or_opengraph_image_available",
        "license_hint": "Public publisher favicon/logo preview; article image unavailable",
        "fetched_at": now,
        "publisher_logo_url": logo,
        "source_domain": domain or None,
        "video_embed_url": page_media.get("video_embed_url"),
        "video_page_url": page_media.get("video_page_url"),
        "video_provider": page_media.get("video_provider"),
        "embed_allowed": False,
    }


def publisher_logo_for(domain_or_url: str | None) -> str | None:
    if not domain_or_url:
        return None
    parsed = urlparse(domain_or_url if "://" in domain_or_url else f"https://{domain_or_url}")
    domain = parsed.netloc or parsed.path
    if not domain or "." not in domain or domain.endswith("-fixtures"):
        return None
    return f"https://www.google.com/s2/favicons?sz=128&domain_url={quote_plus('https://' + domain)}"


def media_caption_for(typology: str, source: SourceConfig) -> str:
    captions = {
        "DIGITAL_ARREST": "Public warning clip mapped to UPI mule cash-out pressure.",
        "UPI_MULE_NETWORK": "Source preview highlights mule account recruitment and rapid cash-out signals.",
        "KYC_UPDATE_PHISHING": "Advisory preview for KYC update lures, APK install prompts, and OTP capture.",
        "LOAN_APP_EXTORTION": "News/social preview around loan-app pressure and repayment mule routing.",
        "INVESTMENT_SCAM": "Public chatter preview for fake trading, task scams, and deposit chains.",
        "DORMANT_ACTIVATION": "Open-web preview for dormant account reactivation claims.",
    }
    return captions.get(typology, f"{source.category} source preview mapped to Indian banking fraud risk.")


def thumbnail_key_for(typology: str, source_id: str) -> str:
    mapping = {
        "DIGITAL_ARREST": "call-center-evidence",
        "UPI_MULE_NETWORK": "upi-mule-network",
        "KYC_UPDATE_PHISHING": "mobile-kyc-phishing",
        "LOAN_APP_EXTORTION": "loan-app-collections",
        "INVESTMENT_SCAM": "investment-scam-feed",
        "DORMANT_ACTIVATION": "dormant-account-map",
        "LAYERING": "layering-flow",
        "STRUCTURING": "structuring-thresholds",
    }
    return mapping.get(typology, _stable_id("THM", source_id, typology).lower())


def geo_scope_for(region: str, typologies: list[str]) -> list[dict[str, Any]]:
    typology = typologies[0] if typologies else "UNCLASSIFIED"
    if "diaspora" in region.lower():
        return [
            {"label": "Delhi NCR", "lat": 28.6139, "lng": 77.2090, "weight": 0.74},
            {"label": "Dubai-India corridor", "lat": 25.2048, "lng": 55.2708, "weight": 0.62},
        ]
    if typology in {"DIGITAL_ARREST", "KYC_UPDATE_PHISHING"}:
        return [
            {"label": "Delhi NCR", "lat": 28.6139, "lng": 77.2090, "weight": 0.82},
            {"label": "Mumbai", "lat": 19.0760, "lng": 72.8777, "weight": 0.78},
            {"label": "Bengaluru", "lat": 12.9716, "lng": 77.5946, "weight": 0.64},
        ]
    if typology in {"UPI_MULE_NETWORK", "LAYERING", "STRUCTURING"}:
        return [
            {"label": "Mumbai", "lat": 19.0760, "lng": 72.8777, "weight": 0.86},
            {"label": "Hyderabad", "lat": 17.3850, "lng": 78.4867, "weight": 0.66},
            {"label": "Kolkata", "lat": 22.5726, "lng": 88.3639, "weight": 0.61},
        ]
    return [
        {"label": "Pune", "lat": 18.5204, "lng": 73.8567, "weight": 0.58},
        {"label": "Chennai", "lat": 13.0827, "lng": 80.2707, "weight": 0.55},
    ]


def public_reach_for(source: SourceConfig, text: str) -> float:
    score = {"tier_0": 0.68, "tier_1": 0.78, "tier_2": 0.58, "tier_3": 0.32}.get(source.tier, 0.35)
    if any(term in text for term in ["upi", "kyc", "digital arrest", "loan app"]):
        score += 0.10
    return min(0.98, score)


def signal_velocity_for(text: str, observed_at: float) -> float:
    age_hours = max(0.0, (_now() - observed_at) / 3600)
    freshness = max(0.20, 1.0 - age_hours / 24)
    keyword_boost = 0.14 if any(term in text for term in ["burst", "rising", "rapid", "cashout", "public"]) else 0.04
    return min(0.99, 0.42 + freshness * 0.34 + keyword_boost)


def source_mix_for(signals: list[ExternalThreatSignal], sources: dict[str, SourceConfig]) -> list[dict[str, Any]]:
    tier_labels = {
        "tier_0": "Official",
        "tier_1": "News",
        "tier_2": "Social",
        "tier_3": "Open Web",
    }
    rows = []
    for tier, label in tier_labels.items():
        tier_sources = [source for source in sources.values() if source.tier == tier]
        tier_signals = [signal for signal in signals if signal.source_tier == tier]
        rows.append({
            "tier": tier,
            "label": label,
            "sources": len(tier_sources),
            "signals": len(tier_signals),
            "trust": round(max((signal.trust_score for signal in tier_signals), default=SOURCE_TIER_TRUST[tier]), 4),
        })
    return rows


def source_health_for(sources: dict[str, SourceConfig]) -> list[dict[str, Any]]:
    now = _now()
    rows = []
    for source in sources.values():
        age = None if source.last_polled_at is None else max(0, int(now - source.last_polled_at))
        rows.append({
            "source_id": source.source_id,
            "tier": source.tier,
            "status": source.last_status,
            "age_sec": age,
            "enabled": source.enabled,
        })
    return rows


def corroboration_rate_for(signals: list[ExternalThreatSignal]) -> float:
    if not signals:
        return 0.0
    corroborated = 0
    for signal in signals:
        peers = [
            other for other in signals
            if other.signal_id != signal.signal_id
            and set(other.typologies).intersection(signal.typologies)
            and other.source_tier != signal.source_tier
        ]
        if signal.corroboration_ids or peers or signal.source_tier == "tier_0":
            corroborated += 1
    return round(corroborated / len(signals), 4)


def velocity_index_for(signals: list[ExternalThreatSignal], timeline: list[dict[str, Any]]) -> float:
    if not signals:
        return 0.0
    recent = sum(point["total"] for point in timeline[-3:])
    baseline = max(1, sum(point["total"] for point in timeline[:6]) / 6)
    signal_velocity = sum(signal.signal_velocity_score for signal in signals) / len(signals)
    return round(min(1.0, 0.45 * signal_velocity + 0.55 * min(1.0, recent / (baseline * 4))), 4)


def channel_exposure_for(signals: list[ExternalThreatSignal], now: float) -> list[dict[str, Any]]:
    channels = ["UPI", "IMPS", "DIGITAL_BANKING", "NEFT", "RTGS", "CARDS", "BRANCH"]
    phase = int(now // 7)
    rows = []
    for idx, channel in enumerate(channels):
        related = [signal for signal in signals if channel in signal.affected_channels]
        base = 0.18 + 0.12 * len(related)
        pulse = ((phase + idx * 3) % 11) / 100
        trust = max((signal.trust_score for signal in related), default=0.42)
        rows.append({
            "channel": channel.replace("DIGITAL_BANKING", "DIGITAL"),
            "exposure": round(min(0.98, base + trust * 0.38 + pulse), 4),
            "signals": len(related),
            "trust": round(trust, 4),
            "delta": round(((phase + idx) % 7 - 3) / 100, 4),
            "velocity": round(min(0.98, max((signal.signal_velocity_score for signal in related), default=0.28) + pulse), 4),
        })
    return sorted(rows, key=lambda row: row["exposure"], reverse=True)


def typology_matrix_for(signals: list[ExternalThreatSignal]) -> list[dict[str, Any]]:
    typologies = sorted({typology for signal in signals for typology in signal.typologies})
    rows = []
    for typology in typologies:
        related = [signal for signal in signals if typology in signal.typologies]
        rows.append({
            "typology": typology,
            "label": typology.replace("_", " ").title(),
            "official": sum(1 for signal in related if signal.source_tier == "tier_0"),
            "news": sum(1 for signal in related if signal.source_tier == "tier_1"),
            "social": sum(1 for signal in related if signal.source_tier == "tier_2"),
            "open_web": sum(1 for signal in related if signal.source_tier == "tier_3"),
            "trust": round(max((signal.trust_score for signal in related), default=0.0), 4),
        })
    return sorted(rows, key=lambda row: (row["trust"], row["official"] + row["news"] + row["social"]), reverse=True)


def signal_timeline_for(signals: list[ExternalThreatSignal], now: float) -> list[dict[str, Any]]:
    rows = []
    signal_factor = max(1, len(signals))
    for idx in range(36):
        age = 35 - idx
        ts = now - age * 600
        phase = int(ts // 300)
        official = max(0, int(2 + signal_factor * 0.55 + ((phase + idx) % 4)))
        news = max(0, int(1 + signal_factor * 0.45 + ((phase + idx * 2) % 5)))
        social = max(0, int(3 + signal_factor * 0.75 + ((phase + idx * 3) % 7)))
        rows.append({
            "time": time.strftime("%H:%M", time.localtime(ts)),
            "official": official,
            "news": news,
            "social": social,
            "total": official + news + social,
            "trust": round(max((s.trust_score for s in signals), default=0.0), 4),
        })
    return rows


def geo_hotspots_for(
    signals: list[ExternalThreatSignal],
    trends: list[FraudTrendCluster],
    now: float,
) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    phase = int(now // 11)
    for signal in signals:
        for item in signal.geo_scope:
            label = str(item["label"])
            row = merged.setdefault(label, {
                "label": label,
                "lat": item["lat"],
                "lng": item["lng"],
                "risk": 0.0,
                "signals": 0,
                "trust": 0.0,
                "typologies": set(),
                "channels": set(),
            })
            row["signals"] += 1
            row["risk"] = max(row["risk"], float(item.get("weight", 0.5)) * signal.trust_score)
            row["trust"] = max(row["trust"], signal.trust_score)
            row["typologies"].update(signal.typologies[:3])
            row["channels"].update(signal.affected_channels[:3])
    if not merged and trends:
        merged["Mumbai"] = {
            "label": "Mumbai",
            "lat": 19.0760,
            "lng": 72.8777,
            "risk": trends[0].trust_score * 0.80,
            "signals": trends[0].evidence_count,
            "trust": trends[0].trust_score,
            "typologies": set(trends[0].typologies),
            "channels": set(trends[0].affected_channels),
        }

    active_typologies = sorted({typology for signal in signals for typology in signal.typologies})
    active_channels = sorted({channel for signal in signals for channel in signal.affected_channels})
    max_trust = max((signal.trust_score for signal in signals), default=0.64)
    for idx, base in enumerate(INDIA_GEO_BASELINE):
        label = base["label"]
        if label in merged:
            continue
        channel_overlap = set(base["channels"]).intersection(active_channels)
        baseline_risk = 0.28 + 0.06 * len(channel_overlap) + ((phase + idx) % 6) / 100
        merged[label] = {
            "label": label,
            "lat": base["lat"],
            "lng": base["lng"],
            "risk": min(0.74, baseline_risk * max_trust),
            "signals": max(1, len(channel_overlap)),
            "trust": round(max(0.48, max_trust - 0.16), 4),
            "typologies": set(active_typologies[:3] or ["UPI_MULE_NETWORK"]),
            "channels": set(base["channels"]),
        }

    rows = []
    for idx, row in enumerate(merged.values()):
        pulse = ((phase + idx * 2) % 9) / 100
        velocity = min(0.98, 0.26 + row["signals"] * 0.06 + pulse)
        risk = round(min(0.99, row["risk"] + pulse), 4)
        rows.append({
            "label": row["label"],
            "lat": row["lat"],
            "lng": row["lng"],
            "risk": risk,
            "signals": row["signals"],
            "trust": round(row["trust"], 4),
            "velocity": round(velocity, 4),
            "delta": round(pulse - 0.04, 4),
            "rank": idx + 1,
            "primary_typology": sorted(row["typologies"])[0] if row["typologies"] else "UNCLASSIFIED",
            "primary_channel": sorted(row["channels"])[0] if row["channels"] else "UPI",
            "typologies": sorted(row["typologies"])[:4],
            "channels": sorted(row["channels"])[:4],
        })
    return sorted(rows, key=lambda item: (item["risk"], item["signals"]), reverse=True)[:24]


def geo_links_for(hotspots: list[dict[str, Any]]) -> list[dict[str, Any]]:
    top = hotspots[:7]
    links = []
    for idx in range(max(0, len(top) - 1)):
        source = top[idx]
        target = top[idx + 1]
        links.append({
            "source": source["label"],
            "target": target["label"],
            "weight": round((source["risk"] + target["risk"]) / 2, 4),
            "channel": source.get("primary_channel") or "UPI",
        })
    if len(top) >= 4:
        links.append({
            "source": top[0]["label"],
            "target": top[3]["label"],
            "weight": round((top[0]["risk"] + top[3]["risk"]) / 2, 4),
            "channel": "IMPS",
        })
    return links


def fusion_graph_for(
    signals: list[ExternalThreatSignal],
    trends: list[FraudTrendCluster],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    nodes: dict[str, dict[str, Any]] = {}
    links: list[dict[str, Any]] = []
    for signal in signals[:10]:
        source_id = f"src:{signal.source_id}"
        nodes[source_id] = {
            "id": source_id,
            "label": signal.source_name,
            "kind": "source",
            "tier": signal.source_tier,
            "trust": signal.trust_score,
        }
        signal_id = f"sig:{signal.signal_id}"
        nodes[signal_id] = {
            "id": signal_id,
            "label": signal.title,
            "kind": "signal",
            "tier": signal.source_tier,
            "trust": signal.trust_score,
        }
        links.append({"source": source_id, "target": signal_id, "weight": signal.trust_score})
    for trend in trends[:6]:
        trend_id = f"trend:{trend.trend_id}"
        nodes[trend_id] = {
            "id": trend_id,
            "label": trend.title,
            "kind": "trend",
            "tier": "cluster",
            "trust": trend.trust_score,
        }
        for evidence_id in trend.evidence_ids[:4]:
            signal_id = f"sig:{evidence_id}"
            if signal_id in nodes:
                links.append({"source": signal_id, "target": trend_id, "weight": trend.trust_score})
    return list(nodes.values()), links


def social_pulse_for(signals: list[ExternalThreatSignal], now: float) -> list[dict[str, Any]]:
    buckets = [
        ("News wires", "tier_1", "finance/cyber desks"),
        ("Public social", "tier_2", "compliant public chatter"),
        ("Official advisories", "tier_0", "regulator/public authority"),
        ("Open web", "tier_3", "advisory-only raw web"),
    ]
    phase = int(now // 5)
    rows = []
    for idx, (label, tier, note) in enumerate(buckets):
        related = [signal for signal in signals if signal.source_tier == tier]
        rows.append({
            "label": label,
            "note": note,
            "mentions": int(len(related) * 14 + 6 + ((phase + idx * 5) % 17)),
            "velocity": round(min(0.98, 0.24 + len(related) * 0.16 + ((phase + idx) % 6) / 100), 4),
            "trust": round(max((signal.trust_score for signal in related), default=SOURCE_TIER_TRUST[tier]), 4),
        })
    return sorted(rows, key=lambda item: item["mentions"], reverse=True)


def source_velocity_series_for(signals: list[ExternalThreatSignal], now: float) -> list[dict[str, Any]]:
    rows = []
    tier_keys = {"tier_0": "official", "tier_1": "news", "tier_2": "social", "tier_3": "open_web"}
    signal_factor = max(1, len(signals))
    for idx in range(48):
        ts = now - (47 - idx) * 300
        phase = int(ts // 180)
        row: dict[str, Any] = {"time": time.strftime("%H:%M", time.localtime(ts))}
        total = 0
        for tier, key in tier_keys.items():
            related = [signal for signal in signals if signal.source_tier == tier]
            value = max(0, int(len(related) * 4 + signal_factor * 0.25 + ((phase + idx + len(key)) % 8)))
            row[key] = value
            total += value
        row["total"] = total
        rows.append(row)
    return rows


def typology_velocity_series_for(signals: list[ExternalThreatSignal], now: float) -> list[dict[str, Any]]:
    top = sorted(
        {typology for signal in signals for typology in signal.typologies},
        key=lambda typology: sum(1 for signal in signals if typology in signal.typologies),
        reverse=True,
    )[:6]
    rows = []
    for typology in top:
        related = [signal for signal in signals if typology in signal.typologies]
        phase = int(now // 240) + len(typology)
        rows.append({
            "typology": typology,
            "label": typology.replace("_", " ").title(),
            "signals": len(related),
            "velocity": round(min(0.99, max((signal.signal_velocity_score for signal in related), default=0.18) + (phase % 7) / 100), 4),
            "trust": round(max((signal.trust_score for signal in related), default=0.0), 4),
            "mentions": int(len(related) * 18 + 8 + (phase % 21)),
            "delta": round(((phase % 9) - 4) / 100, 4),
        })
    return rows


def channel_typology_heatmap_for(signals: list[ExternalThreatSignal]) -> list[dict[str, Any]]:
    typologies = sorted({typology for signal in signals for typology in signal.typologies})
    channels = ["UPI", "IMPS", "DIGITAL_BANKING", "NEFT", "RTGS", "CARDS", "BRANCH"]
    rows = []
    for typology in typologies[:10]:
        related = [signal for signal in signals if typology in signal.typologies]
        row: dict[str, Any] = {
            "typology": typology,
            "label": typology.replace("_", " ").title(),
            "trust": round(max((signal.trust_score for signal in related), default=0.0), 4),
        }
        for channel in channels:
            channel_signals = [signal for signal in related if channel in signal.affected_channels]
            row[channel] = len(channel_signals)
        rows.append(row)
    return sorted(rows, key=lambda row: (row["trust"], sum(row[channel] for channel in channels)), reverse=True)


def geo_layers_for(hotspots: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "hotspots": hotspots,
        "channels": sorted({channel for hotspot in hotspots for channel in hotspot.get("channels", [])}),
        "typologies": sorted({typology for hotspot in hotspots for typology in hotspot.get("typologies", [])}),
        "max_risk": round(max((item.get("risk", 0.0) for item in hotspots), default=0.0), 4),
        "coverage_count": len(hotspots),
    }


def media_evidence_matrix_for(media_previews: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for preview in media_previews:
        status = str(preview.get("preview_status") or preview.get("media_origin") or "generated_fallback")
        row = grouped.setdefault(status, {
            "origin": status,
            "label": status.replace("_", " ").title(),
            "items": 0,
            "resolved": 0,
            "generated": 0,
            "official": 0,
            "news": 0,
            "social": 0,
            "open_web": 0,
            "trust": 0.0,
        })
        row["items"] += 1
        if preview.get("is_real_media") or status in SOURCE_CARD_STATUSES:
            row["resolved"] += 1
        if status == "generated_fallback":
            row["generated"] += 1
        tier = str(preview.get("source_tier") or "")
        if tier == "tier_0":
            row["official"] += 1
        elif tier == "tier_1":
            row["news"] += 1
        elif tier == "tier_2":
            row["social"] += 1
        elif tier == "tier_3":
            row["open_web"] += 1
        row["trust"] = max(row["trust"], float(preview.get("trust_score") or 0.0))
    return sorted(grouped.values(), key=lambda row: (row["resolved"], row["trust"], row["items"]), reverse=True)


def media_health_for(media_previews: list[dict[str, Any]]) -> float:
    if not media_previews:
        return 0.0
    real = sum(1 for item in media_previews if item.get("is_real_media"))
    source_cards = sum(1 for item in media_previews if item.get("preview_status") in SOURCE_CARD_STATUSES)
    generated = sum(1 for item in media_previews if item.get("preview_status") == "generated_fallback")
    score = (real * 1.0 + source_cards * 0.42 + generated * 0.08) / len(media_previews)
    return round(min(1.0, score), 4)


def media_summary_for(media_previews: list[dict[str, Any]], sources: dict[str, SourceConfig], now: float) -> dict[str, Any]:
    real_images = sum(1 for item in media_previews if item.get("preview_status") == "real_image")
    real_videos = sum(1 for item in media_previews if item.get("preview_status") == "real_video_embed")
    source_cards = sum(1 for item in media_previews if item.get("preview_status") == "source_card")
    publisher_logo_only = sum(1 for item in media_previews if item.get("preview_status") == "publisher_logo_only")
    generated_fallbacks = sum(1 for item in media_previews if item.get("preview_status") == "generated_fallback")
    broken = sum(1 for item in media_previews if item.get("preview_status") == "broken" or item.get("image_status") == "broken")
    last_successful_poll = max(
        (float(source.last_polled_at) for source in sources.values() if source.last_status == "ok" and source.last_polled_at),
        default=None,
    )
    stale_sources = sum(
        1
        for source in sources.values()
        if source.enabled
        and source.last_polled_at is not None
        and now - source.last_polled_at > max(source.poll_interval_sec * 2, 900)
    )
    return {
        "live_media": real_images + real_videos,
        "real_images": real_images,
        "real_videos": real_videos,
        "source_cards": source_cards,
        "publisher_logo_only": publisher_logo_only,
        "generated_fallbacks": generated_fallbacks,
        "broken": broken,
        "stale_sources": stale_sources,
        "last_successful_poll": last_successful_poll,
        "health": media_health_for(media_previews),
    }


def playbook_impact_series_for(
    playbooks: list[AdaptivePlaybook],
    trends: list[FraudTrendCluster],
    now: float,
) -> list[dict[str, Any]]:
    trend_by_id = {trend.trend_id: trend for trend in trends}
    rows = []
    for idx, playbook in enumerate(playbooks[:8]):
        trend = trend_by_id.get(playbook.trend_id)
        ttl_remaining = max(0.0, playbook.expires_at - now)
        rows.append({
            "playbook_id": playbook.playbook_id,
            "title": playbook.title,
            "status": playbook.promotion_status,
            "watchlist_terms": len(playbook.watchlist_terms),
            "risk_delta": round(sum(abs(value) for value in playbook.risk_weight_deltas.values()), 4),
            "evidence_count": trend.evidence_count if trend else 0,
            "trust": trend.trust_score if trend else 0.0,
            "ttl_hours": round(ttl_remaining / 3600, 1),
            "rank": idx + 1,
        })
    return rows


def source_freshness_sla_for(sources: dict[str, SourceConfig], now: float) -> list[dict[str, Any]]:
    rows = []
    for source in sources.values():
        age = None if source.last_polled_at is None else max(0, int(now - source.last_polled_at))
        interval = max(60, source.poll_interval_sec)
        rows.append({
            "source_id": source.source_id,
            "name": source.name,
            "tier": source.tier,
            "category": source.category,
            "status": source.last_status,
            "age_sec": age,
            "poll_interval_sec": source.poll_interval_sec,
            "freshness": 0.0 if age is None else round(max(0.0, min(1.0, 1.0 - (age / (interval * 2)))), 4),
            "enabled": source.enabled,
        })
    return sorted(rows, key=lambda row: (row["enabled"], row["freshness"]), reverse=True)


def corroboration_network_for(signals: list[ExternalThreatSignal]) -> dict[str, Any]:
    nodes = [
        {
            "id": signal.signal_id,
            "label": signal.title,
            "tier": signal.source_tier,
            "trust": signal.trust_score,
            "typologies": signal.typologies[:3],
        }
        for signal in signals[:18]
    ]
    node_ids = {node["id"] for node in nodes}
    links = []
    for signal in signals[:18]:
        if signal.signal_id not in node_ids:
            continue
        for related_id in signal.corroboration_ids[:4]:
            if related_id in node_ids and related_id != signal.signal_id:
                links.append({
                    "source": signal.signal_id,
                    "target": related_id,
                    "weight": round(signal.trust_score, 4),
                })
    return {"nodes": nodes, "links": links}


def _india_relevance(text: str, region: str) -> float:
    hits = sum(1 for term in INDIA_RELEVANCE_TERMS if term in text)
    if "india" in region.lower():
        hits += 2
    if "diaspora" in region.lower():
        hits += 1
    return round(min(1.0, hits / 5), 4)


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _now() -> float:
    return round(time.time(), 3)


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _hash_json(payload: dict[str, Any]) -> str:
    data = json.dumps(payload, sort_keys=True, default=str, separators=(",", ":"))
    return _hash_text(data)


def _stable_id(prefix: str, *parts: Any) -> str:
    digest = _hash_text("|".join(str(part) for part in parts))[:12].upper()
    return f"{prefix}-{digest}"


_SERVICE: PreFraudIntelService | None = None


def get_pre_fraud_intel_service() -> PreFraudIntelService:
    global _SERVICE
    if _SERVICE is None:
        _SERVICE = PreFraudIntelService()
    return _SERVICE


def reset_pre_fraud_intel_service() -> PreFraudIntelService:
    service = get_pre_fraud_intel_service()
    service.reset()
    return service
