"""Pre-fraud intelligence layer tests."""

from __future__ import annotations

import asyncio
import os
import time
from types import SimpleNamespace

os.environ.setdefault("PAYFLOW_CPU_ONLY", "1")


def test_pre_fraud_refresh_promotes_guarded_playbooks():
    from src.intel import reset_pre_fraud_intel_service

    service = reset_pre_fraud_intel_service()
    result = service.refresh(seed=2026)

    assert result["signals_added"] >= 5
    assert result["trends"]
    assert result["playbooks"]
    assert result["tuning_status"]["active_playbooks"] >= 1

    signals = service.list_signals(min_trust=0.85)["signals"]
    assert signals
    assert any(signal["source_tier"] == "tier_0" for signal in signals)
    assert all(signal["audit_hash"] for signal in signals)
    assert all(signal["media_preview"]["media_id"] for signal in signals)

    cockpit = service.cockpit()
    assert cockpit["metrics"]["signal_count"] >= 5
    assert cockpit["signal_timeline"]
    assert cockpit["channel_exposure"]
    assert cockpit["geo_hotspots"]
    assert cockpit["media_previews"]
    media = service.media()
    assert media["summary"]["publisher_logo_only"] >= 1
    assert media["summary"]["live_media"] == media["summary"]["real_images"] + media["summary"]["real_videos"]
    assert media["summary"]["live_media"] < media["count"]


def test_low_trust_raw_signal_stays_advisory_without_corroboration():
    from src.intel import reset_pre_fraud_intel_service

    service = reset_pre_fraud_intel_service()
    service.ingest_signal({
        "source_id": "deep_open_web_curated",
        "title": "Unverified rumour about mule accounts",
        "text": "Rumour claims mule account and UPI cashout but source is unverified and possibly false alarm.",
        "typologies": ["UPI_MULE_NETWORK"],
        "affected_channels": ["UPI"],
        "region": "India",
        "observed_at": time.time(),
    })
    service._rebuild_intelligence()

    playbooks = service.list_playbooks()["playbooks"]
    assert playbooks
    assert playbooks[0]["promotion_status"] == "advisory"
    assert service.tuning_status()["active_playbooks"] == 0


def test_invalid_llm_json_is_quarantined():
    from src.intel import reset_pre_fraud_intel_service

    service = reset_pre_fraud_intel_service()
    result = service.validate_llm_extraction("this is not json")

    assert result["status"] == "quarantined"
    assert result["reason"] == "invalid_json"


def test_media_discovery_parses_opengraph_and_jsonld_video(monkeypatch):
    from src.intel import pre_fraud

    class FakeResponse:
        headers = {"content-type": "text/html; charset=utf-8"}

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self, _size):
            return b"""
            <html><head>
              <meta property="og:image" content="/preview.jpg">
              <meta name="twitter:player" content="https://www.youtube.com/watch?v=abc123xyz">
              <script type="application/ld+json">
                {"@context":"https://schema.org","@type":"VideoObject","embedUrl":"https://www.youtube.com/embed/ldjson123","thumbnailUrl":"/video.jpg"}
              </script>
            </head></html>
            """

    pre_fraud._PAGE_MEDIA_CACHE.clear()
    monkeypatch.setattr(pre_fraud, "urlopen", lambda *_args, **_kwargs: FakeResponse())

    media = pre_fraud.discover_page_media("https://example.org/story", timeout_sec=0.1)

    assert media["image_url"] == "https://example.org/preview.jpg"
    assert media["video_embed_url"] == "https://www.youtube.com/embed/abc123xyz"
    assert media["video_provider"] == "youtube"
    assert media["embed_allowed"] is True


def test_bing_news_payloads_use_real_thumbnail_and_reject_irrelevant(monkeypatch):
    from src.intel import pre_fraud

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self, _size):
            return b"""
            <rss version="2.0" xmlns:News="https://www.bing.com/news/search">
              <channel>
                <item>
                  <title>1 complaint, a mule account and 14 states: cyber scam trail</title>
                  <link>http://www.bing.com/news/apiclick.aspx?url=https%3A%2F%2Findianexpress.com%2Farticle%2Fcities%2Fhyderabad%2Fmule-account-cyber-fraud%2F</link>
                  <description>A public cyber fraud complaint led police to mule bank accounts.</description>
                  <pubDate>Tue, 19 May 2026 21:55:00 GMT</pubDate>
                  <News:Source>The Indian Express</News:Source>
                  <News:Image>http://www.bing.com/th?id=ONUT.example&amp;pid=News</News:Image>
                </item>
                <item>
                  <title>Best betting sites in India ranked</title>
                  <link>https://example.com/betting</link>
                  <description>Bookmaker odds and picks.</description>
                  <News:Source>Example</News:Source>
                </item>
              </channel>
            </rss>
            """

    monkeypatch.setattr(pre_fraud, "urlopen", lambda *_args, **_kwargs: FakeResponse())

    payloads = pre_fraud.fetch_bing_news_payloads(["UPI mule account India fraud"], max_records=4, timeout_sec=1)

    assert len(payloads) == 1
    assert payloads[0]["publisher_name"] == "The Indian Express"
    assert payloads[0]["url"].startswith("https://indianexpress.com/")
    assert payloads[0]["image_url"].startswith("https://www.bing.com/th?")


def test_bing_video_payloads_create_youtube_embed(monkeypatch):
    from src.intel import pre_fraud

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self, _size):
            return b'''
            <html><body>
              <div mmeta="{&quot;murl&quot;:&quot;https://www.youtube.com/watch?v=xDo3a0-hdH4&quot;,&quot;turl&quot;:&quot;https://tse3.mm.bing.net/th/id/OVF.example&quot;}">
                <a aria-label="Police Uniform, Video Call Aur Dhamki | Yeh Scam Hai! | #DigitalArrest #CyberDost from YouTube \xc2\xb7 Duration: 17 seconds \xc2\xb7 uploaded by CyberDost"></a>
              </div>
            </body></html>
            '''

    monkeypatch.setattr(pre_fraud, "urlopen", lambda *_args, **_kwargs: FakeResponse())

    payloads = pre_fraud.fetch_bing_video_payloads(["CyberDost digital arrest scam"], max_records=2, timeout_sec=1)

    assert len(payloads) == 1
    assert payloads[0]["video_embed_url"] == "https://www.youtube.com/embed/xDo3a0-hdH4"
    assert payloads[0]["embed_allowed"] is True
    assert payloads[0]["video_provider"] == "youtube"


def test_public_relevance_rejects_unrelated_betting_result():
    from src.intel.pre_fraud import is_relevant_public_signal

    assert not is_relevant_public_signal("Best Betting Sites in India for May 2026 - Top Bookmakers Picks Ranked")
    assert is_relevant_public_signal("How SIM swap fraud can quietly drain your bank account in minutes")


def test_intel_api_routes_and_filters():
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from src.api.routes.intel import router as intel_router
    from src.intel import reset_pre_fraud_intel_service

    reset_pre_fraud_intel_service()
    app = FastAPI()
    app.include_router(intel_router)
    client = TestClient(app)

    sources = client.get("/api/v1/intel/sources")
    assert sources.status_code == 200
    assert len(sources.json()["sources"]) >= 6

    simulated = client.post("/api/v1/intel/simulate-signal", json={"scenario": "digital_arrest_mule"})
    assert simulated.status_code == 200
    assert simulated.json()["tuning_status"]["active_playbooks"] >= 1

    signals = client.get("/api/v1/intel/signals?typology=UPI_MULE_NETWORK&min_trust=0.85")
    assert signals.status_code == 200
    assert signals.json()["signals"]

    trends = client.get("/api/v1/intel/trends")
    assert trends.status_code == 200
    assert trends.json()["trends"]

    playbooks = client.get("/api/v1/intel/playbooks")
    assert playbooks.status_code == 200
    assert playbooks.json()["playbooks"]

    cockpit = client.get("/api/v1/intel/cockpit")
    assert cockpit.status_code == 200
    body = cockpit.json()
    assert body["media_previews"]
    assert body["geo_hotspots"]
    assert body["signal_timeline"]


def test_evidence_package_includes_pre_fraud_context():
    from src.api.ps3_case import build_case_trace, build_evidence_package
    from src.intel import reset_pre_fraud_intel_service

    service = reset_pre_fraud_intel_service()
    service.simulate_signal("digital_arrest_mule")

    meta = {
        "primary_case_id": "PS3-INTEL",
        "scenario_id": "intel-demo",
        "scenario": "rapid_layering",
        "scenario_label": "Rapid Layering Through Multiple Accounts",
        "focus_account_id": "ACC001",
        "focus_txn_id": "TXN001",
        "expected_indicators": ["Compressed multi-hop path"],
        "typologies": ["LAYERING"],
        "recommended_actions": ["Attach preventive intelligence context"],
        "transaction_chain": [
            {
                "type": "transaction",
                "txn_id": "TXN001",
                "sender": "ACC001",
                "receiver": "ACC002",
                "amount_paisa": 12500000,
                "channel": "UPI",
                "fraud_label": "LAYERING",
            }
        ],
        "generated_at": time.time(),
    }
    engine = SimpleNamespace(get_ps3_case=lambda case_id: meta if case_id == "PS3-INTEL" else None)
    orch = SimpleNamespace(_threat_engine=engine)
    trace = build_case_trace(orch, "PS3-INTEL")
    package = asyncio.run(build_evidence_package(orch, "PS3-INTEL"))

    assert trace["pre_fraud_intelligence"]["active_playbooks"]
    intel = package["pre_fraud_intelligence"]
    assert intel["active_playbooks"]
    assert intel["top_trends"]
    assert "Pre-Fraud Intelligence Context" in package["printable_html"]
    assert package["json_payload"]["pre_fraud_intelligence"]["active_playbooks"]
