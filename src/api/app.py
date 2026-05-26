"""
PayFlow -- FastAPI Application Factory
=======================================
Creates and configures the FastAPI application instance for the
real-time monitoring dashboard.  The ``orchestrator`` reference is
injected via ``app.state`` to avoid circular imports — route handlers
access it through ``request.app.state.orchestrator``.

Usage::

    from src.api.app import create_app

    app = create_app(orchestrator=my_orchestrator)
    # Run with uvicorn or embed in an asyncio task
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from src.api.routes.analyst import router as analyst_router
from src.api.routes.analytics import router as analytics_router
from src.api.routes.countermeasures import router as countermeasures_router
from src.api.routes.dashboard import router as dashboard_router
from src.api.routes.fraud import router as fraud_router
from src.api.routes.intel import router as pre_fraud_intel_router
from src.api.routes.intelligence import router as intelligence_router
from src.api.routes.simulation import router as simulation_router

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
TEMPLATES_DIR = PROJECT_ROOT / "frontend" / "templates"
FRONTEND_DIST = PROJECT_ROOT / "frontend" / "app" / "dist"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown lifecycle for the dashboard server."""
    logger.info("Dashboard server starting — templates: %s", TEMPLATES_DIR)
    yield
    # Cleanup: clear broadcaster subscriber queues
    try:
        from src.api.events import EventBroadcaster
        broadcaster = EventBroadcaster.get()
        logger.info(
            "Dashboard shutdown — broadcaster channels: %s",
            broadcaster.snapshot(),
        )
    except Exception:
        pass


def create_app(orchestrator=None) -> FastAPI:
    """
    Build and return a configured FastAPI application.

    Parameters
    ----------
    orchestrator : PayFlowOrchestrator | None
        Optional reference to the running orchestrator for live
        snapshot and graph access.  Stored on ``app.state``.
    """
    app = FastAPI(
        title="PayFlow Dashboard",
        version="0.1.0",
        description="Real-time fraud intelligence monitoring dashboard",
        lifespan=lifespan,
    )

    # Store orchestrator reference for route handlers
    app.state.orchestrator = orchestrator
    try:
        from src.intel import get_pre_fraud_intel_service

        get_pre_fraud_intel_service().refresh(seed=2026)
        logger.info("Pre-fraud intelligence baseline seeded for judge demo")
    except Exception as exc:
        logger.debug("Pre-fraud intelligence baseline unavailable: %s", exc)

    # CORS middleware for development (Vite dev server on :3006 / :5173)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:3000",
            "http://127.0.0.1:3000",
            "http://localhost:3006",
            "http://127.0.0.1:3006",
            "http://localhost:5173",
            "http://127.0.0.1:5173",
        ],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Configure Jinja2 templates
    app.state.templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

    # Register routers
    app.include_router(analyst_router)
    app.include_router(analytics_router)
    app.include_router(countermeasures_router)
    app.include_router(dashboard_router)
    app.include_router(fraud_router)
    app.include_router(pre_fraud_intel_router)
    app.include_router(intelligence_router)
    app.include_router(simulation_router)

    # ── Landing page ──────────────────────────────────────────────
    landing_file = PROJECT_ROOT / "landing.html"

    @app.get("/", response_class=HTMLResponse)
    async def serve_landing():
        return landing_file.read_text(encoding="utf-8")

    @app.get("/landing", response_class=HTMLResponse)
    async def serve_landing_alt():
        return landing_file.read_text(encoding="utf-8")

    @app.get("/ask")
    async def ask_ollama():
        ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434").rstrip("/")
        model = os.getenv("OLLAMA_MODEL", "qwen3.5:4b")
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                f"{ollama_url}/api/generate",
                json={
                    "model": model,
                    "prompt": "Say hello in one sentence.",
                    "stream": False,
                    "keep_alive": "30m",
                    "options": {
                        "num_ctx": 2048,
                        "num_predict": 32,
                        "temperature": 0.1,
                    },
                },
            )
            response.raise_for_status()
            return response.json()

    @app.get("/api/v1/llm/status")
    async def llm_status():
        ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434").rstrip("/")
        target_model = os.getenv("OLLAMA_MODEL", "qwen3.5:4b")
        status = {
            "target_model": target_model,
            "ollama_url": ollama_url,
            "target_installed": False,
            "target_running": False,
            "installed_models": [],
            "running_models": [],
            "reachable": False,
        }
        async with httpx.AsyncClient(timeout=10.0) as client:
            tags = await client.get(f"{ollama_url}/api/tags")
            tags.raise_for_status()
            status["reachable"] = True
            installed = [
                row.get("model") or row.get("name")
                for row in tags.json().get("models", [])
                if isinstance(row, dict) and (row.get("model") or row.get("name"))
            ]
            status["installed_models"] = installed
            status["target_installed"] = any(
                name == target_model or name.startswith(f"{target_model}:")
                for name in installed
            )

            try:
                ps = await client.get(f"{ollama_url}/api/ps")
                ps.raise_for_status()
                running = [
                    row.get("model") or row.get("name")
                    for row in ps.json().get("models", [])
                    if isinstance(row, dict) and (row.get("model") or row.get("name"))
                ]
                status["running_models"] = running
                status["target_running"] = any(
                    name == target_model or name.startswith(f"{target_model}:")
                    for name in running
                )
            except Exception:
                pass

        return status

    # Serve production frontend build if available
    if FRONTEND_DIST.exists() and (FRONTEND_DIST / "index.html").exists():
        assets_dir = FRONTEND_DIST / "assets"
        if assets_dir.exists():
            app.mount(
                "/assets",
                StaticFiles(directory=str(assets_dir)),
                name="frontend-assets",
            )

        @app.get("/app", response_class=HTMLResponse)
        async def serve_spa_root():
            return (FRONTEND_DIST / "index.html").read_text(encoding="utf-8")

        @app.get("/app/{full_path:path}", response_class=HTMLResponse)
        async def serve_spa(full_path: str):
            return (FRONTEND_DIST / "index.html").read_text(encoding="utf-8")

        logger.info("Frontend SPA build mounted from %s", FRONTEND_DIST)

    return app
