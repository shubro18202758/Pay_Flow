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
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from src.api.routes.analyst import router as analyst_router
from src.api.routes.analytics import router as analytics_router
from src.api.routes.dashboard import router as dashboard_router
from src.api.routes.fraud import router as fraud_router
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
    app.include_router(dashboard_router)
    app.include_router(fraud_router)
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

    # Serve production frontend build if available
    if FRONTEND_DIST.exists() and (FRONTEND_DIST / "index.html").exists():
        assets_dir = FRONTEND_DIST / "assets"
        if assets_dir.exists():
            app.mount(
                "/assets",
                StaticFiles(directory=str(assets_dir)),
                name="frontend-assets",
            )

        @app.get("/app/{full_path:path}", response_class=HTMLResponse)
        async def serve_spa(full_path: str):
            return (FRONTEND_DIST / "index.html").read_text()

        logger.info("Frontend SPA build mounted from %s", FRONTEND_DIST)

    return app
