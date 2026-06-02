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
import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from fastapi import FastAPI, Request
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
from src.api.routes.rbac import router as rbac_router
from src.api.routes.simulation import router as simulation_router
from config.settings import OLLAMA_CFG

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

        get_pre_fraud_intel_service().refresh()
        logger.info("Pre-fraud intelligence baseline refreshed from bounded public-source adapters")
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
    app.include_router(rbac_router)
    app.include_router(simulation_router)

    # ── Frontend shell ────────────────────────────────────────────
    # Coolify/uvicorn serves the same React bundle that local Vite uses, so the
    # Union Bank landing and RBAC gates cannot drift from the prototype console.
    landing_file = PROJECT_ROOT / "landing.html"
    frontend_index = FRONTEND_DIST / "index.html"
    has_frontend_build = FRONTEND_DIST.exists() and frontend_index.exists()

    def read_frontend_shell() -> str:
        if has_frontend_build:
            return frontend_index.read_text(encoding="utf-8")
        return landing_file.read_text(encoding="utf-8")

    @app.get("/", response_class=HTMLResponse)
    async def serve_landing():
        return read_frontend_shell()

    @app.get("/landing", response_class=HTMLResponse)
    async def serve_landing_alt():
        return read_frontend_shell()

    @app.get("/ask")
    async def ask_ollama():
        ollama_url = OLLAMA_CFG.base_url.rstrip("/")
        model = OLLAMA_CFG.custom_model
        async with httpx.AsyncClient(timeout=OLLAMA_CFG.request_timeout_sec) as client:
            response = await client.post(
                f"{ollama_url}/api/generate",
                json={
                    "model": model,
                    "prompt": "Reply with exactly one short sentence: Hello from PayFlow.",
                    "stream": False,
                    "think": False,
                    "keep_alive": OLLAMA_CFG.keep_alive,
                    "options": {
                        "num_ctx": OLLAMA_CFG.num_ctx_status,
                        "num_predict": min(96, OLLAMA_CFG.max_predict_tokens),
                        "temperature": OLLAMA_CFG.intent_temperature,
                        "top_k": OLLAMA_CFG.top_k,
                        "top_p": OLLAMA_CFG.top_p,
                        "repeat_penalty": OLLAMA_CFG.repeat_penalty,
                        "num_batch": OLLAMA_CFG.num_batch,
                    },
                },
            )
            response.raise_for_status()
            return response.json()

    @app.get("/api/v1/llm/status")
    async def llm_status(request: Request):
        orch = getattr(request.app.state, "orchestrator", None)
        llm = getattr(orch, "_llm", None) if orch is not None else None
        if llm is not None and hasattr(llm, "status"):
            return await asyncio.to_thread(llm.status)

        ollama_url = OLLAMA_CFG.base_url.rstrip("/")
        target_model = OLLAMA_CFG.custom_model
        required_prefix = OLLAMA_CFG.required_model_prefix.strip().lower()

        def family_ok(model_name: str) -> bool:
            if not OLLAMA_CFG.strict_model_family or not required_prefix:
                return True
            return model_name.strip().lower().startswith(required_prefix)

        status = {
            "base_url": ollama_url,
            "target_model": target_model,
            "fallback_model": OLLAMA_CFG.model,
            "required_model_prefix": OLLAMA_CFG.required_model_prefix,
            "strict_model_family": OLLAMA_CFG.strict_model_family,
            "target_family_ok": family_ok(target_model),
            "ollama_url": ollama_url,
            "target_installed": False,
            "target_running": False,
            "installed_models": [],
            "running_models": [],
            "acceptable_installed": [],
            "reachable": False,
        }
        try:
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
                    name == target_model or name.startswith(target_model)
                    for name in installed
                )
                status["acceptable_installed"] = [
                    name for name in installed if family_ok(name)
                ]

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
                        name == target_model or name.startswith(target_model)
                        for name in running
                    )
                    status["acceptable_running"] = [
                        name for name in running if family_ok(name)
                    ]
                except Exception as exc:
                    status["runtime_probe_error"] = str(exc)
        except Exception as exc:
            status["error"] = str(exc)

        return status

    # Serve production frontend build if available
    if has_frontend_build:
        assets_dir = FRONTEND_DIST / "assets"
        if assets_dir.exists():
            app.mount(
                "/assets",
                StaticFiles(directory=str(assets_dir)),
                name="frontend-assets",
            )

        @app.get("/app", response_class=HTMLResponse)
        async def serve_spa_root():
            return read_frontend_shell()

        @app.get("/app/{full_path:path}", response_class=HTMLResponse)
        async def serve_spa(full_path: str):
            return read_frontend_shell()

        logger.info("Frontend SPA build mounted from %s", FRONTEND_DIST)

    return app
