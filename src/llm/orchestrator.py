"""
PayFlow — LLM Orchestrator Client
===================================
Wraps the Ollama Python SDK with VRAM-aware lifecycle management.
Every call flows through: health_check → VRAM mode switch → inference → release.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import AsyncIterator, Optional

import httpx
import ollama

from config.settings import OLLAMA_CFG
from config.vram_manager import assistant_mode
from src.llm.health_check import check_vram_for_llm

logger = logging.getLogger(__name__)

# Custom model tag built by deploy_ollama.sh
PAYFLOW_MODEL = "payflow-qwen"


class VRAMInsufficientError(RuntimeError):
    """Raised when GPU health check fails before LLM initialization."""


@dataclass
class LLMResponse:
    content: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_duration_ms: float


class PayFlowLLM:
    """
    VRAM-aware LLM client. Enforces the health gate before every inference
    call and manages GPU mode transitions via the VRAM lifecycle manager.

    Usage:
        llm = PayFlowLLM()

        # Synchronous (simple)
        response = llm.query("Analyze this transaction chain for layering...")

        # With structured context
        response = llm.analyze_fraud(
            context={"transactions": [...], "risk_scores": {...}},
            question="Is this a round-tripping pattern?"
        )
    """

    def __init__(
        self,
        model: str = PAYFLOW_MODEL,
        fallback_model: str | None = None,
        skip_health_check: bool = False,
    ):
        self._model = model
        self._fallback_model = fallback_model or OLLAMA_CFG.model
        self._skip_health_check = skip_health_check
        self._client = ollama.Client(host=OLLAMA_CFG.base_url)
        self._gpu_queue = None  # set via set_priority_queue()

    def set_priority_queue(self, queue) -> None:
        """Wire the GPU priority queue for dynamic num_ctx."""
        self._gpu_queue = queue

    def _get_num_ctx(self) -> int:
        """Return the current dynamic num_ctx from the priority queue, or static default."""
        if self._gpu_queue is not None:
            return self._gpu_queue.current_num_ctx
        return OLLAMA_CFG.num_ctx

    def _ensure_model_available(self) -> str:
        """Check if the custom model exists, fall back to base if not."""
        try:
            models = self._client.list()
            if isinstance(models, dict):
                rows = models.get("models", [])
                available = {
                    row.get("model") or row.get("name")
                    for row in rows
                    if isinstance(row, dict) and (row.get("model") or row.get("name"))
                }
            else:
                available = {
                    getattr(m, "model", None) or getattr(m, "name", None)
                    for m in getattr(models, "models", [])
                    if getattr(m, "model", None) or getattr(m, "name", None)
                }

            # Try exact match first, then prefix match for the custom model.
            if self._model in available:
                return self._model
            for name in available:
                if name.startswith(self._model):
                    return name

            # Fall back to the configured base model, also allowing prefix matches
            # because local Ollama installs often expose quantized suffix variants.
            if self._fallback_model in available:
                return self._fallback_model
            for name in available:
                if name.startswith(self._fallback_model):
                    logger.warning(
                        "Custom model '%s' not found. Using compatible local model '%s'.",
                        self._model, name,
                    )
                    return name

            logger.warning(
                "Custom model '%s' not found. Falling back to '%s'.",
                self._model, self._fallback_model,
            )
            return self._fallback_model
        except Exception as exc:
            logger.error("Failed to list Ollama models: %s", exc)
            return self._fallback_model

    def _pre_flight(self) -> str:
        """Run VRAM health check and resolve model name."""
        if not self._skip_health_check:
            result = check_vram_for_llm()
            if not result.passed:
                raise VRAMInsufficientError(result.message)
            logger.info("VRAM health check passed: %s", result.message)

        return self._ensure_model_available()

    def query(
        self,
        prompt: str,
        system: str | None = None,
        temperature: float | None = None,
        max_tokens: int = 2048,
    ) -> LLMResponse:
        """
        Single-turn synchronous query with VRAM lifecycle management.
        Automatically enters assistant_mode, runs inference, then releases.
        """
        model = self._pre_flight()

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        with assistant_mode():
            response = self._client.chat(
                model=model,
                messages=messages,
                options={
                    "temperature": temperature or OLLAMA_CFG.temperature,
                    "num_predict": max_tokens,
                    "num_ctx": self._get_num_ctx(),
                    "top_p": OLLAMA_CFG.top_p,
                },
            )

        return LLMResponse(
            content=response.message.content,
            model=model,
            prompt_tokens=response.prompt_eval_count or 0,
            completion_tokens=response.eval_count or 0,
            total_duration_ms=(response.total_duration or 0) / 1_000_000,
        )

    def analyze_fraud(
        self,
        context: dict,
        question: str,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        """
        Structured fraud analysis query. Injects transaction context into
        the system prompt and asks a targeted forensic question.
        """
        import json

        system_prompt = (
            "You are PayFlow Forensic Analyst. Analyze the following financial data "
            "and respond with structured sections: FINDING, EVIDENCE, RISK ASSESSMENT, "
            "RECOMMENDED ACTION.\n\n"
            "=== FINANCIAL DATA CONTEXT ===\n"
            f"{json.dumps(context, indent=2, default=str)}\n"
            "=== END CONTEXT ===\n"
        )

        return self.query(
            prompt=question,
            system=system_prompt,
            max_tokens=max_tokens,
        )

    def stream_query(
        self,
        prompt: str,
        system: str | None = None,
    ) -> AsyncIterator[str]:
        """
        Streaming inference for real-time UI token delivery.
        Yields content chunks as they arrive from Ollama.
        """
        model = self._pre_flight()

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        # Note: assistant_mode is entered but not exited until stream completes.
        # Caller is responsible for not running analysis workloads during streaming.
        stream = self._client.chat(
            model=model,
            messages=messages,
            stream=True,
            options={
                "temperature": OLLAMA_CFG.temperature,
                "num_ctx": OLLAMA_CFG.num_ctx,
            },
        )

        for chunk in stream:
            token = chunk.message.content
            if token:
                yield token

    def unload(self) -> None:
        """Explicitly unload the model from VRAM."""
        try:
            httpx.post(
                f"{OLLAMA_CFG.base_url}/api/generate",
                json={"model": self._model, "keep_alive": 0},
                timeout=10.0,
            )
            logger.info("Model '%s' unloaded from VRAM.", self._model)
        except Exception as exc:
            logger.warning("Failed to unload model: %s", exc)

    def is_daemon_alive(self) -> bool:
        """Check if the Ollama daemon is responsive."""
        try:
            resp = httpx.get(f"{OLLAMA_CFG.base_url}/api/tags", timeout=5.0)
            return resp.status_code == 200
        except Exception:
            return False
