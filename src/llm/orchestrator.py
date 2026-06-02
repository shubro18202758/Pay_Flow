"""
PayFlow — LLM Orchestrator Client
===================================
Wraps the Ollama Python SDK with VRAM-aware lifecycle management.
Every call flows through: health_check → VRAM mode switch → inference → release.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from dataclasses import dataclass
from typing import Any, AsyncIterator, Optional

import httpx
import ollama

from config.settings import OLLAMA_CFG
from config.vram_manager import assistant_mode, _flush_torch_cache
from src.llm.health_check import check_vram_for_llm

logger = logging.getLogger(__name__)

# Custom model tag built by deploy_ollama.sh, or the deployment-provided model.
PAYFLOW_MODEL = OLLAMA_CFG.custom_model


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
        self._resolved_model: str | None = None
        self._model_checked_at: float = 0.0
        self._request_gate = threading.BoundedSemaphore(
            max(1, OLLAMA_CFG.max_parallel_requests)
        )

    def set_priority_queue(self, queue) -> None:
        """Wire the GPU priority queue for dynamic num_ctx."""
        self._gpu_queue = queue

    def _get_num_ctx(self) -> int:
        """Return the current dynamic num_ctx from the priority queue, or static default."""
        if self._gpu_queue is not None:
            return self._gpu_queue.current_num_ctx
        return OLLAMA_CFG.num_ctx

    def _model_family_allowed(self, model_name: str) -> bool:
        """Return whether a model satisfies the configured Qwen family pin."""
        if not OLLAMA_CFG.strict_model_family:
            return True
        required = OLLAMA_CFG.required_model_prefix.strip().lower()
        if not required:
            return True
        return model_name.strip().lower().startswith(required)

    def _require_allowed_model(self, model_name: str) -> str:
        if self._model_family_allowed(model_name):
            return model_name
        raise RuntimeError(
            "Configured Ollama model "
            f"'{model_name}' violates PAYFLOW_REQUIRED_OLLAMA_PREFIX="
            f"'{OLLAMA_CFG.required_model_prefix}'. Install/use qwen3.5:4b "
            "or set PAYFLOW_STRICT_OLLAMA_MODEL=0 only for explicit tests."
        )

    def _ensure_model_available(self, *, force_refresh: bool = False) -> str:
        """Check if the custom model exists, fall back to base if not."""
        if (
            not force_refresh
            and self._resolved_model is not None
            and time.monotonic() - self._model_checked_at < 60.0
        ):
            return self._resolved_model

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
                self._resolved_model = self._require_allowed_model(self._model)
                self._model_checked_at = time.monotonic()
                return self._resolved_model
            for name in available:
                if name.startswith(self._model) and self._model_family_allowed(name):
                    self._resolved_model = self._require_allowed_model(name)
                    self._model_checked_at = time.monotonic()
                    return self._resolved_model

            # Fall back to the configured base model, also allowing prefix matches
            # because local Ollama installs often expose quantized suffix variants.
            if self._fallback_model in available:
                self._resolved_model = self._require_allowed_model(self._fallback_model)
                self._model_checked_at = time.monotonic()
                return self._resolved_model
            for name in available:
                if name.startswith(self._fallback_model) and self._model_family_allowed(name):
                    logger.warning(
                        "Custom model '%s' not found. Using compatible local model '%s'.",
                        self._model, name,
                    )
                    self._resolved_model = self._require_allowed_model(name)
                    self._model_checked_at = time.monotonic()
                    return self._resolved_model

            if not self._model_family_allowed(self._fallback_model):
                raise RuntimeError(
                    "No acceptable Qwen 3.5 Ollama model was found. "
                    f"Installed models: {sorted(available)}"
                )
            logger.warning(
                "Custom model '%s' not found. Falling back to '%s'.",
                self._model, self._fallback_model,
            )
            self._resolved_model = self._require_allowed_model(self._fallback_model)
            self._model_checked_at = time.monotonic()
            return self._resolved_model
        except RuntimeError:
            raise
        except Exception as exc:
            logger.error("Failed to list Ollama models: %s", exc)
            return self._require_allowed_model(self._fallback_model)

    def _pre_flight(self) -> str:
        """Run VRAM health check and resolve model name."""
        if not self._skip_health_check:
            last_result = None
            for attempt in range(3):
                result = check_vram_for_llm()
                last_result = result
                if result.passed:
                    logger.info("VRAM health check passed: %s", result.message)
                    break
                if attempt < 2:
                    logger.warning(
                        "VRAM health check failed before LLM call (attempt %d/3): %s",
                        attempt + 1,
                        result.message,
                    )
                    _flush_torch_cache()
                    time.sleep(1.5 * (attempt + 1))
            else:
                message = last_result.message if last_result else "Unknown VRAM health check failure"
                raise VRAMInsufficientError(message)

        return self._ensure_model_available()

    def _options(
        self,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        num_ctx: int | None = None,
    ) -> dict[str, Any]:
        return {
            "temperature": OLLAMA_CFG.temperature if temperature is None else temperature,
            "num_predict": max_tokens or OLLAMA_CFG.max_predict_tokens,
            "num_ctx": num_ctx or self._get_num_ctx(),
            "top_k": OLLAMA_CFG.top_k,
            "top_p": OLLAMA_CFG.top_p,
            "repeat_penalty": OLLAMA_CFG.repeat_penalty,
            "num_batch": OLLAMA_CFG.num_batch,
            "seed": OLLAMA_CFG.seed,
        }

    def status(self) -> dict[str, Any]:
        """Return Ollama reachability and model installation/runtime status."""
        target = self._model
        status: dict[str, Any] = {
            "base_url": OLLAMA_CFG.base_url,
            "target_model": target,
            "fallback_model": self._fallback_model,
            "resolved_model": self._resolved_model,
            "required_model_prefix": OLLAMA_CFG.required_model_prefix,
            "strict_model_family": OLLAMA_CFG.strict_model_family,
            "reachable": False,
            "target_installed": False,
            "target_running": False,
            "target_family_ok": self._model_family_allowed(target),
            "installed_models": [],
            "running_models": [],
        }
        try:
            tags = httpx.get(f"{OLLAMA_CFG.base_url}/api/tags", timeout=5.0)
            tags.raise_for_status()
            status["reachable"] = True
            installed = [
                row.get("model") or row.get("name")
                for row in tags.json().get("models", [])
                if isinstance(row, dict) and (row.get("model") or row.get("name"))
            ]
            status["installed_models"] = installed
            status["target_installed"] = any(
                name == target or name.startswith(target)
                for name in installed
            )
            status["acceptable_installed"] = [
                name for name in installed if self._model_family_allowed(name)
            ]
        except Exception as exc:
            status["error"] = str(exc)
            return status

        try:
            ps = httpx.get(f"{OLLAMA_CFG.base_url}/api/ps", timeout=5.0)
            ps.raise_for_status()
            running = [
                row.get("model") or row.get("name")
                for row in ps.json().get("models", [])
                if isinstance(row, dict) and (row.get("model") or row.get("name"))
            ]
            status["running_models"] = running
            status["target_running"] = any(
                name == target or name.startswith(target)
                for name in running
            )
        except Exception:
            pass

        return status

    def warmup(self) -> LLMResponse:
        """Load Qwen with a tiny deterministic request so the first UI query is not cold."""
        return self.query(
            "Reply with exactly: ready",
            temperature=0.0,
            max_tokens=8,
            num_ctx=OLLAMA_CFG.num_ctx_status,
        )

    def query(
        self,
        prompt: str,
        system: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        num_ctx: int | None = None,
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

        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": False,
            "think": False,
            "keep_alive": OLLAMA_CFG.keep_alive,
            "options": self._options(
                temperature=temperature,
                max_tokens=max_tokens,
                num_ctx=num_ctx,
            ),
        }

        with self._request_gate:
            with assistant_mode():
                response = httpx.post(
                    f"{OLLAMA_CFG.base_url}/api/chat",
                    json=payload,
                    timeout=OLLAMA_CFG.request_timeout_sec,
                )
                response.raise_for_status()
                data = response.json()

        message = data.get("message", {}) if isinstance(data, dict) else {}
        content = message.get("content", "") if isinstance(message, dict) else ""
        prompt_tokens = data.get("prompt_eval_count") or 0
        completion_tokens = data.get("eval_count") or 0
        total_duration = data.get("total_duration") or 0

        return LLMResponse(
            content=content,
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_duration_ms=total_duration / 1_000_000,
        )

    async def generate(
        self,
        prompt: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
        num_ctx: int | None = None,
    ) -> str:
        """Async text-generation adapter used by dashboard NL query routes."""
        response = await asyncio.to_thread(
            self.query,
            prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            num_ctx=num_ctx,
        )
        return response.content

    def chat(
        self,
        messages: list[dict],
        *,
        tools: list[dict] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        num_ctx: int | None = None,
        timeout: float | None = None,
        response_format: str | dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Shared non-streaming chat path for agents and API workflows."""
        model = self._pre_flight()
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": False,
            "think": False,
            "keep_alive": OLLAMA_CFG.keep_alive,
            "options": self._options(
                temperature=temperature,
                max_tokens=max_tokens,
                num_ctx=num_ctx,
            ),
        }
        if tools:
            payload["tools"] = tools
        if response_format:
            payload["format"] = response_format

        with self._request_gate:
            with assistant_mode():
                try:
                    response = httpx.post(
                        f"{OLLAMA_CFG.base_url}/api/chat",
                        json=payload,
                        timeout=timeout or OLLAMA_CFG.request_timeout_sec,
                    )
                    response.raise_for_status()
                except httpx.HTTPStatusError as exc:
                    if isinstance(response_format, dict) and exc.response.status_code == 400:
                        payload["format"] = "json"
                        response = httpx.post(
                            f"{OLLAMA_CFG.base_url}/api/chat",
                            json=payload,
                            timeout=timeout or OLLAMA_CFG.request_timeout_sec,
                        )
                        response.raise_for_status()
                    else:
                        raise
                data = response.json()

        message = data.get("message", {}) if isinstance(data, dict) else {}
        raw_tool_calls = message.get("tool_calls", []) if isinstance(message, dict) else []
        parsed_calls: list[dict[str, Any]] = []
        for tc in raw_tool_calls:
            if not isinstance(tc, dict):
                continue
            function_data = tc.get("function", {})
            if not isinstance(function_data, dict):
                continue
            arguments = function_data.get("arguments", {})
            parsed_calls.append({
                "name": function_data.get("name", ""),
                "arguments": arguments,
            })

        return {
            "content": message.get("content", "") if isinstance(message, dict) else "",
            "tool_calls": parsed_calls,
            "model": model,
            "raw": data,
        }

    def analyze_fraud(
        self,
        context: dict,
        question: str,
        max_tokens: int | None = None,
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
            max_tokens=max_tokens or OLLAMA_CFG.max_predict_tokens,
            num_ctx=OLLAMA_CFG.num_ctx_interactive,
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
            keep_alive=OLLAMA_CFG.keep_alive,
            options=self._options(max_tokens=OLLAMA_CFG.max_predict_tokens),
        )

        for chunk in stream:
            if isinstance(chunk, dict):
                message = chunk.get("message", {})
                token = message.get("content", "") if isinstance(message, dict) else ""
            else:
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
