"""
PayFlow -- QLoRA + GRPO Fine-Tuning Pipeline
==============================================
Local fine-tuning pipeline for Qwen 3.5 9B operating within the 8 GB VRAM
constraint of an RTX 4070.

Architecture::

    ┌──────────────────────────────────────────────────────────────────┐
    │                    finetuning_mode() (exclusive GPU)            │
    │                                                                  │
    │  ┌───────────────────┐       ┌────────────────────────────┐     │
    │  │  QLoRAFineTuner   │       │      GRPOTrainer           │     │
    │  │  ─────────────    │       │      ───────────           │     │
    │  │  • load_model()   │──────>│  • train()                 │     │
    │  │  • apply_lora()   │       │  • compute_fraud_reward()  │     │
    │  │  • save_adapter() │<──────│  • generate_completions()  │     │
    │  │  • load_adapter() │       │  • policy_gradient_step()  │     │
    │  └───────────────────┘       └────────────────────────────┘     │
    │                                                                  │
    │  Memory guard: gradient checkpointing + periodic cache flush     │
    └──────────────────────────────────────────────────────────────────┘

VRAM Budget (8 GB ceiling)::

    Component                  VRAM
    ─────────────────────────  ──────
    Base model (NF4 4-bit)     ~2.2 GB  (Qwen 3.5 9B in double-quant NF4)
    LoRA adapters (rank 16)    ~50 MB   (trainable delta matrices)
    Activations + grad ckpt    ~2.5 GB  (offloaded to CPU on demand)
    Optimizer states (AdamW)   ~200 MB  (only LoRA params in 32-bit)
    KV cache (generation)      ~1.5 GB  (GRPO group completions)
    CUDA context + overhead    ~500 MB
    ─────────────────────────  ──────
    Total                      ~7.0 GB  (under 8 GB with margin)

Dependencies (lazy-imported):
    - transformers (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig)
    - peft (LoraConfig, get_peft_model, PeftModel, TaskType)
    - trl (GRPOConfig, GRPOTrainer as TRLGRPOTrainer)
    - bitsandbytes (4-bit CUDA kernels)
    - torch (CUDA memory management)
"""

from __future__ import annotations

import gc
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from config.settings import FINETUNE_CFG, FineTuningConfig

logger = logging.getLogger(__name__)


# ── Dataclasses ──────────────────────────────────────────────────────────────


@dataclass
class FineTuneMetrics:
    """Accumulated metrics from a fine-tuning run."""

    total_steps: int = 0
    completed_epochs: int = 0
    train_loss: float = 0.0
    avg_reward: float = 0.0
    peak_vram_mb: float = 0.0
    wall_clock_seconds: float = 0.0
    checkpoints_saved: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_steps": self.total_steps,
            "completed_epochs": self.completed_epochs,
            "train_loss": round(self.train_loss, 6),
            "avg_reward": round(self.avg_reward, 4),
            "peak_vram_mb": round(self.peak_vram_mb, 1),
            "wall_clock_seconds": round(self.wall_clock_seconds, 2),
            "checkpoints_saved": self.checkpoints_saved,
        }


@dataclass
class RewardSignal:
    """Reward output from fraud topology evaluation."""

    score: float                      # 0.0–1.0 overall reward
    typology_correct: bool            # did the model identify the right fraud type?
    evidence_quality: float           # 0.0–1.0 evidence citation score
    verdict_calibration: float        # 0.0–1.0 confidence calibration

    def to_dict(self) -> dict[str, Any]:
        return {
            "score": round(self.score, 4),
            "typology_correct": self.typology_correct,
            "evidence_quality": round(self.evidence_quality, 4),
            "verdict_calibration": round(self.verdict_calibration, 4),
        }


# ── Memory Management ────────────────────────────────────────────────────────


def aggressive_memory_clear() -> None:
    """
    Flush all reclaimable GPU memory back to the CUDA driver.

    Called between training steps and after generation to prevent OOM during
    the backward pass.  Also triggers Python GC to collect orphan tensors.
    """
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except ImportError:
        pass


def get_peak_vram_mb() -> float:
    """Return peak allocated VRAM in MB since last reset, or 0.0 if unavailable."""
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / (1024 * 1024)
    except ImportError:
        pass
    return 0.0


def reset_peak_vram_tracker() -> None:
    """Reset the CUDA peak-memory tracker for a clean measurement window."""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except ImportError:
        pass


# ── BitsAndBytes Quantization Config ─────────────────────────────────────────


def build_bnb_config(cfg: FineTuningConfig | None = None) -> Any:
    """
    Construct a BitsAndBytesConfig for 4-bit NF4 loading.

    Returns:
        transformers.BitsAndBytesConfig configured for NF4 double-quant.
    """
    from transformers import BitsAndBytesConfig
    import torch

    cfg = cfg or FINETUNE_CFG
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=cfg.quant_type,          # "nf4"
        bnb_4bit_use_double_quant=cfg.use_double_quant,
        bnb_4bit_compute_dtype=torch.bfloat16 if cfg.bf16 else torch.float16,
    )


# ── LoRA Adapter Config ─────────────────────────────────────────────────────


def build_lora_config(cfg: FineTuningConfig | None = None) -> Any:
    """
    Construct a PEFT LoraConfig targeting attention projection matrices.

    Returns:
        peft.LoraConfig with rank, alpha, dropout, and target modules.
    """
    from peft import LoraConfig, TaskType

    cfg = cfg or FINETUNE_CFG
    return LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=list(cfg.target_modules),
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    )


# ── Reward Computation ───────────────────────────────────────────────────────


VALID_TYPOLOGIES = frozenset({
    "layering", "structuring", "round_tripping", "mule_chain",
    "dormant_reactivation", "fan_out", "fan_in", "smurfing",
})


def compute_fraud_reward(
    model_output: str,
    ground_truth_typology: str,
    ground_truth_evidence: list[str],
) -> RewardSignal:
    """
    Evaluate a model completion against ground-truth fraud topology labels.

    The reward is a weighted combination of:
        - Typology identification accuracy   (40%)
        - Evidence citation quality           (35%)
        - Verdict confidence calibration      (25%)

    Args:
        model_output: Raw model generation (expected to contain JSON verdict).
        ground_truth_typology: Correct fraud typology label.
        ground_truth_evidence: List of evidence strings the model should cite.

    Returns:
        RewardSignal with decomposed scores.
    """
    # --- Parse model output ---
    verdict = _extract_verdict_json(model_output)

    # --- Typology match (exact string match, case-insensitive) ---
    predicted_typology = verdict.get("fraud_typology", "").lower().strip()
    typology_correct = predicted_typology == ground_truth_typology.lower().strip()
    typology_score = 1.0 if typology_correct else 0.0

    # --- Evidence quality (Jaccard overlap with ground truth) ---
    predicted_evidence = {
        e.strip().lower() for e in verdict.get("evidence_cited", []) if isinstance(e, str)
    }
    gt_evidence = {e.strip().lower() for e in ground_truth_evidence}
    if gt_evidence:
        intersection = predicted_evidence & gt_evidence
        union = predicted_evidence | gt_evidence
        evidence_quality = len(intersection) / len(union) if union else 0.0
    else:
        evidence_quality = 1.0 if not predicted_evidence else 0.0

    # --- Confidence calibration (penalize over/under confidence) ---
    confidence = verdict.get("confidence", 0.5)
    if not isinstance(confidence, (int, float)):
        confidence = 0.5
    confidence = max(0.0, min(1.0, float(confidence)))
    # Ideal: high confidence when correct, low when incorrect
    if typology_correct:
        verdict_calibration = confidence  # reward high confidence on correct
    else:
        verdict_calibration = 1.0 - confidence  # reward low confidence on wrong

    # --- Weighted composite ---
    score = (
        0.40 * typology_score
        + 0.35 * evidence_quality
        + 0.25 * verdict_calibration
    )

    return RewardSignal(
        score=score,
        typology_correct=typology_correct,
        evidence_quality=evidence_quality,
        verdict_calibration=verdict_calibration,
    )


def _extract_verdict_json(text: str) -> dict[str, Any]:
    """
    Best-effort extraction of a JSON verdict block from model output.

    Handles:
        - Raw JSON string
        - JSON within markdown code fences
        - Fallback to empty dict
    """
    import re

    # Try markdown fenced block first
    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence_match:
        try:
            return json.loads(fence_match.group(1))
        except json.JSONDecodeError:
            pass

    # Try raw JSON object
    brace_match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
    if brace_match:
        try:
            return json.loads(brace_match.group(0))
        except json.JSONDecodeError:
            pass

    return {}


# ── QLoRA Fine-Tuner ─────────────────────────────────────────────────────────


class QLoRAFineTuner:
    """
    Manages the lifecycle of a QLoRA-adapted Qwen 3.5 9B model.

    Handles:
        - 4-bit quantized model loading via bitsandbytes
        - LoRA adapter attachment via PEFT
        - Adapter checkpoint save/load (base model weights are frozen)
        - Trainable parameter accounting
    """

    def __init__(self, config: FineTuningConfig | None = None) -> None:
        self.config = config or FINETUNE_CFG
        self.model: Any = None
        self.tokenizer: Any = None
        self._lora_attached: bool = False
        self._model_loaded: bool = False

    @property
    def is_ready(self) -> bool:
        """True when model is loaded and LoRA adapters are attached."""
        return self._model_loaded and self._lora_attached

    def load_model(self, model_name: str | None = None) -> None:
        """
        Load the base model in 4-bit quantized mode with bitsandbytes.

        Args:
            model_name: HuggingFace model ID. Defaults to config.base_model.
        """
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_name = model_name or self.config.base_model
        bnb_config = build_bnb_config(self.config)

        logger.info("Loading base model '%s' in 4-bit NF4...", model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
        self.model.config.use_cache = False  # incompatible with grad checkpointing

        if self.config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
            logger.info("Gradient checkpointing enabled.")

        self._model_loaded = True
        logger.info(
            "Model loaded. VRAM: %.1f MB allocated.",
            get_peak_vram_mb(),
        )

    def apply_lora(self) -> None:
        """Wrap the loaded model with LoRA adapters via PEFT."""
        if not self._model_loaded:
            raise RuntimeError("Call load_model() before apply_lora().")

        from peft import get_peft_model, prepare_model_for_kbit_training

        self.model = prepare_model_for_kbit_training(self.model)
        lora_config = build_lora_config(self.config)
        self.model = get_peft_model(self.model, lora_config)

        self._lora_attached = True
        trainable, total = self.trainable_param_count()
        logger.info(
            "LoRA adapters attached. Trainable: %s / %s (%.2f%%)",
            f"{trainable:,}", f"{total:,}", 100.0 * trainable / total if total else 0,
        )

    def trainable_param_count(self) -> tuple[int, int]:
        """Return (trainable_params, total_params)."""
        if self.model is None:
            return 0, 0
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        return trainable, total

    def save_adapter(self, output_dir: str | Path | None = None) -> Path:
        """
        Save LoRA adapter weights (not the base model).

        Returns:
            Path to the saved adapter directory.
        """
        if not self._lora_attached:
            raise RuntimeError("No LoRA adapters to save. Call apply_lora() first.")

        save_path = Path(output_dir or self.config.output_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(str(save_path))
        self.tokenizer.save_pretrained(str(save_path))
        logger.info("LoRA adapter saved to %s", save_path)
        return save_path

    def load_adapter(self, adapter_dir: str | Path) -> None:
        """
        Load previously saved LoRA adapter weights onto the base model.

        Args:
            adapter_dir: Path containing adapter_config.json + adapter weights.
        """
        if not self._model_loaded:
            raise RuntimeError("Call load_model() before load_adapter().")

        from peft import PeftModel

        adapter_path = Path(adapter_dir)
        self.model = PeftModel.from_pretrained(self.model, str(adapter_path))
        self._lora_attached = True
        logger.info("LoRA adapter loaded from %s", adapter_path)

    def unload(self) -> None:
        """Release the model and free GPU memory."""
        self.model = None
        self.tokenizer = None
        self._model_loaded = False
        self._lora_attached = False
        aggressive_memory_clear()
        logger.info("Model unloaded, VRAM released.")


# ── GRPO Trainer ─────────────────────────────────────────────────────────────


class GRPOTrainer:
    """
    Group Relative Policy Optimization trainer for fraud-typology alignment.

    GRPO generates a group of completions per prompt, scores them with a
    reward function (fraud accuracy + evidence quality + calibration), and
    performs a policy gradient step that favours higher-reward completions.

    This class wraps TRL's GRPOTrainer with PayFlow-specific reward wiring
    and aggressive memory management for the 8 GB VRAM ceiling.
    """

    def __init__(
        self,
        fine_tuner: QLoRAFineTuner,
        config: FineTuningConfig | None = None,
    ) -> None:
        if not fine_tuner.is_ready:
            raise RuntimeError(
                "QLoRAFineTuner must be loaded and have LoRA adapters applied."
            )
        self.fine_tuner = fine_tuner
        self.config = config or fine_tuner.config
        self.metrics = FineTuneMetrics()
        self._trl_trainer: Any = None

    def build_trl_trainer(self, dataset: Any) -> Any:
        """
        Construct a TRL GRPOTrainer with PayFlow reward function.

        Args:
            dataset: HuggingFace Dataset with 'prompt' and 'ground_truth' columns.

        Returns:
            Configured trl.GRPOTrainer instance.
        """
        from trl import GRPOConfig, GRPOTrainer as TRLGRPOTrainer

        output_dir = str(self.config.output_dir / "grpo_run")
        training_args = GRPOConfig(
            output_dir=output_dir,
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.per_device_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            warmup_ratio=self.config.warmup_ratio,
            weight_decay=self.config.weight_decay,
            lr_scheduler_type=self.config.lr_scheduler_type,
            bf16=self.config.bf16,
            fp16=self.config.fp16,
            gradient_checkpointing=self.config.gradient_checkpointing,
            max_completion_length=self.config.grpo_max_new_tokens,
            num_generations=self.config.grpo_group_size,
            beta=self.config.grpo_beta,
            logging_steps=10,
            save_steps=500,
            save_total_limit=3,
        )

        self._trl_trainer = TRLGRPOTrainer(
            model=self.fine_tuner.model,
            args=training_args,
            train_dataset=dataset,
            processing_class=self.fine_tuner.tokenizer,
            reward_funcs=self._reward_function,
        )

        logger.info("TRL GRPOTrainer built. Output: %s", output_dir)
        return self._trl_trainer

    def _reward_function(self, completions: list[str], **kwargs: Any) -> list[float]:
        """
        Reward function bridge for TRL GRPOTrainer.

        Evaluates each completion against ground truth using
        compute_fraud_reward() and returns scalar rewards.
        """
        ground_truths = kwargs.get("ground_truth", [{}] * len(completions))
        rewards = []
        for completion, gt in zip(completions, ground_truths):
            if isinstance(gt, str):
                gt = json.loads(gt)
            signal = compute_fraud_reward(
                model_output=completion,
                ground_truth_typology=gt.get("typology", "unknown"),
                ground_truth_evidence=gt.get("evidence", []),
            )
            rewards.append(signal.score)
        return rewards

    def train(self, dataset: Any) -> FineTuneMetrics:
        """
        Execute the full GRPO training loop.

        Args:
            dataset: HuggingFace Dataset with columns:
                     - 'prompt': investigation prompt text
                     - 'ground_truth': JSON with 'typology' and 'evidence'

        Returns:
            FineTuneMetrics with training statistics.
        """
        reset_peak_vram_tracker()
        start_time = time.time()

        trainer = self.build_trl_trainer(dataset)

        logger.info("Starting GRPO training...")
        train_result = trainer.train()

        self.metrics.total_steps = train_result.global_step
        self.metrics.completed_epochs = self.config.num_epochs
        self.metrics.train_loss = train_result.training_loss
        self.metrics.peak_vram_mb = get_peak_vram_mb()
        self.metrics.wall_clock_seconds = time.time() - start_time

        # Save final adapter
        save_path = self.fine_tuner.save_adapter(
            self.config.output_dir / "grpo_final"
        )
        self.metrics.checkpoints_saved.append(str(save_path))

        aggressive_memory_clear()
        logger.info(
            "GRPO training complete. Steps: %d, Loss: %.4f, Peak VRAM: %.1f MB",
            self.metrics.total_steps,
            self.metrics.train_loss,
            self.metrics.peak_vram_mb,
        )
        return self.metrics


# ── Dataset Preparation ──────────────────────────────────────────────────────


def prepare_investigation_dataset(
    records: list[dict[str, Any]],
    tokenizer: Any = None,
    max_length: int | None = None,
) -> Any:
    """
    Convert raw investigation records into a HuggingFace Dataset for GRPO.

    Each record should contain:
        - 'prompt': The investigation prompt text
        - 'typology': Ground truth fraud typology label
        - 'evidence': List of ground truth evidence strings

    Args:
        records: List of investigation record dicts.
        tokenizer: Optional tokenizer for length filtering.
        max_length: Max token count to filter (uses config default if None).

    Returns:
        datasets.Dataset with 'prompt' and 'ground_truth' columns.
    """
    from datasets import Dataset

    prompts = []
    ground_truths = []

    for record in records:
        prompt = record.get("prompt", "")
        typology = record.get("typology", "unknown")
        evidence = record.get("evidence", [])

        if not prompt.strip():
            continue

        # Optionally filter by token length
        if tokenizer and max_length:
            tokens = tokenizer.encode(prompt, add_special_tokens=False)
            if len(tokens) > max_length:
                continue

        prompts.append(prompt)
        ground_truths.append(json.dumps({
            "typology": typology,
            "evidence": evidence,
        }))

    return Dataset.from_dict({
        "prompt": prompts,
        "ground_truth": ground_truths,
    })


# ── Convenience Entrypoint ───────────────────────────────────────────────────


def run_finetuning_pipeline(
    records: list[dict[str, Any]],
    config: FineTuningConfig | None = None,
    model_name: str | None = None,
) -> FineTuneMetrics:
    """
    End-to-end fine-tuning pipeline: load → LoRA → GRPO train → save.

    Should be called inside a ``finetuning_mode()`` context manager to
    ensure exclusive GPU access.

    Args:
        records: Training investigation records (prompt + ground truth).
        config: Fine-tuning config (defaults to FINETUNE_CFG singleton).
        model_name: Override base model ID.

    Returns:
        FineTuneMetrics summarising the training run.

    Example::

        from config.vram_manager import finetuning_mode
        from src.llm.finetuning import run_finetuning_pipeline

        with finetuning_mode():
            metrics = run_finetuning_pipeline(training_records)
    """
    config = config or FINETUNE_CFG

    # 1. Initialize QLoRA model
    fine_tuner = QLoRAFineTuner(config)
    fine_tuner.load_model(model_name)
    fine_tuner.apply_lora()

    # 2. Prepare dataset
    dataset = prepare_investigation_dataset(
        records,
        tokenizer=fine_tuner.tokenizer,
        max_length=config.max_seq_length,
    )
    logger.info("Dataset prepared: %d samples.", len(dataset))

    # 3. Run GRPO training
    trainer = GRPOTrainer(fine_tuner, config)
    metrics = trainer.train(dataset)

    # 4. Cleanup
    fine_tuner.unload()

    return metrics
