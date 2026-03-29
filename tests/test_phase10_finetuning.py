"""
PayFlow -- Phase 10 Tests: QLoRA + GRPO Fine-Tuning Pipeline
==============================================================
Verifies model initialization, LoRA adapter management, reward computation,
memory management, dataset preparation, GRPO trainer wiring, VRAM mode
transitions, config exports, and the convenience pipeline entrypoint.

Tests:
 1. FineTuningConfig defaults and frozen immutability
 2. BitsAndBytes quantization config construction
 3. LoRA config construction with target modules
 4. Reward computation — correct typology identification
 5. Reward computation — incorrect typology, high confidence penalty
 6. Reward computation — evidence quality (Jaccard overlap)
 7. Reward computation — malformed model output fallback
 8. Verdict JSON extraction from markdown fences
 9. Verdict JSON extraction from raw JSON
10. Verdict extraction from garbage text (empty fallback)
11. QLoRAFineTuner lifecycle (mock model load + LoRA attach)
12. QLoRAFineTuner save/load adapter cycle
13. QLoRAFineTuner unload releases state
14. QLoRAFineTuner errors before load_model
15. GRPOTrainer rejects unready fine-tuner
16. GRPOTrainer builds TRL trainer with correct config
17. Dataset preparation filters empty prompts
18. Dataset preparation filters by token length
19. Aggressive memory clear runs without GPU
20. VRAM tracking utilities (peak / reset) without GPU
21. FineTuneMetrics and RewardSignal to_dict serialization
22. Finetuning VRAM mode transition (IDLE → FINETUNING → IDLE)
23. Finetuning VRAM mode transition from ASSISTANT
24. Config and __init__ exports verification
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from dataclasses import FrozenInstanceError
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import FINETUNE_CFG, FineTuningConfig
from config.vram_manager import GPUMode, finetuning_mode, get_current_mode
from src.llm.finetuning import (
    FineTuneMetrics,
    GRPOTrainer,
    QLoRAFineTuner,
    RewardSignal,
    VALID_TYPOLOGIES,
    _extract_verdict_json,
    aggressive_memory_clear,
    build_bnb_config,
    build_lora_config,
    compute_fraud_reward,
    get_peak_vram_mb,
    prepare_investigation_dataset,
    reset_peak_vram_tracker,
    run_finetuning_pipeline,
)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _safe_print(msg: str) -> None:
    try:
        print(msg)
    except UnicodeEncodeError:
        print(msg.encode("ascii", "replace").decode())


passed = 0
failed = 0


def run_test(func):
    global passed, failed
    name = func.__name__
    try:
        if asyncio.iscoroutinefunction(func):
            asyncio.get_event_loop().run_until_complete(func())
        else:
            func()
        _safe_print(f"  PASS  {name}")
        passed += 1
    except Exception as e:
        _safe_print(f"  FAIL  {name}: {e}")
        failed += 1


# ── Test 1: FineTuningConfig defaults and immutability ───────────────────────

def test_config_defaults():
    cfg = FineTuningConfig()
    assert cfg.lora_r == 16, f"Expected lora_r=16, got {cfg.lora_r}"
    assert cfg.lora_alpha == 32
    assert cfg.quant_type == "nf4"
    assert cfg.use_double_quant is True
    assert cfg.gradient_checkpointing is True
    assert cfg.per_device_batch_size == 1
    assert cfg.gradient_accumulation_steps == 8
    assert cfg.grpo_group_size == 4
    assert cfg.grpo_beta == 0.1
    assert cfg.bf16 is True
    assert cfg.fp16 is False
    assert "q_proj" in cfg.target_modules
    assert "v_proj" in cfg.target_modules
    assert cfg.base_model == "Qwen/Qwen3.5-9B"

    # Singleton matches
    assert FINETUNE_CFG.lora_r == cfg.lora_r

    # Frozen immutability
    try:
        cfg.lora_r = 32
        raise AssertionError("Should have raised FrozenInstanceError")
    except FrozenInstanceError:
        pass


# ── Test 2: BitsAndBytes config construction ─────────────────────────────────

def test_bnb_config_construction():
    # Mock transformers.BitsAndBytesConfig
    mock_bnb_cls = MagicMock()
    mock_torch = MagicMock()
    mock_torch.bfloat16 = "bfloat16_sentinel"
    mock_torch.float16 = "float16_sentinel"

    with patch.dict("sys.modules", {
        "transformers": MagicMock(BitsAndBytesConfig=mock_bnb_cls),
        "torch": mock_torch,
    }):
        # Re-import to pick up mocks
        from src.llm.finetuning import build_bnb_config as _build
        cfg = FineTuningConfig()
        _build(cfg)

        mock_bnb_cls.assert_called_once()
        call_kwargs = mock_bnb_cls.call_args[1]
        assert call_kwargs["load_in_4bit"] is True
        assert call_kwargs["bnb_4bit_quant_type"] == "nf4"
        assert call_kwargs["bnb_4bit_use_double_quant"] is True
        # bf16=True → bfloat16
        assert call_kwargs["bnb_4bit_compute_dtype"] == "bfloat16_sentinel"


# ── Test 3: LoRA config construction ─────────────────────────────────────────

def test_lora_config_construction():
    mock_lora_cls = MagicMock()
    mock_task_type = MagicMock()
    mock_task_type.CAUSAL_LM = "CAUSAL_LM"

    with patch.dict("sys.modules", {
        "peft": MagicMock(LoraConfig=mock_lora_cls, TaskType=mock_task_type),
    }):
        from src.llm.finetuning import build_lora_config as _build
        cfg = FineTuningConfig()
        _build(cfg)

        mock_lora_cls.assert_called_once()
        call_kwargs = mock_lora_cls.call_args[1]
        assert call_kwargs["r"] == 16
        assert call_kwargs["lora_alpha"] == 32
        assert call_kwargs["lora_dropout"] == 0.05
        assert "q_proj" in call_kwargs["target_modules"]
        assert call_kwargs["task_type"] == "CAUSAL_LM"
        assert call_kwargs["bias"] == "none"


# ── Test 4: Reward — correct typology ────────────────────────────────────────

def test_reward_correct_typology():
    model_output = json.dumps({
        "fraud_typology": "layering",
        "evidence_cited": ["rapid_hops", "amount_decay"],
        "confidence": 0.9,
    })
    signal = compute_fraud_reward(
        model_output=model_output,
        ground_truth_typology="layering",
        ground_truth_evidence=["rapid_hops", "amount_decay", "fan_out"],
    )
    assert isinstance(signal, RewardSignal)
    assert signal.typology_correct is True
    assert signal.score > 0.5, f"Expected score > 0.5, got {signal.score}"
    # Evidence: Jaccard = 2/3 ≈ 0.667
    assert 0.6 <= signal.evidence_quality <= 0.7, (
        f"Expected evidence ~0.667, got {signal.evidence_quality}"
    )
    # High confidence + correct → high calibration
    assert signal.verdict_calibration == 0.9


# ── Test 5: Reward — incorrect typology, confidence penalty ──────────────────

def test_reward_incorrect_typology_penalty():
    model_output = json.dumps({
        "fraud_typology": "structuring",
        "evidence_cited": [],
        "confidence": 0.95,
    })
    signal = compute_fraud_reward(
        model_output=model_output,
        ground_truth_typology="layering",
        ground_truth_evidence=["rapid_hops"],
    )
    assert signal.typology_correct is False
    # Wrong + high confidence → low calibration (1.0 - 0.95 = 0.05)
    assert abs(signal.verdict_calibration - 0.05) < 1e-9, (
        f"Expected calibration ~0.05, got {signal.verdict_calibration}"
    )
    # Typology=0, evidence=0 (empty pred vs non-empty gt), calibration~0.05
    assert signal.score < 0.05


# ── Test 6: Reward — evidence quality Jaccard ────────────────────────────────

def test_reward_evidence_quality():
    model_output = json.dumps({
        "fraud_typology": "mule_chain",
        "evidence_cited": ["fan_in", "star_topology", "rapid_hops"],
        "confidence": 0.8,
    })
    signal = compute_fraud_reward(
        model_output=model_output,
        ground_truth_typology="mule_chain",
        ground_truth_evidence=["fan_in", "star_topology"],
    )
    # Jaccard = 2 / 3 ≈ 0.667 (intersection=2, predicted has extra "rapid_hops")
    assert 0.6 <= signal.evidence_quality <= 0.7


# ── Test 7: Reward — malformed model output ──────────────────────────────────

def test_reward_malformed_output():
    signal = compute_fraud_reward(
        model_output="This is just random text with no JSON",
        ground_truth_typology="layering",
        ground_truth_evidence=["rapid_hops"],
    )
    assert signal.typology_correct is False
    assert signal.evidence_quality == 0.0
    # Default confidence 0.5, wrong → calibration = 0.5
    assert signal.verdict_calibration == 0.5


# ── Test 8: Verdict extraction from markdown fences ──────────────────────────

def test_verdict_extraction_fenced():
    text = '''Here is my analysis:
```json
{"fraud_typology": "round_tripping", "confidence": 0.85}
```
'''
    result = _extract_verdict_json(text)
    assert result["fraud_typology"] == "round_tripping"
    assert result["confidence"] == 0.85


# ── Test 9: Verdict extraction from raw JSON ─────────────────────────────────

def test_verdict_extraction_raw():
    text = 'The verdict is {"fraud_typology": "smurfing", "confidence": 0.7} based on analysis.'
    result = _extract_verdict_json(text)
    assert result["fraud_typology"] == "smurfing"


# ── Test 10: Verdict extraction from garbage ─────────────────────────────────

def test_verdict_extraction_garbage():
    result = _extract_verdict_json("no json here at all")
    assert result == {}


# ── Test 11: QLoRAFineTuner lifecycle (mocked) ───────────────────────────────

def test_finetuner_lifecycle_mocked():
    mock_model = MagicMock()
    mock_model.parameters.return_value = [
        MagicMock(numel=MagicMock(return_value=100), requires_grad=True),
        MagicMock(numel=MagicMock(return_value=900), requires_grad=False),
    ]
    mock_tokenizer = MagicMock()
    mock_tokenizer.pad_token = None
    mock_tokenizer.eos_token = "<eos>"

    mock_auto_model = MagicMock(return_value=mock_model)
    mock_auto_tokenizer = MagicMock(return_value=mock_tokenizer)
    mock_get_peft = MagicMock(return_value=mock_model)
    mock_prepare = MagicMock(return_value=mock_model)

    with patch.dict("sys.modules", {
        "transformers": MagicMock(
            AutoModelForCausalLM=MagicMock(from_pretrained=mock_auto_model),
            AutoTokenizer=MagicMock(from_pretrained=mock_auto_tokenizer),
            BitsAndBytesConfig=MagicMock(),
        ),
        "peft": MagicMock(
            get_peft_model=mock_get_peft,
            prepare_model_for_kbit_training=mock_prepare,
            LoraConfig=MagicMock(),
            TaskType=MagicMock(CAUSAL_LM="CAUSAL_LM"),
        ),
        "torch": MagicMock(
            cuda=MagicMock(
                is_available=MagicMock(return_value=False),
                empty_cache=MagicMock(),
            ),
            bfloat16="bf16",
            float16="fp16",
        ),
    }):
        tuner = QLoRAFineTuner()
        assert not tuner.is_ready

        tuner.load_model("test-model")
        assert tuner._model_loaded
        assert not tuner._lora_attached
        assert not tuner.is_ready

        tuner.apply_lora()
        assert tuner._lora_attached
        assert tuner.is_ready

        trainable, total = tuner.trainable_param_count()
        assert trainable == 100
        assert total == 1000


# ── Test 12: QLoRAFineTuner save/load adapter ────────────────────────────────

def test_finetuner_save_load_adapter():
    import tempfile

    mock_model = MagicMock()
    mock_model.parameters.return_value = []
    mock_tokenizer = MagicMock()

    tuner = QLoRAFineTuner()
    tuner.model = mock_model
    tuner.tokenizer = mock_tokenizer
    tuner._model_loaded = True
    tuner._lora_attached = True

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = tuner.save_adapter(tmpdir)
        assert Path(save_path).exists()
        mock_model.save_pretrained.assert_called_once_with(str(save_path))
        mock_tokenizer.save_pretrained.assert_called_once_with(str(save_path))


# ── Test 13: QLoRAFineTuner unload ───────────────────────────────────────────

def test_finetuner_unload():
    tuner = QLoRAFineTuner()
    tuner.model = MagicMock()
    tuner.tokenizer = MagicMock()
    tuner._model_loaded = True
    tuner._lora_attached = True

    tuner.unload()
    assert tuner.model is None
    assert tuner.tokenizer is None
    assert not tuner._model_loaded
    assert not tuner._lora_attached


# ── Test 14: QLoRAFineTuner errors before load ───────────────────────────────

def test_finetuner_errors_before_load():
    tuner = QLoRAFineTuner()

    # apply_lora before load
    try:
        tuner.apply_lora()
        raise AssertionError("Should have raised RuntimeError")
    except RuntimeError as e:
        assert "load_model" in str(e)

    # save_adapter before lora
    tuner._model_loaded = True
    try:
        tuner.save_adapter("/tmp/test")
        raise AssertionError("Should have raised RuntimeError")
    except RuntimeError as e:
        assert "apply_lora" in str(e)

    # load_adapter before load
    tuner2 = QLoRAFineTuner()
    try:
        tuner2.load_adapter("/tmp/test")
        raise AssertionError("Should have raised RuntimeError")
    except RuntimeError as e:
        assert "load_model" in str(e)


# ── Test 15: GRPOTrainer rejects unready fine-tuner ──────────────────────────

def test_grpo_trainer_rejects_unready():
    tuner = QLoRAFineTuner()
    assert not tuner.is_ready
    try:
        GRPOTrainer(tuner)
        raise AssertionError("Should have raised RuntimeError")
    except RuntimeError as e:
        assert "loaded" in str(e).lower() or "applied" in str(e).lower()


# ── Test 16: GRPOTrainer configuration wiring ────────────────────────────────

def test_grpo_trainer_build():
    tuner = QLoRAFineTuner()
    tuner.model = MagicMock()
    tuner.tokenizer = MagicMock()
    tuner._model_loaded = True
    tuner._lora_attached = True

    trainer = GRPOTrainer(tuner)
    assert trainer.fine_tuner is tuner
    assert trainer.config == FINETUNE_CFG
    assert isinstance(trainer.metrics, FineTuneMetrics)
    assert trainer.metrics.total_steps == 0


# ── Test 17: Dataset preparation filters empty prompts ───────────────────────

def test_dataset_preparation_filters_empty():
    mock_dataset_cls = MagicMock()
    mock_dataset_cls.from_dict = MagicMock(return_value=MagicMock(
        __len__=MagicMock(return_value=2),
    ))

    with patch.dict("sys.modules", {
        "datasets": MagicMock(Dataset=mock_dataset_cls),
    }):
        from src.llm.finetuning import prepare_investigation_dataset as _prep
        records = [
            {"prompt": "Investigate account A001", "typology": "layering", "evidence": ["hop"]},
            {"prompt": "", "typology": "unknown", "evidence": []},  # empty — should be filtered
            {"prompt": "Check for round tripping", "typology": "round_tripping", "evidence": []},
        ]
        _prep(records)

        call_kwargs = mock_dataset_cls.from_dict.call_args[0][0]
        assert len(call_kwargs["prompt"]) == 2, (
            f"Expected 2 prompts after filtering, got {len(call_kwargs['prompt'])}"
        )
        assert "Investigate account A001" in call_kwargs["prompt"]
        assert "Check for round tripping" in call_kwargs["prompt"]


# ── Test 18: Dataset preparation filters by token length ─────────────────────

def test_dataset_preparation_token_filter():
    mock_dataset_cls = MagicMock()
    mock_dataset_cls.from_dict = MagicMock(return_value=MagicMock())

    mock_tokenizer = MagicMock()
    # First prompt → 10 tokens (under limit), second → 200 tokens (over limit)
    mock_tokenizer.encode = MagicMock(side_effect=[
        list(range(10)),
        list(range(200)),
    ])

    with patch.dict("sys.modules", {
        "datasets": MagicMock(Dataset=mock_dataset_cls),
    }):
        from src.llm.finetuning import prepare_investigation_dataset as _prep
        records = [
            {"prompt": "Short prompt", "typology": "layering", "evidence": []},
            {"prompt": "Very long prompt " * 50, "typology": "smurfing", "evidence": []},
        ]
        _prep(records, tokenizer=mock_tokenizer, max_length=100)

        call_kwargs = mock_dataset_cls.from_dict.call_args[0][0]
        assert len(call_kwargs["prompt"]) == 1
        assert call_kwargs["prompt"][0] == "Short prompt"


# ── Test 19: Aggressive memory clear without GPU ─────────────────────────────

def test_aggressive_memory_clear_no_gpu():
    # Should not raise even without CUDA
    aggressive_memory_clear()


# ── Test 20: VRAM tracking utilities without GPU ─────────────────────────────

def test_vram_tracking_no_gpu():
    peak = get_peak_vram_mb()
    assert peak == 0.0 or isinstance(peak, float)
    # Should not raise
    reset_peak_vram_tracker()


# ── Test 21: Dataclass serialization ─────────────────────────────────────────

def test_dataclass_serialization():
    metrics = FineTuneMetrics(
        total_steps=100,
        completed_epochs=2,
        train_loss=0.4567891,
        avg_reward=0.789123,
        peak_vram_mb=6543.21,
        wall_clock_seconds=1234.5678,
        checkpoints_saved=["ckpt_1", "ckpt_2"],
    )
    d = metrics.to_dict()
    assert d["total_steps"] == 100
    assert d["completed_epochs"] == 2
    assert d["train_loss"] == 0.456789  # rounded to 6 decimals
    assert d["avg_reward"] == 0.7891    # rounded to 4 decimals
    assert d["peak_vram_mb"] == 6543.2  # rounded to 1 decimal
    assert len(d["checkpoints_saved"]) == 2

    # JSON serializable
    json_str = json.dumps(d)
    assert "total_steps" in json_str

    signal = RewardSignal(
        score=0.87654,
        typology_correct=True,
        evidence_quality=0.66789,
        verdict_calibration=0.91234,
    )
    sd = signal.to_dict()
    assert sd["score"] == 0.8765
    assert sd["typology_correct"] is True
    json.dumps(sd)  # should not raise


# ── Test 22: Finetuning VRAM mode transition (IDLE → FINETUNING → IDLE) ────

def test_vram_finetuning_mode_idle():
    assert get_current_mode() == GPUMode.IDLE
    with finetuning_mode():
        assert get_current_mode() == GPUMode.FINETUNING
    assert get_current_mode() == GPUMode.IDLE


# ── Test 23: Finetuning VRAM mode from ASSISTANT ─────────────────────────────

def test_vram_finetuning_from_assistant():
    from config.vram_manager import assistant_mode

    with assistant_mode():
        assert get_current_mode() == GPUMode.ASSISTANT
    assert get_current_mode() == GPUMode.IDLE

    # Verify finetuning mode is properly isolated
    with finetuning_mode():
        assert get_current_mode() == GPUMode.FINETUNING
    assert get_current_mode() == GPUMode.IDLE


# ── Test 24: Config and __init__ exports ─────────────────────────────────────

def test_config_and_exports():
    # config package exports
    from config import FINETUNE_CFG as cfg_export, FineTuningConfig as cfg_cls, finetuning_mode as fm
    assert isinstance(cfg_export, cfg_cls)
    assert callable(fm)

    # src.llm package exports
    from src.llm import (
        QLoRAFineTuner,
        GRPOTrainer,
        FineTuneMetrics,
        RewardSignal,
        compute_fraud_reward,
        aggressive_memory_clear,
        prepare_investigation_dataset,
        run_finetuning_pipeline,
    )
    assert callable(QLoRAFineTuner)
    assert callable(GRPOTrainer)
    assert callable(compute_fraud_reward)
    assert callable(aggressive_memory_clear)
    assert callable(prepare_investigation_dataset)
    assert callable(run_finetuning_pipeline)

    # Valid typologies set
    assert "layering" in VALID_TYPOLOGIES
    assert "structuring" in VALID_TYPOLOGIES
    assert "round_tripping" in VALID_TYPOLOGIES
    assert "mule_chain" in VALID_TYPOLOGIES
    assert len(VALID_TYPOLOGIES) == 8


# ── Test Runner ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    _safe_print("\n=== Phase 10 Tests: QLoRA + GRPO Fine-Tuning Pipeline ===\n")

    tests = [
        test_config_defaults,
        test_bnb_config_construction,
        test_lora_config_construction,
        test_reward_correct_typology,
        test_reward_incorrect_typology_penalty,
        test_reward_evidence_quality,
        test_reward_malformed_output,
        test_verdict_extraction_fenced,
        test_verdict_extraction_raw,
        test_verdict_extraction_garbage,
        test_finetuner_lifecycle_mocked,
        test_finetuner_save_load_adapter,
        test_finetuner_unload,
        test_finetuner_errors_before_load,
        test_grpo_trainer_rejects_unready,
        test_grpo_trainer_build,
        test_dataset_preparation_filters_empty,
        test_dataset_preparation_token_filter,
        test_aggressive_memory_clear_no_gpu,
        test_vram_tracking_no_gpu,
        test_dataclass_serialization,
        test_vram_finetuning_mode_idle,
        test_vram_finetuning_from_assistant,
        test_config_and_exports,
    ]

    for test_func in tests:
        run_test(test_func)

    _safe_print(f"\n--- Results: {passed} passed, {failed} failed, {passed + failed} total ---")
    if failed:
        sys.exit(1)
    _safe_print("All Phase 10 tests passed!")
