"""
PayFlow — Hardware-Aware Configuration
=======================================
All VRAM budgets, batch sizes, and concurrency limits are derived from the
RTX 4070 8 GB GDDR6 ceiling. Modify GPU_VRAM_TOTAL_MB if porting to another card.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# ── Project Paths ──────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
LOG_DIR = PROJECT_ROOT / "logs"

# ── Hardware Profile ───────────────────────────────────────────────────────────

GPU_VRAM_TOTAL_MB: int = 8192  # RTX 4070 8 GB


@dataclass(frozen=True)
class VRAMBudget:
    """
    Static VRAM partitioning strategy.

    The Qwen-3.5-9B model at Q4_K_M quantization consumes ~5.5 GB when loaded
    by Ollama. Remaining ~2.5 GB is shared across ML training / GNN inference.

    Key insight: Ollama lazily loads models and can be configured to unload
    after idle timeout (`OLLAMA_KEEP_ALIVE`). We exploit this by defining two
    mutually exclusive execution modes:

        MODE A — "Analysis"  : LLM unloaded, full 8 GB for ML + GNN
        MODE B — "Assistant" : ML/GNN tensors freed, LLM occupies ~5.5 GB

    The orchestrator switches between modes by managing Ollama keepalive and
    calling `torch.cuda.empty_cache()` at transition boundaries.
    """

    total_mb: int = GPU_VRAM_TOTAL_MB

    # Mode A: Analysis (LLM unloaded)
    xgboost_max_mb: int = 1024       # histogram bins + prediction buffer
    gnn_mini_batch_mb: int = 2048    # NeighborLoader batch ceiling
    pytorch_overhead_mb: int = 512   # CUDA context + kernels
    analysis_reserve_mb: int = 4608  # sum of above: safety margin to 8 GB

    # Mode B: Assistant (LLM loaded)
    llm_model_mb: int = 5200        # Qwen-3.5-9B Q4_K_M measured footprint
    llm_kv_cache_mb: int = 1475     # KV cache at q8_0 quant, 16K context
    llm_cuda_overhead_mb: int = 400 # CUDA context + compute buffers
    assistant_reserve_mb: int = 7075

    def validate(self) -> None:
        assert self.analysis_reserve_mb <= self.total_mb, (
            f"Analysis mode exceeds VRAM: {self.analysis_reserve_mb} > {self.total_mb}"
        )
        assert self.assistant_reserve_mb <= self.total_mb, (
            f"Assistant mode exceeds VRAM: {self.assistant_reserve_mb} > {self.total_mb}"
        )


VRAM = VRAMBudget()
VRAM.validate()


# ── Model Hyperparameters (VRAM-constrained) ──────────────────────────────────

@dataclass(frozen=True)
class XGBoostConfig:
    device: str = "cuda"
    tree_method: str = "hist"
    max_bin: int = 128            # lower bins = less VRAM; 128 is the sweet spot
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    max_depth: int = 8
    n_estimators: int = 500
    early_stopping_rounds: int = 25
    eval_metric: str = "aucpr"    # precision-recall AUC for imbalanced fraud data


@dataclass(frozen=True)
class GNNConfig:
    hidden_channels: int = 128
    num_layers: int = 3           # 3-hop neighborhood captures layering chains
    heads: int = 4                # multi-head attention for GAT layers
    dropout: float = 0.3
    # NeighborLoader sampling — controls VRAM per mini-batch
    num_neighbors: list[int] = field(default_factory=lambda: [15, 10, 5])
    batch_size: int = 1024        # target nodes per mini-batch
    num_workers: int = 4          # CPU workers for subgraph sampling


@dataclass(frozen=True)
class OllamaConfig:
    model: str = "qwen3.5:9b-q4_K_M"
    custom_model: str = "payflow-qwen"  # built via scripts/deploy_ollama.sh
    base_url: str = "http://localhost:11434"
    keep_alive: str = "-1"        # permanent residency — never unload from VRAM
    temperature: float = 0.3      # low temp for deterministic fraud analysis
    num_ctx: int = 16384          # context window (tokens) — capped for 8 GB VRAM
    top_p: float = 0.9
    num_batch: int = 256          # smaller prefill batches to limit VRAM spikes
    kv_cache_type: str = "q8_0"   # halves KV cache memory (~2,950 → ~1,475 MB)


# ── Fraud Detection Thresholds ────────────────────────────────────────────────

@dataclass(frozen=True)
class FraudThresholds:
    # Structuring: INR amounts just below CTR reporting limit
    ctr_threshold_inr: int = 10_00_000          # ₹10 lakh
    structuring_band_inr: tuple[int, int] = (8_00_000, 9_99_999)

    # Dormant account: days inactive before flagging reactivation
    dormant_days: int = 180

    # Layering: max minutes between hops to count as "rapid"
    rapid_layering_window_minutes: int = 30
    min_layering_hops: int = 3

    # Round-tripping: cycle detection depth limit
    max_cycle_length: int = 10

    # Risk scoring
    high_risk_score: float = 0.85
    medium_risk_score: float = 0.60


# ── Audit Ledger ─────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class LedgerConfig:
    """Tamper-evident audit ledger configuration."""
    db_path: Path = DATA_DIR / "ledger2.db"
    key_dir: Path = ARTIFACTS_DIR / "ledger"
    checkpoint_interval: int = 100          # Merkle checkpoint every N blocks
    enable_signing: bool = True             # Ed25519 signatures (disable for tests)


# ── Singleton Instances ───────────────────────────────────────────────────────

XGBOOST_CFG = XGBoostConfig()
GNN_CFG = GNNConfig()
OLLAMA_CFG = OllamaConfig()
FRAUD_THRESHOLDS = FraudThresholds()
LEDGER_CFG = LedgerConfig()


# ── Circuit Breaker ──────────────────────────────────────────────────────────

@dataclass(frozen=True)
class CircuitBreakerConfig:
    """Smart contract circuit breaker configuration."""
    consensus_threshold: float = 0.80       # weighted score to trigger freeze
    ml_weight: float = 0.35                 # weight for ML risk_score
    gnn_weight: float = 0.35               # weight for GNN risk_score
    graph_evidence_weight: float = 0.30     # weight for structural evidence
    cooldown_seconds: int = 300             # 5 min cooldown per node
    freeze_ttl_seconds: int = 3600          # 1 hour auto-unfreeze
    max_frozen_nodes: int = 10_000          # cap to prevent runaway freezes
    require_investigation: bool = True      # wait for GNN result before triggering


CIRCUIT_BREAKER_CFG = CircuitBreakerConfig()


# ── Investigator Agent ──────────────────────────────────────────────────────

@dataclass(frozen=True)
class InvestigatorAgentConfig:
    """LangGraph investigator agent configuration."""
    max_iterations: int = 5              # max think-act-observe loops
    thinking_temperature: float = 0.3    # low temp for deterministic CoT reasoning
    verdict_temperature: float = 0.1     # even lower for final verdict
    max_thinking_tokens: int = 4096      # token budget per thinking step
    max_verdict_tokens: int = 2048       # token budget for final verdict
    tool_timeout_seconds: int = 30       # per-tool execution timeout
    enable_cot_trace: bool = True        # log full CoT trace for audit


INVESTIGATOR_CFG = InvestigatorAgentConfig()


# ── Unstructured NLU Sub-Agent ───────────────────────────────────────────

@dataclass(frozen=True)
class UnstructuredAgentConfig:
    """NLU sub-agent configuration for unstructured data analysis."""
    analysis_temperature: float = 0.2    # lower than main agent for precision
    max_findings_to_report: int = 15     # cap findings per analysis
    risk_modifier_ceiling: float = 0.30  # max positive risk adjustment
    risk_modifier_floor: float = -0.05   # risk reduction when no findings
    se_weight: float = 0.40              # social engineering weight
    la_weight: float = 0.35              # linguistic anomaly weight
    da_weight: float = 0.25              # device anomaly weight


UNSTRUCTURED_AGENT_CFG = UnstructuredAgentConfig()


# ── Fine-Tuning (QLoRA + GRPO) ─────────────────────────────────────────────

@dataclass(frozen=True)
class FineTuningConfig:
    """
    QLoRA + GRPO fine-tuning configuration for 8 GB VRAM.

    The Qwen 3.5 9B model is loaded in 4-bit NormalFloat (NF4) quantization
    via bitsandbytes. LoRA adapters target the attention projection matrices
    (q_proj, k_proj, v_proj, o_proj) with a small rank to keep trainable
    parameters under ~50 MB. Gradient checkpointing offloads activation
    memory to CPU, trading ~30% wall-clock time for ~60% peak VRAM reduction.

    GRPO (Group Relative Policy Optimization) uses grouped completions
    per prompt and ranks them by a reward signal derived from fraud-typology
    correctness, evidence citation quality, and verdict calibration.
    """

    # ── QLoRA Adapter ──────────────────────────────────────────────────────
    base_model: str = "Qwen/Qwen3.5-9B"
    lora_r: int = 16                      # LoRA rank (16 strikes rank vs. VRAM)
    lora_alpha: int = 32                  # scaling factor = alpha / r = 2.0
    lora_dropout: float = 0.05
    target_modules: tuple[str, ...] = (
        "q_proj", "k_proj", "v_proj", "o_proj",
    )
    quant_type: str = "nf4"               # NormalFloat 4-bit (bitsandbytes)
    use_double_quant: bool = True         # nested quantization saves ~0.4 GB

    # ── Training Loop ──────────────────────────────────────────────────────
    learning_rate: float = 2e-4
    num_epochs: int = 3
    per_device_batch_size: int = 1        # micro-batch=1 for 8 GB ceiling
    gradient_accumulation_steps: int = 8  # effective batch = 1 × 8 = 8
    max_seq_length: int = 2048            # investigation context window
    warmup_ratio: float = 0.03
    weight_decay: float = 0.01
    lr_scheduler_type: str = "cosine"
    gradient_checkpointing: bool = True   # mandatory on 8 GB
    fp16: bool = False                    # bf16 preferred on Ada Lovelace
    bf16: bool = True

    # ── GRPO ───────────────────────────────────────────────────────────────
    grpo_group_size: int = 4              # completions per prompt
    grpo_temperature: float = 0.7         # sampling temp for diversity
    grpo_beta: float = 0.1               # KL penalty coefficient
    grpo_max_new_tokens: int = 1024

    # ── Memory Management ──────────────────────────────────────────────────
    vram_safety_margin_mb: int = 512      # reserve for CUDA context + spikes
    flush_cache_every_n_steps: int = 10   # periodic torch.cuda.empty_cache()

    # ── Paths ──────────────────────────────────────────────────────────────
    output_dir: Path = ARTIFACTS_DIR / "models" / "qlora_adapters"
    dataset_dir: Path = DATA_DIR / "finetune"


FINETUNE_CFG = FineTuningConfig()


# ── Real-Time CDC Streaming ──────────────────────────────────────────────────

@dataclass(frozen=True)
class StreamingConfig:
    """
    Real-time CDC streaming pipeline configuration.

    Controls the BankingDatabase CDC trigger system, CDCReader polling,
    StreamingConsumer batching, and BankingEndpointSimulator parameters.
    """
    db_path: Path = DATA_DIR / "banking.db"
    cdc_poll_interval_ms: int = 10            # base poll interval (busy)
    cdc_max_poll_interval_ms: int = 100       # ceiling when idle
    cdc_batch_size: int = 256                 # events per micro-batch
    cdc_batch_timeout_sec: float = 0.5        # flush timeout
    cdc_log_retention: int = 10_000           # keep last N processed
    endpoint_tps: float = 50.0                # transactions per second
    endpoint_burst_multiplier: float = 3.0    # peak burst factor
    endpoint_fraud_ratio: float = 0.05        # 5% injected fraud
    consumer_concurrency: int = 3             # parallel consumer invocations
    enable_validation: bool = True            # CRC32 validation


STREAMING_CFG = StreamingConfig()


# ── Human-in-the-Loop (HITL) Escalation ──────────────────────────────────────

@dataclass(frozen=True)
class HITLConfig:
    """
    Human-in-the-Loop escalation configuration.

    Controls confidence thresholds per fraud typology, human analyst API
    endpoint, graph context packaging, and escalation lifecycle.
    Complex typologies (LAYERING, ROUND_TRIPPING, STRUCTURING) require
    higher confidence for autonomous verdicts; below-threshold cases
    are escalated to a human analyst.
    """
    # Global default confidence threshold for autonomous verdicts
    default_confidence_threshold: float = 0.75
    # Per-typology overrides (higher for complex ML typologies)
    typology_thresholds: dict = field(default_factory=lambda: {
        "LAYERING": 0.80,
        "ROUND_TRIPPING": 0.80,
        "STRUCTURING": 0.80,
        "DORMANT_ACTIVATION": 0.65,
        "PROFILE_MISMATCH": 0.65,
    })
    # Human analyst API endpoint
    analyst_endpoint_url: str = "http://localhost:8000/api/v1/analyst/escalation"
    analyst_timeout_seconds: int = 10
    analyst_max_retries: int = 2
    # Graph context packaging
    graph_context_k_hops: int = 3
    max_evidence_items: int = 20
    # Escalation tracking
    max_pending_escalations: int = 100
    escalation_ttl_seconds: int = 3600  # 1 hour before auto-resolve


HITL_CFG = HITLConfig()


# ── Agent-Triggered Circuit Breaker ──────────────────────────────────────────

@dataclass(frozen=True)
class AgentBreakerConfig:
    """
    Configuration for agent-triggered blockchain circuit breaker listeners.

    Controls the confidence thresholds that determine when an agent verdict
    triggers autonomous defensive actions (node freeze, device ban, routing
    pause) on the local blockchain.  Two tiers:

    - CRITICAL (>= critical_confidence_threshold): full freeze + device ban
      + routing pause + immutable ledger anchoring with ZKP proof.
    - HIGH_SUSPICION (>= high_suspicion_threshold): routing pause only.
    """
    # Tier thresholds
    critical_confidence_threshold: float = 0.95
    high_suspicion_threshold: float = 0.80
    # Device fingerprint ban
    device_ban_ttl_seconds: int = 7200          # 2 hours
    max_banned_devices: int = 5000
    # Routing pause
    routing_pause_ttl_seconds: int = 3600       # 1 hour
    # Mule cascade freeze
    enable_mule_cascade_freeze: bool = True
    max_cascade_nodes: int = 10


AGENT_BREAKER_CFG = AgentBreakerConfig()


# ── Dashboard Configuration ──────────────────────────────────────────────────

@dataclass(frozen=True)
class DashboardConfig:
    """
    Real-time web dashboard configuration.

    Controls the embedded FastAPI + SSE dashboard that visualises the
    transaction graph topology, agent Chain-of-Thought reasoning, live
    risk scores, and circuit breaker status in a browser.
    """
    enable_dashboard: bool = True
    host: str = "127.0.0.1"
    port: int = 8000
    max_graph_nodes_display: int = 500
    max_graph_edges_display: int = 1000
    sse_keepalive_seconds: int = 15
    sse_queue_max_size: int = 256
    system_telemetry_interval_sec: float = 1.0


DASHBOARD_CFG = DashboardConfig()


# ── GPU Concurrency Management ───────────────────────────────────────────────

@dataclass(frozen=True)
class GPUConcurrencyConfig:
    """
    Advanced GPU concurrency configuration for cooperative VRAM sharing.

    Replaces the old binary exclusive-mode model with a priority-based
    arbitration system that keeps the LLM permanently resident in VRAM
    while dynamically scaling its KV cache and offloading GNN to CPU
    when VRAM pressure is critical.
    """
    # VRAM pressure thresholds (MB) — hysteresis to prevent oscillation
    vram_critical_threshold_mb: float = 7800.0   # activate CPU offload
    vram_high_threshold_mb: float = 7500.0       # reduce KV cache
    vram_normal_threshold_mb: float = 6500.0     # clear pressure (resume)

    # Dynamic context window scaling for LLM KV cache
    num_ctx_full: int = 16384        # normal pressure: full context
    num_ctx_medium: int = 8192       # high pressure: halved context
    num_ctx_minimal: int = 4096      # critical: minimal context

    # Concurrency limits
    max_concurrent_gnn: int = 1      # max simultaneous GNN inferences
    max_concurrent_ml: int = 1       # max simultaneous ML training ops

    # Timeouts
    gpu_acquire_timeout_sec: float = 30.0  # max wait for GPU semaphore


GPU_CONCURRENCY_CFG = GPUConcurrencyConfig()


# ── Threat Simulation Engine ─────────────────────────────────────────────────

@dataclass(frozen=True)
class SimulationConfig:
    """Threat Simulation Engine configuration for live demo attack scenarios."""
    # Drip-feed pacing
    default_event_interval_sec: float = 0.5   # seconds between events
    burst_interval_sec: float = 0.05          # fast-paced attacks (phishing)

    # UPI Mule Network defaults
    upi_mule_count: int = 6
    upi_total_amount_inr: int = 15_00_000     # 15 lakh

    # Circular Laundering defaults
    circular_shell_count: int = 5
    circular_hop_amount_inr: int = 25_00_000  # 25 lakh

    # Velocity Phishing defaults
    phishing_credential_attempts: int = 8
    phishing_unauthorized_txns: int = 5
    phishing_geo_spread: int = 4

    # Engine limits
    max_concurrent_attacks: int = 3
    max_events_per_attack: int = 200


SIMULATION_CFG = SimulationConfig()


# ── Environment Overrides ─────────────────────────────────────────────────────

if os.getenv("PAYFLOW_CPU_ONLY"):
    # Graceful fallback for machines without NVIDIA GPU
    XGBOOST_CFG = XGBoostConfig(device="cpu", tree_method="hist")
