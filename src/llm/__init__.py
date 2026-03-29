"""PayFlow — LLM Orchestration Package."""

from src.llm.agent import AgentMetrics, AgentState, InvestigatorAgent, VerdictPayload
from src.llm.health_check import (
    GPUDeviceInfo,
    HealthCheckResult,
    check_vram_for_analysis,
    check_vram_for_llm,
    print_gpu_diagnostic,
    query_gpu,
)
from src.llm.orchestrator import LLMResponse, PayFlowLLM, VRAMInsufficientError
from src.llm.prompts import (
    COT_ACTIVATION_PREFIX,
    INVESTIGATOR_SYSTEM_PROMPT,
    VERDICT_SCHEMA,
    build_cot_prompt,
    build_investigation_prompt,
    build_verdict_prompt,
)
from src.llm.tools import TOOL_SCHEMAS, ToolCall, ToolExecutor, ToolResult
from src.llm.unstructured_agent import UnstructuredAnalysisAgent
from src.llm.unstructured_models import (
    FindingConfidence,
    SemanticAnomalyType,
    SemanticFinding,
    UnstructuredAgentMetrics,
    UnstructuredAnalysisResult,
    UnstructuredPayload,
)
from src.llm.unstructured_prompts import (
    build_consensus_injection_prompt,
    build_unstructured_analysis_prompt,
)
from src.llm.finetuning import (
    FineTuneMetrics,
    GRPOTrainer,
    QLoRAFineTuner,
    RewardSignal,
    aggressive_memory_clear,
    compute_fraud_reward,
    prepare_investigation_dataset,
    run_finetuning_pipeline,
)

__all__ = [
    # Orchestrator
    "PayFlowLLM",
    "LLMResponse",
    "VRAMInsufficientError",
    # Health Check
    "query_gpu",
    "GPUDeviceInfo",
    "HealthCheckResult",
    "check_vram_for_llm",
    "check_vram_for_analysis",
    "print_gpu_diagnostic",
    # Investigator Agent
    "InvestigatorAgent",
    "VerdictPayload",
    "AgentState",
    "AgentMetrics",
    # Tools
    "ToolExecutor",
    "ToolCall",
    "ToolResult",
    "TOOL_SCHEMAS",
    # Prompts
    "INVESTIGATOR_SYSTEM_PROMPT",
    "COT_ACTIVATION_PREFIX",
    "VERDICT_SCHEMA",
    "build_investigation_prompt",
    "build_cot_prompt",
    "build_verdict_prompt",
    # NLU Sub-Agent (Phase 9B)
    "UnstructuredAnalysisAgent",
    "UnstructuredPayload",
    "UnstructuredAnalysisResult",
    "SemanticFinding",
    "SemanticAnomalyType",
    "FindingConfidence",
    "UnstructuredAgentMetrics",
    "build_unstructured_analysis_prompt",
    "build_consensus_injection_prompt",
    # Fine-Tuning Pipeline (Phase 10)
    "QLoRAFineTuner",
    "GRPOTrainer",
    "FineTuneMetrics",
    "RewardSignal",
    "compute_fraud_reward",
    "aggressive_memory_clear",
    "prepare_investigation_dataset",
    "run_finetuning_pipeline",
]
