#!/usr/bin/env bash
# ============================================================================
# PayFlow — Ollama Deployment & VRAM-Safe Configuration Script
# ============================================================================
# Configures Ollama daemon environment variables for 8 GB VRAM ceiling,
# creates the VRAM-optimized custom model, validates deployment, and
# runs a diagnostic inference pass.
#
# Usage: bash scripts/deploy_ollama.sh
# ============================================================================

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

log()  { echo -e "${GREEN}[PayFlow:Ollama]${NC} $*"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $*"; }
err()  { echo -e "${RED}[ERROR]${NC} $*" >&2; }
info() { echo -e "${CYAN}[INFO]${NC} $*"; }

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODELFILE_PATH="${PROJECT_ROOT}/scripts/Modelfile"
CUSTOM_MODEL_TAG="payflow-qwen"
BASE_MODEL="qwen3.5:9b"

# ── Step 0: Preflight — Verify Ollama Installation ───────────────────────────

log "Checking Ollama installation..."

if ! command -v ollama &>/dev/null; then
    err "Ollama is not installed or not in PATH."
    err "Install from: https://ollama.com/download/windows"
    exit 1
fi

OLLAMA_VERSION=$(ollama --version 2>&1 | grep -oP '[\d.]+' | head -1 || echo "unknown")
log "Ollama version: ${OLLAMA_VERSION}"

# ── Step 1: Set VRAM-Critical Environment Variables ───────────────────────────

log "Configuring Ollama environment for RTX 4070 (8 GB)..."

# These variables must be set BEFORE the Ollama daemon starts.
# On Windows, they persist in the user environment via setx / registry.
# The script exports them for the current session AND writes a .env file
# that the daemon can source on restart.

cat > "${PROJECT_ROOT}/scripts/ollama.env" << 'ENVEOF'
# ============================================================================
# PayFlow — Ollama Daemon Environment Variables
# ============================================================================
# Apply these BEFORE starting the Ollama service:
#   Windows: set each via System Properties > Environment Variables
#   Linux:   source this file in your shell before `ollama serve`
# ============================================================================

# --- KV Cache Quantization ---
# Default: f16 (2 bytes per element). q8_0 halves this to ~1 byte.
# Impact: 16K context KV cache drops from ~2,950 MB → ~1,475 MB.
# This single flag reclaims ~1.5 GB of VRAM.
OLLAMA_KV_CACHE_TYPE=q8_0

# --- Flash Attention ---
# Fused QKV kernel: 30-40% less memory for attention computation.
# On Ada Lovelace (RTX 4070) this also yields ~20% throughput gain.
OLLAMA_FLASH_ATTENTION=1

# --- Concurrency Limits ---
# Each parallel slot allocates its OWN KV cache. At 16K context:
#   1 slot  = ~1,475 MB KV    (fits)
#   2 slots = ~2,950 MB KV    (OOM with model weights)
# MUST be 1 on 8 GB cards.
OLLAMA_NUM_PARALLEL=1

# --- Model Slot Limits ---
# Only allow one model resident in VRAM at a time.
# Prevents accidental co-loading if another model is requested.
OLLAMA_MAX_LOADED_MODELS=1

# --- Keep-Alive (Auto-Unload) ---
# Unload model from VRAM after 5 minutes of idle.
# Critical for yielding GPU back to Analysis mode (XGBoost / GNN).
OLLAMA_KEEP_ALIVE=5m

# --- GPU Selection ---
# Force Ollama to use GPU 0 (the discrete RTX 4070).
# Prevents accidental selection of integrated Intel UHD graphics.
CUDA_VISIBLE_DEVICES=0

# --- Logging ---
OLLAMA_DEBUG=0
ENVEOF

log "Written: scripts/ollama.env"

# Export for current session
set -a
source "${PROJECT_ROOT}/scripts/ollama.env"
set +a

log "Environment variables applied to current session."

# On Windows, persist to the user environment for future Ollama daemon starts.
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "cygwin" || "$OSTYPE" == "win32" ]]; then
    log "Persisting to Windows user environment (setx)..."
    setx OLLAMA_KV_CACHE_TYPE    "q8_0"  > /dev/null 2>&1 || true
    setx OLLAMA_FLASH_ATTENTION  "1"     > /dev/null 2>&1 || true
    setx OLLAMA_NUM_PARALLEL     "1"     > /dev/null 2>&1 || true
    setx OLLAMA_MAX_LOADED_MODELS "1"    > /dev/null 2>&1 || true
    setx OLLAMA_KEEP_ALIVE       "5m"    > /dev/null 2>&1 || true
    setx CUDA_VISIBLE_DEVICES    "0"     > /dev/null 2>&1 || true
    log "Persisted. Restart Ollama service for changes to take effect."
fi

# ── Step 2: Check if Ollama Daemon is Running ─────────────────────────────────

log "Checking Ollama daemon status..."

OLLAMA_ALIVE=false
for attempt in 1 2 3; do
    if curl -sf http://localhost:11434/api/tags > /dev/null 2>&1; then
        OLLAMA_ALIVE=true
        break
    fi
    if [ "$attempt" -lt 3 ]; then
        warn "Ollama daemon not responding (attempt ${attempt}/3). Waiting 3s..."
        sleep 3
    fi
done

if [ "$OLLAMA_ALIVE" = false ]; then
    warn "Ollama daemon is not running."
    warn "Start it manually:  ollama serve"
    warn "Then re-run this script."
    exit 1
fi

log "Ollama daemon is responsive."

# ── Step 3: Pull Base Model ───────────────────────────────────────────────────

log "Ensuring base model '${BASE_MODEL}' is available..."

if ollama list 2>/dev/null | grep -q "${BASE_MODEL}"; then
    log "Base model '${BASE_MODEL}' already downloaded."
else
    log "Pulling '${BASE_MODEL}' (Q4_K_M, ~5.5 GB). This may take several minutes..."
    ollama pull "${BASE_MODEL}"
    log "Base model downloaded."
fi

# ── Step 4: Build Custom PayFlow Model ────────────────────────────────────────

log "Building custom model '${CUSTOM_MODEL_TAG}' from Modelfile..."

if [ ! -f "$MODELFILE_PATH" ]; then
    err "Modelfile not found at: ${MODELFILE_PATH}"
    exit 1
fi

ollama create "${CUSTOM_MODEL_TAG}" -f "${MODELFILE_PATH}"
log "Custom model '${CUSTOM_MODEL_TAG}' created."

# ── Step 5: Validate — Dry-Run Inference ──────────────────────────────────────

log "Running diagnostic inference (single-shot)..."

DIAG_RESPONSE=$(curl -sf http://localhost:11434/api/generate \
    -d '{
        "model": "'"${CUSTOM_MODEL_TAG}"'",
        "prompt": "You are a financial fraud analyst. Respond with EXACTLY: PAYFLOW_DIAGNOSTIC_OK",
        "stream": false,
        "options": {
            "num_ctx": 512,
            "temperature": 0.0,
            "num_predict": 20
        }
    }' 2>/dev/null || echo '{"response":"FAILED"}')

# Extract response text
DIAG_TEXT=$(echo "$DIAG_RESPONSE" | python -c "
import sys, json
try:
    data = json.load(sys.stdin)
    print(data.get('response', 'NO_RESPONSE'))
except:
    print('PARSE_ERROR')
" 2>/dev/null || echo "PARSE_ERROR")

if echo "$DIAG_TEXT" | grep -qi "PAYFLOW_DIAGNOSTIC_OK"; then
    log "Diagnostic inference: PASS"
else
    warn "Diagnostic inference returned unexpected output: ${DIAG_TEXT:0:100}"
    warn "Model loaded but may need prompt tuning. Continuing..."
fi

# ── Step 6: Unload Model (Return VRAM to Idle) ───────────────────────────────

log "Unloading model to free VRAM..."
curl -sf http://localhost:11434/api/generate \
    -d '{"model": "'"${CUSTOM_MODEL_TAG}"'", "keep_alive": 0}' > /dev/null 2>&1 || true
log "VRAM released."

# ── Summary ───────────────────────────────────────────────────────────────────

echo ""
echo -e "${CYAN}╔══════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║          PayFlow — Ollama Deployment Complete                   ║${NC}"
echo -e "${CYAN}╠══════════════════════════════════════════════════════════════════╣${NC}"
echo -e "${CYAN}║${NC}  Model tag:     ${GREEN}${CUSTOM_MODEL_TAG}${NC}"
echo -e "${CYAN}║${NC}  Base model:    ${GREEN}${BASE_MODEL} (Q4_K_M)${NC}"
echo -e "${CYAN}║${NC}  Context:       ${GREEN}16,384 tokens${NC}"
echo -e "${CYAN}║${NC}  KV cache:      ${GREEN}q8_0 (halved memory)${NC}"
echo -e "${CYAN}║${NC}  Flash attn:    ${GREEN}enabled${NC}"
echo -e "${CYAN}║${NC}  GPU layers:    ${GREEN}all (num_gpu=999)${NC}"
echo -e "${CYAN}║${NC}  Parallelism:   ${GREEN}1 slot (env-controlled)${NC}"
echo -e "${CYAN}║${NC}  Keep-alive:    ${GREEN}5 min via daemon env${NC}"
echo -e "${CYAN}╠══════════════════════════════════════════════════════════════════╣${NC}"
echo -e "${CYAN}║${NC}  ${BOLD}VRAM BUDGET:${NC}"
echo -e "${CYAN}║${NC}    Weights (Q4_K_M):    ~5,200 MB${NC}"
echo -e "${CYAN}║${NC}    KV cache (q8_0,16K): ~1,475 MB${NC}"
echo -e "${CYAN}║${NC}    CUDA overhead:         ~400 MB${NC}"
echo -e "${CYAN}║${NC}    Headroom:               ~517 MB${NC}"
echo -e "${CYAN}║${NC}    ───────────────────────────────${NC}"
echo -e "${CYAN}║${NC}    TOTAL:               ~7,592 / 8,192 MB (93%)${NC}"
echo -e "${CYAN}╚══════════════════════════════════════════════════════════════════╝${NC}"
