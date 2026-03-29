#!/usr/bin/env bash
# ============================================================================
# PayFlow — Environment Initialization Script
# ============================================================================
# Target: Windows 11 / Lenovo Legion 7i / RTX 4070 8GB / CUDA 12.x
# Run from Git Bash or WSL: bash scripts/init_env.sh
# ============================================================================

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

log()  { echo -e "${GREEN}[PayFlow]${NC} $*"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $*"; }
err()  { echo -e "${RED}[ERROR]${NC} $*" >&2; }

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${PROJECT_ROOT}/.venv"
PYTHON_CMD="python"

# ── Step 0: Preflight Checks ─────────────────────────────────────────────────

log "Running preflight checks..."

# Python 3.11+
if ! command -v $PYTHON_CMD &>/dev/null; then
    PYTHON_CMD="python3"
fi
if ! command -v $PYTHON_CMD &>/dev/null; then
    err "Python not found. Install Python 3.11+ from https://python.org"
    exit 1
fi

PY_VERSION=$($PYTHON_CMD -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
PY_MAJOR=$($PYTHON_CMD -c "import sys; print(sys.version_info.major)")
PY_MINOR=$($PYTHON_CMD -c "import sys; print(sys.version_info.minor)")

if [ "$PY_MAJOR" -lt 3 ] || ([ "$PY_MAJOR" -eq 3 ] && [ "$PY_MINOR" -lt 11 ]); then
    err "Python ${PY_VERSION} found, but 3.11+ is required."
    exit 1
fi
log "Python ${PY_VERSION} ✓"

# NVIDIA GPU check
if command -v nvidia-smi &>/dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits 2>/dev/null | head -1)
    GPU_VRAM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1)
    DRIVER_VER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)
    log "GPU: ${GPU_NAME} (${GPU_VRAM} MB VRAM, driver ${DRIVER_VER}) ✓"
else
    warn "nvidia-smi not found. GPU features will be unavailable."
    warn "Set PAYFLOW_CPU_ONLY=1 to suppress GPU-related warnings at runtime."
fi

# Ollama check
if command -v ollama &>/dev/null; then
    OLLAMA_VER=$(ollama --version 2>/dev/null || echo "unknown")
    log "Ollama: ${OLLAMA_VER} ✓"
else
    warn "Ollama not found. Install from https://ollama.com for LLM features."
fi

# ── Step 1: Create Virtual Environment ────────────────────────────────────────

if [ -d "$VENV_DIR" ]; then
    log "Virtual environment already exists at ${VENV_DIR}"
else
    log "Creating virtual environment..."
    $PYTHON_CMD -m venv "$VENV_DIR"
    log "Created .venv ✓"
fi

# Activate
if [ -f "${VENV_DIR}/Scripts/activate" ]; then
    source "${VENV_DIR}/Scripts/activate"    # Windows Git Bash
elif [ -f "${VENV_DIR}/bin/activate" ]; then
    source "${VENV_DIR}/bin/activate"        # Linux/macOS/WSL
fi

log "Upgrading pip..."
pip install --upgrade pip --quiet

# ── Step 2: Install PyTorch with CUDA 12.x ───────────────────────────────────

log "Installing PyTorch (CUDA 12.8)..."
pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu128 \
    --quiet

# Verify CUDA availability
$PYTHON_CMD -c "
import torch
if torch.cuda.is_available():
    print(f'  PyTorch {torch.__version__} with CUDA {torch.version.cuda} ✓')
    print(f'  Device: {torch.cuda.get_device_name(0)}')
    print(f'  VRAM: {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB')
else:
    print('  PyTorch installed (CPU only — CUDA not detected)')
"

# ── Step 3: Install PyG (requires torch first) ───────────────────────────────

log "Installing PyTorch Geometric + extensions..."
pip install torch-geometric \
    --quiet

# PyG optional extensions for fast sparse ops
pip install torch-scatter torch-sparse torch-cluster \
    -f https://data.pyg.org/whl/torch-$(python -c "import torch; print(torch.__version__.split('+')[0])")+cu128.html \
    --quiet 2>/dev/null || warn "PyG C++ extensions not prebuilt for this torch version. JIT compilation will be used."

# ── Step 4: Install Project Dependencies ──────────────────────────────────────

log "Installing PayFlow dependencies..."
pip install -e "${PROJECT_ROOT}[dev]" --quiet

# ── Step 5: Pull Ollama Model ─────────────────────────────────────────────────

if command -v ollama &>/dev/null; then
    log "Pulling Qwen-3.5 9B model (this may take a while on first run)..."
    ollama pull qwen3.5:9b || warn "Could not pull model. Ensure Ollama daemon is running."
else
    warn "Skipping model pull (Ollama not installed)."
fi

# ── Step 6: Initialize Git ────────────────────────────────────────────────────

cd "$PROJECT_ROOT"
if [ ! -d ".git" ]; then
    log "Initializing Git repository..."
    git init
    git add -A
    git commit -m "feat: initialize PayFlow project scaffold

- Modular architecture: ingestion, ML, graph, blockchain, LLM, API
- VRAM-constrained config for RTX 4070 8GB
- XGBoost CUDA + PyG NeighborLoader + embedded Merkle ledger
- Dual GPU mode manager (Analysis vs Assistant)"
    log "Initial commit created ✓"
fi

# ── Summary ───────────────────────────────────────────────────────────────────

echo ""
echo -e "${CYAN}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║          PayFlow Environment Initialized                    ║${NC}"
echo -e "${CYAN}╠══════════════════════════════════════════════════════════════╣${NC}"
echo -e "${CYAN}║${NC}  Activate:  ${GREEN}source .venv/Scripts/activate${NC}                   ${CYAN}║${NC}"
echo -e "${CYAN}║${NC}  Run API:   ${GREEN}uvicorn src.api.main:app --reload${NC}               ${CYAN}║${NC}"
echo -e "${CYAN}║${NC}  Tests:     ${GREEN}pytest${NC}                                          ${CYAN}║${NC}"
echo -e "${CYAN}║${NC}  Lint:      ${GREEN}ruff check src/${NC}                                  ${CYAN}║${NC}"
echo -e "${CYAN}╚══════════════════════════════════════════════════════════════╝${NC}"
