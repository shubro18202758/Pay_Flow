FROM node:22-bookworm-slim AS frontend

WORKDIR /app/frontend/app
COPY frontend/app/package*.json ./
RUN npm ci --legacy-peer-deps
COPY frontend/app ./
RUN npm run build


FROM python:3.11-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PAYFLOW_CPU_ONLY=1 \
    PAYFLOW_HOST=0.0.0.0 \
    PAYFLOW_MAX_AGENT_TASKS=0 \
    OLLAMA_URL=http://ollama-u6tzx1:11434 \
    OLLAMA_MODEL=qwen3.5:4b \
    PORT=8000

WORKDIR /app

COPY pyproject.toml requirements.deploy.txt README.md LICENSE ./
COPY config ./config
COPY src ./src
COPY frontend/templates ./frontend/templates
COPY main.py landing.html ./
COPY scripts ./scripts
COPY --from=frontend /app/frontend/app/dist ./frontend/app/dist

RUN python -m pip install --upgrade pip \
    && python -m pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu "torch==2.5.1+cpu" \
    && python -m pip install --no-cache-dir -r requirements.deploy.txt \
    && mkdir -p data/raw data/processed data/synthetic artifacts/models artifacts/evidence artifacts/reports logs

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=30s --start-period=180s --retries=6 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/', timeout=20).read(1)"

CMD ["sh", "-c", "python main.py --serve --cpu-only ${PAYFLOW_SKIP_LLM:+--skip-llm} --events ${PAYFLOW_EVENTS:-1500} --accounts ${PAYFLOW_ACCOUNTS:-600} --fraud-ratio ${PAYFLOW_FRAUD_RATIO:-0.08} --dashboard-host ${PAYFLOW_HOST:-0.0.0.0} --dashboard-port ${PORT:-8000}"]
