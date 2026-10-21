# One image, two services (API and UI) — see docker-compose.yml.
FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    ARBITER_DB_PATH=/data/arbiter.db

WORKDIR /app

# Dependencies first (cached layer): exact versions the test suite ran against.
COPY requirements.lock ./
RUN pip install -r requirements.lock

# Application code, config, demo fixture and evaluation results (for the analytics page).
COPY pyproject.toml config.yaml ./
COPY src ./src
COPY scripts ./scripts
COPY data/demo ./data/demo
COPY reports/*.md reports/*.json ./reports/
RUN pip install --no-deps . && rm -rf build \
    && useradd --create-home --uid 1000 arbiter \
    && mkdir -p /data /app/logs \
    && chown -R arbiter:arbiter /data /app/logs

USER arbiter
EXPOSE 8000 8501

# Default: the API. docker-compose overrides the command for the UI.
CMD ["uvicorn", "arbiter.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
