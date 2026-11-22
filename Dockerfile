FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim AS uv

FROM python:3.11-slim-bookworm

COPY --from=uv /uv /uvx /bin/

RUN apt-get update \
    && apt-get install --no-install-recommends -y libgomp1 \
    && rm -rf /var/lib/apt/lists/*

ENV UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1 \
    persist_dir=/app/chroma_db \
    LOG_DIR=/app/logs

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY app ./app

RUN mkdir -p /app/chroma_db /app/logs

EXPOSE 8000

CMD ["/opt/venv/bin/uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
