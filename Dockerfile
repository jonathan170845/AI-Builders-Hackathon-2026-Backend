FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:0.12.13 /uv /usr/local/bin/uv
WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy PATH="/app/.venv/bin:$PATH"
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project
COPY app ./app
COPY alembic ./alembic
COPY prepared-data ./prepared-data
COPY alembic.ini ./
RUN uv sync --frozen --no-dev && useradd --create-home appuser && mkdir -p /state && chown appuser /state
USER appuser
ENV APP_ENV=production APP_HOST=0.0.0.0 APP_PORT=8000 DATABASE_URL=sqlite:////state/veritas.db
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health/live', timeout=3)"
CMD ["veritas-api"]
