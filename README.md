# Veritas Backend

FastAPI backend for Veritas, a decision-intelligence application that evaluates
assumptions, unit economics, historical evidence, and validation plans for a business decision.

## Prerequisites

- Python 3.12 or later.
- `uv` to synchronize dependencies from the lockfile.

## Run locally

```powershell
cd AI-Builders-Hackathon-2026-Backend
uv sync --extra dev --locked
Copy-Item .env.example .env
uv run veritas-api
```

Interactive documentation is available at `http://127.0.0.1:8000/docs`; the raw schema is at
`/openapi.json`. The local Vite frontend (`http://localhost:5173`) is the default allowed CORS
origin. Change `FRONTEND_ORIGINS` in `.env` to allow other origins, using comma-separated values.
The entry point reads `APP_HOST`, `APP_PORT`, and `APP_ENV`; `APP_ENV=development` enables reload.

## Bootstrap endpoints

- `GET /health/live` — confirms that the HTTP process is alive.
- `GET /health/ready` — confirms that application dependencies are ready.
- `POST /api/v1/analyses` — creates an analysis job with `queued` status.
- `GET /api/v1/analyses/{id}` — reads the current temporary job status.

Jobs are currently kept in memory to validate the API contract. Persistent storage and a background
worker will replace this implementation.

`/health/live` only checks the HTTP process. `/health/ready` returns HTTP `200` when ready, or HTTP
`503` with a local check summary when a dependency is not ready.

`financialInputs` is optional. When provided, every field is required; `monthlyOrders` is an integer
and monetary values are bounded to keep later calculations safe. Monetary values are unit-agnostic,
so the frontend must not assume a currency symbol.

## Financial stress test

`app.services.financial.run_financial_stress_test` is the source of truth for unit economics. It
calculates monetary values with `Decimal(str(input))` and applies `ROUND_HALF_UP` to two decimal
places only when mapping to the API. Percentages and runway are rounded to four decimal places.
`runwayMonths` is `null` when there is no burn, and `breakEvenOrders` is `null` when the contribution
margin per order is not positive. `FINANCIAL_LOW_RUNWAY_MONTHS` defines the runway warning threshold
(default: 6 months).

IDX comparisons are *directional* only: the contribution-margin ratio is compared with IDX gross
margin and the response contains an explicit disclaimer. Other IDX metrics are not returned because
the current input does not provide accounting debt, assets, or operating cash flow.

## LLM pipeline and cache

`AnalysisPipeline` is an orchestration service called by a background worker, so the bootstrap
endpoint does not yet execute an LLM directly. The pipeline requests validated JSON for assumption
extraction, retrieval queries, grounded evaluation, and conservative review. It fails the entire
analysis when any stage is invalid; partial reports are never published.

The evaluator receives only an evidence packet for each assumption. Only historical failures can
support a claim; company analogues are always labelled as context-only. When retrieval is empty, the
pipeline returns `Insufficient Evidence` without calling the evaluator. Financial results always
come from `financial.py`, not from LLM output. The decision and all evidence text are treated as
untrusted data and explicitly delimited in prompts.

For OpenRouter, set `OPENROUTER_API_KEY` and `OPENROUTER_MODEL` only in the backend `.env` file. The
client uses explicit timeouts and retries only timeouts/network errors, HTTP 429, and 5xx responses.
It also limits global concurrency, calls per analysis, response size, output tokens, and provider-
reported cost. Never put the key in the frontend or logs. Change `PROMPT_VERSION` whenever prompt
behaviour changes so old cache entries are not reused.

The SQLite cache is an exact-key cache based on the model, prompt version, system prompt, payload,
sampling parameters (including temperature), and response schema. It stores structured output and
may therefore contain sensitive data; a hash key **does not** conceal cache contents. Do not enable
or share the cache database without appropriate access and retention policies. Use
`LLM_CACHE_ENABLED=false` while debugging. Set retention and the row limit through
`LLM_CACHE_RETENTION_DAYS` and `LLM_CACHE_MAX_ROWS`.

## Database and migrations

Before using the worker or cache with a new database, run this from the backend root:

```powershell
uv run alembic upgrade head
```

The initial migration creates only `llm_cache`; analysis tables will use the same SQLAlchemy and
Alembic foundation. `DATABASE_URL` may use SQLite for the MVP. Do not commit a local database file
or cache contents to Git.

## Configuration

Copy `.env.example` to `.env`. Do not commit `.env` files or API keys.
`EMBEDDING_MODEL_PATH_OR_ID` must point to the same SentenceTransformer model recorded in the
artifact manifest. For offline deployment, place the audited model on disk and use its path; the API
process must not rely on downloading a model during startup.

## Prepare retrieval artifacts

The raw artifacts in `../AI-Builders-Hackathon-2026-Data/Data_Final` are incomplete for runtime:
historical-failure embeddings and a manifest are intentionally not generated by the API. After
verifying dataset and model licences and provenance, prepare a separate Git-ignored directory once:

```powershell
uv run python scripts/prepare_data.py `
  --source-dir ../AI-Builders-Hackathon-2026-Data/Data_Final `
  --output-dir ./prepared-data `
  --model <audited-model-path-or-id> `
  --model-id <model-id> `
  --model-revision <immutable-revision>
```

Set `DATA_DIR=./prepared-data` and `EMBEDDING_MODEL_PATH_OR_ID` to the same model. Use
`RETRIEVAL_MAX_TOP_K` (1--50, default 10) to cap evidence returned for each query.
`prepare_data.py` performs a full NaN/Inf scan, builds retrieval text and historical-failure
embeddings, then records checksums, sample checksums, stable-ID ordering, dtype, dimensions, metric,
normalization, and model revision in `artifact-manifest.json`. At startup, the API verifies the
manifest and versioned samples without duplicating the array or performing a full scan. Readiness
remains `503` until artifacts and the model are validated.

Audit the model used by the source notebook before use: pin an immutable revision and do not enable
remote custom code. Benchmark separately before choosing FAISS or HNSW. Benchmarking must record
load, encoding, and search times separately without storing decision text in logs.

## Code quality

```powershell
uv run ruff check .
uv run pytest
```
