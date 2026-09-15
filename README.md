# Veritas Backend

FastAPI backend for Veritas, a decision-intelligence application that evaluates
assumptions, unit economics, historical evidence, and validation plans for a business decision.

## Prerequisites

- Python 3.12 or later.
- `uv` to synchronize dependencies from the lockfile.

## Run locally

```powershell
cd AI-Builders-Hackathon-2026-Backend
uv sync --extra dev --frozen
Copy-Item .env.example .env
uv run veritas-api
```

Interactive documentation is available at `http://127.0.0.1:8000/docs`; the raw schema is at
`/openapi.json`. The local Vite frontend (`http://localhost:5173`) is the default allowed CORS
origin. Change `FRONTEND_ORIGINS` in `.env` to allow other origins, using comma-separated values.
The entry point reads `APP_HOST`, `APP_PORT`, and `APP_ENV`; `APP_ENV=development` enables reload.

## API endpoints

- `GET /health/live` — confirms that the HTTP process is alive.
- `GET /health/ready` — confirms that application dependencies are ready.
- `POST /api/v1/analyses` — creates an analysis job with `queued` status.
- `GET /api/v1/analyses/{id}` — reads the current persisted job status or result.
- `GET /api/v1/analyses?limit=20&cursor=...` — lists compact newest-first job history.

Jobs are persisted in SQLite or PostgreSQL through SQLAlchemy and run through FastAPI
`BackgroundTasks`. This is a single-process MVP: do not run multiple API workers until an external
task queue is introduced. A process restart marks any job left in `queued` or `processing` as failed with
`WORKER_INTERRUPTED`, so polling clients never wait indefinitely.

`MAX_CONCURRENT_ANALYSES` controls in-process worker concurrency (default `1`) and
`MAX_QUEUED_ANALYSES` caps waiting jobs (default `10`). A full queue returns HTTP `503` with
`CAPACITY_EXCEEDED` and a `Retry-After` header; no job row is created. The frontend should disable
repeated submits while its create request is active.

`/health/live` only checks the HTTP process. `/health/ready` returns HTTP `200` only after database
migrations, artifacts, and the embedding model are ready; otherwise it returns HTTP `503` with a
local check summary.

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

Migrations create `llm_cache` and `analyses`, including stable history indexes. SQLite connections
use WAL mode and `SQLITE_BUSY_TIMEOUT_MS` (default `5000`) to improve concurrent read/write behavior
for the demo. `DATABASE_URL` may use SQLite for the MVP. Do not commit a local database file or cache
contents to Git.

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


## Verified local artifact handoff (2026-09-11)

The source copy under `E:/BINUS_CODING/Ai Builders Hackhaton/Data Final` contains
115,798 CSV **records** and a `(115798, 384)` float32 embedding matrix. Counting physical
CSV lines is not a record count because quoted text contains newlines. The source notebook
exports both from the same candidate dataframe with `normalize_embeddings=True`.
Twelve evenly spaced records re-encoded with the supplied local model have cosine similarity
approximately 1.0 with saved vectors. No full company re-embedding is needed for this copy.

The local SentenceTransformer model uses the v6 module layout. Dependencies now support v6.
The upstream immutable Hub revision is not available locally; do not invent one. Preparation
records a SHA-256 identity of model weights/config/tokenizer files instead, checks 32 company
samples against that exact model, fully checks vector norms/finite values, and generates failure
vectors. Runtime verifies the model identity before accepting analyses.

```powershell
uv run python scripts/prepare_data.py --source-dir "E:/BINUS_CODING/Ai Builders Hackhaton/Data Final" --output-dir ./prepared-data --model "E:/BINUS_CODING/Ai Builders Hackhaton/AI-Builders-Hackhaton-2026-Backend/models/multilingual-minilm" --model-id sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
```

Set `DATA_DIR=./prepared-data` and `EMBEDDING_MODEL_PATH_OR_ID` to the model directory.
Source artifacts are never overwritten. Dataset redistribution permissions are unverified;
keep the original and prepared data local until provenance/licensing is confirmed. This is
separate from the locally supplied model card's Apache-2.0 declaration.

## Quality and contract checks

```powershell
uv sync --extra dev --frozen
uv run --frozen ruff check .
uv run --frozen pytest --cov=app.services --cov=app.api --cov-report=term-missing
uv run --frozen python scripts/export_contract.py --check
```

After an intentional API change, run `uv run python scripts/export_contract.py` and review the
contract diff. Frontend uses the exported OpenAPI schema and its generated TypeScript types.
Static backend type checking is deferred (SHOULD); runtime schema tests and offline integration
tests remain required. Tests marked `data` skip unless `RUN_DATA_INTEGRATION=1`; no test makes
live provider calls by default. A live OpenRouter end-to-end run requires an explicitly
configured provider and model; fixture success does not prove provider availability.

## Containers and operations

Set `LOCAL_MODEL_DIR` and optionally `PREPARED_DATA_DIR` to prepared host directories,
then `docker compose up --build`. Compose applies migrations through a separate one-shot service,
mounts data/model read-only, and keeps SQLite in a named volume. Frontend opens at
`http://localhost:8080`, backend at `http://localhost:8000`; both bind to loopback. The frontend
serves SPA fallback for `/analyses/:id` and `/history`. Liveness checks process health; inspect
`/health/ready` before the demo to verify actual data/model/database readiness.

For local development, apply `uv run alembic upgrade head` before `uv run veritas-api`, then
run `pnpm install --frozen-lockfile` and `pnpm run dev` in the frontend repo. Missing migrations,
model identity mismatch, or missing artifacts yield readiness 503. History remains readable
when only the retrieval service is unavailable. Configure CORS for the precise frontend origin.
Provider keys stay in backend `.env`; missing credentials lead to a typed failed analysis.

Single process only. `MAX_QUEUED_ANALYSES` is at least 1; `ANALYSIS_TIMEOUT_SECONDS` defaults to
600. Provider calls have an actual wall-clock timeout and streamed byte limit. On timeout,
Python threads already doing CPU work finish naturally; do not treat timeout as forced thread
termination. Structured logs contain request IDs, stage and errors, without decision/prompt bodies.

Before backing up SQLite, stop the API and copy the database plus existing WAL/SHM files
(or use SQLite's backup API). Restore while the API is stopped and rerun migrations before
startup. Do not expose the unauthenticated MVP publicly without network access restrictions
or rate limiting. Local demo mode remains explicitly labelled in the frontend.

Plan at least 2 GB disk for environment/model/artifacts and initially allow 4 GB free RAM;
these are planning allowances, not measured minimums. Record machine, data rows, dimensions,
cold/warm startup, query time, and RAM for the demo environment. Review provider quota and
one uncached decision before presenting. Fallback is the labelled static demo or an exact
cache hit, never a cached result belonging to a different decision.


The notebook's `inclusionai/ling-3.0-flash-fin:free` provider currently rejects JSON mode
with HTTP 400 (unsupported `structured-outputs`). For this model set `LLM_RESPONSE_FORMAT=text`.
The adapter still sends the full JSON schema in the system prompt and strictly validates the
returned JSON/fenced JSON. For a provider supporting JSON mode use `json_object` (default).
The format is included in cache identity; invalid text never becomes a successful report.

The local Ling reasoning model exhausted the original 1,200-token cap during evidence evaluation
(`finish_reason=length`). The verified local configuration uses `LLM_MAX_TOKENS=4096`.
Truncated output now returns `LLM_OUTPUT_TRUNCATED`; reasoning text is never parsed as the final answer.
See [OpenRouter reasoning token documentation](https://openrouter.ai/docs/guides/best-practices/reasoning-tokens).
For a repeatable live check, run `uv run python scripts/smoke_api.py --live` with the server running;
add `--financial` to verify known unit-economics values. These commands use a synthetic decision,
create history entries, and call the configured provider. Without `--live`, the script only reads health/history.
