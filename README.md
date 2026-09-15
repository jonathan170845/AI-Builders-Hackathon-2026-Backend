# Veritas Backend

FastAPI backend for **Veritas**, a decision-intelligence application that evaluates business assumptions using historical evidence, unit-economics stress tests, and validation plans.

## Features

- Creates asynchronous analysis jobs and persists their status and results.
- Extracts business assumptions, retrieves historical evidence, and produces grounded evaluations.
- Calculates unit economics deterministically; financial results never come from the LLM.
- Provides cursor-paginated analysis history and an OpenAPI contract.
- Supports SQLite for the MVP and PostgreSQL through SQLAlchemy.
- Caches LLM responses by model, prompt, payload, sampling parameters, and response schema.

## Prerequisites

- Python 3.12 or later.
- [uv](https://docs.astral.sh/uv/) to manage dependencies.
- A prepared retrieval-artifact directory and its matching SentenceTransformer model.
- An OpenRouter API key and model to run analyses to completion.

## Run locally

From the backend root:

```powershell
uv sync --extra dev --frozen
Copy-Item .env.example .env
uv run alembic upgrade head
uv run veritas-api
```

The server runs at `http://127.0.0.1:8000`. Interactive API documentation is available at
[`/docs`](http://127.0.0.1:8000/docs), and the raw contract is at
[`/openapi.json`](http://127.0.0.1:8000/openapi.json).

`APP_ENV=development` (the default) enables reload. Set `APP_HOST`, `APP_PORT`, and
`FRONTEND_ORIGINS` in `.env` as needed. `FRONTEND_ORIGINS` accepts comma-separated origins, for
example `http://localhost:5173,http://localhost:3000`.

> `.env` may contain credentials. Do not commit it or expose provider keys to the frontend.

## Minimum configuration

Copy `.env.example` to `.env`, then set these values after preparing the artifacts:

```dotenv
DATA_DIR=./prepared-data
EMBEDDING_MODEL_PATH_OR_ID=C:/path/to/model
OPENROUTER_API_KEY=...
OPENROUTER_MODEL=provider/model
```

| Area | Key settings | Purpose |
| --- | --- | --- |
| Application | `APP_ENV`, `APP_HOST`, `APP_PORT`, `FRONTEND_ORIGINS` | API runtime and CORS. |
| Database | `DATABASE_URL`, `SQLITE_BUSY_TIMEOUT_MS` | Defaults to `sqlite:///./veritas.db`. |
| Retrieval | `DATA_DIR`, `EMBEDDING_MODEL_PATH_OR_ID`, `RETRIEVAL_MAX_TOP_K` | Artifact and model must match the manifest. |
| LLM | `OPENROUTER_API_KEY`, `OPENROUTER_MODEL`, `LLM_RESPONSE_FORMAT` | The provider is enabled only when both OpenRouter settings exist. |
| Work queue | `MAX_CONCURRENT_ANALYSES`, `MAX_QUEUED_ANALYSES`, `ANALYSIS_TIMEOUT_SECONDS` | Defaults: one worker, ten queued jobs, 600-second timeout. |
| Cache | `LLM_CACHE_ENABLED`, `LLM_CACHE_RETENTION_DAYS`, `LLM_CACHE_MAX_ROWS` | Cached structured output must be treated as sensitive data. |

Use `LLM_RESPONSE_FORMAT=text` for providers that do not support structured output. The returned
JSON is still strictly validated. If a reasoning model needs more output capacity, set
`LLM_MAX_TOKENS=4096`; the example default remains `1200`.

## Prepare retrieval artifacts

The API does not build embeddings at startup. After checking dataset and model provenance and
licenses, prepare artifacts once in a Git-ignored directory:

```powershell
uv run python scripts/prepare_data.py `
  --source-dir ../AI-Builders-Hackathon-2026-Data/Data_Final `
  --output-dir ./prepared-data `
  --model <model-path-or-model-id> `
  --model-id <model-id> `
  --model-revision <immutable-revision>
```

The script validates `NaN`/`Inf` values, builds historical-failure retrieval text and embeddings,
and writes `artifact-manifest.json`. The manifest records checksums, embedding shape and dtype,
stable ID ordering, metric, normalization, and model identity. At startup, the API validates the
manifest and model identity before accepting analysis requests.

For the verified local data and model copy, use:

```powershell
uv run python scripts/prepare_data.py `
  --source-dir "E:/BINUS_CODING/Ai Builders Hackhaton/Data Final" `
  --output-dir ./prepared-data `
  --model "E:/BINUS_CODING/Ai Builders Hackhaton/AI-Builders-Hackhaton-2026-Backend/models/multilingual-minilm" `
  --model-id sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
```

Set `DATA_DIR` to the output directory and `EMBEDDING_MODEL_PATH_OR_ID` to that same model. Do
not overwrite source artifacts or redistribute data until redistribution permission is confirmed.

## Health checks

| Endpoint | Purpose |
| --- | --- |
| `GET /health/live` | Confirms that the HTTP process is alive. |
| `GET /health/ready` | Confirms that database migrations, artifacts, and the embedding model are ready. Returns `503` with check details otherwise. |

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health/ready
```

## Analysis API

| Endpoint | Response |
| --- | --- |
| `POST /api/v1/analyses` | Creates a job and returns `202 Accepted`. The `Location` header identifies the job URL. |
| `GET /api/v1/analyses/{id}` | Returns the latest `queued`, `processing`, `completed`, or `failed` job. |
| `GET /api/v1/analyses?limit=20&cursor=...` | Returns newest-first job history. `limit` is 1--100. |

Create an analysis:

```powershell
$body = @{
  decision = "Pilot a campus food-delivery service before expanding to another campus."
  financialInputs = @{
    monthlyOrders = 100
    revenuePerOrder = 10
    variableCostPerOrder = 3
    promoSubsidy = 0
    deliveryCost = 1
    fixedCost = 400
    driverCost = 100
    cashBalance = 5000
  }
} | ConvertTo-Json -Depth 3

$job = Invoke-RestMethod `
  -Method Post `
  -Uri http://127.0.0.1:8000/api/v1/analyses `
  -ContentType application/json `
  -Body $body
$job
```

Poll its result:

```powershell
Invoke-RestMethod "http://127.0.0.1:8000/api/v1/analyses/$($job.id)"
```

`decision` must contain 10--5,000 characters. `financialInputs` is optional, but every field is
required when it is supplied. `monthlyOrders` is an integer; money values must be non-negative and
have at most four decimal places. Amounts are unit-agnostic, so clients must not assume a currency.

Errors use this shape and include a `requestId` for tracing:

```json
{
  "error": {
    "code": "ARTIFACTS_NOT_READY",
    "message": "Evidence service is not ready",
    "requestId": "..."
  }
}
```

Jobs use FastAPI `BackgroundTasks` and are intended for a single-process MVP. Do not run multiple
API workers until an external task queue is introduced. A process restart changes jobs left in
`queued` or `processing` to `failed` with `WORKER_INTERRUPTED`. When the queue is full, the API
returns `503 CAPACITY_EXCEEDED` with `Retry-After`; it does not create a job row.

## Analysis behaviour

The pipeline requests validated JSON for assumption extraction, retrieval queries, grounded
evaluation, and conservative review. An invalid or failed stage prevents publication of partial
reports. Historical failures are risk context, not predictions; company analogues are always
context-only. If retrieval finds no evidence, the evaluator is not called and the result is
`Insufficient Evidence`.

Financial calculations use `Decimal(str(input))`, applying `ROUND_HALF_UP` only while mapping to
the API. `runwayMonths` is `null` when there is no burn; `breakEvenOrders` is `null` when the
contribution margin per order is not positive. `FINANCIAL_LOW_RUNWAY_MONTHS` sets the low-runway
warning threshold (default: six months).

## Database, checks, and API contract

Run migrations before using a new database:

```powershell
uv run alembic upgrade head
```

SQLite uses WAL mode and `SQLITE_BUSY_TIMEOUT_MS` to improve concurrent read/write behavior. Do
not commit a local database or cache contents. To back up SQLite, stop the API and copy the database
with any existing WAL/SHM files, or use SQLite's backup API.

Run quality and contract checks:

```powershell
uv sync --extra dev --frozen
uv run --frozen ruff check .
uv run --frozen pytest --cov=app.services --cov=app.api --cov-report=term-missing
uv run --frozen python scripts/export_contract.py --check
```

After an intentional API change, run `uv run python scripts/export_contract.py` and review the
update to `contracts/openapi.json`. Tests marked `data` require `RUN_DATA_INTEGRATION=1`; default
tests never call a live provider.

With the server running, use the smoke tests:

```powershell
uv run python scripts/smoke_api.py
uv run python scripts/smoke_api.py --live --financial
```

The second command creates synthetic history and calls the configured provider.

## Docker Compose

Ensure the sibling frontend repository exists at `../AI-Builders-Hackathon-2026-Frontend`, then
set the model and artifact paths in the active environment:

```powershell
$env:LOCAL_MODEL_DIR = "C:/path/to/model"
$env:PREPARED_DATA_DIR = "C:/path/to/prepared-data"
docker compose up --build
```

Compose runs migrations as a one-shot service, mounts model and data read-only, and stores SQLite
in a named volume. The frontend is available at `http://localhost:8080` and the backend at
`http://localhost:8000`; both bind to loopback only.

## Operational notes

- This MVP has no authentication, rate limiting, or distributed worker. Do not expose it publicly without appropriate network controls.
- Provider keys must not appear in logs. The LLM client limits timeout, retries, concurrency, calls, response bytes, output tokens, and reported cost per analysis.
- Allocate at least 2 GB disk and initially allow 4 GB free RAM for the demo. These are planning allowances, not measured minimums.
- Before a demo, check `/health/ready`, provider quota, an uncached result, and runtime metrics on the environment being used.
