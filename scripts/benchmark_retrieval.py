"""Local benchmark; synthetic queries only, no provider calls."""

import json
import platform
import time

from app.core.config import Settings
from app.main import initialize_retrieval

started = time.perf_counter()
artifacts, service = initialize_retrieval(Settings())
load_seconds = time.perf_counter() - started
queries = [
    "small business software subscription",
    "delivery cost and cash runway",
    "new city customer demand",
]
measurements = []
for query in queries:
    start = time.perf_counter()
    service.company_analogues(query, top_k=5)
    service.historical_failures(query, top_k=5)
    measurements.append(round(time.perf_counter() - start, 4))
print(
    json.dumps(
        {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "companies": len(artifacts.company_records),
            "dimensions": artifacts.embedding_dimension,
            "top_k": 5,
            "queries": len(queries),
            "startup_seconds": round(load_seconds, 3),
            "warm_company_and_failure_query_seconds": measurements,
        },
        indent=2,
    )
)
