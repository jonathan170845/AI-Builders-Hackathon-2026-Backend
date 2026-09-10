from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from app.services.artifacts import (
    MANIFEST_FILENAME,
    MANIFEST_SCHEMA_VERSION,
    artifact_entry,
    ordered_values_sha256,
)


def make_artifact_directory(path: Path) -> Path:
    metadata = path / "software_it_candidate_metadata.csv"
    with metadata.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["id", "name", "retrieval_text"])
        writer.writeheader()
        writer.writerows(
            [
                {"id": "company-a", "name": "Alpha", "retrieval_text": "alpha software"},
                {"id": "company-b", "name": "Beta", "retrieval_text": "beta delivery"},
                {"id": "company-c", "name": "Gamma", "retrieval_text": "gamma finance"},
            ]
        )
    company_embeddings = path / "software_it_candidate_embeddings.npy"
    np.save(company_embeddings, np.array([[1, 0], [0, 1], [1, 0]], dtype=np.float32))

    documents = path / "startup_failure_documents.json"
    raw_cases = [
        {
            "company": "Failed Alpha",
            "funding": "$1M",
            "category": "Seed",
            "investors": ["Investor A"],
            "failure_reason": "No demand",
            "prompt": "Why did it fail?",
            "retrieval_text": "Alpha failure no demand",
        },
        {
            "company": "Failed Beta",
            "funding": "$2M",
            "category": "Series A",
            "investors": [],
            "failure_reason": "Costs exceeded revenue",
            "prompt": "Why did it fail?",
            "retrieval_text": "Beta failure costs",
        },
    ]
    raw_source = path / "startup-failure-insights.json"
    raw_source.write_text(json.dumps(raw_cases), encoding="utf-8")
    for case in raw_cases:
        case["retrieval_text"] = (
            f"Company: {case['company']}\nFailure Reason: {case['failure_reason']}\n"
            f"Failure Question: {case['prompt']}"
        )
    documents.write_text(json.dumps(raw_cases), encoding="utf-8")
    failure_embeddings = path / "startup_failure_embeddings.npy"
    np.save(failure_embeddings, np.array([[1, 0], [0, 1]], dtype=np.float32))

    overall = path / "idx_financial_benchmark_overall.csv"
    overall.write_text(
        "benchmark_scope,ratio,count,median\nIDX_ALL,gross_margin,1,0.2\n", encoding="utf-8"
    )
    period = path / "idx_financial_benchmark_by_period.csv"
    period.write_text(
        "benchmark_scope,ratio,count,median\nIDX_PERIOD,gross_margin,1,0.2\n", encoding="utf-8"
    )
    ratio = path / "idx_financial_ratio_benchmark_ready.csv"
    ratio.write_text("ticker,year\nTEST,2024\n", encoding="utf-8")

    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "embedding_model": {
            "id": "fixture",
            "revision": "test",
            "dimension": 2,
            "metric": "dot_product",
            "normalization": "l2",
        },
        "metadata_stable_id_column": "id",
        "metadata_stable_ids_sha256": ordered_values_sha256(
            ["company-a", "company-b", "company-c"]
        ),
        "artifacts": {
            "company_metadata": artifact_entry(metadata, rows=3, stable_id_column="id"),
            "company_embeddings": artifact_entry(
                company_embeddings, rows=3, dimension=2, dtype="float32"
            ),
            "failure_documents": artifact_entry(documents, rows=2),
            "failure_embeddings": artifact_entry(
                failure_embeddings, rows=2, dimension=2, dtype="float32"
            ),
            "failure_cases": artifact_entry(raw_source, rows=2),
            "idx_benchmark_overall": artifact_entry(overall, rows=1),
            "idx_benchmark_by_period": artifact_entry(period, rows=1),
            "idx_ratio_benchmark": artifact_entry(ratio, rows=1),
        },
    }
    (path / MANIFEST_FILENAME).write_text(json.dumps(manifest), encoding="utf-8")
    return path
