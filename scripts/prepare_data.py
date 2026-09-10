"""Prepare a self-describing retrieval artifact directory from the raw Data_Final source.

Run with ``uv run python scripts/prepare_data.py --source-dir <Data_Final>``.
This deliberately performs the expensive full checks once, before the API starts.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from app.services.artifacts import (
    MANIFEST_FILENAME,
    MANIFEST_SCHEMA_VERSION,
    artifact_entry,
    build_failure_retrieval_text,
    ordered_values_sha256,
)
from app.services.retrieval import SentenceTransformerEncoder

SOURCE_FILENAMES = (
    "software_it_candidate_metadata.csv",
    "software_it_candidate_embeddings.npy",
    "startup-failure-insights.json",
    "idx_financial_benchmark_overall.csv",
    "idx_financial_benchmark_by_period.csv",
    "idx_financial_ratio_benchmark_ready.csv",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--model", required=True, help="Local model path or Hugging Face model ID")
    parser.add_argument(
        "--model-id", required=True, help="Audited immutable model ID recorded in manifest"
    )
    parser.add_argument("--model-revision", required=True, help="Audited immutable model revision")
    return parser.parse_args()


def copy_if_needed(source: Path, destination: Path) -> Path:
    target = destination / source.name
    if source.resolve() != target.resolve():
        shutil.copy2(source, target)
    return target


def csv_row_count(path: Path) -> int:
    with path.open(newline="", encoding="utf-8") as csv_file:
        return sum(1 for _ in csv.DictReader(csv_file))


def full_embedding_validation(path: Path, label: str) -> tuple[int, int, str]:
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    if array.ndim != 2 or not np.issubdtype(array.dtype, np.floating):
        raise ValueError(f"{label} must be a two-dimensional floating-point array")
    for start in range(0, array.shape[0], 8192):
        if not np.isfinite(array[start : start + 8192]).all():
            raise ValueError(f"{label} contains NaN or Infinity")
    return int(array.shape[0]), int(array.shape[1]), str(array.dtype)


def load_failure_cases(path: Path) -> list[dict[str, Any]]:
    cases = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(cases, list) or not all(isinstance(case, dict) for case in cases):
        raise ValueError("startup-failure-insights.json must contain a list of objects")
    documents = []
    for case in cases:
        document = dict(case)
        document["retrieval_text"] = build_failure_retrieval_text(case)
        if not document["retrieval_text"]:
            raise ValueError("A failure case creates an empty retrieval document")
        documents.append(document)
    return documents


def company_ids(path: Path) -> list[str]:
    with path.open(newline="", encoding="utf-8") as csv_file:
        ids = [(row.get("id") or "").strip() for row in csv.DictReader(csv_file)]
    if not ids or not all(ids) or len(ids) != len(set(ids)):
        raise ValueError("Company metadata IDs must be present and unique")
    return ids


def main() -> None:
    args = parse_args()
    source_dir = args.source_dir.resolve()
    output_dir = (args.output_dir or args.source_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    source_paths = {filename: source_dir / filename for filename in SOURCE_FILENAMES}
    missing = [filename for filename, path in source_paths.items() if not path.is_file()]
    if missing:
        raise SystemExit(f"Source artifacts are missing: {', '.join(missing)}")

    copied = {name: copy_if_needed(path, output_dir) for name, path in source_paths.items()}
    company_count, dimension, company_dtype = full_embedding_validation(
        copied["software_it_candidate_embeddings.npy"], "Company embeddings"
    )
    ids = company_ids(copied["software_it_candidate_metadata.csv"])
    if len(ids) != company_count:
        raise SystemExit("Company metadata row count does not match company embeddings")

    documents = load_failure_cases(copied["startup-failure-insights.json"])
    documents_path = output_dir / "startup_failure_documents.json"
    documents_path.write_text(
        json.dumps(documents, ensure_ascii=False, separators=(",", ":")), encoding="utf-8"
    )

    encoder = SentenceTransformerEncoder(str(args.model), args.model_revision)
    if encoder.dimension() != dimension:
        raise SystemExit("Embedding model dimension does not match company embeddings")
    failure_vectors = np.asarray(
        encoder.encode([document["retrieval_text"] for document in documents])
    )
    if (
        failure_vectors.shape != (len(documents), dimension)
        or not np.isfinite(failure_vectors).all()
    ):
        raise SystemExit("Historical failure embedding output is invalid")
    failure_path = output_dir / "startup_failure_embeddings.npy"
    np.save(failure_path, failure_vectors.astype(np.float32, copy=False))
    failure_count, failure_dimension, failure_dtype = full_embedding_validation(
        failure_path, "Historical failure embeddings"
    )
    if failure_dimension != dimension:
        raise SystemExit("Historical failure embedding dimension does not match company embeddings")

    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "embedding_model": {
            "id": args.model_id,
            "revision": args.model_revision,
            "dimension": dimension,
            "metric": "dot_product",
            "normalization": "l2",
        },
        "metadata_stable_id_column": "id",
        "metadata_stable_ids_sha256": ordered_values_sha256(ids),
        "artifacts": {
            "company_metadata": artifact_entry(
                copied["software_it_candidate_metadata.csv"], rows=len(ids), stable_id_column="id"
            ),
            "company_embeddings": artifact_entry(
                copied["software_it_candidate_embeddings.npy"],
                rows=company_count,
                dimension=dimension,
                dtype=company_dtype,
            ),
            "failure_cases": artifact_entry(
                copied["startup-failure-insights.json"], rows=len(documents)
            ),
            "failure_documents": artifact_entry(documents_path, rows=len(documents)),
            "failure_embeddings": artifact_entry(
                failure_path, rows=failure_count, dimension=failure_dimension, dtype=failure_dtype
            ),
            "idx_benchmark_overall": artifact_entry(
                copied["idx_financial_benchmark_overall.csv"],
                rows=csv_row_count(copied["idx_financial_benchmark_overall.csv"]),
            ),
            "idx_benchmark_by_period": artifact_entry(
                copied["idx_financial_benchmark_by_period.csv"],
                rows=csv_row_count(copied["idx_financial_benchmark_by_period.csv"]),
            ),
            "idx_ratio_benchmark": artifact_entry(
                copied["idx_financial_ratio_benchmark_ready.csv"],
                rows=csv_row_count(copied["idx_financial_ratio_benchmark_ready.csv"]),
            ),
        },
    }
    (output_dir / MANIFEST_FILENAME).write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        f"Prepared {output_dir} with {company_count} companies and {failure_count} failure cases."
    )


if __name__ == "__main__":
    main()
