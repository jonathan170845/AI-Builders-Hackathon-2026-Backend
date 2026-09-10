from __future__ import annotations

import csv
import hashlib
import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

MANIFEST_FILENAME = "artifact-manifest.json"
MANIFEST_SCHEMA_VERSION = 1
SAMPLE_BYTES = 4096

REQUIRED_COMPANY_COLUMNS = frozenset({"id", "name", "retrieval_text"})
REQUIRED_FAILURE_FIELDS = frozenset({"company", "failure_reason", "prompt"})
REQUIRED_BENCHMARK_COLUMNS = frozenset({"benchmark_scope", "ratio", "count", "median"})


class ArtifactValidationError(RuntimeError):
    """A startup-safe artifact error; `public_message` must not reveal server paths."""

    def __init__(self, public_message: str) -> None:
        super().__init__(public_message)
        self.public_message = public_message


@dataclass(frozen=True, slots=True)
class CompanyRecord:
    company_id: str
    name: str
    retrieval_text: str
    short_description: str | None
    categories: str | None
    locations: str | None
    company_type: str | None
    employee_min: float | None
    employee_max: float | None
    founded_year: int | None
    funding_total: float | None


@dataclass(frozen=True, slots=True)
class FailureDocument:
    case_index: int
    company: str
    funding: str | None
    category: str | None
    investors: tuple[str, ...]
    failure_reason: str
    prompt: str
    retrieval_text: str


@dataclass(frozen=True, slots=True)
class DataArtifacts:
    """Validated, process-wide data required by the evidence retrieval layer."""

    manifest: dict[str, Any]
    company_records: tuple[CompanyRecord, ...]
    company_embeddings: NDArray[np.floating[Any]]
    failure_documents: tuple[FailureDocument, ...]
    failure_embeddings: NDArray[np.floating[Any]]
    idx_benchmark_overall: tuple[dict[str, str], ...]
    idx_benchmark_by_period: tuple[dict[str, str], ...]
    idx_ratio_benchmark: tuple[dict[str, str], ...]

    @property
    def embedding_dimension(self) -> int:
        return int(self.company_embeddings.shape[1])


def build_failure_retrieval_text(case: dict[str, Any]) -> str:
    """Build the same deterministic evidence text formerly created in the notebook."""
    parts = [
        f"Company: {str(case.get('company') or '').strip()}",
        f"Failure Reason: {str(case.get('failure_reason') or '').strip()}",
    ]
    prompt = str(case.get("prompt") or "").strip()
    if prompt:
        parts.append(f"Failure Question: {prompt}")
    completion = str(case.get("completion") or "").strip()
    if completion:
        parts.append(f"Failure Summary: {completion}")
    return "\n".join(part for part in parts if part.split(": ", 1)[-1])


def ordered_values_sha256(values: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(value.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as artifact_file:
        for block in iter(lambda: artifact_file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_sample_sha256(path: Path) -> str:
    """Small deterministic fingerprint used at startup; full hash is made by preparation."""
    size = path.stat().st_size
    offsets = sorted({0, max(0, size // 2 - SAMPLE_BYTES // 2), max(0, size - SAMPLE_BYTES)})
    digest = hashlib.sha256()
    with path.open("rb") as artifact_file:
        for offset in offsets:
            artifact_file.seek(offset)
            digest.update(artifact_file.read(SAMPLE_BYTES))
    return digest.hexdigest()


def artifact_entry(path: Path, *, rows: int | None = None, **extra: Any) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "filename": path.name,
        "size_bytes": path.stat().st_size,
        "sha256": file_sha256(path),
        "sample_sha256": file_sample_sha256(path),
    }
    if rows is not None:
        entry["rows"] = rows
    entry.update(extra)
    return entry


def _read_manifest(data_dir: Path) -> dict[str, Any]:
    path = data_dir / MANIFEST_FILENAME
    if not path.is_file():
        raise ArtifactValidationError("Artifact manifest is missing; run scripts/prepare_data.py")
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactValidationError("Artifact manifest is invalid") from exc
    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ArtifactValidationError("Artifact manifest schema version is unsupported")
    if not isinstance(manifest.get("artifacts"), dict):
        raise ArtifactValidationError("Artifact manifest does not describe artifacts")
    return manifest


def _artifact_path(data_dir: Path, manifest: dict[str, Any], name: str) -> Path:
    entry = manifest["artifacts"].get(name)
    if not isinstance(entry, dict) or not isinstance(entry.get("filename"), str):
        raise ArtifactValidationError(f"Artifact manifest entry '{name}' is missing")
    filename = Path(entry["filename"])
    if filename.name != entry["filename"]:
        raise ArtifactValidationError(f"Artifact manifest entry '{name}' is invalid")
    path = data_dir / filename
    if not path.is_file():
        raise ArtifactValidationError(f"Required artifact '{name}' is missing")
    if path.stat().st_size != entry.get("size_bytes"):
        raise ArtifactValidationError(f"Artifact '{name}' has an unexpected size")
    if file_sample_sha256(path) != entry.get("sample_sha256"):
        raise ArtifactValidationError(f"Artifact '{name}' does not match its manifest")
    return path


def _as_optional_text(value: str | None) -> str | None:
    value = (value or "").strip()
    return value or None


def _as_optional_float(value: str | None) -> float | None:
    value = _as_optional_text(value)
    if value is None:
        return None
    try:
        return float(value)
    except ValueError as exc:
        raise ArtifactValidationError("Company metadata contains an invalid numeric value") from exc


def _as_optional_int(value: str | None) -> int | None:
    number = _as_optional_float(value)
    return int(number) if number is not None else None


def _read_company_records(path: Path) -> tuple[CompanyRecord, ...]:
    try:
        with path.open(newline="", encoding="utf-8") as csv_file:
            reader = csv.DictReader(csv_file)
            if reader.fieldnames is None or not REQUIRED_COMPANY_COLUMNS.issubset(
                reader.fieldnames
            ):
                raise ArtifactValidationError("Company metadata schema is unsupported")
            records = tuple(
                CompanyRecord(
                    company_id=(row["id"] or "").strip(),
                    name=(row["name"] or "").strip(),
                    retrieval_text=(row["retrieval_text"] or "").strip(),
                    short_description=_as_optional_text(row.get("short_description")),
                    categories=_as_optional_text(row.get("categories")),
                    locations=_as_optional_text(row.get("locations")),
                    company_type=_as_optional_text(row.get("company_type")),
                    employee_min=_as_optional_float(row.get("employee_min")),
                    employee_max=_as_optional_float(row.get("employee_max")),
                    founded_year=_as_optional_int(row.get("founded_year")),
                    funding_total=_as_optional_float(row.get("funding_total")),
                )
                for row in reader
            )
    except (OSError, csv.Error) as exc:
        raise ArtifactValidationError("Company metadata cannot be read") from exc
    if not records or any(not row.company_id or not row.retrieval_text for row in records):
        raise ArtifactValidationError("Company metadata contains required empty fields")
    if len({row.company_id for row in records}) != len(records):
        raise ArtifactValidationError("Company metadata IDs are not unique")
    return records


def _read_failure_documents(path: Path) -> tuple[FailureDocument, ...]:
    try:
        raw_documents = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactValidationError("Historical failure documents are invalid") from exc
    if not isinstance(raw_documents, list):
        raise ArtifactValidationError("Historical failure documents schema is unsupported")
    documents: list[FailureDocument] = []
    for index, item in enumerate(raw_documents):
        if not isinstance(item, dict) or not REQUIRED_FAILURE_FIELDS.issubset(item):
            raise ArtifactValidationError("Historical failure documents schema is unsupported")
        investors = item.get("investors") or []
        if not isinstance(investors, list):
            raise ArtifactValidationError("Historical failure investors schema is unsupported")
        document = FailureDocument(
            case_index=index,
            company=str(item["company"]).strip(),
            funding=_as_optional_text(item.get("funding")),
            category=_as_optional_text(item.get("category")),
            investors=tuple(str(investor) for investor in investors),
            failure_reason=str(item["failure_reason"]).strip(),
            prompt=str(item["prompt"]).strip(),
            retrieval_text=str(item.get("retrieval_text") or "").strip(),
        )
        if not document.company or not document.failure_reason or not document.retrieval_text:
            raise ArtifactValidationError(
                "Historical failure documents contain required empty fields"
            )
        documents.append(document)
    if not documents:
        raise ArtifactValidationError("Historical failure documents are empty")
    return tuple(documents)


def _read_failure_cases(path: Path) -> list[dict[str, Any]]:
    try:
        cases = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactValidationError("Historical failure source data is invalid") from exc
    if not isinstance(cases, list) or not all(isinstance(case, dict) for case in cases):
        raise ArtifactValidationError("Historical failure source data schema is unsupported")
    for case in cases:
        if not REQUIRED_FAILURE_FIELDS.issubset(case):
            raise ArtifactValidationError("Historical failure source data schema is unsupported")
    return cases


def _read_csv_rows(
    path: Path, required_columns: frozenset[str], label: str
) -> tuple[dict[str, str], ...]:
    try:
        with path.open(newline="", encoding="utf-8") as csv_file:
            reader = csv.DictReader(csv_file)
            if reader.fieldnames is None or not required_columns.issubset(reader.fieldnames):
                raise ArtifactValidationError(f"{label} schema is unsupported")
            return tuple({key: value or "" for key, value in row.items()} for row in reader)
    except (OSError, csv.Error) as exc:
        raise ArtifactValidationError(f"{label} cannot be read") from exc


def _load_embeddings(path: Path, label: str) -> NDArray[np.floating[Any]]:
    try:
        embeddings = np.load(path, mmap_mode="r", allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise ArtifactValidationError(f"{label} embeddings are invalid") from exc
    if (
        not isinstance(embeddings, np.ndarray)
        or embeddings.ndim != 2
        or not np.issubdtype(embeddings.dtype, np.floating)
    ):
        raise ArtifactValidationError(f"{label} embeddings must be a two-dimensional float matrix")
    if embeddings.shape[0] == 0 or embeddings.shape[1] == 0:
        raise ArtifactValidationError(f"{label} embeddings are empty")
    sample = embeddings[
        np.linspace(0, embeddings.shape[0] - 1, min(32, embeddings.shape[0]), dtype=int)
    ]
    if not np.isfinite(sample).all():
        raise ArtifactValidationError(f"{label} embeddings contain invalid values")
    return embeddings


def _assert_entry_rows(manifest: dict[str, Any], name: str, rows: int) -> None:
    if manifest["artifacts"][name].get("rows") != rows:
        raise ArtifactValidationError(f"Artifact '{name}' row count does not match its manifest")


def load_data_artifacts(data_dir: Path) -> DataArtifacts:
    """Load once at startup, checking provenance and alignment without a full RAM copy."""
    manifest = _read_manifest(data_dir)
    paths = {
        name: _artifact_path(data_dir, manifest, name)
        for name in (
            "company_metadata",
            "company_embeddings",
            "failure_documents",
            "failure_embeddings",
            "failure_cases",
            "idx_benchmark_overall",
            "idx_benchmark_by_period",
            "idx_ratio_benchmark",
        )
    }
    company_records = _read_company_records(paths["company_metadata"])
    company_embeddings = _load_embeddings(paths["company_embeddings"], "Company")
    failure_documents = _read_failure_documents(paths["failure_documents"])
    failure_cases = _read_failure_cases(paths["failure_cases"])
    failure_embeddings = _load_embeddings(paths["failure_embeddings"], "Historical failure")
    idx_overall = _read_csv_rows(
        paths["idx_benchmark_overall"], REQUIRED_BENCHMARK_COLUMNS, "IDX overall benchmark"
    )
    idx_period = _read_csv_rows(
        paths["idx_benchmark_by_period"], REQUIRED_BENCHMARK_COLUMNS, "IDX period benchmark"
    )
    idx_ratio = _read_csv_rows(
        paths["idx_ratio_benchmark"], frozenset({"ticker"}), "IDX ratio benchmark"
    )

    for name, rows in (
        ("company_metadata", len(company_records)),
        ("company_embeddings", int(company_embeddings.shape[0])),
        ("failure_documents", len(failure_documents)),
        ("failure_embeddings", int(failure_embeddings.shape[0])),
        ("failure_cases", len(failure_cases)),
        ("idx_benchmark_overall", len(idx_overall)),
        ("idx_benchmark_by_period", len(idx_period)),
        ("idx_ratio_benchmark", len(idx_ratio)),
    ):
        _assert_entry_rows(manifest, name, rows)

    if company_embeddings.shape[0] != len(company_records):
        raise ArtifactValidationError("Company metadata and embeddings are not aligned")
    if failure_embeddings.shape[0] != len(failure_documents):
        raise ArtifactValidationError("Historical failure documents and embeddings are not aligned")
    if len(failure_cases) != len(failure_documents):
        raise ArtifactValidationError(
            "Historical failure source data and documents are not aligned"
        )
    for case, document in zip(failure_cases, failure_documents, strict=True):
        if (
            str(case["company"]).strip() != document.company
            or str(case["failure_reason"]).strip() != document.failure_reason
            or build_failure_retrieval_text(case) != document.retrieval_text
        ):
            raise ArtifactValidationError(
                "Historical failure source data and documents are not aligned"
            )
    if company_embeddings.shape[1] != failure_embeddings.shape[1]:
        raise ArtifactValidationError("Company and historical failure embedding dimensions differ")

    embedding = manifest.get("embedding_model")
    if not isinstance(embedding, dict) or embedding.get("dimension") != int(
        company_embeddings.shape[1]
    ):
        raise ArtifactValidationError("Embedding model dimension does not match artifacts")
    if embedding.get("metric") != "dot_product" or embedding.get("normalization") != "l2":
        raise ArtifactValidationError("Embedding metric or normalization is unsupported")
    if manifest.get("metadata_stable_ids_sha256") != ordered_values_sha256(
        record.company_id for record in company_records
    ):
        raise ArtifactValidationError("Company metadata order does not match its manifest")

    return DataArtifacts(
        manifest=manifest,
        company_records=company_records,
        company_embeddings=company_embeddings,
        failure_documents=failure_documents,
        failure_embeddings=failure_embeddings,
        idx_benchmark_overall=idx_overall,
        idx_benchmark_by_period=idx_period,
        idx_ratio_benchmark=idx_ratio,
    )
