from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from app.services.artifacts import CompanyRecord, DataArtifacts, FailureDocument

DEFAULT_MAX_TOP_K = 10


class RetrievalError(ValueError):
    """Raised for invalid retrieval input or an incompatible encoder."""


class QueryEncoder(Protocol):
    def encode(self, texts: Sequence[str]) -> NDArray[np.floating]: ...

    def dimension(self) -> int: ...


@dataclass(frozen=True, slots=True)
class CompanyAnalogue:
    rank: int
    company_id: str
    name: str
    semantic_score: float
    short_description: str | None
    categories: str | None
    locations: str | None
    company_type: str | None
    employee_min: float | None
    employee_max: float | None
    founded_year: int | None
    funding_total: float | None

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class HistoricalFailure:
    rank: int
    case_index: int
    company: str
    semantic_score: float
    funding: str | None
    category: str | None
    investors: tuple[str, ...]
    failure_reason: str
    prompt: str

    def as_dict(self) -> dict[str, object]:
        value = asdict(self)
        value["investors"] = list(self.investors)
        return value


@dataclass(frozen=True, slots=True)
class AssumptionEvidence:
    assumption_id: str
    assumption: str
    retrieval_query: str
    company_analogues: tuple[CompanyAnalogue, ...]
    historical_failures: tuple[HistoricalFailure, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "assumption_id": self.assumption_id,
            "assumption": self.assumption,
            "retrieval_query": self.retrieval_query,
            "company_analogue_count": len(self.company_analogues),
            "company_analogues": [item.as_dict() for item in self.company_analogues],
            "historical_failure_count": len(self.historical_failures),
            "historical_failures": [item.as_dict() for item in self.historical_failures],
        }


class SentenceTransformerEncoder:
    """Adapter that keeps the optional model dependency out of DTOs and tests."""

    def __init__(self, model_path_or_id: str, revision: str | None) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - packaging/runtime guard
            raise RetrievalError("sentence-transformers is not installed") from exc
        try:
            self._model = SentenceTransformer(
                model_path_or_id,
                revision=revision,
                model_kwargs={"trust_remote_code": False},
            )
        except Exception as exc:  # model errors must not leak filesystem/cache locations
            raise RetrievalError("Embedding model could not be loaded") from exc

    def dimension(self) -> int:
        dimension = self._model.get_sentence_embedding_dimension()
        if not isinstance(dimension, int):
            raise RetrievalError("Embedding model does not report a valid dimension")
        return dimension

    def encode(self, texts: Sequence[str]) -> NDArray[np.floating]:
        encoded = self._model.encode(
            list(texts), convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False
        )
        return np.asarray(encoded)


def _normalized_query(query: object) -> str:
    if not isinstance(query, str):
        raise RetrievalError("Retrieval query must be a string")
    normalized = " ".join(query.split())
    if not normalized:
        raise RetrievalError("Retrieval query must not be empty")
    return normalized


def _validate_top_k(top_k: int, *, maximum: int) -> int:
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1 or top_k > maximum:
        raise RetrievalError(f"top_k must be an integer between 1 and {maximum}")
    return top_k


def _rank_indices(scores: NDArray[np.floating], top_k: int) -> NDArray[np.intp]:
    """Order equal similarities by original row index for reproducible evidence."""
    indices = np.arange(scores.shape[0])
    order = np.lexsort((indices, -scores))
    return order[:top_k]


class RetrievalService:
    def __init__(
        self, artifacts: DataArtifacts, encoder: QueryEncoder, *, max_top_k: int = DEFAULT_MAX_TOP_K
    ) -> None:
        if encoder.dimension() != artifacts.embedding_dimension:
            raise RetrievalError("Embedding model dimension does not match prepared artifacts")
        self._artifacts = artifacts
        self._encoder = encoder
        self._max_top_k = max_top_k

    def _encode_query(self, query: object) -> tuple[str, NDArray[np.floating]]:
        normalized = _normalized_query(query)
        vector = np.asarray(self._encoder.encode([normalized]))
        if vector.shape != (1, self._artifacts.embedding_dimension):
            raise RetrievalError("Query embedding dimension does not match prepared artifacts")
        if not np.isfinite(vector).all():
            raise RetrievalError("Query embedding contains invalid values")
        norm = float(np.linalg.norm(vector[0]))
        if norm == 0:
            raise RetrievalError("Query embedding must not be zero")
        return normalized, vector[0] / norm

    def company_analogues(self, query: object, *, top_k: int = 5) -> tuple[CompanyAnalogue, ...]:
        top_k = min(
            _validate_top_k(top_k, maximum=self._max_top_k), len(self._artifacts.company_records)
        )
        _, vector = self._encode_query(query)
        scores = np.asarray(self._artifacts.company_embeddings @ vector)
        return tuple(
            _company_result(rank, self._artifacts.company_records[index], float(scores[index]))
            for rank, index in enumerate(_rank_indices(scores, top_k), start=1)
        )

    def historical_failures(
        self, query: object, *, top_k: int = 5
    ) -> tuple[HistoricalFailure, ...]:
        top_k = min(
            _validate_top_k(top_k, maximum=self._max_top_k), len(self._artifacts.failure_documents)
        )
        _, vector = self._encode_query(query)
        scores = np.asarray(self._artifacts.failure_embeddings @ vector)
        return tuple(
            _failure_result(rank, self._artifacts.failure_documents[index], float(scores[index]))
            for rank, index in enumerate(_rank_indices(scores, top_k), start=1)
        )

    def retrieve_assumptions(
        self,
        assumptions: Iterable[object],
        *,
        top_company_analogues: int = 5,
        top_failure_cases: int = 5,
    ) -> tuple[AssumptionEvidence, ...]:
        evidence: list[AssumptionEvidence] = []
        for position, raw_assumption in enumerate(assumptions, start=1):
            assumption_id, assumption, query = _normalize_assumption(raw_assumption, position)
            evidence.append(
                AssumptionEvidence(
                    assumption_id=assumption_id,
                    assumption=assumption,
                    retrieval_query=query,
                    company_analogues=self.company_analogues(query, top_k=top_company_analogues),
                    historical_failures=self.historical_failures(query, top_k=top_failure_cases),
                )
            )
        return tuple(evidence)


def _company_result(rank: int, record: CompanyRecord, score: float) -> CompanyAnalogue:
    return CompanyAnalogue(
        rank=rank,
        company_id=record.company_id,
        name=record.name,
        semantic_score=round(score, 6),
        short_description=record.short_description,
        categories=record.categories,
        locations=record.locations,
        company_type=record.company_type,
        employee_min=record.employee_min,
        employee_max=record.employee_max,
        founded_year=record.founded_year,
        funding_total=record.funding_total,
    )


def _failure_result(rank: int, document: FailureDocument, score: float) -> HistoricalFailure:
    return HistoricalFailure(
        rank=rank,
        case_index=document.case_index,
        company=document.company,
        semantic_score=round(score, 6),
        funding=document.funding,
        category=document.category,
        investors=document.investors,
        failure_reason=document.failure_reason,
        prompt=document.prompt,
    )


def _normalize_assumption(raw: object, position: int) -> tuple[str, str, str]:
    if isinstance(raw, dict):
        assumption_id = str(raw.get("id") or raw.get("assumption_id") or f"A{position}").strip()
        assumption = str(raw.get("assumption") or raw.get("text") or "").strip()
        query = str(
            raw.get("retrieval_query")
            or raw.get("semantic_query")
            or raw.get("search_query")
            or assumption
        ).strip()
    else:
        assumption_id = f"A{position}"
        assumption = str(raw).strip()
        query = assumption
    if not assumption_id or not assumption:
        raise RetrievalError("Each assumption needs an ID and non-empty text")
    return assumption_id, assumption, _normalized_query(query)
