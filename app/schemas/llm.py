"""Strict, internal contracts for every structured LLM response."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.schemas.analysis import AssessmentLevel


class LLMStructuredModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class ExtractedAssumption(LLMStructuredModel):
    id: str = Field(pattern=r"^A[1-5]$")
    text: str = Field(min_length=10, max_length=1_000)
    category: Literal["demand", "unit_economics", "operations", "market", "regulatory"]


class AssumptionExtraction(LLMStructuredModel):
    assumptions: list[ExtractedAssumption] = Field(min_length=3, max_length=5)

    @field_validator("assumptions")
    @classmethod
    def ids_must_be_unique(cls, values: list[ExtractedAssumption]) -> list[ExtractedAssumption]:
        if len({item.id for item in values}) != len(values):
            raise ValueError("assumption IDs must be unique")
        return values


class RetrievalQuery(LLMStructuredModel):
    assumption_id: str = Field(pattern=r"^A[1-5]$")
    query: str = Field(min_length=3, max_length=1_000)


class RetrievalQueryBatch(LLMStructuredModel):
    queries: list[RetrievalQuery] = Field(min_length=3, max_length=5)

    @field_validator("queries")
    @classmethod
    def ids_must_be_unique(cls, values: list[RetrievalQuery]) -> list[RetrievalQuery]:
        if len({item.assumption_id for item in values}) != len(values):
            raise ValueError("retrieval-query assumption IDs must be unique")
        return values


class EvidenceReference(LLMStructuredModel):
    """Only historical cases can support a claim; company analogues are context-only."""

    kind: Literal["historical_failure"]
    id: str = Field(min_length=1, max_length=200)


class GroundedEvaluation(LLMStructuredModel):
    assumption_id: str = Field(pattern=r"^A[1-5]$")
    assessment: AssessmentLevel
    summary: str = Field(min_length=1, max_length=2_000)
    evidence_refs: list[EvidenceReference] = Field(default_factory=list, max_length=10)
    evidence_gaps: list[str] = Field(default_factory=list, max_length=10)


class GroundingReview(LLMStructuredModel):
    assumption_id: str = Field(pattern=r"^A[1-5]$")
    disposition: Literal["keep", "weaken", "discard"]
    summary: str = Field(min_length=1, max_length=2_000)
    evidence_refs: list[EvidenceReference] = Field(default_factory=list, max_length=10)
