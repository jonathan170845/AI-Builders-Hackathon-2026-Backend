from __future__ import annotations

import math
from datetime import datetime
from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator
from pydantic.alias_generators import to_camel


class ApiModel(BaseModel):
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True, extra="forbid")


FiniteNonNegative = Annotated[float, Field(ge=0)]


class FinancialInputs(ApiModel):
    monthly_orders: FiniteNonNegative
    revenue_per_order: FiniteNonNegative
    variable_cost_per_order: FiniteNonNegative
    promo_subsidy: FiniteNonNegative
    delivery_cost: FiniteNonNegative
    fixed_cost: FiniteNonNegative
    driver_cost: FiniteNonNegative
    cash_balance: FiniteNonNegative

    @field_validator("*")
    @classmethod
    def values_must_be_finite(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("must be a finite number")
        return value


class CreateAnalysisRequest(ApiModel):
    decision: str = Field(
        ...,
        examples=["Scale on-demand grocery delivery to 5 new cities"],
    )
    financial_inputs: FinancialInputs | None = Field(
        default=None,
        examples=[
            {
                "monthlyOrders": 1_000_000,
                "revenuePerOrder": 50,
                "variableCostPerOrder": 35,
                "promoSubsidy": 10,
                "deliveryCost": 8,
                "fixedCost": 5_000_000,
                "driverCost": 3_000_000,
                "cashBalance": 500_000_000,
            }
        ],
    )

    @field_validator("decision")
    @classmethod
    def normalize_decision(cls, value: str) -> str:
        normalized = value.strip()
        if not 10 <= len(normalized) <= 5_000:
            raise ValueError("must contain between 10 and 5000 characters after trimming")
        return normalized


class AnalysisStatus(StrEnum):
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class AnalysisStage(StrEnum):
    QUEUED = "queued"
    EXTRACTING_ASSUMPTIONS = "extracting_assumptions"
    RETRIEVING_EVIDENCE = "retrieving_evidence"
    CALCULATING_FINANCIALS = "calculating_financials"
    REVIEWING_EVIDENCE = "reviewing_evidence"
    BUILDING_REPORT = "building_report"


class AnalysisAccepted(ApiModel):
    id: str
    status: Literal[AnalysisStatus.QUEUED] = AnalysisStatus.QUEUED
    created_at: datetime


class AnalysisPending(ApiModel):
    id: str
    status: Literal[AnalysisStatus.QUEUED, AnalysisStatus.PROCESSING]
    stage: AnalysisStage
    created_at: datetime
    updated_at: datetime


class FinancialResults(ApiModel):
    contribution_margin: float
    contribution_margin_pct: float
    operating_profit: float
    monthly_burn: float
    runway_months: float | None
    break_even_orders: int | None


AssessmentLevel = Literal[
    "Insufficient Evidence", "Partially Supported", "Well Supported", "Contradicted"
]


class Assumption(ApiModel):
    id: str
    text: str
    assessment: AssessmentLevel
    summary: str
    evidence_gaps: list[str]
    validation_experiment: str


class FailureMechanism(ApiModel):
    id: str
    title: str
    description: str
    source: str
    year: int


class CompanyAnalogue(ApiModel):
    id: str
    name: str
    context: str
    outcome: Literal["Failed", "Pivoted", "Succeeded"]
    year_range: str
    relevance: str


class ValidationExperiment(ApiModel):
    id: str
    title: str
    description: str
    linked_assumption_id: str
    time_to_run: str
    cost_level: Literal["Low", "Medium", "High"]


class SourceItem(ApiModel):
    id: str
    title: str
    type: Literal["Academic", "Industry Report", "News", "Case Study", "Financial Filing"]
    year: int
    url: str


class IdxBenchmark(ApiModel):
    metric: str
    value: str
    benchmark: str
    status: Literal["Below Benchmark", "At Benchmark", "Above Benchmark"]


class AnalysisResult(ApiModel):
    id: str
    decision_title: str
    summary: str
    assumptions: list[Assumption]
    direct_evidence_coverage: str
    financial_warnings: list[str]
    financial_results: FinancialResults
    failure_mechanisms: list[FailureMechanism]
    company_analogues: list[CompanyAnalogue]
    validation_experiments: list[ValidationExperiment]
    idx_benchmarks: list[IdxBenchmark]
    sources: list[SourceItem]
    date: str
    status: Literal["Completed"] = "Completed"
    assumption_count: int
    financial_warning_count: int


class AnalysisCompleted(ApiModel):
    id: str
    status: Literal[AnalysisStatus.COMPLETED] = AnalysisStatus.COMPLETED
    created_at: datetime
    updated_at: datetime
    result: AnalysisResult


class AnalysisFailed(ApiModel):
    id: str
    status: Literal[AnalysisStatus.FAILED] = AnalysisStatus.FAILED
    created_at: datetime
    updated_at: datetime
    message: str


AnalysisDetail = Annotated[
    AnalysisPending | AnalysisCompleted | AnalysisFailed,
    Field(discriminator="status"),
]
