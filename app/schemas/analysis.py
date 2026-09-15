from __future__ import annotations

import math
from datetime import datetime
from decimal import Decimal
from enum import StrEnum
from typing import Annotated, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator
from pydantic.alias_generators import to_camel


class ApiModel(BaseModel):
    model_config = ConfigDict(
        alias_generator=to_camel,
        populate_by_name=True,
        extra="forbid",
        allow_inf_nan=False,
        json_schema_serialization_defaults_required=True,
    )


MAX_MONTHLY_ORDERS = 10**12
MAX_MONEY_VALUE = 10**15
MAX_MONEY_DECIMAL_PLACES = 4

MonthlyOrders = Annotated[int, Field(ge=0, le=MAX_MONTHLY_ORDERS)]
MoneyInput = Annotated[float, Field(ge=0, le=MAX_MONEY_VALUE)]
AnalysisId = UUID


class FinancialInputs(ApiModel):
    monthly_orders: MonthlyOrders
    revenue_per_order: MoneyInput
    variable_cost_per_order: MoneyInput
    promo_subsidy: MoneyInput
    delivery_cost: MoneyInput
    fixed_cost: MoneyInput
    driver_cost: MoneyInput
    cash_balance: MoneyInput

    @field_validator(
        "revenue_per_order",
        "variable_cost_per_order",
        "promo_subsidy",
        "delivery_cost",
        "fixed_cost",
        "driver_cost",
        "cash_balance",
    )
    @classmethod
    def values_must_be_finite(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("must be a finite number")
        if Decimal(str(value)).as_tuple().exponent < -MAX_MONEY_DECIMAL_PLACES:
            raise ValueError(f"must have at most {MAX_MONEY_DECIMAL_PLACES} decimal places")
        return value


class CreateAnalysisRequest(ApiModel):
    decision: str = Field(
        ...,
        min_length=10,
        max_length=5_000,
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

    @field_validator("decision", mode="before")
    @classmethod
    def normalize_decision(cls, value: object) -> object:
        return value.strip() if isinstance(value, str) else value


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
    id: AnalysisId
    status: Literal[AnalysisStatus.QUEUED] = AnalysisStatus.QUEUED
    created_at: datetime


class AnalysisPending(ApiModel):
    id: AnalysisId
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

EvidenceDirection = Literal[
    "supports",
    "contradicts",
    "mixed",
    "neutral",
]


class EvidenceStrengthBreakdown(ApiModel):
    directness: float = Field(ge=0.0, le=2.0)
    decision_specificity: float = Field(ge=0.0, le=2.0)
    observed_evidence: float = Field(ge=0.0, le=2.0)
    comparability: float = Field(ge=0.0, le=1.5)
    coverage: float = Field(ge=0.0, le=1.5)
    source_quality: float = Field(ge=0.0, le=1.0)


class FinancialWarningCode(StrEnum):
    NEGATIVE_CONTRIBUTION_MARGIN = "NEGATIVE_CONTRIBUTION_MARGIN"
    NEGATIVE_OPERATING_PROFIT = "NEGATIVE_OPERATING_PROFIT"
    NON_POSITIVE_NET_REVENUE = "NON_POSITIVE_NET_REVENUE"
    LOW_RUNWAY = "LOW_RUNWAY"
    BREAK_EVEN_UNREACHABLE = "BREAK_EVEN_UNREACHABLE"


class Assumption(ApiModel):
    id: str
    text: str

    assessment: AssessmentLevel
    summary: str

    evidence_strength_score: float | None = Field(
        default=None,
        ge=0.0,
        le=10.0,
    )

    evidence_direction: EvidenceDirection | None = None

    score_breakdown: EvidenceStrengthBreakdown | None = None

    user_evidence_found: bool = False

    score_reasoning: list[str] = Field(
        default_factory=list
    )

    evidence_gaps: list[str]

    validation_experiment: str

    evidence_refs: list[str] = Field(
        default_factory=list
    )


class FailureMechanism(ApiModel):
    id: str
    title: str
    description: str
    source: str
    year: int | None


class CompanyAnalogue(ApiModel):
    id: str
    name: str
    context: str
    outcome: Literal["Failed", "Pivoted", "Succeeded"] | None
    year_range: str | None
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
    year: int | None
    url: str | None


class IdxBenchmark(ApiModel):
    metric: str
    value: str
    benchmark: str
    status: Literal["Below Benchmark", "At Benchmark", "Above Benchmark"]
    comparison_type: Literal["directional"]
    benchmark_metric: Literal["gross_margin"]
    sample_size: int = Field(ge=1)
    disclaimer: str


class AnalysisResult(ApiModel):
    id: AnalysisId
    decision_title: str
    summary: str
    assumptions: list[Assumption]
    direct_evidence_coverage: str
    financial_warnings: list[FinancialWarningCode]
    financial_results: FinancialResults | None
    failure_mechanisms: list[FailureMechanism]
    company_analogues: list[CompanyAnalogue]
    validation_experiments: list[ValidationExperiment]
    idx_benchmarks: list[IdxBenchmark]
    sources: list[SourceItem]
    date: str
    assumption_count: int
    financial_warning_count: int


class AnalysisCompleted(ApiModel):
    id: AnalysisId
    status: Literal[AnalysisStatus.COMPLETED] = AnalysisStatus.COMPLETED
    stage: None = None
    created_at: datetime
    updated_at: datetime
    result: AnalysisResult


class AnalysisJobError(ApiModel):
    code: str
    message: str


class AnalysisFailed(ApiModel):
    id: AnalysisId
    status: Literal[AnalysisStatus.FAILED] = AnalysisStatus.FAILED
    stage: AnalysisStage
    created_at: datetime
    updated_at: datetime
    error: AnalysisJobError


class AnalysisHistoryItem(ApiModel):
    id: AnalysisId
    decision: str
    status: AnalysisStatus
    created_at: datetime
    updated_at: datetime
    assumption_count: int = Field(ge=0)
    financial_warning_count: int = Field(ge=0)


class AnalysisHistory(ApiModel):
    items: list[AnalysisHistoryItem]
    next_cursor: str | None


AnalysisDetail = Annotated[
    AnalysisPending | AnalysisCompleted | AnalysisFailed,
    Field(discriminator="status"),
]
