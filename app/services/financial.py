"""Deterministic financial stress-test and directional IDX comparison.

All calculation is performed with ``Decimal``.  Conversion to JSON-safe floats is
kept in the mapper at the API boundary so the service remains usable without
FastAPI, a database, an LLM, or NumPy.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from decimal import ROUND_CEILING, ROUND_HALF_UP, Decimal, InvalidOperation

from app.schemas.analysis import FinancialInputs, FinancialWarningCode, IdxBenchmark

MONEY_PLACES = Decimal("0.01")
PERCENTAGE_PLACES = Decimal("0.0001")
DEFAULT_LOW_RUNWAY_MONTHS = Decimal("6")
IDX_DIRECTIONAL_DISCLAIMER = (
    "Contribution margin is not identical to accounting gross margin; this IDX comparison is "
    "directional only."
)


@dataclass(frozen=True, slots=True)
class FinancialStressTest:
    """Decimal-valued calculation result before it crosses the JSON boundary."""

    contribution_margin: Decimal
    contribution_margin_pct: Decimal
    operating_profit: Decimal
    monthly_burn: Decimal
    runway_months: Decimal | None
    break_even_orders: int | None
    net_revenue_per_order: Decimal
    contribution_margin_per_order: Decimal
    warnings: tuple[FinancialWarningCode, ...]

    def to_api_result(self) -> dict[str, float | int | None]:
        """Round and convert values to finite JSON-native primitives."""
        return {
            "contribution_margin": _money_to_float(self.contribution_margin),
            "contribution_margin_pct": _percentage_to_float(self.contribution_margin_pct),
            "operating_profit": _money_to_float(self.operating_profit),
            "monthly_burn": _money_to_float(self.monthly_burn),
            "runway_months": (
                _percentage_to_float(self.runway_months) if self.runway_months is not None else None
            ),
            "break_even_orders": self.break_even_orders,
        }


@dataclass(frozen=True, slots=True)
class FinancialEvidence:
    """Complete financial section of an analysis result, before API serialization."""

    financial_results: dict[str, float | int | None] | None
    financial_warnings: tuple[FinancialWarningCode, ...]
    idx_benchmarks: tuple[IdxBenchmark, ...]


def run_financial_stress_test(
    inputs: FinancialInputs,
    *,
    low_runway_months: Decimal = DEFAULT_LOW_RUNWAY_MONTHS,
) -> FinancialStressTest:
    """Calculate the financial stress test from the public request schema.

    Monetary outputs use two decimal places and percentages (including runway in
    months) use four places, both with ``ROUND_HALF_UP``.  The break-even ceiling
    is calculated before monetary display rounding.
    """
    if low_runway_months < 0:
        raise ValueError("low_runway_months must be non-negative")

    monthly_orders = Decimal(inputs.monthly_orders)
    revenue_per_order = _decimal_from_input(inputs.revenue_per_order)
    variable_cost_per_order = _decimal_from_input(inputs.variable_cost_per_order)
    promo_subsidy = _decimal_from_input(inputs.promo_subsidy)
    delivery_cost = _decimal_from_input(inputs.delivery_cost)
    fixed_cost = _decimal_from_input(inputs.fixed_cost)
    driver_cost = _decimal_from_input(inputs.driver_cost)
    cash_balance = _decimal_from_input(inputs.cash_balance)

    net_revenue_per_order = revenue_per_order - promo_subsidy
    variable_cost_per_order_total = variable_cost_per_order + delivery_cost
    contribution_margin_per_order = net_revenue_per_order - variable_cost_per_order_total
    contribution_margin = contribution_margin_per_order * monthly_orders
    operating_profit = contribution_margin - fixed_cost - driver_cost
    monthly_burn = -operating_profit if operating_profit < 0 else Decimal("0")
    runway_months = cash_balance / monthly_burn if monthly_burn > 0 else None
    break_even_orders = (
        int(
            ((fixed_cost + driver_cost) / contribution_margin_per_order).to_integral_value(
                rounding=ROUND_CEILING
            )
        )
        if contribution_margin_per_order > 0
        else None
    )
    contribution_margin_pct = (
        contribution_margin_per_order / net_revenue_per_order * Decimal("100")
        if net_revenue_per_order > 0
        else Decimal("0")
    )

    warnings: list[FinancialWarningCode] = []
    if net_revenue_per_order <= 0:
        warnings.append(FinancialWarningCode.NON_POSITIVE_NET_REVENUE)
    if contribution_margin_per_order < 0:
        warnings.append(FinancialWarningCode.NEGATIVE_CONTRIBUTION_MARGIN)
    if operating_profit < 0:
        warnings.append(FinancialWarningCode.NEGATIVE_OPERATING_PROFIT)
    if runway_months is not None and runway_months < low_runway_months:
        warnings.append(FinancialWarningCode.LOW_RUNWAY)
    if break_even_orders is None:
        warnings.append(FinancialWarningCode.BREAK_EVEN_UNREACHABLE)

    return FinancialStressTest(
        contribution_margin=contribution_margin,
        contribution_margin_pct=contribution_margin_pct,
        operating_profit=operating_profit,
        monthly_burn=monthly_burn,
        runway_months=runway_months,
        break_even_orders=break_even_orders,
        net_revenue_per_order=net_revenue_per_order,
        contribution_margin_per_order=contribution_margin_per_order,
        warnings=tuple(warnings),
    )


def build_financial_evidence(
    inputs: FinancialInputs | None,
    benchmark_rows: Iterable[dict[str, str]],
    *,
    low_runway_months: Decimal = DEFAULT_LOW_RUNWAY_MONTHS,
) -> FinancialEvidence:
    """Build the complete financial response portion for an optional input object."""
    if inputs is None:
        return FinancialEvidence(
            financial_results=None,
            financial_warnings=(),
            idx_benchmarks=(),
        )
    stress_test = run_financial_stress_test(inputs, low_runway_months=low_runway_months)
    return FinancialEvidence(
        financial_results=stress_test.to_api_result(),
        financial_warnings=stress_test.warnings,
        idx_benchmarks=benchmark_against_idx(stress_test, benchmark_rows),
    )


def benchmark_against_idx(
    stress_test: FinancialStressTest, benchmark_rows: Iterable[dict[str, str]]
) -> tuple[IdxBenchmark, ...]:
    """Compare the unit-economics margin to IDX gross margin directionally.

    Cash, debt, and operating-cash-flow ratios have no compatible request input,
    so this deliberately returns no invented comparison for them.  A non-positive
    net revenue also has no meaningful gross-margin analogue.
    """
    if stress_test.net_revenue_per_order <= 0:
        return ()

    rows = [row for row in benchmark_rows if row.get("ratio") == "gross_margin"]
    if len(rows) != 1:
        return ()
    row = rows[0]
    try:
        p25 = _decimal_from_text(row["p25"])
        median = _decimal_from_text(row["median"])
        p75 = _decimal_from_text(row["p75"])
        sample_size = int(row["count"])
    except (KeyError, ValueError):
        return ()
    if sample_size <= 0 or not p25 <= median <= p75:
        return ()

    margin_ratio = stress_test.contribution_margin_per_order / stress_test.net_revenue_per_order
    if margin_ratio < p25:
        position = "Below Benchmark"
    elif margin_ratio <= p75:
        position = "At Benchmark"
    else:
        position = "Above Benchmark"

    return (
        IdxBenchmark(
            metric="Contribution margin ratio (directional)",
            value=_format_percent(margin_ratio),
            benchmark=(
                f"IDX gross margin: P25 {_format_percent(p25)}, median "
                f"{_format_percent(median)}, P75 {_format_percent(p75)}"
            ),
            status=position,
            comparison_type="directional",
            benchmark_metric="gross_margin",
            sample_size=sample_size,
            disclaimer=IDX_DIRECTIONAL_DISCLAIMER,
        ),
    )


def _decimal_from_input(value: float) -> Decimal:
    return _decimal_from_text(str(value))


def _decimal_from_text(value: str) -> Decimal:
    try:
        decimal_value = Decimal(value)
    except (InvalidOperation, ValueError) as exc:
        raise ValueError("value must be a finite decimal") from exc
    if not decimal_value.is_finite():
        raise ValueError("value must be a finite decimal")
    return decimal_value


def _money_to_float(value: Decimal) -> float:
    return _finite_float(value.quantize(MONEY_PLACES, rounding=ROUND_HALF_UP))


def _percentage_to_float(value: Decimal) -> float:
    return _finite_float(value.quantize(PERCENTAGE_PLACES, rounding=ROUND_HALF_UP))


def _finite_float(value: Decimal) -> float:
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError("financial result is outside the finite JSON range")
    return converted


def _format_percent(value: Decimal) -> str:
    return f"{_percentage_to_float(value * Decimal('100')):.4f}%"
