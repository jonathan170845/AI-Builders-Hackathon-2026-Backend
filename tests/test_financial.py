from __future__ import annotations

from decimal import Decimal

import pytest

from app.schemas.analysis import FinancialInputs, FinancialWarningCode
from app.services.financial import (
    benchmark_against_idx,
    build_financial_evidence,
    run_financial_stress_test,
)


def make_inputs(**overrides: int | float) -> FinancialInputs:
    values: dict[str, int | float] = {
        "monthly_orders": 1_000_000,
        "revenue_per_order": 50,
        "variable_cost_per_order": 35,
        "promo_subsidy": 10,
        "delivery_cost": 8,
        "fixed_cost": 5_000_000,
        "driver_cost": 3_000_000,
        "cash_balance": 500_000_000,
    }
    values.update(overrides)
    return FinancialInputs(**values)


def test_golden_financial_stress_test_matches_documented_formula():
    stress_test = run_financial_stress_test(make_inputs())

    assert stress_test.to_api_result() == {
        "contribution_margin": -3_000_000.0,
        "contribution_margin_pct": -6.0,
        "operating_profit": -11_000_000.0,
        "monthly_burn": 11_000_000.0,
        "runway_months": 45.4545,
        "break_even_orders": None,
    }
    assert stress_test.warnings == (
        FinancialWarningCode.NEGATIVE_CONTRIBUTION_MARGIN,
        FinancialWarningCode.NEGATIVE_OPERATING_PROFIT,
        FinancialWarningCode.BREAK_EVEN_UNREACHABLE,
    )


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        (
            {
                "monthly_orders": 0,
                "revenue_per_order": 0,
                "variable_cost_per_order": 0,
                "promo_subsidy": 0,
                "delivery_cost": 0,
                "fixed_cost": 0,
                "driver_cost": 0,
                "cash_balance": 0,
            },
            {"runway_months": None, "break_even_orders": None},
        ),
        (
            {"promo_subsidy": 60},
            {"runway_months": Decimal("8.196721311475409836065573770"), "break_even_orders": None},
        ),
        (
            {
                "revenue_per_order": 50,
                "variable_cost_per_order": 40,
                "promo_subsidy": 0,
                "delivery_cost": 10,
            },
            {"runway_months": Decimal("62.5"), "break_even_orders": None},
        ),
        (
            {
                "revenue_per_order": 100,
                "variable_cost_per_order": 20,
                "promo_subsidy": 0,
                "delivery_cost": 0,
            },
            {"runway_months": None, "break_even_orders": 100_000},
        ),
        (
            {"cash_balance": 0},
            {"runway_months": Decimal("0"), "break_even_orders": None},
        ),
    ],
)
def test_financial_edge_cases(overrides, expected):
    stress_test = run_financial_stress_test(make_inputs(**overrides))

    if expected["runway_months"] is None:
        assert stress_test.runway_months is None
    else:
        assert abs(stress_test.runway_months - expected["runway_months"]) < Decimal("1e-24")
    assert stress_test.break_even_orders == expected["break_even_orders"]
    assert all(
        value is None or isinstance(value, (float, int))
        for value in stress_test.to_api_result().values()
    )


def test_decimal_rounding_and_low_runway_warning_are_deterministic():
    stress_test = run_financial_stress_test(
        make_inputs(
            monthly_orders=1,
            revenue_per_order=1.005,
            variable_cost_per_order=0,
            promo_subsidy=0,
            delivery_cost=0,
            fixed_cost=1.505,
            driver_cost=0,
            cash_balance=1,
        ),
        low_runway_months=Decimal("2"),
    )

    assert stress_test.to_api_result()["contribution_margin"] == 1.01
    assert stress_test.to_api_result()["runway_months"] == 2.0
    assert FinancialWarningCode.LOW_RUNWAY not in stress_test.warnings


def test_idx_benchmark_is_directional_and_skips_incompatible_metrics():
    stress_test = run_financial_stress_test(
        make_inputs(
            monthly_orders=1,
            revenue_per_order=100,
            variable_cost_per_order=20,
            promo_subsidy=0,
            delivery_cost=0,
            fixed_cost=0,
            driver_cost=0,
        )
    )
    benchmarks = benchmark_against_idx(
        stress_test,
        [
            {"ratio": "gross_margin", "count": "12", "p25": "0.2", "median": "0.4", "p75": "0.6"},
            {"ratio": "debt_to_assets", "count": "12", "p25": "0.2", "median": "0.4", "p75": "0.6"},
        ],
    )

    assert len(benchmarks) == 1
    benchmark = benchmarks[0]
    assert benchmark.status == "Above Benchmark"
    assert benchmark.value == "80.0000%"
    assert benchmark.comparison_type == "directional"
    assert "not identical" in benchmark.disclaimer


def test_idx_benchmark_is_not_created_for_nonpositive_net_revenue_or_bad_rows():
    nonpositive_revenue = run_financial_stress_test(make_inputs(promo_subsidy=50))

    assert benchmark_against_idx(nonpositive_revenue, []) == ()
    assert benchmark_against_idx(run_financial_stress_test(make_inputs()), []) == ()


def test_missing_financial_inputs_returns_explicitly_empty_financial_evidence():
    evidence = build_financial_evidence(None, [])

    assert evidence.financial_results is None
    assert evidence.financial_warnings == ()
    assert evidence.idx_benchmarks == ()


def test_financial_margin_uses_gross_revenue_as_denominator():
    stress_test = run_financial_stress_test(
        make_inputs(
            monthly_orders=3600,
            revenue_per_order=25,
            variable_cost_per_order=10,
            promo_subsidy=3,
            delivery_cost=2,
            fixed_cost=18000,
            driver_cost=0,
            cash_balance=90000,
        )
    )

    result = stress_test.to_api_result()

    assert result["contribution_margin"] == 36000.0
    assert result["contribution_margin_pct"] == 40.0
    assert result["operating_profit"] == 18000.0
    assert result["monthly_burn"] == 0.0
    assert result["runway_months"] is None
    assert result["break_even_orders"] == 1800