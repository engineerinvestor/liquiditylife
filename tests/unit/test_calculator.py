"""Tests for the asset allocation calculator."""

from __future__ import annotations

import pytest

from liquiditylife.calculator.mappings import (
    expense_ratio_to_friction,
    risk_tolerance_to_gamma,
    scenario_key,
)
from liquiditylife.calculator.recommend import UserInputs, recommend
from liquiditylife.calculator.tables import PrecomputedTable


class TestMappings:
    def test_low_expense_ratio(self) -> None:
        assert expense_ratio_to_friction(0.20) == "low_friction"

    def test_med_expense_ratio(self) -> None:
        assert expense_ratio_to_friction(0.45) == "med_friction"

    def test_high_expense_ratio(self) -> None:
        assert expense_ratio_to_friction(0.70) == "high_friction"

    def test_aggressive_risk(self) -> None:
        assert risk_tolerance_to_gamma(1) == "aggressive"
        assert risk_tolerance_to_gamma(2) == "aggressive"

    def test_moderate_risk(self) -> None:
        assert risk_tolerance_to_gamma(3) == "moderate"

    def test_conservative_risk(self) -> None:
        assert risk_tolerance_to_gamma(4) == "conservative"
        assert risk_tolerance_to_gamma(5) == "conservative"

    def test_scenario_key(self) -> None:
        assert scenario_key("high_friction", "moderate") == "high_friction_moderate"


class TestUserInputs:
    def test_valid(self) -> None:
        inputs = UserInputs(
            age=35, annual_income=150_000, liquid_savings=200_000,
            monthly_fixed_expenses=5_000, risk_tolerance=3,
        )
        assert inputs.age == 35

    def test_invalid_age(self) -> None:
        with pytest.raises(ValueError, match="Age must be between"):
            UserInputs(
                age=15, annual_income=100_000, liquid_savings=50_000,
                monthly_fixed_expenses=3_000, risk_tolerance=3,
            )

    def test_invalid_income(self) -> None:
        with pytest.raises(ValueError, match="income must be positive"):
            UserInputs(
                age=35, annual_income=0, liquid_savings=50_000,
                monthly_fixed_expenses=3_000, risk_tolerance=3,
            )

    def test_invalid_risk(self) -> None:
        with pytest.raises(ValueError, match="Risk tolerance must be 1-5"):
            UserInputs(
                age=35, annual_income=100_000, liquid_savings=50_000,
                monthly_fixed_expenses=3_000, risk_tolerance=6,
            )


class TestPrecomputedTable:
    def _make_table(self) -> PrecomputedTable:
        """Create a simple linear table for testing."""
        ages = [25, 35, 45]
        m_grid = [0.5, 1.0, 2.0, 5.0, 10.0]
        # Stock share increases with m_t (more wealth = more stocks)
        stock_share = [
            [0.05, 0.10, 0.15, 0.25, 0.35],  # age 25
            [0.10, 0.15, 0.20, 0.30, 0.40],  # age 35
            [0.15, 0.20, 0.25, 0.35, 0.45],  # age 45
        ]
        return PrecomputedTable(ages=ages, m_grid=m_grid, stock_share=stock_share)

    def test_lookup_on_grid(self) -> None:
        table = self._make_table()
        assert abs(table.lookup(25, 1.0) - 0.10) < 1e-6

    def test_lookup_interpolated(self) -> None:
        table = self._make_table()
        share = table.lookup(25, 0.75)
        assert 0.05 < share < 0.15  # between grid points

    def test_lookup_clamped(self) -> None:
        table = self._make_table()
        share = table.lookup(25, 100.0)  # way above grid
        assert abs(share - 0.35) < 1e-6  # clamps to max

    def test_lookup_bounds(self) -> None:
        table = self._make_table()
        share = table.lookup(25, 1.0)
        assert 0.0 <= share <= 1.0

    def test_round_trip(self) -> None:
        table = self._make_table()
        data = table.to_dict()
        restored = PrecomputedTable.from_dict(data)
        assert restored.ages == table.ages
        assert abs(restored.lookup(35, 2.0) - table.lookup(35, 2.0)) < 1e-10


class TestRecommend:
    def _make_tables(self) -> dict[str, PrecomputedTable]:
        """Create minimal tables covering all 9 scenarios."""
        ages = list(range(25, 100))
        m_grid = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]
        tables = {}
        from liquiditylife.calculator.mappings import ALL_SCENARIO_KEYS
        for key in ALL_SCENARIO_KEYS:
            # Higher friction → lower stock shares
            base = 0.30 if "low" in key else 0.20 if "med" in key else 0.10
            # Conservative → lower shares
            if "conservative" in key:
                base -= 0.05
            elif "aggressive" in key:
                base += 0.05
            stock_share = []
            for age in ages:
                age_factor = 1.0 + (age - 25) * 0.005
                row = [
                    min(0.95, max(0.01, base * age_factor * (1 + 0.02 * i)))
                    for i in range(len(m_grid))
                ]
                stock_share.append(row)
            tables[key] = PrecomputedTable(ages=ages, m_grid=m_grid, stock_share=stock_share)
        return tables

    def test_basic_recommendation(self) -> None:
        tables = self._make_tables()
        inputs = UserInputs(
            age=35, annual_income=150_000, liquid_savings=200_000,
            monthly_fixed_expenses=5_000, risk_tolerance=3,
        )
        rec = recommend(inputs, tables)
        assert 0 < rec.stock_share_pct < 100
        assert rec.emergency_fund_months >= 6
        assert rec.stocks_dollars + rec.safe_dollars == pytest.approx(200_000, abs=1)

    def test_trajectory_ages(self) -> None:
        tables = self._make_tables()
        inputs = UserInputs(
            age=35, annual_income=100_000, liquid_savings=100_000,
            monthly_fixed_expenses=4_000, risk_tolerance=3,
        )
        rec = recommend(inputs, tables)
        assert len(rec.trajectory) == 3
        assert rec.trajectory[0].age == 40
        assert rec.trajectory[1].age == 45
        assert rec.trajectory[2].age == 50

    def test_high_friction_lower_stocks(self) -> None:
        tables = self._make_tables()
        low_exp = UserInputs(
            age=35, annual_income=100_000, liquid_savings=100_000,
            monthly_fixed_expenses=2_000, risk_tolerance=3,
        )
        high_exp = UserInputs(
            age=35, annual_income=100_000, liquid_savings=100_000,
            monthly_fixed_expenses=6_000, risk_tolerance=3,
        )
        rec_low = recommend(low_exp, tables)
        rec_high = recommend(high_exp, tables)
        assert rec_low.stock_share_pct >= rec_high.stock_share_pct

    def test_zero_savings(self) -> None:
        tables = self._make_tables()
        inputs = UserInputs(
            age=30, annual_income=100_000, liquid_savings=0,
            monthly_fixed_expenses=3_000, risk_tolerance=3,
        )
        rec = recommend(inputs, tables)
        assert rec.stocks_dollars == 0
        assert rec.safe_dollars == 0

    def test_old_age_trajectory_truncated(self) -> None:
        tables = self._make_tables()
        inputs = UserInputs(
            age=90, annual_income=50_000, liquid_savings=500_000,
            monthly_fixed_expenses=3_000, risk_tolerance=3,
        )
        rec = recommend(inputs, tables)
        assert len(rec.trajectory) <= 2  # only +5 and +10 fit under 99
