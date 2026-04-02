"""Tests for budget constraint transitions."""

import math

from liquiditylife.model.budget import (
    end_of_period_savings,
    next_period_coh_retired,
    next_period_coh_working,
    portfolio_return,
)
from liquiditylife.processes.adjustment_cost import AdjustmentCostModel
from liquiditylife.processes.illiquid import IlliquidWealthRule


class TestEndOfPeriodSavings:
    def test_no_adjustment_cost(self) -> None:
        adj = AdjustmentCostModel(phi_c=0.0)
        s = end_of_period_savings(m_t=5.0, c_t=2.0, cm_lag=2.0, adj_cost=adj)
        assert math.isclose(s, 3.0)

    def test_with_adjustment_cost(self) -> None:
        adj = AdjustmentCostModel(phi_c=10.0)
        # c_t = 0.9, cm_lag = 1.0 -> cost = 5 * 0.01 / 1.0 = 0.05
        s = end_of_period_savings(m_t=5.0, c_t=0.9, cm_lag=1.0, adj_cost=adj)
        expected = 5.0 - 0.9 - 0.05
        assert math.isclose(s, expected)

    def test_no_cost_when_increasing(self) -> None:
        adj = AdjustmentCostModel(phi_c=10.0)
        s = end_of_period_savings(m_t=5.0, c_t=1.1, cm_lag=1.0, adj_cost=adj)
        assert math.isclose(s, 3.9)


class TestPortfolioReturn:
    def test_all_bonds(self) -> None:
        r = portfolio_return(theta=0.0, r_stock=0.08, rf=0.02)
        assert math.isclose(r, 1.02)

    def test_all_stocks(self) -> None:
        r = portfolio_return(theta=1.0, r_stock=0.08, rf=0.02)
        assert math.isclose(r, 1.08)

    def test_half_half(self) -> None:
        r = portfolio_return(theta=0.5, r_stock=0.08, rf=0.02)
        assert math.isclose(r, 1.05)


class TestNextPeriodCoh:
    def test_working_basic(self) -> None:
        illiq = IlliquidWealthRule()
        m = next_period_coh_working(
            savings=3.0, r_port=1.05, y_ratio=1.0, illiquid=illiq
        )
        # 3.0 * 1.05 / 1.0 + 0.50 = 3.65
        assert math.isclose(m, 3.65)

    def test_retired_basic(self) -> None:
        illiq = IlliquidWealthRule()
        m = next_period_coh_retired(
            savings=3.0, r_port=1.05, y_ratio=1.0, illiquid=illiq
        )
        # 3.0 * 1.05 / 1.0 + 0.60 = 3.75
        assert math.isclose(m, 3.75)
