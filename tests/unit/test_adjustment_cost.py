"""Tests for AdjustmentCostModel."""

import math

import pytest

from liquiditylife.processes.adjustment_cost import AdjustmentCostModel


class TestAdjustmentCost:
    def test_no_cost_when_phi_zero(self) -> None:
        ac = AdjustmentCostModel(phi_c=0.0)
        assert ac.cost(0.8, 1.0) == 0.0

    def test_no_cost_when_increasing(self) -> None:
        ac = AdjustmentCostModel(phi_c=10.0)
        assert ac.cost(1.1, 1.0) == 0.0

    def test_no_cost_when_equal(self) -> None:
        ac = AdjustmentCostModel(phi_c=10.0)
        assert ac.cost(1.0, 1.0) == 0.0

    def test_cost_when_decreasing(self) -> None:
        ac = AdjustmentCostModel(phi_c=10.0)
        # cost = (10/2) * (1.0 - 0.9)^2 / 1.0 = 5 * 0.01 = 0.05
        cost = ac.cost(0.9, 1.0)
        assert math.isclose(cost, 0.05)

    def test_cost_larger_drop(self) -> None:
        ac = AdjustmentCostModel(phi_c=10.0)
        # cost = (10/2) * (1.0 - 0.5)^2 / 1.0 = 5 * 0.25 = 1.25
        cost = ac.cost(0.5, 1.0)
        assert math.isclose(cost, 1.25)

    def test_total_expenditure(self) -> None:
        ac = AdjustmentCostModel(phi_c=10.0)
        total = ac.total_expenditure(0.9, 1.0)
        assert math.isclose(total, 0.9 + 0.05)

    def test_total_expenditure_no_cost(self) -> None:
        ac = AdjustmentCostModel(phi_c=10.0)
        total = ac.total_expenditure(1.1, 1.0)
        assert math.isclose(total, 1.1)

    def test_invalid_phi_c(self) -> None:
        with pytest.raises(ValueError, match="phi_c must be non-negative"):
            AdjustmentCostModel(phi_c=-1.0)
