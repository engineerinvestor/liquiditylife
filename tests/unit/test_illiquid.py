"""Tests for IlliquidWealthRule."""

import math

import pytest

from liquiditylife.processes.illiquid import IlliquidWealthRule


class TestIlliquidWealthRule:
    def test_defaults(self) -> None:
        iw = IlliquidWealthRule()
        assert iw.s == 0.15
        assert iw.tau == 0.35
        assert iw.S == 0.60

    def test_disposable_share(self) -> None:
        iw = IlliquidWealthRule()
        # 1 - 0.15 - 0.35 = 0.50
        assert math.isclose(iw.disposable_share, 0.50)

    def test_disposable_income(self) -> None:
        iw = IlliquidWealthRule()
        assert math.isclose(iw.disposable_income(100.0), 50.0)

    def test_retirement_flow(self) -> None:
        iw = IlliquidWealthRule()
        assert math.isclose(iw.retirement_flow(100.0), 60.0)

    def test_invalid_rate(self) -> None:
        with pytest.raises(ValueError, match="Rate must be in"):
            IlliquidWealthRule(s=-0.1)

    def test_serialization_round_trip(self) -> None:
        iw = IlliquidWealthRule()
        iw2 = IlliquidWealthRule.model_validate_json(iw.model_dump_json())
        assert iw == iw2
