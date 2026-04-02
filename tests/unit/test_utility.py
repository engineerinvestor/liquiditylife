"""Tests for Epstein-Zin and CRRA utility functions."""

import math

from liquiditylife.core.preferences import Preferences
from liquiditylife.model.utility import crra_utility, ez_utility, terminal_utility


class TestEZUtility:
    def test_positive_output(self) -> None:
        prefs = Preferences(gamma=5.0, psi=0.5, beta=0.85)
        v = ez_utility(1.0, 1.0, prefs)
        assert v > 0

    def test_higher_c_higher_v(self) -> None:
        prefs = Preferences(gamma=5.0, psi=0.5, beta=0.85)
        v1 = ez_utility(1.0, 1.0, prefs)
        v2 = ez_utility(2.0, 1.0, prefs)
        assert v2 > v1

    def test_higher_ev_higher_v(self) -> None:
        prefs = Preferences(gamma=5.0, psi=0.5, beta=0.85)
        v1 = ez_utility(1.0, 1.0, prefs)
        v2 = ez_utility(1.0, 2.0, prefs)
        assert v2 > v1

    def test_log_eis_case(self) -> None:
        prefs = Preferences(gamma=5.0, psi=1.0, beta=0.85)
        v = ez_utility(1.0, 1.0, prefs)
        # c^(1-beta) * ev^beta = 1^0.15 * 1^0.85 = 1
        assert math.isclose(v, 1.0, rel_tol=1e-9)

    def test_small_consumption_floored(self) -> None:
        prefs = Preferences(gamma=5.0, psi=0.5, beta=0.85)
        v = ez_utility(0.0, 1.0, prefs)
        assert v > 0  # should not crash


class TestTerminalUtility:
    def test_positive(self) -> None:
        prefs = Preferences(gamma=5.0, psi=0.5, beta=0.85)
        v = terminal_utility(5.0, prefs)
        assert v == 5.0

    def test_floor(self) -> None:
        prefs = Preferences(gamma=5.0, psi=0.5, beta=0.85)
        v = terminal_utility(0.0, prefs)
        assert v > 0


class TestCRRAUtility:
    def test_log_case(self) -> None:
        u = crra_utility(math.e, 1.0)
        assert math.isclose(u, 1.0, rel_tol=1e-9)

    def test_power_case(self) -> None:
        u = crra_utility(2.0, 2.0)
        # 2^(-1) / (-1) = -0.5
        assert math.isclose(u, -0.5, rel_tol=1e-9)

    def test_monotone_in_c(self) -> None:
        assert crra_utility(2.0, 5.0) < crra_utility(3.0, 5.0)
