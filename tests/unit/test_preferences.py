"""Tests for Preferences domain model."""

import math

import pytest

from liquiditylife.core.preferences import Preferences


class TestPreferencesConstruction:
    def test_baseline_values(self) -> None:
        p = Preferences(gamma=5.0, psi=0.5, beta=0.85)
        assert p.gamma == 5.0
        assert p.psi == 0.5
        assert p.beta == 0.85

    def test_frozen(self) -> None:
        p = Preferences(gamma=5.0, psi=0.5, beta=0.85)
        with pytest.raises(Exception):  # noqa: B017
            p.gamma = 10.0  # type: ignore[misc]

    def test_invalid_gamma(self) -> None:
        with pytest.raises(ValueError, match="gamma must be positive"):
            Preferences(gamma=-1.0, psi=0.5, beta=0.85)

    def test_invalid_psi(self) -> None:
        with pytest.raises(ValueError, match="psi must be positive"):
            Preferences(gamma=5.0, psi=0.0, beta=0.85)

    def test_invalid_beta_low(self) -> None:
        with pytest.raises(ValueError, match="beta must be in"):
            Preferences(gamma=5.0, psi=0.5, beta=0.0)

    def test_invalid_beta_high(self) -> None:
        with pytest.raises(ValueError, match="beta must be in"):
            Preferences(gamma=5.0, psi=0.5, beta=1.0)


class TestPreferencesProperties:
    def test_theta_baseline(self) -> None:
        p = Preferences(gamma=5.0, psi=0.5, beta=0.85)
        # theta = (1 - 5) / (1 - 1/0.5) = -4 / -1 = 4
        assert math.isclose(p.theta, 4.0)

    def test_theta_log_eis(self) -> None:
        p = Preferences(gamma=5.0, psi=1.0, beta=0.85)
        assert p.theta == float("inf")

    def test_is_crra_true(self) -> None:
        p = Preferences(gamma=5.0, psi=0.2, beta=0.85)
        assert p.is_crra

    def test_is_crra_false(self) -> None:
        p = Preferences(gamma=5.0, psi=0.5, beta=0.85)
        assert not p.is_crra


class TestPreferencesSerialization:
    def test_json_round_trip(self) -> None:
        p = Preferences(gamma=5.0, psi=0.5, beta=0.85)
        json_str = p.model_dump_json()
        p2 = Preferences.model_validate_json(json_str)
        assert p == p2
