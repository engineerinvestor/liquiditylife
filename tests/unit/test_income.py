"""Tests for IncomeProcess and IncomeMixtureParams."""

import math

import pytest

from liquiditylife.processes.income import IncomeMixtureParams, IncomeProcess


class TestIncomeMixtureParams:
    def test_mu_bad_no_news(self) -> None:
        m = IncomeMixtureParams()
        assert math.isclose(m.mu_bad(0.0, 0.0), m.mu_eta_bad_base)

    def test_mu_bad_positive_cf(self) -> None:
        m = IncomeMixtureParams()
        # Positive cash-flow news raises bad-tail mean
        assert m.mu_bad(0.01, 0.0) > m.mu_bad(0.0, 0.0)

    def test_mu_good_no_news(self) -> None:
        m = IncomeMixtureParams()
        assert math.isclose(m.mu_good(0.0, 0.0), m.mu_eta_good_base)

    def test_invalid_p_eta_bad(self) -> None:
        with pytest.raises(ValueError, match="p_eta_bad must be in"):
            IncomeMixtureParams(p_eta_bad=1.5)


class TestIncomeProcess:
    def test_defaults(self) -> None:
        ip = IncomeProcess()
        assert ip.sigma_eps == 0.20
        assert ip.sigma_eta == 0.25

    def test_age_drift_at_25(self) -> None:
        ip = IncomeProcess()
        # At age 25, drift should be 0 (normalized to zero at entry)
        assert math.isclose(ip.age_drift(25), 0.0)

    def test_age_drift_positive_early(self) -> None:
        ip = IncomeProcess()
        # Early career should have positive drift
        assert ip.age_drift(30) > 0.0

    def test_persistent_earnings_transition(self) -> None:
        ip = IncomeProcess()
        log_y = 0.0
        eta = 0.05
        age = 30
        result = ip.persistent_earnings_transition(log_y, age, eta)
        expected = log_y + ip.age_drift(age) + eta
        assert math.isclose(result, expected)

    def test_invalid_sigma(self) -> None:
        with pytest.raises(ValueError, match="sigma must be positive"):
            IncomeProcess(sigma_eps=0.0)

    def test_serialization_round_trip(self) -> None:
        ip = IncomeProcess()
        ip2 = IncomeProcess.model_validate_json(ip.model_dump_json())
        assert ip == ip2
