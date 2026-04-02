"""Tests for AssetReturnProcess and CrashMixtureParams."""

import math

import pytest

from liquiditylife.processes.asset_returns import AssetReturnProcess, CrashMixtureParams


class TestCrashMixtureParams:
    def test_zero_mean_constraint_cf(self) -> None:
        c = CrashMixtureParams()
        unconditional = c.p_crash * c.mu_cf_crash + (1 - c.p_crash) * c.mu_cf_no_crash
        assert math.isclose(unconditional, 0.0, abs_tol=1e-12)

    def test_zero_mean_constraint_dr(self) -> None:
        c = CrashMixtureParams()
        unconditional = c.p_crash * c.mu_dr_crash + (1 - c.p_crash) * c.mu_dr_no_crash
        assert math.isclose(unconditional, 0.0, abs_tol=1e-12)

    def test_cov_matrix_shape(self) -> None:
        c = CrashMixtureParams()
        assert c.cov_matrix.shape == (2, 2)

    def test_cov_matrix_symmetric(self) -> None:
        c = CrashMixtureParams()
        assert math.isclose(c.cov_matrix[0, 1], c.cov_matrix[1, 0])

    def test_invalid_p_crash(self) -> None:
        with pytest.raises(ValueError, match="p_crash must be in"):
            CrashMixtureParams(p_crash=0.0)


class TestAssetReturnProcess:
    def test_defaults(self) -> None:
        ar = AssetReturnProcess()
        assert ar.rf == 0.02
        assert ar.x_bar == 0.05

    def test_evolve_premium_at_mean(self) -> None:
        ar = AssetReturnProcess()
        # At x_t = x_bar with zero innovation, x_{t+1} = x_bar
        x_next = ar.evolve_premium(ar.x_bar, 0.0)
        assert math.isclose(x_next, ar.x_bar)

    def test_evolve_premium_positive_shock(self) -> None:
        ar = AssetReturnProcess()
        x_next = ar.evolve_premium(ar.x_bar, 0.01)
        assert x_next > ar.x_bar

    def test_ndr_from_xi(self) -> None:
        ar = AssetReturnProcess()
        ndr = ar.ndr_from_xi(0.01)
        assert math.isclose(ndr, 0.96 * 0.01)

    def test_realized_return(self) -> None:
        ar = AssetReturnProcess()
        # R = rf + x_t + NCF - NDR
        r = ar.realized_return(x_t=0.05, ncf=0.02, ndr=0.01)
        assert math.isclose(r, 0.02 + 0.05 + 0.02 - 0.01)

    def test_sigma_x_unconditional(self) -> None:
        ar = AssetReturnProcess()
        expected = 0.023 / math.sqrt(1 - 0.85**2)
        assert math.isclose(ar.sigma_x_unconditional, expected)

    def test_invalid_phi_x(self) -> None:
        with pytest.raises(ValueError, match="phi_x must satisfy"):
            AssetReturnProcess(phi_x=1.0)

    def test_serialization_round_trip(self) -> None:
        ar = AssetReturnProcess()
        ar2 = AssetReturnProcess.model_validate_json(ar.model_dump_json())
        assert ar == ar2
