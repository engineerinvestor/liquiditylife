"""Asset return process with crash/non-crash mixture and return news decomposition."""

from __future__ import annotations

import math

import numpy as np
from pydantic import BaseModel, computed_field, field_validator

from liquiditylife._types import ArrayFloat


class CrashMixtureParams(BaseModel, frozen=True):
    """Parameters for the crash/non-crash mixture of (NCF, NDR) news.

    The no-crash means are derived from the zero-mean constraint:
    ``E[N] = p_crash * mu_crash + (1 - p_crash) * mu_no_crash = 0``.
    """

    p_crash: float = 0.15
    """Probability of a crash regime."""

    mu_cf_crash: float = -0.092
    """Mean cash-flow news in crash regime."""

    mu_dr_crash: float = 0.158
    """Mean discount-rate news in crash regime (positive = bad: prices fall)."""

    sigma_cf: float = 0.075
    """Conditional std dev of cash-flow news."""

    sigma_dr: float = 0.12
    """Conditional std dev of discount-rate news."""

    rho_cf_dr: float = -0.75
    """Correlation between NCF and NDR within each regime."""

    @field_validator("p_crash")
    @classmethod
    def _p_crash_valid(cls, v: float) -> float:
        if not 0 < v < 1:
            msg = "p_crash must be in (0, 1)"
            raise ValueError(msg)
        return v

    @computed_field  # type: ignore[prop-decorator]
    @property
    def mu_cf_no_crash(self) -> float:
        """No-crash cash-flow news mean (zero-mean constraint)."""
        return -self.p_crash * self.mu_cf_crash / (1.0 - self.p_crash)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def mu_dr_no_crash(self) -> float:
        """No-crash discount-rate news mean (zero-mean constraint)."""
        return -self.p_crash * self.mu_dr_crash / (1.0 - self.p_crash)

    @property
    def cov_matrix(self) -> ArrayFloat:
        """2x2 covariance matrix for (NCF, NDR) conditional on regime."""
        cov = self.rho_cf_dr * self.sigma_cf * self.sigma_dr
        return np.array(
            [[self.sigma_cf**2, cov], [cov, self.sigma_dr**2]],
            dtype=np.float64,
        )


class AssetReturnProcess(BaseModel, frozen=True):
    """Risky and risk-free return structure.

    The equity premium follows an AR(1) process. Stock returns are
    decomposed into cash-flow news (NCF) and discount-rate news (NDR).
    """

    rf: float = 0.02
    """Risk-free rate."""

    x_bar: float = 0.05
    """Unconditional mean equity premium."""

    phi_x: float = 0.85
    """AR(1) persistence of the equity premium."""

    sigma_xi: float = 0.023
    """Innovation std dev of the equity premium process."""

    rho_cs: float = 0.96
    """Mapping from premium innovation to discount-rate news."""

    crash: CrashMixtureParams = CrashMixtureParams()
    """Crash/non-crash mixture parameters for return news."""

    @field_validator("phi_x")
    @classmethod
    def _phi_x_stationary(cls, v: float) -> float:
        if abs(v) >= 1:
            msg = "phi_x must satisfy |phi_x| < 1 for stationarity"
            raise ValueError(msg)
        return v

    @property
    def sigma_x_unconditional(self) -> float:
        """Unconditional std dev of the equity premium state."""
        return self.sigma_xi / math.sqrt(1.0 - self.phi_x**2)

    def evolve_premium(self, x_t: float, xi: float) -> float:
        """Transition the equity premium state: x_{t+1} = x_bar + phi_x*(x_t - x_bar) + xi."""
        return self.x_bar + self.phi_x * (x_t - self.x_bar) + xi

    def ndr_from_xi(self, xi: float) -> float:
        """Compute discount-rate news from the premium innovation."""
        return self.rho_cs * xi

    def realized_return(self, x_t: float, ncf: float, ndr: float) -> float:
        """Compute the realized stock return from the equity premium and news.

        R_{stock} = rf + x_t + NCF - NDR
        (NDR positive = prices fall, so subtract from return)
        """
        return self.rf + x_t + ncf - ndr
