"""Income process with persistent/transitory decomposition and crash linkage."""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel, field_validator

from liquiditylife._types import ArrayFloat

# Polynomial approximation of the Guvenen, Ozkan, and Song (2014)
# life-cycle log-earnings age-drift profile (normalised to zero at age 25).
# Coefficients for: g(age) = a0 + a1*(age-25) + a2*(age-25)^2 + a3*(age-25)^3
_GOS_COEFFS: ArrayFloat = np.array([0.0, 0.035, -0.0005, 2.0e-6], dtype=np.float64)


class IncomeMixtureParams(BaseModel, frozen=True):
    """Parameters linking persistent earnings shocks to return news.

    The persistent shock eta is drawn from a mixture:
    - With probability p_eta_bad: eta ~ N(mu_bad(NCF, NDR), sigma_eta_tilde^2)
    - With probability 1 - p_eta_bad: eta ~ N(mu_good(NCF, NDR), sigma_eta_tilde^2)

    The conditional means depend on return news:
      mu_bad  = mu_eta_bad_base + lambda_cf_bad * NCF + lambda_dr_bad * NDR
      mu_good = mu_eta_good_base + lambda_cf_good * NCF + lambda_dr_good * NDR

    Slope parameters are calibrated so that a 1% increase in NCF raises
    average persistent earnings by ~0.2% and raises the bad-tail mean by ~0.9%
    (public approximation of the paper's qualitative targets).
    """

    p_eta_bad: float = 0.15
    """Probability of a bad persistent earnings shock."""

    mu_eta_bad_base: float = -0.10
    """Base mean of the bad-tail persistent shock (log scale)."""

    mu_eta_good_base: float = 0.0
    """Base mean of the good-regime persistent shock."""

    sigma_eta_tilde: float = 0.15
    """Conditional std dev of persistent shock within each regime."""

    lambda_cf_bad: float = 0.9
    """Sensitivity of bad-tail mean to cash-flow news."""

    lambda_dr_bad: float = -0.5
    """Sensitivity of bad-tail mean to discount-rate news."""

    lambda_cf_good: float = 0.2
    """Sensitivity of good-regime mean to cash-flow news."""

    lambda_dr_good: float = -0.1
    """Sensitivity of good-regime mean to discount-rate news."""

    @field_validator("p_eta_bad")
    @classmethod
    def _p_valid(cls, v: float) -> float:
        if not 0 < v < 1:
            msg = "p_eta_bad must be in (0, 1)"
            raise ValueError(msg)
        return v

    def mu_bad(self, ncf: float, ndr: float) -> float:
        """Conditional mean of the bad-tail persistent shock."""
        return self.mu_eta_bad_base + self.lambda_cf_bad * ncf + self.lambda_dr_bad * ndr

    def mu_good(self, ncf: float, ndr: float) -> float:
        """Conditional mean of the good-regime persistent shock."""
        return self.mu_eta_good_base + self.lambda_cf_good * ncf + self.lambda_dr_good * ndr


class IncomeProcess(BaseModel, frozen=True):
    """Nonfinancial income with persistent and transitory components.

    Income: Y_t = Y_bar_t * exp(eps_t), where Y_bar_t is persistent
    earnings following a random walk with age drift, and eps_t is a
    transitory shock.
    """

    sigma_eps: float = 0.20
    """Std dev of transitory income shock (log scale)."""

    sigma_eta: float = 0.25
    """Overall std dev of persistent shock (including mixture dispersion)."""

    mixture: IncomeMixtureParams = IncomeMixtureParams()
    """Mixture parameters linking persistent shocks to return news."""

    age_drift_source: str = "guvenen_2014"
    """Source for the age-drift profile."""

    @field_validator("sigma_eps", "sigma_eta")
    @classmethod
    def _sigma_positive(cls, v: float) -> float:
        if v <= 0:
            msg = "sigma must be positive"
            raise ValueError(msg)
        return v

    def age_drift(self, age: int) -> float:
        """Log-earnings age drift g(age) from the Guvenen et al. (2014) profile."""
        t = float(age - 25)
        return float(
            _GOS_COEFFS[0]
            + _GOS_COEFFS[1] * t
            + _GOS_COEFFS[2] * t**2
            + _GOS_COEFFS[3] * t**3
        )

    def persistent_earnings_transition(
        self, log_y_bar: float, age: int, eta: float
    ) -> float:
        """Transition persistent log-earnings: log Y_bar_{t+1} = log Y_bar_t + g(age) + eta."""
        return log_y_bar + self.age_drift(age) + eta
