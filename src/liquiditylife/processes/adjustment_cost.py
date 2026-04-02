"""Asymmetric consumption adjustment cost model."""

from __future__ import annotations

from pydantic import BaseModel, field_validator


class AdjustmentCostModel(BaseModel, frozen=True):
    """Asymmetric quadratic cost of reducing consumption below prior level.

    Cost applies only when C_t < C_{t-1}:
      Phi_C = (phi_c / 2) * max(0, C_{t-1} - C_t)^2 / C_{t-1}

    Higher phi_c generates greater precautionary savings and lower
    optimal stock shares, matching observed savings responses in the data.
    """

    phi_c: float = 0.0
    """Adjustment cost parameter. Paper baseline values: 0, 5, 10."""

    @field_validator("phi_c")
    @classmethod
    def _phi_c_non_negative(cls, v: float) -> float:
        if v < 0:
            msg = "phi_c must be non-negative"
            raise ValueError(msg)
        return v

    def cost(self, c_t: float, c_lag: float) -> float:
        """Compute the adjustment cost Phi_C(C_t, C_{t-1}).

        Returns 0 when c_t >= c_lag (no cost for increasing consumption).
        """
        if self.phi_c == 0.0 or c_t >= c_lag:
            return 0.0
        shortfall = c_lag - c_t
        return (self.phi_c / 2.0) * shortfall**2 / c_lag

    def total_expenditure(self, c_t: float, c_lag: float) -> float:
        """Total resources consumed: C_t + Phi_C(C_t, C_{t-1})."""
        return c_t + self.cost(c_t, c_lag)
