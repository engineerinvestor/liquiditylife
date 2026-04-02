"""Household preference specification."""

from __future__ import annotations

import math

from pydantic import BaseModel, field_validator


class Preferences(BaseModel, frozen=True):
    """Epstein-Zin preferences for the household.

    When ``psi == 1 / gamma``, this collapses to standard CRRA utility.
    """

    gamma: float
    """Relative risk aversion."""

    psi: float
    """Elasticity of intertemporal substitution."""

    beta: float
    """Time discount factor."""

    @field_validator("gamma")
    @classmethod
    def _gamma_positive(cls, v: float) -> float:
        if v <= 0:
            msg = "gamma must be positive"
            raise ValueError(msg)
        return v

    @field_validator("psi")
    @classmethod
    def _psi_positive(cls, v: float) -> float:
        if v <= 0:
            msg = "psi must be positive"
            raise ValueError(msg)
        return v

    @field_validator("beta")
    @classmethod
    def _beta_in_unit(cls, v: float) -> float:
        if not 0 < v < 1:
            msg = "beta must be in (0, 1)"
            raise ValueError(msg)
        return v

    @property
    def theta(self) -> float:
        """Epstein-Zin aggregation exponent: (1 - gamma) / (1 - 1/psi).

        Returns ``float('inf')`` when ``psi == 1`` (log EIS), signaling
        that the log-utility recursion should be used instead.
        """
        denom = 1.0 - 1.0 / self.psi
        if math.isclose(denom, 0.0, abs_tol=1e-12):
            return float("inf")
        return (1.0 - self.gamma) / denom

    @property
    def is_crra(self) -> bool:
        """True when EZ collapses to CRRA (psi == 1/gamma)."""
        return math.isclose(self.psi, 1.0 / self.gamma, rel_tol=1e-9)
