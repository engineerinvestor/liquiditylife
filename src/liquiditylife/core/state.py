"""Household state representations."""

from __future__ import annotations

from pydantic import BaseModel


class HouseholdState(BaseModel, frozen=True):
    """Canonical scaled state at the start of a period.

    All wealth and consumption variables are expressed relative to
    persistent earnings (the paper's dimensionality reduction).
    """

    age: int
    """Current age."""

    x_t: float
    """Equity premium state (AR(1) process)."""

    m_t: float
    """Scaled cash-on-hand (M_t / Y_t^P)."""

    cm_t: float
    """Lagged consumption ratio (C_{t-1} / Y_{t-1}^P)."""


class UnscaledState(BaseModel, frozen=True):
    """Unscaled state for debugging and reference.

    Tracks absolute dollar values rather than ratios.
    """

    age: int
    x_t: float
    M_t: float
    """Absolute cash-on-hand."""

    C_lag: float
    """Absolute lagged consumption."""

    Y_t: float
    """Persistent earnings level."""

    def to_scaled(self) -> HouseholdState:
        """Convert to the canonical scaled representation."""
        return HouseholdState(
            age=self.age,
            x_t=self.x_t,
            m_t=self.M_t / self.Y_t,
            cm_t=self.C_lag / self.Y_t,
        )
