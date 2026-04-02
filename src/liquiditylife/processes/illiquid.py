"""Illiquid wealth rules for working life and retirement."""

from __future__ import annotations

from pydantic import BaseModel, field_validator


class IlliquidWealthRule(BaseModel, frozen=True):
    """Working-life illiquid saving and retirement flow income.

    During working life, a share ``s`` of income is diverted to illiquid
    savings, and a share ``tau`` goes to taxes/contributions, leaving
    ``(1 - s - tau)`` as disposable liquid income.

    In retirement, flow income is ``S * Y_bar_terminal`` (a proxy for
    Social Security / pension income tied to terminal persistent earnings).
    """

    s: float = 0.15
    """Share of working-age income allocated to illiquid savings."""

    tau: float = 0.35
    """Tax/contribution rate on working-age income."""

    S: float = 0.60
    """Retirement income replacement rate (fraction of terminal persistent earnings)."""

    @field_validator("s", "tau", "S")
    @classmethod
    def _rate_valid(cls, v: float) -> float:
        if not 0 <= v <= 1:
            msg = "Rate must be in [0, 1]"
            raise ValueError(msg)
        return v

    @property
    def disposable_share(self) -> float:
        """Share of gross income available as liquid resources during work."""
        return 1.0 - self.s - self.tau

    def disposable_income(self, y_t: float) -> float:
        """Liquid disposable income during working life."""
        return self.disposable_share * y_t

    def retirement_flow(self, y_bar_terminal: float) -> float:
        """Retirement flow income proxy."""
        return self.S * y_bar_terminal
