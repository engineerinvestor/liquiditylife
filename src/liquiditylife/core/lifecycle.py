"""Lifecycle age structure."""

from __future__ import annotations

from pydantic import BaseModel, model_validator


class Lifecycle(BaseModel, frozen=True):
    """Age boundaries for the household lifecycle."""

    age_start: int = 25
    """Age at which the household enters the model."""

    age_retire: int = 60
    """Age at which the household retires (first retirement period)."""

    age_max: int = 99
    """Terminal age (last period alive)."""

    @model_validator(mode="after")
    def _ages_ordered(self) -> Lifecycle:
        if not (self.age_start < self.age_retire < self.age_max):
            msg = "Must have age_start < age_retire < age_max"
            raise ValueError(msg)
        return self

    @property
    def n_working_periods(self) -> int:
        """Number of working-life periods (age_start to age_retire - 1)."""
        return self.age_retire - self.age_start

    @property
    def n_retirement_periods(self) -> int:
        """Number of retirement periods (age_retire to age_max)."""
        return self.age_max - self.age_retire + 1

    @property
    def n_total_periods(self) -> int:
        """Total number of periods."""
        return self.age_max - self.age_start + 1

    @property
    def ages(self) -> range:
        """Range of all ages in the model."""
        return range(self.age_start, self.age_max + 1)

    def is_retired(self, age: int) -> bool:
        """Check if the household is retired at the given age."""
        return age >= self.age_retire
