"""Simulation result container."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from liquiditylife._types import ArrayFloat
from liquiditylife.calibrations.bundles import CalibrationBundle


class SimulationPaths(BaseModel):
    """Household simulation paths over the lifecycle.

    All arrays have shape ``(n_households, n_periods)``.
    """

    model_config = {"arbitrary_types_allowed": True}

    wealth: ArrayFloat
    """Cash-on-hand at start of each period."""

    consumption: ArrayFloat
    """Consumption chosen each period."""

    stock_share: ArrayFloat
    """Stock share of liquid wealth each period."""

    income: ArrayFloat
    """Realized income each period (scaled)."""

    equity_premium: ArrayFloat
    """Equity premium state each period."""

    stock_return: ArrayFloat
    """Realized stock return each period."""

    persistent_earnings: ArrayFloat
    """Log persistent earnings each period."""


class SimulationResult(BaseModel):
    """Complete simulation output with metadata."""

    model_config = {"arbitrary_types_allowed": True}

    n_households: int
    seed: int
    calibration: CalibrationBundle
    ages: ArrayFloat
    paths: SimulationPaths
    metadata: dict[str, Any] = {}
