"""Calibration bundle combining all model components."""

from __future__ import annotations

import hashlib
from typing import Literal

from pydantic import BaseModel

from liquiditylife.core.lifecycle import Lifecycle
from liquiditylife.core.preferences import Preferences
from liquiditylife.processes.adjustment_cost import AdjustmentCostModel
from liquiditylife.processes.asset_returns import AssetReturnProcess
from liquiditylife.processes.illiquid import IlliquidWealthRule
from liquiditylife.processes.income import IncomeProcess


class CalibrationBundle(BaseModel, frozen=True):
    """Complete model calibration.

    Composes all domain objects into a single immutable bundle with
    provenance tracking.
    """

    name: str
    """Human-readable calibration name."""

    description: str
    """Brief description of what this calibration represents."""

    source: Literal["published_baseline", "public_approximation", "user_custom"]
    """Provenance category for result attribution."""

    preferences: Preferences
    lifecycle: Lifecycle
    asset_returns: AssetReturnProcess
    income: IncomeProcess
    adjustment_cost: AdjustmentCostModel
    illiquid: IlliquidWealthRule

    def fingerprint(self) -> str:
        """Hex digest uniquely identifying this calibration for caching."""
        payload = self.model_dump_json(indent=None)
        return hashlib.sha256(payload.encode()).hexdigest()[:16]
