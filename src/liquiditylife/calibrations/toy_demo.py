"""Toy demo calibration for fast testing and exploration."""

from __future__ import annotations

from liquiditylife.calibrations.bundles import CalibrationBundle
from liquiditylife.core.lifecycle import Lifecycle
from liquiditylife.core.preferences import Preferences
from liquiditylife.processes.adjustment_cost import AdjustmentCostModel
from liquiditylife.processes.asset_returns import AssetReturnProcess
from liquiditylife.processes.illiquid import IlliquidWealthRule
from liquiditylife.processes.income import IncomeProcess


def make_toy_demo() -> CalibrationBundle:
    """Create a small-grid toy calibration for fast exploration.

    Uses a shorter lifecycle (age 25-40-50) and the same economic
    structure as the baseline. Intended for testing and tutorials.
    """
    return CalibrationBundle(
        name="toy_demo_small_grid",
        description="Toy demo with short lifecycle for fast testing",
        source="public_approximation",
        preferences=Preferences(gamma=5.0, psi=0.5, beta=0.85),
        lifecycle=Lifecycle(age_start=25, age_retire=40, age_max=50),
        asset_returns=AssetReturnProcess(
            rf=0.02,
            x_bar=0.05,
            phi_x=0.85,
            sigma_xi=0.023,
            rho_cs=0.96,
        ),
        income=IncomeProcess(
            sigma_eps=0.20,
            sigma_eta=0.25,
        ),
        adjustment_cost=AdjustmentCostModel(phi_c=5.0),
        illiquid=IlliquidWealthRule(s=0.15, tau=0.35, S=0.60),
    )
