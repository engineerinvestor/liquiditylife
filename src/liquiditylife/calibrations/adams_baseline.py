"""Adams (2026) baseline calibration factory."""

from __future__ import annotations

from liquiditylife.calibrations.bundles import CalibrationBundle
from liquiditylife.core.lifecycle import Lifecycle
from liquiditylife.core.preferences import Preferences
from liquiditylife.processes.adjustment_cost import AdjustmentCostModel
from liquiditylife.processes.asset_returns import AssetReturnProcess
from liquiditylife.processes.illiquid import IlliquidWealthRule
from liquiditylife.processes.income import IncomeProcess


def make_adams_baseline(phi_c: float = 10.0) -> CalibrationBundle:
    """Create an Adams (2026) baseline calibration.

    Args:
        phi_c: Consumption adjustment cost parameter. Paper values: 0, 5, 10.
            Default 10.0 matches the paper's empirically-preferred specification.
    """
    return CalibrationBundle(
        name=f"adams_baseline_phi_c_{phi_c:.0f}",
        description=(
            f"Adams (2026) baseline with phi_c={phi_c}. "
            "Income process parameters are public approximations."
        ),
        source="public_approximation",
        preferences=Preferences(gamma=5.0, psi=0.5, beta=0.85),
        lifecycle=Lifecycle(age_start=25, age_retire=60, age_max=99),
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
        adjustment_cost=AdjustmentCostModel(phi_c=phi_c),
        illiquid=IlliquidWealthRule(s=0.15, tau=0.35, S=0.60),
    )
