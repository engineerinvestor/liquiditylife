"""Backward induction solver for the life-cycle model."""

from __future__ import annotations

import logging
import time
from typing import Any

from pydantic import BaseModel

from liquiditylife._types import ArrayFloat
from liquiditylife.calibrations.bundles import CalibrationBundle
from liquiditylife.core.policy import PolicyFunction
from liquiditylife.solve.bellman import bellman_operator
from liquiditylife.solve.grids import GridSpec, build_grids, default_grid_spec
from liquiditylife.solve.interpolation import build_interpolator
from liquiditylife.solve.quadrature import QuadratureSpec, build_shock_grid

logger = logging.getLogger(__name__)


class SolverConfig(BaseModel, frozen=True):
    """Configuration for the backward induction solver."""

    grid_spec: GridSpec = GridSpec()
    """State grid specification."""

    quad_spec: QuadratureSpec = QuadratureSpec()
    """Quadrature specification for shock integration."""

    n_c_grid: int = 80
    """Number of consumption grid points per state point."""

    n_theta_grid: int = 21
    """Number of stock share grid points per state point."""

    verbose: bool = True
    """Print progress during solve."""


class SolvedModel(BaseModel):
    """Result of solving the life-cycle model.

    Contains policy functions, value functions, and metadata
    for all ages in the lifecycle.
    """

    model_config = {"arbitrary_types_allowed": True}

    calibration: CalibrationBundle
    solver_config: SolverConfig
    policies: dict[int, PolicyFunction]
    value_functions: dict[int, ArrayFloat]
    grids: tuple[ArrayFloat, ArrayFloat, ArrayFloat]
    solve_time_seconds: float
    metadata: dict[str, Any] = {}


def solve_model(
    cal: CalibrationBundle,
    config: SolverConfig | None = None,
) -> SolvedModel:
    """Solve the life-cycle model via backward induction.

    Args:
        cal: Calibration bundle defining the economic model.
        config: Solver configuration. If None, uses defaults
            with grid bounds derived from the calibration.

    Returns:
        A ``SolvedModel`` containing policy and value functions at every age.
    """
    if config is None:
        grid_spec = default_grid_spec(cal)
        config = SolverConfig(grid_spec=grid_spec)

    lc = cal.lifecycle
    grid_x, grid_m, grid_cm = build_grids(config.grid_spec)
    shock_grid = build_shock_grid(cal, config.quad_spec)

    policies: dict[int, PolicyFunction] = {}
    value_functions: dict[int, ArrayFloat] = {}

    v_next_interp = None
    t_start = time.perf_counter()

    # Backward induction from age_max to age_start
    for age in range(lc.age_max, lc.age_start - 1, -1):
        if config.verbose:
            logger.info("Solving age %d / %d", age, lc.age_max)

        V, C_pol, theta_pol = bellman_operator(
            age=age,
            grid_x=grid_x,
            grid_m=grid_m,
            grid_cm=grid_cm,
            v_next_interp=v_next_interp,
            cal=cal,
            shock_grid=shock_grid,
            n_c_grid=config.n_c_grid,
            n_theta_grid=config.n_theta_grid,
        )

        value_functions[age] = V

        # Build interpolator for this age's value function
        v_interp = build_interpolator(grid_x, grid_m, grid_cm, V)

        # Build policy function with interpolators
        pf = PolicyFunction(
            age=age,
            consumption=C_pol,
            stock_share=theta_pol,
            value=V,
            grid_x=grid_x,
            grid_m=grid_m,
            grid_cm=grid_cm,
        )
        c_interp = build_interpolator(grid_x, grid_m, grid_cm, C_pol)
        theta_interp = build_interpolator(grid_x, grid_m, grid_cm, theta_pol)
        pf.set_interpolators(c_interp, theta_interp, v_interp)

        policies[age] = pf
        v_next_interp = v_interp

    solve_time = time.perf_counter() - t_start

    if config.verbose:
        logger.info("Solve complete in %.1f seconds", solve_time)

    return SolvedModel(
        calibration=cal,
        solver_config=config,
        policies=policies,
        value_functions=value_functions,
        grids=(grid_x, grid_m, grid_cm),
        solve_time_seconds=solve_time,
    )
