"""Policy surfaces and comparative statics."""

from __future__ import annotations

import numpy as np
import pandas as pd

from liquiditylife._types import ArrayFloat
from liquiditylife.calibrations.registry import load_calibration
from liquiditylife.simulate.engine import simulate_cohorts
from liquiditylife.solve.solver import SolvedModel, SolverConfig, solve_model


def policy_surface(
    solution: SolvedModel,
    age: int,
    x_values: ArrayFloat | None = None,
    m_values: ArrayFloat | None = None,
    cm_t: float = 0.5,
) -> pd.DataFrame:
    """Evaluate policy function over a 2D grid at a fixed age and cm_t.

    Args:
        solution: Solved model.
        age: Age at which to evaluate.
        x_values: Equity premium values. Defaults to solution grid.
        m_values: Cash-on-hand values. Defaults to solution grid.
        cm_t: Fixed lagged consumption ratio.

    Returns:
        DataFrame with columns (x_t, m_t, consumption, stock_share, value).
    """
    pf = solution.policies[age]

    if x_values is None:
        x_values = pf.grid_x
    if m_values is None:
        m_values = pf.grid_m

    rows: list[dict[str, float]] = []
    for x in x_values:
        for m in m_values:
            rows.append({
                "x_t": float(x),
                "m_t": float(m),
                "consumption": pf.consume(float(x), float(m), cm_t),
                "stock_share": pf.stock_share_at(float(x), float(m), cm_t),
                "value": pf.value_at(float(x), float(m), cm_t),
            })

    return pd.DataFrame(rows)


def comparative_statics(
    calibration_names: list[str],
    config: SolverConfig | None = None,
) -> dict[str, SolvedModel]:
    """Solve multiple calibrations for comparison.

    Args:
        calibration_names: List of calibration names to solve.
        config: Optional solver configuration (shared across all calibrations).

    Returns:
        Dict mapping calibration name to solved model.
    """
    results: dict[str, SolvedModel] = {}
    for name in calibration_names:
        cal = load_calibration(name)
        results[name] = solve_model(cal, config)
    return results


def age_profile(
    solution: SolvedModel,
    n_households: int = 10_000,
    seed: int = 42,
) -> pd.DataFrame:
    """Compute age-averaged statistics from simulation.

    Args:
        solution: Solved model.
        n_households: Number of households to simulate.
        seed: Random seed.

    Returns:
        DataFrame with one row per age and columns for mean/median statistics.
    """
    sim = simulate_cohorts(solution, n_households=n_households, seed=seed)
    lc = solution.calibration.lifecycle

    rows: list[dict[str, float]] = []
    for t, age in enumerate(range(lc.age_start, lc.age_max + 1)):
        rows.append({
            "age": float(age),
            "mean_stock_share": float(np.mean(sim.paths.stock_share[:, t])),
            "median_stock_share": float(np.median(sim.paths.stock_share[:, t])),
            "mean_wealth": float(np.mean(sim.paths.wealth[:, t])),
            "median_wealth": float(np.median(sim.paths.wealth[:, t])),
            "mean_consumption": float(np.mean(sim.paths.consumption[:, t])),
        })

    return pd.DataFrame(rows)
