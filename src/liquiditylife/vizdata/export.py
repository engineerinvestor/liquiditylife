"""Visualization-friendly data export."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import xarray as xr

from liquiditylife.simulate.result import SimulationResult
from liquiditylife.solve.solver import SolvedModel


def solution_to_dataframe(
    solution: SolvedModel,
    ages: list[int] | None = None,
) -> pd.DataFrame:
    """Flatten policy grids into a tidy DataFrame.

    Args:
        solution: Solved model.
        ages: Ages to include. Defaults to all.

    Returns:
        DataFrame with columns (age, x_t, m_t, cm_t, consumption, stock_share, value).
    """
    if ages is None:
        ages = sorted(solution.policies.keys())

    rows: list[dict[str, float]] = []
    for age in ages:
        pf = solution.policies[age]
        for ix, x in enumerate(pf.grid_x):
            for im, m in enumerate(pf.grid_m):
                for icm, cm in enumerate(pf.grid_cm):
                    rows.append({
                        "age": float(age),
                        "x_t": float(x),
                        "m_t": float(m),
                        "cm_t": float(cm),
                        "consumption": float(pf.consumption_grid[ix, im, icm]),
                        "stock_share": float(pf.stock_share_grid[ix, im, icm]),
                        "value": float(pf.value_grid[ix, im, icm]),
                    })

    return pd.DataFrame(rows)


def simulation_to_xarray(result: SimulationResult) -> xr.Dataset:
    """Convert simulation paths to an xarray Dataset.

    Args:
        result: Simulation result.

    Returns:
        Dataset with dimensions (household, period).
    """
    lc = result.calibration.lifecycle
    ages = list(range(lc.age_start, lc.age_max + 1))

    return xr.Dataset(
        {
            "wealth": (["household", "age"], result.paths.wealth),
            "consumption": (["household", "age"], result.paths.consumption),
            "stock_share": (["household", "age"], result.paths.stock_share),
            "income": (["household", "age"], result.paths.income),
            "equity_premium": (["household", "age"], result.paths.equity_premium),
            "stock_return": (["household", "age"], result.paths.stock_return),
        },
        coords={
            "household": range(result.n_households),
            "age": ages,
        },
        attrs={
            "calibration_name": result.calibration.name,
            "seed": result.seed,
            "n_households": result.n_households,
        },
    )


def to_json_payload(df: pd.DataFrame) -> str:
    """Convert a DataFrame to compact JSON for web dashboards."""
    result = df.to_json(orient="records")
    return str(result)


def to_parquet(df: pd.DataFrame, path: Path) -> None:
    """Save a DataFrame to Parquet format."""
    df.to_parquet(path, index=False)
