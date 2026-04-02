"""Precomputed lookup tables for instant policy queries."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from liquiditylife.calculator.mappings import (
    FRICTION_PHI_C,
    RISK_GAMMA,
    scenario_key,
)

logger = logging.getLogger(__name__)

_DEFAULT_TABLES_PATH = Path(__file__).parent / "data" / "default_tables.json"

# Default m_t grid for the lookup tables (wealth-to-income ratio)
_M_GRID = np.array(
    [0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0,
     5.0, 6.0, 7.0, 8.0, 10.0, 12.0, 15.0, 20.0, 25.0, 30.0],
    dtype=np.float64,
)


class PrecomputedTable:
    """A 2D lookup table: age x m_t -> stock_share."""

    def __init__(
        self,
        ages: list[int],
        m_grid: list[float],
        stock_share: list[list[float]],
    ) -> None:
        self.ages = ages
        self.m_grid = np.array(m_grid, dtype=np.float64)
        self._stock_share = np.array(stock_share, dtype=np.float64)

    def lookup(self, age: int, m_t: float) -> float:
        """Interpolate stock share at (age, m_t).

        Uses nearest-age and linear interpolation over m_t.
        """
        # Nearest age
        age_idx = min(
            range(len(self.ages)),
            key=lambda i: abs(self.ages[i] - age),
        )

        # Linear interpolation over m_t
        m_clamped = float(np.clip(m_t, self.m_grid[0], self.m_grid[-1]))
        share = float(np.interp(m_clamped, self.m_grid, self._stock_share[age_idx]))
        return max(0.0, min(1.0, share))

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict."""
        return {
            "ages": self.ages,
            "m_grid": self.m_grid.tolist(),
            "stock_share": self._stock_share.tolist(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PrecomputedTable:
        """Deserialize from a JSON-compatible dict."""
        return cls(
            ages=data["ages"],
            m_grid=data["m_grid"],
            stock_share=data["stock_share"],
        )


def precompute_tables(
    age_start: int = 25,
    age_retire: int = 60,
    age_max: int = 99,
) -> dict[str, PrecomputedTable]:
    """Precompute lookup tables for all 9 scenarios by solving the model.

    This runs the full solver 9 times (once per scenario). Takes several
    minutes but only needs to be done once.
    """
    from liquiditylife.calibrations.bundles import CalibrationBundle
    from liquiditylife.core.lifecycle import Lifecycle
    from liquiditylife.core.preferences import Preferences
    from liquiditylife.processes.adjustment_cost import AdjustmentCostModel
    from liquiditylife.processes.asset_returns import AssetReturnProcess
    from liquiditylife.processes.illiquid import IlliquidWealthRule
    from liquiditylife.processes.income import IncomeProcess
    from liquiditylife.solve.grids import GridSpec
    from liquiditylife.solve.quadrature import QuadratureSpec
    from liquiditylife.solve.solver import SolverConfig, solve_model

    config = SolverConfig(
        grid_spec=GridSpec(
            x_points=5, m_points=15, cm_points=5,
            x_min=-0.02, x_max=0.12, m_max=30.0,
        ),
        quad_spec=QuadratureSpec(n_xi=5, n_ncf=5, n_eta=5, n_eps=3),
        n_c_grid=50,
        n_theta_grid=21,
        verbose=False,
    )

    lc = Lifecycle(age_start=age_start, age_retire=age_retire, age_max=age_max)
    ages = list(lc.ages)
    m_grid = _M_GRID.tolist()

    tables: dict[str, PrecomputedTable] = {}

    for friction_name, phi_c in FRICTION_PHI_C.items():
        for risk_name, gamma in RISK_GAMMA.items():
            key = scenario_key(friction_name, risk_name)
            logger.info("Precomputing %s (phi_c=%.0f, gamma=%.0f)...", key, phi_c, gamma)

            cal = CalibrationBundle(
                name=key,
                description=f"Calculator scenario: {key}",
                source="public_approximation",
                preferences=Preferences(gamma=gamma, psi=0.5, beta=0.85),
                lifecycle=lc,
                asset_returns=AssetReturnProcess(),
                income=IncomeProcess(),
                adjustment_cost=AdjustmentCostModel(phi_c=phi_c),
                illiquid=IlliquidWealthRule(),
            )

            solution = solve_model(cal, config)

            # Extract stock share at x_t = x_bar (mean) and cm_t = 0.5 (typical)
            x_mid = cal.asset_returns.x_bar
            cm_mid = 0.5
            stock_share_grid: list[list[float]] = []

            for age in ages:
                pf = solution.policies[age]
                row = [
                    pf.stock_share_at(x_mid, m, cm_mid) for m in m_grid
                ]
                stock_share_grid.append(row)

            tables[key] = PrecomputedTable(
                ages=ages,
                m_grid=m_grid,
                stock_share=stock_share_grid,
            )

    return tables


def export_tables_json(tables: dict[str, PrecomputedTable], path: Path) -> None:
    """Write precomputed tables to a JSON file."""
    data = {key: table.to_dict() for key, table in tables.items()}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, separators=(",", ":")))
    logger.info("Exported %d tables to %s", len(tables), path)


def load_tables_json(path: Path | None = None) -> dict[str, PrecomputedTable]:
    """Load precomputed tables from a JSON file.

    Args:
        path: Path to JSON file. Defaults to the shipped default tables.
    """
    if path is None:
        path = _DEFAULT_TABLES_PATH
    if not path.exists():
        msg = (
            f"Lookup tables not found at {path}. "
            "Run 'liquiditylife calculator precompute' to generate them."
        )
        raise FileNotFoundError(msg)
    data = json.loads(path.read_text())
    return {key: PrecomputedTable.from_dict(val) for key, val in data.items()}
