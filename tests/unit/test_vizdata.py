"""Tests for visualization data export."""

from __future__ import annotations

import pytest

from liquiditylife.calibrations.registry import load_calibration
from liquiditylife.simulate.engine import simulate_cohorts
from liquiditylife.solve.grids import GridSpec
from liquiditylife.solve.quadrature import QuadratureSpec
from liquiditylife.solve.solver import SolverConfig, solve_model
from liquiditylife.vizdata.export import (
    simulation_to_xarray,
    solution_to_dataframe,
    to_json_payload,
)


@pytest.mark.slow
class TestVizDataExport:
    @pytest.fixture(scope="class")
    def solved(self) -> object:
        cal = load_calibration("toy_demo_small_grid")
        config = SolverConfig(
            grid_spec=GridSpec(
                x_points=3, m_points=5, cm_points=3,
                x_min=-0.02, x_max=0.12, m_max=10.0,
            ),
            quad_spec=QuadratureSpec(n_xi=3, n_ncf=3, n_eta=3, n_eps=3),
            n_c_grid=20,
            n_theta_grid=5,
            verbose=False,
        )
        return solve_model(cal, config)

    def test_solution_to_dataframe_shape(self, solved: object) -> None:
        df = solution_to_dataframe(solved, ages=[25, 30])  # type: ignore[arg-type]
        assert len(df) > 0
        assert set(df.columns) == {
            "age", "x_t", "m_t", "cm_t", "consumption", "stock_share", "value"
        }

    def test_solution_to_dataframe_all_ages(self, solved: object) -> None:
        df = solution_to_dataframe(solved)  # type: ignore[arg-type]
        assert len(df) > 0

    def test_simulation_to_xarray(self, solved: object) -> None:
        sim = simulate_cohorts(solved, n_households=20, seed=42)  # type: ignore[arg-type]
        ds = simulation_to_xarray(sim)
        assert "wealth" in ds
        assert "stock_share" in ds
        assert ds.sizes["household"] == 20

    def test_to_json_payload(self, solved: object) -> None:
        df = solution_to_dataframe(solved, ages=[25])  # type: ignore[arg-type]
        json_str = to_json_payload(df)
        assert isinstance(json_str, str)
        assert len(json_str) > 0
