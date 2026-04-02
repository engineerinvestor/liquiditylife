"""Tests for solution IO/caching."""

from __future__ import annotations

from pathlib import Path

import pytest

from liquiditylife.calibrations.registry import load_calibration
from liquiditylife.io.cache import load_solution, save_solution
from liquiditylife.solve.grids import GridSpec
from liquiditylife.solve.quadrature import QuadratureSpec
from liquiditylife.solve.solver import SolverConfig, solve_model


@pytest.mark.slow
class TestIOCache:
    def test_save_load_round_trip(self, tmp_path: Path) -> None:
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
        solution = solve_model(cal, config)

        save_path = tmp_path / "test_solution.pkl"
        save_solution(solution, save_path)
        assert save_path.exists()

        loaded = load_solution(save_path)
        assert loaded.calibration.name == solution.calibration.name
        assert loaded.calibration.fingerprint() == solution.calibration.fingerprint()
        assert set(loaded.policies.keys()) == set(solution.policies.keys())
