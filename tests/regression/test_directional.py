"""Directional tests: policy functions should satisfy economic monotonicity."""

import numpy as np
import pytest

from liquiditylife.calibrations.registry import load_calibration
from liquiditylife.solve.grids import GridSpec
from liquiditylife.solve.quadrature import QuadratureSpec
from liquiditylife.solve.solver import SolvedModel, SolverConfig, solve_model


@pytest.mark.slow
class TestDirectionalProperties:
    @pytest.fixture(scope="class")
    def solution(self) -> SolvedModel:
        cal = load_calibration("toy_demo_small_grid")
        config = SolverConfig(
            grid_spec=GridSpec(
                x_points=3, m_points=8, cm_points=3,
                x_min=-0.02, x_max=0.12, m_max=10.0,
            ),
            quad_spec=QuadratureSpec(n_xi=3, n_ncf=3, n_eta=3, n_eps=3),
            n_c_grid=30,
            n_theta_grid=7,
            verbose=False,
        )
        return solve_model(cal, config)

    def test_consumption_weakly_increasing_in_m(self, solution: SolvedModel) -> None:
        """Consumption should generally increase with cash-on-hand."""
        mid_age = (solution.calibration.lifecycle.age_start
                   + solution.calibration.lifecycle.age_max) // 2
        pf = solution.policies[mid_age]
        for ix in range(len(pf.grid_x)):
            for icm in range(len(pf.grid_cm)):
                c_slice = pf.consumption_grid[ix, :, icm]
                diffs = np.diff(c_slice)
                violations = np.sum(diffs < -0.01)
                assert violations <= 1, (
                    f"Consumption not monotone in m at ix={ix}, icm={icm}"
                )

    def test_value_increasing_in_m(self, solution: SolvedModel) -> None:
        """Value function should increase with cash-on-hand."""
        mid_age = (solution.calibration.lifecycle.age_start
                   + solution.calibration.lifecycle.age_max) // 2
        V = solution.value_functions[mid_age]
        for ix in range(V.shape[0]):
            for icm in range(V.shape[2]):
                v_slice = V[ix, :, icm]
                diffs = np.diff(v_slice)
                violations = np.sum(diffs < -1e-6)
                assert violations <= 1, (
                    f"Value not monotone in m at ix={ix}, icm={icm}"
                )
