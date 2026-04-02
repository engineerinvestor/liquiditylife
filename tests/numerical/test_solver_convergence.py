"""Tests for end-to-end solver on toy calibration."""

import numpy as np
import pytest

from liquiditylife.calibrations.registry import load_calibration
from liquiditylife.solve.grids import GridSpec
from liquiditylife.solve.quadrature import QuadratureSpec
from liquiditylife.solve.solver import SolverConfig, solve_model


@pytest.mark.slow
class TestSolverToyDemo:
    def test_solve_completes(self) -> None:
        """Solve a tiny grid and verify basic properties."""
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

        # Should have policies for every age
        for age in cal.lifecycle.ages:
            assert age in solution.policies

        # Consumption should be positive
        for age, pf in solution.policies.items():
            assert np.all(pf.consumption_grid > 0), f"Negative consumption at age {age}"

        # Stock share should be in [0, 1]
        for age, pf in solution.policies.items():
            assert np.all(pf.stock_share_grid >= 0), f"Negative stock share at age {age}"
            assert np.all(pf.stock_share_grid <= 1), f"Stock share > 1 at age {age}"

        # Value function should be positive
        for age, V in solution.value_functions.items():
            assert np.all(V > 0), f"Non-positive value at age {age}"

    def test_consumption_monotone_in_m(self) -> None:
        """Consumption should generally increase with cash-on-hand."""
        cal = load_calibration("toy_demo_small_grid")
        config = SolverConfig(
            grid_spec=GridSpec(
                x_points=3, m_points=8, cm_points=3,
                x_min=-0.02, x_max=0.12, m_max=10.0,
            ),
            quad_spec=QuadratureSpec(n_xi=3, n_ncf=3, n_eta=3, n_eps=3),
            n_c_grid=30,
            n_theta_grid=5,
            verbose=False,
        )
        solution = solve_model(cal, config)

        # Check monotonicity at a middle age, middle x, middle cm
        mid_age = (cal.lifecycle.age_start + cal.lifecycle.age_max) // 2
        pf = solution.policies[mid_age]
        c_slice = pf.consumption_grid[1, :, 1]  # fix x and cm indices
        # Allow small violations due to coarse grid
        diffs = np.diff(c_slice)
        n_violations = np.sum(diffs < -1e-4)
        assert n_violations <= 1, f"Too many monotonicity violations: {n_violations}"
