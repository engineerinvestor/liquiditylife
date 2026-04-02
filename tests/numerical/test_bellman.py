"""Tests for the Bellman operator."""

import numpy as np

from liquiditylife.calibrations.registry import load_calibration
from liquiditylife.solve.bellman import bellman_operator
from liquiditylife.solve.grids import GridSpec, build_grids
from liquiditylife.solve.quadrature import QuadratureSpec, build_shock_grid


class TestBellmanTerminal:
    def test_terminal_age_consumes_all(self) -> None:
        cal = load_calibration("toy_demo_small_grid")
        spec = GridSpec(x_points=3, m_points=5, cm_points=3, m_max=10.0)
        grid_x, grid_m, grid_cm = build_grids(spec)
        quad = QuadratureSpec(n_xi=3, n_ncf=3, n_eta=3, n_eps=3)
        sg = build_shock_grid(cal, quad)

        V, C, theta = bellman_operator(
            age=cal.lifecycle.age_max,
            grid_x=grid_x,
            grid_m=grid_m,
            grid_cm=grid_cm,
            v_next_interp=None,
            cal=cal,
            shock_grid=sg,
            n_c_grid=20,
            n_theta_grid=5,
        )

        # At terminal age, should consume everything
        for im in range(len(grid_m)):
            np.testing.assert_allclose(
                C[:, im, :], grid_m[im], rtol=1e-8
            )
        # Stock share should be 0 at terminal
        np.testing.assert_allclose(theta, 0.0)
        # Value should be positive
        assert np.all(V > 0)
