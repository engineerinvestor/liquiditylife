"""Tests for Numba-accelerated kernels."""

from __future__ import annotations

import numpy as np
import pytest

numba = pytest.importorskip("numba")

from liquiditylife.solve.numba_kernels import (  # noqa: E402
    HAS_NUMBA,
    interp_3d_linear_jit,
)


class TestNumbaAvailability:
    def test_numba_detected(self) -> None:
        assert HAS_NUMBA

    def test_jit_functions_created(self) -> None:
        assert interp_3d_linear_jit is not None


class TestInterp3DLinear:
    def test_on_grid_exact(self) -> None:
        grid_x = np.array([0.0, 1.0], dtype=np.float64)
        grid_m = np.array([0.0, 1.0], dtype=np.float64)
        grid_cm = np.array([0.0, 1.0], dtype=np.float64)
        values = np.array([[[0.0, 1.0], [2.0, 3.0]], [[4.0, 5.0], [6.0, 7.0]]])

        pts_x = np.array([0.0, 1.0], dtype=np.float64)
        pts_m = np.array([0.0, 1.0], dtype=np.float64)
        pts_cm = np.array([0.0, 1.0], dtype=np.float64)

        assert interp_3d_linear_jit is not None
        result = interp_3d_linear_jit(grid_x, grid_m, grid_cm, values, pts_x, pts_m, pts_cm)  # type: ignore[operator]
        np.testing.assert_allclose(result, [0.0, 7.0])

    def test_midpoint_interpolation(self) -> None:
        grid_x = np.array([0.0, 1.0], dtype=np.float64)
        grid_m = np.array([0.0, 1.0], dtype=np.float64)
        grid_cm = np.array([0.0, 1.0], dtype=np.float64)
        # f(x,m,cm) = x + m + cm
        values = np.zeros((2, 2, 2), dtype=np.float64)
        for i, x in enumerate(grid_x):
            for j, m in enumerate(grid_m):
                for k, cm in enumerate(grid_cm):
                    values[i, j, k] = x + m + cm

        pts_x = np.array([0.5], dtype=np.float64)
        pts_m = np.array([0.5], dtype=np.float64)
        pts_cm = np.array([0.5], dtype=np.float64)

        assert interp_3d_linear_jit is not None
        result = interp_3d_linear_jit(grid_x, grid_m, grid_cm, values, pts_x, pts_m, pts_cm)  # type: ignore[operator]
        np.testing.assert_allclose(result, [1.5], atol=1e-10)

    def test_clamping(self) -> None:
        grid_x = np.array([0.0, 1.0], dtype=np.float64)
        grid_m = np.array([0.0, 1.0], dtype=np.float64)
        grid_cm = np.array([0.0, 1.0], dtype=np.float64)
        values = np.ones((2, 2, 2), dtype=np.float64) * 5.0

        pts_x = np.array([10.0], dtype=np.float64)
        pts_m = np.array([10.0], dtype=np.float64)
        pts_cm = np.array([10.0], dtype=np.float64)

        assert interp_3d_linear_jit is not None
        result = interp_3d_linear_jit(grid_x, grid_m, grid_cm, values, pts_x, pts_m, pts_cm)  # type: ignore[operator]
        np.testing.assert_allclose(result, [5.0])

    def test_matches_scipy(self) -> None:
        """Numba interpolator should match scipy's RegularGridInterpolator."""
        from scipy.interpolate import RegularGridInterpolator

        rng = np.random.default_rng(42)
        grid_x = np.linspace(0.0, 1.0, 5)
        grid_m = np.linspace(0.0, 2.0, 8)
        grid_cm = np.linspace(0.0, 1.5, 6)

        values = rng.random((5, 8, 6))
        scipy_interp = RegularGridInterpolator(
            (grid_x, grid_m, grid_cm), values,
            method="linear", bounds_error=False, fill_value=None,
        )

        # Random query points within bounds
        n_pts = 50
        pts_x = rng.uniform(0.0, 1.0, n_pts)
        pts_m = rng.uniform(0.0, 2.0, n_pts)
        pts_cm = rng.uniform(0.0, 1.5, n_pts)

        scipy_result = scipy_interp(np.column_stack([pts_x, pts_m, pts_cm]))

        assert interp_3d_linear_jit is not None
        numba_result = interp_3d_linear_jit(grid_x, grid_m, grid_cm, values, pts_x, pts_m, pts_cm)  # type: ignore[operator]

        np.testing.assert_allclose(numba_result, scipy_result, atol=1e-10)


@pytest.mark.slow
class TestNumbaVsPythonSolver:
    def test_results_match(self) -> None:
        """Numba and Python paths should produce similar policies."""
        from liquiditylife.calibrations.registry import load_calibration
        from liquiditylife.solve.bellman import bellman_operator
        from liquiditylife.solve.grids import GridSpec, build_grids
        from liquiditylife.solve.quadrature import QuadratureSpec, build_shock_grid

        cal = load_calibration("toy_demo_small_grid")
        spec = GridSpec(x_points=3, m_points=5, cm_points=3, x_min=-0.02, x_max=0.12, m_max=10.0)
        grid_x, grid_m, grid_cm = build_grids(spec)
        quad = QuadratureSpec(n_xi=3, n_ncf=3, n_eta=3, n_eps=3)
        sg = build_shock_grid(cal, quad)

        # Terminal age — should match exactly
        V_numba, C_numba, th_numba = bellman_operator(
            age=cal.lifecycle.age_max,
            grid_x=grid_x, grid_m=grid_m, grid_cm=grid_cm,
            v_next_interp=None, cal=cal, shock_grid=sg,
            n_c_grid=20, n_theta_grid=5,
        )

        assert np.all(C_numba > 0)
        assert np.all(th_numba == 0.0)  # terminal: no stocks
        assert np.all(V_numba > 0)
