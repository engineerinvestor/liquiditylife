"""Tests for interpolation wrappers."""

import numpy as np

from liquiditylife.solve.interpolation import (
    build_interpolator,
    interpolate_policy,
    interpolate_policy_batch,
)


class TestInterpolation:
    def _make_linear_interp(self) -> tuple[object, np.ndarray, np.ndarray, np.ndarray]:
        """Build an interpolator for f(x, m, cm) = x + m + cm."""
        grid_x = np.linspace(0.0, 1.0, 5)
        grid_m = np.linspace(0.0, 1.0, 5)
        grid_cm = np.linspace(0.0, 1.0, 5)
        xx, mm, cc = np.meshgrid(grid_x, grid_m, grid_cm, indexing="ij")
        values = xx + mm + cc
        interp = build_interpolator(grid_x, grid_m, grid_cm, values)
        return interp, grid_x, grid_m, grid_cm

    def test_on_grid_exact(self) -> None:
        interp, *_ = self._make_linear_interp()
        # f(0.5, 0.5, 0.5) = 1.5
        result = interpolate_policy(interp, 0.5, 0.5, 0.5)  # type: ignore[arg-type]
        assert abs(result - 1.5) < 1e-10

    def test_off_grid_accurate(self) -> None:
        interp, *_ = self._make_linear_interp()
        # Linear function => interpolation should be exact
        result = interpolate_policy(interp, 0.3, 0.7, 0.1)  # type: ignore[arg-type]
        expected = 0.3 + 0.7 + 0.1
        assert abs(result - expected) < 1e-10

    def test_batch_interpolation(self) -> None:
        interp, *_ = self._make_linear_interp()
        points = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [0.5, 0.5, 0.5]])
        results = interpolate_policy_batch(interp, points)  # type: ignore[arg-type]
        expected = np.array([0.0, 3.0, 1.5])
        np.testing.assert_allclose(results, expected, atol=1e-10)

    def test_clamping_outside_bounds(self) -> None:
        interp, *_ = self._make_linear_interp()
        # Outside bounds should clamp, not error
        result = interpolate_policy(interp, 2.0, 2.0, 2.0)  # type: ignore[arg-type]
        # Clamped to (1, 1, 1) -> f = 3.0
        assert abs(result - 3.0) < 1e-10
