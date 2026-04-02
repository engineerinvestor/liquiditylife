"""Tests for state grid construction."""

import numpy as np

from liquiditylife.calibrations.registry import load_calibration
from liquiditylife.solve.grids import (
    GridSpec,
    build_cm_grid,
    build_grids,
    build_m_grid,
    build_x_grid,
    default_grid_spec,
)


class TestGridConstruction:
    def test_x_grid_shape(self) -> None:
        spec = GridSpec()
        grid = build_x_grid(spec)
        assert len(grid) == spec.x_points

    def test_x_grid_monotone(self) -> None:
        grid = build_x_grid(GridSpec())
        assert np.all(np.diff(grid) > 0)

    def test_m_grid_shape(self) -> None:
        spec = GridSpec()
        grid = build_m_grid(spec)
        assert len(grid) == spec.m_points

    def test_m_grid_monotone(self) -> None:
        grid = build_m_grid(GridSpec())
        assert np.all(np.diff(grid) > 0)

    def test_m_grid_dense_at_low(self) -> None:
        grid = build_m_grid(GridSpec())
        # Spacing should be smaller at low end (geomspace)
        assert grid[1] - grid[0] < grid[-1] - grid[-2]

    def test_cm_grid_shape(self) -> None:
        spec = GridSpec()
        grid = build_cm_grid(spec)
        assert len(grid) == spec.cm_points

    def test_cm_grid_monotone(self) -> None:
        grid = build_cm_grid(GridSpec())
        assert np.all(np.diff(grid) > 0)

    def test_build_grids_returns_three(self) -> None:
        grids = build_grids(GridSpec())
        assert len(grids) == 3

    def test_m_grid_bounds(self) -> None:
        spec = GridSpec(m_min=0.01, m_max=30.0)
        grid = build_m_grid(spec)
        assert np.isclose(grid[0], 0.01)
        assert np.isclose(grid[-1], 30.0)


class TestDefaultGridSpec:
    def test_default_grid_spec_from_calibration(self) -> None:
        cal = load_calibration("adams_baseline")
        spec = default_grid_spec(cal)
        assert spec.x_min < cal.asset_returns.x_bar
        assert spec.x_max > cal.asset_returns.x_bar
