"""Interpolation wrappers for value and policy functions."""

from __future__ import annotations

import logging

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from liquiditylife._types import ArrayFloat

logger = logging.getLogger(__name__)


def build_interpolator(
    grid_x: ArrayFloat,
    grid_m: ArrayFloat,
    grid_cm: ArrayFloat,
    values: ArrayFloat,
) -> RegularGridInterpolator:
    """Build a 3D linear interpolator over (x_t, m_t, cm_t).

    Uses nearest-neighbor extrapolation outside grid bounds.

    Args:
        grid_x: 1D array of equity premium grid points (sorted).
        grid_m: 1D array of cash-on-hand grid points (sorted).
        grid_cm: 1D array of lagged consumption ratio grid points (sorted).
        values: 3D array of shape ``(len(grid_x), len(grid_m), len(grid_cm))``.
    """
    return RegularGridInterpolator(
        (grid_x, grid_m, grid_cm),
        values,
        method="linear",
        bounds_error=False,
        fill_value=None,  # use nearest extrapolation
    )


def interpolate_policy(
    interp: RegularGridInterpolator,
    x_t: float,
    m_t: float,
    cm_t: float,
) -> float:
    """Evaluate the interpolator at a single point with clamping."""
    point = _clamp_point(interp, x_t, m_t, cm_t)
    return float(interp(point.reshape(1, 3))[0])


def interpolate_policy_batch(
    interp: RegularGridInterpolator,
    points: ArrayFloat,
) -> ArrayFloat:
    """Evaluate the interpolator at multiple points.

    Args:
        interp: The interpolator object.
        points: Array of shape ``(n, 3)`` with columns (x_t, m_t, cm_t).
    """
    clamped = _clamp_points(interp, points)
    result: ArrayFloat = np.asarray(interp(clamped), dtype=np.float64)
    return result


def _clamp_point(
    interp: RegularGridInterpolator, x_t: float, m_t: float, cm_t: float
) -> ArrayFloat:
    """Clamp a single point to the interpolator's grid bounds."""
    grids = interp.grid
    return np.array(
        [
            np.clip(x_t, grids[0][0], grids[0][-1]),
            np.clip(m_t, grids[1][0], grids[1][-1]),
            np.clip(cm_t, grids[2][0], grids[2][-1]),
        ],
        dtype=np.float64,
    )


def _clamp_points(
    interp: RegularGridInterpolator, points: ArrayFloat
) -> ArrayFloat:
    """Clamp an array of points to the interpolator's grid bounds."""
    grids = interp.grid
    clamped = points.copy()
    for dim in range(3):
        clamped[:, dim] = np.clip(clamped[:, dim], grids[dim][0], grids[dim][-1])
    return clamped
