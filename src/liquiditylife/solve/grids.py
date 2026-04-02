"""State grid construction for the dynamic programming solver."""

from __future__ import annotations

import math

import numpy as np
from pydantic import BaseModel

from liquiditylife._types import ArrayFloat
from liquiditylife.calibrations.bundles import CalibrationBundle


class GridSpec(BaseModel, frozen=True):
    """Specification for the discretized state grids."""

    x_points: int = 15
    """Number of grid points for the equity premium state."""

    m_points: int = 40
    """Number of grid points for scaled cash-on-hand."""

    cm_points: int = 10
    """Number of grid points for the lagged consumption ratio."""

    x_min: float = -0.05
    """Lower bound of the equity premium grid."""

    x_max: float = 0.15
    """Upper bound of the equity premium grid."""

    m_min: float = 0.01
    """Lower bound of scaled cash-on-hand grid."""

    m_max: float = 30.0
    """Upper bound of scaled cash-on-hand grid."""

    cm_min: float = 0.01
    """Lower bound of the lagged consumption ratio grid."""

    cm_max: float = 3.0
    """Upper bound of the lagged consumption ratio grid."""


def build_x_grid(spec: GridSpec) -> ArrayFloat:
    """Build a linearly-spaced grid for the equity premium state."""
    return np.linspace(spec.x_min, spec.x_max, spec.x_points, dtype=np.float64)


def build_m_grid(spec: GridSpec) -> ArrayFloat:
    """Build a geometrically-spaced grid for cash-on-hand (dense at low values)."""
    return np.geomspace(spec.m_min, spec.m_max, spec.m_points, dtype=np.float64)


def build_cm_grid(spec: GridSpec) -> ArrayFloat:
    """Build a linearly-spaced grid for the lagged consumption ratio."""
    return np.linspace(spec.cm_min, spec.cm_max, spec.cm_points, dtype=np.float64)


def build_grids(spec: GridSpec) -> tuple[ArrayFloat, ArrayFloat, ArrayFloat]:
    """Build all three state grids from the specification."""
    return build_x_grid(spec), build_m_grid(spec), build_cm_grid(spec)


def default_grid_spec(cal: CalibrationBundle) -> GridSpec:
    """Derive sensible grid bounds from a calibration bundle.

    The equity premium range is set to x_bar +/- 3 * unconditional std dev.
    """
    ar = cal.asset_returns
    sigma_unc = ar.sigma_xi / math.sqrt(1.0 - ar.phi_x**2)
    x_min = ar.x_bar - 3.0 * sigma_unc
    x_max = ar.x_bar + 3.0 * sigma_unc
    return GridSpec(x_min=x_min, x_max=x_max)
