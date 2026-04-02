"""Policy function representation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

import numpy as np

if TYPE_CHECKING:
    from liquiditylife._types import ArrayFloat


class _Interpolator(Protocol):
    def __call__(self, xi: ArrayFloat) -> ArrayFloat: ...


class PolicyFunction:
    """Solved policy and value function at a single age.

    Stores 3D arrays over (x_t, m_t, cm_t) grids. Interpolation
    is set up via ``set_interpolator`` after the solve phase builds
    the interpolation objects.
    """

    def __init__(
        self,
        age: int,
        consumption: ArrayFloat,
        stock_share: ArrayFloat,
        value: ArrayFloat,
        grid_x: ArrayFloat,
        grid_m: ArrayFloat,
        grid_cm: ArrayFloat,
    ) -> None:
        self.age = age
        self._consumption = consumption
        self._stock_share = stock_share
        self._value = value
        self.grid_x = grid_x
        self.grid_m = grid_m
        self.grid_cm = grid_cm
        self._interp_c: _Interpolator | None = None
        self._interp_theta: _Interpolator | None = None
        self._interp_v: _Interpolator | None = None

    def set_interpolators(
        self,
        interp_c: _Interpolator,
        interp_theta: _Interpolator,
        interp_v: _Interpolator,
    ) -> None:
        """Attach interpolation callables for off-grid queries."""
        self._interp_c = interp_c
        self._interp_theta = interp_theta
        self._interp_v = interp_v

    def consume(self, x_t: float, m_t: float, cm_t: float) -> float:
        """Optimal consumption at (x_t, m_t, cm_t)."""
        if self._interp_c is None:
            msg = "Interpolator not set; call set_interpolators first"
            raise RuntimeError(msg)
        return float(self._interp_c(np.array([[x_t, m_t, cm_t]])))

    def stock_share_at(self, x_t: float, m_t: float, cm_t: float) -> float:
        """Optimal stock share at (x_t, m_t, cm_t)."""
        if self._interp_theta is None:
            msg = "Interpolator not set; call set_interpolators first"
            raise RuntimeError(msg)
        return float(self._interp_theta(np.array([[x_t, m_t, cm_t]])))

    def value_at(self, x_t: float, m_t: float, cm_t: float) -> float:
        """Value function at (x_t, m_t, cm_t)."""
        if self._interp_v is None:
            msg = "Interpolator not set; call set_interpolators first"
            raise RuntimeError(msg)
        return float(self._interp_v(np.array([[x_t, m_t, cm_t]])))

    @property
    def consumption_grid(self) -> ArrayFloat:
        """Raw consumption policy array."""
        return self._consumption

    @property
    def stock_share_grid(self) -> ArrayFloat:
        """Raw stock share policy array."""
        return self._stock_share

    @property
    def value_grid(self) -> ArrayFloat:
        """Raw value function array."""
        return self._value
