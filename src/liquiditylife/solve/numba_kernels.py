"""Optional Numba-accelerated inner loops for the Bellman operator.

When Numba is installed, the solver dispatches to JIT-compiled kernels
that replace scipy's interpolator and Python loop overhead with compiled
machine code. Falls back to pure-Python/NumPy when Numba is unavailable.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from liquiditylife._types import ArrayFloat

try:
    import numba  # type: ignore

    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False


def _make_jit_functions() -> (
    tuple[object, object] | None
):
    """Create JIT-compiled functions. Returns None if Numba unavailable."""
    if not HAS_NUMBA:
        return None

    @numba.njit(cache=True)  # type: ignore[untyped-decorator]
    def _interp_3d_linear(
        grid_x: ArrayFloat,
        grid_m: ArrayFloat,
        grid_cm: ArrayFloat,
        values: ArrayFloat,
        pts_x: ArrayFloat,
        pts_m: ArrayFloat,
        pts_cm: ArrayFloat,
    ) -> ArrayFloat:
        """3D trilinear interpolation with clamping (Numba-compiled)."""
        n = len(pts_x)
        result = np.empty(n, dtype=np.float64)
        nx = len(grid_x)
        nm = len(grid_m)
        ncm = len(grid_cm)

        for i in range(n):
            # Clamp to grid bounds
            x = min(max(pts_x[i], grid_x[0]), grid_x[nx - 1])
            m = min(max(pts_m[i], grid_m[0]), grid_m[nm - 1])
            c = min(max(pts_cm[i], grid_cm[0]), grid_cm[ncm - 1])

            # Find cell indices via binary search
            ix = np.searchsorted(grid_x, x) - 1
            ix = min(max(ix, 0), nx - 2)
            im = np.searchsorted(grid_m, m) - 1
            im = min(max(im, 0), nm - 2)
            ic = np.searchsorted(grid_cm, c) - 1
            ic = min(max(ic, 0), ncm - 2)

            # Interpolation weights
            dx = grid_x[ix + 1] - grid_x[ix]
            wx = (x - grid_x[ix]) / dx if dx > 0 else 0.0

            dm = grid_m[im + 1] - grid_m[im]
            wm = (m - grid_m[im]) / dm if dm > 0 else 0.0

            dc = grid_cm[ic + 1] - grid_cm[ic]
            wc = (c - grid_cm[ic]) / dc if dc > 0 else 0.0

            # Trilinear interpolation (8 corners)
            v000 = values[ix, im, ic]
            v001 = values[ix, im, ic + 1]
            v010 = values[ix, im + 1, ic]
            v011 = values[ix, im + 1, ic + 1]
            v100 = values[ix + 1, im, ic]
            v101 = values[ix + 1, im, ic + 1]
            v110 = values[ix + 1, im + 1, ic]
            v111 = values[ix + 1, im + 1, ic + 1]

            v00 = v000 * (1.0 - wc) + v001 * wc
            v01 = v010 * (1.0 - wc) + v011 * wc
            v10 = v100 * (1.0 - wc) + v101 * wc
            v11 = v110 * (1.0 - wc) + v111 * wc

            v0 = v00 * (1.0 - wm) + v01 * wm
            v1 = v10 * (1.0 - wm) + v11 * wm

            result[i] = v0 * (1.0 - wx) + v1 * wx

        return result

    @numba.njit(cache=True)  # type: ignore[untyped-decorator]
    def bellman_operator_numba(
        grid_x: ArrayFloat,
        grid_m: ArrayFloat,
        grid_cm: ArrayFloat,
        v_next: ArrayFloat,
        # Shock arrays (flattened tensor product)
        shock_xi: ArrayFloat,
        shock_ncf: ArrayFloat,
        shock_ndr: ArrayFloat,
        shock_eta: ArrayFloat,
        shock_eps: ArrayFloat,
        shock_weights: ArrayFloat,
        # Scalar model parameters
        gamma: float,
        beta: float,
        psi: float,
        rf: float,
        x_bar: float,
        phi_x: float,
        phi_c: float,
        illiq_disp_share: float,
        illiq_S: float,
        age_drift_val: float,
        is_terminal: bool,
        is_next_retired: bool,
        # Search grid sizes
        n_c_grid: int,
        n_theta_grid: int,
    ) -> tuple[ArrayFloat, ArrayFloat, ArrayFloat]:
        """Bellman operator for a single age (Numba-compiled)."""
        nx = len(grid_x)
        nm = len(grid_m)
        ncm = len(grid_cm)
        n_shocks = len(shock_weights)
        EPS = 1e-10
        SAVINGS_FLOOR = 1e-8

        V = np.full((nx, nm, ncm), -np.inf, dtype=np.float64)
        C_pol = np.zeros((nx, nm, ncm), dtype=np.float64)
        theta_pol = np.zeros((nx, nm, ncm), dtype=np.float64)

        # EZ utility parameter
        rho = 1.0 - 1.0 / psi

        # Pre-build theta grid
        theta_grid = np.linspace(0.0, 1.0, n_theta_grid)

        for ix in range(nx):
            x_t = grid_x[ix]
            for im in range(nm):
                m_t = grid_m[im]
                for icm in range(ncm):
                    cm_t = grid_cm[icm]

                    if is_terminal:
                        V[ix, im, icm] = max(m_t, EPS)
                        C_pol[ix, im, icm] = m_t
                        theta_pol[ix, im, icm] = 0.0
                        continue

                    # Feasible consumption range
                    c_min = 1e-4
                    c_max = m_t - SAVINGS_FLOOR
                    if c_max <= c_min:
                        V[ix, im, icm] = max(m_t, EPS)
                        C_pol[ix, im, icm] = m_t
                        theta_pol[ix, im, icm] = 0.0
                        continue

                    c_grid = np.linspace(c_min, c_max, n_c_grid)

                    best_v = -np.inf
                    best_c = c_min
                    best_theta = 0.0

                    for ic in range(n_c_grid):
                        c_t_val = c_grid[ic]

                        # Inline adjustment cost
                        if phi_c > 0.0 and c_t_val < cm_t:
                            shortfall = cm_t - c_t_val
                            adj_cost = (phi_c / 2.0) * shortfall * shortfall / cm_t
                        else:
                            adj_cost = 0.0

                        savings = m_t - c_t_val - adj_cost
                        if savings < SAVINGS_FLOOR:
                            continue

                        for ith in range(n_theta_grid):
                            theta_val = theta_grid[ith]

                            # Compute next-period states for all shocks
                            pts_x = np.empty(n_shocks)
                            pts_m = np.empty(n_shocks)
                            pts_cm = np.empty(n_shocks)

                            for s in range(n_shocks):
                                # Next equity premium
                                x_next = x_bar + phi_x * (x_t - x_bar) + shock_xi[s]

                                # Stock return and portfolio return
                                r_stock = rf + x_t + shock_ncf[s] - shock_ndr[s]
                                r_port = 1.0 + rf + theta_val * (r_stock - rf)

                                # Earnings growth
                                y_ratio = np.exp(age_drift_val + shock_eta[s])

                                # Next cash-on-hand
                                financial = savings * r_port / y_ratio
                                if is_next_retired:
                                    m_next = financial + illiq_S
                                else:
                                    m_next = (
                                        financial
                                        + illiq_disp_share * np.exp(shock_eps[s])
                                    )

                                pts_x[s] = x_next
                                pts_m[s] = max(m_next, EPS)
                                pts_cm[s] = max(c_t_val / y_ratio, EPS)

                            # Interpolate V_next at all shock nodes
                            v_next_vals = _interp_3d_linear(
                                grid_x, grid_m, grid_cm, v_next,
                                pts_x, pts_m, pts_cm,
                            )

                            # Floor values
                            for s in range(n_shocks):
                                if v_next_vals[s] < EPS:
                                    v_next_vals[s] = EPS

                            # Certainty equivalent
                            one_minus_gamma = 1.0 - gamma
                            expectation = 0.0
                            for s in range(n_shocks):
                                expectation += (
                                    shock_weights[s]
                                    * v_next_vals[s] ** one_minus_gamma
                                )

                            if expectation <= 0.0:
                                continue

                            ce = expectation ** (1.0 / one_minus_gamma)

                            # EZ utility
                            c_safe = max(c_t_val, EPS)
                            ce_safe = max(ce, EPS)
                            term_c = (1.0 - beta) * c_safe**rho
                            term_v = beta * ce_safe**rho
                            v_now = max((term_c + term_v) ** (1.0 / rho), EPS)

                            if v_now > best_v:
                                best_v = v_now
                                best_c = c_t_val
                                best_theta = theta_val

                    if best_v > -np.inf:
                        V[ix, im, icm] = best_v
                        C_pol[ix, im, icm] = best_c
                        theta_pol[ix, im, icm] = best_theta
                    else:
                        V[ix, im, icm] = max(m_t, EPS)
                        C_pol[ix, im, icm] = m_t
                        theta_pol[ix, im, icm] = 0.0

        return V, C_pol, theta_pol

    return _interp_3d_linear, bellman_operator_numba


# Create JIT functions at import time (lazy compilation on first call)
_jit_funcs = _make_jit_functions()

if _jit_funcs is not None:
    interp_3d_linear_jit, bellman_operator_numba_jit = _jit_funcs
else:
    interp_3d_linear_jit = None
    bellman_operator_numba_jit = None
