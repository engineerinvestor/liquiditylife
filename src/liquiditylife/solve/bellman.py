"""Single-period Bellman operator for backward induction."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from liquiditylife.calibrations.bundles import CalibrationBundle
from liquiditylife.core.preferences import Preferences
from liquiditylife.model.budget import (
    end_of_period_savings,
)
from liquiditylife.model.utility import ez_utility, terminal_utility
from liquiditylife.processes.adjustment_cost import AdjustmentCostModel
from liquiditylife.processes.asset_returns import AssetReturnProcess
from liquiditylife.processes.illiquid import IlliquidWealthRule
from liquiditylife.processes.income import IncomeProcess
from liquiditylife.solve.quadrature import ShockGrid

if TYPE_CHECKING:
    from liquiditylife._types import ArrayFloat

# Minimum savings floor
_SAVINGS_FLOOR = 1e-8


def bellman_operator(
    age: int,
    grid_x: ArrayFloat,
    grid_m: ArrayFloat,
    grid_cm: ArrayFloat,
    v_next_interp: RegularGridInterpolator | None,
    cal: CalibrationBundle,
    shock_grid: ShockGrid,
    n_c_grid: int = 80,
    n_theta_grid: int = 21,
) -> tuple[ArrayFloat, ArrayFloat, ArrayFloat]:
    """Solve the Bellman equation at a single age for all state grid points.

    Args:
        age: Current age.
        grid_x: Equity premium grid.
        grid_m: Cash-on-hand grid.
        grid_cm: Lagged consumption ratio grid.
        v_next_interp: Interpolator for next-period value function.
            None at the terminal age.
        cal: Calibration bundle.
        shock_grid: Precomputed quadrature nodes and weights.
        n_c_grid: Number of consumption grid points for search.
        n_theta_grid: Number of stock share grid points for search.

    Returns:
        Tuple of (value, consumption_policy, stock_share_policy) arrays,
        each of shape ``(n_x, n_m, n_cm)``.
    """
    # Try Numba-accelerated path
    from liquiditylife.solve.numba_kernels import (
        HAS_NUMBA,
        bellman_operator_numba_jit,
    )

    prefs = cal.preferences
    adj = cal.adjustment_cost
    ar = cal.asset_returns
    ip = cal.income
    lc = cal.lifecycle
    illiq = cal.illiquid

    is_terminal = age == lc.age_max
    is_next_retired = lc.is_retired(age + 1) if not is_terminal else False

    if HAS_NUMBA and bellman_operator_numba_jit is not None:
        # Extract V_next array for Numba (or zeros for terminal)
        if v_next_interp is not None:
            v_next_arr = np.asarray(v_next_interp.values, dtype=np.float64)
        else:
            v_next_arr = np.zeros(
                (len(grid_x), len(grid_m), len(grid_cm)), dtype=np.float64
            )

        result: tuple[ArrayFloat, ArrayFloat, ArrayFloat] = bellman_operator_numba_jit(  # type: ignore[operator]
            grid_x, grid_m, grid_cm, v_next_arr,
            shock_grid.xi, shock_grid.ncf, shock_grid.ndr,
            shock_grid.eta, shock_grid.eps, shock_grid.weights,
            prefs.gamma, prefs.beta, prefs.psi,
            ar.rf, ar.x_bar, ar.phi_x,
            adj.phi_c,
            illiq.disposable_share, illiq.S,
            ip.age_drift(age),
            is_terminal, is_next_retired,
            n_c_grid, n_theta_grid,
        )
        return result

    # Python fallback path
    n_x = len(grid_x)
    n_m = len(grid_m)
    n_cm = len(grid_cm)

    V = np.full((n_x, n_m, n_cm), -np.inf, dtype=np.float64)
    C_pol = np.zeros((n_x, n_m, n_cm), dtype=np.float64)
    theta_pol = np.zeros((n_x, n_m, n_cm), dtype=np.float64)

    # Stock share grid
    theta_grid = np.linspace(0.0, 1.0, n_theta_grid, dtype=np.float64)

    for ix in range(n_x):
        x_t = float(grid_x[ix])
        for im in range(n_m):
            m_t = float(grid_m[im])
            for icm in range(n_cm):
                cm_t = float(grid_cm[icm])

                if is_terminal:
                    # Consume everything at terminal age
                    V[ix, im, icm] = terminal_utility(m_t, prefs)
                    C_pol[ix, im, icm] = m_t
                    theta_pol[ix, im, icm] = 0.0
                    continue

                # Feasible consumption range
                c_min = 1e-4
                # Max consumption: m_t minus minimum savings
                c_max_raw = m_t - _SAVINGS_FLOOR
                if c_max_raw <= c_min:
                    # Not enough resources — consume everything
                    V[ix, im, icm] = terminal_utility(m_t, prefs)
                    C_pol[ix, im, icm] = m_t
                    theta_pol[ix, im, icm] = 0.0
                    continue

                c_max = c_max_raw
                c_grid = np.geomspace(c_min, c_max, n_c_grid, dtype=np.float64)

                best_v = -np.inf
                best_c = c_min
                best_theta = 0.0

                for c_t in c_grid:
                    savings = end_of_period_savings(
                        m_t, float(c_t), cm_t, adj
                    )
                    if savings < _SAVINGS_FLOOR:
                        continue

                    for theta in theta_grid:
                        # Compute expected continuation value
                        ev = _compute_expected_value(
                            x_t=x_t,
                            savings=savings,
                            theta=float(theta),
                            c_t=float(c_t),
                            age=age,
                            gamma=prefs.gamma,
                            v_next_interp=v_next_interp,  # type: ignore[arg-type]
                            shock_grid=shock_grid,
                            ar=ar,
                            ip=ip,
                            illiq=illiq,
                            is_next_retired=is_next_retired,
                        )

                        if ev <= 0:
                            continue

                        v_now = ez_utility(float(c_t), ev, prefs)

                        if v_now > best_v:
                            best_v = v_now
                            best_c = float(c_t)
                            best_theta = float(theta)

                # Golden-section refinement
                if best_v > -np.inf:
                    c_spacing = (c_max_raw - c_min) / max(n_c_grid - 1, 1)
                    best_c, best_v = _refine_golden(
                        best_c, c_spacing, c_min, c_max_raw,
                        best_theta, x_t, m_t, cm_t,
                        prefs, adj, ar, ip, illiq, age,
                        v_next_interp, shock_grid,  # type: ignore[arg-type]
                        is_next_retired, is_consumption=True,
                    )
                    th_spacing = 1.0 / max(n_theta_grid - 1, 1)
                    best_theta, best_v = _refine_golden(
                        best_theta, th_spacing, 0.0, 1.0,
                        best_c, x_t, m_t, cm_t,
                        prefs, adj, ar, ip, illiq, age,
                        v_next_interp, shock_grid,  # type: ignore[arg-type]
                        is_next_retired, is_consumption=False,
                    )
                    V[ix, im, icm] = best_v
                    C_pol[ix, im, icm] = best_c
                    theta_pol[ix, im, icm] = best_theta
                else:
                    V[ix, im, icm] = terminal_utility(m_t, prefs)
                    C_pol[ix, im, icm] = m_t
                    theta_pol[ix, im, icm] = 0.0

    return V, C_pol, theta_pol


def _eval_python(
    c_val: float,
    theta_val: float,
    x_t: float,
    m_t: float,
    cm_t: float,
    prefs: Preferences,
    adj: AdjustmentCostModel,
    ar: AssetReturnProcess,
    ip: IncomeProcess,
    illiq: IlliquidWealthRule,
    age: int,
    v_next_interp: RegularGridInterpolator,
    shock_grid: ShockGrid,
    is_next_retired: bool,
) -> float:
    """Evaluate Bellman objective at a single (c, theta) pair (Python path)."""
    savings = end_of_period_savings(m_t, c_val, cm_t, adj)
    if savings < _SAVINGS_FLOOR:
        return -np.inf
    ev = _compute_expected_value(
        x_t=x_t, savings=savings, theta=theta_val, c_t=c_val,
        age=age, gamma=prefs.gamma, v_next_interp=v_next_interp,
        shock_grid=shock_grid, ar=ar, ip=ip, illiq=illiq,
        is_next_retired=is_next_retired,
    )
    if ev <= 0:
        return -np.inf
    return ez_utility(c_val, ev, prefs)


def _refine_golden(
    best_val: float,
    spacing: float,
    lo_bound: float,
    hi_bound: float,
    other_val: float,
    x_t: float,
    m_t: float,
    cm_t: float,
    prefs: Preferences,
    adj: AdjustmentCostModel,
    ar: AssetReturnProcess,
    ip: IncomeProcess,
    illiq: IlliquidWealthRule,
    age: int,
    v_next_interp: RegularGridInterpolator,
    shock_grid: ShockGrid,
    is_next_retired: bool,
    *,
    is_consumption: bool,
) -> tuple[float, float]:
    """Golden-section refinement around the grid-search optimum."""
    gr = 0.6180339887498949
    a = max(lo_bound, best_val - spacing)
    b = min(hi_bound, best_val + spacing)

    def obj(v: float) -> float:
        if is_consumption:
            return _eval_python(
                v, other_val, x_t, m_t, cm_t,
                prefs, adj, ar, ip, illiq, age,
                v_next_interp, shock_grid, is_next_retired,
            )
        return _eval_python(
            other_val, v, x_t, m_t, cm_t,
            prefs, adj, ar, ip, illiq, age,
            v_next_interp, shock_grid, is_next_retired,
        )

    c1 = b - gr * (b - a)
    c2 = a + gr * (b - a)
    v1 = obj(c1)
    v2 = obj(c2)
    for _ in range(20):
        if v1 < v2:
            a = c1
            c1, v1 = c2, v2
            c2 = a + gr * (b - a)
            v2 = obj(c2)
        else:
            b = c2
            c2, v2 = c1, v1
            c1 = b - gr * (b - a)
            v1 = obj(c1)

    mid = (a + b) / 2.0
    v_mid = obj(mid)
    return mid, v_mid


def _compute_expected_value(
    x_t: float,
    savings: float,
    theta: float,
    c_t: float,
    age: int,
    gamma: float,
    v_next_interp: RegularGridInterpolator,
    shock_grid: ShockGrid,
    ar: AssetReturnProcess,
    ip: IncomeProcess,
    illiq: IlliquidWealthRule,
    is_next_retired: bool,
) -> float:
    """Compute the certainty equivalent of next-period value.

    CE = (sum_i w_i * V_{t+1}(s'_i)^(1-gamma))^(1/(1-gamma))
    """

    # Vectorised computation over all quadrature nodes
    xi = shock_grid.xi
    ncf = shock_grid.ncf
    eta = shock_grid.eta
    eps = shock_grid.eps
    weights = shock_grid.weights

    # Next-period equity premium
    x_next = ar.x_bar + ar.phi_x * (x_t - ar.x_bar) + xi

    # Realized stock returns
    r_stock = ar.rf + x_t + ncf - shock_grid.ndr

    # Portfolio returns
    r_port = 1.0 + ar.rf + theta * (r_stock - ar.rf)

    # Persistent earnings growth
    g_a = ip.age_drift(age)
    y_ratio = np.exp(g_a + eta)

    # Next-period cash-on-hand
    financial = savings * r_port / y_ratio
    if is_next_retired:
        m_next = financial + illiq.S
    else:
        m_next = financial + illiq.disposable_share * np.exp(eps)

    # Lagged consumption ratio
    cm_next = c_t / y_ratio

    # Clamp to grid bounds
    grids = v_next_interp.grid
    x_next_c = np.clip(x_next, grids[0][0], grids[0][-1])
    m_next_c = np.clip(m_next, grids[1][0], grids[1][-1])
    cm_next_c = np.clip(cm_next, grids[2][0], grids[2][-1])

    # Evaluate next-period value
    points = np.column_stack([x_next_c, m_next_c, cm_next_c])
    v_next_vals: ArrayFloat = np.asarray(v_next_interp(points), dtype=np.float64)

    # Floor values
    v_next_vals = np.maximum(v_next_vals, 1e-10)

    # Certainty equivalent: (sum w * V^(1-gamma))^(1/(1-gamma))
    # For gamma > 1, 1-gamma < 0, so V^(1-gamma) is well-defined for V > 0
    powered = v_next_vals ** (1.0 - gamma)
    expectation = float(np.sum(weights * powered))

    if expectation <= 0:
        return 0.0

    ce = expectation ** (1.0 / (1.0 - gamma))
    return float(ce)
