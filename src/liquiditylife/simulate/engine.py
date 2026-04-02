"""Cohort simulation engine."""

from __future__ import annotations

import numpy as np

from liquiditylife._types import ArrayFloat
from liquiditylife.simulate.result import SimulationPaths, SimulationResult
from liquiditylife.solve.solver import SolvedModel


def simulate_cohorts(
    solution: SolvedModel,
    n_households: int = 10_000,
    seed: int = 42,
) -> SimulationResult:
    """Forward-simulate household cohorts using solved policy functions.

    Args:
        solution: A solved model containing policy functions at every age.
        n_households: Number of households to simulate.
        seed: Random seed for reproducibility.

    Returns:
        A ``SimulationResult`` with full lifecycle paths.
    """
    rng = np.random.default_rng(seed)
    cal = solution.calibration
    lc = cal.lifecycle
    ar = cal.asset_returns
    ip = cal.income
    illiq = cal.illiquid
    adj = cal.adjustment_cost

    n_periods = lc.n_total_periods
    ages = np.arange(lc.age_start, lc.age_max + 1, dtype=np.float64)

    # Allocate path arrays
    wealth = np.zeros((n_households, n_periods), dtype=np.float64)
    consumption = np.zeros((n_households, n_periods), dtype=np.float64)
    stock_share = np.zeros((n_households, n_periods), dtype=np.float64)
    income = np.zeros((n_households, n_periods), dtype=np.float64)
    eq_premium = np.zeros((n_households, n_periods), dtype=np.float64)
    stock_ret = np.zeros((n_households, n_periods), dtype=np.float64)
    log_y_bar = np.zeros((n_households, n_periods), dtype=np.float64)

    # Initial conditions
    x_t = np.full(n_households, ar.x_bar, dtype=np.float64)
    m_t = rng.lognormal(mean=0.0, sigma=0.5, size=n_households)
    cm_t = np.full(n_households, 0.5, dtype=np.float64)
    log_yp = np.zeros(n_households, dtype=np.float64)

    for t in range(n_periods):
        age = lc.age_start + t

        # Record state
        wealth[:, t] = m_t
        eq_premium[:, t] = x_t
        log_y_bar[:, t] = log_yp

        # Look up policy
        pf = solution.policies[age]
        interp_c = pf._interp_c
        interp_theta = pf._interp_theta

        if interp_c is None or interp_theta is None:
            msg = f"Policy interpolators not set for age {age}"
            raise RuntimeError(msg)

        # Query policies (vectorised) — clamp to grid bounds
        x_clamped = np.clip(x_t, pf.grid_x[0], pf.grid_x[-1])
        m_clamped = np.clip(m_t, pf.grid_m[0], pf.grid_m[-1])
        cm_clamped = np.clip(cm_t, pf.grid_cm[0], pf.grid_cm[-1])
        points = np.column_stack([x_clamped, m_clamped, cm_clamped])

        c_t: ArrayFloat = np.maximum(
            np.asarray(interp_c(points), dtype=np.float64), 1e-6
        )
        theta_t: ArrayFloat = np.clip(
            np.asarray(interp_theta(points), dtype=np.float64), 0.0, 1.0
        )

        consumption[:, t] = c_t
        stock_share[:, t] = theta_t

        # Draw shocks for transition
        xi = rng.normal(0.0, ar.sigma_xi, size=n_households)

        # Crash/non-crash mixture for NCF
        crash_mask = rng.random(n_households) < ar.crash.p_crash
        ncf = np.where(
            crash_mask,
            rng.normal(ar.crash.mu_cf_crash, ar.crash.sigma_cf, n_households),
            rng.normal(ar.crash.mu_cf_no_crash, ar.crash.sigma_cf, n_households),
        )
        ndr = ar.rho_cs * xi

        # Income shocks
        eta = rng.normal(0.0, ip.sigma_eta, size=n_households)
        eps = rng.normal(0.0, ip.sigma_eps, size=n_households)

        # Realized stock return
        r_stock = ar.rf + x_t + ncf - ndr
        stock_ret[:, t] = r_stock
        income[:, t] = np.exp(log_yp + eps)

        if t < n_periods - 1:
            # Transition
            cost = np.where(
                c_t < cm_t,
                (adj.phi_c / 2.0) * ((cm_t - c_t) ** 2) / np.maximum(cm_t, 1e-10),
                0.0,
            )
            savings = np.maximum(m_t - c_t - cost, 1e-10)

            r_port = 1.0 + ar.rf + theta_t * (r_stock - ar.rf)

            g_a = ip.age_drift(age)
            y_ratio = np.exp(g_a + eta)

            financial = savings * r_port / y_ratio

            is_retired = lc.is_retired(age + 1)
            if is_retired:
                m_next = financial + illiq.S
            else:
                m_next = financial + illiq.disposable_share * np.exp(eps)

            # Update state
            x_t = ar.x_bar + ar.phi_x * (x_t - ar.x_bar) + xi
            m_t = np.maximum(m_next, 1e-10)
            cm_t = np.maximum(c_t / y_ratio, 1e-10)
            log_yp = log_yp + g_a + eta

    return SimulationResult(
        n_households=n_households,
        seed=seed,
        calibration=cal,
        ages=ages,
        paths=SimulationPaths(
            wealth=wealth,
            consumption=consumption,
            stock_share=stock_share,
            income=income,
            equity_premium=eq_premium,
            stock_return=stock_ret,
            persistent_earnings=log_y_bar,
        ),
    )
