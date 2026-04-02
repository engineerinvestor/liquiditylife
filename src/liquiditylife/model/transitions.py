"""Full state transition logic combining processes and budget."""

from __future__ import annotations

import math

from liquiditylife.calibrations.bundles import CalibrationBundle
from liquiditylife.core.state import HouseholdState
from liquiditylife.model.budget import (
    end_of_period_savings,
    next_period_coh_retired,
    next_period_coh_working,
    portfolio_return,
)


def transition_state(
    state: HouseholdState,
    c_t: float,
    theta: float,
    xi: float,
    ncf: float,
    ndr: float,
    eta: float,
    eps: float,
    cal: CalibrationBundle,
) -> HouseholdState:
    """Compute the next-period state given current state, controls, and shocks.

    Args:
        state: Current household state (scaled).
        c_t: Scaled consumption chosen this period.
        theta: Stock share of liquid wealth.
        xi: Equity premium innovation.
        ncf: Cash-flow news realization.
        ndr: Discount-rate news realization.
        eta: Persistent income shock.
        eps: Transitory income shock.
        cal: Calibration bundle.

    Returns:
        Next-period household state.
    """
    ar = cal.asset_returns
    ip = cal.income
    lc = cal.lifecycle
    adj = cal.adjustment_cost
    illiq = cal.illiquid

    # 1. Next-period equity premium
    x_next = ar.evolve_premium(state.x_t, xi)

    # 2. Realized stock return
    r_stock = ar.realized_return(state.x_t, ncf, ndr)

    # 3. End-of-period savings (scaled)
    savings = end_of_period_savings(state.m_t, c_t, state.cm_t, adj)
    savings = max(savings, 0.0)

    # 4. Portfolio return
    r_port = portfolio_return(theta, r_stock, ar.rf)

    # 5. Persistent earnings growth ratio: Y_{t+1}^P / Y_t^P
    g_a = ip.age_drift(state.age)
    y_ratio = math.exp(g_a + eta)

    # 6. Next-period cash-on-hand (scaled)
    is_retired = lc.is_retired(state.age + 1)
    if is_retired:
        m_next = next_period_coh_retired(savings, r_port, y_ratio, illiq)
    else:
        # Add transitory income component
        m_next = next_period_coh_working(savings, r_port, y_ratio, illiq)
        m_next += illiq.disposable_share * (math.exp(eps) - 1.0)

    # 7. Lagged consumption ratio for next period
    cm_next = c_t / y_ratio

    return HouseholdState(
        age=state.age + 1,
        x_t=x_next,
        m_t=max(m_next, 1e-10),
        cm_t=max(cm_next, 1e-10),
    )
