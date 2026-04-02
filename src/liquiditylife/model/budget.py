"""Budget constraint and cash-on-hand transitions."""

from __future__ import annotations

from liquiditylife.processes.adjustment_cost import AdjustmentCostModel
from liquiditylife.processes.illiquid import IlliquidWealthRule


def end_of_period_savings(
    m_t: float,
    c_t: float,
    cm_lag: float,
    adj_cost: AdjustmentCostModel,
) -> float:
    """Compute end-of-period savings (scaled by persistent earnings).

    s_t = m_t - c_t - Phi_C(c_t, cm_lag)

    Args:
        m_t: Scaled cash-on-hand.
        c_t: Scaled consumption.
        cm_lag: Lagged consumption ratio (C_{t-1} / Y_{t-1}^P).
        adj_cost: Adjustment cost model.

    Returns:
        End-of-period liquid savings (scaled).
    """
    cost = adj_cost.cost(c_t, cm_lag)
    return m_t - c_t - cost


def portfolio_return(theta: float, r_stock: float, rf: float) -> float:
    """Portfolio gross return: R_port = 1 + rf + theta * (R_stock - rf).

    Args:
        theta: Stock share of liquid wealth.
        r_stock: Realized stock return (net).
        rf: Risk-free rate.
    """
    return 1.0 + rf + theta * (r_stock - rf)


def next_period_coh_working(
    savings: float,
    r_port: float,
    y_ratio: float,
    illiquid: IlliquidWealthRule,
) -> float:
    """Next-period scaled cash-on-hand during working life.

    m_{t+1} = savings * R_port / (Y_{t+1}^P / Y_t^P) + (1 - s - tau) * (Y_{t+1} / Y_{t+1}^P)

    In scaled form with y_ratio = Y_{t+1}^P / Y_t^P and transitory component:
    m_{t+1} = savings * R_port / y_ratio + disposable_share * exp(eps)

    Args:
        savings: End-of-period scaled savings.
        r_port: Gross portfolio return.
        y_ratio: Persistent earnings growth Y_{t+1}^P / Y_t^P.
        illiquid: Illiquid wealth rule.
    """
    financial = savings * r_port / y_ratio
    return financial + illiquid.disposable_share


def next_period_coh_retired(
    savings: float,
    r_port: float,
    y_ratio: float,
    illiquid: IlliquidWealthRule,
) -> float:
    """Next-period scaled cash-on-hand during retirement.

    In retirement, flow income is S * Y_bar_terminal (constant in levels),
    so in scaled form: m_{t+1} = savings * R_port / y_ratio + S

    Args:
        savings: End-of-period scaled savings.
        r_port: Gross portfolio return.
        y_ratio: Persistent earnings growth (typically 1.0 in retirement).
        illiquid: Illiquid wealth rule.
    """
    financial = savings * r_port / y_ratio
    return financial + illiquid.S
