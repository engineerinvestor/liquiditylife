"""Epstein-Zin utility and terminal value functions."""

from __future__ import annotations

import math

from liquiditylife.core.preferences import Preferences

# Floor to prevent log(0) or negative bases in power functions
_EPS = 1e-10


def ez_utility(c: float, ev_next: float, prefs: Preferences) -> float:
    """Epstein-Zin one-period utility recursion.

    V_t = [(1-beta)*c^rho + beta*CE^rho]^(1/rho)
    where rho = 1 - 1/psi and CE = (E[V^(1-gamma)])^(1/(1-gamma))

    For the case where psi == 1 (log EIS):
    V_t = c^(1 - beta) * ev_next^beta

    Args:
        c: Current consumption (must be positive).
        ev_next: Certainty equivalent of next-period value
            (E[V_{t+1}^(1-gamma)])^(1/(1-gamma)).
        prefs: Household preferences.

    Returns:
        Current-period value V_t.
    """
    c = max(c, _EPS)
    ev_next = max(ev_next, _EPS)

    if math.isinf(prefs.theta):
        # Log EIS case: V = c^(1-beta) * ev_next^beta
        return float(c ** (1.0 - prefs.beta) * ev_next**prefs.beta)

    rho = 1.0 - 1.0 / prefs.psi  # = (1-gamma)/theta

    term_c = (1.0 - prefs.beta) * c**rho
    term_v = prefs.beta * ev_next**rho

    return float(max((term_c + term_v) ** (1.0 / rho), _EPS))


def terminal_utility(c: float, prefs: Preferences) -> float:
    """Terminal period utility (no continuation value).

    At the final age, the household consumes all remaining resources.
    V_T = ((1 - beta) * c^(1 - 1/psi))^(1/(1 - 1/psi)) = (1-beta)^(psi/(psi-1)) * c
    Simplified: proportional to c, so we use V_T = c for simplicity
    (the constant cancels in the optimisation).
    """
    return max(c, _EPS)


def crra_utility(c: float, gamma: float) -> float:
    """Standard CRRA flow utility for reference/debugging.

    u(c) = c^(1-gamma) / (1-gamma)  if gamma != 1
    u(c) = log(c)                    if gamma == 1
    """
    c = max(c, _EPS)
    if math.isclose(gamma, 1.0, rel_tol=1e-9):
        return math.log(c)
    return float(c ** (1.0 - gamma) / (1.0 - gamma))
