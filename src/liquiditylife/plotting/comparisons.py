"""Multi-calibration comparison plots."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from liquiditylife.solve.solver import SolvedModel


def plot_friction_comparison(
    solutions: dict[str, SolvedModel],
    age: int = 35,
    m_values: list[float] | None = None,
    ax: Axes | None = None,
) -> Axes:
    """Overlay stock share vs cash-on-hand for multiple friction levels.

    Args:
        solutions: Dict mapping label to solved model.
        age: Age at which to compare.
        m_values: Cash-on-hand values to evaluate. Defaults to grid.
        ax: Matplotlib axes. Created if None.

    Returns:
        The axes with the plot.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    if ax is None:
        _, ax = plt.subplots()

    x_mid = 0.05  # mean equity premium
    cm_mid = 0.5  # typical lagged consumption ratio

    for label, solution in solutions.items():
        pf = solution.policies[age]
        m_vals = pf.grid_m if m_values is None else np.array(m_values, dtype=np.float64)

        shares = [pf.stock_share_at(x_mid, float(m), cm_mid) for m in m_vals]
        ax.plot(m_vals, shares, label=label)

    ax.set_xlabel("Cash-on-Hand (m_t)")
    ax.set_ylabel("Optimal Stock Share")
    ax.set_title(f"Stock Share at Age {age}: Friction Comparison")
    ax.legend()
    return ax
