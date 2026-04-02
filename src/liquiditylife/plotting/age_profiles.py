"""Age profile plots."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from liquiditylife.simulate.result import SimulationResult


def plot_age_stock_share(
    results: dict[str, SimulationResult],
    ax: Axes | None = None,
) -> Axes:
    """Plot age vs mean stock share for multiple calibrations.

    Args:
        results: Dict mapping label to simulation result.
        ax: Matplotlib axes. Created if None.

    Returns:
        The axes with the plot.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    if ax is None:
        _, ax = plt.subplots()

    for label, sim in results.items():
        mean_share = np.mean(sim.paths.stock_share, axis=0)
        ages = sim.ages
        ax.plot(ages, mean_share, label=label)

    ax.set_xlabel("Age")
    ax.set_ylabel("Mean Stock Share")
    ax.set_title("Optimal Stock Share by Age")
    ax.legend()
    return ax


def plot_age_wealth(
    results: dict[str, SimulationResult],
    ax: Axes | None = None,
) -> Axes:
    """Plot age vs median wealth for multiple calibrations.

    Args:
        results: Dict mapping label to simulation result.
        ax: Matplotlib axes. Created if None.

    Returns:
        The axes with the plot.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    if ax is None:
        _, ax = plt.subplots()

    for label, sim in results.items():
        median_wealth = np.median(sim.paths.wealth, axis=0)
        ages = sim.ages
        ax.plot(ages, median_wealth, label=label)

    ax.set_xlabel("Age")
    ax.set_ylabel("Median Wealth (scaled)")
    ax.set_title("Liquid Wealth by Age")
    ax.legend()
    return ax
