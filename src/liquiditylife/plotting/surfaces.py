"""Policy surface plots."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from matplotlib.axes import Axes


def plot_policy_surface(
    surface_df: pd.DataFrame,
    variable: str = "stock_share",
    ax: Axes | None = None,
) -> Axes:
    """Plot a 2D heatmap of a policy variable over (x_t, m_t).

    Args:
        surface_df: DataFrame from ``policy_surface()`` with columns
            x_t, m_t, and the variable to plot.
        variable: Column name to plot.
        ax: Matplotlib axes. Created if None.

    Returns:
        The axes with the plot.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots()

    pivot = surface_df.pivot_table(index="m_t", columns="x_t", values=variable)
    im = ax.pcolormesh(
        pivot.columns.values,
        pivot.index.values,
        pivot.values,
        shading="auto",
    )
    ax.set_xlabel("Equity Premium (x_t)")
    ax.set_ylabel("Cash-on-Hand (m_t)")
    ax.set_title(f"Policy: {variable}")
    if ax.figure is not None:
        ax.figure.colorbar(im, ax=ax)
    return ax
