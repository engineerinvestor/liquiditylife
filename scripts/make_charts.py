#!/usr/bin/env python3
"""Generate Twitter-ready data visualizations from precomputed calculator tables.

Usage:
    python scripts/make_charts.py

Output:
    figures/*.png (1200x675, 2x DPI for retina)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from liquiditylife.calculator.tables import load_tables_json

FIGURES_DIR = Path(__file__).parent.parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

# Style constants
COLORS = {
    "teal": "#00d4aa",
    "coral": "#ff6b6b",
    "gold": "#ffd93d",
    "blue": "#4ecdc4",
    "purple": "#a855f7",
    "white": "#e0e0e0",
    "gray": "#888888",
}
BG_COLOR = "#0f0f23"
GRID_COLOR = "#1e1e3a"
FONT_COLOR = "#e0e0e0"
ANNOTATION_COLOR = "#666680"

WIDTH = 1200
HEIGHT = 675


def _base_layout(title: str) -> dict:  # type: ignore[type-arg]
    return dict(
        title=dict(text=title, font=dict(size=22, color=FONT_COLOR), x=0.5),
        paper_bgcolor=BG_COLOR,
        plot_bgcolor=BG_COLOR,
        font=dict(family="Inter, Helvetica, Arial, sans-serif", color=FONT_COLOR, size=14),
        xaxis=dict(gridcolor=GRID_COLOR, zeroline=False),
        yaxis=dict(gridcolor=GRID_COLOR, zeroline=False),
        legend=dict(
            bgcolor="rgba(0,0,0,0)", bordercolor="rgba(0,0,0,0)",
            font=dict(size=13), x=0.02, y=0.98,
        ),
        margin=dict(l=70, r=30, t=70, b=70),
        annotations=[dict(
            text="liquiditylife | Adams (2026)",
            xref="paper", yref="paper", x=0.99, y=-0.12,
            showarrow=False, font=dict(size=10, color=ANNOTATION_COLOR),
        )],
    )


def _save(fig: go.Figure, name: str) -> None:
    path = FIGURES_DIR / f"{name}.png"
    fig.write_image(str(path), width=WIDTH, height=HEIGHT, scale=2)
    print(f"  Saved {path}")


def chart_1_age_profile() -> None:
    """Stock share vs age for 3 friction levels."""
    tables = load_tables_json()
    m_t = 2.0  # moderate wealth-to-income

    fig = go.Figure()
    scenarios = [
        ("low_friction_moderate", "No friction (flexible spending)", COLORS["teal"], "dot"),
        ("med_friction_moderate", "Moderate friction", COLORS["gold"], "dash"),
        ("high_friction_moderate", "High friction (rigid expenses)", COLORS["coral"], "solid"),
    ]

    for key, label, color, dash in scenarios:
        t = tables[key]
        ages = t.ages
        shares = [t.lookup(age, m_t) * 100 for age in ages]
        fig.add_trace(go.Scatter(
            x=ages, y=shares, name=label, mode="lines",
            line=dict(color=color, width=3, dash=dash),
        ))

    # Retirement line
    fig.add_vline(x=60, line=dict(color=ANNOTATION_COLOR, width=1, dash="dot"))
    fig.add_annotation(x=60, y=95, text="Retirement", showarrow=False,
                       font=dict(size=11, color=ANNOTATION_COLOR))

    layout = _base_layout("Optimal Stock Share by Age")
    layout["xaxis"]["title"] = "Age"
    layout["yaxis"]["title"] = "Stock Share of Liquid Wealth (%)"
    layout["yaxis"]["range"] = [0, 105]
    fig.update_layout(**layout)
    _save(fig, "01_age_profile")


def chart_2_wealth_heatmap() -> None:
    """Heatmap: stock share over age x wealth-to-income."""
    tables = load_tables_json()
    t = tables["high_friction_moderate"]

    ages = t.ages
    m_grid = t.m_grid
    z = np.array([[t.lookup(age, m) * 100 for m in m_grid] for age in ages])

    fig = go.Figure(data=go.Heatmap(
        x=ages, y=m_grid, z=z.T,
        colorscale=[
            [0, "#0f0f23"], [0.2, "#1e3a5f"], [0.4, "#00d4aa"],
            [0.6, "#ffd93d"], [0.8, "#ff6b6b"], [1.0, "#ff2222"],
        ],
        colorbar=dict(title="Stock %", tickfont=dict(color=FONT_COLOR)),
        zmin=0, zmax=100,
    ))

    fig.add_vline(x=60, line=dict(color="white", width=1, dash="dot"))
    fig.add_annotation(x=60, y=28, text="Retirement", showarrow=False,
                       font=dict(size=11, color="white"))

    layout = _base_layout("Optimal Stock Share: Age vs Wealth-to-Income")
    layout["xaxis"]["title"] = "Age"
    layout["yaxis"]["title"] = "Liquid Wealth / Annual Income"
    layout["yaxis"]["type"] = "log"
    fig.update_layout(**layout)
    _save(fig, "02_wealth_heatmap")


def chart_3_friction_effect() -> None:
    """Stock share vs wealth at age 35 for 3 friction levels."""
    tables = load_tables_json()
    age = 35

    fig = go.Figure()
    scenarios = [
        ("low_friction_moderate", "No friction", COLORS["teal"]),
        ("med_friction_moderate", "Moderate friction", COLORS["gold"]),
        ("high_friction_moderate", "High friction", COLORS["coral"]),
    ]

    m_range = np.linspace(0.1, 15, 100)
    for key, label, color in scenarios:
        t = tables[key]
        shares = [t.lookup(age, m) * 100 for m in m_range]
        fig.add_trace(go.Scatter(
            x=m_range, y=shares, name=label, mode="lines",
            line=dict(color=color, width=3),
        ))

    layout = _base_layout("How Fixed Expenses Change Your Optimal Allocation (Age 35)")
    layout["xaxis"]["title"] = "Liquid Wealth / Annual Income"
    layout["yaxis"]["title"] = "Optimal Stock Share (%)"
    layout["yaxis"]["range"] = [0, 105]
    fig.update_layout(**layout)
    _save(fig, "03_friction_effect")


def chart_4_risk_tolerance() -> None:
    """Stock share vs wealth at age 35 for 3 risk levels (high friction)."""
    tables = load_tables_json()
    age = 35

    fig = go.Figure()
    scenarios = [
        ("high_friction_aggressive", "Aggressive (low risk aversion)", COLORS["teal"]),
        ("high_friction_moderate", "Moderate", COLORS["gold"]),
        ("high_friction_conservative", "Conservative (high risk aversion)", COLORS["coral"]),
    ]

    m_range = np.linspace(0.1, 15, 100)
    for key, label, color in scenarios:
        t = tables[key]
        shares = [t.lookup(age, m) * 100 for m in m_range]
        fig.add_trace(go.Scatter(
            x=m_range, y=shares, name=label, mode="lines",
            line=dict(color=color, width=3),
        ))

    layout = _base_layout("Risk Tolerance vs Liquidity (Age 35, High Fixed Expenses)")
    layout["xaxis"]["title"] = "Liquid Wealth / Annual Income"
    layout["yaxis"]["title"] = "Optimal Stock Share (%)"
    layout["yaxis"]["range"] = [0, 105]
    fig.update_layout(**layout)
    _save(fig, "04_risk_tolerance")


def chart_5_working_vs_retired() -> None:
    """Side-by-side: age 40 vs age 65 stock share across wealth."""
    tables = load_tables_json()
    t = tables["high_friction_moderate"]

    m_range = np.linspace(0.1, 15, 100)
    shares_40 = [t.lookup(40, m) * 100 for m in m_range]
    shares_65 = [t.lookup(65, m) * 100 for m in m_range]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=m_range, y=shares_40, name="Age 40 (working)",
        mode="lines", line=dict(color=COLORS["coral"], width=3),
    ))
    fig.add_trace(go.Scatter(
        x=m_range, y=shares_65, name="Age 65 (retired)",
        mode="lines", line=dict(color=COLORS["teal"], width=3),
    ))

    # Shaded area showing the gap
    fig.add_trace(go.Scatter(
        x=np.concatenate([m_range, m_range[::-1]]),
        y=np.concatenate([shares_65, shares_40[::-1]]),  # type: ignore[arg-type]
        fill="toself", fillcolor="rgba(0,212,170,0.1)",
        line=dict(color="rgba(0,0,0,0)"), showlegend=False,
    ))

    layout = _base_layout("The Retirement Paradox: Retirees Can Hold More Stocks")
    layout["xaxis"]["title"] = "Liquid Wealth / Annual Income"
    layout["yaxis"]["title"] = "Optimal Stock Share (%)"
    layout["yaxis"]["range"] = [0, 105]
    fig.update_layout(**layout)
    _save(fig, "05_working_vs_retired")


def main() -> None:
    print("Generating Twitter-ready charts...")
    chart_1_age_profile()
    chart_2_wealth_heatmap()
    chart_3_friction_effect()
    chart_4_risk_tolerance()
    chart_5_working_vs_retired()
    print(f"\nDone! Charts saved to {FIGURES_DIR}/")


if __name__ == "__main__":
    main()
