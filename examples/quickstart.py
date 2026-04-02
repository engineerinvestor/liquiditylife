#!/usr/bin/env python3
"""Quickstart: solve, simulate, and explore a lifecycle portfolio model.

Run with:
    python examples/quickstart.py
"""

from __future__ import annotations

from liquiditylife.calibrations import load_calibration
from liquiditylife.simulate import simulate_cohorts
from liquiditylife.solve import solve_model
from liquiditylife.solve.grids import GridSpec
from liquiditylife.solve.quadrature import QuadratureSpec
from liquiditylife.solve.solver import SolverConfig
from liquiditylife.sweep import age_profile, policy_surface


def main() -> None:
    # --- 1. Load a calibration bundle ---
    cal = load_calibration("toy_demo_small_grid")
    print(f"Calibration: {cal.name}")
    print(f"  Ages: {cal.lifecycle.age_start}-{cal.lifecycle.age_max}")
    print(f"  Friction (phi_c): {cal.adjustment_cost.phi_c}")
    print(f"  Risk aversion (gamma): {cal.preferences.gamma}")
    print()

    # --- 2. Solve the model (small grid for speed) ---
    config = SolverConfig(
        grid_spec=GridSpec(
            x_points=5, m_points=10, cm_points=5,
            x_min=-0.02, x_max=0.12, m_max=15.0,
        ),
        quad_spec=QuadratureSpec(n_xi=3, n_ncf=3, n_eta=3, n_eps=3),
        n_c_grid=30,
        n_theta_grid=11,
        verbose=False,
    )
    print("Solving... ", end="", flush=True)
    solution = solve_model(cal, config)
    print(f"done in {solution.solve_time_seconds:.1f}s")
    print()

    # --- 3. Simulate household cohorts ---
    print("Simulating 1,000 households... ", end="", flush=True)
    sim = simulate_cohorts(solution, n_households=1_000, seed=42)
    print(f"done ({sim.n_households} households, {len(sim.ages)} periods)")
    print()

    # --- 4. Age profile summary ---
    profile = age_profile(solution, n_households=1_000, seed=42)
    print("Age Profile (selected ages):")
    selected = profile[profile["age"].isin([25, 30, 35, 40, 45, 50])]
    print(selected.to_string(index=False))
    print()

    # --- 5. Policy surface at a specific age ---
    surface = policy_surface(solution, age=30)
    print(f"Policy surface at age 30: {len(surface)} grid points")
    ss = surface["stock_share"]
    print(f"  Stock share range: [{ss.min():.3f}, {ss.max():.3f}]")
    cs = surface["consumption"]
    print(f"  Consumption range: [{cs.min():.3f}, {cs.max():.3f}]")
    print()

    # --- 6. Compare calibrations ---
    print("Comparing frictionless vs high friction at age 30:")
    for name in ["adams_frictionless", "adams_high_friction"]:
        c = load_calibration(name)
        s = solve_model(c, config)
        p = age_profile(s, n_households=500, seed=42)
        row = p[p["age"] == 30.0].iloc[0]
        print(f"  {name}: mean stock share = {row['mean_stock_share']:.3f}")


if __name__ == "__main__":
    main()
