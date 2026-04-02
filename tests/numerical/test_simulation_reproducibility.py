"""Tests for simulation engine."""

import numpy as np
import pytest

from liquiditylife.calibrations.registry import load_calibration
from liquiditylife.simulate.engine import simulate_cohorts
from liquiditylife.solve.grids import GridSpec
from liquiditylife.solve.quadrature import QuadratureSpec
from liquiditylife.solve.solver import SolverConfig, solve_model


def _solve_tiny() -> object:
    """Solve a tiny grid for testing."""
    cal = load_calibration("toy_demo_small_grid")
    config = SolverConfig(
        grid_spec=GridSpec(
            x_points=3, m_points=5, cm_points=3,
            x_min=-0.02, x_max=0.12, m_max=10.0,
        ),
        quad_spec=QuadratureSpec(n_xi=3, n_ncf=3, n_eta=3, n_eps=3),
        n_c_grid=20,
        n_theta_grid=5,
        verbose=False,
    )
    return solve_model(cal, config)


@pytest.mark.slow
class TestSimulation:
    def test_shapes(self) -> None:
        solution = _solve_tiny()
        sim = simulate_cohorts(solution, n_households=50, seed=42)  # type: ignore[arg-type]
        lc = sim.calibration.lifecycle
        assert sim.paths.wealth.shape == (50, lc.n_total_periods)
        assert sim.paths.consumption.shape == (50, lc.n_total_periods)
        assert sim.paths.stock_share.shape == (50, lc.n_total_periods)

    def test_positive_wealth(self) -> None:
        solution = _solve_tiny()
        sim = simulate_cohorts(solution, n_households=50, seed=42)  # type: ignore[arg-type]
        assert np.all(sim.paths.wealth > 0)

    def test_stock_share_bounds(self) -> None:
        solution = _solve_tiny()
        sim = simulate_cohorts(solution, n_households=50, seed=42)  # type: ignore[arg-type]
        assert np.all(sim.paths.stock_share >= 0)
        assert np.all(sim.paths.stock_share <= 1)

    def test_positive_consumption(self) -> None:
        solution = _solve_tiny()
        sim = simulate_cohorts(solution, n_households=50, seed=42)  # type: ignore[arg-type]
        assert np.all(sim.paths.consumption > 0)

    def test_seed_reproducibility(self) -> None:
        solution = _solve_tiny()
        sim1 = simulate_cohorts(solution, n_households=20, seed=123)  # type: ignore[arg-type]
        sim2 = simulate_cohorts(solution, n_households=20, seed=123)  # type: ignore[arg-type]
        np.testing.assert_array_equal(sim1.paths.wealth, sim2.paths.wealth)
        np.testing.assert_array_equal(sim1.paths.consumption, sim2.paths.consumption)

    def test_different_seeds_differ(self) -> None:
        solution = _solve_tiny()
        sim1 = simulate_cohorts(solution, n_households=20, seed=1)  # type: ignore[arg-type]
        sim2 = simulate_cohorts(solution, n_households=20, seed=2)  # type: ignore[arg-type]
        assert not np.allclose(sim1.paths.wealth, sim2.paths.wealth)
