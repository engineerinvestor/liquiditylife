"""Comparative statics: frictionless vs high-friction calibrations."""

from __future__ import annotations

import numpy as np
import pytest

from liquiditylife.calibrations.registry import load_calibration
from liquiditylife.simulate.engine import simulate_cohorts
from liquiditylife.simulate.result import SimulationResult
from liquiditylife.solve.grids import GridSpec
from liquiditylife.solve.quadrature import QuadratureSpec
from liquiditylife.solve.solver import SolvedModel, SolverConfig, solve_model


def _solve_and_simulate(
    name: str, config: SolverConfig, n_households: int = 100, seed: int = 42
) -> tuple[SolvedModel, SimulationResult]:
    cal = load_calibration(name)
    solution = solve_model(cal, config)
    sim = simulate_cohorts(solution, n_households=n_households, seed=seed)
    return solution, sim


@pytest.mark.slow
class TestComparativeStatics:
    @pytest.fixture(scope="class")
    def config(self) -> SolverConfig:
        return SolverConfig(
            grid_spec=GridSpec(
                x_points=3, m_points=6, cm_points=3,
                x_min=-0.02, x_max=0.12, m_max=10.0,
            ),
            quad_spec=QuadratureSpec(n_xi=3, n_ncf=3, n_eta=3, n_eps=3),
            n_c_grid=25,
            n_theta_grid=7,
            verbose=False,
        )

    @pytest.fixture(scope="class")
    def frictionless_sim(self, config: SolverConfig) -> SimulationResult:
        _, sim = _solve_and_simulate("adams_frictionless", config)
        return sim

    @pytest.fixture(scope="class")
    def high_friction_sim(self, config: SolverConfig) -> SimulationResult:
        _, sim = _solve_and_simulate("adams_high_friction", config)
        return sim

    def test_frictionless_higher_stock_share(
        self,
        frictionless_sim: SimulationResult,
        high_friction_sim: SimulationResult,
    ) -> None:
        """Frictionless calibration should have higher mean stock share."""
        lc = frictionless_sim.calibration.lifecycle

        for check_age in [30, 35]:
            t = check_age - lc.age_start
            if t >= frictionless_sim.paths.stock_share.shape[1]:
                continue
            mean_fl = float(np.mean(frictionless_sim.paths.stock_share[:, t]))
            mean_hf = float(np.mean(high_friction_sim.paths.stock_share[:, t]))
            assert mean_fl >= mean_hf - 0.05, (
                f"At age {check_age}: frictionless ({mean_fl:.3f}) "
                f"should >= high friction ({mean_hf:.3f})"
            )
