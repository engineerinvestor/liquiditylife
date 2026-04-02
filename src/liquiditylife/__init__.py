"""liquiditylife: Life-cycle portfolio choice with liquidity risk."""

from liquiditylife.calculator.recommend import Recommendation, UserInputs, recommend
from liquiditylife.calibrations import load_calibration
from liquiditylife.simulate import simulate_cohorts
from liquiditylife.solve import SolvedModel, SolverConfig, solve_model
from liquiditylife.sweep import age_profile, comparative_statics, policy_surface

__version__ = "0.1.0"

__all__ = [
    "Recommendation",
    "SolvedModel",
    "SolverConfig",
    "UserInputs",
    "__version__",
    "age_profile",
    "comparative_statics",
    "load_calibration",
    "policy_surface",
    "recommend",
    "simulate_cohorts",
    "solve_model",
]
