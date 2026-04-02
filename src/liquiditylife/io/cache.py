"""Solution caching via pickle."""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from liquiditylife.solve.solver import SolvedModel

logger = logging.getLogger(__name__)

_DEFAULT_CACHE_DIR = Path.home() / ".liquiditylife" / "cache"


def save_solution(solution: SolvedModel, path: Path | None = None) -> Path:
    """Save a solved model to disk.

    Args:
        solution: The solved model to save.
        path: File path. Defaults to ``~/.liquiditylife/cache/<fingerprint>.pkl``.

    Returns:
        The path where the solution was saved.
    """
    if path is None:
        _DEFAULT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        fp = solution.calibration.fingerprint()
        path = _DEFAULT_CACHE_DIR / f"{fp}.pkl"

    with path.open("wb") as f:
        pickle.dump(solution, f, protocol=pickle.HIGHEST_PROTOCOL)

    logger.info("Saved solution to %s", path)
    return path


def load_solution(path: Path) -> SolvedModel:
    """Load a solved model from disk.

    Args:
        path: Path to the pickled solution.

    Returns:
        The deserialized ``SolvedModel``.
    """
    with path.open("rb") as f:
        solution: SolvedModel = pickle.load(f)
    logger.info("Loaded solution from %s", path)
    return solution
