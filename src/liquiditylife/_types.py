"""Shared type aliases for liquiditylife."""

from typing import Any

import numpy as np
import numpy.typing as npt

ArrayFloat = npt.NDArray[np.float64]
Scalar = float | np.floating[Any]
