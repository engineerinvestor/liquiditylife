"""Calibration registry for named parameter bundles."""

from __future__ import annotations

from typing import Any

from liquiditylife.calibrations.adams_baseline import make_adams_baseline
from liquiditylife.calibrations.bundles import CalibrationBundle
from liquiditylife.calibrations.toy_demo import make_toy_demo

_REGISTRY: dict[str, tuple[Any, dict[str, Any]]] = {
    "adams_baseline": (make_adams_baseline, {"phi_c": 10.0}),
    "adams_frictionless": (make_adams_baseline, {"phi_c": 0.0}),
    "adams_moderate_friction": (make_adams_baseline, {"phi_c": 5.0}),
    "adams_high_friction": (make_adams_baseline, {"phi_c": 10.0}),
    "toy_demo_small_grid": (make_toy_demo, {}),
}


def load_calibration(name: str, **overrides: float) -> CalibrationBundle:
    """Load a named calibration bundle.

    Args:
        name: Calibration name (see ``list_calibrations()``).
        **overrides: Keyword arguments forwarded to the factory function,
            overriding default parameter values.

    Returns:
        A frozen ``CalibrationBundle``.

    Raises:
        KeyError: If the name is not in the registry.
    """
    if name not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY))
        msg = f"Unknown calibration {name!r}. Available: {available}"
        raise KeyError(msg)
    factory, defaults = _REGISTRY[name]
    kwargs = {**defaults, **overrides}
    result: CalibrationBundle = factory(**kwargs)
    return result


def list_calibrations() -> list[str]:
    """Return sorted list of available calibration names."""
    return sorted(_REGISTRY)
