# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`liquiditylife` is an open-source Python package for life-cycle portfolio choice with liquidity risk, labor-income risk, and consumption adjustment frictions. It implements a structural model inspired by Patrick Adams' paper "Stocks for the Long Run or Liquidity? Tax Data Evidence and Portfolio Choice Implications." (MIT Sloan, 2026).

The package is a **policy engine**, not a data replication pipeline. It computes optimal consumption and equity allocation policies via backward induction, then simulates household cohorts and exports visualization-ready outputs.

## Build & Development

Requires Python >= 3.11. Uses a venv at `.venv/`:

```bash
source .venv/bin/activate
pip install -e ".[dev]"             # editable install with all dev deps
```

## Common Commands

```bash
# Type checking (strict mode with pydantic plugin)
mypy src/

# Linting and formatting
ruff check src/ tests/
ruff check src/ tests/ --fix

# Tests (117+ tests, ~33s for full suite)
pytest tests/                       # all tests
pytest tests/unit/                  # fast unit tests only
pytest tests/ -m "not slow"         # skip solver/simulation tests
pytest tests/ -m slow               # only solver/simulation tests
pytest tests/unit/test_calibrations.py  # single test file

# CLI
liquiditylife list-calibrations
liquiditylife solve --calibration toy_demo_small_grid
```

## Architecture

Source lives in `src/liquiditylife/` (src layout for PyPI). Three layers:

- **Layer A (Economic domain):** `core/` (Preferences, Lifecycle, HouseholdState, PolicyFunction), `calibrations/` (CalibrationBundle, registry, named bundles), `processes/` (AssetReturnProcess, IncomeProcess, AdjustmentCostModel, IlliquidWealthRule)
- **Layer B (Numerical engine):** `model/` (utility, budget, transitions), `solve/` (grids, quadrature, interpolation, bellman, solver), `simulate/` (engine, result), `sweep/` (surface, comparative statics)
- **Layer C (Visualization):** `vizdata/` (DataFrame/xarray/JSON export), `plotting/` (matplotlib helpers)

Supporting: `io/` (pickle cache), `cli/` (Click CLI)

## Key Technical Details

- All domain models use **pydantic v2** with `frozen=True`
- **mypy --strict** with pydantic plugin; all code must be fully typed
- **ruff** for linting (py311 target, 99 char line length)
- Solver uses grid search over (C, theta) with Gaussian quadrature — the critical performance bottleneck is `solve/bellman.py`
- Tests use `@pytest.mark.slow` for solver/simulation tests

## Calibration Bundles

5 named bundles in `calibrations/registry.py`: adams_baseline, adams_frictionless, adams_moderate_friction, adams_high_friction, toy_demo_small_grid. Use `load_calibration("name")` to get a `CalibrationBundle`.

## Design Rules

- Headless core, UI-ready outputs — no mandatory app dependency
- All result objects carry calibration provenance metadata (`source` field)
- No hidden global config; explicit model creation
- Income process parameters marked as `source="public_approximation"` (derived from paper's qualitative targets, not confidential data)
