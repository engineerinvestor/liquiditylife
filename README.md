# liquiditylife

An open-source Python package for **liquidity-aware life-cycle portfolio choice**, built for interactive exploration of portfolio choice under labor-income risk and consumption rigidity.

[![PyPI](https://img.shields.io/pypi/v/liquiditylife)](https://pypi.org/project/liquiditylife/)
[![Python](https://img.shields.io/pypi/pyversions/liquiditylife)](https://pypi.org/project/liquiditylife/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

`liquiditylife` implements a life-cycle portfolio choice model with:

- **Risky labor income** correlated with stock-market crashes
- **Consumption adjustment frictions** (asymmetric costs of cutting spending)
- **Time-varying equity premium** with return predictability
- **Illiquid savings** during working life and retirement flow income

The package solves for optimal consumption and equity allocation policies via backward induction, simulates household cohorts, and exports visualization-ready outputs for dashboards and research.

## Installation

```bash
pip install liquiditylife
```

With optional extras:

```bash
pip install liquiditylife[viz]   # Streamlit, Plotly, Panel
pip install liquiditylife[fast]  # Numba acceleration
pip install liquiditylife[dev]   # Development tools
```

## Quick Start

```python
from liquiditylife.calibrations import load_calibration
from liquiditylife.solve import solve_model
from liquiditylife.simulate import simulate_cohorts
from liquiditylife.sweep import policy_surface

cal = load_calibration("adams_high_friction")
solution = solve_model(cal)

sim = simulate_cohorts(solution, n_households=100_000, seed=42)

surface = policy_surface(solution, age=35)
```

## CLI

```bash
liquiditylife solve --calibration adams_high_friction
liquiditylife simulate --calibration adams_high_friction --n 100000
liquiditylife sweep policy-surface --age 35 --calibration adams_high_friction
liquiditylife list-calibrations
```

## Citation

This package implements a model inspired by:

> Adams, Patrick. "Stocks for the Long Run or Liquidity? Tax Data Evidence and Portfolio Choice Implications."
> MIT Sloan School of Management, January 7, 2026.
> [https://patrick-adams.com/jmp](https://patrick-adams.com/jmp)

## License

MIT
