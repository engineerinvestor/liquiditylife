# liquiditylife

An open-source Python package for **liquidity-aware life-cycle portfolio choice**, built for interactive exploration of portfolio choice under labor-income risk and consumption rigidity.

[![CI](https://github.com/engineerinvestor/liquiditylife/actions/workflows/ci.yml/badge.svg)](https://github.com/engineerinvestor/liquiditylife/actions/workflows/ci.yml)
[![Docs](https://github.com/engineerinvestor/liquiditylife/actions/workflows/docs.yml/badge.svg)](https://engineerinvestor.github.io/liquiditylife/)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](https://github.com/engineerinvestor/liquiditylife)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

`liquiditylife` implements a life-cycle portfolio choice model with:

- **Risky labor income** correlated with stock-market crashes
- **Consumption adjustment frictions** (asymmetric costs of cutting spending)
- **Time-varying equity premium** with return predictability
- **Illiquid savings** during working life and retirement flow income

The package solves for optimal consumption and equity allocation policies via backward induction, simulates household cohorts, and exports visualization-ready outputs for dashboards and research.

## Asset Allocation Calculator

Get an instant, personalized recommendation without running the solver:

```python
from liquiditylife import recommend, UserInputs

rec = recommend(UserInputs(
    age=35,
    annual_income=150_000,
    liquid_savings=200_000,
    monthly_fixed_expenses=5_000,
    risk_tolerance=3,  # 1=aggressive, 5=conservative
))

print(f"Stock share: {rec.stock_share_pct}%")
print(f"Emergency fund: {rec.emergency_fund_months} months")
print(f"Stocks: ${rec.stocks_dollars:,.0f}, Safe: ${rec.safe_dollars:,.0f}")
```

Or from the command line:

```bash
liquiditylife calculator recommend \
  --age 35 --income 150000 --savings 200000 --expenses 5000 --risk 3
```

See the [Calculator docs](https://engineerinvestor.github.io/liquiditylife/calculator/) for details.

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

## Quick Start (Research API)

```python
from liquiditylife import load_calibration, solve_model, simulate_cohorts, policy_surface

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
liquiditylife calculator recommend --age 35 --income 150000 --savings 200000 --expenses 5000 --risk 3
```

## Documentation

Full documentation is available at [engineerinvestor.github.io/liquiditylife](https://engineerinvestor.github.io/liquiditylife/).

## Citation

This package implements a model inspired by:

> Adams, Patrick. "Stocks for the Long Run or Liquidity? Tax Data Evidence and Portfolio Choice Implications."
> MIT Sloan School of Management, January 7, 2026.
> [https://patrick-adams.com/jmp](https://patrick-adams.com/jmp)

If you use `liquiditylife` in academic work, please cite the software as:

```bibtex
@software{liquiditylife,
  author       = {Engineer Investor},
  title        = {liquiditylife: Life-Cycle Portfolio Choice with Liquidity Risk},
  year         = {2026},
  url          = {https://github.com/engineerinvestor/liquiditylife},
  version      = {0.1.0},
  license      = {MIT}
}
```

## License

MIT
