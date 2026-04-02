# liquiditylife

An open-source Python package for **liquidity-aware life-cycle portfolio choice**.

## Overview

`liquiditylife` implements a life-cycle portfolio choice model inspired by
[Adams (2026)](https://patrick-adams.com/jmp), featuring:

- Risky labor income correlated with stock-market crashes
- Asymmetric consumption adjustment frictions
- Time-varying equity premium with return predictability
- Illiquid savings during working life and retirement flow income

## Quick Start

```python
from liquiditylife.calibrations import load_calibration
from liquiditylife.solve import solve_model
from liquiditylife.simulate import simulate_cohorts

cal = load_calibration("adams_high_friction")
solution = solve_model(cal)
sim = simulate_cohorts(solution, n_households=10_000, seed=42)
```

## Installation

```bash
pip install liquiditylife
```

## Citation

> Adams, Patrick. "Stocks for the Long Run or Liquidity? Tax Data Evidence
> and Portfolio Choice Implications." MIT Sloan School of Management,
> January 7, 2026. [https://patrick-adams.com/jmp](https://patrick-adams.com/jmp)
