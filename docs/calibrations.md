# Calibrations

## Available Bundles

| Name | $\phi_c$ | Description |
|------|----------|-------------|
| `adams_baseline` | 10 | Baseline with high friction (paper-preferred) |
| `adams_frictionless` | 0 | No consumption adjustment costs |
| `adams_moderate_friction` | 5 | Moderate friction |
| `adams_high_friction` | 10 | High friction (same as baseline) |
| `toy_demo_small_grid` | 5 | Short lifecycle (25-40-50) for fast testing |

## Baseline Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| $\gamma$ | 5 | Relative risk aversion |
| $\psi$ | 0.5 | Elasticity of intertemporal substitution |
| $\beta$ | 0.85 | Time discount factor |
| $r_f$ | 0.02 | Risk-free rate |
| $\bar{x}$ | 0.05 | Mean equity premium |
| $\phi_x$ | 0.85 | Equity premium persistence |
| $\rho_{cs}$ | 0.96 | Premium innovation to discount-rate news |
| $s$ | 0.15 | Illiquid savings share (working life) |
| $\tau$ | 0.35 | Tax/contribution rate |
| $S$ | 0.60 | Retirement income replacement rate |
| $\sigma_\varepsilon$ | 0.20 | Transitory income shock std dev |
| $\sigma_\eta$ | 0.25 | Persistent income shock std dev |

## Custom Calibrations

```python
from liquiditylife.calibrations import load_calibration

# Override specific parameters
cal = load_calibration("adams_baseline", phi_c=7.5)
```
