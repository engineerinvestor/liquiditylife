# Economic Model

## Overview

The model is a life-cycle problem where a household chooses consumption and
portfolio allocation each period, facing risky labor income, time-varying
equity returns, and asymmetric consumption adjustment costs.

## State Variables (Scaled Mode)

| Variable | Description |
|----------|-------------|
| age | Current age |
| $x_t$ | Equity premium state (AR(1)) |
| $m_t$ | Cash-on-hand / persistent earnings |
| $cm_t$ | Lagged consumption ratio |

## Controls

- $c_t$: consumption (scaled)
- $\theta_t$: stock share of liquid wealth, $\theta \in [0, 1]$

## Key Equations

**Equity premium transition:**
$$x_{t+1} = \bar{x} + \phi_x (x_t - \bar{x}) + \xi_{t+1}$$

**Realized stock return:**
$$R_{stock} = r_f + x_t + N_{CF} - N_{DR}$$

**Consumption adjustment cost (asymmetric, applies only when $c_t < c_{t-1}$):**
$$\Phi_C = \frac{\phi_c}{2} \frac{(\max(0, cm_t - c_t))^2}{cm_t}$$

**Epstein-Zin utility recursion:**
$$V_t = [(1-\beta) c_t^\rho + \beta \text{CE}_t^\rho]^{1/\rho}$$

where $\rho = 1 - 1/\psi$ and $\text{CE}_t = (\mathbb{E}[V_{t+1}^{1-\gamma}])^{1/(1-\gamma)}$.

## Paper vs. Package

| Aspect | Paper | Package |
|--------|-------|---------|
| Income process params | Estimated from confidential tax data | Public approximation from qualitative targets |
| Crash mixture | Calibrated to data | Parametric approximation |
| Grid resolution | Not specified | Configurable via `GridSpec` |
| Quadrature | "Gaussian quadrature over 4 shocks" | Gauss-Hermite with configurable nodes |

## Approximations

All income process slope parameters (lambda coefficients linking persistent
earnings shocks to return news) are derived from the paper's qualitative
targets rather than confidential estimation. These are marked with
`source="public_approximation"` in the calibration metadata.
