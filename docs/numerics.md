# Numerical Methods

## Solver

Backward induction over age, from `age_max` to `age_start`.

At each age and state grid point, the Bellman equation is solved via
grid search over consumption and stock share, with Gaussian quadrature
integration over stochastic shocks.

## State Grids

| Dimension | Method | Default |
|-----------|--------|---------|
| $x_t$ (equity premium) | `np.linspace` | 15 points, $\bar{x} \pm 3\sigma_{unc}$ |
| $m_t$ (cash-on-hand) | `np.geomspace` | 40 points, [0.01, 30.0] |
| $cm_t$ (lagged consumption) | `np.linspace` | 10 points, [0.01, 3.0] |

The geometric spacing for $m_t$ provides higher density at low wealth
levels, where policy functions curve most sharply.

## Quadrature

Gauss-Hermite quadrature (probabilist's convention) over 4 shocks:

1. Equity premium innovation ($\xi$): 5 nodes
2. Cash-flow news ($N_{CF}$): 5 nodes (with crash/non-crash mixture)
3. Persistent income shock ($\eta$): 5 nodes
4. Transitory income shock ($\varepsilon$): 3 nodes

Total integration points: up to $5 \times 10 \times 5 \times 3 = 750$
(doubled NCF nodes for crash mixture).

## Interpolation

Linear interpolation via `scipy.interpolate.RegularGridInterpolator` with
nearest-neighbor extrapolation outside grid bounds.

## Performance

| Calibration | Grid | Time (no Numba) |
|-------------|------|-----------------|
| `toy_demo_small_grid` | 3x5x3 | ~12 seconds |
| Full baseline | 15x40x10 | ~30 minutes |
