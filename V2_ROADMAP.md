# liquiditylife v2 Roadmap

## v1 Status

v1 is complete and published to PyPI. It delivers:

- Backward induction DP solver with Numba JIT (60x speedup)
- 5 calibration bundles, 9 calculator scenarios
- Cohort simulation, policy surfaces, CLI, docs
- 152+ tests, mypy strict, CI/CD

## v1 Known Limitation: Non-Monotonic Policy Surfaces

**Problem:** Stock share is non-monotonic in wealth-to-income (m_t) when the adjustment cost is active (phi_c > 0). The policy dips at moderate wealth, rises again at high wealth. The frictionless case (phi_c=0) is perfectly monotonic.

**Root cause:** The asymmetric adjustment cost creates a kink in the Bellman objective at c = cm_t (consumption equals lagged consumption). With a coarse cm_t state grid (3-5 points), trilinear interpolation across this kink introduces non-convexities in the interpolated value function that propagate through backward induction.

**Why v1 can't fix it:** More grid points don't help — the issue persists at 40 m_t points with 5-node quadrature. The problem is structural: linear interpolation + kink = non-convex value function. The fix requires either a much finer cm_t grid (25+ points, 5x solve time) or a fundamentally different numerical method.

**Impact:** Calculator recommendations are directionally correct (high friction < low friction, conservative < aggressive) but the intermediate wealth region shows artifacts. Charts of stock share vs wealth are jagged rather than smooth.

---

## v2 Goals

### 1. Endogenous Grid Method (EGM) for the Adjustment Cost Kink

**What:** Replace the grid-search-over-consumption approach with an endogenous grid method that solves the Euler equation directly, handling the adjustment cost kink analytically.

**Why:** EGM avoids interpolating the value function at the kink. Instead of searching over a consumption grid and evaluating V(c), it inverts the first-order condition to find c*(m) directly. The adjustment cost creates two regimes (c < cm_t and c >= cm_t), each with its own FOC. EGM solves both, finds the regime boundary endogenously, and produces a smooth, monotonic policy function.

**References:**
- Carroll (2006), "The Method of Endogenous Gridpoints for Solving Dynamic Stochastic Optimization Problems"
- Iskhakov et al. (2017), "The Endogenous Grid Method for Discrete-Continuous Dynamic Choice Models" (handles exactly this type of kink)

**Implementation:**
- New `solve/egm.py` module alongside the existing grid-search solver
- Solve the FOC in each regime separately
- Stitch regimes at the endogenous kink point
- Produces smooth, monotonic consumption and savings policies
- Stock share still optimized via grid search over theta (EGM handles consumption only)

**Expected outcome:** Smooth, monotonic policy surfaces suitable for publication-quality charts.

### 2. Finer cm_t Grid with Adaptive Refinement

**What:** If EGM is too complex, an intermediate fix: increase the cm_t grid from 5 to 25 points with adaptive refinement around the cm_t values where the kink matters most.

**Why:** More cm_t resolution reduces the interpolation error across the kink, even without eliminating it entirely.

**Trade-off:** 5x more cm_t points = 5x solve time per scenario. With Numba, a single scenario would go from ~5 min to ~25 min (full lifecycle). Total precompute for 9 scenarios: ~4 hours.

### 3. JAX Backend for Sweeps

**What:** Optional JAX backend for vectorized, GPU-accelerated parameter sweeps.

**Why:** The current Numba solver is sequential over state grid points. JAX could vectorize the Bellman evaluation over all (ix, im, icm) simultaneously on GPU, potentially 10-100x faster for large grids.

**When:** After EGM is working. JAX + EGM would enable real-time interactive dashboards with full-fidelity solutions.

### 4. Numba Parallelization

**What:** Add `numba.prange` over the outermost state dimension in the Bellman kernel.

**Why:** The current Numba kernel is single-threaded. Parallelizing the outer loop over ix (equity premium grid) would give 4-8x speedup on multi-core CPUs with no algorithmic changes.

**Implementation:** Change `for ix in range(nx)` to `for ix in numba.prange(nx)` and ensure no shared mutable state across iterations (already the case — each ix writes to independent output slices).

### 5. Full-Lifecycle Default Tables with EGM

**What:** Once EGM produces smooth policies, regenerate the default calculator tables at high fidelity and ship them with the package.

**Why:** The current default tables show the non-monotonic artifacts. EGM-based tables would be publication-quality.

### 6. Interactive Dashboard

**What:** Streamlit or Panel app that lets users explore policy surfaces, run the calculator, and compare calibrations interactively.

**Why:** Visual exploration is the primary use case for educators and financial planners. Smooth policies (from EGM) are a prerequisite — jagged charts would undermine credibility.

### 7. Publication-Quality Visualizations

**What:** Plotly/matplotlib charts suitable for Twitter, blog posts, and papers.

**Why:** Blocked by the non-monotonicity issue in v1. Once EGM produces smooth surfaces, the `scripts/make_charts.py` infrastructure is ready to generate them.

---

## v2 Priority Order

| Priority | Item | Depends On | Effort |
|----------|------|-----------|--------|
| **P0** | EGM solver for adjustment cost kink | — | 2-3 weeks |
| **P0** | Regenerate smooth default tables | EGM | 1 day |
| **P1** | Numba prange parallelization | — | 2-3 days |
| **P1** | Publication-quality charts | Smooth tables | 1-2 days |
| **P2** | Interactive dashboard (Streamlit) | Smooth tables | 1 week |
| **P2** | Adaptive cm_t grid (fallback if EGM delayed) | — | 3-5 days |
| **P3** | JAX backend | EGM working | 2-3 weeks |

---

## What NOT to do in v2

- **Don't add more calibration bundles** — 5 is enough until the solver is validated against published figures
- **Don't build a full web app** — Streamlit demo is sufficient for v2
- **Don't add multi-asset** — out of scope per SPEC section 5.2
- **Don't attempt exact paper replication** — requires confidential data parameters not available publicly
