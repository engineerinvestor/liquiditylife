## `liquiditylife` (working name)

### Open-source Python package for life-cycle portfolio choice with liquidity risk, labor-income risk, and consumption adjustment frictions

## 1. Executive summary

`liquiditylife` is an open-source Python package for implementing and exploring a life-cycle portfolio choice model in the spirit of Patrick Adams’ “Stocks for the Long Run or Liquidity? Tax Data Evidence and Portfolio Choice Implications.” The package is designed to support:

* research-grade model simulation,
* policy-function computation,
* calibration and sensitivity analysis,
* scenario comparison,
* and interactive visualization tools such as Streamlit, Plotly Dash, Panel, or web calculators.

The package will not attempt to recreate the confidential tax-data pipeline from the paper in v1. Instead, it will implement a **paper-consistent structural model core** with public calibration bundles and fast sweepable outputs suitable for visual interfaces. The underlying model structure is motivated by Adams’ annual life-cycle problem with risky labor income, time-varying expected returns, consumption adjustment frictions, illiquid saving during working life, retirement flow income, scaled state reduction, and Gaussian-quadrature-based numerical integration.   

## 2. Recommendation

### Recommendation

**Build it.**

### Why

The model fills a real gap between:

* simplistic “100% stocks for the long run” tools,
* academic code that is hard to reuse,
* and public-facing visual tools that ignore labor-income risk and liquidity needs.

This package would let users explore questions like:

* How does the optimal stock share change with age?
* How sensitive is policy to liquid wealth, fixed expenses, labor-income risk, and adjustment frictions?
* How large does a safe buffer need to be before equity allocation rises materially?
* How do working-age and retired households differ?
* How much of the result is driven by discount-rate news versus income risk versus spending rigidity?

These are exactly the mechanisms emphasized in the paper. Adams’ model uses a time-varying equity premium, risky earnings correlated with stock-return news, and an asymmetric consumption adjustment cost; high adjustment costs are needed to match the observed savings response, and they sharply reduce optimal stock shares for working-age households.   

### Why not overreach in v1

I do not recommend:

* building the confidential tax-data estimation pipeline into v1,
* building a full general-equilibrium model,
* adding many asset classes at launch,
* or making the package UI-dependent.

The paper’s value for open-source users is in the **structural policy engine**, not in reproducing inaccessible data plumbing. Also, interactive tools need fast, stable, serializable outputs more than they need every empirical appendix on day one.

## 3. Product vision

### Vision statement

Create the default open-source Python framework for **liquidity-aware life-cycle portfolio choice**, where users can move from:

* economic assumptions,
* to calibrated model objects,
* to optimal policy functions,
* to household simulations,
* to scenario sweeps,
* to interactive visualizations,

without rewriting custom academic scripts.

### Primary value proposition

The package should make it easy to answer questions like:

* “What is the optimal stock share of liquid wealth for a 35-year-old with high income risk and sticky expenses?”
* “How does a higher emergency-fund buffer change the stock allocation surface?”
* “What happens when consumption adjustment frictions rise from 0 to 10?”
* “How much does correlated labor-income risk matter versus pure return predictability?”
* “Why might retirees hold more equities than some working-age households in this framework?”

## 4. Design principles

1. **Paper-consistent, not paper-mystical**
   Implement the model transparently and explicitly. Where the paper leaves room for numerical interpretation, document the choice.

2. **Headless core, UI-ready outputs**
   The package should power many interfaces, not bundle one mandatory app.

3. **Fast sweeps matter**
   Interactive tools need cached grids, interpolation, and reusable simulation objects.

4. **Calibration is modular**
   Users should be able to swap in baseline, frictionless, high-friction, or custom calibrations.

5. **Separation of model and presentation**
   Core economics and numerics belong in the engine; charts and dashboards are adapters.

6. **Reproducibility over cleverness**
   Every computed surface or simulation should be traceable to a named calibration and solver configuration.

## 5. Scope

### 5.1 In scope for v1

* structural life-cycle model engine
* dynamic programming solver
* simulation engine
* calibration bundles
* sensitivity/sweep engine
* policy-function export
* visualization-friendly outputs
* plotting helpers
* notebooks and examples
* CLI/API for reproducible runs

### 5.2 Out of scope for v1

* confidential administrative tax-data estimation
* full empirical replication of Section 4 regressions
* endogenous labor supply
* housing as a separately optimized state variable
* general equilibrium with endogenous asset prices
* multi-country tax systems
* full asset-location tax optimizer
* multi-asset international portfolio choice

## 6. Economic model summary

The v1 package shall implement a model with the following core structure:

* annual time steps,
* working-age and retirement phases,
* liquid wealth allocated between a risk-free bond and risky aggregate stock,
* a time-varying one-period-ahead equity premium following an AR(1),
* realized stock returns decomposed into cash-flow news and discount-rate news,
* labor income with persistent and transitory components,
* persistent earnings shocks whose distribution depends on contemporaneous stock-return news,
* asymmetric quadratic costs of reducing consumption below the previous period’s level,
* a fixed share of working-life income allocated to illiquid wealth,
* a retirement flow-income proxy tied to terminal persistent earnings,
* solution using scaled state variables and Gaussian quadrature.    

### 6.1 Important implementation note

The package should distinguish between:

* **paper-consistent mode**: faithful to the published equations and baseline calibration structure
* **exploration mode**: allows softened assumptions, alternative distributions, faster approximations, and richer visualization sweeps

## 7. High-level architecture

```text
liquiditylife/
├── calibrations/     # named parameter bundles
├── core/             # domain models and contracts
├── processes/        # return and income stochastic processes
├── model/            # budget constraints, utility, transitions
├── solve/            # dynamic programming, interpolation, quadrature
├── simulate/         # cohort and household simulations
├── sweep/            # parameter grids and comparative statics
├── vizdata/          # UI-friendly serialization / caching
├── plotting/         # matplotlib/plotly-ready helpers
├── io/               # save/load configs and results
├── cli/              # command line entry points
├── examples/         # tutorials and demos
└── tests/            # verification and regression tests
```

## 8. Recommended package layers

### Layer A: Economic domain layer

Defines the conceptual model:

* household lifecycle
* preferences
* budget rules
* state variables
* shocks
* calibrations

### Layer B: Numerical engine layer

Handles:

* quadrature nodes and weights
* state grids
* backward induction
* interpolation
* simulation draws
* caching of policy functions

### Layer C: Visualization interface layer

Produces:

* age-policy curves
* policy surfaces
* cohort distributions
* scenario comparison payloads
* JSON/xarray/pandas outputs for dashboards

## 9. Core abstractions

### 9.1 `Preferences`

Represents household preferences.

Fields:

* `gamma: float` — relative risk aversion
* `psi: float` — elasticity of intertemporal substitution
* `beta: float` — time discount factor

Notes:

* Baseline paper values discussed in the model section include `gamma = 5`, `psi = 0.5`, and `beta = 0.85`. 

### 9.2 `Lifecycle`

Represents the age structure.

Fields:

* `age_start: int`
* `age_retire: int`
* `age_max: int`

### 9.3 `AssetReturnProcess`

Represents the risky and risk-free return structure.

Fields:

* `rf: float`
* `x_bar: float`
* `phi_x: float`
* `sigma_r: float`
* `rho_cs: float = 0.96`
* crash-mixture parameters for `(NCF, NDR)`

Responsibilities:

* evolve the equity premium state
* generate `NDR` from premium innovation
* sample joint news shocks
* compute realized stock returns

The paper specifies an AR(1) equity premium and discount-rate news derived from that state, with a crash/non-crash mixture for cash-flow and discount-rate news.  

### 9.4 `IncomeProcess`

Represents nonfinancial income.

Fields:

* `age_drift_curve`
* `sigma_eps`
* `sigma_eta`
* `p_eta_bad`
* coefficients linking `eta` distribution to `(NCF, NDR)`

Responsibilities:

* decompose income into persistent and transitory components
* evolve persistent earnings
* allow state-dependent bad-tail earnings events

The paper models income as `Y = Y_bar * exp(eps)`, with persistent log earnings following a random walk plus age drift, and persistent shock distribution conditioned on contemporaneous return-news components. 

### 9.5 `AdjustmentCostModel`

Represents consumption down-adjustment frictions.

Fields:

* `phi_c: float`

Responsibilities:

* compute `Phi_C(C_t, C_{t-1})`
* expose marginal and average adjustment-cost diagnostics

The paper uses an asymmetric quadratic adjustment cost that applies only when current consumption falls below prior consumption. 

### 9.6 `IlliquidWealthRule`

Represents working-life illiquid saving and retirement flow income.

Fields:

* `s: float`
* `S: float`
* `tau: float`

Responsibilities:

* apply working-age illiquid contribution share
* apply retirement flow income proxy
* compute available liquid resources

The paper’s baseline discussion uses `s = 0.15`, `S = 0.60`, and `tau = 0.35`.  

### 9.7 `HouseholdState`

Canonical state at start of period.

Fields:

* `age`
* `x_t`
* `m_t` or `M_t`
* `lagged_consumption_ratio`
* optional `Y_t` for unscaled mode

Notes:
The paper states the natural state variables are age, equity premium, cash-on-hand, lagged consumption, and persistent earnings, and then reduces the problem by scaling by persistent earnings. 

### 9.8 `PolicyFunction`

Represents solved controls.

Methods:

* `consume(state) -> float`
* `stock_share(state) -> float`
* `value(state) -> float`

### 9.9 `SimulationResult`

Represents simulated cohorts.

Fields:

* paths of states
* controls
* returns
* income
* savings flows
* summary statistics
* calibration metadata
* solver metadata

## 10. Mathematical model requirements

### 10.1 State representation

The solver shall support two modes:

#### A. Scaled-state mode (default)

Tracks:

* age
* equity premium `x_t`
* cash-on-hand relative to persistent earnings `m_t`
* lagged-consumption ratio

This follows the paper’s dimensionality reduction approach. 

#### B. Unscaled mode (debug/reference)

Tracks:

* age
* `x_t`
* `M_t`
* `C_{t-1}`
* `Y_t`

Use for testing and sanity checks.

### 10.2 Controls

Per period, the household chooses:

* current consumption `C_t`
* stock share of liquid investable wealth `theta_t`

### 10.3 Transition equations

The engine shall implement:

* equity premium transition
* stock return realization from news decomposition
* persistent earnings transition
* transitory shock application
* cash-on-hand law of motion
* retirement regime switch

### 10.4 Adjustment cost function

The engine shall implement the asymmetric quadratic form used in the paper as the default specification, while allowing alternative friction models via plugin architecture. 

### 10.5 Retirement treatment

The baseline shall mirror the paper’s parsimonious retirement-income proxy rather than requiring full accumulated retirement-account accounting. This keeps the state space manageable and faithful to the intended mechanism. 

## 11. Numerical methods

### 11.1 Solver method

Recommended v1 method:

* backward induction over age
* discretized state grids
* Gaussian quadrature integration over shocks
* interpolation for value and policy functions

This matches the paper’s description closely enough to preserve interpretability while remaining open-source friendly. The paper explicitly states that Gaussian quadrature is used over the four shocks. 

### 11.2 Grid strategy

Recommended state grids:

* `x_t`: moderately fine grid over equity-premium state
* `m_t`: nonlinear grid, denser near low cash-on-hand
* lagged-consumption ratio: coarse-to-medium grid
* age: integer annual grid

### 11.3 Interpolation

Recommended:

* linear or monotone cubic interpolation for policy functions
* explicit extrapolation guards
* cached interpolation objects

### 11.4 Simulation

The package shall support:

* single-household path simulation
* many-household cohort simulation
* lifecycle cross-sectional snapshots
* deterministic scenario playback
* Monte Carlo runs with reproducible seeds

### 11.5 Performance recommendation

For v1:

* NumPy + SciPy baseline
* Numba optional acceleration

For v2:

* JAX optional backend for sweeps and auto-vectorization

I would not make JAX mandatory in v1. Interactive tools benefit more from stable cached surfaces than from a hard JAX dependency.

## 12. Calibration system

### 12.1 Named calibration bundles

Ship with at least:

* `adams_baseline`
* `adams_frictionless`
* `adams_moderate_friction`
* `adams_high_friction`
* `toy_demo_small_grid`

### 12.2 Baseline parameters

The baseline bundle should expose, at minimum:

* `gamma = 5`
* `psi = 0.5`
* `beta = 0.85`
* `tau = 0.35`
* `s = 0.15`
* `S = 0.60`
* `rf = 0.02`
* `x_bar = 0.05`
* `phi_x = 0.85`
* `rho_cs = 0.96`
* `phi_c` selectable across values like `0`, `5`, `10`

These values are discussed in the paper’s model section, and `phi_c = 10` is highlighted as much closer to the empirically estimated savings response than lower-friction alternatives.  

### 12.3 Calibration philosophy

The package shall explicitly separate:

* **published baseline parameters**
* **public approximation parameters**
* **user custom parameters**

Every result object must record which category was used.

## 13. API design

### 13.1 Example user workflow

```python
from liquiditylife.calibrations import load_calibration
from liquiditylife.solve import solve_model
from liquiditylife.simulate import simulate_cohorts
from liquiditylife.sweep import policy_surface

cal = load_calibration("adams_high_friction")
solution = solve_model(cal)

sim = simulate_cohorts(
    solution,
    n_households=100_000,
    seed=42
)

surface = policy_surface(
    solution,
    age=35,
    x_values="default",
    m_values="default"
)
```

### 13.2 API rules

* Keep model creation explicit.
* No hidden global config.
* Return rich results, not bare arrays.
* Expose solver diagnostics.
* Make all major objects serializable.
* Preserve calibration provenance in outputs.

## 14. Visualization requirements

This is central to the project.

### 14.1 Visualization-friendly outputs

The package shall export:

* pandas DataFrames
* xarray Datasets
* compact JSON payloads
* parquet files for cached sweeps

### 14.2 Required visualization primitives

The package shall support generation of:

* age vs average optimal stock share
* age vs median liquid wealth
* stock share as a surface over `(age, liquid wealth/income)`
* stock share as a surface over `(x_t, m_t)`
* policy comparison under `phi_c = 0, 5, 10`
* simulated distribution of drawdowns after income shocks
* savings response plots analogous to model Figure 10
* policy plots analogous to model Figures 11 and 12

The paper reports that without adjustment frictions the average optimal stock share is about 50% across much of working life, while higher friction markedly lowers those shares and raises wealth accumulation. Those are exactly the kinds of outputs a dashboard should make explorable. 

### 14.3 Dashboard adapter recommendation

Provide optional helpers for:

* Streamlit
* Plotly
* Panel

But keep them in extras:

```bash
pip install liquiditylife[viz]
```

### 14.4 Important UI recommendation

Do not solve the full dynamic program on every slider move in a web app. Precompute and cache:

* calibration solutions,
* common policy grids,
* comparison surfaces,
* and simulation summaries.

Interactive tools should query cached surfaces first.

## 15. Research and educational modes

### 15.1 Research mode

Priorities:

* fidelity
* reproducibility
* calibration transparency
* simulation scale
* publication-quality plots

### 15.2 Educational mode

Priorities:

* speed
* small grids
* intuitive surfaces
* explainers for mechanism decomposition

### 15.3 Mechanism-decomposition mode

Support toggling:

* income correlation off/on
* adjustment cost off/on
* return predictability off/on
* illiquid saving off/on

This is one of the most valuable features for teaching the paper’s intuition.

## 16. Validation and verification

### 16.1 Unit tests

Test:

* AR(1) equity-premium transitions
* return decomposition accounting
* income-process moments
* adjustment-cost function values
* budget-constraint transitions
* retirement regime switch
* scaling/unscaling consistency

### 16.2 Numerical tests

Test:

* policy monotonicity in cash-on-hand where expected
* stable convergence as grids are refined
* quadrature integration sanity
* reproducibility under fixed seeds

### 16.3 Economic-regression tests

The package should include checks that, under shipped calibrations:

* frictionless mode gives materially higher stock shares than high-friction mode
* higher `phi_c` raises precautionary wealth accumulation
* high-friction mode produces larger savings responses to persistent income shocks
* working-age stock shares can fall well below frictionless values

These directions are supported by the paper’s reported results.  

### 16.4 Non-goal for validation

Do not claim “exact replication of the paper” unless the package reproduces published figures to a documented tolerance under a documented calibration.

## 17. CLI requirements

Example commands:

```bash
liquiditylife solve --calibration adams_high_friction
liquiditylife simulate --calibration adams_high_friction --n 100000
liquiditylife sweep policy-surface --age 35 --calibration adams_high_friction
liquiditylife export vizdata --calibration adams_high_friction
```

## 18. Documentation requirements

The repo shall include:

* `README.md`
* `SPEC.md`
* `docs/model.md`
* `docs/calibrations.md`
* `docs/numerics.md`
* `docs/viz.md`
* `examples/quickstart.ipynb`
* `examples/streamlit_demo.ipynb`

### 18.1 Documentation philosophy

Separate:

* what the paper says,
* what the package implements,
* and where the package deliberately approximates or generalizes.

## 19. Dependencies

### Required

* NumPy
* SciPy
* pandas
* xarray
* pydantic or dataclasses
* matplotlib

### Optional

* Numba
* Plotly
* Streamlit
* JAX
* pyarrow

### Dependency philosophy

Keep base install light:

```bash
pip install liquiditylife
```

Extras:

```bash
pip install liquiditylife[viz]
pip install liquiditylife[fast]
pip install liquiditylife[dev]
```

## 20. What not to over-engineer

Do **not** overbuild:

* bespoke PDE machinery
* distributed compute orchestration
* a giant calibration zoo
* a full frontend framework
* a full empirical estimation stack

The core win is a trustworthy model engine with great outputs.

## 21. Acceptance criteria

The package will be considered v1-ready when it can:

1. Solve the lifecycle model under at least three named calibrations.
2. Export policy functions for consumption and stock share.
3. Simulate cohorts over the full lifecycle.
4. Reproduce the directional comparative statics of frictionless vs high-friction cases.
5. Generate visualization-ready surfaces over age and liquid-wealth states.
6. Serialize results with calibration provenance.
7. Pass unit, numerical, and regression tests.
8. Provide at least one notebook and one dashboard example.
9. Document all approximations relative to the paper.
10. Run on a laptop without GPU requirements.

## 22. Roadmap

### Phase 1 — Core model skeleton

* domain models
* calibration object
* return and income processes
* budget constraint and transitions

### Phase 2 — Solver

* state grids
* backward induction
* quadrature integration
* policy interpolation

### Phase 3 — Simulation + sweeps

* cohort simulation
* comparative statics
* cached surfaces
* summary metrics

### Phase 4 — Visualization layer

* plot helpers
* viz payload exports
* Streamlit demo

### Phase 5 — Validation + docs

* solver tests
* economic-regression tests
* notebooks
* public release

## 23. Final recommendation

**Proceed.**

Best version of this project:

* open-source,
* model-first,
* visualization-friendly,
* and honest about what is paper-consistent versus paper-identical.

Best v1 framing:

> “An open-source lifecycle liquidity-risk modeling engine inspired by Adams (2026), built for interactive exploration of portfolio choice under labor-income risk and consumption rigidity.”

Not recommended for v1 framing:

> “Exact replication of the Adams paper.”

That claim is harder to defend unless you match published figures and calibrations extremely closely.

---

My recommendation in one sentence: **build a policy-engine package that exposes the paper’s mechanisms cleanly and powers interactive visualizations, rather than trying to ship a full confidential-data replication stack on day one.** The uploaded paper is a strong foundation for that approach.  
