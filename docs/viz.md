# Visualization

## Export Formats

```python
from liquiditylife.vizdata.export import (
    solution_to_dataframe,
    simulation_to_xarray,
    to_json_payload,
    to_parquet,
)

# Policy grids as tidy DataFrame
df = solution_to_dataframe(solution, ages=[30, 40, 50])

# Simulation paths as xarray Dataset
ds = simulation_to_xarray(sim_result)

# JSON for web dashboards
json_str = to_json_payload(df)

# Parquet for cached sweeps
to_parquet(df, Path("policies.parquet"))
```

## Plotting

```python
from liquiditylife.plotting.age_profiles import plot_age_stock_share
from liquiditylife.plotting.surfaces import plot_policy_surface
from liquiditylife.plotting.comparisons import plot_friction_comparison
from liquiditylife.sweep.surface import policy_surface

# Age profiles
plot_age_stock_share({"High friction": sim_hf, "Frictionless": sim_fl})

# Policy surface heatmap
surface = policy_surface(solution, age=35)
plot_policy_surface(surface, variable="stock_share")

# Friction comparison
plot_friction_comparison({"phi_c=0": sol_0, "phi_c=10": sol_10}, age=35)
```

## Dashboard Integration

Install optional dashboard dependencies:

```bash
pip install liquiditylife[viz]
```

For interactive dashboards, precompute and cache solutions rather than
solving on every slider move. Use `io.cache.save_solution()` and
`io.cache.load_solution()`.
