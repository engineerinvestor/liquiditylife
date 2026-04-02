# Asset Allocation Calculator

A lightweight tool that translates the model's conclusions into instant, personalized asset allocation recommendations — no solver required.

## How It Works

The calculator uses **precomputed lookup tables** derived from solving the model across 9 scenarios (3 expense friction levels x 3 risk tolerance levels). Given a user's age, income, savings, expenses, and risk tolerance, it interpolates the cached policy surface and returns a recommendation in microseconds.

## Quick Start

### Python API

```python
from liquiditylife.calculator import UserInputs, recommend

rec = recommend(UserInputs(
    age=35,
    annual_income=150_000,
    liquid_savings=200_000,
    monthly_fixed_expenses=5_000,
    risk_tolerance=3,  # 1=aggressive, 5=conservative
))

print(f"Stock share: {rec.stock_share_pct}%")
print(f"Emergency fund: {rec.emergency_fund_months} months")
print(f"Stocks: ${rec.stocks_dollars:,.0f}")
print(f"Safe assets: ${rec.safe_dollars:,.0f}")
```

### CLI

```bash
liquiditylife calculator recommend \
  --age 35 --income 150000 --savings 200000 --expenses 5000 --risk 3
```

## User Inputs

| Input | Description | Maps to |
|-------|-------------|---------|
| `age` | Current age (18-99) | Age in the lifecycle model |
| `annual_income` | Gross annual income ($) | Denominator for wealth-to-income ratio |
| `liquid_savings` | Total liquid savings ($) | Cash-on-hand ratio (m_t) |
| `monthly_fixed_expenses` | Fixed monthly costs ($) | Expense friction level (phi_c) |
| `risk_tolerance` | 1 (aggressive) to 5 (conservative) | Risk aversion (gamma) |

## Output

The `Recommendation` object includes:

- **`stock_share_pct`** — Recommended % of liquid wealth in stocks
- **`emergency_fund_months`** — Minimum months of expenses in safe assets
- **`stocks_dollars`** / **`safe_dollars`** — Dollar breakdown
- **`trajectory`** — How the recommendation changes at ages +5, +10, +15
- **`sensitivity_extra_savings`** — Stock share if savings were 50% higher

## Scenario Matrix

The calculator maps user inputs to one of 9 precomputed scenarios:

| Expense Ratio | Risk 1-2 | Risk 3 | Risk 4-5 |
|--------------|----------|--------|----------|
| < 30% of income | gamma=3, phi_c=0 | gamma=5, phi_c=0 | gamma=8, phi_c=0 |
| 30-60% | gamma=3, phi_c=5 | gamma=5, phi_c=5 | gamma=8, phi_c=5 |
| > 60% | gamma=3, phi_c=10 | gamma=5, phi_c=10 | gamma=8, phi_c=10 |

## Custom Tables

To generate full-lifecycle tables (ages 25-99) with higher accuracy:

```bash
liquiditylife calculator precompute --output my_tables.json
```

Then load them:

```python
from liquiditylife.calculator import load_tables_json, recommend, UserInputs
from pathlib import Path

tables = load_tables_json(Path("my_tables.json"))
rec = recommend(UserInputs(...), tables=tables)
```

## Web Integration

The precomputed tables export as compact JSON (~50-200KB). A web frontend can:

1. Fetch the JSON once (cache it)
2. Accept user inputs via form fields
3. Interpolate the lookup table client-side
4. Display the recommendation instantly

No server-side computation is needed.

## Disclaimer

This calculator provides approximate asset allocation guidance based on an academic model. It is not financial advice. Consult a qualified financial advisor for personalized recommendations.
