"""User-facing recommendation engine."""

from __future__ import annotations

import math

from pydantic import BaseModel, field_validator

from liquiditylife.calculator.mappings import (
    expense_ratio_to_friction,
    risk_tolerance_to_gamma,
    scenario_key,
)
from liquiditylife.calculator.tables import PrecomputedTable, load_tables_json


class UserInputs(BaseModel, frozen=True):
    """Inputs a DIY financial planner would know."""

    age: int
    """Current age (25-99)."""

    annual_income: float
    """Gross annual income in dollars."""

    liquid_savings: float
    """Total liquid savings: bank accounts + brokerage (not retirement accounts)."""

    monthly_fixed_expenses: float
    """Monthly fixed expenses: mortgage/rent, childcare, insurance, etc."""

    risk_tolerance: int
    """Risk tolerance on a 1-5 scale (1=aggressive, 5=conservative)."""

    @field_validator("age")
    @classmethod
    def _age_valid(cls, v: int) -> int:
        if not 18 <= v <= 99:
            msg = "Age must be between 18 and 99"
            raise ValueError(msg)
        return v

    @field_validator("annual_income")
    @classmethod
    def _income_positive(cls, v: float) -> float:
        if v <= 0:
            msg = "Annual income must be positive"
            raise ValueError(msg)
        return v

    @field_validator("liquid_savings")
    @classmethod
    def _savings_non_negative(cls, v: float) -> float:
        if v < 0:
            msg = "Liquid savings cannot be negative"
            raise ValueError(msg)
        return v

    @field_validator("risk_tolerance")
    @classmethod
    def _risk_valid(cls, v: int) -> int:
        if not 1 <= v <= 5:
            msg = "Risk tolerance must be 1-5"
            raise ValueError(msg)
        return v


class TrajectoryPoint(BaseModel, frozen=True):
    """A single point in the age trajectory."""

    age: int
    stock_share_pct: float
    stocks_dollars: float


class Recommendation(BaseModel, frozen=True):
    """The calculator's output."""

    stock_share_pct: float
    """Recommended stock share as a percentage (0-100)."""

    emergency_fund_months: int
    """Recommended emergency fund in months of expenses."""

    stocks_dollars: float
    """Dollar amount to hold in stocks."""

    safe_dollars: float
    """Dollar amount to hold in safe assets (bonds, cash)."""

    trajectory: list[TrajectoryPoint]
    """How the recommendation changes at future ages."""

    sensitivity_extra_savings: float
    """Stock share (%) if savings were 50% higher."""

    friction_level: str
    """Expense friction bucket used."""

    risk_level: str
    """Risk tolerance bucket used."""

    expense_ratio: float
    """Annual fixed expenses as a fraction of income."""

    wealth_to_income: float
    """Liquid savings divided by annual income."""


def recommend(
    inputs: UserInputs,
    tables: dict[str, PrecomputedTable] | None = None,
) -> Recommendation:
    """Generate an asset allocation recommendation from user inputs.

    Args:
        inputs: User-provided financial inputs.
        tables: Precomputed lookup tables. If None, loads default tables.

    Returns:
        A ``Recommendation`` with stock share, emergency fund, and trajectory.
    """
    if tables is None:
        tables = load_tables_json()

    # Map inputs to model coordinates
    m_t = inputs.liquid_savings / inputs.annual_income
    expense_ratio = (inputs.monthly_fixed_expenses * 12) / inputs.annual_income

    friction = expense_ratio_to_friction(expense_ratio)
    risk = risk_tolerance_to_gamma(inputs.risk_tolerance)
    key = scenario_key(friction, risk)

    table = tables[key]

    # Current recommendation
    stock_share = table.lookup(inputs.age, m_t)
    stock_share_pct = round(stock_share * 100, 1)

    # Emergency fund: at least 6 months, scale with expense ratio
    ef_months = max(6, math.ceil(12 * expense_ratio))

    # Dollar breakdown
    stocks_dollars = round(inputs.liquid_savings * stock_share, 0)
    safe_dollars = round(inputs.liquid_savings - stocks_dollars, 0)

    # Trajectory at +5, +10, +15 years
    trajectory: list[TrajectoryPoint] = []
    for delta in [5, 10, 15]:
        future_age = inputs.age + delta
        if future_age > 99:
            break
        future_share = table.lookup(future_age, m_t)
        trajectory.append(TrajectoryPoint(
            age=future_age,
            stock_share_pct=round(future_share * 100, 1),
            stocks_dollars=round(inputs.liquid_savings * future_share, 0),
        ))

    # Sensitivity: what if savings were 50% higher?
    m_t_extra = (inputs.liquid_savings * 1.5) / inputs.annual_income
    sens_share = table.lookup(inputs.age, m_t_extra)

    return Recommendation(
        stock_share_pct=stock_share_pct,
        emergency_fund_months=ef_months,
        stocks_dollars=stocks_dollars,
        safe_dollars=safe_dollars,
        trajectory=trajectory,
        sensitivity_extra_savings=round(sens_share * 100, 1),
        friction_level=friction,
        risk_level=risk,
        expense_ratio=round(expense_ratio, 3),
        wealth_to_income=round(m_t, 2),
    )
