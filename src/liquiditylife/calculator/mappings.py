"""Mappings from user-facing inputs to model parameters."""

from __future__ import annotations

# Friction levels (phi_c) based on expense rigidity
FRICTION_LOW = "low_friction"  # phi_c = 0
FRICTION_MED = "med_friction"  # phi_c = 5
FRICTION_HIGH = "high_friction"  # phi_c = 10

FRICTION_PHI_C: dict[str, float] = {
    FRICTION_LOW: 0.0,
    FRICTION_MED: 5.0,
    FRICTION_HIGH: 10.0,
}

# Risk tolerance levels mapped to gamma
RISK_AGGRESSIVE = "aggressive"  # gamma = 3
RISK_MODERATE = "moderate"  # gamma = 5
RISK_CONSERVATIVE = "conservative"  # gamma = 8

RISK_GAMMA: dict[str, float] = {
    RISK_AGGRESSIVE: 3.0,
    RISK_MODERATE: 5.0,
    RISK_CONSERVATIVE: 8.0,
}


def expense_ratio_to_friction(expense_ratio: float) -> str:
    """Map annual fixed expenses / income ratio to a friction bucket.

    Args:
        expense_ratio: (monthly_fixed_expenses * 12) / annual_income

    Returns:
        Friction level key: 'low_friction', 'med_friction', or 'high_friction'.
    """
    if expense_ratio < 0.30:
        return FRICTION_LOW
    if expense_ratio < 0.60:
        return FRICTION_MED
    return FRICTION_HIGH


def risk_tolerance_to_gamma(score: int) -> str:
    """Map a 1-5 risk tolerance score to a gamma bucket.

    Args:
        score: 1 (most aggressive) to 5 (most conservative).

    Returns:
        Risk level key: 'aggressive', 'moderate', or 'conservative'.
    """
    if score <= 2:
        return RISK_AGGRESSIVE
    if score <= 3:
        return RISK_MODERATE
    return RISK_CONSERVATIVE


def scenario_key(friction: str, risk: str) -> str:
    """Build the lookup table key from friction and risk levels."""
    return f"{friction}_{risk}"


ALL_SCENARIO_KEYS: list[str] = [
    scenario_key(f, r)
    for f in [FRICTION_LOW, FRICTION_MED, FRICTION_HIGH]
    for r in [RISK_AGGRESSIVE, RISK_MODERATE, RISK_CONSERVATIVE]
]
