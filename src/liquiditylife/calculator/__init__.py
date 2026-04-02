"""Lightweight asset allocation calculator using precomputed lookup tables."""

from liquiditylife.calculator.recommend import Recommendation, UserInputs, recommend
from liquiditylife.calculator.tables import (
    export_tables_json,
    load_tables_json,
    precompute_tables,
)

__all__ = [
    "Recommendation",
    "UserInputs",
    "export_tables_json",
    "load_tables_json",
    "precompute_tables",
    "recommend",
]
