"""Synthetic data generation for the cash-flow prediction prototype."""
from synth.generator import (
    DiscretionaryCategory,
    HouseholdProfile,
    IncomeSource,
    PRESET_PROFILES,
    RecurringBill,
    SyntheticDataset,
    generate,
)

__all__ = [
    "DiscretionaryCategory",
    "HouseholdProfile",
    "IncomeSource",
    "PRESET_PROFILES",
    "RecurringBill",
    "SyntheticDataset",
    "generate",
]
