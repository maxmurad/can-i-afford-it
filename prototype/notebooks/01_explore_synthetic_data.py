"""M1 — Explore the synthetic data.

Runnable today. This is the exploratory pass that confirms the generator
produces realistic ledgers before any model is built.

Run from the prototype/ directory:
    uv run python notebooks/01_explore_synthetic_data.py

Or open it cell-by-cell — the `# %%` markers make it a Jupyter-style
notebook in VS Code / PyCharm.
"""
# %%
import sys
from pathlib import Path

# make `synth` importable when run as a plain script from notebooks/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import pandas as pd

from synth.generator import PRESET_PROFILES, generate

pd.set_option("display.width", 120)
pd.set_option("display.max_columns", 12)

# %%
# Generate one household from each preset and print a summary.
datasets = {}
for key in PRESET_PROFILES:
    ds = generate(PRESET_PROFILES[key], start="2024-01-01", n_days=365, seed=0)
    datasets[key] = ds
    print(ds.summary())
    print("-" * 64)

# %%
# Look at the raw ledger for the primary persona.
ds = datasets["steady_biweekly"]
print("\nFirst 15 transactions:")
print(ds.transactions.head(15).to_string(index=False))

# %%
# Recurring vs discretionary breakdown.
t = ds.transactions
print("\nTransaction mix by type:")
print(t.groupby("txn_type").agg(n=("amount", "size"), total=("amount", "sum")))

print("\nSpend by category (debits only):")
debits = t[t.amount < 0]
print(debits.groupby("category")["amount"].agg(["size", "sum"]).sort_values("sum"))

# %%
# Ground truth: the schedules behind the data — this is what Stage 1/2 must recover.
print("\nGround-truth income sources:")
for src in ds.ground_truth["income"]:
    print(f"  {src.name:20s} {src.merchant:28s} {src.frequency:12s} ~${src.amount_mean:,.0f}")

print("\nGround-truth recurring bills:")
for bill in ds.ground_truth["recurring"]:
    print(f"  {bill.name:20s} {bill.merchant:28s} day {bill.day_of_month:2d}  ~${bill.amount_mean:,.0f}")

# %%
# Plot the running balance for every preset — eyeball realism and overdraft risk.
fig, axes = plt.subplots(len(datasets), 1, figsize=(11, 3 * len(datasets)), sharex=True)
for ax, (key, ds) in zip(axes, datasets.items()):
    ds.balance_series.plot(ax=ax, lw=1.2)
    ax.axhline(0, color="red", lw=0.8, ls="--")
    ax.set_title(f"{key} — running balance")
    ax.set_ylabel("$")
fig.tight_layout()
out = Path(__file__).resolve().parent.parent / "figures"
out.mkdir(exist_ok=True)
fig.savefig(out / "01_balance_trajectories.png", dpi=120)
print(f"\nSaved balance plot -> {out / '01_balance_trajectories.png'}")

# %%
# Sanity checks you'd want before trusting the generator.
for key, ds in datasets.items():
    t = ds.transactions
    assert t["date"].is_monotonic_increasing, f"{key}: dates not sorted"
    assert (t["is_recurring"] == (t["source_id"].notna())).all(), f"{key}: label mismatch"
    recon = ds.profile.starting_balance + t["amount"].sum()
    assert abs(recon - ds.balance_series.iloc[-1]) < 0.01, f"{key}: balance reconciliation off"
print("\nAll sanity checks passed.")
