"""M6 — Run the pipeline against the real Plaid API (sandbox).

This is the milestone where the engine stops being a pure synthetic-data
exercise and gets plumbed through a production financial-data API.

IMPORTANT — what this proves and what it does not:
  - PROVES: the full integration path works — Plaid auth, Item creation,
    /transactions/get pagination, the Plaid->internal schema adapter, and
    balance reconstruction from current-balance + transaction history.
  - DOES NOT PROVE: model performance on independent real-world data.
    Plaid's default sandbox user is too thin (~48 txns) to backtest, so we
    push our OWN generator households through Plaid's custom-sandbox-user
    API and pull them back. The data is still ours, round-tripped — the
    calibration numbers mirror M5. Independent validation needs real
    consented accounts, which only the post-sandbox step provides.

Setup (one-time):
  1. plaid.com -> Dashboard -> Team Settings -> Keys
  2. Put client_id + Sandbox secret in prototype/.env:
       PLAID_CLIENT_ID=...
       PLAID_SECRET=...
       PLAID_ENV=sandbox
  3. uv sync

Run from prototype/:
  uv run python notebooks/06_plaid_calibration.py

First run hits the Plaid sandbox API (creates 4 custom Items, pulls each
back) and caches everything to prototype/cache/. Re-runs read from cache.
"""
# %%
import datetime
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import pandas as pd

from data.normalize import (
    plaid_to_ledger,
    reconstruct_balance_series,
    summarize_snapshot,
)
from data.plaid_loader import fetch_custom_snapshot
from eval.backtest import run_real_data_backtest
from synth.generator import PRESET_PROFILES, generate

pd.set_option("display.width", 140)

# We push 150-day households, but Plaid's headless sandbox flow caps the
# RETURNED history at ~90 days (days_requested is only settable via the full
# Link UI). So the round-trip yields ~90 days; a 14-day holdout leaves ~75
# days of history. That is deliberately thin — M6 is the integration proof,
# not the calibration verdict. M5's 80-household synthetic backtest remains
# the substantive calibration evidence.
PLAID_TEST_DAYS = 150
HOLDOUT_DAYS = 14


# %%
# Generate households, push each through Plaid's custom sandbox user, pull back.
end = datetime.date.today()
start = (end - datetime.timedelta(days=PLAID_TEST_DAYS)).isoformat()
print(f"Generating households ({start} -> {end}) and round-tripping through Plaid sandbox...")
print("(first run hits the API and caches; re-runs read cache)\n")

datasets = []
for i, (name, profile) in enumerate(PRESET_PROFILES.items()):
    ds = generate(profile, start=start, n_days=PLAID_TEST_DAYS, seed=i)
    current_balance = float(ds.balance_series.iloc[-1])

    snapshot = fetch_custom_snapshot(name, ds.transactions, current_balance)
    ledger = plaid_to_ledger(snapshot)
    print("  " + summarize_snapshot(snapshot, ledger))

    balance = reconstruct_balance_series(ledger, current_balance=snapshot.current_balance)
    if len(balance) < HOLDOUT_DAYS + 60:
        print(f"    skipping {name} — only {len(balance)} days after round-trip")
        continue
    datasets.append((name, ledger, balance))

print(f"\n{len(datasets)} households usable for the Plaid-sourced backtest.")


# %%
# Run the M5-style calibration backtest on the Plaid round-tripped data.
print(f"\nRunning real-data backtest (holdout = {HOLDOUT_DAYS} days)...")
out = run_real_data_backtest(
    datasets,
    holdout_days=HOLDOUT_DAYS,
    n_sims=2000,
    decisions_per_run=15,
    safety_buffer=100.0,
)
cal = out["calibration"]
dec = out["decisions"]
decisions_df = out["decisions_df"]


# %%
# Reliability curve.
print("\n" + "=" * 78)
print("Plaid round-trip reliability curve  (stated vs empirical coverage)")
print("-" * 78)
for stated, empirical in sorted(cal.coverage_curve.items()):
    gap = empirical - stated
    sign = "+" if gap >= 0 else "-"
    print(f"  stated {stated:.0%}    empirical {empirical:>5.0%}    gap {sign}{abs(gap):.0%}")
print(f"\n  mean calibration error: {cal.calibration_error:.1%}")
print("=" * 78)


# %%
# Decision confusion matrix.
if dec is not None:
    total = dec.true_affordable + dec.true_not_affordable + dec.false_affordable + dec.false_not_affordable
    print("\n" + "=" * 78)
    print(f"Affordability decisions on Plaid-sourced data  (N = {total})")
    print("-" * 78)
    print(f"                              actually-affordable    actually-breached")
    print(f"  predicted affordable        {dec.true_affordable:>14,}            {dec.false_affordable:>14,}  <-- FALSE AFFORDABLE")
    print(f"  predicted not-affordable    {dec.false_not_affordable:>14,}            {dec.true_not_affordable:>14,}")
    print("-" * 78)
    print(f"  FALSE-AFFORDABLE RATE:    {dec.false_affordable_rate:.1%}")
    print("=" * 78)


# %%
# Save the reliability diagram.
fig, ax = plt.subplots(figsize=(6, 6))
ax.plot([0, 1], [0, 1], color="gray", lw=1.0, ls="--", label="ideal (calibrated)")
stated_list = sorted(cal.coverage_curve.keys())
empirical_list = [cal.coverage_curve[s] for s in stated_list]
ax.plot(stated_list, empirical_list, "o-", color="C2", lw=2, markersize=10, label="Plaid round-trip")
for s, e in zip(stated_list, empirical_list):
    ax.annotate(f"{e:.0%}", (s, e), textcoords="offset points", xytext=(8, -4), fontsize=9)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_xlabel("Stated coverage")
ax.set_ylabel("Empirical coverage")
ax.set_title("Reliability diagram — Plaid sandbox round-trip")
ax.legend(loc="upper left", fontsize=9)
ax.grid(alpha=0.3)
fig.tight_layout()
fig_dir = Path(__file__).resolve().parent.parent / "figures"
fig_dir.mkdir(exist_ok=True)
out_path = fig_dir / "06_plaid_reliability.png"
fig.savefig(out_path, dpi=120)
plt.close(fig)
print(f"\nSaved reliability diagram -> {out_path}")


# %%
# Verdict.
print("\n" + "=" * 78)
print("M6 verdict")
print("-" * 78)
print("  INTEGRATION PATH — validated end-to-end against the real Plaid API:")
print("    - sandbox auth + Item creation")
print("    - custom-user transaction injection")
print("    - /transactions/refresh + /transactions/get pagination")
print("    - Plaid -> internal schema adapter (sign flip, merchant fallback)")
print("    - daily-balance reconstruction from current balance + history")
if dec is not None:
    total = dec.true_affordable + dec.true_not_affordable + dec.false_affordable + dec.false_not_affordable
    print(f"\n  ROUND-TRIP METRICS (thin — see caveat):  N={total} decisions")
    print(f"    calibration error:     {cal.calibration_error:.1%}")
    print(f"    false-affordable rate: {dec.false_affordable_rate:.1%}")
print()
print("  CAVEAT: Plaid's headless sandbox caps history at ~90 days, leaving")
print("  only ~75 days to fit on — too thin for a meaningful calibration")
print("  verdict. These numbers are noisy and are NOT the calibration result.")
print("  The substantive calibration evidence is M5 (80 synthetic households,")
print("  notebooks/05_calibration.py): calibration error 8.2%, FAR 0.6%.")
print()
print("  M6's deliverable is the PLUMBING PROOF above. Full-history validation")
print("  needs the real Link flow (days_requested) or real consented accounts.")
print("=" * 78)
