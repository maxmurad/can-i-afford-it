"""M2 — Recurring & income detection, scored against synthetic ground truth.

Run from prototype/:
    uv run python notebooks/02_recurring_detection.py

What this proves:
  - Stage 1 (recurring.py) correctly recovers schedules with high precision
    and recall against ground-truth labels.
  - Recovered cadences match the ground-truth cadences.
  - The irregular_gig income stream is correctly NOT promoted to a regular
    schedule but IS picked up as irregular by Stage 2.
"""
# %%
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from cashflow.income import forecast_income
from cashflow.recurring import detect_recurring, split_recurring_vs_discretionary
from eval.backtest import score_detection
from synth.generator import PRESET_PROFILES, generate

pd.set_option("display.width", 140)
pd.set_option("display.max_columns", 12)


# %%
# Run detection on every preset and score against ground truth.
print("=" * 92)
print(f"{'profile':<22} {'preds':>6} {'truth':>6}   {'precision':>10} {'recall':>8} {'cadence':>9}")
print("-" * 92)

results = {}
for key, profile in PRESET_PROFILES.items():
    ds = generate(profile, start="2024-01-01", n_days=365, seed=0)
    schedules = detect_recurring(ds.transactions)
    score = score_detection(schedules, ds.transactions, ds.ground_truth)
    results[key] = (ds, schedules, score)
    print(
        f"{key:<22} "
        f"{score.n_predicted_schedules:>6} "
        f"{score.n_ground_truth_sources:>6}   "
        f"{score.recurring_precision:>9.2%} "
        f"{score.recurring_recall:>7.2%} "
        f"{score.schedule_cadence_accuracy:>8.2%}"
    )
print("=" * 92)


# %%
# Drill into the primary persona — what schedules did we recover?
ds, schedules, score = results["steady_biweekly"]
print(f"\n[steady_biweekly] {len(schedules)} schedules detected:\n")
print(f"{'merchant_key':<30} {'cadence':<14} {'amount':>10} {'next_date':<12} {'conf':>5} {'income':>7}")
print("-" * 92)
for s in schedules:
    print(
        f"{s.merchant_key[:30]:<30} "
        f"{s.cadence:<14} "
        f"${s.amount_mean:>9,.2f} "
        f"{s.next_date.date().isoformat():<12} "
        f"{s.confidence:>5.0%} "
        f"{str(s.is_income):>7}"
    )


# %%
# The irregular_gig profile is the hard case — Stage 1 should REJECT the
# gig income (irregular gaps), and Stage 2 should pick it up as irregular.
ds, schedules, score = results["irregular_gig"]
gig_in_schedules = any(
    "uber" in s.merchant_key for s in schedules if s.is_income
)
print(f"\n[irregular_gig] gig income promoted to a REGULAR schedule? {gig_in_schedules}")
print(f"                (should be False — gig income is irregular by construction)")

# Now ask Stage 2 to forecast — it should produce irregular events.
income_forecast = forecast_income(schedules, ds.transactions, horizon_days=30)
print(f"\nIncome forecast for next 30 days:")
print(f"  regime:     {income_forecast.regime}")
print(f"  confidence: {income_forecast.confidence:.0%}")
print(f"  events:     {len(income_forecast.events)}")
for e in income_forecast.events[:6]:
    print(
        f"    {e.expected_date.date()}  ±{e.date_std_days:>3.0f}d  "
        f"${e.amount_mean:>6,.0f} ±${e.amount_std:>4,.0f}  ({e.regime})"
    )
if len(income_forecast.events) > 6:
    print(f"    ... +{len(income_forecast.events) - 6} more")


# %%
# Compare to steady_biweekly — should be regular, tight bands.
ds, schedules, _ = results["steady_biweekly"]
fc = forecast_income(schedules, ds.transactions, horizon_days=30)
print(f"\n[steady_biweekly] regime={fc.regime} confidence={fc.confidence:.0%} events={len(fc.events)}")
for e in fc.events:
    print(
        f"    {e.expected_date.date()}  ±{e.date_std_days:>3.0f}d  "
        f"${e.amount_mean:>6,.0f} ±${e.amount_std:>4,.0f}  ({e.regime})"
    )


# %%
# Stress test: many small monthly subs, all named differently. Look at what
# every schedule covers (debits + credits together) vs the ground-truth set.
ds, schedules, score = results["subscription_heavy"]
predicted_ids = set()
for s in schedules:
    predicted_ids.update(s.member_txn_ids)
true_recurring_ids = set(ds.transactions.loc[ds.transactions["is_recurring"], "txn_id"])
missed = true_recurring_ids - predicted_ids
extras = predicted_ids - true_recurring_ids

print(f"\n[subscription_heavy] detection breakdown (debits + credits together):")
print(f"  true recurring txns:        {len(true_recurring_ids)}")
print(f"  predicted recurring txns:   {len(predicted_ids)}")
print(f"  missed (false negatives):   {len(missed)}")
print(f"  spurious (false positives): {len(extras)}")
if missed:
    miss_df = ds.transactions[ds.transactions["txn_id"].isin(missed)]
    print(f"\n  missed transactions by source:")
    print(miss_df.groupby("source_id")["txn_id"].count().sort_values(ascending=False).head(8))


# %%
# Overall verdict.
print("\n" + "=" * 92)
all_good = True
for key, (ds, schedules, score) in results.items():
    p_ok = score.recurring_precision >= 0.90
    r_ok = score.recurring_recall >= 0.85
    c_ok = score.schedule_cadence_accuracy >= 0.85
    status = "PASS" if (p_ok and r_ok and c_ok) else "REVIEW"
    if status != "PASS":
        all_good = False
    print(f"  {key:<22} {status}   (p={score.recurring_precision:.2%}, r={score.recurring_recall:.2%}, cadence={score.schedule_cadence_accuracy:.2%})")
print("=" * 92)
print(f"\nM2 detection: {'all profiles PASS — ready for M3' if all_good else 'review weak profiles before proceeding'}")
