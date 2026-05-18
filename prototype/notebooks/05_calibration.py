"""M5 — Calibration and decision evaluation across many trajectories.

This is the moment the engine is statistically PROVEN OR SENT BACK.
Single-trajectory eyeball coverage (the M4 demo) is NOT a calibration
test — that's what this notebook actually does.

Run from prototype/:
    uv run python notebooks/05_calibration.py

What this produces:
  - Reliability curve: stated vs empirical coverage for the 50/80/90%
    predictive intervals, pooled across all (profile, seed, day) points.
    A calibrated engine sits on the diagonal.
  - figures/05_reliability.png : reliability diagram (stated vs empirical).
  - Per-profile coverage table.
  - Decision confusion matrix on ~K random affordability questions per run,
    with the headline FALSE-AFFORDABLE rate — the trust-destroying error.

Settings:
  - 4 profiles × 20 seeds = 80 households
  - 30-day horizon, 2000 sims per projection
  - 10 random affordability questions per run = 800 decisions total
"""
# %%
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from eval.backtest import run_full_backtest, score_calibration

pd.set_option("display.width", 140)


# %%
print("Running full backtest — 4 profiles × 20 seeds = 80 households...")
out = run_full_backtest(
    seeds=range(20),
    holdout_days=30,
    n_sims=2000,
    decisions_per_run=10,
    safety_buffer=100.0,
)

cal = out["calibration"]
dec = out["decisions"]
results = out["results"]
decisions_df = out["decisions_df"]

print(
    f"\n  total points (profile × seed × day):  {len(results.actual_values):>5,}"
    f"\n  total affordability decisions:        {len(decisions_df):>5,}"
)


# %%
# Pooled reliability curve — the headline calibration result.
print("\n" + "=" * 78)
print("Pooled reliability curve  (stated coverage vs empirical)")
print("-" * 78)
print(f"  {'stated':>10}   {'empirical':>10}   gap")
for stated, empirical in sorted(cal.coverage_curve.items()):
    gap = empirical - stated
    sign = "+" if gap >= 0 else "-"
    print(f"  {stated:>9.0%}    {empirical:>9.0%}    {sign}{abs(gap):.0%}")
print(f"\n  mean calibration error:  {cal.calibration_error:.1%}")
print("=" * 78)


# %%
# Per-profile coverage breakdown — does any profile lie outside calibration?
print("\nPer-profile coverage breakdown:")
profiles = pd.unique(results.profile_labels)
header = f"  {'profile':<22}"
intervals = [(0.5, 25, 75), (0.8, 10, 90), (0.9, 5, 95)]
for stated, _, _ in intervals:
    header += f"  {f'{stated:.0%} interval':>13}"
print(header)
print("-" * len(header))
for profile in profiles:
    mask = results.profile_labels == profile
    row = f"  {profile:<22}"
    for stated, lo_p, hi_p in intervals:
        lo = results.bands[lo_p][mask]
        hi = results.bands[hi_p][mask]
        actual = results.actual_values[mask]
        emp = float(((actual >= lo) & (actual <= hi)).mean())
        row += f"        {emp:>5.0%}    "
    print(row)


# %%
# Decision evaluation — the headline FALSE-AFFORDABLE rate.
total = dec.true_affordable + dec.true_not_affordable + dec.false_affordable + dec.false_not_affordable
print("\n" + "=" * 78)
print("Affordability decision confusion matrix  (N = {:,})".format(total))
print("-" * 78)
print(f"                              actually-affordable    actually-breached")
print(f"  predicted affordable        {dec.true_affordable:>14,}            {dec.false_affordable:>14,}  <-- FALSE AFFORDABLE")
print(f"  predicted not-affordable    {dec.false_not_affordable:>14,}            {dec.true_not_affordable:>14,}")
print("-" * 78)
print(f"  FALSE-AFFORDABLE RATE (headline):     {dec.false_affordable_rate:>6.1%}")
print(f"  false-not-affordable rate (conservative): "
      f"{dec.false_not_affordable / total if total else 0:.1%}")
print("=" * 78)


# %%
# Per-profile decision breakdown.
print("\nPer-profile decision breakdown:")
print(f"  {'profile':<22}  {'N':>5}  {'FA':>4}  {'TA':>4}  {'FNA':>4}  {'TNA':>4}  {'FAR':>6}")
for profile in profiles:
    sub = decisions_df[decisions_df["profile"] == profile]
    pa = sub["verdict"] == "affordable"
    br = sub["actually_breached"]
    fa = int((pa & br).sum())
    ta = int((pa & ~br).sum())
    fna = int((~pa & ~br).sum())
    tna = int((~pa & br).sum())
    n = len(sub)
    far = fa / n if n else 0.0
    print(f"  {profile:<22}  {n:>5}  {fa:>4}  {ta:>4}  {fna:>4}  {tna:>4}  {far:>5.1%}")


# %%
# Save the reliability diagram.
fig, ax = plt.subplots(figsize=(6, 6))
ax.plot([0, 1], [0, 1], color="gray", lw=1.0, ls="--", label="ideal (calibrated)")
stated_list = sorted(cal.coverage_curve.keys())
empirical_list = [cal.coverage_curve[s] for s in stated_list]
ax.plot(stated_list, empirical_list, "o-", color="C0", lw=2, markersize=10, label="observed")
for s, e in zip(stated_list, empirical_list):
    ax.annotate(f"{e:.0%}", (s, e), textcoords="offset points", xytext=(8, -4), fontsize=9)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_xlabel("Stated coverage")
ax.set_ylabel("Empirical coverage")
ax.set_title("Reliability diagram — calibration of predictive intervals")
ax.legend(loc="upper left", fontsize=9)
ax.grid(alpha=0.3)
fig.tight_layout()
out_path = Path(__file__).resolve().parent.parent / "figures" / "05_reliability.png"
out_path.parent.mkdir(exist_ok=True)
fig.savefig(out_path, dpi=120)
plt.close(fig)
print(f"\nSaved reliability diagram -> {out_path}")


# %%
# Verdict.
#
# Two metrics matter, and they measure different things:
#
#   1. CALIBRATION ERROR — how honest are the per-day predictive intervals?
#      Target: <= 10%. Under-coverage means tails are tighter than reality.
#
#   2. FALSE-AFFORDABLE RATE — the OPERATIONAL metric. How often does the
#      engine tell a user "yes" when reality breached the buffer?
#      Target: <= 5%. This is what protects user trust.
#
# These can diverge — the affordability query uses the MIN-BALANCE
# distribution (heavier-tailed than per-day bands) plus a conservative
# verdict threshold, so it can be robust even when per-day intervals
# are slightly tight. That's the design.
print("\n" + "=" * 78)
ce_ok = cal.calibration_error <= 0.10
far_ok = dec.false_affordable_rate <= 0.05
print("M5 verdict")
print(
    f"  calibration error:        {cal.calibration_error:>5.1%}   "
    f"({'PASS — intervals roughly honest' if ce_ok else 'REVIEW — intervals systematically tight'})"
)
print(
    f"  false-affordable rate:    {dec.false_affordable_rate:>5.1%}   "
    f"({'PASS — operationally safe' if far_ok else 'FAIL — engine misleading users'})"
)
print()
print("  Interpretation:")
if ce_ok and far_ok:
    print("    Engine is statistically calibrated AND operationally safe. Ready.")
elif far_ok and not ce_ok:
    print("    Operationally safe (low false-affordable rate) DESPITE per-day intervals")
    print("    being slightly tight. The decision logic uses min-balance tail risk,")
    print("    which compensates. Engine is shippable; widen Stage-3 amount tails")
    print("    (mixture-of-Normals or Student's t) to improve calibration further.")
elif ce_ok and not far_ok:
    print("    Intervals look calibrated but the decision threshold lets too many")
    print("    false-affordable through. Tighten verdict thresholds.")
else:
    print("    Both metrics off. Investigate Stage 3 amount distribution and the")
    print("    affordability-query decision thresholds.")
print("=" * 78)
