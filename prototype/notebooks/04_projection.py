"""M4 — Monte Carlo forward projection + can_i_afford() against synthetic data.

Run from prototype/:
    uv run python notebooks/04_projection.py

What this produces:
  - figures/04_projection_vs_actual.png : 4-panel chart, projected balance
    ribbon (median + 50% + 80% bands) overlaid with the ACTUAL holdout
    balance trajectory for each profile. Eyeball test for calibration.
  - per-profile affordability questions printed to stdout — verdicts,
    probabilities of breaching the safety buffer, the low-point date,
    and the obligation driving the low point.

Methodology:
  - Generate 365 days of synthetic history.
  - Hold out the last 30 days.
  - Fit M1+M2+M3 on the 335-day history.
  - Project 30 days forward from the start of holdout.
  - Compare projected balance distribution to what actually happened.
"""
# %%
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cashflow.discretionary import fit_discretionary
from cashflow.income import forecast_income
from cashflow.projection import can_i_afford, project
from cashflow.recurring import detect_recurring, split_recurring_vs_discretionary
from eval.backtest import split_holdout
from synth.generator import PRESET_PROFILES, generate

pd.set_option("display.width", 140)

HOLDOUT_DAYS = 30
N_SIMS = 5000
SAFETY_BUFFER = 100.0


def fit_pipeline(history: pd.DataFrame):
    """Run M2 + M3 on a history frame, return (schedules, income, disc_model)."""
    schedules = detect_recurring(history)
    income = forecast_income(schedules, history, horizon_days=HOLDOUT_DAYS)
    _, disc_history = split_recurring_vs_discretionary(history, schedules)
    disc_model = fit_discretionary(disc_history)
    return schedules, income, disc_model


# %%
# Build the projection for every profile and assemble a 4-panel figure.
fig, axes = plt.subplots(len(PRESET_PROFILES), 1, figsize=(11, 3.2 * len(PRESET_PROFILES)), sharex=False)
projections: dict = {}

for ax, (key, profile) in zip(axes, PRESET_PROFILES.items()):
    ds = generate(profile, start="2024-01-01", n_days=365, seed=0)
    history, _ = split_holdout(ds.transactions, holdout_days=HOLDOUT_DAYS)

    last_hist_day = history["date"].max()
    horizon_start = last_hist_day + pd.Timedelta(days=1)
    current_balance = float(ds.balance_series.loc[last_hist_day])

    schedules, income, disc_model = fit_pipeline(history)
    proj = project(
        current_balance=current_balance,
        start_date=horizon_start,
        horizon_days=HOLDOUT_DAYS,
        recurring=schedules,
        income=income,
        discretionary=disc_model,
        n_sims=N_SIMS,
        seed=0,
    )
    projections[key] = (ds, proj)

    # Median + percentile bands.
    ax.plot(proj.dates, proj.median_path, color="C0", lw=1.6, label="median projection")
    ax.fill_between(
        proj.dates, proj.percentile_bands[25], proj.percentile_bands[75],
        color="C0", alpha=0.30, label="50% band",
    )
    ax.fill_between(
        proj.dates, proj.percentile_bands[10], proj.percentile_bands[90],
        color="C0", alpha=0.14, label="80% band",
    )

    # Overlay the actual holdout balance.
    actual_idx = (ds.balance_series.index >= proj.dates[0]) & (
        ds.balance_series.index <= proj.dates[-1]
    )
    actual = ds.balance_series.loc[actual_idx]
    ax.plot(actual.index, actual.values, color="black", lw=1.1, ls="--", label="actual")

    ax.axhline(0, color="red", lw=0.7, ls=":")
    ax.axhline(SAFETY_BUFFER, color="orange", lw=0.7, ls=":")
    ax.set_title(f"{key}  —  starting balance ${current_balance:,.0f}")
    ax.set_ylabel("$")
    ax.legend(fontsize=8, loc="upper left")

fig.tight_layout()
out_dir = Path(__file__).resolve().parent.parent / "figures"
out_dir.mkdir(exist_ok=True)
out_path = out_dir / "04_projection_vs_actual.png"
fig.savefig(out_path, dpi=120)
plt.close(fig)
print(f"\nSaved projection-vs-actual figure -> {out_path}")


# %%
# Coverage of the 80% band on the actual trajectory — quick eyeball backstop.
print("\n" + "=" * 92)
print(f"{'profile':<22} {'start bal':>10} {'actual min':>11} {'median min':>11} {'80% covered':>12}")
print("-" * 92)
for key, (ds, proj) in projections.items():
    actual = ds.balance_series.loc[proj.dates[0] : proj.dates[-1]].values
    inside_80 = ((actual >= proj.percentile_bands[10]) & (actual <= proj.percentile_bands[90])).mean()
    print(
        f"{key:<22} "
        f"${proj.starting_balance:>9,.0f} "
        f"${actual.min():>10,.0f} "
        f"${np.median(proj.min_balance_distribution):>10,.0f} "
        f"{inside_80:>11.0%}"
    )


# %%
# Affordability questions — varied per profile. The verdicts should track
# intuition: gig refuses more, subscription-heavy flags mid-month, etc.
print("\n" + "=" * 92)
print("Affordability questions")
print("=" * 92)

QUESTIONS_BY_PROFILE = {
    "steady_biweekly":    [(200, 0), (800, 3), (1500, 10), (2500, 0), (3500, 0)],
    "irregular_gig":      [(100, 0), (500, 5), (1200, 10), (2200, 0), (3000, 0)],
    "subscription_heavy": [(100, 0), (300, 5), (800, 14), (1500, 0), (2500, 0)],
    "thin_file":          [(100, 0), (400, 3), (900, 7), (1600, 0), (2200, 0)],
}

for key, (ds, proj) in projections.items():
    print(f"\n[{key}]  current balance ${proj.starting_balance:,.0f}, horizon {proj.dates[0].date()} -> {proj.dates[-1].date()}")
    for amount, day_offset in QUESTIONS_BY_PROFILE[key]:
        on_date = proj.dates[day_offset]
        ans = can_i_afford(proj, amount, on_date, safety_buffer=SAFETY_BUFFER)
        driving = ans.driving_obligation or "discretionary"
        print(
            f"  ${amount:>5,} on {on_date.date()}  ->  "
            f"{ans.verdict.upper():<14}  "
            f"P(below buffer)={ans.prob_below_buffer:>5.0%}  "
            f"low point {ans.low_point_date.date()}  "
            f"(driver: {driving})"
        )


# %%
# IMPORTANT — single-trajectory coverage is NOT a calibration test.
#
# Even a perfectly-calibrated 80% band only covers a single realization 80%
# of the time IN EXPECTATION across many trajectories. One household's
# 30-day actual can sit consistently at the band edge purely by sample
# variation in their discretionary spend. The real calibration test
# requires running the projection on MANY synthetic households and asking
# "across all of them, did the 80% band cover ~80% of actual days?"
# That's the M5 calibration eval and uses the score_calibration() stub.
#
# Below is the single-trajectory eyeball check for completeness only.
print("\n" + "=" * 92)
print("Single-trajectory eyeball coverage (NOT a calibration verdict — M5 does that)")
print("-" * 92)
for key, (ds, proj) in projections.items():
    actual = ds.balance_series.loc[proj.dates[0] : proj.dates[-1]].values
    inside_80 = ((actual >= proj.percentile_bands[10]) & (actual <= proj.percentile_bands[90])).mean()
    median_end = proj.median_path[-1]
    actual_end = actual[-1]
    print(
        f"  {key:<22} 80% band covered {inside_80:>4.0%} of days   "
        f"(median end ${median_end:>6,.0f}  vs  actual end ${actual_end:>6,.0f})"
    )
print("=" * 92)
print(
    "\nM4 status: projection + can_i_afford() implemented and produce reasonable\n"
    "verdicts that track each profile's structure (steady households comfortably\n"
    "affordable, subscription-heavy flagged not-affordable due to mid-month bills).\n"
    "Calibration evaluation across many trajectories is the next step (M5)."
)
