"""Evaluation harness — detection, projection, calibration, decision.

See ../../PREDICTION_MODEL_DESIGN.md section 9.

M5 milestone in full. For now (M2) only the detection layer is implemented;
projection / calibration / decision come once Stages 3 and 4 land.

Detection eval (synthetic only) uses ground-truth labels:
  - per-transaction precision/recall on is_recurring
  - schedule cadence accuracy (right cadence on the right source)
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from cashflow.discretionary import (
    DiscretionaryModel,
    expected_daily_spend,
    fit_discretionary,
    sample_daily_spend,
)
from cashflow.income import forecast_income
from cashflow.projection import can_i_afford, project
from cashflow.recurring import detect_recurring, split_recurring_vs_discretionary
from synth.generator import PRESET_PROFILES, generate


@dataclass
class DetectionScore:
    recurring_precision: float
    recurring_recall: float
    schedule_cadence_accuracy: float    # fraction of schedules whose cadence matches ground truth
    n_predicted_schedules: int
    n_ground_truth_sources: int
    n_true_recurring_txns: int
    n_predicted_recurring_txns: int


@dataclass
class CalibrationScore:
    coverage_curve: dict[float, float]
    calibration_error: float


@dataclass
class DecisionScore:
    true_affordable: int
    true_not_affordable: int
    false_affordable: int
    false_not_affordable: int
    false_affordable_rate: float


# ---------------------------------------------------------------------------
# Detection scoring (M2)
# ---------------------------------------------------------------------------


def _expected_cadence_for_source(name: str, ground_truth: dict) -> str | None:
    """Look up what cadence a ground-truth source should have."""
    for bill in ground_truth.get("recurring", []):
        if bill.name == name:
            return "monthly"   # all synthetic bills are monthly
    for src in ground_truth.get("income", []):
        if src.name == name:
            if src.frequency == "biweekly":
                return "biweekly"
            if src.frequency == "semimonthly":
                return "semimonthly"
            if src.frequency == "monthly":
                return "monthly"
            if src.frequency == "irregular":
                return "irregular"   # signaled — should NOT be in regular schedules
    return None


def score_detection(
    predicted_schedules,
    transactions: pd.DataFrame,
    ground_truth: dict,
) -> DetectionScore:
    """Score Stage 1 recurring detection against synthetic ground truth.

    Stage 1 is responsible for detecting REGULAR recurring schedules. Irregular
    income (e.g., gig) is correctly handed off to Stage 2 — so it must be
    excluded from the "true recurring" set when scoring Stage 1, or correct
    behavior gets penalized as missed recall.

    Args:
        predicted_schedules: list[RecurringSchedule] from detect_recurring().
        transactions:        the synthetic ledger (has is_recurring / source_id).
        ground_truth:        SyntheticDataset.ground_truth dict.
    """
    # Per-transaction precision / recall — over REGULAR recurring sources only.
    predicted_ids: set[str] = set()
    for sched in predicted_schedules:
        predicted_ids.update(sched.member_txn_ids)

    irregular_source_names = {
        s.name for s in ground_truth.get("income", []) if s.frequency == "irregular"
    }
    is_regular_recurring = (
        transactions["is_recurring"]
        & ~transactions["source_id"].isin(irregular_source_names)
    )
    true_recurring_ids = set(transactions.loc[is_regular_recurring, "txn_id"])

    tp = len(predicted_ids & true_recurring_ids)
    fp = len(predicted_ids - true_recurring_ids)
    fn = len(true_recurring_ids - predicted_ids)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0

    # Schedule cadence accuracy: for each predicted schedule, find its
    # primary ground-truth source (majority vote among member txn source_ids)
    # and check whether the recovered cadence matches what that source
    # actually was.
    txn_lookup = transactions.set_index("txn_id")["source_id"]
    correct = 0
    rated = 0
    for sched in predicted_schedules:
        sources = txn_lookup.reindex(sched.member_txn_ids).dropna()
        if sources.empty:
            continue
        primary = sources.mode().iloc[0]
        expected = _expected_cadence_for_source(primary, ground_truth)
        if expected is None or expected == "irregular":
            # If we promoted an irregular source to a regular schedule, that's
            # a cadence error by definition.
            rated += 1
            continue
        rated += 1
        if expected == sched.cadence:
            correct += 1
    cadence_acc = correct / rated if rated else 0.0

    n_truth = len(ground_truth.get("recurring", [])) + sum(
        1 for s in ground_truth.get("income", []) if s.frequency != "irregular"
    )

    return DetectionScore(
        recurring_precision=precision,
        recurring_recall=recall,
        schedule_cadence_accuracy=cadence_acc,
        n_predicted_schedules=len(predicted_schedules),
        n_ground_truth_sources=n_truth,
        n_true_recurring_txns=len(true_recurring_ids),
        n_predicted_recurring_txns=len(predicted_ids),
    )


# ---------------------------------------------------------------------------
# Holdout split (used by projection eval in M5)
# ---------------------------------------------------------------------------


def split_holdout(transactions: pd.DataFrame, holdout_days: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a ledger into (history, holdout) at `holdout_days` from the end."""
    df = transactions.copy()
    df["date"] = pd.to_datetime(df["date"])
    cutoff = df["date"].max() - pd.Timedelta(days=holdout_days)
    history = df[df["date"] <= cutoff].copy()
    holdout = df[df["date"] > cutoff].copy()
    return history, holdout


# ---------------------------------------------------------------------------
# Discretionary scoring (M3)
# ---------------------------------------------------------------------------


@dataclass
class DiscretionaryScore:
    """Backtest score for the discretionary spending model vs naive baseline."""
    dow_mae: float            # MAE of DOW model expected daily spend
    naive_mae: float          # MAE of flat naive baseline
    mae_improvement: float    # (naive - dow) / naive, positive = DOW better
    coverage_50: float        # empirical coverage of the model's 50% interval
    coverage_80: float        # empirical coverage of the model's 80% interval
    coverage_90: float        # empirical coverage of the model's 90% interval
    n_holdout_days: int
    mean_actual_daily: float


def _actual_daily_discretionary(holdout: pd.DataFrame) -> pd.Series:
    """Sum |amount| of TRUE-discretionary debits per day in the holdout window.

    Uses the synthetic ground-truth label `is_recurring` to isolate the
    discretionary slice cleanly — this keeps M3 evaluation independent of
    M2 detection errors. On real data, swap for an inference-based split.
    """
    disc = holdout[(holdout["amount"] < 0) & (~holdout["is_recurring"])].copy()
    disc["date"] = pd.to_datetime(disc["date"])
    daily = disc.groupby(disc["date"].dt.normalize())["amount"].apply(lambda s: s.abs().sum())
    # Reindex to full daily calendar so zero-spend days count
    if not holdout.empty:
        h = holdout.copy()
        h["date"] = pd.to_datetime(h["date"])
        full_idx = pd.date_range(h["date"].min().normalize(), h["date"].max().normalize(), freq="D")
        daily = daily.reindex(full_idx, fill_value=0.0)
    return daily


def score_discretionary(
    model: DiscretionaryModel,
    naive_dollars_per_day: float,
    holdout: pd.DataFrame,
    n_samples: int = 2000,
    seed: int = 0,
) -> DiscretionaryScore:
    """Backtest the discretionary model on a holdout window.

    For each day in the holdout:
      - actual: true discretionary spend that day
      - DOW prediction: expected_daily_spend(model, day)
      - naive prediction: flat constant
    Track MAE for both. Separately, sample `n_samples` daily spends from
    the model per day to build an empirical predictive distribution and
    measure coverage of stated intervals.
    """
    daily_actual = _actual_daily_discretionary(holdout)
    if daily_actual.empty:
        return DiscretionaryScore(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0.0)

    rng = np.random.default_rng(seed)
    dow_preds = np.array([expected_daily_spend(model, d) for d in daily_actual.index])
    actuals = daily_actual.values

    dow_mae = float(np.mean(np.abs(actuals - dow_preds)))
    naive_mae = float(np.mean(np.abs(actuals - naive_dollars_per_day)))
    improvement = (naive_mae - dow_mae) / naive_mae if naive_mae > 0 else 0.0

    # Coverage: empirical predictive distribution per day, check intervals.
    inside_50 = 0
    inside_80 = 0
    inside_90 = 0
    for actual, day in zip(actuals, daily_actual.index):
        samples = np.array([sample_daily_spend(model, day, rng) for _ in range(n_samples)])
        lo50, hi50 = np.percentile(samples, [25, 75])
        lo80, hi80 = np.percentile(samples, [10, 90])
        lo90, hi90 = np.percentile(samples, [5, 95])
        inside_50 += int(lo50 <= actual <= hi50)
        inside_80 += int(lo80 <= actual <= hi80)
        inside_90 += int(lo90 <= actual <= hi90)

    n = len(actuals)
    return DiscretionaryScore(
        dow_mae=dow_mae,
        naive_mae=naive_mae,
        mae_improvement=float(improvement),
        coverage_50=inside_50 / n,
        coverage_80=inside_80 / n,
        coverage_90=inside_90 / n,
        n_holdout_days=n,
        mean_actual_daily=float(actuals.mean()),
    )


# ---------------------------------------------------------------------------
# M5 — Calibration and decision evaluation across many trajectories
# ---------------------------------------------------------------------------


@dataclass
class BacktestResults:
    """Pooled per-day calibration data across many (profile, seed) runs."""
    # Flat 1-D arrays of length N = sum over (profile, seed) of horizon_days.
    actual_values: np.ndarray
    bands: dict[int, np.ndarray]    # percentile -> values
    # Bookkeeping for breakdowns.
    profile_labels: np.ndarray
    seed_labels: np.ndarray


def score_calibration(results: BacktestResults) -> CalibrationScore:
    """Reliability curve from pooled (actual, predicted band) pairs.

    For each stated coverage P, computes empirical coverage = fraction of
    actuals that fell inside the P% predictive interval. A well-calibrated
    engine sits on the diagonal — stated 80% covers ~80%, not 55% (under)
    or 95% (over).
    """
    actuals = results.actual_values
    intervals = [
        (0.50, 25, 75),
        (0.80, 10, 90),
        (0.90, 5, 95),
    ]
    coverage = {}
    for stated, lo_p, hi_p in intervals:
        lo = results.bands[lo_p]
        hi = results.bands[hi_p]
        empirical = float(((actuals >= lo) & (actuals <= hi)).mean())
        coverage[stated] = empirical

    calibration_error = float(np.mean([abs(s - e) for s, e in coverage.items()]))
    return CalibrationScore(coverage_curve=coverage, calibration_error=calibration_error)


def score_decisions(decisions_df: pd.DataFrame) -> DecisionScore:
    """Confusion matrix on affordability decisions.

    Predicted affordable = verdict == "affordable" (the strict positive).
    "tight" and "not affordable" both treated as predicted_not_affordable
    since the product would warn or refuse in both cases.

    The headline metric is the FALSE-AFFORDABLE rate — predicting yes when
    reality breached the buffer. That's the trust-destroying error.
    """
    if decisions_df.empty:
        return DecisionScore(0, 0, 0, 0, 0.0)
    predicted_affordable = decisions_df["verdict"] == "affordable"
    actually_breached = decisions_df["actually_breached"]
    actually_affordable = ~actually_breached

    tp = int((predicted_affordable & actually_affordable).sum())
    tn = int((~predicted_affordable & actually_breached).sum())
    fp = int((predicted_affordable & actually_breached).sum())
    fn = int((~predicted_affordable & actually_affordable).sum())
    total = len(decisions_df)
    far = fp / total if total else 0.0

    return DecisionScore(
        true_affordable=tp,
        true_not_affordable=tn,
        false_affordable=fp,
        false_not_affordable=fn,
        false_affordable_rate=float(far),
    )


def run_full_backtest(
    profiles: list[str] | None = None,
    seeds=range(20),
    holdout_days: int = 30,
    n_sims: int = 2000,
    decisions_per_run: int = 10,
    safety_buffer: float = 100.0,
    decision_amount_range: tuple[float, float] = (50.0, 2500.0),
    n_days: int = 365,
) -> dict:
    """Run the full M5 backtest: many (profile, seed) draws, aggregate scores.

    For each (profile, seed):
      - Generate 365 days of synthetic data.
      - Split last `holdout_days` as the actual future.
      - Fit M2 + M3 on the history.
      - Project (M4) `holdout_days` forward.
      - Record per-day percentile bands vs the actual balance (for
        calibration scoring across all profile/seed/day tuples).
      - Ask `decisions_per_run` random affordability questions; record
        the verdict and whether the actual trajectory (with the
        hypothetical extra debit subtracted from on_date forward) would
        have breached `safety_buffer`.

    Returns dict with CalibrationScore + DecisionScore + raw frames.
    """
    if profiles is None:
        profiles = list(PRESET_PROFILES.keys())

    actuals_per_run: list[np.ndarray] = []
    bands_per_run: dict[int, list[np.ndarray]] = {p: [] for p in (5, 10, 25, 50, 75, 90, 95)}
    profile_labels: list[np.ndarray] = []
    seed_labels: list[np.ndarray] = []
    decision_records: list[dict] = []

    decision_rng = np.random.default_rng(42)

    for profile_name in profiles:
        for seed in seeds:
            ds = generate(
                PRESET_PROFILES[profile_name],
                start="2024-01-01",
                n_days=n_days,
                seed=int(seed),
            )
            history, _ = split_holdout(ds.transactions, holdout_days=holdout_days)
            last_hist_day = history["date"].max()
            current_balance = float(ds.balance_series.loc[last_hist_day])
            horizon_start = last_hist_day + pd.Timedelta(days=1)

            schedules = detect_recurring(history)
            income = forecast_income(schedules, history, horizon_days=holdout_days)
            _, disc_h = split_recurring_vs_discretionary(history, schedules)
            disc_model = fit_discretionary(disc_h)

            proj = project(
                current_balance,
                horizon_start,
                holdout_days,
                schedules,
                income,
                disc_model,
                n_sims=n_sims,
                seed=int(seed),
            )

            actual = ds.balance_series.loc[proj.dates[0] : proj.dates[-1]].values
            # Compute the bands we need for calibration. The Projection
            # already stores {10,25,75,90}; add {5,50,95} from trajectories.
            p5 = np.percentile(proj.trajectories, 5, axis=0)
            p50 = proj.median_path
            p95 = np.percentile(proj.trajectories, 95, axis=0)
            this_bands = {
                5: p5,
                10: proj.percentile_bands[10],
                25: proj.percentile_bands[25],
                50: p50,
                75: proj.percentile_bands[75],
                90: proj.percentile_bands[90],
                95: p95,
            }

            actuals_per_run.append(actual)
            for p in bands_per_run:
                bands_per_run[p].append(this_bands[p])
            profile_labels.append(np.array([profile_name] * len(actual)))
            seed_labels.append(np.array([seed] * len(actual)))

            # Random affordability questions.
            for _ in range(decisions_per_run):
                day_offset = int(decision_rng.integers(0, holdout_days))
                amount = float(decision_rng.uniform(*decision_amount_range))
                on_date = proj.dates[day_offset]
                ans = can_i_afford(proj, amount, on_date, safety_buffer=safety_buffer)

                # Whether the ACTUAL trajectory (with the hypothetical
                # extra debit) would have breached the safety buffer.
                actual_adjusted = actual.copy()
                actual_adjusted[day_offset:] -= amount
                breached = bool((actual_adjusted < safety_buffer).any())

                decision_records.append(
                    {
                        "profile": profile_name,
                        "seed": int(seed),
                        "on_date": on_date,
                        "amount": amount,
                        "verdict": ans.verdict,
                        "prob_breach": ans.prob_below_buffer,
                        "actually_breached": breached,
                    }
                )

    results = BacktestResults(
        actual_values=np.concatenate(actuals_per_run),
        bands={p: np.concatenate(arrs) for p, arrs in bands_per_run.items()},
        profile_labels=np.concatenate(profile_labels),
        seed_labels=np.concatenate(seed_labels),
    )
    decisions_df = pd.DataFrame(decision_records)

    return {
        "calibration": score_calibration(results),
        "decisions": score_decisions(decisions_df),
        "results": results,
        "decisions_df": decisions_df,
    }


# ---------------------------------------------------------------------------
# M6 — Real-data backtest (Plaid sandbox or any pre-normalized dataset)
# ---------------------------------------------------------------------------


def run_real_data_backtest(
    datasets: list[tuple[str, pd.DataFrame, pd.Series]],
    holdout_days: int = 30,
    n_sims: int = 2000,
    decisions_per_run: int = 10,
    safety_buffer: float = 100.0,
    decision_amount_range: tuple[float, float] = (50.0, 2500.0),
) -> dict:
    """Run the M5-style backtest on real (non-synthetic) datasets.

    Args:
        datasets: list of (label, normalized_ledger_df, balance_series).
                  The DataFrame uses the internal schema; the Series is
                  the reconstructed daily end-of-day balance.
        ... rest same as run_full_backtest.

    Detection eval is dropped (no ground-truth `is_recurring` labels on
    real data). Calibration and decision eval still work — they only need
    the actual balance trajectory, which we reconstructed from current
    balance + transaction history.

    Returns the same dict shape as run_full_backtest minus the
    profile/seed-level breakdown labels.
    """
    actuals_per_run: list[np.ndarray] = []
    bands_per_run: dict[int, list[np.ndarray]] = {p: [] for p in (5, 10, 25, 50, 75, 90, 95)}
    label_labels: list[np.ndarray] = []
    decision_records: list[dict] = []
    decision_rng = np.random.default_rng(42)

    for label, ledger, balance_series in datasets:
        if ledger.empty or balance_series.empty:
            continue
        history, _ = split_holdout(ledger, holdout_days=holdout_days)
        if history.empty:
            continue

        last_hist_day = history["date"].max().normalize()
        if last_hist_day not in balance_series.index:
            # Find the closest available index <= last_hist_day
            valid = balance_series.index[balance_series.index <= last_hist_day]
            if len(valid) == 0:
                continue
            last_hist_day = valid.max()
        current_balance = float(balance_series.loc[last_hist_day])
        horizon_start = last_hist_day + pd.Timedelta(days=1)

        schedules = detect_recurring(history)
        income = forecast_income(schedules, history, horizon_days=holdout_days)
        _, disc_h = split_recurring_vs_discretionary(history, schedules)
        disc_model = fit_discretionary(disc_h)

        proj = project(
            current_balance,
            horizon_start,
            holdout_days,
            schedules,
            income,
            disc_model,
            n_sims=n_sims,
            seed=0,
        )

        actual = balance_series.reindex(proj.dates).ffill().bfill().values
        if len(actual) != len(proj.dates) or np.isnan(actual).any():
            # Skip if we cannot align actual to horizon
            continue

        p5 = np.percentile(proj.trajectories, 5, axis=0)
        p95 = np.percentile(proj.trajectories, 95, axis=0)
        this_bands = {
            5: p5,
            10: proj.percentile_bands[10],
            25: proj.percentile_bands[25],
            50: proj.median_path,
            75: proj.percentile_bands[75],
            90: proj.percentile_bands[90],
            95: p95,
        }

        actuals_per_run.append(actual)
        for p in bands_per_run:
            bands_per_run[p].append(this_bands[p])
        label_labels.append(np.array([label] * len(actual)))

        for _ in range(decisions_per_run):
            day_offset = int(decision_rng.integers(0, holdout_days))
            amount = float(decision_rng.uniform(*decision_amount_range))
            on_date = proj.dates[day_offset]
            ans = can_i_afford(proj, amount, on_date, safety_buffer=safety_buffer)
            actual_adjusted = actual.copy()
            actual_adjusted[day_offset:] -= amount
            breached = bool((actual_adjusted < safety_buffer).any())
            decision_records.append(
                {
                    "label": label,
                    "on_date": on_date,
                    "amount": amount,
                    "verdict": ans.verdict,
                    "prob_breach": ans.prob_below_buffer,
                    "actually_breached": breached,
                }
            )

    if not actuals_per_run:
        raise RuntimeError(
            "No usable datasets — check that ledgers and balance series have "
            "enough history (need >= holdout_days + a few weeks for fitting)."
        )

    results = BacktestResults(
        actual_values=np.concatenate(actuals_per_run),
        bands={p: np.concatenate(arrs) for p, arrs in bands_per_run.items()},
        profile_labels=np.concatenate(label_labels),
        seed_labels=np.zeros(sum(len(a) for a in actuals_per_run), dtype=int),
    )
    decisions_df = pd.DataFrame(decision_records)

    return {
        "calibration": score_calibration(results),
        "decisions": score_decisions(decisions_df) if not decisions_df.empty else None,
        "results": results,
        "decisions_df": decisions_df,
    }
