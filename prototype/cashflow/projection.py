"""Stage 4 — Forward projection and the affordability query.

See ../../PREDICTION_MODEL_DESIGN.md section 6.

The projection is built by MONTE CARLO SIMULATION, not point estimation.

A single simulated trajectory:
  - starts from the current known balance
  - walks forward day by day to the horizon
  - on each day: applies recurring obligations due (sampling amount),
    applies income expected that day (sampling date + amount),
    and applies one sampled day of discretionary spend

Run thousands of trajectories -> a distribution of balance paths. Report
percentile bands AND — critically — the distribution of the MINIMUM
balance reached over the horizon.

The affordability query injects an extra debit into every trajectory and
reports P(min balance < safety buffer). A wrong "affordable" is the
expensive error — decision thresholds are deliberately conservative.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from cashflow.discretionary import DiscretionaryModel, sample_daily_spend_batch
from cashflow.income import IncomeForecast
from cashflow.recurring import CADENCE_PERIOD, RecurringSchedule


@dataclass
class Projection:
    """The Monte Carlo projection output."""
    dates: pd.DatetimeIndex
    trajectories: np.ndarray            # (n_sims, n_days) end-of-day balances
    median_path: np.ndarray             # (n_days,) median balance per day
    percentile_bands: dict[int, np.ndarray]  # {10, 25, 75, 90} -> (n_days,)
    min_balance_distribution: np.ndarray     # (n_sims,) min over horizon per sim
    starting_balance: float
    recurring: list[RecurringSchedule]
    income: IncomeForecast


@dataclass
class AffordabilityAnswer:
    """Result of one 'can I afford $X on date D?' question."""
    amount: float
    on_date: pd.Timestamp
    prob_below_buffer: float
    expected_min_balance: float
    low_point_date: pd.Timestamp
    driving_obligation: str | None
    verdict: str                        # "affordable" | "tight" | "not affordable"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _schedule_occurrences_in_window(
    sched: RecurringSchedule,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> list[pd.Timestamp]:
    """Every projected occurrence date of `sched` within [start, end]."""
    period = CADENCE_PERIOD[sched.cadence]
    out: list[pd.Timestamp] = []
    d = sched.next_date
    # Defensive: if next_date is far past end_date, bail quickly.
    if d > end_date + pd.Timedelta(days=period):
        return out
    while d <= end_date:
        if d >= start_date:
            out.append(d)
        d = d + pd.Timedelta(days=period)
    return out


# ---------------------------------------------------------------------------
# Projection
# ---------------------------------------------------------------------------


def project(
    current_balance: float,
    start_date,
    horizon_days: int,
    recurring: list[RecurringSchedule],
    income: IncomeForecast,
    discretionary: DiscretionaryModel,
    n_sims: int = 5000,
    seed: int = 0,
) -> Projection:
    """Run the Monte Carlo forward projection of the household's balance."""
    rng = np.random.default_rng(seed)
    start_date = pd.Timestamp(start_date)
    end_date = start_date + pd.Timedelta(days=horizon_days - 1)
    dates = pd.date_range(start_date, end_date, freq="D")
    n_days = len(dates)

    # Per-day signed delta array: (n_sims, n_days).
    deltas = np.zeros((n_sims, n_days), dtype=np.float64)

    # Recurring DEBITS — fixed schedule date, amount sampled per sim.
    for sched in recurring:
        if sched.is_income:
            continue
        for d in _schedule_occurrences_in_window(sched, start_date, end_date):
            day_idx = (d - start_date).days
            amounts = rng.normal(sched.amount_mean, sched.amount_std, size=n_sims)
            np.clip(amounts, 1.0, None, out=amounts)
            deltas[:, day_idx] -= amounts

    # INCOME — regular events land on a fixed day; irregular events have
    # per-sim date jitter modeled with np.add.at (handles repeated indices).
    for event in income.events:
        base_idx = (event.expected_date - start_date).days
        if base_idx >= n_days + 7:
            continue  # well past the horizon
        amounts = rng.normal(event.amount_mean, event.amount_std, size=n_sims)
        np.clip(amounts, 1.0, None, out=amounts)

        if event.regime == "regular":
            if 0 <= base_idx < n_days:
                deltas[:, base_idx] += amounts
            # Drop events before/after horizon silently.
        else:
            # Irregular — sample date jitter per sim, clip into window.
            jitter = rng.normal(0, event.date_std_days, size=n_sims).round().astype(int)
            target_idx = np.clip(base_idx + jitter, 0, n_days - 1)
            np.add.at(deltas, (np.arange(n_sims), target_idx), amounts)

    # DISCRETIONARY — one batched sample per day, subtracted from deltas.
    for d_idx, day in enumerate(dates):
        daily = sample_daily_spend_batch(discretionary, day, rng, n_sims)
        deltas[:, d_idx] -= daily

    # Compose running balance: starting + cumulative daily delta.
    trajectories = current_balance + deltas.cumsum(axis=1)

    return Projection(
        dates=dates,
        trajectories=trajectories,
        median_path=np.median(trajectories, axis=0),
        percentile_bands={
            10: np.percentile(trajectories, 10, axis=0),
            25: np.percentile(trajectories, 25, axis=0),
            75: np.percentile(trajectories, 75, axis=0),
            90: np.percentile(trajectories, 90, axis=0),
        },
        min_balance_distribution=trajectories.min(axis=1),
        starting_balance=float(current_balance),
        recurring=recurring,
        income=income,
    )


# ---------------------------------------------------------------------------
# Affordability query
# ---------------------------------------------------------------------------


def can_i_afford(
    projection: Projection,
    amount: float,
    on_date,
    safety_buffer: float = 100.0,
    tight_threshold: float = 0.15,
    refuse_threshold: float = 0.40,
) -> AffordabilityAnswer:
    """Answer 'can I afford $X on date D?' against the projection.

    Injects an extra debit of `amount` on `on_date` into every trajectory
    (by subtracting `amount` from every day at or after `on_date`) and
    computes the probability the minimum balance over the horizon dips
    below `safety_buffer`.

    Verdict thresholds default to deliberately conservative values —
    a wrong "affordable" is the expensive error.
    """
    on_date = pd.Timestamp(on_date).normalize()
    start = pd.Timestamp(projection.dates[0]).normalize()
    end = pd.Timestamp(projection.dates[-1]).normalize()
    if on_date < start or on_date > end:
        raise ValueError(
            f"on_date {on_date.date()} is outside the projection window "
            f"[{start.date()}, {end.date()}]"
        )
    day_idx = (on_date - start).days

    adjusted = projection.trajectories.copy()
    adjusted[:, day_idx:] -= amount

    min_per_sim = adjusted.min(axis=1)
    prob_breach = float((min_per_sim < safety_buffer).mean())

    if prob_breach < tight_threshold:
        verdict = "affordable"
    elif prob_breach < refuse_threshold:
        verdict = "tight"
    else:
        verdict = "not affordable"

    # Best estimate of when the low point lands (median across sims).
    typical_low_idx = int(np.median(adjusted.argmin(axis=1)))
    low_point_date = pd.Timestamp(projection.dates[typical_low_idx])

    # Best-effort driving obligation: largest recurring debit in
    # [low_point - 3 days, low_point]. Likely the bill blamed in the UX.
    driving = None
    biggest_amount = 0.0
    window_start = low_point_date - pd.Timedelta(days=3)
    for sched in projection.recurring:
        if sched.is_income:
            continue
        for d in _schedule_occurrences_in_window(sched, start, end):
            if window_start <= d <= low_point_date and sched.amount_mean > biggest_amount:
                biggest_amount = sched.amount_mean
                driving = sched.merchant_key

    return AffordabilityAnswer(
        amount=float(amount),
        on_date=on_date,
        prob_below_buffer=prob_breach,
        expected_min_balance=float(min_per_sim.mean()),
        low_point_date=low_point_date,
        driving_obligation=driving,
        verdict=verdict,
    )
