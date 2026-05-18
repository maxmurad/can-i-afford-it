"""Stage 2 — Income timing.

See ../../PREDICTION_MODEL_DESIGN.md section 4.

Why income gets its own stage: the entire projection pivots on WHEN money
arrives. A wrong income date cascades into every downstream answer.

What Stage 2 does:
  1. Take the Stage-1 schedules flagged is_income=True — these are the
     regular paychecks (biweekly / semimonthly / monthly).
  2. Separately scan credit transactions for streams Stage 1 didn't catch
     because their gaps are too irregular — that's irregular income (gig,
     commission, variable hours).
  3. Classify overall regime as 'regular' or 'irregular' and project
     income events over the forecast horizon, each with date AND amount
     uncertainty (NOT a point estimate).

Irregular income deliberately gets wide date_std_days values — the
downstream projection bands should be wider, which is correct.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from cashflow.recurring import (
    CADENCE_PERIOD,
    MIN_GROUP_SIZE,
    RecurringSchedule,
    normalize_merchant,
)

# Threshold for calling an unscheduled credit stream "irregular income".
# Real income streams should have many events across the year.
MIN_IRREGULAR_INCOME_EVENTS = 8


@dataclass
class IncomeEvent:
    """A single projected future income event, with uncertainty."""
    expected_date: pd.Timestamp
    date_std_days: float
    amount_mean: float
    amount_std: float
    source: str          # merchant_key (or label) this event comes from
    regime: str          # "regular" | "irregular"


@dataclass
class IncomeForecast:
    """The projected income schedule over the forecast horizon."""
    regime: str
    events: list[IncomeEvent]
    confidence: float


def _project_regular(sched: RecurringSchedule, end_date: pd.Timestamp) -> list[IncomeEvent]:
    """Walk a regular cadence forward to end_date."""
    events: list[IncomeEvent] = []
    period = CADENCE_PERIOD[sched.cadence]
    d = sched.next_date
    while d <= end_date:
        events.append(
            IncomeEvent(
                expected_date=d,
                date_std_days=1.0,            # regular paychecks are punctual
                amount_mean=sched.amount_mean,
                amount_std=sched.amount_std,
                source=sched.merchant_key,
                regime="regular",
            )
        )
        d = d + pd.Timedelta(days=period)
    return events


def _detect_irregular_streams(
    transactions: pd.DataFrame,
    already_scheduled_txn_ids: set[str],
) -> list[tuple[str, pd.Series, list[int]]]:
    """Find credit-side merchant groups Stage 1 missed that look like income.

    Returns list of (merchant_key, amounts_series, gaps_list).
    """
    df = transactions.copy()
    df["date"] = pd.to_datetime(df["date"])
    credits = df[(df["amount"] > 0) & (~df["txn_id"].isin(already_scheduled_txn_ids))]
    if credits.empty:
        return []
    credits = credits.assign(merchant_key=credits["merchant"].map(normalize_merchant))

    streams: list[tuple[str, pd.Series, list[int]]] = []
    for key, grp in credits.groupby("merchant_key", sort=False):
        if len(grp) < MIN_IRREGULAR_INCOME_EVENTS:
            continue
        grp_sorted = grp.sort_values("date")
        dates = grp_sorted["date"].tolist()
        gaps = [int((dates[i + 1] - dates[i]).days) for i in range(len(dates) - 1)]
        streams.append((str(key), grp_sorted["amount"], gaps))
    return streams


def _project_irregular(
    merchant_key: str,
    amounts: pd.Series,
    gaps: list[int],
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> list[IncomeEvent]:
    """Project irregular income forward by sampling mean gap / amount.

    Wide date uncertainty is the point — downstream bands should reflect it.
    """
    if not gaps:
        return []
    mean_gap = sum(gaps) / len(gaps)
    # gap uncertainty: roughly std of historical gaps
    if len(gaps) > 1:
        gap_std = (sum((g - mean_gap) ** 2 for g in gaps) / (len(gaps) - 1)) ** 0.5
    else:
        gap_std = mean_gap / 2

    amt_mean = float(amounts.mean())
    amt_std = float(amounts.std(ddof=0))

    events: list[IncomeEvent] = []
    d = start_date + pd.Timedelta(days=mean_gap)
    while d <= end_date:
        events.append(
            IncomeEvent(
                expected_date=d,
                date_std_days=float(max(gap_std, 1.0)),
                amount_mean=amt_mean,
                amount_std=amt_std,
                source=merchant_key,
                regime="irregular",
            )
        )
        d = d + pd.Timedelta(days=mean_gap)
    return events


def forecast_income(
    schedules: list[RecurringSchedule],
    transactions: pd.DataFrame,
    horizon_days: int,
    as_of: pd.Timestamp | None = None,
) -> IncomeForecast:
    """Project income events over the next `horizon_days`.

    Args:
        schedules:    all Stage-1 schedules (function picks is_income=True).
        transactions: full ledger — used to find irregular streams Stage 1 missed.
        horizon_days: forecast length in days.
        as_of:        anchor date for "now" (defaults to last txn date).

    Returns:
        IncomeForecast with chronologically-ordered events.
    """
    if as_of is None:
        as_of = pd.Timestamp(pd.to_datetime(transactions["date"]).max())
    end_date = as_of + pd.Timedelta(days=horizon_days)

    income_schedules = [s for s in schedules if s.is_income]
    events: list[IncomeEvent] = []

    # Regular income from Stage 1.
    for sched in income_schedules:
        events.extend(_project_regular(sched, end_date))

    # Irregular income from anything Stage 1 didn't pick up.
    scheduled_ids: set[str] = set()
    for s in income_schedules:
        scheduled_ids.update(s.member_txn_ids)

    for merchant_key, amounts, gaps in _detect_irregular_streams(transactions, scheduled_ids):
        events.extend(
            _project_irregular(merchant_key, amounts, gaps, as_of, end_date)
        )

    events.sort(key=lambda e: e.expected_date)

    # Overall regime: regular if any regular events exist; else irregular if any.
    if any(e.regime == "regular" for e in events):
        regime = "regular"
    elif events:
        regime = "irregular"
    else:
        regime = "unknown"

    # Confidence: average of Stage-1 confidence for regular schedules;
    # for irregular, a modest fixed value (the band-widening already
    # encodes our uncertainty downstream).
    if income_schedules:
        confidence = sum(s.confidence for s in income_schedules) / len(income_schedules)
    elif regime == "irregular":
        confidence = 0.5
    else:
        confidence = 0.0

    return IncomeForecast(regime=regime, events=events, confidence=float(confidence))
