"""Stage 1 — Recurring and bill detection.

See ../../PREDICTION_MODEL_DESIGN.md section 3.

Approach:
  1. Normalize merchant strings to a stable stem (strip trailing
     store-numbers / txn-ids).
  2. Group by (normalized merchant, debit/credit sign). Never mix signs.
  3. Within each group, test inter-transaction gaps against known cadences
     (weekly / biweekly / semimonthly / monthly / quarterly / annual).
  4. Promote groups with strong cadence fit to RecurringSchedule objects.
     Amount learned as a distribution (mean/std) — utility bills vary.

The cadence test:
  - Compute median inter-transaction gap.
  - Pick the candidate cadence whose period is closest to that median.
  - Disambiguate biweekly vs semimonthly (both have ~14-day medians) using
    the standard deviation of gaps: biweekly is tight, semimonthly alternates.
  - Confirm by requiring >= MIN_CADENCE_FIT fraction of gaps within tolerance.
"""
from __future__ import annotations

import re
import statistics
from dataclasses import dataclass

import pandas as pd


# Cadence catalog: (name, period_days, tolerance_days)
CADENCES: list[tuple[str, int, int]] = [
    ("weekly", 7, 2),
    ("biweekly", 14, 2),
    ("semimonthly", 15, 3),
    ("monthly", 30, 4),
    ("quarterly", 91, 14),
    ("annual", 365, 30),
]

# Mapping for projection forward (income.py reads this too).
CADENCE_PERIOD: dict[str, int] = {name: period for name, period, _ in CADENCES}

# Detection thresholds — tuned against synthetic ground truth.
MIN_GROUP_SIZE = 3              # need >= 3 events to even consider recurring
MIN_CADENCE_FIT = 0.6           # >= 60% of gaps must fall within tolerance


@dataclass
class RecurringSchedule:
    """A detected recurring obligation (the model's recovered estimate)."""
    merchant_key: str
    cadence: str
    amount_mean: float
    amount_std: float
    next_date: pd.Timestamp
    confidence: float
    member_txn_ids: list[str]
    is_income: bool = False


# ---------------------------------------------------------------------------
# Merchant normalization
# ---------------------------------------------------------------------------

# trailing or embedded "#1234" tags
_HASH_TAG = re.compile(r"#\s*\d+")
# long numeric runs (transaction ids, 6+ digits)
_LONG_DIGITS = re.compile(r"\b\d{6,}\b")
# whitespace collapse
_WS = re.compile(r"\s+")


def normalize_merchant(raw: str) -> str:
    """Collapse a raw merchant string to a stable matching key.

    Strips trailing store-number tags ('#4471') and long numeric txn-ids,
    lowercases, collapses whitespace.

    >>> normalize_merchant("NETFLIX.COM #4471")
    'netflix.com'
    >>> normalize_merchant("AMAZON  87654321")
    'amazon'
    >>> normalize_merchant("Whole Foods #12")
    'whole foods'
    """
    s = (raw or "").lower()
    s = _HASH_TAG.sub("", s)
    s = _LONG_DIGITS.sub("", s)
    s = _WS.sub(" ", s)
    return s.strip()


# ---------------------------------------------------------------------------
# Cadence classification
# ---------------------------------------------------------------------------


def _classify_cadence(gaps: list[int]) -> tuple[str, int, float] | None:
    """Pick the best cadence for an ordered list of inter-event gaps in days.

    Returns (cadence_name, period_days, confidence) or None if no cadence
    fits well enough. Confidence is the fraction of gaps within tolerance.
    """
    if len(gaps) < MIN_GROUP_SIZE - 1:  # gaps = len(events) - 1
        return None

    median = statistics.median(gaps)

    # Closest cadence to the median.
    name, period, tol = min(CADENCES, key=lambda c: abs(c[1] - median))

    # Biweekly vs semimonthly look identical on median alone — both ~14-15.
    # Disambiguate with gap standard deviation: biweekly is tight, semimonthly
    # alternates 13/15 and crosses month boundaries (Feb shortens, etc.).
    if 13 <= median <= 16 and len(gaps) >= 3:
        stdev = statistics.pstdev(gaps)
        if stdev < 1.2:
            name, period, tol = "biweekly", 14, 2
        else:
            name, period, tol = "semimonthly", 15, 3

    fit = sum(1 for g in gaps if abs(g - period) <= tol) / len(gaps)
    if fit < MIN_CADENCE_FIT:
        return None
    return name, period, fit


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def detect_recurring(transactions: pd.DataFrame) -> list[RecurringSchedule]:
    """Detect all recurring schedules in a transaction ledger.

    Args:
        transactions: ledger with columns at least
            [txn_id, date, amount, merchant]  (amount: -debit, +credit)

    Returns:
        list of RecurringSchedule, sorted by next_date. Credit streams
        come back with is_income=True for Stage 2 to consume.
    """
    if transactions.empty:
        return []

    df = transactions.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["merchant_key"] = df["merchant"].map(normalize_merchant)
    df["sign"] = df["amount"].apply(lambda x: 1 if x > 0 else -1)

    schedules: list[RecurringSchedule] = []

    # Group by (merchant_key, sign) so credits and debits never mix.
    for (key, sign), grp in df.groupby(["merchant_key", "sign"], sort=False):
        if len(grp) < MIN_GROUP_SIZE:
            continue
        grp_sorted = grp.sort_values("date")
        dates = grp_sorted["date"].tolist()
        gaps = [int((dates[i + 1] - dates[i]).days) for i in range(len(dates) - 1)]

        cadence_info = _classify_cadence(gaps)
        if cadence_info is None:
            continue
        cadence, period, confidence = cadence_info

        amounts_abs = grp_sorted["amount"].abs()
        next_date = pd.Timestamp(dates[-1]) + pd.Timedelta(days=period)

        schedules.append(
            RecurringSchedule(
                merchant_key=str(key),
                cadence=cadence,
                amount_mean=float(amounts_abs.mean()),
                amount_std=float(amounts_abs.std(ddof=0)),
                next_date=next_date,
                confidence=float(confidence),
                member_txn_ids=grp_sorted["txn_id"].tolist(),
                is_income=(sign > 0),
            )
        )

    schedules.sort(key=lambda s: s.next_date)
    return schedules


def split_recurring_vs_discretionary(
    transactions: pd.DataFrame,
    schedules: list[RecurringSchedule],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Partition the DEBIT ledger into (recurring, discretionary) frames.

    Credit transactions are excluded from both — income is handled by Stage 2.
    """
    debit_only = transactions[transactions["amount"] < 0].copy()
    recurring_ids: set[str] = set()
    for s in schedules:
        if not s.is_income:
            recurring_ids.update(s.member_txn_ids)
    recurring_df = debit_only[debit_only["txn_id"].isin(recurring_ids)].copy()
    discretionary_df = debit_only[~debit_only["txn_id"].isin(recurring_ids)].copy()
    return recurring_df, discretionary_df
