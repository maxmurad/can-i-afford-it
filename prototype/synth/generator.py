"""Synthetic household transaction generator.

Produces realistic transaction ledgers for synthetic households WITH
ground-truth labels, so the cash-flow models can be developed and the
evaluation harness can score detection against known truth — all before
real anonymized data is available.

What it gives you:
  - a transaction DataFrame with signed amounts, merchant strings, accounts
  - ground-truth columns (is_recurring, source_id, txn_type, category)
  - the underlying schedules (income sources + recurring bills) so detection
    can be scored against the true cadence/amount, not just per-transaction
  - a running balance series

Design choices:
  - Fully reproducible via `seed`.
  - Profiles are data, not code — see PRESET_PROFILES for examples and add
    your own (irregular income, thin file, subscription-heavy, etc.).
  - `description_noise=True` appends junk (store #, txn id) to merchant
    strings so Stage 1 detection has to do real fuzzy matching.

Usage:
    from synth.generator import generate, PRESET_PROFILES
    ds = generate(PRESET_PROFILES["steady_biweekly"], start="2024-01-01",
                  n_days=365, seed=0)
    ds.transactions.head()
    ds.ground_truth          # schedules behind the data
    ds.balance_series        # date -> running balance
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration dataclasses
# ---------------------------------------------------------------------------


@dataclass
class IncomeSource:
    """A stream of credit (income) transactions."""
    name: str                       # ground-truth label, e.g. "primary_paycheck"
    merchant: str                   # description stem, e.g. "ACME CORP PAYROLL"
    amount_mean: float
    amount_std: float               # 0.0 for a perfectly fixed paycheck
    frequency: str                  # "biweekly" | "semimonthly" | "monthly" | "irregular"
    anchor_day: int = 1             # monthly: day-of-month; semimonthly: first of the pair
    second_day: int = 15            # semimonthly only: the second pay day
    # irregular-income controls (ignored unless frequency == "irregular"):
    irregular_min_gap: int = 3
    irregular_max_gap: int = 21


@dataclass
class RecurringBill:
    """A regular debit obligation — rent, utility, subscription, insurance."""
    name: str                       # ground-truth label, e.g. "rent"
    merchant: str                   # description stem, e.g. "GREENWAY PROPERTY MGMT"
    category: str                   # "housing" | "utilities" | "subscription" | "insurance" | ...
    amount_mean: float
    amount_std: float               # 0.0 = fixed (rent); >0 = variable (utility)
    day_of_month: int               # 1-28, the day it usually posts
    day_jitter: int = 0             # +/- days of randomness around day_of_month


@dataclass
class DiscretionaryCategory:
    """A statistical stream of irregular discretionary spending."""
    name: str                       # "groceries" | "dining" | "fuel" | "shopping" | ...
    merchants: list[str]            # pool of merchant strings to draw from
    daily_txn_rate: float           # mean transactions per day (Poisson)
    amount_mean: float              # mean transaction amount (dollars)
    amount_std: float               # spread of transaction amount
    weekend_multiplier: float = 1.0  # >1 = more spending Sat/Sun


@dataclass
class HouseholdProfile:
    """A complete synthetic household specification."""
    name: str
    starting_balance: float
    income_sources: list[IncomeSource]
    recurring_bills: list[RecurringBill]
    discretionary: list[DiscretionaryCategory]
    description_noise: bool = False  # append store#/txn-id junk to descriptions


@dataclass
class SyntheticDataset:
    """Output of `generate`: the ledger plus everything needed to score against."""
    transactions: pd.DataFrame       # the labeled ledger
    ground_truth: dict               # {"income": [...], "recurring": [...]}
    balance_series: pd.Series        # date -> running balance
    profile: HouseholdProfile

    def summary(self) -> str:
        t = self.transactions
        debits = t.loc[t.amount < 0, "amount"].sum()
        credits = t.loc[t.amount > 0, "amount"].sum()
        rec = t.loc[t.is_recurring].shape[0]
        return (
            f"Household: {self.profile.name}\n"
            f"  transactions:      {len(t)}  ({rec} recurring, {len(t) - rec} discretionary)\n"
            f"  date range:        {t.date.min().date()} -> {t.date.max().date()}\n"
            f"  total credits:     ${credits:,.0f}\n"
            f"  total debits:      ${-debits:,.0f}\n"
            f"  net:               ${credits + debits:,.0f}\n"
            f"  starting balance:  ${self.profile.starting_balance:,.0f}\n"
            f"  ending balance:    ${self.balance_series.iloc[-1]:,.0f}\n"
            f"  min balance:       ${self.balance_series.min():,.0f}\n"
            f"  days below $0:     {(self.balance_series < 0).sum()}"
        )


# ---------------------------------------------------------------------------
# Date helpers
# ---------------------------------------------------------------------------


def _month_day(year: int, month: int, day: int) -> date:
    """A date clamped to a valid day-of-month (handles day 31 in Feb, etc.)."""
    # day is assumed 1-28 for bills, but clamp defensively anyway
    import calendar

    last = calendar.monthrange(year, month)[1]
    return date(year, month, min(day, last))


def _iter_months(start: date, end: date):
    """Yield (year, month) pairs covering [start, end]."""
    y, m = start.year, start.month
    while date(y, m, 1) <= end:
        yield y, m
        m += 1
        if m > 12:
            m, y = 1, y + 1


# ---------------------------------------------------------------------------
# Per-stream generators
# ---------------------------------------------------------------------------


def _gen_income(src: IncomeSource, start: date, end: date, rng) -> list[dict]:
    """Generate dated credit events for one income source."""
    rows: list[dict] = []
    dates: list[date] = []

    if src.frequency == "biweekly":
        # anchor on the first occurrence within the window
        d = start
        while d <= end:
            dates.append(d)
            d += timedelta(days=14)
    elif src.frequency == "semimonthly":
        for y, m in _iter_months(start, end):
            for dom in (src.anchor_day, src.second_day):
                d = _month_day(y, m, dom)
                if start <= d <= end:
                    dates.append(d)
    elif src.frequency == "monthly":
        for y, m in _iter_months(start, end):
            d = _month_day(y, m, src.anchor_day)
            if start <= d <= end:
                dates.append(d)
    elif src.frequency == "irregular":
        d = start + timedelta(days=int(rng.integers(0, src.irregular_max_gap)))
        while d <= end:
            dates.append(d)
            gap = int(rng.integers(src.irregular_min_gap, src.irregular_max_gap + 1))
            d += timedelta(days=gap)
    else:
        raise ValueError(f"unknown income frequency: {src.frequency!r}")

    for d in dates:
        amt = float(rng.normal(src.amount_mean, src.amount_std))
        amt = max(amt, 1.0)  # income is always positive
        rows.append(
            {
                "date": d,
                "amount": round(amt, 2),
                "merchant": src.merchant,
                "txn_type": "income",
                "category": "income",
                "is_recurring": True,
                "source_id": src.name,
            }
        )
    return rows


def _gen_bill(bill: RecurringBill, start: date, end: date, rng) -> list[dict]:
    """Generate dated debit events for one recurring bill."""
    rows: list[dict] = []
    for y, m in _iter_months(start, end):
        jitter = int(rng.integers(-bill.day_jitter, bill.day_jitter + 1)) if bill.day_jitter else 0
        dom = min(max(bill.day_of_month + jitter, 1), 28)
        d = _month_day(y, m, dom)
        if not (start <= d <= end):
            continue
        amt = float(rng.normal(bill.amount_mean, bill.amount_std))
        amt = max(amt, 1.0)
        rows.append(
            {
                "date": d,
                "amount": -round(amt, 2),  # debit
                "merchant": bill.merchant,
                "txn_type": "bill",
                "category": bill.category,
                "is_recurring": True,
                "source_id": bill.name,
            }
        )
    return rows


def _gen_discretionary(cat: DiscretionaryCategory, start: date, end: date, rng) -> list[dict]:
    """Generate dated debit events for one discretionary spending category."""
    rows: list[dict] = []
    n_days = (end - start).days + 1
    for i in range(n_days):
        d = start + timedelta(days=i)
        is_weekend = d.weekday() >= 5
        rate = cat.daily_txn_rate * (cat.weekend_multiplier if is_weekend else 1.0)
        n_txns = rng.poisson(rate)
        for _ in range(n_txns):
            amt = float(rng.normal(cat.amount_mean, cat.amount_std))
            amt = max(amt, 1.0)
            merchant = str(rng.choice(cat.merchants))
            rows.append(
                {
                    "date": d,
                    "amount": -round(amt, 2),
                    "merchant": merchant,
                    "txn_type": "discretionary",
                    "category": cat.name,
                    "is_recurring": False,
                    "source_id": None,
                }
            )
    return rows


def _add_description_noise(merchant: str, rng) -> str:
    """Append realistic junk (store #, txn id) so detection must fuzzy-match."""
    roll = rng.random()
    if roll < 0.45:
        return f"{merchant} #{int(rng.integers(100, 9999))}"
    if roll < 0.7:
        return f"{merchant} {int(rng.integers(10000000, 99999999))}"
    return merchant


# ---------------------------------------------------------------------------
# Top-level generate
# ---------------------------------------------------------------------------


def generate(
    profile: HouseholdProfile,
    start: str | date = "2024-01-01",
    n_days: int = 365,
    seed: int = 0,
) -> SyntheticDataset:
    """Generate a synthetic transaction ledger for one household.

    Args:
        profile:  the household specification (see PRESET_PROFILES).
        start:    first date of the window (ISO string or date).
        n_days:   length of the window in days.
        seed:     RNG seed — same seed + profile => identical output.

    Returns:
        SyntheticDataset with .transactions, .ground_truth, .balance_series.
    """
    rng = np.random.default_rng(seed)
    start_d = date.fromisoformat(start) if isinstance(start, str) else start
    end_d = start_d + timedelta(days=n_days - 1)

    rows: list[dict] = []
    for src in profile.income_sources:
        rows.extend(_gen_income(src, start_d, end_d, rng))
    for bill in profile.recurring_bills:
        rows.extend(_gen_bill(bill, start_d, end_d, rng))
    for cat in profile.discretionary:
        rows.extend(_gen_discretionary(cat, start_d, end_d, rng))

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date", kind="stable").reset_index(drop=True)

    # Optionally roughen the merchant strings.
    if profile.description_noise:
        df["merchant"] = df["merchant"].map(lambda m: _add_description_noise(m, rng))

    # Single account for the prototype; multi-account is a later extension.
    df["account_id"] = "checking"
    df["txn_id"] = [f"t{i:06d}" for i in range(len(df))]

    # Running balance.
    df["balance_after"] = profile.starting_balance + df["amount"].cumsum()
    balance_series = (
        df.set_index("date")["balance_after"].groupby(level=0).last()
    )
    # reindex to a full daily calendar so gaps are filled forward
    full_idx = pd.date_range(start_d, end_d, freq="D")
    balance_series = balance_series.reindex(full_idx).ffill()
    balance_series.iloc[0] = (
        balance_series.iloc[0]
        if not pd.isna(balance_series.iloc[0])
        else profile.starting_balance
    )
    balance_series = balance_series.ffill()

    # Column order for readability.
    df = df[
        [
            "txn_id",
            "date",
            "account_id",
            "amount",
            "merchant",
            "txn_type",
            "category",
            "is_recurring",
            "source_id",
            "balance_after",
        ]
    ]

    ground_truth = {
        "income": [src for src in profile.income_sources],
        "recurring": [bill for bill in profile.recurring_bills],
    }

    return SyntheticDataset(
        transactions=df,
        ground_truth=ground_truth,
        balance_series=balance_series,
        profile=profile,
    )


# ---------------------------------------------------------------------------
# Preset profiles — starting points; copy and edit to make your own
# ---------------------------------------------------------------------------

_GROCERIES = ["WHOLE FOODS", "TRADER JOES", "SAFEWAY", "KROGER", "ALDI"]
_DINING = ["STARBUCKS", "CHIPOTLE", "SWEETGREEN", "DOORDASH", "LOCAL CAFE"]
_FUEL = ["SHELL", "CHEVRON", "EXXON", "COSTCO GAS"]
_SHOPPING = ["AMAZON", "TARGET", "WALMART", "BEST BUY", "CVS"]

PRESET_PROFILES: dict[str, HouseholdProfile] = {
    # The PRD's primary persona: steady biweekly paycheck, normal bill load.
    "steady_biweekly": HouseholdProfile(
        name="steady_biweekly",
        starting_balance=1800.0,
        income_sources=[
            IncomeSource(
                name="primary_paycheck",
                merchant="ACME CORP PAYROLL",
                amount_mean=2150.0,
                amount_std=40.0,
                frequency="biweekly",
            )
        ],
        recurring_bills=[
            RecurringBill("rent", "GREENWAY PROPERTY MGMT", "housing", 1650.0, 0.0, 1),
            RecurringBill("electric", "CITY POWER & LIGHT", "utilities", 110.0, 35.0, 12, day_jitter=2),
            RecurringBill("internet", "XFINITY", "utilities", 75.0, 0.0, 8),
            RecurringBill("phone", "VERIZON WIRELESS", "utilities", 90.0, 5.0, 18),
            RecurringBill("car_insurance", "GEICO", "insurance", 145.0, 0.0, 22),
            RecurringBill("netflix", "NETFLIX.COM", "subscription", 15.49, 0.0, 5),
            RecurringBill("spotify", "SPOTIFY USA", "subscription", 11.99, 0.0, 14),
            RecurringBill("gym", "PLANET FITNESS", "subscription", 24.99, 0.0, 25),
        ],
        discretionary=[
            DiscretionaryCategory("groceries", _GROCERIES, 0.55, 62.0, 28.0, weekend_multiplier=1.5),
            DiscretionaryCategory("dining", _DINING, 0.7, 16.0, 9.0, weekend_multiplier=1.8),
            DiscretionaryCategory("fuel", _FUEL, 0.18, 48.0, 12.0),
            DiscretionaryCategory("shopping", _SHOPPING, 0.35, 41.0, 35.0, weekend_multiplier=1.4),
        ],
        description_noise=True,
    ),
    # Hard case: gig income — variable amount AND variable timing.
    "irregular_gig": HouseholdProfile(
        name="irregular_gig",
        starting_balance=900.0,
        income_sources=[
            IncomeSource(
                name="gig_income",
                merchant="UBER DRIVER PARTNER",
                amount_mean=620.0,
                amount_std=220.0,
                frequency="irregular",
                irregular_min_gap=2,
                irregular_max_gap=11,
            )
        ],
        recurring_bills=[
            RecurringBill("rent", "SUNSET APARTMENTS LLC", "housing", 1200.0, 0.0, 1),
            RecurringBill("electric", "PG&E", "utilities", 95.0, 30.0, 14, day_jitter=3),
            RecurringBill("phone", "T-MOBILE", "utilities", 65.0, 0.0, 20),
            RecurringBill("car_insurance", "PROGRESSIVE", "insurance", 165.0, 0.0, 10),
            RecurringBill("netflix", "NETFLIX.COM", "subscription", 15.49, 0.0, 7),
        ],
        discretionary=[
            DiscretionaryCategory("groceries", _GROCERIES, 0.45, 48.0, 22.0, weekend_multiplier=1.3),
            DiscretionaryCategory("dining", _DINING, 0.5, 13.0, 7.0, weekend_multiplier=1.6),
            DiscretionaryCategory("fuel", _FUEL, 0.4, 42.0, 11.0),  # gig driver buys lots of gas
        ],
        description_noise=True,
    ),
    # Subscription-heavy: many small recurring debits — stresses Stage 1 detection.
    "subscription_heavy": HouseholdProfile(
        name="subscription_heavy",
        starting_balance=2600.0,
        income_sources=[
            IncomeSource(
                name="primary_paycheck",
                merchant="TECHCO INC PAYROLL",
                amount_mean=2900.0,
                amount_std=30.0,
                frequency="semimonthly",
                anchor_day=15,
                second_day=28,
            )
        ],
        recurring_bills=[
            RecurringBill("rent", "URBAN LIVING TRUST", "housing", 2100.0, 0.0, 1),
            RecurringBill("electric", "CONED", "utilities", 130.0, 40.0, 11, day_jitter=2),
            RecurringBill("internet", "SPECTRUM", "utilities", 80.0, 0.0, 6),
            RecurringBill("phone", "AT&T WIRELESS", "utilities", 95.0, 5.0, 19),
            RecurringBill("renters_insurance", "LEMONADE", "insurance", 18.0, 0.0, 3),
            RecurringBill("netflix", "NETFLIX.COM", "subscription", 22.99, 0.0, 5),
            RecurringBill("spotify", "SPOTIFY USA", "subscription", 11.99, 0.0, 8),
            RecurringBill("hulu", "HULU", "subscription", 17.99, 0.0, 12),
            RecurringBill("disney", "DISNEY PLUS", "subscription", 13.99, 0.0, 15),
            RecurringBill("nyt", "NYTIMES", "subscription", 17.0, 0.0, 17),
            RecurringBill("icloud", "APPLE.COM/BILL", "subscription", 9.99, 0.0, 21),
            RecurringBill("chatgpt", "OPENAI", "subscription", 20.0, 0.0, 23),
            RecurringBill("gym", "EQUINOX", "subscription", 215.0, 0.0, 25),
            RecurringBill("peloton", "PELOTON", "subscription", 44.0, 0.0, 27),
        ],
        discretionary=[
            DiscretionaryCategory("groceries", _GROCERIES, 0.5, 70.0, 30.0, weekend_multiplier=1.4),
            DiscretionaryCategory("dining", _DINING, 1.1, 22.0, 14.0, weekend_multiplier=1.7),
            DiscretionaryCategory("shopping", _SHOPPING, 0.5, 55.0, 48.0, weekend_multiplier=1.3),
        ],
        description_noise=True,
    ),
    # Thin file: short history, few accounts, sparse activity.
    "thin_file": HouseholdProfile(
        name="thin_file",
        starting_balance=600.0,
        income_sources=[
            IncomeSource(
                name="primary_paycheck",
                merchant="RETAIL STORE 4471 PAYROLL",
                amount_mean=780.0,
                amount_std=80.0,
                frequency="biweekly",
            )
        ],
        recurring_bills=[
            RecurringBill("rent", "ROOMMATE ZELLE", "housing", 800.0, 0.0, 3, day_jitter=2),
            RecurringBill("phone", "CRICKET WIRELESS", "utilities", 50.0, 0.0, 15),
        ],
        discretionary=[
            DiscretionaryCategory("groceries", _GROCERIES, 0.4, 35.0, 18.0),
            DiscretionaryCategory("dining", _DINING, 0.4, 11.0, 6.0, weekend_multiplier=1.5),
        ],
        description_noise=False,
    ),
}


if __name__ == "__main__":
    # Quick self-test: generate every preset and print a summary.
    for key in PRESET_PROFILES:
        ds = generate(PRESET_PROFILES[key], start="2024-01-01", n_days=365, seed=0)
        print(ds.summary())
        print("-" * 60)
