"""Stage 3 — Discretionary spending forecast.

See ../../PREDICTION_MODEL_DESIGN.md section 5.

After recurring obligations and income are removed, what remains is
discretionary spending. It is NOT modeled as discrete predictable events;
it is modeled as a STATISTICAL PROCESS the projection step samples from
thousands of times.

The model:
  - For each day-of-week (0=Mon ... 6=Sun), learn:
      * txn_rate     = mean number of discretionary transactions per day
      * amount_mean  = mean per-transaction amount
      * amount_std   = spread of per-transaction amount
  - Sampling one day: draw n ~ Poisson(rate), then n amounts from a
    clipped Normal(mean, std), return their sum.

The KEY interface for M4 is `sample_daily_spend(model, day, rng)` —
the Monte Carlo projection will call it thousands of times.

Deferred refinements (don't add until baseline proves well-calibrated):
  - category-level rates
  - month-phase effects
  - paycheck-proximity effects
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class DiscretionaryModel:
    """Fitted discretionary-spending process — DOW-aware Poisson + Normal-clipped amount."""
    dow_txn_rate: dict[int, float]
    dow_amount_mean: dict[int, float]
    dow_amount_std: dict[int, float]
    overall_daily_mean: float
    n_history_days: int


def fit_discretionary(discretionary_txns: pd.DataFrame) -> DiscretionaryModel:
    """Fit a day-of-week-aware spending process from leftover (non-recurring) debits.

    Args:
        discretionary_txns: the non-recurring frame from
            recurring.split_recurring_vs_discretionary().

    Returns:
        DiscretionaryModel ready to be sampled.
    """
    if discretionary_txns.empty:
        return DiscretionaryModel(
            dow_txn_rate={d: 0.0 for d in range(7)},
            dow_amount_mean={d: 0.0 for d in range(7)},
            dow_amount_std={d: 0.0 for d in range(7)},
            overall_daily_mean=0.0,
            n_history_days=0,
        )

    df = discretionary_txns.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["dow"] = df["date"].dt.dayofweek
    df["abs_amount"] = df["amount"].abs()

    # Per-day transaction counts over a complete daily calendar so days with
    # zero discretionary spend pull the DOW rate down correctly.
    date_range = pd.date_range(df["date"].min(), df["date"].max(), freq="D")
    n_days = len(date_range)
    daily_counts = (
        df.groupby("date").size().reindex(date_range, fill_value=0)
    )
    daily_counts = pd.DataFrame(
        {"n": daily_counts.values, "dow": daily_counts.index.dayofweek}
    )
    dow_txn_rate = daily_counts.groupby("dow")["n"].mean().to_dict()

    # Per-transaction amount stats by DOW (across all txns falling on each DOW).
    dow_amount_mean = df.groupby("dow")["abs_amount"].mean().to_dict()
    dow_amount_std = df.groupby("dow")["abs_amount"].std(ddof=0).to_dict()

    overall_amt_mean = float(df["abs_amount"].mean())
    overall_amt_std = float(df["abs_amount"].std(ddof=0))
    for d in range(7):
        dow_txn_rate.setdefault(d, 0.0)
        dow_amount_mean.setdefault(d, overall_amt_mean)
        dow_amount_std.setdefault(d, overall_amt_std)

    overall_daily_mean = float(df["abs_amount"].sum() / max(n_days, 1))

    return DiscretionaryModel(
        dow_txn_rate={int(k): float(v) for k, v in dow_txn_rate.items()},
        dow_amount_mean={int(k): float(v) for k, v in dow_amount_mean.items()},
        dow_amount_std={int(k): float(v) for k, v in dow_amount_std.items()},
        overall_daily_mean=overall_daily_mean,
        n_history_days=n_days,
    )


def sample_daily_spend(
    model: DiscretionaryModel,
    day: pd.Timestamp,
    rng: np.random.Generator,
) -> float:
    """Sample ONE plausible day of total discretionary spend (a positive number).

    Called many times per simulated trajectory by the Monte Carlo projection.
    """
    dow = pd.Timestamp(day).dayofweek
    rate = model.dow_txn_rate.get(dow, 0.0)
    n = int(rng.poisson(rate))
    if n == 0:
        return 0.0
    mean = model.dow_amount_mean.get(dow, 0.0)
    std = model.dow_amount_std.get(dow, 0.0)
    amounts = rng.normal(mean, std, size=n)
    amounts = np.clip(amounts, 1.0, None)
    return float(amounts.sum())


def expected_daily_spend(model: DiscretionaryModel, day: pd.Timestamp) -> float:
    """Analytical expected spend on `day` given its day-of-week."""
    dow = pd.Timestamp(day).dayofweek
    return model.dow_txn_rate.get(dow, 0.0) * model.dow_amount_mean.get(dow, 0.0)


def sample_daily_spend_batch(
    model: DiscretionaryModel,
    day: pd.Timestamp,
    rng: np.random.Generator,
    n_samples: int,
) -> np.ndarray:
    """Vectorized: sample `n_samples` plausible daily spends for `day`.

    Equivalent to calling sample_daily_spend() n_samples times but ~100x faster
    — the M4 Monte Carlo projection calls this once per simulated day.
    """
    dow = pd.Timestamp(day).dayofweek
    rate = model.dow_txn_rate.get(dow, 0.0)
    counts = rng.poisson(rate, size=n_samples)
    if counts.max() == 0 or rate == 0.0:
        return np.zeros(n_samples, dtype=np.float64)

    mean = model.dow_amount_mean.get(dow, 0.0)
    std = model.dow_amount_std.get(dow, 0.0)
    max_n = int(counts.max())

    # Draw a (n_samples, max_n) matrix of amounts, then sum only `counts[i]`
    # of them in each row via a mask. Avoids Python-level looping.
    amounts = rng.normal(mean, std, size=(n_samples, max_n))
    np.clip(amounts, 1.0, None, out=amounts)
    mask = np.arange(max_n) < counts[:, None]
    return (amounts * mask).sum(axis=1)


def naive_baseline(discretionary_txns: pd.DataFrame) -> float:
    """Flat average $/day — the bar the DOW model must beat."""
    if discretionary_txns.empty:
        return 0.0
    df = discretionary_txns.copy()
    df["date"] = pd.to_datetime(df["date"])
    date_range = pd.date_range(df["date"].min(), df["date"].max(), freq="D")
    return float(df["amount"].abs().sum() / max(len(date_range), 1))
