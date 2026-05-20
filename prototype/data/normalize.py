"""Schema normalization — Plaid → internal transaction ledger.

The cashflow engine (cashflow/recurring, cashflow/income, etc.) works on
a single normalized DataFrame with these columns:

    txn_id, date, account_id, amount, merchant
    (optionally: txn_type, category, is_recurring, source_id)

Plaid's transaction shape uses the OPPOSITE SIGN CONVENTION — positive
amounts are debits. We flip the sign here so downstream code is unchanged.

Also reconstructs the daily-balance series from current balance + the
transactions, by walking backward in time.
"""
from __future__ import annotations

import pandas as pd

from data.plaid_loader import PlaidSnapshot


def plaid_to_ledger(snapshot: PlaidSnapshot) -> pd.DataFrame:
    """Convert raw Plaid transactions into the internal ledger schema."""
    rows = []
    for t in snapshot.transactions:
        # Plaid: positive = debit (outflow). Internal: negative = debit.
        amount = -float(t["amount"])
        merchant = (
            t.get("merchant_name")
            or t.get("name")
            or "UNKNOWN"
        )
        rows.append(
            {
                "txn_id": t["transaction_id"],
                "date": pd.to_datetime(t.get("date") or t.get("authorized_date")),
                "account_id": t["account_id"],
                "amount": round(amount, 2),
                "merchant": str(merchant).strip(),
                # On real data we don't have ground-truth labels.
                "is_recurring": False,
                "source_id": None,
                "txn_type": "income" if amount > 0 else "debit",
                "category": (t.get("category") or [None])[0],
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.sort_values("date", kind="stable").reset_index(drop=True)
    return df


def reconstruct_balance_series(
    ledger: pd.DataFrame,
    current_balance: float,
    as_of: pd.Timestamp | None = None,
) -> pd.Series:
    """Reconstruct the running daily balance series from current balance.

    Plaid gives us `current balance` and the transaction list. To get
    end-of-day balance for any historical date D:

        balance(D) = current_balance - sum(amount for txns with date > D)

    (Subtracting future signed amounts "undoes" them — amount is negative
    for debits in our schema, so subtracting a negative debit removes the
    debit, i.e., the historical balance was higher than today's.)

    Returns a daily-indexed Series, forward-filled across no-txn days.
    """
    if ledger.empty:
        return pd.Series(dtype=float)

    as_of = as_of or ledger["date"].max()
    daily = ledger.groupby(ledger["date"].dt.normalize())["amount"].sum().sort_index()
    full_idx = pd.date_range(daily.index.min(), as_of, freq="D")
    daily = daily.reindex(full_idx, fill_value=0.0)

    # End-of-day balance: current_balance minus sum of amounts AFTER each day.
    # Compute reverse cumulative sum of daily deltas (exclusive of day D).
    reverse_cum_after = daily[::-1].cumsum()[::-1] - daily   # sum of amounts strictly AFTER day
    balance = current_balance - reverse_cum_after
    return balance


def summarize_snapshot(snapshot: PlaidSnapshot, ledger: pd.DataFrame) -> str:
    """One-line readable summary of a Plaid snapshot post-normalization."""
    if ledger.empty:
        return f"{snapshot.label}: <empty ledger>"
    n_txns = len(ledger)
    date_min = ledger["date"].min().date()
    date_max = ledger["date"].max().date()
    debits = ledger.loc[ledger.amount < 0, "amount"].abs().sum()
    credits = ledger.loc[ledger.amount > 0, "amount"].sum()
    return (
        f"{snapshot.label:<14} {n_txns:>5,} txns  "
        f"{date_min} -> {date_max}  "
        f"debits ${debits:>10,.0f}  credits ${credits:>10,.0f}  "
        f"current ${snapshot.current_balance:>9,.0f}"
    )
