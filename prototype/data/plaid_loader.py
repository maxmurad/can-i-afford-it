"""Plaid sandbox loader.

Pulls real-shaped transaction data from Plaid's sandbox so the M2-M5 pipeline
can be exercised against the schema and edge cases of a production API rather
than the synthetic generator. Caches raw responses to disk so re-runs don't hit
the API every time (and anyone cloning the repo can replay if they have a
cached response).

Setup:
  .env must contain:
    PLAID_CLIENT_ID=...           # from dashboard.plaid.com
    PLAID_SECRET=...              # the SANDBOX secret
    PLAID_ENV=sandbox             # default

Flow:
  1. /sandbox/public_token/create -> public_token
  2. /item/public_token/exchange  -> access_token (cached in cache/.plaid_state.json)
  3. /transactions/sync (paginated) -> all sandbox transactions
  4. /accounts/get -> current balance per account
  5. Result cached as cache/plaid_<institution>.json

The sandbox default user (`user_good`) returns ~24 months of synthetic
transactions for any standard institution, which is more than enough for
the M5-style calibration backtest.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import httpx
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

PLAID_BASE_URLS = {
    "sandbox": "https://sandbox.plaid.com",
    "development": "https://development.plaid.com",
    "production": "https://production.plaid.com",
}

# Cache lives at prototype/cache/ (gitignored).
CACHE_DIR = Path(__file__).resolve().parent.parent / "cache"
CACHE_DIR.mkdir(exist_ok=True)
STATE_FILE = CACHE_DIR / ".plaid_state.json"

# A few sandbox institutions worth pulling. Add more by visiting
# https://plaid.com/docs/sandbox/institutions/
DEFAULT_INSTITUTIONS = {
    "platypus":   "ins_109508",  # First Platypus Bank
    "gingham":    "ins_109509",  # First Gingham Credit Union
    "houndstooth": "ins_109510", # Houndstooth Bank
}


# ---------------------------------------------------------------------------
# Plaid raw-response cache type
# ---------------------------------------------------------------------------


@dataclass
class PlaidSnapshot:
    """A cached, raw-Plaid-shape pull for one institution."""
    institution_id: str
    label: str
    transactions: list[dict]      # raw Plaid transaction dicts
    accounts: list[dict]          # raw Plaid account dicts (have balances)

    @property
    def current_balance(self) -> float:
        """Sum of `available` (or `current`) balances across all accounts."""
        total = 0.0
        for a in self.accounts:
            bal = a.get("balances", {})
            total += float(bal.get("available") or bal.get("current") or 0.0)
        return total


# ---------------------------------------------------------------------------
# Low-level HTTP helpers
# ---------------------------------------------------------------------------


def _client_info() -> tuple[str, str, str]:
    client_id = os.environ.get("PLAID_CLIENT_ID")
    secret = os.environ.get("PLAID_SECRET") or os.environ.get("PLAID_SANDBOX_SECRET")
    env = os.environ.get("PLAID_ENV", "sandbox")
    if not client_id or not secret:
        raise RuntimeError(
            "PLAID_CLIENT_ID and PLAID_SECRET must be set in .env (see .env.example)"
        )
    if env not in PLAID_BASE_URLS:
        raise ValueError(f"PLAID_ENV must be one of {list(PLAID_BASE_URLS)}, got {env!r}")
    return client_id, secret, PLAID_BASE_URLS[env]


def _post(path: str, payload: dict) -> dict:
    client_id, secret, base_url = _client_info()
    body = {"client_id": client_id, "secret": secret, **payload}
    resp = httpx.post(f"{base_url}{path}", json=body, timeout=30.0)
    if resp.status_code != 200:
        try:
            err = resp.json()
        except Exception:
            err = {"error": resp.text}
        raise RuntimeError(f"Plaid {path} failed [{resp.status_code}]: {err}")
    return resp.json()


# ---------------------------------------------------------------------------
# Token + state management
# ---------------------------------------------------------------------------


def _load_state() -> dict:
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text())
        except Exception:
            return {}
    return {}


def _save_state(state: dict) -> None:
    STATE_FILE.write_text(json.dumps(state, indent=2))


def get_or_create_access_token(institution_id: str) -> str:
    """Get a cached access_token for this institution, or bootstrap one via sandbox.

    Access tokens are durable in Plaid; we cache them and re-use across runs.
    """
    state = _load_state()
    key = f"access_token::{institution_id}"
    if key in state and state[key]:
        return state[key]

    pub = _post(
        "/sandbox/public_token/create",
        {"institution_id": institution_id, "initial_products": ["transactions"]},
    )
    exc = _post("/item/public_token/exchange", {"public_token": pub["public_token"]})
    state[key] = exc["access_token"]
    _save_state(state)
    return state[key]


# ---------------------------------------------------------------------------
# Transaction fetching
# ---------------------------------------------------------------------------


def _fetch_transactions(
    access_token: str,
    days: int = 730,
    expected_count: int | None = None,
) -> list[dict]:
    """Pull every transaction via /transactions/get, paginating with offset.

    Two async hurdles in Plaid, both handled here:
      1. Right after Item creation, /transactions/get returns PRODUCT_NOT_READY
         while the initial pull runs — we retry on that.
      2. Plaid then runs an async HISTORICAL UPDATE that extends the window
         from ~30 days to the full history. Until it finishes, the response
         under-reports total_transactions. We fire /transactions/refresh to
         kick it, then poll total_transactions until it reaches expected_count
         (known for the custom user) or plateaus.
    """
    end = date.today()
    start = end - timedelta(days=days)

    def _get(offset: int, count: int) -> dict:
        payload = {
            "access_token": access_token,
            "start_date": start.isoformat(),
            "end_date": end.isoformat(),
            "options": {"count": count, "offset": offset},
        }
        for _ in range(20):
            try:
                return _post("/transactions/get", payload)
            except RuntimeError as e:
                if "PRODUCT_NOT_READY" in str(e):
                    time.sleep(3)
                    continue
                raise
        raise RuntimeError(
            "Plaid /transactions/get never became ready (PRODUCT_NOT_READY "
            "after ~60s). Re-run — the sandbox item should be ready by then."
        )

    # Kick the async historical update so the full window materializes.
    try:
        _post("/transactions/refresh", {"access_token": access_token})
    except RuntimeError:
        pass  # /transactions/refresh isn't always needed; polling still works.

    # Poll total_transactions until the historical update is complete:
    # either it reaches the count we know we pushed, or it plateaus.
    prev_total = -1
    stable = 0
    for _ in range(40):  # ~2 min cap
        total = _get(0, 1).get("total_transactions", 0)
        if expected_count is not None and total >= expected_count:
            break
        if total == prev_total and total > 0:
            stable += 1
            if stable >= 4:
                break
        else:
            stable = 0
        prev_total = total
        time.sleep(3)

    # Paginate the complete set.
    all_txns: list[dict] = []
    offset = 0
    count = 500
    while True:
        resp = _get(offset, count)
        batch = resp.get("transactions", [])
        all_txns.extend(batch)
        total = resp.get("total_transactions", 0)
        if not batch or len(all_txns) >= total:
            break
        offset += count

    return all_txns


def fetch_snapshot(
    institution_id: str,
    label: str | None = None,
    force_refresh: bool = False,
) -> PlaidSnapshot:
    """Return a PlaidSnapshot for `institution_id`, hitting cache first."""
    label = label or institution_id
    cache_file = CACHE_DIR / f"plaid_{label}.json"

    # Use the cache only if it exists AND actually has transactions — an
    # empty cached pull (e.g. from a warm-up race) should be re-fetched.
    if cache_file.exists() and not force_refresh:
        data = json.loads(cache_file.read_text())
        if data.get("transactions"):
            return PlaidSnapshot(
                institution_id=data["institution_id"],
                label=data["label"],
                transactions=data["transactions"],
                accounts=data["accounts"],
            )

    access_token = get_or_create_access_token(institution_id)
    txns = _fetch_transactions(access_token)
    accounts_resp = _post("/accounts/get", {"access_token": access_token})
    accounts = accounts_resp.get("accounts", [])

    data = {
        "institution_id": institution_id,
        "label": label,
        "transactions": txns,
        "accounts": accounts,
    }
    cache_file.write_text(json.dumps(data, indent=2, default=str))
    return PlaidSnapshot(
        institution_id=institution_id,
        label=label,
        transactions=txns,
        accounts=accounts,
    )


def fetch_default_snapshots(force_refresh: bool = False) -> list[PlaidSnapshot]:
    """Pull all institutions in DEFAULT_INSTITUTIONS.

    NOTE: Plaid's default sandbox user (`user_good`) carries only a thin,
    recent transaction set (~48 txns / ~80 days) — too sparse for the
    calibration backtest. Use the custom-user path below for real testing.
    """
    return [
        fetch_snapshot(inst_id, label=label, force_refresh=force_refresh)
        for label, inst_id in DEFAULT_INSTITUTIONS.items()
    ]


# ---------------------------------------------------------------------------
# Custom sandbox user — inject a rich transaction history via Plaid's API
# ---------------------------------------------------------------------------
#
# Plaid's default sandbox user is too thin to backtest. The documented way
# to get rich sandbox data is the "custom user": create an Item with
# override_username="user_custom" and an override_password that is a JSON
# spec of accounts and transactions.
#
# We use this to push generator-built households THROUGH the real Plaid API
# and pull them back. This validates the full integration path — auth, item
# creation, /transactions/get pagination, the schema adapter, and balance
# reconstruction — against genuine Plaid endpoints. It does NOT independently
# re-validate the model (the data is still ours, just round-tripped);
# that requires real consented accounts.


def synth_to_plaid_override(transactions, current_balance: float) -> dict:
    """Convert an internal-schema ledger into a Plaid custom-user override.

    Plaid's sign convention is the opposite of ours: in Plaid a positive
    transaction amount is a debit (money out). Our internal schema uses
    negative for debits — so we flip the sign on the way out.

    Args:
        transactions: internal-schema DataFrame (txn_id, date, amount, merchant).
        current_balance: the account's current/ending balance.
    """
    txns = []
    for row in transactions.itertuples(index=False):
        d = pd.Timestamp(row.date).date().isoformat()
        txns.append(
            {
                "date_transacted": d,
                "date_posted": d,
                "amount": round(-float(row.amount), 2),  # flip sign for Plaid
                "description": str(row.merchant),
                "currency": "USD",
            }
        )
    return {
        "override_accounts": [
            {
                "type": "depository",
                "subtype": "checking",
                "starting_balance": round(float(current_balance), 2),
                "meta": {"name": "Checking"},
                "transactions": txns,
            }
        ]
    }


def create_custom_sandbox_item(label: str, override: dict, force_new: bool = False) -> str:
    """Create (or reuse) a Plaid sandbox Item populated with `override` data.

    The access token is cached in cache/.plaid_state.json keyed by label so
    re-runs reuse the same Item.
    """
    state = _load_state()
    key = f"custom_access_token::{label}"
    if key in state and state[key] and not force_new:
        return state[key]

    options = {
        "override_username": "user_custom",
        "override_password": json.dumps(override),
    }
    # NOTE: Plaid's headless sandbox endpoint (/sandbox/public_token/create)
    # caps returned transaction history at ~90 days. `transactions.days_requested`
    # (max 730) is only settable through the full Link flow, which requires a
    # browser UI — not feasible from a script. So the custom-user round-trip
    # yields ~90 days regardless. That's enough to validate the integration
    # plumbing; deeper history needs real consented accounts.
    pub = _post(
        "/sandbox/public_token/create",
        {
            "institution_id": "ins_109508",  # any institution; custom user supplies the data
            "initial_products": ["transactions"],
            "options": options,
        },
    )
    exc = _post("/item/public_token/exchange", {"public_token": pub["public_token"]})
    state[key] = exc["access_token"]
    _save_state(state)
    return state[key]


def fetch_custom_snapshot(
    label: str,
    transactions,
    current_balance: float,
    force_refresh: bool = False,
) -> PlaidSnapshot:
    """Push an internal-schema ledger through Plaid as a custom sandbox user,
    then pull it back via the real API. Cached after the first run.
    """
    expected = len(transactions)
    cache_file = CACHE_DIR / f"plaid_custom_{label}.json"
    if cache_file.exists() and not force_refresh:
        data = json.loads(cache_file.read_text())
        cached = data.get("transactions", [])
        # Plaid's headless sandbox returns at most ~90 days regardless of how
        # much history we push, so a non-empty cache IS the complete result.
        # (Pass force_refresh=True to deliberately re-pull.)
        if cached:
            return PlaidSnapshot(
                institution_id=data["institution_id"],
                label=data["label"],
                transactions=cached,
                accounts=data["accounts"],
            )

    override = synth_to_plaid_override(transactions, current_balance)
    access_token = create_custom_sandbox_item(label, override, force_new=force_refresh)
    txns = _fetch_transactions(access_token, expected_count=expected)
    accounts_resp = _post("/accounts/get", {"access_token": access_token})
    accounts = accounts_resp.get("accounts", [])

    data = {
        "institution_id": "ins_109508",
        "label": label,
        "transactions": txns,
        "accounts": accounts,
    }
    cache_file.write_text(json.dumps(data, indent=2, default=str))
    return PlaidSnapshot(
        institution_id="ins_109508",
        label=label,
        transactions=txns,
        accounts=accounts,
    )
