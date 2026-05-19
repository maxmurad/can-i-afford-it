# Can I Afford It?

A consumer-banking copilot that answers forward-looking affordability questions — *"can I afford this $X spend right now, given my bills and paychecks?"* — with calibrated confidence and an attributable low-point date.

Built free-first as a public-good passion project, with an earned, conditional path to monetization if and only if the product proves genuinely engaging at scale.

## What's in this repo

The project is two artifacts: a written strategy and a working technical prototype.

[Market & PRD](Market_and_PRD.md) lays out the consumer-banking pain point this targets (forward-looking cash-flow uncertainty), maps the competitive landscape (PFM tools, cash-advance apps, bank-native forecasts, aggregation infrastructure), explains why no incumbent solves the specific affordability job well, and defines a v1 product concept built around a single sentence-length answer rather than a dashboard.

[Prediction Model Design](PREDICTION_MODEL_DESIGN.md) is the technical companion: a four-stage forecasting pipeline (recurring-bill detection, income timing, discretionary spending, Monte Carlo projection) feeding a calibrated affordability query, with an explicit evaluation methodology that treats calibration — not point accuracy — as the metric that matters.

[`prototype/`](prototype/) is the buildable, runnable Python implementation. All five core milestones from the design doc are complete; see the prototype's [README](prototype/README.md) for the pipeline diagram, build status, and headline results.

## Quick start

```bash
git clone https://github.com/maxmurad/can-i-afford-it.git
cd can-i-afford-it/prototype
uv sync
uv run python notebooks/05_calibration.py
```

That runs the full 80-household calibration backtest end-to-end and reproduces the headline numbers below.

## Headline results from the prototype

Validated on 80 synthetic households (4 personas × 20 seeds) with 800 random affordability questions:

- **False-affordable rate: 0.6%** — the trust-destroying error (telling a user "yes" before an overdraft) is rare by design.
- **Recurring-bill detection: 100% precision, 100% recall, 100% cadence accuracy** across all four personas including the subscription-heavy stress case.
- **Calibration error: 8.2%**, with the predictive intervals slightly tight — a known limitation of the single-Normal-per-day-of-week amount model. The improvement direction is identified (mixture-of-Normals or Student's t in Stage 3).

![Reliability diagram — calibration of predictive intervals](prototype/figures/05_reliability.png)

## What the API looks like

After fitting Stages 1–3 on a household's transaction history, the Monte Carlo projection and affordability query are a few lines:

```python
from cashflow.projection import project, can_i_afford

projection = project(
    current_balance=4_000,
    start_date="2024-12-01",
    horizon_days=30,
    recurring=schedules,            # from detect_recurring()
    income=income_forecast,         # from forecast_income()
    discretionary=discretionary_model,  # from fit_discretionary()
    n_sims=5_000,
)

answer = can_i_afford(projection, amount=500, on_date="2024-12-15")

answer.verdict             # -> "tight"
answer.prob_below_buffer   # -> 0.19
answer.low_point_date      # -> Timestamp("2024-12-22")
answer.driving_obligation  # -> "geico"
```

The verdict is graded ("affordable" / "tight" / "not affordable") rather than a bare yes/no, and the engine surfaces *when* the low point lands and *which* recurring obligation drives it — so the user gets a reason, not just an answer.

## What this is not

This is not a venture pitch and not an app. It is a strategy document plus a notebook-provable engine, demonstrating that the forecasting problem behind the product can be solved with calibrated confidence. The next milestone (M6) is to swap synthetic data for real anonymized transactions; the evaluation harness is designed for that swap to be mechanical.

## Author

Max Murad — ex-JPMorgan Chase consumer-banking data platform (~12 years), most recently at Meta on Ads ML and Quest personalization.
