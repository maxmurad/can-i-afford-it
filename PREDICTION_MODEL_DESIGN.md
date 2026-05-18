# Prediction Model — Technical Design

*The core deliverable of the project. A notebook-provable cash-flow forecasting engine that answers "can I afford it?" with calibrated confidence. This document specifies the pipeline, the evaluation methodology, and the build path.*

---

## 1. Purpose and scope

The prediction engine takes a household's transaction history across all connected accounts and produces a forward projection of their cash position — and, against that projection, answers point-in-time affordability questions. Everything else in the product (the UI, the conversation layer, the alerts) is a presentation of this engine's output. If the engine is not accurate and well-calibrated, nothing built on top of it can be trusted.

The prototype's job is to prove the engine works *before* a single screen is designed. Proof means two things: the projections are accurate enough to be useful, and — more importantly — the confidence bands are honest, meaning an "80% likely" claim is true about 80% of the time. The deliverable is a notebook (or set of notebooks) plus a reusable Python package, not an app.

This document assumes synthetic data for development (Section 8), with a defined path to real anonymized data later.

## 2. Pipeline overview

The engine is a four-stage pipeline followed by a projection step:

The input is a normalized transaction ledger: dated, signed amounts (negative for debits, positive for credits) with merchant/description strings and account identifiers, plus current account balances. Stage one separates the recurring structure from the noise — detecting bills, subscriptions, and other regular obligations. Stage two handles income as a distinct, higher-stakes special case of recurrence. Stage three models everything left over — discretionary spending — as a statistical process rather than discrete events. Stage four assembles a forward projection by combining the known recurring schedule, the predicted income schedule, and many sampled discretionary trajectories into a distribution over future balances. The affordability query is then a question asked of that distribution.

The deliberate architectural choice, carried over from the PRD: the forecasting is **deterministic and statistical**, not LLM-driven. The language model's role is described in Section 7 and is strictly downstream of the numbers.

## 3. Stage 1 — Recurring and bill detection

The goal is to identify every transaction that belongs to a regular obligation, and to recover the *schedule* behind it: amount distribution, cadence, and next expected date.

The approach is grouping followed by periodicity testing. Transactions are first grouped by a normalized merchant key — lowercased, stripped of trailing transaction IDs and store numbers, collapsed to a stable stem — combined with amount similarity. Within each candidate group, the engine tests for regular cadence: are the inter-transaction gaps clustered around a recognizable period (weekly, biweekly, semimonthly, monthly, quarterly, annual)? A group that shows both a stable amount band and a regular cadence is promoted to a recurring schedule. Amount can vary (a utility bill is recurring but not fixed), so the test is cadence-regularity-dominant with amount treated as a distribution to learn rather than a constant to match.

The output is a set of recurring schedules, each with: a merchant key, an amount distribution (mean and spread), a cadence, a confidence score, and a next-expected-date. The leftover transactions — those that join no regular group — pass through to Stage 3.

Because synthetic data carries ground-truth `is_recurring` and `source_id` labels, this stage can be scored directly with precision and recall (Section 9).

## 4. Stage 2 — Income timing

Income is structurally a recurrence problem, but it gets its own stage because the cost of getting it wrong is asymmetric and large: the entire projection pivots on when money arrives. It is detected among credit transactions using the same grouping-plus-periodicity logic, but with income-specific handling.

The engine must classify the income regime. **Regular income** — a biweekly or semimonthly paycheck — is the easy case: detect the cadence, learn the amount distribution, project the next several pay dates with high confidence. **Irregular income** — gig work, commission, variable hours — is the hard case, and whether to support it in the prototype is an open question (Section 12); if included, it is modeled as a distribution over both gap length and amount rather than a fixed schedule, which materially widens the confidence bands downstream.

The output is a projected income schedule: a list of expected future credit events, each with a date distribution and an amount distribution.

## 5. Stage 3 — Discretionary spending forecast

After recurring obligations and income are removed, what remains is discretionary spending — groceries, dining, fuel, shopping, the long tail. This is not modeled as discrete predictable events; it is modeled as a *statistical process*.

The baseline model is a daily spending rate: the expected number of discretionary transactions per day and the amount distribution per transaction, learned from history. The first refinement is a day-of-week profile, because discretionary spending is strongly weekly — weekends differ from weekdays. Further refinements (category-level rates, month-phase effects, paycheck-proximity effects) are deliberately deferred; the prototype should establish that a simple, well-calibrated discretionary model beats a naive one before adding structure.

The output is not a point forecast. It is a generator: something the projection step can sample from repeatedly to produce many plausible discretionary trajectories.

## 6. Stage 4 — Forward projection and confidence

This is where the engine earns or loses trust. The projection is built by **Monte Carlo simulation**, not point estimation.

A single simulated trajectory starts from the current known balance and walks forward day by day to the horizon (a natural horizon is "through the next one or two paychecks"). On each day it applies the recurring obligations due that day (sampling each amount from its learned distribution), applies any income expected that day (sampling date and amount from their distributions), and applies a sampled day of discretionary spending from the Stage 3 generator. Running this thousands of times produces a distribution of balance trajectories. From that ensemble the engine reports percentile bands — a median path and, say, 10th/25th/75th/90th percentile envelopes — and, critically, the distribution of the *minimum* balance reached over the horizon.

The affordability query is then a clean question asked of this ensemble. "Can I afford $X on date D?" becomes: inject an additional $X debit on date D into every trajectory, and report the probability that the minimum balance over the horizon falls below the user's safety buffer. The answer is never a bare yes/no — it is that probability, the date the low point occurs, and the obligation driving it. A wrong "yes" is far more damaging than a wrong "no" (Section 9), so the decision threshold is deliberately conservative and the buffer is user-set.

## 7. The LLM's role

The language model is explicitly **not** the predictor. It sits downstream of the Monte Carlo output and does three things. It translates the numeric result into plain language — "yes, but you'll dip to about $40 next Thursday before payday because of your insurance autopay." It handles the natural-language conversation, so the user can ask in their own words. And it incorporates unstructured, one-off context the statistical model cannot know — "I have a $500 car repair coming Friday" — by turning that into a concrete scenario adjustment (an extra modeled debit) and re-running the projection. Keeping the math deterministic and the language model on conversation and explanation is both the correct engineering and the credibility story: the accuracy claim rests on the statistics, not on a model that might hallucinate a number.

## 8. Synthetic data strategy

Real anonymized transaction data is the eventual requirement, but it is also a chicken-and-egg problem: hard to obtain, and you cannot design the pipeline against data you do not have. Synthetic data resolves this and is genuinely the right first move, for three reasons. It provides **ground truth** — the generator knows exactly which transactions are recurring, what the true income schedule is, what the true future looks like — which makes the detection stages directly scoreable. It allows **deliberate edge cases** — irregular income, subscription-heavy households, thin files — to be generated on demand rather than waited for. And it makes the whole pipeline and evaluation harness developable today.

The synthetic generator (shipped in this foundation, see `prototype/synth/generator.py`) produces realistic households from configurable profiles, with labeled transactions and a recoverable ground-truth schedule. The path to real data — Plaid sandbox first, then a small set of consented real accounts — is a later milestone, and the eval harness is designed to run unchanged on real data (it simply loses the detection-eval layer that depends on synthetic labels).

## 9. Evaluation methodology

This is the most important section, because for this project the evaluation *is* the deliverable. There are three layers.

**Detection evaluation** uses the synthetic ground-truth labels. It scores Stage 1 and Stage 2 directly: precision and recall on which transactions are recurring, and accuracy of the recovered schedules (cadence correct, next-date within tolerance, amount distribution close). This layer only exists on synthetic data and is how the detection stages are tuned.

**Projection evaluation** is a backtest and works on real or synthetic data. Hold out the last N days of history, build the projection from everything before, then compare the projected balance trajectory to what actually happened. Point accuracy is measured with mean absolute error on the daily balance path — but point accuracy is the *lesser* metric here.

**Calibration** is the metric that matters most. A projection that is roughly accurate but honest about its uncertainty is far more valuable than a precise-looking one that is overconfident. Calibration is measured with a reliability check: for the stated P% predictive interval, what fraction of held-out actuals actually fall inside it? Plot stated coverage against empirical coverage across many backtested households and dates; a well-calibrated engine sits on the diagonal. An engine whose 80% bands only contain the truth 55% of the time is dangerous regardless of its median accuracy, because the product will tell users they are safe when they are not.

**Affordability-decision evaluation** is the end-to-end test that mirrors the actual product. Over many historical (date, hypothetical-spend) points, record what the engine would have answered — affordable or not — and whether an overdraft (or buffer breach) actually occurred within the horizon. The result is a confusion matrix, scored with deliberate asymmetry: a false "affordable" (told the user yes, they overdrafted) is weighted far more heavily than a false "not affordable." The headline number for the prototype is the false-affordable rate, driven as low as possible, with the conservatism of the decision threshold tuned against it.

## 10. Milestones

The build path runs in six steps. **M1** is the synthetic generator and exploratory analysis — shipped in this foundation; the next move is to run it and look at the data. **M2** is recurring and income detection, validated against synthetic ground truth until precision and recall are strong. **M3** is the discretionary spending model, established as beating a naive baseline. **M4** is the Monte Carlo projection and the affordability query. **M5** is the full evaluation harness and the first calibration report — the moment the engine is either proven or sent back. **M6** swaps synthetic data for real anonymized data and re-runs the harness.

## 11. Project structure

The foundation ships as `prototype/` inside this folder:

```
prototype/
  pyproject.toml          # uv-managed deps
  .gitignore
  README.md
  synth/
    generator.py          # synthetic household transaction generator — IMPLEMENTED
  cashflow/
    recurring.py          # Stage 1 — recurring & bill detection (stub)
    income.py             # Stage 2 — income timing (stub)
    discretionary.py      # Stage 3 — discretionary spend model (stub)
    projection.py         # Stage 4 — Monte Carlo projection + affordability query (stub)
  eval/
    backtest.py           # detection / projection / calibration / decision eval (stub)
  notebooks/
    01_explore_synthetic_data.py   # M1 EDA — runnable today
```

The generator is fully implemented. The `cashflow/` and `eval/` modules are specified stubs — typed signatures and detailed docstrings describing exactly what each must do — so the build can proceed module by module against the milestones above.

## 12. Open technical questions

Three questions should be resolved early. Whether to support **irregular income** in the prototype — it is a large, underserved segment, but it widens every confidence band and complicates Stage 2 materially; the prototype may be stronger proving the regular-income case cleanly first. What the right **projection horizon** is — "through the next two paychecks" is a reasonable default, but it interacts with how far the discretionary model can be trusted. And how to set the **safety buffer** — a fixed dollar amount, a function of the household's own volatility, or user-chosen — which is as much a product decision as a modeling one.
