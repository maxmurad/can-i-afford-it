"""Cash-flow prediction engine — the four-stage pipeline.

Stage 1  recurring.py     recurring & bill detection
Stage 2  income.py        income timing
Stage 3  discretionary.py discretionary spending model
Stage 4  projection.py    Monte Carlo projection + affordability query

See ../../PREDICTION_MODEL_DESIGN.md for the full design.
"""
