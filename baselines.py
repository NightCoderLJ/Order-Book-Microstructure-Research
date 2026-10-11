"""
baselines.py -- simple, non-fitted price-movement baselines that the
Phase 4 model will be compared against.

Takes in: features.py's output table (data/features/features.parquet, or
any DataFrame with `midprice`, `spread`, and `queue_imbalance_w5`
columns), one row per event, in event order.

Produces: for a given forecast horizon h, a continuous predicted score per
event, using only information available at or before that event (no
lookahead) -- evaluate.py ranks these scores against the realized forward
return for IC, and their sign against the label for accuracy.

Every baseline shares the same `fn(df, h) -> pd.Series` signature (even
the three that ignore h) so evaluate.py can loop over BASELINES generically
for every horizon without special-casing any one of them.
"""

import pandas as pd


def last_direction(df: pd.DataFrame, h: int) -> pd.Series:
    """Predicts that the most recent single-event price move continues:
    score = midprice(t) - midprice(t-1). Ignores h -- it only ever looks
    one event back, regardless of how far ahead it's being scored."""
    return df["midprice"].diff(1)


def momentum(df: pd.DataFrame, h: int) -> pd.Series:
    """Predicts trend continuation: score = midprice(t) - midprice(t-h),
    the same h-event window as the forecast horizon it's being evaluated
    on -- i.e. "the realized return over the last h events repeats over
    the next h events." This is the momentum baseline the model's IC
    comparison is defined against."""
    return df["midprice"].diff(h)


def mean_reversion(df: pd.DataFrame, h: int) -> pd.Series:
    """Predicts trend reversal: the negative of momentum()'s score -- "the
    realized return over the last h events reverses over the next h"."""
    return -df["midprice"].diff(h)


def spread_only(df: pd.DataFrame, h: int) -> pd.Series:
    """Uses the current bid-ask spread, unmodified, as the predicted
    score. Ignores h. A single-feature control: this dataset's Level-10
    depth truncation crosses the book on ~99% of events (reference/01), so
    spread is frequently negative and may carry real directional signal
    here rather than being a pure noise control -- see
    reference/03_labels_and_baselines.md."""
    return df["spread"]


def imbalance_only(df: pd.DataFrame, h: int) -> pd.Series:
    """Uses the smallest-window (5-event) queue imbalance, unmodified, as
    the predicted score -- a heavier bid queue (positive imbalance)
    predicts the price rises, per the queue-imbalance literature. Ignores
    h; queue_imbalance_w5 is already backward-looking over its own trailing
    5 events regardless of the forecast horizon."""
    return df["queue_imbalance_w5"]


BASELINES = {
    "last_direction": last_direction,
    "momentum": momentum,
    "mean_reversion": mean_reversion,
    "spread_only": spread_only,
    "imbalance_only": imbalance_only,
}
