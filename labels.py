"""
labels.py -- short-horizon price-movement classification labels, computed
from the midprice series that features.py already produces.

Takes in: features.py's output table (data/features/features.parquet, or
any DataFrame with a `midprice` column, one row per event, in event
order -- event order matters since horizons are event counts, not time).

Produces: one label column per forecast horizon in config.yaml:
labels.horizons -- `label_h{h}` in {-1.0, 0.0, +1.0, NaN} -- where
y = sign(midprice(t+h) - midprice(t)), unless that move is smaller than
the relative threshold config.yaml: labels.epsilon, in which case y = 0
("flat"). NaN for the last h events of the day, where midprice(t+h)
doesn't exist yet.
"""

import pandas as pd
import yaml


def load_config(path: str = "config.yaml") -> dict:
    """Loads the small set of named assumptions this pipeline depends on."""
    with open(path) as f:
        return yaml.safe_load(f)


def forward_return(midprice: pd.Series, h: int) -> pd.Series:
    """Relative forward return over h events: midprice(t+h) / midprice(t) - 1.

    midprice: per-event midprice series, in event order (row t is event t,
    row t+h is h events later -- an event-count lookahead, not time-based,
    consistent with features.py's event-count windows).

    Why relative, not raw price difference: a fixed epsilon threshold on
    the *relative* return is comparable across the day even though the
    stock's price level (and therefore the raw tick-to-tick jump size)
    can drift. Used both to build labels below and as evaluate.py's
    continuous IC target, so both are scored against the exact same
    quantity.

    NaN for the last h events of the day, where midprice(t+h) does not
    exist yet -- not an error, just no ground truth available there.
    """
    return midprice.shift(-h) / midprice - 1.0


def label_horizon(midprice: pd.Series, h: int, epsilon: float) -> pd.Series:
    """One horizon's classification label: +1 if the h-event forward return
    exceeds +epsilon, -1 if it's below -epsilon, 0 ("flat") otherwise, NaN
    where forward_return() is NaN (end-of-day, no ground truth yet).

    Why a threshold instead of a bare sign(): a bare sign() would label
    every microscopic, noise-level return as directional, which floods the
    +1/-1 classes with moves too small to be tradeable. epsilon carves out
    an explicit "flat" class for those -- see config.yaml: labels.epsilon
    and reference/03_labels_and_baselines.md for how its value was chosen.
    """
    ret = forward_return(midprice, h)
    label = pd.Series(0.0, index=midprice.index)
    label[ret > epsilon] = 1.0
    label[ret < -epsilon] = -1.0
    label[ret.isna()] = float("nan")
    return label


def add_labels(df: pd.DataFrame, horizons: list, epsilon: float) -> pd.DataFrame:
    """Adds one `label_h{h}` column per horizon in `horizons` to a copy of
    `df` (which must have a `midprice` column), for every h in
    config.yaml: labels.horizons -- so all downstream evaluation is done on
    labels built from the exact same midprice series."""
    df = df.copy()
    for h in horizons:
        df[f"label_h{h}"] = label_horizon(df["midprice"], h, epsilon)
    return df


def main():
    config = load_config()
    df = pd.read_parquet(config["features"]["output_file"])
    df = add_labels(df, config["labels"]["horizons"], config["labels"]["epsilon"])

    print(f"rows: {len(df):,}")
    print(f"epsilon: {config['labels']['epsilon']}")
    for h in config["labels"]["horizons"]:
        counts = df[f"label_h{h}"].value_counts(dropna=True).sort_index()
        total = counts.sum()
        print(f"label_h{h} distribution: "
              f"{ {k: f'{v:,} ({v/total:.1%})' for k, v in counts.items()} }")


if __name__ == "__main__":
    main()
