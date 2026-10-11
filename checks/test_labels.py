"""Unit tests for labels.py: hand-computed forward-return and label checks
on a synthetic midprice series."""

import numpy as np
import pandas as pd

from labels import add_labels, forward_return, label_horizon

# midprice moves by +1%, -3%, then holds flat for the rest -- picked so
# h=1 forward returns land cleanly above/below/inside an epsilon=0.02 band.
MIDPRICE = pd.Series([100.0, 101.0, 98.0, 98.0, 98.0, 98.0])


def test_forward_return_formula():
    ret = forward_return(MIDPRICE, h=1)
    # (101 - 100) / 100
    assert ret.iloc[0] == (101.0 / 100.0) - 1.0
    # (98 - 101) / 101
    assert ret.iloc[1] == (98.0 / 101.0) - 1.0


def test_forward_return_is_nan_for_last_h_rows():
    ret = forward_return(MIDPRICE, h=2)
    assert ret.iloc[-2:].isna().all()
    assert ret.iloc[:-2].notna().all()


def test_label_horizon_directional_and_flat_with_threshold():
    # h=1, epsilon=0.02: row0 return=+0.01 (< eps -> flat), row1
    # return=-0.0297 (< -eps -> -1), row2 return=0 (flat).
    label = label_horizon(MIDPRICE, h=1, epsilon=0.02)
    assert label.iloc[0] == 0.0
    assert label.iloc[1] == -1.0
    assert label.iloc[2] == 0.0


def test_label_horizon_directional_with_tight_epsilon():
    # same series, epsilon=0.005: row0 return=+0.01 now exceeds eps -> +1.
    label = label_horizon(MIDPRICE, h=1, epsilon=0.005)
    assert label.iloc[0] == 1.0


def test_label_horizon_nan_at_end_of_series():
    label = label_horizon(MIDPRICE, h=2, epsilon=0.02)
    assert label.iloc[-2:].isna().all()


def test_add_labels_adds_one_column_per_horizon():
    df = pd.DataFrame({"midprice": MIDPRICE})
    out = add_labels(df, horizons=[1, 2], epsilon=0.02)

    assert "label_h1" in out.columns
    assert "label_h2" in out.columns
    # add_labels must not mutate the caller's DataFrame in place.
    assert "label_h1" not in df.columns
