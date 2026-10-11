"""Unit tests for baselines.py: each baseline's formula, hand-computed on
a small synthetic feature table."""

import pandas as pd

from baselines import (
    BASELINES,
    imbalance_only,
    last_direction,
    mean_reversion,
    momentum,
    spread_only,
)

DF = pd.DataFrame({
    "midprice": [100.0, 102.0, 101.0, 105.0, 104.0],
    "spread": [1.0, 1.2, 0.8, -0.5, 0.3],
    "queue_imbalance_w5": [0.1, 0.2, -0.1, 0.4, -0.3],
})


def test_last_direction_is_one_event_diff():
    out = last_direction(DF, h=10)
    assert out.iloc[1] == 102.0 - 100.0
    assert out.iloc[2] == 101.0 - 102.0
    assert pd.isna(out.iloc[0])


def test_momentum_is_h_event_diff():
    out = momentum(DF, h=2)
    assert out.iloc[2] == 101.0 - 100.0
    assert out.iloc[3] == 105.0 - 102.0
    assert pd.isna(out.iloc[0])
    assert pd.isna(out.iloc[1])


def test_mean_reversion_is_negative_momentum():
    out_mom = momentum(DF, h=2)
    out_rev = mean_reversion(DF, h=2)
    assert (out_rev.dropna() == -out_mom.dropna()).all()


def test_spread_only_passes_spread_through_unmodified():
    out = spread_only(DF, h=5)
    assert list(out) == list(DF["spread"])


def test_imbalance_only_passes_queue_imbalance_w5_through_unmodified():
    out = imbalance_only(DF, h=5)
    assert list(out) == list(DF["queue_imbalance_w5"])


def test_all_baselines_share_the_same_callable_signature():
    for name, fn in BASELINES.items():
        out = fn(DF, h=5)
        assert isinstance(out, pd.Series), name
        assert len(out) == len(DF), name
