"""Unit tests for viz.py: each plotting function is checked to produce a
non-empty PNG on a small synthetic table, independent of the real day's
data (data/features/features.parquet) -- these tests must pass even
before that file has been generated."""

import numpy as np
import pandas as pd

from model import feature_columns
from viz import (
    fit_xgboost_for_importance,
    plot_book_evolution,
    plot_drawdown_curve,
    plot_feature_importance,
    plot_ic_by_horizon,
    plot_imbalance_distribution,
    plot_pnl_curve,
    plot_predicted_vs_actual,
    plot_spread_over_time,
)


def _assert_nonempty_png(path: str) -> None:
    assert path.endswith(".png")
    with open(path, "rb") as f:
        content = f.read()
    assert len(content) > 0
    assert content[:8] == b"\x89PNG\r\n\x1a\n"  # PNG magic bytes


def _book_like_df(n: int) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    bid = 100.0 + rng.normal(size=n).cumsum() * 0.01
    ask = bid + rng.uniform(0.5, 1.5, size=n)
    df = pd.DataFrame({
        "best_bid": bid,
        "best_ask": ask,
        "spread": ask - bid,
        "queue_imbalance_w5": rng.uniform(-1, 1, size=n),
    })
    df.index = pd.MultiIndex.from_arrays(
        [range(1, n + 1), np.linspace(34200.0, 34260.0, n)], names=["event_id", "timestamp"]
    )
    return df


def test_plot_book_evolution_writes_png(tmp_path):
    df = _book_like_df(50)
    path = plot_book_evolution(df, str(tmp_path), n_events=20)
    _assert_nonempty_png(path)


def test_plot_spread_over_time_writes_png(tmp_path):
    df = _book_like_df(50)
    path = plot_spread_over_time(df, str(tmp_path))
    _assert_nonempty_png(path)


def test_plot_imbalance_distribution_writes_png(tmp_path):
    df = _book_like_df(50)
    path = plot_imbalance_distribution(df, str(tmp_path))
    _assert_nonempty_png(path)


def _model_ready_df(n: int) -> pd.DataFrame:
    """n rows with every feature_columns() column present, and
    label_h1 a perfectly learnable function of queue_imbalance_w5 -- same
    synthetic-signal pattern as checks/test_model.py, so fit_xgboost_for_importance
    trains against >1 label class."""
    rng = np.random.default_rng(0)
    df = pd.DataFrame({col: rng.normal(size=n) for col in feature_columns()})
    label = pd.Series(rng.choice([-1.0, 0.0, 1.0], size=n))
    df["queue_imbalance_w5"] = label
    df["label_h1"] = label
    return df


def test_fit_xgboost_for_importance_covers_every_feature_column():
    df = _model_ready_df(120)
    config = {"model": {"xgboost": {
        "n_estimators": 10, "max_depth": 2, "learning_rate": 0.3, "subsample": 1.0, "colsample_bytree": 1.0,
    }}}
    xgb = fit_xgboost_for_importance(df, config, "label_h1", train_range=range(0, 120))
    assert len(xgb.feature_importances_) == len(feature_columns())


def test_plot_feature_importance_writes_png(tmp_path):
    df = _model_ready_df(120)
    config = {"model": {"xgboost": {
        "n_estimators": 10, "max_depth": 2, "learning_rate": 0.3, "subsample": 1.0, "colsample_bytree": 1.0,
    }}}
    xgb = fit_xgboost_for_importance(df, config, "label_h1", train_range=range(0, 120))
    path = plot_feature_importance(xgb, str(tmp_path), horizon=1)
    _assert_nonempty_png(path)


def test_plot_predicted_vs_actual_writes_png(tmp_path):
    rng = np.random.default_rng(0)
    n = 200
    midprice = pd.Series(100.0 + rng.normal(size=n).cumsum() * 0.01)
    score = pd.Series(rng.normal(size=n))
    path = plot_predicted_vs_actual(score, midprice, horizon=5, out_dir=str(tmp_path))
    _assert_nonempty_png(path)


def test_plot_ic_by_horizon_writes_png(tmp_path):
    baseline_results = pd.DataFrame({
        "baseline": ["imbalance_only", "momentum", "imbalance_only", "momentum"],
        "horizon": [5, 5, 10, 10],
        "ic": [0.06, 0.02, 0.08, 0.01],
    })
    model_results = pd.DataFrame({
        "model": ["xgboost", "xgboost"],
        "horizon": [5, 10],
        "ic": [0.063, 0.062],
    })
    path = plot_ic_by_horizon(baseline_results, model_results, str(tmp_path))
    _assert_nonempty_png(path)


def _trades_df(pnls) -> pd.DataFrame:
    return pd.DataFrame({"net_pnl_dollars": pnls})


def test_plot_pnl_curve_writes_png(tmp_path):
    trades_by_mode = {
        "signal replay": _trades_df([1.0, -0.5, 2.0, 0.3]),
        "paper-trading": _trades_df([-10.0, -5.0, -20.0]),
    }
    path = plot_pnl_curve(trades_by_mode, str(tmp_path))
    _assert_nonempty_png(path)


def test_plot_drawdown_curve_writes_png(tmp_path):
    trades_by_mode = {
        "signal replay": _trades_df([1.0, -0.5, 2.0, 0.3]),
        "paper-trading": _trades_df([-10.0, -5.0, -20.0]),
    }
    path = plot_drawdown_curve(trades_by_mode, str(tmp_path))
    _assert_nonempty_png(path)
