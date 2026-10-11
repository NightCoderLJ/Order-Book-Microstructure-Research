"""
viz.py -- Phase 6 diagnostic plots for the full pipeline.

Takes in: features.py's output table (data/features/features.parquet)
with labels.py's labels added; evaluate.py's baseline/model IC tables;
one representative XGBoost fit (via model.py's helpers) for feature
importance; model.py's out-of-sample prediction series (via backtest.py's
generate_oos_predictions); and backtest.py's signal/paper-trading replay
trades.

Produces: one PNG per plot, written under config.yaml: viz.output_dir --
book evolution, spread over time, imbalance distribution, feature
importance, predicted vs. actual return, IC by horizon, PnL curve, and
drawdown curve -- so every number already reported in reference/ and README.md
has a corresponding picture a reviewer can open directly.

Why every plotting function takes already-computed data (not file paths
or config) rather than reaching into disk/model state itself: this keeps
each function pure and independently testable on small synthetic frames
(see checks/test_viz.py), and lets main.py (Phase 6's single pipeline
driver) reuse whatever it already computed for its own stage instead of
recomputing it a second time here.

main() is the exception (like every other stage's main()): it reloads
data.parquet from disk, refits/re-derives what it needs, and calls the
pure plotting functions -- so `python viz.py` works standalone, the same
way `python backtest.py` recomputes its own out-of-sample predictions
independently of model.py's own main().
"""

import os

import matplotlib

matplotlib.use("Agg")  # no display in this environment -- write PNGs only
import matplotlib.pyplot as plt
import pandas as pd
import yaml
from sklearn.utils.class_weight import compute_sample_weight

from evaluate import evaluate_baselines, evaluate_model, walk_forward_splits
from labels import add_labels, forward_return
from model import feature_columns, make_linear_model, make_predict_fold_fn, make_xgboost_model, normalize


def load_config(path: str = "config.yaml") -> dict:
    """Loads the small set of named assumptions this pipeline depends on."""
    with open(path) as f:
        return yaml.safe_load(f)


def _savefig(fig, out_dir: str, name: str) -> str:
    """Writes `fig` to `out_dir/name`, creating the directory if needed,
    and closes it (matplotlib figures are not garbage-collected
    automatically and this script generates several in one process)."""
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# -- 1. book evolution --


def plot_book_evolution(df: pd.DataFrame, out_dir: str, n_events: int) -> str:
    """Plots best_bid/best_ask over the first `n_events` rows of `df`
    (indexed by (event_id, timestamp), as produced by features.py).

    Why the first n_events, not the full day: reference/01_ingestion_and_book.md
    documents that this Level-10 reconstruction is genuinely uncrossed
    (best_bid < best_ask) only for its first ~1.3% of events and
    permanently crossed after that -- plotting a window spanning both
    regimes (config.yaml: viz.book_evolution_events) makes that
    documented transition visible, not just a printed count.
    """
    sample = df.iloc[:n_events]
    ts = sample.index.get_level_values("timestamp")

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(ts, sample["best_bid"], label="best_bid", linewidth=0.7)
    ax.plot(ts, sample["best_ask"], label="best_ask", linewidth=0.7)
    ax.set_xlabel("time (seconds since midnight)")
    ax.set_ylabel("price (raw units, dollars x price_scale)")
    ax.set_title(f"Book evolution: best bid/ask, first {n_events:,} events")
    ax.legend()
    return _savefig(fig, out_dir, "01_book_evolution.png")


# -- 2. spread over time --


def plot_spread_over_time(df: pd.DataFrame, out_dir: str) -> str:
    """Plots the instantaneous `spread` column over the full day.

    Why it looks mostly negative: this dataset's book is crossed on
    ~98.8% of events (reference/01_ingestion_and_book.md) -- this is a
    faithful picture of the reconstruction, not a plotting bug, and is
    exactly the finding backtest.py's assumed_half_spread_raw derivation
    is based on.
    """
    ts = df.index.get_level_values("timestamp")

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(ts, df["spread"], linewidth=0.3)
    ax.axhline(0.0, color="black", linewidth=0.5, linestyle="--")
    ax.set_xlabel("time (seconds since midnight)")
    ax.set_ylabel("spread (raw units)")
    ax.set_title("Spread over the full trading day")
    return _savefig(fig, out_dir, "02_spread_over_time.png")


# -- 3. imbalance distribution --


def plot_imbalance_distribution(df: pd.DataFrame, out_dir: str) -> str:
    """Histogram of `queue_imbalance_w5` -- baselines.py's imbalance_only
    baseline (Stage 3's strongest baseline) uses this column unmodified as
    its score, so its distribution is worth seeing directly."""
    values = df["queue_imbalance_w5"].dropna()

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(values, bins=100)
    ax.set_xlabel("queue_imbalance_w5")
    ax.set_ylabel("event count")
    ax.set_title("Queue imbalance distribution (5-event window)")
    return _savefig(fig, out_dir, "03_imbalance_distribution.png")


# -- 4. feature importance --


_LABEL_TO_CLASS_IDX = {-1.0: 0, 0.0: 1, 1.0: 2}


def fit_xgboost_for_importance(df: pd.DataFrame, config: dict, label_col: str, train_range: range):
    """Fits one XGBoost classifier on `df.iloc[train_range]` using
    model.py's own feature_columns()/normalize()/make_xgboost_model(), and
    returns it fitted -- purely so plot_feature_importance() has a real
    fitted model's .feature_importances_ to show.

    Why one extra fit here instead of reusing evaluate_model()'s output:
    evaluate_model() only returns aggregate IC/accuracy metrics, never the
    fitted model object itself (by design -- model.py deliberately doesn't
    persist any single fold's model, see model.py's docstring). This is a
    single representative fit for visualization, not a new evaluation
    claim -- the walk-forward IC comparison in reference/04_model_and_evaluation.md
    remains the actual evaluation.
    """
    cols = feature_columns()
    train_df = df.iloc[train_range.start:train_range.stop]
    valid = train_df[cols].notna().all(axis=1) & train_df[label_col].notna()
    x_train_raw = train_df.loc[valid, cols]
    y_train = train_df.loc[valid, label_col]

    x_train, _ = normalize(x_train_raw, x_train_raw)
    weights = compute_sample_weight("balanced", y_train)
    y_fit = y_train.map(_LABEL_TO_CLASS_IDX)

    xgb = make_xgboost_model(config)
    xgb.fit(x_train, y_fit, sample_weight=weights)
    return xgb


def plot_feature_importance(model, out_dir: str, horizon: int) -> str:
    """Horizontal bar chart of a fitted XGBoost model's
    .feature_importances_ (gain-based, normalized to sum to 1) against
    model.feature_columns(), sorted ascending so the largest bar is on
    top."""
    importances = pd.Series(model.feature_importances_, index=feature_columns()).sort_values()

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.barh(importances.index, importances.values)
    ax.set_xlabel("XGBoost feature importance (gain, normalized)")
    ax.set_title(f"Feature importance -- xgboost @ h={horizon}, last fold's train block")
    return _savefig(fig, out_dir, "04_feature_importance.png")


# -- 5. predicted vs. actual return --


def plot_predicted_vs_actual(oos_score: pd.Series, midprice: pd.Series, horizon: int, out_dir: str) -> str:
    """Hexbin density plot of an out-of-sample prediction score against
    the realized forward return it's trying to predict (labels.forward_return),
    restricted to rows where both are defined.

    Why hexbin, not a scatter: this dataset has ~400k events, and IC
    (Stage 4's actual metric) is a rank correlation over the *whole*
    series, not something visible from a handful of points -- hexbin shows
    the full joint density instead of an overplotted point cloud.
    """
    ret = forward_return(midprice, horizon)
    valid = oos_score.notna() & ret.notna()

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.hexbin(oos_score[valid], ret[valid], gridsize=60, mincnt=1, cmap="viridis")
    ax.axhline(0.0, color="white", linewidth=0.5, linestyle="--")
    ax.axvline(0.0, color="white", linewidth=0.5, linestyle="--")
    ax.set_xlabel(f"predicted score (P(up) - P(down)), h={horizon}")
    ax.set_ylabel(f"realized forward return, h={horizon}")
    ax.set_title("Predicted score vs. realized forward return (out-of-sample)")
    return _savefig(fig, out_dir, "05_predicted_vs_actual_return.png")


# -- 6. IC by horizon --


def plot_ic_by_horizon(baseline_results: pd.DataFrame, model_results: pd.DataFrame, out_dir: str) -> str:
    """Grouped bar chart of IC by horizon: the best baseline per horizon
    (from evaluate_baselines()'s output) next to every model in
    model_results (evaluate_model()'s output for one or more models) --
    the same comparison model.py's main() prints as a table, in picture
    form."""
    best_baseline = baseline_results.loc[baseline_results.groupby("horizon")["ic"].idxmax()]
    best_baseline = best_baseline[["horizon", "baseline", "ic"]].rename(columns={"baseline": "series"})

    model_rows = model_results[["horizon", "model", "ic"]].rename(columns={"model": "series"})
    combined = pd.concat([best_baseline, model_rows], ignore_index=True)

    horizons = sorted(combined["horizon"].unique())
    series_names = combined["series"].unique()
    width = 0.8 / len(series_names)

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, series in enumerate(series_names):
        sub = combined[combined["series"] == series].set_index("horizon").reindex(horizons)
        positions = [h_idx + i * width for h_idx in range(len(horizons))]
        ax.bar(positions, sub["ic"], width=width, label=series)

    ax.set_xticks([h_idx + width * (len(series_names) - 1) / 2 for h_idx in range(len(horizons))])
    ax.set_xticklabels([str(h) for h in horizons])
    ax.axhline(0.0, color="black", linewidth=0.5)
    ax.set_xlabel("forecast horizon (events)")
    ax.set_ylabel("Spearman IC")
    ax.set_title("IC by horizon: best baseline vs. models")
    ax.legend()
    return _savefig(fig, out_dir, "06_ic_by_horizon.png")


# -- 7 & 8. PnL and drawdown curves --


def plot_pnl_curve(trades_by_mode: dict, out_dir: str) -> str:
    """Cumulative net PnL over trade sequence, one line per mode in
    `trades_by_mode` (e.g. {"signal": signal_trades_df, "paper": paper_trades_df},
    each with a net_pnl_dollars column as produced by backtest.run_replay()).

    One subplot per mode rather than one shared axis: signal replay's
    +$3,448 and paper-trading's -$1.5M (reference/05_replay_and_costs.md) are
    ~3-4 orders of magnitude apart, so a shared y-axis would flatten the
    signal-replay curve to a visually invisible line.
    """
    fig, axes = plt.subplots(len(trades_by_mode), 1, figsize=(9, 3.5 * len(trades_by_mode)), squeeze=False)
    for ax, (mode, trades) in zip(axes[:, 0], trades_by_mode.items()):
        cum_pnl = trades["net_pnl_dollars"].cumsum()
        ax.plot(range(len(cum_pnl)), cum_pnl, linewidth=0.8)
        ax.axhline(0.0, color="black", linewidth=0.5, linestyle="--")
        ax.set_xlabel("trade sequence")
        ax.set_ylabel("cumulative PnL ($)")
        ax.set_title(f"Cumulative PnL -- {mode}")
    fig.tight_layout()
    return _savefig(fig, out_dir, "07_pnl_curve.png")


def plot_drawdown_curve(trades_by_mode: dict, out_dir: str) -> str:
    """Running drawdown (cumulative PnL minus its running max) over trade
    sequence, one subplot per mode -- same magnitude reasoning as
    plot_pnl_curve()."""
    fig, axes = plt.subplots(len(trades_by_mode), 1, figsize=(9, 3.5 * len(trades_by_mode)), squeeze=False)
    for ax, (mode, trades) in zip(axes[:, 0], trades_by_mode.items()):
        cum_pnl = trades["net_pnl_dollars"].cumsum()
        running_max = cum_pnl.cummax()
        drawdown = cum_pnl - running_max
        ax.fill_between(range(len(drawdown)), drawdown, 0.0, step=None, alpha=0.6)
        ax.set_xlabel("trade sequence")
        ax.set_ylabel("drawdown ($)")
        ax.set_title(f"Drawdown -- {mode}")
    fig.tight_layout()
    return _savefig(fig, out_dir, "08_drawdown_curve.png")


def main():
    # Deferred imports: backtest.py imports evaluate.py/model.py at module
    # load time, so importing it back at viz.py's own module load time
    # would work but is only needed by this script entry point -- same
    # "only main() needs it" convention model.py/backtest.py already use
    # for their own cross-module imports.
    from backtest import generate_oos_predictions, run_replay
    from ingest import iter_events

    config = load_config()
    out_dir = config["viz"]["output_dir"]
    horizons = config["labels"]["horizons"]
    n_splits = config["evaluate"]["n_splits"]
    bt = config["backtest"]

    df = pd.read_parquet(config["features"]["output_file"])
    df = add_labels(df, horizons, config["labels"]["epsilon"])

    paths = []

    print("plotting book evolution / spread / imbalance...")
    paths.append(plot_book_evolution(df, out_dir, config["viz"]["book_evolution_events"]))
    paths.append(plot_spread_over_time(df, out_dir))
    paths.append(plot_imbalance_distribution(df, out_dir))

    print("fitting one representative model for feature importance...")
    purge = max(horizons)
    folds = walk_forward_splits(len(df), n_splits, purge)
    last_train_range, _ = folds[-1]
    xgb = fit_xgboost_for_importance(df, config, f"label_h{bt['horizon']}", last_train_range)
    paths.append(plot_feature_importance(xgb, out_dir, bt["horizon"]))

    print("generating out-of-sample predictions for predicted-vs-actual...")
    fold_fn = make_predict_fold_fn(config, make_xgboost_model, feature_columns(), uses_class_index=True)
    oos = generate_oos_predictions(df, bt["horizon"], n_splits, fold_fn)
    paths.append(plot_predicted_vs_actual(oos["score"], df["midprice"], bt["horizon"], out_dir))

    print("scoring baselines and models for IC-by-horizon...")
    baseline_results = evaluate_baselines(df, horizons, n_splits)
    linear_fn = make_predict_fold_fn(config, make_linear_model, feature_columns(), uses_class_index=False)
    xgb_fn = make_predict_fold_fn(config, make_xgboost_model, feature_columns(), uses_class_index=True)
    model_results = pd.concat([
        evaluate_model(df, horizons, n_splits, linear_fn, "logistic_regression"),
        evaluate_model(df, horizons, n_splits, xgb_fn, "xgboost"),
    ], ignore_index=True)
    paths.append(plot_ic_by_horizon(baseline_results, model_results, out_dir))

    print("replaying signal and paper-trading modes for PnL/drawdown...")
    trades_by_mode = {}
    for mode in ("signal", "paper"):
        event_chunks = iter_events(config["data"]["message_file"], chunk_size=config["ingest"]["chunk_size"])
        result = run_replay(config, oos.reset_index(drop=True), mode, event_chunks)
        label = "signal replay" if mode == "signal" else f"paper-trading ({bt['order_type']})"
        trades_by_mode[label] = result["trades"]
    paths.append(plot_pnl_curve(trades_by_mode, out_dir))
    paths.append(plot_drawdown_curve(trades_by_mode, out_dir))

    print()
    print(f"wrote {len(paths)} plots to {out_dir}/:")
    for path in paths:
        print(f"  {path}")


if __name__ == "__main__":
    main()
