# Order Book Microstructure Research

A research pipeline that reconstructs a limit order book from raw L2
market-data messages, extracts microstructure features, and evaluates a
short-horizon price-movement forecast against momentum baselines.

**Status: all six phases are complete** -- ingestion + book
reconstruction, microstructure features, labels + baselines, model +
evaluation, replay/execution realism, and visualization + a single
end-to-end pipeline driver.

## One sentence each

- **Dataset**: LOBSTER Level-10 sample data for AAPL, 2012-06-21, full
  continuous trading session (400,391 events).
- **Reconstruction logic**: `book.py`'s `OrderBook` applies one canonical
  event at a time (add / partial cancel / full cancel / visible execution
  / hidden execution / halt), maintaining sorted bid/ask price levels and
  an `order_id -> order` map, with O(1)-per-event invariant checks
  (non-negative sizes, no crossed book, best-bid/ask consistency) that are
  logged and counted rather than silently dropped.
- **Features**: `features.py` replays events through `OrderBook` and emits
  one row per event -- top-of-book (best bid/ask, midprice, spread) plus
  microstructure signals (OFI, queue imbalance, microprice, weighted
  microprice, liquidity pressure, arrival/cancel rate, trade imbalance)
  aggregated over trailing 5/10/20/50-event windows, written to Parquet.
- **Labels**: `labels.py` computes `sign(midprice(t+h) - midprice(t))`
  with a relative-return flat-band `epsilon`, for h in 5/10/100 events.
- **Baseline**: `baselines.py` implements five non-fitted heuristics
  (last-direction, momentum, mean-reversion, spread-only,
  imbalance-only); `imbalance_only` has the strongest IC of the five.
- **Evaluation**: `evaluate.py` splits the day into walk-forward
  (purged), time-based folds and reports Spearman IC and directional
  accuracy per (baseline, horizon) -- see
  [reference/03_labels_and_baselines.md](reference/03_labels_and_baselines.md).
- **Model**: `model.py` trains a logistic regression and an XGBoost
  classifier on the 5 named microstructure feature groups, retrained
  fresh on every walk-forward fold with train-set-only normalization;
  neither clearly beats Stage 3's strongest baseline (`imbalance_only`,
  IC 0.05-0.08) except a near-tie at the shortest horizon -- see
  [reference/04_model_and_evaluation.md](reference/04_model_and_evaluation.md) for
  the full IC/accuracy/precision/recall/AUC comparison and a per-feature-
  group ablation.
- **Replay/execution realism**: `backtest.py` replays the full day event
  by event, injecting `model.py`'s out-of-sample (walk-forward) XGBoost
  predictions, in a no-cost "signal replay" mode and a cost-aware
  "paper-trading" mode (latency, fees, slippage, and -- for passive
  orders -- queue-position loss); the raw signal has a small positive
  edge (+$3,448/day, Sharpe 201) that a realistic spread-crossing cost
  overwhelms by ~460x (aggressive paper-trading loses on every trade) --
  see [reference/05_replay_and_costs.md](reference/05_replay_and_costs.md) for the
  full cost breakdown and a data-quality finding (this Level-10 sample's
  book is only genuinely uncrossed for its first ~1.3%) that shaped the
  cost model.
- **Visualization**: `viz.py` renders 8 plots (book evolution, spread over
  time, imbalance distribution, XGBoost feature importance, predicted-vs-
  actual return, IC by horizon, PnL curve, drawdown curve) from the exact
  tables/series above to `reports/figures/` -- see
  [reference/06_visualization.md](reference/06_visualization.md) and "Plots" below.
  `main.py` runs every stage above in dependency order with one command
  (`python main.py`), so the whole pipeline is reproducible from
  `config.yaml` and `data/raw/*.csv` alone.

## Setup

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

(A dataclass `slots=True` used in `ingest.py`/`book.py` requires Python
3.10+; the system default `python3` may be older.)

### Getting the data

`data/raw/*.csv` is not committed to version control (see `.gitignore`).
LOBSTER's sample-data page now gates downloads behind a request form, but
the same AAPL 2012-06-21 Level-10 sample used here is mirrored at
[huggingface.co/datasets/totalorganfailure/lobster-data](https://huggingface.co/datasets/totalorganfailure/lobster-data).
Download the two files under `LOBSTER_SampleFile_AAPL_2012-06-21_10/` into
`data/raw/`:

```
data/raw/AAPL_2012-06-21_34200000_57600000_message_10.csv
data/raw/AAPL_2012-06-21_34200000_57600000_orderbook_10.csv
```

## Running it

```bash
.venv/bin/pytest              # unit tests + full-day reconciliation (~5s)
.venv/bin/python main.py      # the whole pipeline, one command (~6-7 min measured end-to-end, see below)

# or run each stage individually, in this order (each reads the previous
# stage's output file):
.venv/bin/python replay.py    # full-day replay: throughput + event counts (~1.4s)
.venv/bin/python features.py  # full-day feature table -> data/features/features.parquet (~4s)
.venv/bin/python labels.py    # label class balance per horizon (<1s)
.venv/bin/python evaluate.py  # baseline IC/accuracy table (walk-forward test folds) (~1s)
.venv/bin/python model.py     # model vs. best-baseline IC + feature-group ablation (~4 min: retrains ~180 models)
.venv/bin/python backtest.py  # signal-replay vs. paper-trading PnL/Sharpe (~15s: retrains xgboost on 5 folds)
.venv/bin/python viz.py       # 8 plots -> reports/figures/ (~2 min: retrains xgboost, replays the day twice)
```

## Layout

```
config.yaml       named assumptions (data paths, price scale, checkpoint interval, chunk size, feature windows/depth, label horizons/epsilon, walk-forward splits)
ingest.py         streams LOBSTER message/orderbook CSVs into canonical Event/OrderbookSnapshot records
book.py           OrderBook: event-by-event book reconstruction, invariants, snapshots
features.py       microstructure features (OFI, queue imbalance, microprice, spread, liquidity pressure) over event windows
labels.py         short-horizon price-movement classification labels (sign of forward return past an epsilon flat-band)
baselines.py      five non-fitted heuristics: last-direction, momentum, mean-reversion, spread-only, imbalance-only
evaluate.py       walk-forward (purged) time split; scores baselines'/model's IC, accuracy, precision/recall/AUC on the test folds
model.py          linear (logistic regression) and XGBoost classifiers, retrained per walk-forward fold with train-set-only normalization; feature-group ablation
backtest.py       event-driven replay injecting model.py's out-of-sample predictions; signal-only vs. cost-aware paper-trading (latency, fees, slippage, queue position) PnL/Sharpe/fill-rate comparison
replay.py         driver: runs a full day through ingest+book, reports throughput and reconciliation
viz.py            8 diagnostic plots (book evolution, spread, imbalance distribution, feature importance, predicted-vs-actual return, IC by horizon, PnL curve, drawdown curve) -> reports/figures/
main.py           single-command driver: runs every stage above in dependency order, from config.yaml alone
checks/           unit tests (add/cancel/execute/crossed-book edge cases, feature formulas, labels, baselines, evaluation, model, replay/costs, plots, pipeline ordering) + full-day reconciliation
reference/        one page per pipeline stage (see reference/01_ingestion_and_book.md, reference/02_features.md, reference/03_labels_and_baselines.md, reference/04_model_and_evaluation.md, reference/05_replay_and_costs.md, reference/06_visualization.md)
data/raw/         LOBSTER input CSVs (gitignored, see "Getting the data" above)
reports/figures/  PNGs written by viz.py (gitignored, regenerated by running python viz.py)
```

## Plots

`python viz.py` (or `python main.py`, which calls it as its last stage)
writes 8 PNGs to `reports/figures/`:

| file | shows |
|---|---|
| `reports/figures/01_book_evolution.png` | best bid/ask over the first 8,000 events -- spans both the day's brief genuinely-uncrossed opening window and the point it becomes permanently crossed |
| `reports/figures/02_spread_over_time.png` | instantaneous spread over the full day |
| `reports/figures/03_imbalance_distribution.png` | histogram of `queue_imbalance_w5` (the strongest Stage 3 baseline's raw input) |
| `reports/figures/04_feature_importance.png` | XGBoost gain-based feature importance, `backtest.model` @ `backtest.horizon` |
| `reports/figures/05_predicted_vs_actual_return.png` | out-of-sample predicted score vs. realized forward return (hexbin density) |
| `reports/figures/06_ic_by_horizon.png` | best baseline vs. logistic regression vs. XGBoost, IC per horizon |
| `reports/figures/07_pnl_curve.png` | cumulative PnL, signal replay vs. paper-trading |
| `reports/figures/08_drawdown_curve.png` | running drawdown, same two modes |

See [reference/06_visualization.md](reference/06_visualization.md) for what each plot
is built from and why.

## Benchmark

Full trading day, 400,391 events: **317,609 events/sec** (1.26s). See
[reference/01_ingestion_and_book.md](reference/01_ingestion_and_book.md) for the
full writeup, including a documented limitation of LOBSTER's Level-10
sample tier that explains the (expected, non-zero) invariant counts.
