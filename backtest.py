"""
backtest.py -- Phase 5: replays the reconstructed order book with the
Phase 4 model's out-of-sample predictions injected event-by-event, under
named cost/latency assumptions (config.yaml: backtest.*), in two modes:

  - "signal" replay: marks each directional prediction to the midprice
    `backtest.horizon` events later, with no execution simulation at all
    (no latency, fees, slippage, or fills) -- isolates the model's raw
    prediction quality from any execution assumption.
  - "paper" (paper-trading) replay: converts each prediction into an
    order (config.yaml: backtest.order_type, aggressive or passive),
    simulates its fill against the live book -- latency delay,
    spread-crossing, slippage, a fixed fee, and, for passive orders,
    queue-position loss -- then marks filled positions to midprice the
    same `backtest.horizon` events after the signal, so the two modes
    are directly comparable.

Takes in: features.py's output table (data/features/features.parquet,
with labels.py's label_h{horizon} column added) for out-of-sample
predictions and midprices, plus a fresh raw event stream (see ingest.py)
replayed through a new OrderBook (see book.py) for paper-trading's fill
simulation -- the parquet's windowed columns don't carry the per-event
detail (exact executed size at one resting price) needed to simulate
queue position, so paper-trading mode re-walks the raw messages once.

Produces: one row per realized (closed) trade for the requested mode,
plus a metrics dict (PnL, Sharpe, win rate, max drawdown, fill rate,
slippage-adjusted edge) via compute_metrics(), computed identically for
both modes. Running `python backtest.py` prints both summaries.

Why predictions are precomputed, not fit live: Phase 4 (model.py/
evaluate.py) already established walk-forward retraining as the only
leak-free way to score this model. generate_oos_predictions() reuses
evaluate.py's identical folds to build one full-day, leak-free
prediction series *before* the replay starts, then "injects" it event by
event during the loop below -- there is no live/incremental fitting
here, by design.
"""

from collections import deque
from typing import Dict, List

import pandas as pd
import yaml

from book import OrderBook
from evaluate import walk_forward_splits
from ingest import iter_events
from labels import add_labels


def load_config(path: str = "config.yaml") -> dict:
    """Loads the small set of named assumptions this pipeline depends on."""
    with open(path) as f:
        return yaml.safe_load(f)


# -- out-of-sample prediction series --


def generate_oos_predictions(df: pd.DataFrame, horizon: int, n_splits: int, predict_fold_fn) -> pd.DataFrame:
    """Builds one full-day, leak-free (score, pred_class) series for
    `horizon` by running the same walk-forward folds evaluate_model()
    uses (see evaluate.walk_forward_splits) and concatenating each
    fold's held-out test predictions, instead of aggregating them into
    IC/accuracy metrics the way evaluate_model() does.

    predict_fold_fn: a closure from model.make_predict_fold_fn(), e.g.
    make_predict_fold_fn(config, make_xgboost_model, feature_columns(), uses_class_index=True).

    Rows before the first fold's test window starts (the initial
    train-only block) get NaN -- there is no leak-free way to score them
    without training on future data, so the replay simply has no signal
    to trade there.
    """
    label_col = f"label_h{horizon}"
    folds = walk_forward_splits(len(df), n_splits, purge=horizon)

    score = pd.Series(float("nan"), index=df.index)
    pred_class = pd.Series(float("nan"), index=df.index)
    for train_range, test_range in folds:
        fold_score, fold_pred_class, _ = predict_fold_fn(df, train_range, test_range, label_col)
        score.iloc[test_range.start:test_range.stop] = fold_score.to_numpy()
        pred_class.iloc[test_range.start:test_range.stop] = fold_pred_class.to_numpy()
    return pd.DataFrame({"score": score, "pred_class": pred_class})


# -- event-driven replay engine --


def run_replay(config: dict, oos: pd.DataFrame, mode: str, event_chunks) -> Dict:
    """Streams `event_chunks` (chunks of ingest.Event, e.g. from
    ingest.iter_events -- passed in, not read from config directly, so
    this can be driven by a synthetic event stream in tests) through a
    fresh OrderBook one event at a time, injecting oos.pred_class[i] at
    each event i (0-indexed row position, aligned to sequence_id - 1),
    and either:

      mode="signal": opens a position at the current midprice with no
        execution model at all, closing it `horizon` events later at the
        midprice then in effect.
      mode="paper": converts the signal into an order per
        config.yaml: backtest.order_type, subject to latency, fees,
        slippage (aggressive), and queue position (passive) -- see the
        module docstring -- before opening a position identically to
        signal mode once (and if) it fills.

    Returns {"trades": DataFrame of closed, filled trades (one row per
    trade: gen_idx, direction, entry_price_raw, exit_price_raw,
    fee_dollars, net_pnl_dollars), "total_signals": int}. total_signals
    is every event where oos.pred_class was non-flat, whether or not it
    ever became a trade -- compute_metrics() uses it as the fill-rate
    denominator.

    Why one engine for both modes instead of two: both need the exact
    same book-driven exit price at `horizon` events after entry, so
    sharing one event loop guarantees the two modes are scored on
    identical exits and differ only in how (and whether) the entry fills.
    """
    bt = config["backtest"]
    horizon = bt["horizon"]
    position_size = bt["position_size"]
    price_scale = config["data"]["price_scale"]
    fee_dollars = bt["fee_per_share"] * position_size
    slippage_raw = bt["slippage_ticks"] * bt["tick_size_raw"]
    half_spread_raw = bt["assumed_half_spread_raw"]
    latency_seconds = bt["latency_seconds"]
    order_type = bt["order_type"]

    pred_class = oos["pred_class"].to_numpy()

    open_positions: Dict[int, dict] = {}  # close_idx -> position
    pending_arrivals = deque()            # paper mode: orders in flight (latency)
    pending_fills: List[dict] = []        # paper mode, passive: orders resting, waiting on queue
    trades: List[dict] = []
    total_signals = 0

    book = OrderBook(checkpoint_interval=config["book"]["checkpoint_interval"])

    for chunk in event_chunks:
        for event in chunk:
            book.apply_event(event)
            i = event.sequence_id - 1
            ts = event.timestamp_exchange
            bb, ba = book.best_bid(), book.best_ask()

            # 1. close any position whose holding period ends here
            position = open_positions.pop(i, None)
            if position is not None and bb is not None and ba is not None:
                exit_price = (bb + ba) / 2.0
                gross_pnl_raw = position["direction"] * (exit_price - position["entry_price"]) * position_size
                trades.append({
                    "gen_idx": position["gen_idx"],
                    "direction": position["direction"],
                    "entry_price_raw": position["entry_price"],
                    "exit_price_raw": exit_price,
                    "fee_dollars": position["fee_dollars"],
                    "net_pnl_dollars": gross_pnl_raw / price_scale - position["fee_dollars"],
                })

            if mode == "paper":
                if order_type == "passive":
                    _update_pending_fills(pending_fills, event, book, open_positions, horizon,
                                           position_size, fee_dollars)

                while pending_arrivals and pending_arrivals[0]["arrival_time"] <= ts:
                    order = pending_arrivals.popleft()
                    _resolve_arrival(order, order_type, book, open_positions, pending_fills,
                                      horizon, fee_dollars, slippage_raw, half_spread_raw)

            # signal: a fresh, non-flat, out-of-sample prediction at this event
            pc = pred_class[i]
            if pd.notna(pc) and pc != 0.0:
                total_signals += 1
                direction = 1.0 if pc > 0 else -1.0
                if mode == "signal":
                    if bb is not None and ba is not None:
                        open_positions[i + horizon] = {
                            "gen_idx": i, "direction": direction,
                            "entry_price": (bb + ba) / 2.0, "fee_dollars": 0.0,
                        }
                else:
                    pending_arrivals.append({
                        "arrival_time": ts + latency_seconds, "direction": direction, "gen_idx": i,
                    })

    return {"trades": pd.DataFrame(trades), "total_signals": total_signals}


def _resolve_arrival(order: dict, order_type: str, book: OrderBook, open_positions: Dict[int, dict],
                      pending_fills: List[dict], horizon: int,
                      fee_dollars: float, slippage_raw: int, half_spread_raw: int) -> None:
    """Handles one order's arrival at the exchange (after its latency
    delay has elapsed), against the book's *current* state.

    aggressive: fills at the current midprice, moved `half_spread_raw +
    slippage_raw` against the order (buy pays midprice plus that much;
    sell receives midprice minus that much) -- an *assumed* spread-
    crossing cost (config.yaml: backtest.assumed_half_spread_raw) rather
    than the live best_bid/best_ask. Why: this LOBSTER Level-10
    reconstruction's best_bid/best_ask is only genuinely valid (bb < ba)
    on ~1.2% of the day's events (see reference/01_ingestion_and_book.md's
    `crossed_book` invariant count) -- for the other ~98.8%, one side can
    be a stale, out-of-money price from an order outside the tracked
    depth that was never cancelled, so crossing it directly would
    fabricate a fictitious risk-free trade rather than cost a real one.
    Declines to fill only if the book has no two-sided liquidity at all
    (midprice undefined).

    passive: joins the back of its own side's queue at the current best
    price, recording that price and the size already resting ahead of it
    (queue_ahead) so _update_pending_fills() can track when enough of
    the queue ahead has traded through. This does *not* need the same
    guard as aggressive: resting behind a real order and waiting for real
    executions never fabricates a trade -- a resting order queued behind
    a stale price simply never accumulates real executions and times out
    in _update_pending_fills(), which is a real (if uninteresting)
    outcome, not a fabricated one. If its own side is empty, there is no
    reference price to rest at, so it is dropped (rare -- only possible
    near the open or a halt).
    """
    direction = order["direction"]
    gen_idx = order["gen_idx"]
    bb, ba = book.best_bid(), book.best_ask()

    if order_type == "aggressive":
        if bb is None or ba is None:
            return  # no two-sided liquidity to compute a midprice from
        midprice = (bb + ba) / 2.0
        cost = half_spread_raw + slippage_raw
        entry_price = midprice + cost if direction > 0 else midprice - cost
        open_positions[gen_idx + horizon] = {
            "gen_idx": gen_idx, "direction": direction, "entry_price": entry_price, "fee_dollars": fee_dollars,
        }
    else:  # passive
        own_price = bb if direction > 0 else ba
        if own_price is None:
            return  # no resting liquidity on our own side to queue behind
        levels = book.bid_levels if direction > 0 else book.ask_levels
        queue_ahead = levels.get(own_price, 0)
        pending_fills.append({
            "price": own_price, "direction": direction, "gen_idx": gen_idx,
            "queue_ahead": queue_ahead, "cum_exec": 0, "deadline_idx": gen_idx + horizon,
        })


def _update_pending_fills(pending_fills: List[dict], event, book: OrderBook, open_positions: Dict[int, dict],
                           horizon: int, position_size: int, fee_dollars: float) -> None:
    """Advances every resting passive order's queue tracking by one event,
    in place, removing orders that fill, time out, or lose their queue
    position.

    Only EXECUTE_VISIBLE events count toward cum_exec: book.py treats
    EXECUTE_HIDDEN as a no-op against the displayed book (see book.py's
    _apply_execute_hidden), so a hidden execution does not consume a
    displayed resting order's queue position either. A pending order
    fills once cum_exec reaches the size that was resting ahead of it
    plus its own size (queue_ahead + position_size) at the same
    price/side. It is dropped, unfilled, if its price level disappears
    from the book entirely (the order it was queued behind was wiped out
    -- "queue-position loss") or if `horizon` events pass without a fill.
    """
    i = event.sequence_id - 1
    still_pending = []
    for pending in pending_fills:
        own_side = "B" if pending["direction"] > 0 else "A"

        if event.event_type == "EXECUTE_VISIBLE" and event.side == own_side and event.price == pending["price"]:
            pending["cum_exec"] += event.size

        if pending["cum_exec"] >= pending["queue_ahead"] + position_size:
            open_positions[pending["gen_idx"] + horizon] = {
                "gen_idx": pending["gen_idx"], "direction": pending["direction"],
                "entry_price": pending["price"], "fee_dollars": fee_dollars,
            }
            continue  # filled -- drop from pending_fills

        levels = book.bid_levels if pending["direction"] > 0 else book.ask_levels
        still_resting = pending["price"] in levels
        if not still_resting or i >= pending["deadline_idx"]:
            continue  # lost queue position, or timed out -- drop, unfilled

        still_pending.append(pending)

    pending_fills[:] = still_pending


def compute_metrics(trades: pd.DataFrame, total_signals: int, config: dict) -> dict:
    """Computes PnL, Sharpe, win rate, max drawdown, fill rate, and
    slippage-adjusted edge from one mode's closed trades.

    Sharpe: mean / std of the per-trade net PnL series, annualized by
    sqrt(n_trades * backtest.trading_days_per_year) -- since this replay
    covers exactly one trading day, "trades per day" is just n_trades,
    and the annualization assumes (a named, documented simplification)
    that a similar number of similarly-distributed trades recur on other
    days. NaN if fewer than 2 trades or the PnL series has zero variance.

    fill_rate: filled trades / total_signals generated -- always 1.0 for
    signal-replay mode (every signal is assumed to fill), and the
    fraction of orders that actually filled for paper-trading mode.

    slippage_adjusted_edge: mean net PnL per trade in dollars, i.e. the
    average edge per trade *after* every named cost in config.yaml:
    backtest has already been subtracted -- distinct from `pnl_dollars`
    (the total across all trades), so a reviewer can see per-trade edge
    and cumulative PnL side by side.
    """
    n_trades = len(trades)
    if n_trades == 0:
        return {
            "n_trades": 0, "fill_rate": 0.0 if total_signals else float("nan"),
            "pnl_dollars": 0.0, "sharpe": float("nan"), "win_rate": float("nan"),
            "max_drawdown_dollars": 0.0, "slippage_adjusted_edge_dollars": float("nan"),
        }

    pnl = trades["net_pnl_dollars"]
    cum_pnl = pnl.cumsum()
    running_max = cum_pnl.cummax()
    max_drawdown = (cum_pnl - running_max).min()

    ann_factor = (n_trades * config["backtest"]["trading_days_per_year"]) ** 0.5
    std = pnl.std(ddof=1)
    sharpe = (pnl.mean() / std * ann_factor) if n_trades > 1 and std > 0 else float("nan")

    return {
        "n_trades": n_trades,
        "fill_rate": n_trades / total_signals if total_signals > 0 else float("nan"),
        "pnl_dollars": pnl.sum(),
        "sharpe": sharpe,
        "win_rate": (pnl > 0).mean(),
        "max_drawdown_dollars": max_drawdown,
        "slippage_adjusted_edge_dollars": pnl.mean(),
    }


def main():
    # Deferred import: model.py imports evaluate.py, and evaluate.py is
    # imported at module load time above -- importing model.py back here
    # too would work, but is only ever needed by this script entry point,
    # not by anything else in this module, so it's kept local like
    # model.py does for its own evaluate.py import.
    from model import feature_columns, make_predict_fold_fn, make_xgboost_model, make_linear_model

    config = load_config()
    bt = config["backtest"]
    horizon = bt["horizon"]

    df = pd.read_parquet(config["features"]["output_file"])
    df = add_labels(df, [horizon], config["labels"]["epsilon"])

    model_fn = make_xgboost_model if bt["model"] == "xgboost" else make_linear_model
    fold_fn = make_predict_fold_fn(config, model_fn, feature_columns(), uses_class_index=(bt["model"] == "xgboost"))
    oos = generate_oos_predictions(df, horizon, config["evaluate"]["n_splits"], fold_fn)

    oos = oos.reset_index(drop=True)

    print(f"model={bt['model']} horizon={horizon} order_type={bt['order_type']}")
    print()
    for mode in ("signal", "paper"):
        # Fresh generator per mode: iter_events() is single-use, and each
        # mode needs to replay the full day from a clean OrderBook.
        event_chunks = iter_events(config["data"]["message_file"], chunk_size=config["ingest"]["chunk_size"])
        result = run_replay(config, oos, mode, event_chunks)
        metrics = compute_metrics(result["trades"], result["total_signals"], config)
        label = "signal replay" if mode == "signal" else f"paper-trading replay ({bt['order_type']})"
        print(f"=== {label} ===")
        for key, value in metrics.items():
            print(f"  {key:<32} {value:,.4f}" if isinstance(value, float) else f"  {key:<32} {value}")
        print()


if __name__ == "__main__":
    main()
