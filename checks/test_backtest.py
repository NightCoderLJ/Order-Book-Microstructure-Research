"""Unit tests for backtest.py: signal-replay PnL, paper-trading fills
(aggressive slippage/fees, passive queue fill and queue-position loss),
metrics arithmetic, and out-of-sample prediction concatenation -- all on
small, hand-computed synthetic event streams rather than the full day's
data, so each assertion's expected number can be checked by hand."""

import pandas as pd

from backtest import compute_metrics, generate_oos_predictions, run_replay
from ingest import Event

BASE_CONFIG = {
    "data": {"price_scale": 1},
    "book": {"checkpoint_interval": 100_000},
    "backtest": {
        "horizon": 2, "position_size": 1, "fee_per_share": 0.0, "slippage_ticks": 0,
        "tick_size_raw": 1, "assumed_half_spread_raw": 0, "latency_seconds": 0.0,
        "order_type": "aggressive", "trading_days_per_year": 252,
    },
}


def _event(seq, t, event_type, side, price, size, order_id):
    return Event(timestamp_exchange=t, sequence_id=seq, event_type=event_type,
                 side=side, price=price, size=size, order_id=order_id)


def _oos(n, signal_index, pred_class=1.0):
    pc = [float("nan")] * n
    pc[signal_index] = pred_class
    return pd.DataFrame({"score": pc, "pred_class": pc})


def test_signal_replay_marks_to_midprice_h_events_later_with_no_costs():
    # bb/ba after each event: (100,110)->105, then an ask improvement to
    # 108 two events later -> exit midprice 104. Signal fires right after
    # the book first has two-sided liquidity (i=1).
    events = [[
        _event(1, 0.0, "ADD", "B", 100, 10, 1),
        _event(2, 0.1, "ADD", "A", 110, 10, 2),   # i=1: signal fires here, entry mid=105
        _event(3, 0.2, "ADD", "B", 98, 5, 3),     # i=2: doesn't move best bid/ask
        _event(4, 0.3, "ADD", "A", 108, 5, 4),    # i=3 == close_idx: ask improves, exit mid=104
    ]]
    oos = _oos(4, signal_index=1)

    result = run_replay(BASE_CONFIG, oos, "signal", events)
    trades = result["trades"]

    assert result["total_signals"] == 1
    assert len(trades) == 1
    row = trades.iloc[0]
    assert row["gen_idx"] == 1
    assert row["entry_price_raw"] == 105.0
    assert row["exit_price_raw"] == 104.0
    assert row["fee_dollars"] == 0.0
    assert row["net_pnl_dollars"] == -1.0  # (104 - 105) * 1 share, no costs


def test_paper_aggressive_fill_pays_assumed_spread_slippage_and_fee():
    # Aggressive fills price off the *midprice* plus an assumed half-spread
    # plus slippage -- not the live best_ask -- see _resolve_arrival's
    # docstring for why (this dataset's best_ask/best_bid is only
    # genuinely valid ~1.2% of the day).
    config = {**BASE_CONFIG, "backtest": {**BASE_CONFIG["backtest"],
              "order_type": "aggressive", "slippage_ticks": 1, "tick_size_raw": 1,
              "assumed_half_spread_raw": 2, "fee_per_share": 0.5}}
    events = [[
        _event(1, 0.0, "ADD", "B", 100, 10, 1),
        _event(2, 0.1, "ADD", "A", 110, 10, 2),   # i=1: signal fires, order arrives next event
        _event(3, 0.2, "ADD", "B", 98, 5, 3),     # i=2: arrival -- midprice=(100+110)/2=105, +2 spread +1 slippage=108
        _event(4, 0.3, "ADD", "A", 108, 5, 4),    # i=3 == close_idx: exit mid=(100+108)/2=104
    ]]
    oos = _oos(4, signal_index=1)

    result = run_replay(config, oos, "paper", events)
    trades = result["trades"]

    assert len(trades) == 1
    row = trades.iloc[0]
    assert row["entry_price_raw"] == 108.0        # midprice 105 + half_spread 2 + slippage 1
    assert row["fee_dollars"] == 0.5
    assert row["net_pnl_dollars"] == (104.0 - 108.0) * 1 - 0.5


def test_paper_passive_fills_after_enough_volume_trades_through_queue():
    config = {**BASE_CONFIG, "backtest": {**BASE_CONFIG["backtest"],
              "order_type": "passive", "horizon": 6, "fee_per_share": 0.2}}
    events = [[
        _event(1, 0.0, "ADD", "B", 100, 10, 1),
        _event(2, 0.1, "ADD", "A", 110, 10, 2),          # i=1: signal fires (buy)
        _event(3, 0.2, "ADD", "B", 100, 3, 3),           # i=2: arrival, queue_ahead = 10+3 = 13
        _event(4, 0.3, "ADD", "B", 100, 5, 4),           # i=3: order behind us, doesn't count
        _event(5, 0.4, "EXECUTE_VISIBLE", "B", 100, 10, 1),  # i=4: order1 fully trades, cum_exec=10
        _event(6, 0.5, "EXECUTE_VISIBLE", "B", 100, 3, 3),   # i=5: order3 fully trades, cum_exec=13
        _event(7, 0.6, "EXECUTE_VISIBLE", "B", 100, 1, 4),   # i=6: 1 more share -> cum_exec=14, filled
        _event(8, 0.7, "ADD", "A", 105, 2, 5),           # i=7 == close_idx: exit mid=(100+105)/2=102.5
    ]]
    oos = _oos(8, signal_index=1)

    result = run_replay(config, oos, "paper", events)
    trades = result["trades"]

    assert len(trades) == 1
    row = trades.iloc[0]
    assert row["entry_price_raw"] == 100.0   # passive fill: own quoted price, no slippage
    assert row["exit_price_raw"] == 102.5
    assert row["fee_dollars"] == 0.2
    assert row["net_pnl_dollars"] == (102.5 - 100.0) * 1 - 0.2


def test_paper_passive_order_loses_queue_position_when_level_empties():
    config = {**BASE_CONFIG, "backtest": {**BASE_CONFIG["backtest"], "order_type": "passive", "horizon": 5}}
    events = [[
        _event(1, 0.0, "ADD", "B", 100, 10, 1),
        _event(2, 0.1, "ADD", "A", 110, 10, 2),               # i=1: signal fires
        _event(3, 0.2, "ADD", "B", 100, 3, 3),                # i=2: arrival, queue_ahead = 13
        _event(4, 0.3, "EXECUTE_VISIBLE", "B", 100, 10, 1),   # i=3: cum_exec=10, level=3
        _event(5, 0.4, "EXECUTE_VISIBLE", "B", 100, 3, 3),    # i=4: cum_exec=13 (<14), level=0 -> removed
        _event(6, 0.5, "ADD", "B", 90, 1, 5),                 # i=5: filler
    ]]
    oos = _oos(6, signal_index=1)

    result = run_replay(config, oos, "paper", events)

    assert result["total_signals"] == 1
    assert len(result["trades"]) == 0  # never filled: queue position lost when the level emptied


def test_paper_aggressive_prices_off_midprice_even_when_book_is_crossed():
    # This dataset's Level-10 book is crossed (best_bid >= best_ask) on
    # ~98.8% of events (reference/01) -- best_ask can be a stale, out-of-money
    # price. Aggressive fills must not cross that literal price (it would
    # fabricate a fictitious profit); they price off the midprice instead,
    # which stays well-defined and consistent even when bb >= ba.
    config = {**BASE_CONFIG, "backtest": {**BASE_CONFIG["backtest"], "assumed_half_spread_raw": 5}}
    events = [[
        _event(1, 0.0, "ADD", "B", 200, 5, 1),
        _event(2, 0.1, "ADD", "A", 100, 5, 2),  # i=1: signal fires; book is crossed (bb=200 >= ba=100)
        _event(3, 0.2, "ADD", "B", 90, 1, 3),   # i=2: arrival -- midprice=(200+100)/2=150, +5 assumed spread=155
        _event(4, 0.3, "ADD", "A", 108, 5, 4),  # i=3 == close_idx
    ]]
    oos = _oos(4, signal_index=1)

    result = run_replay(config, oos, "paper", events)
    trades = result["trades"]

    assert len(trades) == 1
    assert trades.iloc[0]["entry_price_raw"] == 155.0  # never touches the stale ask(100) or bid(200) directly


def test_paper_aggressive_declines_without_two_sided_liquidity():
    config = BASE_CONFIG
    events = [[
        _event(1, 0.0, "ADD", "B", 100, 10, 1),  # i=0: signal fires here (only one side of book exists)
        _event(2, 0.1, "ADD", "B", 99, 5, 2),    # i=1: arrival -- still no ask at all -> declines
        _event(3, 0.2, "ADD", "B", 98, 5, 3),    # i=2 == close_idx
    ]]
    oos = _oos(3, signal_index=0)

    result = run_replay(config, oos, "paper", events)

    assert result["total_signals"] == 1
    assert len(result["trades"]) == 0


def test_compute_metrics_matches_hand_computed_values():
    trades = pd.DataFrame({"net_pnl_dollars": [1.0, -0.5, 2.0]})
    metrics = compute_metrics(trades, total_signals=4, config=BASE_CONFIG)

    assert metrics["n_trades"] == 3
    assert metrics["fill_rate"] == 3 / 4
    assert metrics["pnl_dollars"] == 2.5
    assert metrics["win_rate"] == 2 / 3
    # cumulative pnl = [1.0, 0.5, 2.5]; running max = [1.0, 1.0, 2.5]; drawdown = [0, -0.5, 0]
    assert metrics["max_drawdown_dollars"] == -0.5
    assert metrics["slippage_adjusted_edge_dollars"] == 2.5 / 3


def test_compute_metrics_handles_zero_trades():
    metrics = compute_metrics(pd.DataFrame(columns=["net_pnl_dollars"]), total_signals=2, config=BASE_CONFIG)
    assert metrics["n_trades"] == 0
    assert metrics["fill_rate"] == 0.0
    assert metrics["pnl_dollars"] == 0.0


def test_generate_oos_predictions_concatenates_fold_test_windows():
    n = 20
    df = pd.DataFrame({"label_h2": [0.0] * n})

    def fake_predict_fold_fn(df, train_range, test_range, label_col):
        idx = df.iloc[test_range.start:test_range.stop].index
        # score/pred_class equal to the test row's position, so each
        # fold's slice is identifiable in the concatenated output.
        score = pd.Series(list(test_range), index=idx, dtype=float)
        pred_class = pd.Series([1.0] * len(idx), index=idx)
        proba = pd.DataFrame(index=idx)
        return score, pred_class, proba

    oos = generate_oos_predictions(df, horizon=2, n_splits=3, predict_fold_fn=fake_predict_fold_fn)

    # first block (before fold 1's test window) has no OOS prediction
    block = n // 4  # walk_forward_splits(20, 3, purge=2) block size
    assert oos["score"].iloc[:block].isna().all()
    # every row from the first fold's test window onward is filled in,
    # and each row's score equals its own row position (per the fake fn)
    assert oos["score"].iloc[block:].tolist() == list(range(block, n))
    assert (oos["pred_class"].iloc[block:] == 1.0).all()
