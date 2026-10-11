"""Unit tests for features.py: hand-computed formula checks on a synthetic
5-level book, plus rolling-window aggregation checks."""

import pandas as pd

from book import OrderBook
from features import (
    add_rolling_windows,
    instantaneous_microstructure,
    ofi_contribution,
    top_level,
    top_of_book,
    trade_signed_volume,
)
from ingest import Event


def ev(event_type, side, price, size, order_id, sequence_id=1):
    return Event(
        timestamp_exchange=float(sequence_id),
        sequence_id=sequence_id,
        event_type=event_type,
        side=side,
        price=price,
        size=size,
        order_id=order_id,
    )


def synthetic_5_level_book() -> OrderBook:
    """5 price levels per side:
    bids: 100 (10), 99 (20), 98 (30), 97 (40), 96 (50)  -- best bid = 100
    asks: 101 (5),  102 (15), 103 (25), 104 (35), 105 (45) -- best ask = 101
    """
    book = OrderBook()
    for i, (price, size) in enumerate([(100, 10), (99, 20), (98, 30), (97, 40), (96, 50)]):
        book.apply_event(ev("ADD", "B", price, size, order_id=i))
    for i, (price, size) in enumerate([(101, 5), (102, 15), (103, 25), (104, 35), (105, 45)]):
        book.apply_event(ev("ADD", "A", price, size, order_id=100 + i))
    return book


# -- top_of_book --


def test_top_of_book_formulas_on_synthetic_book():
    book = synthetic_5_level_book()
    fields = top_of_book(book)

    assert fields["best_bid"] == 100
    assert fields["best_ask"] == 101
    assert fields["midprice"] == 100.5
    assert fields["spread"] == 1
    assert fields["relative_spread"] == 1 / 100.5
    assert fields["bid_size_l1"] == 10
    assert fields["ask_size_l1"] == 5


def test_top_of_book_is_nan_when_a_side_is_empty():
    book = OrderBook()
    book.apply_event(ev("ADD", "B", 100, 10, order_id=1))

    fields = top_of_book(book)
    assert fields["best_ask"] is None
    assert pd.isna(fields["midprice"])
    assert pd.isna(fields["spread"])


# -- instantaneous_microstructure --


def test_queue_imbalance_formula_l1_only():
    book = synthetic_5_level_book()
    fields = instantaneous_microstructure(book, depth=5)

    # (bid_l1 - ask_l1) / (bid_l1 + ask_l1) = (10 - 5) / (10 + 5)
    assert fields["queue_imbalance"] == (10 - 5) / (10 + 5)


def test_microprice_formula_l1_only():
    book = synthetic_5_level_book()
    fields = instantaneous_microstructure(book, depth=5)

    # (best_bid * ask_l1 + best_ask * bid_l1) / (bid_l1 + ask_l1)
    expected = (100 * 5 + 101 * 10) / (10 + 5)
    assert fields["microprice"] == expected


def test_liquidity_pressure_and_weighted_microprice_use_depth_aggregate():
    book = synthetic_5_level_book()
    fields = instantaneous_microstructure(book, depth=3)

    bid_depth_size = 10 + 20 + 30  # top 3 bid levels
    ask_depth_size = 5 + 15 + 25   # top 3 ask levels

    expected_pressure = (bid_depth_size - ask_depth_size) / (bid_depth_size + ask_depth_size)
    expected_wmp = (100 * ask_depth_size + 101 * bid_depth_size) / (bid_depth_size + ask_depth_size)

    assert fields["liquidity_pressure"] == expected_pressure
    assert fields["weighted_microprice"] == expected_wmp


def test_weighted_microprice_at_depth_1_matches_l1_microprice():
    # depth=1 aggregates only the best level per side, so weighted_microprice
    # collapses to exactly the same formula/value as the L1 microprice.
    book = synthetic_5_level_book()
    fields = instantaneous_microstructure(book, depth=1)

    assert fields["weighted_microprice"] == fields["microprice"]


def test_weighted_microprice_differs_from_microprice_when_depth_greater_than_1():
    book = synthetic_5_level_book()
    fields = instantaneous_microstructure(book, depth=3)

    assert fields["weighted_microprice"] != fields["microprice"]


# -- ofi_contribution (Cont, Kukanov & Stoikov 2014) --


def test_ofi_same_price_size_increase_on_bid_is_bullish():
    # bid size grows at the same best price (e.g. an ADD behind the queue);
    # ask unchanged -> e_n = (15 - 10) - 0 = 5
    prev_bid, curr_bid = (100, 10), (100, 15)
    prev_ask, curr_ask = (105, 8), (105, 8)

    assert ofi_contribution(prev_bid, prev_ask, curr_bid, curr_ask) == 5


def test_ofi_bid_price_improvement_is_bullish():
    # a new, better bid appears -> dW_bid = curr_size = 5; ask unchanged
    prev_bid, curr_bid = (100, 10), (101, 5)
    prev_ask, curr_ask = (105, 8), (105, 8)

    assert ofi_contribution(prev_bid, prev_ask, curr_bid, curr_ask) == 5


def test_ofi_ask_price_improvement_down_is_bearish():
    # a new, more aggressive (lower) ask appears -> dW_ask = curr_size = 6
    # bid unchanged -> e_n = 0 - 6 = -6
    prev_bid, curr_bid = (100, 10), (100, 10)
    prev_ask, curr_ask = (105, 8), (103, 6)

    assert ofi_contribution(prev_bid, prev_ask, curr_bid, curr_ask) == -6


def test_ofi_bid_side_emptying_is_bearish():
    # best bid cancelled entirely, no bid left -> dW_bid = -prev_size = -10
    prev_bid, curr_bid = (100, 10), None
    prev_ask, curr_ask = (105, 8), (105, 8)

    assert ofi_contribution(prev_bid, prev_ask, curr_bid, curr_ask) == -10


def test_ofi_ask_side_emptying_is_bullish():
    # best ask cancelled entirely, no ask left -> dW_ask = -prev_size = -8
    # e_n = 0 - (-8) = 8
    prev_bid, curr_bid = (100, 10), (100, 10)
    prev_ask, curr_ask = (105, 8), None

    assert ofi_contribution(prev_bid, prev_ask, curr_bid, curr_ask) == 8


def test_top_level_matches_book_state():
    book = synthetic_5_level_book()
    assert top_level(book, "B") == (100, 10)
    assert top_level(book, "A") == (101, 5)


# -- trade_signed_volume --


def test_trade_signed_volume_buyer_initiated_when_resting_side_is_ask():
    event = ev("EXECUTE_VISIBLE", "A", 101, 7, order_id=1)
    assert trade_signed_volume(event) == 7


def test_trade_signed_volume_seller_initiated_when_resting_side_is_bid():
    event = ev("EXECUTE_VISIBLE", "B", 100, 7, order_id=1)
    assert trade_signed_volume(event) == -7


def test_trade_signed_volume_counts_hidden_executions_too():
    event = ev("EXECUTE_HIDDEN", "A", 101, 3, order_id=0)
    assert trade_signed_volume(event) == 3


def test_trade_signed_volume_zero_for_non_trade_events():
    event = ev("ADD", "B", 100, 10, order_id=1)
    assert trade_signed_volume(event) == 0.0


# -- add_rolling_windows --


def test_arrival_rate_and_ofi_rolling_window_hand_computed():
    # 5 synthetic raw rows: is_add flags and ofi contributions chosen so the
    # window=5 rolling values can be hand-verified from the raw column,
    # independent of any real book replay.
    df = pd.DataFrame({
        "is_add": [1.0, 1.0, 0.0, 1.0, 0.0],
        "is_cancel": [0.0, 0.0, 1.0, 0.0, 1.0],
        "queue_imbalance": [0.2, 0.2, 0.2, 0.2, 0.2],
        "microprice": [100.0] * 5,
        "weighted_microprice": [100.0] * 5,
        "liquidity_pressure": [0.1] * 5,
        "ofi": [5.0, -2.0, 3.0, 1.0, -1.0],
        "trade_signed_volume": [0.0, 10.0, -4.0, 0.0, 6.0],
    })

    out = add_rolling_windows(df, windows=[5])

    # arrival_rate_w5 at the last row = fraction of ADDs across all 5 rows
    assert out["arrival_rate_w5"].iloc[-1] == 3 / 5
    # cancel_rate_w5 at the last row = fraction of cancels across all 5 rows
    assert out["cancel_rate_w5"].iloc[-1] == 2 / 5
    # ofi_w5 at the last row = sum of all 5 contributions
    assert out["ofi_w5"].iloc[-1] == 5.0 - 2.0 + 3.0 + 1.0 - 1.0
    # partial window at row 0 (min_periods=1): just that row's own value
    assert out["ofi_w5"].iloc[0] == 5.0
    assert out["arrival_rate_w5"].iloc[0] == 1.0


def test_trade_imbalance_rolling_window_hand_computed():
    df = pd.DataFrame({
        "is_add": [0.0] * 4,
        "is_cancel": [0.0] * 4,
        "queue_imbalance": [0.0] * 4,
        "microprice": [0.0] * 4,
        "weighted_microprice": [0.0] * 4,
        "liquidity_pressure": [0.0] * 4,
        "ofi": [0.0] * 4,
        "trade_signed_volume": [10.0, -4.0, 6.0, 0.0],
    })

    out = add_rolling_windows(df, windows=[4])

    # (buy - sell) / (buy + sell) volume over the whole window:
    # signed sum = 10 - 4 + 6 + 0 = 12; absolute sum = 10 + 4 + 6 + 0 = 20
    assert out["trade_imbalance_w4"].iloc[-1] == 12 / 20


def test_trade_imbalance_is_zero_when_no_trades_in_window():
    df = pd.DataFrame({
        "is_add": [1.0, 1.0],
        "is_cancel": [0.0, 0.0],
        "queue_imbalance": [0.0, 0.0],
        "microprice": [0.0, 0.0],
        "weighted_microprice": [0.0, 0.0],
        "liquidity_pressure": [0.0, 0.0],
        "ofi": [0.0, 0.0],
        "trade_signed_volume": [0.0, 0.0],
    })

    out = add_rolling_windows(df, windows=[2])
    assert out["trade_imbalance_w2"].iloc[-1] == 0.0


def test_add_rolling_windows_drops_raw_columns():
    df = pd.DataFrame({
        "is_add": [1.0], "is_cancel": [0.0], "queue_imbalance": [0.1],
        "microprice": [100.0], "weighted_microprice": [100.0],
        "liquidity_pressure": [0.1], "ofi": [1.0], "trade_signed_volume": [0.0],
    })

    out = add_rolling_windows(df, windows=[5])
    for raw_col in ["is_add", "is_cancel", "queue_imbalance", "microprice",
                    "weighted_microprice", "liquidity_pressure", "ofi", "trade_signed_volume"]:
        assert raw_col not in out.columns
