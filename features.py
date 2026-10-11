"""
features.py -- computes microstructure features from a reconstructed
OrderBook, one row per applied event.

Takes in: Event objects (see ingest.py), replayed one at a time through an
OrderBook (see book.py).

Produces: a table with one row per event --
  - top-of-book fields (best bid/ask, midprice, spread, relative spread,
    top-level sizes), computed instantaneously at that event
  - microstructure signals (order flow imbalance, queue imbalance,
    microprice, weighted microprice, liquidity pressure, order arrival
    rate, cancel rate, trade imbalance), each aggregated over trailing
    event-count windows of 5/10/20/50 events (config.yaml: features.windows)

Written to config.yaml: features.output_file as Parquet, indexed by
(event_id, timestamp).

Why event-count windows and not time or depth-slope windows: those
variants are deferred until later ablations show they're needed (YAGNI)
-- event count is also the natural unit here since book.py itself steps
one event at a time.
"""

import os
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd
import yaml

from book import OrderBook
from ingest import Event, iter_events

Level = Optional[Tuple[int, int]]  # (price, size) at the top of one side, or None if empty


def load_config(path: str = "config.yaml") -> dict:
    """Loads the small set of named assumptions this pipeline depends on."""
    with open(path) as f:
        return yaml.safe_load(f)


# -- instantaneous, per-event fields (computed from current book state) --


def top_of_book(book: OrderBook) -> Dict[str, float]:
    """Best bid/ask, midprice, spread, relative spread, and top-level sizes
    at the book's current state.

    Why: these are the plain top-of-book quantities item 1 of the Phase 2
    spec asks for -- unlike the microstructure signals below, they are not
    aggregated over event windows.
    """
    bb, ba = book.best_bid(), book.best_ask()
    if bb is None or ba is None:
        # Only possible on the first few events of a replay, before both
        # sides have resting liquidity -- not an invariant violation.
        return {
            "best_bid": bb, "best_ask": ba, "midprice": float("nan"),
            "spread": float("nan"), "relative_spread": float("nan"),
            "bid_size_l1": float("nan"), "ask_size_l1": float("nan"),
        }
    bid_size = book.bid_levels[bb]
    ask_size = book.ask_levels[ba]
    midprice = (bb + ba) / 2
    spread = ba - bb
    return {
        "best_bid": bb, "best_ask": ba, "midprice": midprice,
        "spread": spread, "relative_spread": spread / midprice,
        "bid_size_l1": bid_size, "ask_size_l1": ask_size,
    }


def instantaneous_microstructure(book: OrderBook, depth: int) -> Dict[str, float]:
    """Point-in-time microstructure signals that depend only on the book's
    current state (as opposed to OFI/arrival/cancel/trade-imbalance, which
    are inherently flow quantities defined over a window of events).

    Returns queue_imbalance/microprice (L1 only) and their depth-aggregated
    counterparts liquidity_pressure/weighted_microprice (using the top
    `depth` levels per side, config.yaml: features.microprice_depth). Each
    pair uses the same two formulas -- an imbalance and a size-weighted
    price -- just at two different depths, so the depth-k versions add
    information beyond L1 without requiring a full depth-slope model.

    microprice formula (Stoikov): weighting each side's *own* price by the
    *opposite* side's resting size, so a heavier bid queue pulls the
    fair-value price up toward the ask, and vice versa.
    """
    bb, ba = book.best_bid(), book.best_ask()
    if bb is None or ba is None:
        return {
            "queue_imbalance": float("nan"), "microprice": float("nan"),
            "weighted_microprice": float("nan"), "liquidity_pressure": float("nan"),
        }
    bid_l1 = book.bid_levels[bb]
    ask_l1 = book.ask_levels[ba]
    queue_imbalance = (bid_l1 - ask_l1) / (bid_l1 + ask_l1)
    microprice = (bb * ask_l1 + ba * bid_l1) / (bid_l1 + ask_l1)

    bids_d, asks_d = book.depth(depth)
    bid_depth_size = sum(size for _, size in bids_d)
    ask_depth_size = sum(size for _, size in asks_d)
    liquidity_pressure = (bid_depth_size - ask_depth_size) / (bid_depth_size + ask_depth_size)
    weighted_microprice = (bb * ask_depth_size + ba * bid_depth_size) / (bid_depth_size + ask_depth_size)

    return {
        "queue_imbalance": queue_imbalance, "microprice": microprice,
        "weighted_microprice": weighted_microprice, "liquidity_pressure": liquidity_pressure,
    }


def top_level(book: OrderBook, side: str) -> Level:
    """(price, aggregate_size) at the best level of `side`, or None if that
    side is currently empty. Used to snapshot pre/post-event state for
    ofi_contribution(), which needs both."""
    price = book.best_bid() if side == "B" else book.best_ask()
    if price is None:
        return None
    levels = book.bid_levels if side == "B" else book.ask_levels
    return (price, levels[price])


# -- order flow imbalance (Cont, Kukanov & Stoikov 2014) --


def _delta_w_bid(prev: Level, curr: Level) -> float:
    if prev is None and curr is None:
        return 0.0
    if prev is None:  # bid side went from empty to non-empty: an improvement from -inf
        return curr[1]
    if curr is None:  # bid side emptied out: equivalent to the price falling
        return -prev[1]
    prev_price, prev_size = prev
    curr_price, curr_size = curr
    if curr_price > prev_price:
        return curr_size
    if curr_price == prev_price:
        return curr_size - prev_size
    return -prev_size


def _delta_w_ask(prev: Level, curr: Level) -> float:
    if prev is None and curr is None:
        return 0.0
    if prev is None:  # ask side went from empty to non-empty: an improvement from +inf
        return curr[1]
    if curr is None:  # ask side emptied out: equivalent to the price rising
        return -prev[1]
    prev_price, prev_size = prev
    curr_price, curr_size = curr
    if curr_price < prev_price:
        return curr_size
    if curr_price == prev_price:
        return curr_size - prev_size
    return -prev_size


def ofi_contribution(prev_bid: Level, prev_ask: Level, curr_bid: Level, curr_ask: Level) -> float:
    """Single-event order flow imbalance term e_n = dW_bid - dW_ask (Cont,
    Kukanov & Stoikov, "The Price Impact of Order Book Events", 2014).

    Each side's dW term is +size for a price improvement, +/-(size delta)
    for a same-price size change, and -size for a price deterioration/
    cancellation of the best level -- so e_n is positive when order flow
    is net bullish (bid strengthens and/or ask weakens) and negative when
    net bearish. Summed (not averaged) over a window to get the window's
    OFI, since OFI is a flow quantity, not a level quantity.
    """
    return _delta_w_bid(prev_bid, curr_bid) - _delta_w_ask(prev_ask, curr_ask)


# -- per-event flow flags (arrival / cancel / trade) --


def trade_signed_volume(event: Event) -> float:
    """Signed trade size for one event: positive if the aggressor bought
    (a resting sell/'A' order was hit), negative if the aggressor sold (a
    resting buy/'B' order was hit), 0 for non-trade events.

    Both EXECUTE_VISIBLE and EXECUTE_HIDDEN count: LOBSTER's message file
    reports Direction/Price/Size for hidden executions too (see
    ingest.py), so both are real trade prints even though book.py treats
    EXECUTE_HIDDEN as a book-state no-op.
    """
    if event.event_type not in ("EXECUTE_VISIBLE", "EXECUTE_HIDDEN"):
        return 0.0
    if event.side == "A":
        return float(event.size)
    if event.side == "B":
        return -float(event.size)
    return 0.0


# -- table construction --


def _raw_event_rows(events: Iterable[Event], book: OrderBook, depth: int) -> List[dict]:
    """Replays `events` through `book` one at a time, producing one raw
    (unwindowed) row per event: top-of-book fields plus the instantaneous
    and per-event flow values that add_rolling_windows() aggregates into
    the final windowed microstructure columns.
    """
    rows = []
    prev_bid: Level = None
    prev_ask: Level = None

    for event in events:
        book.apply_event(event)
        curr_bid = top_level(book, "B")
        curr_ask = top_level(book, "A")

        row = {"event_id": event.sequence_id, "timestamp": event.timestamp_exchange}
        row.update(top_of_book(book))
        row.update(instantaneous_microstructure(book, depth))
        row["ofi"] = ofi_contribution(prev_bid, prev_ask, curr_bid, curr_ask)
        row["is_add"] = 1.0 if event.event_type == "ADD" else 0.0
        row["is_cancel"] = 1.0 if event.event_type in ("CANCEL_PARTIAL", "CANCEL_FULL") else 0.0
        row["trade_signed_volume"] = trade_signed_volume(event)
        rows.append(row)

        prev_bid, prev_ask = curr_bid, curr_ask

    return rows


# raw column -> (windowed feature name, rolling aggregation)
_MEAN_COLUMNS = {
    "queue_imbalance": "queue_imbalance",
    "microprice": "microprice",
    "weighted_microprice": "weighted_microprice",
    "liquidity_pressure": "liquidity_pressure",
    "is_add": "arrival_rate",
    "is_cancel": "cancel_rate",
}
_SUM_COLUMNS = {"ofi": "ofi"}


def add_rolling_windows(df: pd.DataFrame, windows: List[int]) -> pd.DataFrame:
    """Expands the raw per-event columns into windowed microstructure
    features over each event-count window in `windows`, then drops the raw
    columns (only the windowed features and top-of-book fields are kept in
    the output table, per the Phase 2 spec).

    Aggregation per feature:
      - queue_imbalance/microprice/weighted_microprice/liquidity_pressure:
        rolling mean (they are point-in-time levels; averaging smooths
        noise without changing units)
      - arrival_rate/cancel_rate: rolling mean of a 0/1 indicator, i.e. the
        fraction of the trailing window that was an ADD/cancel event
      - ofi: rolling sum (OFI is a flow quantity, defined by accumulation)
      - trade_imbalance: rolling sum of signed trade volume divided by
        rolling sum of absolute trade volume, i.e. (buy - sell) / (buy +
        sell) volume over the window; 0 when there were no trades in it

    min_periods=1 so the first w-1 rows of the file get a value computed
    over however many events are actually available, rather than NaN --
    documented in reference/02_features.md.
    """
    for w in windows:
        for raw_col, out_name in {**_MEAN_COLUMNS}.items():
            df[f"{out_name}_w{w}"] = df[raw_col].rolling(window=w, min_periods=1).mean()
        for raw_col, out_name in _SUM_COLUMNS.items():
            df[f"{out_name}_w{w}"] = df[raw_col].rolling(window=w, min_periods=1).sum()

        signed = df["trade_signed_volume"].rolling(window=w, min_periods=1).sum()
        absolute = df["trade_signed_volume"].abs().rolling(window=w, min_periods=1).sum()
        df[f"trade_imbalance_w{w}"] = (signed / absolute).where(absolute > 0, 0.0)

    raw_columns = list(_MEAN_COLUMNS) + list(_SUM_COLUMNS) + ["trade_signed_volume"]
    return df.drop(columns=raw_columns)


def build_feature_table(config: dict, book: Optional[OrderBook] = None) -> pd.DataFrame:
    """Runs the full Phase 2 pipeline: streams the configured message file
    through `book` (a fresh OrderBook if none is passed), then builds and
    returns the windowed feature table, indexed by (event_id, timestamp).
    """
    book = book if book is not None else OrderBook(checkpoint_interval=config["book"]["checkpoint_interval"])
    chunk_size = config["ingest"]["chunk_size"]
    depth = config["features"]["microprice_depth"]
    windows = config["features"]["windows"]

    rows: List[dict] = []
    for chunk in iter_events(config["data"]["message_file"], chunk_size=chunk_size):
        rows.extend(_raw_event_rows(chunk, book, depth))

    df = pd.DataFrame(rows)
    df = add_rolling_windows(df, windows)
    return df.set_index(["event_id", "timestamp"])


def write_features(df: pd.DataFrame, path: str) -> None:
    """Writes the feature table to `path` as Parquet, creating the parent
    directory if needed (data/features/ is gitignored, not checked in)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_parquet(path)


def main():
    config = load_config()
    df = build_feature_table(config)
    write_features(df, config["features"]["output_file"])

    print(f"rows written: {len(df):,}")
    print(f"columns:      {len(df.columns)}")
    print(f"output:       {config['features']['output_file']}")
    print()
    print("sample row (first event with full trade/arrival history):")
    print(df.iloc[[-1]].T)


if __name__ == "__main__":
    main()
