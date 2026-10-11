"""
book.py -- reconstructs a tick-accurate limit order book from a stream of
canonical order-book events.

Takes in: Event objects (see ingest.py) applied one at a time via
OrderBook.apply_event().

Produces: live queryable book state (best_bid(), best_ask(), depth(n)) and
point-in-time snapshots (snapshot(index)).

Invariants enforced after every single event (logged and counted on
violation, never raised -- a bad row should degrade visibly, not crash a
400k-event replay):
  - order and price-level sizes never go negative
  - the book is never crossed (best_bid < best_ask)
  - best_bid()/best_ask() are consistent with the top of the sorted levels

All three checks are O(1) per event: only the specific order/level touched
by the current event can newly violate an invariant, so there is no need to
rescan the whole book after every update.
"""

import bisect
import logging
from collections import Counter
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

DEFAULT_CHECKPOINT_INTERVAL = 100_000


class OrderBook:
    """Tracks resting bid/ask price levels and an order_id -> order map,
    updated one LOBSTER event at a time. Stateful by nature -- the one
    piece of this pipeline that earns being a class."""

    def __init__(self, checkpoint_interval: int = DEFAULT_CHECKPOINT_INTERVAL):
        self.bid_prices: List[int] = []  # sorted ascending; best bid = last
        self.ask_prices: List[int] = []  # sorted ascending; best ask = first
        self.bid_levels: Dict[int, int] = {}  # price -> aggregate resting size
        self.ask_levels: Dict[int, int] = {}  # price -> aggregate resting size
        self.orders: Dict[int, list] = {}  # order_id -> [side, price, size]

        self.events_applied = 0
        self.event_type_counts: Counter = Counter()
        self.violations: Counter = Counter()

        self.checkpoint_interval = checkpoint_interval
        self.snapshots: Dict[int, dict] = {}

    # -- public queries --

    def best_bid(self) -> Optional[int]:
        """Highest resting buy price, or None if the bid side is empty."""
        return self.bid_prices[-1] if self.bid_prices else None

    def best_ask(self) -> Optional[int]:
        """Lowest resting sell price, or None if the ask side is empty."""
        return self.ask_prices[0] if self.ask_prices else None

    def depth(self, n: int) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
        """Top n (price, aggregate_size) levels per side, best-first.

        Why: this is the shape features.py will need later (top-of-book and
        queue-imbalance features are computed over exactly this depth slice).
        """
        bids = [(p, self.bid_levels[p]) for p in reversed(self.bid_prices[-n:])]
        asks = [(p, self.ask_levels[p]) for p in self.ask_prices[:n]]
        return bids, asks

    def snapshot(self, index: int) -> dict:
        """Captures the full book state at the given event index.

        Stored on self.snapshots[index] and returned, so a caller can either
        keep it for later audit or use the return value directly. Called
        automatically every `checkpoint_interval` events by apply_event so a
        long replay has bounded-memory checkpoints without extra wiring.
        """
        snap = {
            "index": index,
            "bids": sorted(self.bid_levels.items(), reverse=True),
            "asks": sorted(self.ask_levels.items()),
        }
        self.snapshots[index] = snap
        return snap

    # -- event application --

    def apply_event(self, event) -> None:
        """Applies one canonical Event and enforces invariants afterward.

        Why: keeping dispatch, mutation, and invariant checks in one place
        means every code path that changes book state also gets checked --
        there is no way to mutate the book without also validating it.
        """
        handler = _HANDLERS.get(event.event_type)
        if handler is None:
            logger.warning(
                "seq %s: no handler for event_type %s", event.sequence_id, event.event_type
            )
            self.violations["unknown_event_type"] += 1
        else:
            handler(self, event)

        self.event_type_counts[event.event_type] += 1
        self.events_applied += 1
        self._check_invariants(event)

        if self.events_applied % self.checkpoint_interval == 0:
            self.snapshot(self.events_applied)

    # -- internal level helpers --

    def _levels_for(self, side: str):
        return (self.bid_prices, self.bid_levels) if side == "B" else (self.ask_prices, self.ask_levels)

    def _add_level(self, prices: List[int], levels: Dict[int, int], price: int, size: int) -> None:
        if price in levels:
            levels[price] += size
        else:
            levels[price] = size
            bisect.insort(prices, price)

    def _remove_level_if_empty(self, prices: List[int], levels: Dict[int, int], price: int) -> None:
        if levels.get(price, 0) <= 0:
            levels.pop(price, None)
            i = bisect.bisect_left(prices, price)
            if i < len(prices) and prices[i] == price:
                prices.pop(i)

    # -- event handlers (one per canonical event_type) --

    def _apply_add(self, event) -> None:
        if event.order_id in self.orders:
            logger.warning(
                "seq %s: duplicate order_id %s on ADD; overwriting", event.sequence_id, event.order_id
            )
            self.violations["duplicate_order_id"] += 1
        prices, levels = self._levels_for(event.side)
        self.orders[event.order_id] = [event.side, event.price, event.size]
        self._add_level(prices, levels, event.price, event.size)

    def _apply_reduce(self, event, remove_fully: bool) -> None:
        """Shared logic for CANCEL_PARTIAL, CANCEL_FULL, and EXECUTE_VISIBLE,
        which all reduce a known order's resting size by some quantity."""
        order = self.orders.get(event.order_id)
        if order is None:
            # Expected and frequent on real LOBSTER data, not a parsing bug:
            # a Level-N message file only emits ADDs for orders that affect
            # the top-N depth. Orders resting deeper than that boundary can
            # still be cancelled/executed later -- those messages reference
            # an order_id we were never given. Logged at debug (not warning)
            # so a normal run isn't flooded; still counted for reconciliation.
            logger.debug(
                "seq %s: %s references unknown order_id %s",
                event.sequence_id, event.event_type, event.order_id,
            )
            self.violations["unknown_order_id"] += 1
            return

        side, price, remaining = order
        qty = remaining if remove_fully else event.size
        prices, levels = self._levels_for(side)

        new_remaining = remaining - qty
        if new_remaining < 0:
            logger.warning(
                "seq %s: order %s size went negative; clamping to 0",
                event.sequence_id, event.order_id,
            )
            self.violations["negative_order_size"] += 1
            new_remaining = 0

        new_level_size = levels.get(price, 0) - qty
        if new_level_size < 0:
            logger.warning(
                "seq %s: level %s size went negative; clamping to 0", event.sequence_id, price
            )
            self.violations["negative_level_size"] += 1
            new_level_size = 0

        levels[price] = new_level_size
        order[2] = new_remaining
        if order[2] <= 0:
            self.orders.pop(event.order_id, None)
        self._remove_level_if_empty(prices, levels, price)

    def _apply_cancel_partial(self, event) -> None:
        self._apply_reduce(event, remove_fully=False)

    def _apply_cancel_full(self, event) -> None:
        self._apply_reduce(event, remove_fully=True)

    def _apply_execute_visible(self, event) -> None:
        self._apply_reduce(event, remove_fully=False)

    def _apply_execute_hidden(self, event) -> None:
        # Hidden orders are never displayed, so they never entered orders/
        # levels via ADD -- there is nothing to mutate. Still counted above
        # in event_type_counts so message-count reconciliation accounts for it.
        pass

    def _apply_halt(self, event) -> None:
        # Trading halt marker row; carries no book state to apply.
        pass

    # -- invariant checks --

    def _check_invariants(self, event) -> None:
        bb, ba = self.best_bid(), self.best_ask()
        if bb is not None and ba is not None and bb >= ba:
            # Also expected on a Level-N reconstruction: an order that falls
            # outside the top-N depth can stop being cancelled/executed in
            # our view (see _apply_reduce), so it can linger as a stale
            # extreme price and eventually cross a level on the other side
            # that has the same issue. Debug-level for the same reason as
            # unknown_order_id above -- see reference/01_ingestion_and_book.md.
            logger.debug(
                "seq %s: crossed book, best_bid=%s >= best_ask=%s", event.sequence_id, bb, ba
            )
            self.violations["crossed_book"] += 1

        if self.bid_prices and self.bid_levels.get(self.bid_prices[-1], 0) <= 0:
            logger.warning(
                "seq %s: best_bid %s has non-positive size", event.sequence_id, self.bid_prices[-1]
            )
            self.violations["stale_best_bid"] += 1

        if self.ask_prices and self.ask_levels.get(self.ask_prices[0], 0) <= 0:
            logger.warning(
                "seq %s: best_ask %s has non-positive size", event.sequence_id, self.ask_prices[0]
            )
            self.violations["stale_best_ask"] += 1


_HANDLERS = {
    "ADD": OrderBook._apply_add,
    "CANCEL_PARTIAL": OrderBook._apply_cancel_partial,
    "CANCEL_FULL": OrderBook._apply_cancel_full,
    "EXECUTE_VISIBLE": OrderBook._apply_execute_visible,
    "EXECUTE_HIDDEN": OrderBook._apply_execute_hidden,
    "HALT": OrderBook._apply_halt,
}
