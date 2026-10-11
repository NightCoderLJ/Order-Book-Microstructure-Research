"""Unit tests for book.py: OrderBook state transitions and invariants."""

from book import OrderBook
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


def test_add_sets_best_bid_and_ask():
    book = OrderBook()
    book.apply_event(ev("ADD", "B", 100, 10, order_id=1))
    book.apply_event(ev("ADD", "A", 105, 8, order_id=2))

    assert book.best_bid() == 100
    assert book.best_ask() == 105
    assert book.depth(1) == ([(100, 10)], [(105, 8)])


def test_add_aggregates_size_at_same_price():
    book = OrderBook()
    book.apply_event(ev("ADD", "B", 100, 10, order_id=1))
    book.apply_event(ev("ADD", "B", 100, 5, order_id=2))

    assert book.depth(1) == ([(100, 15)], [])


def test_cancel_partial_reduces_size():
    book = OrderBook()
    book.apply_event(ev("ADD", "B", 100, 10, order_id=1))
    book.apply_event(ev("CANCEL_PARTIAL", "B", 100, 4, order_id=1))

    assert book.depth(1) == ([(100, 6)], [])
    assert book.orders[1][2] == 6
    assert book.violations == {}


def test_cancel_full_removes_order_but_keeps_other_orders_at_level():
    book = OrderBook()
    book.apply_event(ev("ADD", "B", 100, 10, order_id=1))
    book.apply_event(ev("ADD", "B", 100, 5, order_id=2))
    book.apply_event(ev("CANCEL_FULL", "B", 100, 0, order_id=1))

    assert 1 not in book.orders
    assert book.depth(1) == ([(100, 5)], [])
    assert book.best_bid() == 100


def test_execute_visible_partial_then_full_fill_removes_level():
    book = OrderBook()
    book.apply_event(ev("ADD", "B", 100, 10, order_id=1))
    book.apply_event(ev("EXECUTE_VISIBLE", "B", 100, 4, order_id=1))
    assert book.depth(1) == ([(100, 6)], [])

    book.apply_event(ev("EXECUTE_VISIBLE", "B", 100, 6, order_id=1))
    assert book.best_bid() is None
    assert book.depth(1) == ([], [])
    assert 1 not in book.orders
    assert book.violations == {}


def test_execute_hidden_is_a_noop_for_book_state():
    book = OrderBook()
    book.apply_event(ev("ADD", "B", 100, 10, order_id=1))
    book.apply_event(ev("EXECUTE_HIDDEN", "B", 999, 1, order_id=999))

    assert book.depth(1) == ([(100, 10)], [])
    assert book.event_type_counts["EXECUTE_HIDDEN"] == 1
    assert book.violations == {}


def test_unknown_order_id_is_logged_and_counted_not_raised():
    book = OrderBook()
    book.apply_event(ev("CANCEL_PARTIAL", "B", 100, 4, order_id=999))

    assert book.violations["unknown_order_id"] == 1
    assert book.events_applied == 1


def test_duplicate_order_id_on_add_is_flagged():
    book = OrderBook()
    book.apply_event(ev("ADD", "B", 100, 10, order_id=1))
    book.apply_event(ev("ADD", "B", 100, 5, order_id=1))

    assert book.violations["duplicate_order_id"] == 1


def test_crossed_book_is_detected_not_prevented():
    book = OrderBook()
    book.apply_event(ev("ADD", "A", 100, 10, order_id=1))
    book.apply_event(ev("ADD", "B", 150, 10, order_id=2))  # bid above resting ask

    assert book.violations["crossed_book"] == 1
    assert book.best_bid() == 150
    assert book.best_ask() == 100


def test_depth_returns_top_n_best_first_both_sides():
    book = OrderBook()
    for i, price in enumerate([100, 99, 98]):
        book.apply_event(ev("ADD", "B", price, 1, order_id=i))
    for i, price in enumerate([101, 102, 103]):
        book.apply_event(ev("ADD", "A", price, 1, order_id=100 + i))

    bids, asks = book.depth(2)
    assert bids == [(100, 1), (99, 1)]
    assert asks == [(101, 1), (102, 1)]


def test_snapshot_captures_current_state():
    book = OrderBook()
    book.apply_event(ev("ADD", "B", 100, 10, order_id=1))
    book.apply_event(ev("ADD", "A", 105, 8, order_id=2))

    snap = book.snapshot(book.events_applied)

    assert snap["bids"] == [(100, 10)]
    assert snap["asks"] == [(105, 8)]
    assert book.snapshots[book.events_applied] is snap


def test_periodic_checkpointing_every_n_events():
    book = OrderBook(checkpoint_interval=3)
    for i in range(3):
        book.apply_event(ev("ADD", "B", 100 - i, 1, order_id=i))

    assert 3 in book.snapshots
    assert 1 not in book.snapshots
    assert 2 not in book.snapshots
