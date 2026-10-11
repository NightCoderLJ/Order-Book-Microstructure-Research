"""Unit tests for ingest.py: LOBSTER row parsing and chunked streaming."""

from ingest import (
    Event,
    iter_events,
    iter_orderbook_snapshots,
    parse_message_row,
    parse_orderbook_row,
)


def test_parse_message_row_add_buy():
    row = ["34200.004241176", "1", "16113575", "18", "5853300", "1"]
    event = parse_message_row(row, sequence_id=1)
    assert event == Event(
        timestamp_exchange=34200.004241176,
        sequence_id=1,
        event_type="ADD",
        side="B",
        price=5853300,
        size=18,
        order_id=16113575,
    )


def test_parse_message_row_sell_side():
    row = ["34200.1", "1", "42", "10", "100", "-1"]
    event = parse_message_row(row, sequence_id=2)
    assert event.side == "A"


def test_parse_message_row_event_type_map():
    expected = {
        "1": "ADD",
        "2": "CANCEL_PARTIAL",
        "3": "CANCEL_FULL",
        "4": "EXECUTE_VISIBLE",
        "5": "EXECUTE_HIDDEN",
        "7": "HALT",
    }
    for type_code, event_type in expected.items():
        row = ["0.0", type_code, "1", "10", "100", "1"]
        assert parse_message_row(row, sequence_id=1).event_type == event_type


def test_iter_events_chunking_and_sequence_ids(tmp_path):
    csv_path = tmp_path / "messages.csv"
    lines = [f"{i}.0,1,{i},10,100,1" for i in range(1, 26)]  # 25 rows
    csv_path.write_text("\n".join(lines) + "\n")

    chunks = list(iter_events(str(csv_path), chunk_size=10))

    assert [len(c) for c in chunks] == [10, 10, 5]
    all_events = [e for chunk in chunks for e in chunk]
    assert [e.sequence_id for e in all_events] == list(range(1, 26))


def test_parse_orderbook_row_two_levels():
    # AskP1,AskS1,BidP1,BidS1, AskP2,AskS2,BidP2,BidS2
    row = ["105", "5", "100", "3", "106", "8", "99", "4"]
    snap = parse_orderbook_row(row, sequence_id=1, levels=2)
    assert snap.asks == ((105, 5), (106, 8))
    assert snap.bids == ((100, 3), (99, 4))


def test_iter_orderbook_snapshots_streams_all_rows(tmp_path):
    csv_path = tmp_path / "orderbook.csv"
    rows = ["105,5,100,3" for _ in range(7)]
    csv_path.write_text("\n".join(rows) + "\n")

    chunks = list(iter_orderbook_snapshots(str(csv_path), levels=1, chunk_size=3))

    assert [len(c) for c in chunks] == [3, 3, 1]
    all_snaps = [s for chunk in chunks for s in chunk]
    assert [s.sequence_id for s in all_snaps] == list(range(1, 8))
