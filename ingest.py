"""
ingest.py -- parses raw LOBSTER message and orderbook files into typed,
streamed records.

Takes in:
  - a LOBSTER "message" CSV (one row per book-changing event: Time, Type,
    OrderID, Size, Price, Direction)
  - a LOBSTER "orderbook" CSV (one row per message, giving the top-N
    aggregated price/size levels immediately after that event is applied)

Produces:
  - Event objects in the canonical schema (timestamp_exchange, sequence_id,
    event_type, side, price, size, order_id) consumed by book.py
  - OrderbookSnapshot objects (top-N bid/ask levels) parsed from the
    orderbook file for ad-hoc inspection; not fed into book.py's
    reconstruction. LOBSTER's Level-N message file only emits messages for
    order flow that affects the top-N depth, so this file cannot be used to
    validate book.py byte-for-byte -- see reference/01_ingestion_and_book.md.

Both files are read row-by-row via csv.reader and handed out in fixed-size
chunks (see config.yaml: ingest.chunk_size) so peak memory stays bounded
regardless of file size -- nothing is loaded fully into memory.
"""

import csv
from dataclasses import dataclass
from typing import Iterator, List, Optional, Tuple

# LOBSTER message "Type" column -> canonical event_type.
# Direction always identifies the side of the RESTING order being affected,
# not the side of an incoming aggressor -- see LOBSTER's data description.
EVENT_TYPE_MAP = {
    1: "ADD",              # new limit order
    2: "CANCEL_PARTIAL",   # partial cancellation (reduces size)
    3: "CANCEL_FULL",      # full deletion of a resting order
    4: "EXECUTE_VISIBLE",  # execution against a visible resting order
    5: "EXECUTE_HIDDEN",   # execution against a hidden (undisplayed) order
    7: "HALT",             # trading halt indicator row
}

SIDE_MAP = {1: "B", -1: "A"}


@dataclass(frozen=True, slots=True)
class Event:
    """One canonical order-book event, in the schema book.py expects."""

    timestamp_exchange: float
    sequence_id: int
    event_type: str
    side: Optional[str]  # 'B' or 'A'; None for HALT rows
    price: int           # LOBSTER integer units: dollars * price_scale
    size: int
    order_id: int


def parse_message_row(row: List[str], sequence_id: int) -> Event:
    """Converts one raw LOBSTER message row into a canonical Event.

    Why: isolating this in one place means the rest of the pipeline never
    has to know LOBSTER's column order or type codes.
    """
    time_s, type_code, order_id, size, price, direction = row
    type_code = int(type_code)
    direction = int(direction)
    return Event(
        timestamp_exchange=float(time_s),
        sequence_id=sequence_id,
        event_type=EVENT_TYPE_MAP.get(type_code, f"UNKNOWN_{type_code}"),
        side=SIDE_MAP.get(direction),
        price=int(price),
        size=int(size),
        order_id=int(order_id),
    )


def iter_events(path: str, chunk_size: int = 10_000) -> Iterator[List[Event]]:
    """Streams a LOBSTER message file as chunks of canonical Events.

    sequence_id is the 1-indexed row position in the file, which LOBSTER
    guarantees is already in exchange-priority order -- so it doubles as
    our reconciliation key (messages in vs. events applied).
    """
    with open(path, newline="") as f:
        reader = csv.reader(f)
        chunk: List[Event] = []
        sequence_id = 0
        for row in reader:
            sequence_id += 1
            chunk.append(parse_message_row(row, sequence_id))
            if len(chunk) >= chunk_size:
                yield chunk
                chunk = []
        if chunk:
            yield chunk


@dataclass(frozen=True, slots=True)
class OrderbookSnapshot:
    """Top-N (price, size) levels from LOBSTER's own orderbook file, for
    one message index. Used only to validate book.py's reconstruction."""

    sequence_id: int
    bids: Tuple[Tuple[int, int], ...]  # best-first
    asks: Tuple[Tuple[int, int], ...]  # best-first


def parse_orderbook_row(row: List[str], sequence_id: int, levels: int) -> OrderbookSnapshot:
    """Converts one raw LOBSTER orderbook row into an OrderbookSnapshot.

    LOBSTER lays each row out as repeating groups of four columns:
    AskPrice_i, AskSize_i, BidPrice_i, BidSize_i for i = 1..levels. Empty
    levels are marked with sentinel prices and size 0; callers should
    ignore entries with size 0 rather than treat the sentinel as a price.
    """
    bids = []
    asks = []
    for i in range(levels):
        ask_price, ask_size, bid_price, bid_size = (int(x) for x in row[4 * i : 4 * i + 4])
        asks.append((ask_price, ask_size))
        bids.append((bid_price, bid_size))
    return OrderbookSnapshot(sequence_id=sequence_id, bids=tuple(bids), asks=tuple(asks))


def iter_orderbook_snapshots(path: str, levels: int, chunk_size: int = 10_000) -> Iterator[List[OrderbookSnapshot]]:
    """Streams a LOBSTER orderbook file as chunks of OrderbookSnapshots."""
    with open(path, newline="") as f:
        reader = csv.reader(f)
        chunk: List[OrderbookSnapshot] = []
        sequence_id = 0
        for row in reader:
            sequence_id += 1
            chunk.append(parse_orderbook_row(row, sequence_id, levels))
            if len(chunk) >= chunk_size:
                yield chunk
                chunk = []
        if chunk:
            yield chunk
