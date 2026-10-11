# Stage 1: Ingestion and Book Reconstruction

## What problem does this stage solve?

Raw exchange data arrives as a flat stream of order-level events (new
orders, cancels, executions). Nothing in that stream directly says "the
best bid is $X" -- that has to be derived by replaying every event against
a running model of the book. This stage turns the raw event stream into:

1. a typed, memory-bounded stream of canonical events (`ingest.py`)
2. a live, queryable limit order book reconstructed one event at a time
   (`book.py`)
3. a driver that replays a full trading day and reports throughput and
   reconciliation counts (`replay.py`)

Everything downstream (features, labels, model, backtest) reads from the
book this stage produces, so its correctness is the foundation the rest of
the resume claims ("tick-accurate", "sub-ms", IC comparisons) rest on.

## Data

- **Venue / feed**: NASDAQ, via LOBSTER (derived from NASDAQ TotalView-ITCH)
- **Instrument**: AAPL
- **Sample period**: 2012-06-21, full continuous trading session
  (09:30:00.000 - 16:00:00.000)
- **Book depth of the raw file**: LOBSTER "Level 10" (top 10 price levels
  per side)
- **Files** (`data/raw/`, ~110MB total, not committed to version control):
  - `AAPL_2012-06-21_34200000_57600000_message_10.csv` -- 400,391 rows, one
    per book-changing event
  - `AAPL_2012-06-21_34200000_57600000_orderbook_10.csv` -- 400,391 rows,
    one per message, giving LOBSTER's own top-10 snapshot after that message

## Inputs and outputs

### `ingest.py`

**Input**: one row of the LOBSTER message file:
`Time, Type, OrderID, Size, Price, Direction`

**Output**: a canonical `Event`:

| field               | type          | meaning                                                    |
|---------------------|---------------|-------------------------------------------------------------|
| `timestamp_exchange` | float        | seconds since midnight, exchange clock                     |
| `sequence_id`        | int          | 1-indexed row position in the message file                 |
| `event_type`         | str          | `ADD`, `CANCEL_PARTIAL`, `CANCEL_FULL`, `EXECUTE_VISIBLE`, `EXECUTE_HIDDEN`, `HALT` |
| `side`               | `'B'`/`'A'`/None | side of the *resting* order the event applies to (LOBSTER's `Direction` always identifies the resting order, not an incoming aggressor) |
| `price`              | int          | LOBSTER's native units: dollars x 10,000 (e.g. `5853300` = $585.33) |
| `size`               | int          | shares |
| `order_id`           | int          | LOBSTER order reference number (`0` for `EXECUTE_HIDDEN`, since hidden orders have no visible id) |

Events are streamed in fixed-size chunks (`ingest.chunk_size` in
`config.yaml`, default 10,000 rows) via `iter_events()`, so peak memory is
bounded regardless of file size -- nothing is loaded fully into memory.

`ingest.py` also provides `iter_orderbook_snapshots()` to parse the
orderbook file into `OrderbookSnapshot` records (top-N `(price, size)`
pairs per side) for ad-hoc inspection. These are **not** fed into
`book.py` -- see "Known limitation" below for why they can't be used to
validate the reconstruction.

### `book.py`

**Input**: one `Event` at a time, via `OrderBook.apply_event(event)`.

**Output** (queryable at any point during replay):

- `best_bid()` / `best_ask()` -- highest resting buy / lowest resting sell
  price, or `None` if that side is empty
- `depth(n)` -- top `n` `(price, aggregate_size)` levels per side, best-first
- `snapshot(index)` -- full book state (all price levels) at the given
  event index; also called automatically every `book.checkpoint_interval`
  events (default 100,000) so a long replay has bounded-memory checkpoints

### `replay.py`

Ties the two together: streams the configured message file through a
fresh `OrderBook` and reports rows read, events applied, per-event-type
counts, invariant violation counts, and events/sec.

## Assumptions (named in `config.yaml`)

| assumption | value | why |
|---|---|---|
| `data.orderbook_levels` | 10 | matches the LOBSTER file's own depth |
| `data.price_scale` | 10,000 | LOBSTER's fixed-point price convention |
| `book.checkpoint_interval` | 100,000 events | bounded-memory audit points for a long replay |
| `ingest.chunk_size` | 10,000 rows | streaming batch size; does not affect correctness |

## Invariants enforced (after every single event, not just at the end)

`OrderBook._check_invariants()` runs after every `apply_event()` call and
is O(1) per event (only the level/order touched by that event can newly
violate an invariant, so there's no need to rescan the whole book):

- **non-negative sizes**: if an order or level's size would go negative,
  it's clamped to 0 and counted (`negative_order_size` /
  `negative_level_size`) rather than silently allowed to drift
- **no crossed book**: `best_bid() >= best_ask()` is counted as
  `crossed_book`
- **best bid/ask consistency**: the top of the sorted price-level list must
  itself have positive size (`stale_best_bid` / `stale_best_ask`) --
  catches a desync between the sorted price list and the level-size dict
- **known order references**: a cancel/execute for an `order_id` never
  seen via `ADD` is counted as `unknown_order_id` rather than raised

Nothing is ever silently dropped. `OrderBook.violations` (a `Counter`) and
`OrderBook.event_type_counts` give a full accounting of every message
applied, which is what `checks/test_full_day.py` checks.

## Known limitation: LOBSTER Level-10 files can't be fully reconciled against their own reference book

This is the single most important finding from building this stage, so
it's worth stating precisely.

A LOBSTER "Level N" message file only contains messages for order flow
that affects the top-N depth. Resting liquidity deeper than that boundary
generates **no message at all** -- and when it later surfaces into the
top-N view (because the levels above it got cancelled or executed away),
it just appears in `orderbook_10.csv` with no corresponding `ADD` in
`message_10.csv`. This was confirmed by direct inspection: the price level
`5854000` first appears in `orderbook_10.csv` at row 147 with a resting
size of 50 shares, but grepping the entire message file for an `ADD` at
that price returns nothing.

Consequence: reconstructing the book from `message_10.csv` alone will
accumulate orders that, in reality, were later cancelled or executed
"out of view" (beyond depth 10) -- but since those removal messages
were never recorded either, our reconstruction has no way to know. Over a
full trading day this produces two specific, *expected* invariant counts
on this dataset:

- `unknown_order_id` (8,381): a cancel/execute referencing an order that
  was resting before the sample window started, or that entered the book
  beyond the visible top-10 boundary
- `crossed_book` (395,590): a stale, never-cancelled order from one of the
  above ends up as the extreme best bid or best ask, crossing a similarly
  stale order on the other side

Both counts are logged at `DEBUG` level (not `WARNING`) specifically
because they're expected and frequent on this data, not a sign of a bug --
see the comments in `book.py::_apply_reduce` and `_check_invariants`.
`checks/test_full_day.py` pins both to their exact observed value as a
regression guard, and separately asserts that the invariants which *would*
indicate a real bug (negative sizes, duplicate order ids) are always
exactly zero -- which they are.

Practical implication for later stages: features/labels derived from
`best_bid()`/`best_ask()`/`depth(n)` should be understood as "this
project's own reconstruction from the public Level-10 sample," not a
byte-for-byte match to LOBSTER's commercial reconstruction. For a paid,
higher-level LOBSTER file (e.g. Level 50) or full-depth ITCH, this
limitation would not apply.

## Benchmark

Full trading day (400,391 events), measured by `replay.py`:

```
throughput: 317,609 events/sec  (1.26s wall clock)
```

Run it yourself: `python replay.py` (from the project root, with the venv
active).
