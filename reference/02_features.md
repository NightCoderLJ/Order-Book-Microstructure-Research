# Stage 2: Microstructure Features

## What problem does this stage solve?

`book.py` gives a live, queryable order book, but a forecasting model needs
a fixed-width numeric row per event, not a queryable object. This stage
(`features.py`) replays the day's events through a fresh `OrderBook` one at
a time and, at every event, reads off:

1. **top-of-book** fields -- the plain state of the book right now (best
   bid/ask, midprice, spread, top-level sizes)
2. **microstructure signals** -- order flow imbalance (OFI), queue
   imbalance, microprice, weighted microprice, liquidity pressure, order
   arrival rate, cancel rate, and trade imbalance, each aggregated over a
   trailing window of events

The result is one row per event, written to Parquet, that `labels.py` and
`model.py` (Phase 3+, not yet built) will read as their feature matrix.

## Inputs and outputs

**Input**: the same LOBSTER message file as `book.py`/`replay.py`, streamed
through `ingest.iter_events()` and applied to a fresh `OrderBook` one event
at a time via `OrderBook.apply_event()`.

**Output**: a `pandas.DataFrame` (and Parquet file at
`features.output_file` in `config.yaml`, default `data/features/features.parquet`),
one row per event, indexed by `(event_id, timestamp)`:

| field | type | meaning |
|---|---|---|
| `event_id` (index) | int | `Event.sequence_id` -- 1-indexed row position in the message file |
| `timestamp` (index) | float | `Event.timestamp_exchange` -- seconds since midnight |

**Top-of-book columns** (instantaneous, not windowed):

| column | meaning |
|---|---|
| `best_bid` / `best_ask` | LOBSTER integer price units (dollars x `price_scale`) |
| `midprice` | `(best_bid + best_ask) / 2` |
| `spread` | `best_ask - best_bid` |
| `relative_spread` | `spread / midprice` |
| `bid_size_l1` / `ask_size_l1` | aggregate resting size at the best bid/ask level |

All five are `NaN` for the handful of events at the very start of the day
before both sides have resting liquidity -- not an invariant violation,
just an empty side.

**Microstructure columns**: each of the 8 signals below appears as 4
columns, one per event-count window `w` in `features.windows`
(`config.yaml`, default `[5, 10, 20, 50]`): `{signal}_w5`, `_w10`, `_w20`,
`_w50`. 8 signals x 4 windows = 32 columns.

| signal | formula | aggregation over window `w` |
|---|---|---|
| `ofi` | Cont/Kukanov/Stoikov (2014) per-event order flow term `e_n = dW_bid - dW_ask` (see below) | rolling **sum** over the trailing `w` events |
| `queue_imbalance` | `(bid_size_l1 - ask_size_l1) / (bid_size_l1 + ask_size_l1)` | rolling **mean** |
| `microprice` | `(best_bid * ask_size_l1 + best_ask * bid_size_l1) / (bid_size_l1 + ask_size_l1)` | rolling **mean** |
| `liquidity_pressure` | same formula as `queue_imbalance`, but using aggregate size over the top `features.microprice_depth` levels per side instead of just L1 | rolling **mean** |
| `weighted_microprice` | same formula as `microprice`, but using the same depth-aggregated sizes as `liquidity_pressure` instead of L1 | rolling **mean** |
| `arrival_rate` | 1 if the event is `ADD` else 0 | rolling **mean** (= fraction of the trailing window that was an arrival) |
| `cancel_rate` | 1 if the event is `CANCEL_PARTIAL`/`CANCEL_FULL` else 0 | rolling **mean** |
| `trade_imbalance` | signed trade volume per event: `+size` if the aggressor bought, `-size` if the aggressor sold, `0` for non-trades (see below) | `sum(signed) / sum(abs(signed))` over the window; `0` if there were no trades in it |

`min_periods=1` on every rolling window: the first `w-1` rows of the day
are computed over however many events are actually available, rather than
`NaN`. This trades a slightly noisier value at the very start of the day
for never dropping rows -- acceptable here since the model (Phase 3+) will
window/split by time anyway and the first handful of rows out of 400,391
are immaterial.

## Why these formulas

- **`microprice`** (Stoikov): each side's own price is weighted by the
  *opposite* side's resting size, so a heavier bid queue pulls the
  fair-value price up toward the ask, and vice versa.
- **`liquidity_pressure` / `weighted_microprice`**: the same two formulas
  as `queue_imbalance`/`microprice`, just computed on aggregate size over
  `microprice_depth` levels instead of L1 only. This adds real information
  beyond the top level without needing a full depth-slope model, which is
  deferred to later ablations.
- **`ofi`**: the per-event contribution `e_n = dW_bid - dW_ask` is defined
  event-by-event, comparing the best bid/ask price and size immediately
  before vs. immediately after the event:
  - a **price improvement** on a side contributes its new size
  - a **same-price size change** contributes the size delta
  - a **price deterioration** (or that side's best level disappearing
    entirely) contributes the negative of its old size
  Summed (not averaged) over the window, since OFI is a flow quantity by
  definition -- an average would silently rescale it depending on the
  window length.
- **`trade_imbalance`**: `event.side` in the canonical `Event` schema
  always identifies the *resting* order's side (see `ingest.py`), not the
  aggressor's -- so a trade against a resting ask (`side == 'A'`) means the
  aggressor bought, and a trade against a resting bid (`side == 'B'`) means
  the aggressor sold. Both `EXECUTE_VISIBLE` and `EXECUTE_HIDDEN` count:
  LOBSTER reports `Direction`/`Price`/`Size` for hidden executions too, so
  both are real trade prints even though `book.py` treats `EXECUTE_HIDDEN`
  as a book-state no-op.
- **event-count windows, not time-based or depth-slope**: those variants
  are deferred until later ablations show they're needed. Event count is
  also the natural unit here, since
  `book.py` itself steps one event at a time.

## Assumptions (named in `config.yaml`)

| assumption | value | why |
|---|---|---|
| `features.windows` | `[5, 10, 20, 50]` | trailing event-count windows every microstructure signal is aggregated over |
| `features.microprice_depth` | 5 | depth (per side) used for `liquidity_pressure`/`weighted_microprice`; half of the available Level-10 depth |
| `features.output_file` | `data/features/features.parquet` | where the feature table is written (gitignored, regenerated by running `python features.py`) |

## Sanity checks / invariants

- `add_rolling_windows()` drops every raw (unwindowed) intermediate column
  before returning, so the output table contains exactly the top-of-book
  fields plus the windowed microstructure columns -- nothing else leaks
  through.
- `checks/test_features.py` hand-computes each of the 8 formulas above (plus
  top-of-book) on a synthetic 5-level book, and separately hand-verifies
  the rolling-window aggregation (sum vs. mean vs. ratio) on a small
  synthetic event sequence, independent of any real book replay.
- Because this dataset's Level-10 depth truncation produces a crossed book
  on ~99% of events (documented in
  [reference/01_ingestion_and_book.md](01_ingestion_and_book.md)), `spread` and
  `relative_spread` are frequently negative on this sample file. That is a
  faithful reflection of the underlying book state, not a bug in
  `features.py` -- see the sample row below, taken from one of the ~4,800
  events where the book happens to be uncrossed, for a "clean" example.

## Sample row

Event 4 (`event_id=4`, `timestamp=34200.025551909`), one of the early,
uncrossed events of the day. Prices are LOBSTER's native integer units
(dollars x 10,000, e.g. `5853300` = $585.33):

```
best_bid                  5853300.0
best_ask                  5859100.0
midprice                  5856200.0
spread                       5800.0
relative_spread           0.000990
bid_size_l1                    18.0
ask_size_l1                    18.0
queue_imbalance_w5               0.0
microprice_w5             5856200.0
weighted_microprice_w5    5857650.0
liquidity_pressure_w5           0.5
arrival_rate_w5                  1.0
cancel_rate_w5                   0.0
ofi_w5                           0.0
trade_imbalance_w5               0.0
... (same 8 signals repeated for w10/w20/w50)
```

Run it yourself: `python features.py` (from the project root, with the
venv active) -- prints row/column counts and the last row of the table,
then writes the full table to `features.output_file`.
