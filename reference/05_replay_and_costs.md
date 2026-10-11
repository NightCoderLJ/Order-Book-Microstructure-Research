# Stage 5: Replay and Costs

## What problem does this stage solve?

`model.py`/`evaluate.py` (Stage 4) established the model's *predictive*
quality (IC, accuracy) on held-out, walk-forward test folds. This stage
asks the practical follow-up question: if you actually traded on those
predictions, what would happen? `backtest.py` replays the reconstructed
order book event by event, injecting the model's out-of-sample prediction
at each step, in two modes:

- **signal replay**: marks each directional prediction to the midprice
  `backtest.horizon` events later, with *no* execution model at all (no
  latency, fees, slippage, or fills). This isolates the model's raw
  prediction quality, in dollar terms, from any cost assumption.
- **paper-trading replay**: converts each prediction into an order
  (`backtest.order_type`: aggressive or passive), simulates its fill
  against the live book under named latency/cost assumptions, and marks
  filled positions to midprice the same `horizon` events later -- so the
  two modes are directly comparable, and any gap between them is
  attributable entirely to execution realism, not to a different
  prediction series or holding period.

## Inputs and outputs

**Input**: `features.py`'s output table (`data/features/features.parquet`)
plus `labels.py`'s `label_h{horizon}` column, for out-of-sample
predictions and midprices; and a fresh raw event stream (`ingest.py`)
replayed through a new `OrderBook` (`book.py`) for paper-trading's fill
simulation. The parquet's windowed columns don't carry the per-event
detail (exact executed size at one resting price) needed to simulate
queue position, so paper-trading mode re-walks the raw messages once.

**Out-of-sample predictions**: `generate_oos_predictions()` reuses
`evaluate.walk_forward_splits()` and `model.make_predict_fold_fn()`
(Stage 4, unchanged) to build one full-day, leak-free `(score,
pred_class)` series for `backtest.model` @ `backtest.horizon`, by
concatenating each fold's held-out test predictions instead of
aggregating them into IC. Rows before the first fold's test window (the
initial train-only block) have no prediction -- there is no leak-free way
to score them without training on future data.

**`run_replay()` output**: one row per realized (closed) trade --
`gen_idx`, `direction`, `entry_price_raw`, `exit_price_raw`,
`fee_dollars`, `net_pnl_dollars` -- plus `total_signals` (every event
where the prediction was non-flat, whether or not it became a trade).

**`compute_metrics()` output**: `n_trades`, `fill_rate`, `pnl_dollars`,
`sharpe`, `win_rate`, `max_drawdown_dollars`,
`slippage_adjusted_edge_dollars` -- computed identically for both modes.

## Assumptions and parameters (named in `config.yaml: backtest`)

| assumption | value | why |
|---|---|---|
| `model` / `horizon` | `xgboost` @ `5` | Stage 4's best-performing (model, horizon) pair (IC 0.0633) and its shortest horizon, giving the replay the freshest, most frequent signal |
| `position_size` | `100` shares | fixed size per trade -- no position-sizing or capital model (out of scope) |
| `latency_seconds` | `0.0005` (0.5ms) | one-way delay from signal to order arrival; LOBSTER carries no network-latency field, so this stands in for a colo-to-matching-engine round trip on 2012-era commodity hardware |
| `fee_per_share` | `$0.0005` | flat per-share fee on every fill, both order types -- a simplified stand-in for a real maker/taker schedule |
| `slippage_ticks` | `1` tick ($0.01) | adverse price movement applied only to aggressive fills, approximating impact/drift during the aggressor's execution window |
| `tick_size_raw` | `100` raw units | this sample's $0.01 tick, expressed in LOBSTER's `price_scale=10000` integer units -- the same fact `labels.py`'s `epsilon` is based on |
| `assumed_half_spread_raw` | `1200` raw units ($0.12) | see "Why spread-crossing cost is an assumption, not a live quote" below |
| `order_type` | `aggressive` | which fill model paper-trading uses (see below) |
| `trading_days_per_year` | `252` | used only to annualize Sharpe from this single day's per-trade PnL series |

## Why spread-crossing cost is an assumption, not a live quote

The obvious way to price an aggressive fill is "cross the live spread":
buy at `best_ask()`, sell at `best_bid()`. That was the first
implementation here, and it produced an impossible result: paper-trading
PnL of **+$38.4M** with a **100% win rate**. The cause is a real, already
-documented limitation of this dataset, not a new bug:

`book.py`'s Level-10 reconstruction has a **crossed book
(`best_bid() >= best_ask()`) on 395,590 of 400,391 events (98.8%)** -- see
[reference/01_ingestion_and_book.md](01_ingestion_and_book.md)'s
`crossed_book` invariant count. Digging into *when* the book is valid
shows it is worse than a uniform 1.2%: **every genuinely valid
(uncrossed) event falls in indices 3-5,121 -- the first ~1.3% of the day
-- and the book never becomes valid again for the remaining ~395k
events.** Once a stale, never-cancelled order from outside the tracked
top-10 depth becomes the extreme best bid or ask on one side, it stays
there, permanently crossing that side against the other. Since the
walk-forward model's earliest out-of-sample prediction (Stage 4's first
fold, after the purge) doesn't start until well past event 5,121, **every
signal this replay ever acts on occurs in the permanently-crossed
regime.** Crossing `best_ask()` directly there means crossing a stale,
out-of-money price that isn't real, executable liquidity -- hence the
fabricated risk-free profit.

`features.py`/`labels.py` already compute `midprice = (bb+ba)/2` from
this same crossed book throughout Stages 1-4, and that's an accepted
simplification there: IC is a *rank*-based, relative statistic, so a
consistently-computed midprice still carries a usable directional signal
even when its absolute level is corrupted by one stale side. Stage 5
claims something stronger -- a real fill at a real, executable price --
so the fix is:

- **Aggressive fills** price off the **midprice**, moved
  `assumed_half_spread_raw + slippage_ticks` against the order, using the
  *named, config-driven* `assumed_half_spread_raw` instead of the live
  (frequently invalid) `best_ask()`/`best_bid()`. `1200` raw units
  ($0.12) is the median observed spread during the day's ~1.2% valid
  window -- an empirically grounded, but likely **conservative/wide**
  stand-in for the other ~98.8% (real intraday AAPL in 2012, one of the
  most liquid stocks traded, typically quoted a materially tighter spread
  than its first few seconds of trading; there is no way to verify the
  true intraday spread on this specific Level-10 sample, so this is
  reported as a named assumption with that caveat attached, not
  presented as measured fact).
- **Passive fills** still queue behind the book's live `best_bid()` /
  `best_ask()` directly (see "Passive order simulation" below) -- this
  does *not* need the same guard, because resting behind a real order and
  waiting for real executions never fabricates a trade. A passive order
  queued behind a stale price simply never accumulates real executions
  and times out, which is a real (if uninteresting) outcome, not a
  fabricated one.
- **Exit pricing** (both modes) still marks to the live midprice, exactly
  as `labels.py` does -- unchanged from before this fix, since marking a
  position to the pipeline's existing fair-value convention isn't the
  same claim as fabricating an executed fill against it.

## Passive order simulation (queue position)

A passive order joins the back of its own side's queue at the current
`best_bid()`/`best_ask()`, recording the aggregate size already resting
there (`queue_ahead`). It fills once the cumulative size of
`EXECUTE_VISIBLE` events at that exact price and side reaches
`queue_ahead + position_size` -- i.e., enough real order flow has traded
through everyone ahead of it. It is dropped, unfilled ("queue-position
loss"), if that price level disappears from the book entirely before
that happens, or if `horizon` events pass without a fill.
`EXECUTE_HIDDEN` events don't count: `book.py` already treats them as a
no-op against the displayed book (hidden liquidity was never part of any
displayed queue), so they can't consume one either.

## Signal replay vs. paper-trading (full day, `xgboost` @ h=5)

```
                                signal replay   paper-trading (aggressive)   paper-trading (passive)
n_trades                              123,200                       115,144                          7
fill_rate                              1.0000                        0.9346                    0.0001
pnl_dollars                          3,448.00                 -1,500,998.70                  -3,017.35
sharpe                                201.31                       -110,186.68                  -139.20
win_rate                               0.0717                        0.0000                     0.0000
max_drawdown_dollars                  -180.50                    -1,500,985.65                 -2,827.30
slippage_adjusted_edge_dollars          0.0280                         -13.04                    -431.05
```

Reproduce with `python backtest.py` (requires `data/features/features.parquet`
to exist -- run `python features.py` first; also retrains `xgboost` on 5
walk-forward folds, ~15s). The table's `aggressive` column is
`backtest.py`'s printed output at the current `config.yaml` default
(`order_type: aggressive`); the `passive` column is the same run with
`order_type: passive`, included here for comparison.

**Read, honestly**:

- **Signal replay's raw edge is real but tiny.** $3,448 total profit
  across 123,200 signals, or 2.8 cents per 100-share trade -- consistent
  with Stage 4's modest IC (0.0633 at h=5). `win_rate` of only 7.2% looks
  alarming next to a positive `sharpe`, but is expected: most 5-event
  midprice moves are near zero regardless of direction (labels.py: ~95%
  of h=5 moves are "flat"), so most signal trades post a tiny loss or
  wash while a smaller number of correctly-called larger moves carry the
  total.
- **Paper-trading (aggressive) is unambiguously unprofitable, and by a
  wide margin.** Every single trade in the aggressive run loses money --
  `win_rate` is exactly 0.0. The reason is directly quantifiable: the
  assumed round-trip cost (`assumed_half_spread_raw + slippage_ticks +
  fee`, ~$0.13/share) is roughly **460x** the raw signal's average edge
  per share (~$0.00028, from signal replay's $0.028/trade over 100
  shares). No realistic fee schedule closes a 460x gap. This is the
  single most important result in this stage: **a statistically real,
  positive-IC signal is not automatically a profitable strategy** --
  transaction costs this dataset can't even measure precisely (see the
  spread-assumption caveat above) already dominate it by two orders of
  magnitude.
- **Paper-trading (passive) has an almost-zero fill rate (0.01%, 7 of
  123,200 signals).** This is also a real finding, not a bug: `best_bid`/
  `best_ask` aggregate sizes on a stock as heavily traded as AAPL are
  typically large, and only 5 events' worth of real executions occur
  within the `horizon=5` window this replay allows before timing out --
  nowhere near enough volume to trade through a realistic queue in that
  short a wait. A passive strategy on this signal would need either a
  much longer patience window or to target thinner, less-competed price
  levels, neither of which this Phase 5 scope models.
- **Sharpe/max-drawdown numbers should be read as directional, not
  literal.** The annualization (`sqrt(n_trades * 252)`) assumes a similar
  number of similarly-distributed trades recur on other days -- a named
  simplification for a single-day dataset, not a claim about real
  year-over-year variance.

## Sanity checks / invariants

- `checks/test_backtest.py` hand-computes, on small synthetic event
  streams: signal-replay entry/exit marking with no costs; aggressive
  fills paying the assumed half-spread, slippage, and fee; passive fills
  accumulating queue position correctly and filling once enough volume
  trades through; passive orders losing queue position when their price
  level empties before filling; aggressive fills pricing off the
  midprice (not the live spread) even when the synthetic book is
  crossed, and declining only when there's no two-sided liquidity at all;
  `compute_metrics()`'s PnL/win-rate/drawdown arithmetic against
  hand-computed expected values; and `generate_oos_predictions()`'s
  fold-concatenation alignment against a fake `predict_fold_fn`.
