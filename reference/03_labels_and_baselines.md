# Stage 3: Labels and Baselines

## What problem does this stage solve?

`features.py` gives one row per event with the current book state, but a
forecasting model needs a ground-truth target and something to be
evaluated against. This stage adds three modules:

- **`labels.py`** turns the event-level `midprice` column into a
  classification label: did the price go up, down, or stay flat, `h`
  events later?
- **`baselines.py`** implements five simple, non-fitted heuristics --
  last-direction, momentum, mean-reversion, spread-only, imbalance-only --
  that the Phase 4 model must beat.
- **`evaluate.py`** splits the day into walk-forward (purged), time-based
  folds and scores every baseline, at every horizon, on the folds' test
  windows, reporting IC and directional accuracy.

## Inputs and outputs

**Input**: `features.py`'s output table (`data/features/features.parquet`),
one row per event in event order, specifically its `midprice`, `spread`,
and `queue_imbalance_w5` columns.

**`labels.py` output**: `add_labels()` returns the same table with one
`label_h{h}` column added per horizon `h` in `config.yaml: labels.horizons`:

| column | type | meaning |
|---|---|---|
| `label_h5` / `label_h10` / `label_h100` | float in `{-1.0, 0.0, 1.0, NaN}` | `sign(midprice(t+h) - midprice(t))` if the relative move exceeds `epsilon`, else `0.0` ("flat"); `NaN` for the last `h` events of the day, where `midprice(t+h)` doesn't exist yet |

Label definition, precisely:

```
ret   = midprice(t+h) / midprice(t) - 1
label =  1  if ret >  epsilon
      = -1  if ret < -epsilon
      =  0  otherwise
      = NaN if midprice(t+h) doesn't exist (end of day)
```

**`baselines.py` output**: each of the 5 functions in `BASELINES` takes the
feature table and a horizon `h`, and returns a continuous predicted score
per event (positive = predicts price up, negative = predicts price down),
using only information at or before that event -- no lookahead.

**`evaluate.py` output**: `evaluate_baselines()` returns one row per
`(baseline, horizon)` pair with the mean IC and mean directional accuracy
across the walk-forward folds' test windows (see table below).

## Assumptions and parameters (named in `config.yaml`)

| assumption | value | why |
|---|---|---|
| `labels.horizons` | `[5, 10, 100]` | event-count forecast horizons (event-based, matching `features.py`'s event-count windows -- not time-based) |
| `labels.epsilon` | `0.00001` (1e-5, relative return) | this sample's tick size is $0.01 (100 raw price units at `price_scale=10000`) and its mean midprice is ~$585.09, so one tick is a ~1.71e-5 relative move. `epsilon=1e-5` (~0.58 ticks) keeps sub-tick noise out of the directional classes without requiring a full tick move to count as "moved" -- see the resulting class balance below |
| `evaluate.n_splits` | `5` | number of walk-forward folds; with ~400k events this gives 5 test windows of ~67k events each, enough per fold for a stable IC estimate |
| purge length | `max(labels.horizons)` = `100` | *not* a separate config value -- derived in `evaluate.py` from `labels.horizons` so it can never drift out of sync with the labels it protects against leakage |

Resulting label class balance (from `python labels.py` on the full day,
400,391 events):

| horizon | -1 (down) | 0 (flat) | +1 (up) |
|---|---|---|---|
| h=5 | 2.6% | 94.9% | 2.5% |
| h=10 | 4.2% | 92.0% | 3.8% |
| h=100 | 11.6% | 79.2% | 9.3% |

As expected, the flat class shrinks and the directional classes grow as
the horizon lengthens -- more events means more room for the price to
move a full tick or more.

## Walk-forward (purged) split

`evaluate.walk_forward_splits(n_rows, n_splits, purge)` divides the day's
events into `n_splits + 1` equal-width, contiguous, time-ordered blocks.
Fold `i`'s test block is block `i+1`; its train block is every block up
to and including block `i`. This is an **expanding-window** walk-forward
split -- no shuffling, per the project's time-based-splits-only rule.

**Purging**: the last `purge` rows of each fold's train block are dropped
before the test boundary. Without this, a train-set row's label (which
looks `h` events into the *future*) could be computed from midprice values
that fall inside the test block -- a direct leak of test-period
information into what's nominally the train set. Purge length is set to
`max(labels.horizons) = 100`, the longest lookahead any label uses, so no
label at the boundary can reach past the purged region.

Baselines here are closed-form formulas, not fitted models, so they don't
actually use the train block -- but `walk_forward_splits()` still returns
it so Phase 4's `model.py` can reuse the identical folds for a fair,
leakage-free comparison against these same baselines.

## Baseline definitions

All five share the signature `fn(df, h) -> pd.Series` (a continuous score
per event), even though three of them ignore `h`:

| baseline | formula | predicts |
|---|---|---|
| `last_direction` | `midprice(t) - midprice(t-1)` | the most recent single-event move continues |
| `momentum` | `midprice(t) - midprice(t-h)` | the realized h-event return repeats over the next h events |
| `mean_reversion` | `-(midprice(t) - midprice(t-h))` | the realized h-event return reverses |
| `spread_only` | `spread` (unmodified) | a single-feature control -- does the current spread alone carry directional signal? |
| `imbalance_only` | `queue_imbalance_w5` (unmodified) | a heavier bid queue predicts the price rises (queue-imbalance literature) |

`spread_only` is not a noise control on this dataset the way it would be
on a normal book: Level-10 depth truncation crosses the book on ~99% of
events (see [reference/01_ingestion_and_book.md](01_ingestion_and_book.md)), so
`spread` is frequently negative and its sign may itself carry information
about which side is more aggressively priced.

## Evaluation metrics

- **IC** (`rank_ic`): Spearman rank correlation between the baseline's
  score and the realized forward return (`labels.forward_return()`), on
  rows where both are defined. Computed as the Pearson correlation of
  ranks rather than via pandas' `method='spearman'` (which needs `scipy`,
  not otherwise a dependency here) -- ranking then Pearson-correlating is
  the same statistic.
- **Accuracy** (`directional_accuracy`): fraction of rows where
  `sign(score)` matches `label_h{h}`, restricted to rows with a non-NaN
  label and a *non-zero* predicted sign (a baseline emitting exactly 0 is
  "no call", not "predicted flat", so it's excluded rather than scored as
  a miss). Because ~79-95% of labels are flat (0) and a continuous score
  is rarely exactly 0, most baselines' accuracy is dominated by
  automatically-wrong calls on flat-labeled rows -- **IC, not accuracy, is
  the primary metric for this stage**, consistent with the project's
  IC-based comparison.

## Baseline IC / accuracy table (full day, 5 walk-forward test folds)

```
      baseline  horizon      ic  accuracy  n_folds
last_direction        5 -0.0519    0.0797        5
last_direction       10 -0.0319    0.1351        5
last_direction      100 -0.0225    0.3522        5
      momentum        5 -0.0229    0.0823        5
      momentum       10  0.0052    0.1316        5
      momentum      100 -0.0152    0.3492        5
mean_reversion        5  0.0229    0.1302        5
mean_reversion       10 -0.0052    0.1717        5
mean_reversion      100  0.0152    0.3623        5
   spread_only        5  0.0019    0.0180        5
   spread_only       10  0.0045    0.0311        5
   spread_only      100  0.0004    0.1002        5
imbalance_only        5  0.0622    0.0201        5
imbalance_only       10  0.0788    0.0349        5
imbalance_only      100  0.0512    0.1051        5
```

Reproduce with `python evaluate.py` (requires `data/features/features.parquet`
to already exist -- run `python features.py` first).

Read: `last_direction` and `momentum` have *negative* IC at short horizons
(short-horizon continuation doesn't hold here -- `mean_reversion`, their
mirror image, has the matching positive IC instead). `imbalance_only` has
the strongest IC of all five baselines at every horizon (0.05-0.08) --
this is the number Phase 4's model needs to beat, not just the
conventional "momentum" baseline, since it's the strongest one measured
here.

## Sanity checks / invariants

- `checks/test_labels.py` hand-computes `forward_return()` and
  `label_horizon()` on a small synthetic midprice series with a known
  epsilon crossing.
- `checks/test_baselines.py` hand-computes each of the 5 baseline formulas
  on a small synthetic feature table, and checks all 5 share the same
  callable signature.
- `checks/test_evaluate.py` checks `walk_forward_splits()` produces
  non-overlapping, time-ordered, train-strictly-before-test folds, that
  purging only shrinks the train block (never the test block), hand-
  computes `rank_ic()`/`directional_accuracy()` on tiny known series, and
  checks `evaluate_baselines()`'s output shape end-to-end on a synthetic
  table.
