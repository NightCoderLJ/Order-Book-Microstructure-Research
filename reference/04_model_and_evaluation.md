# Stage 4: Model and Evaluation

## What problem does this stage solve?

`baselines.py`/`evaluate.py` (Stage 3) established that simple, non-fitted
heuristics carry some directional signal -- `imbalance_only` is the
strongest, with IC 0.05-0.08 depending on horizon. This stage asks
whether a *fitted* model, trained on the full microstructure feature set
instead of one feature at a time, can beat that. Two models are trained:

- **`model.py`**: a multinomial logistic regression ("linear baseline")
  and an XGBoost classifier, both trained on `features.py`'s feature
  table against `labels.py`'s `label_h{h}` classification labels, and
  retrained fresh on every walk-forward fold (no single fit on the whole
  day -- see "Walk-forward retraining" below).
- **`evaluate.py`** (extended): `evaluate_model()` scores a fitted model
  exactly like `evaluate_baselines()` scores a baseline -- same folds,
  same IC/accuracy definitions -- plus precision, recall, and AUC, which
  only make sense for a probabilistic classifier.

This stage also runs an **ablation**: retrain the XGBoost model with each
of the 5 named feature groups removed, one at a time, to see which
features are actually load-bearing for IC.

## Inputs and outputs

**Input**: `features.py`'s output table plus `labels.py`'s `label_h{h}`
columns (the same table `evaluate.py` already consumes for baselines).

**Model input features**: exactly the 5 microstructure feature groups
named in the project's scope, each expanded across
`features.py`'s 4 event-count windows (5/10/20/50) -- 22 columns total:

| group | columns |
|---|---|
| `ofi` | `ofi_w5`, `ofi_w10`, `ofi_w20`, `ofi_w50` |
| `queue_imbalance` | `queue_imbalance_w5/10/20/50` |
| `microprice` | `microprice_w5/10/20/50`, `weighted_microprice_w5/10/20/50` |
| `spread` | `spread`, `relative_spread` |
| `liquidity_pressure` | `liquidity_pressure_w5/10/20/50` |

Raw price *levels* (`best_bid`/`best_ask`/`midprice`) and
`arrival_rate`/`cancel_rate`/`trade_imbalance` are deliberately excluded:
the former are non-stationary across the day (the model would key off
"what price regime is this" rather than microstructure imbalance), and
the latter aren't part of the 5 named groups the ablation spec asks for.

**`model.py` output**: no persisted model artifacts -- this is a research
comparison, not a deployed system. `make_predict_fold_fn()` returns a
closure that fits a fresh model on one fold's train block and scores its
test block; `evaluate.evaluate_model()` drives that closure across all
folds and horizons.

**`evaluate.py` output** (`evaluate_model()`): one row per
`(model, horizon)` with mean IC, accuracy, precision, recall, and AUC
across the walk-forward folds' test windows -- directly comparable to
`evaluate_baselines()`'s output since both use identical folds.

## Assumptions and parameters (named in `config.yaml`)

| assumption | value | why |
|---|---|---|
| `model.logistic_regression.max_iter` | 1000 | enough iterations for `lbfgs` to converge on ~22 z-scored features |
| `model.logistic_regression.C` | 1.0 | sklearn default -- this is a baseline, not the model being tuned |
| `model.xgboost.n_estimators` | 150 | modest depth/tree count -- the model is refit ~180 times (2 models + 5 ablations, x3 horizons, x5 folds) |
| `model.xgboost.max_depth` | 4 | shallow trees, to limit overfitting on noisy tick data |
| `model.xgboost.learning_rate` | 0.05 | paired with 150 trees for a smooth, non-overfit fit |
| `model.xgboost.subsample` / `colsample_bytree` | 0.8 / 0.8 | standard row/column subsampling for variance reduction |

Class imbalance: `label_h{h}` is 79-95% "flat" (see
[reference/03_labels_and_baselines.md](03_labels_and_baselines.md)). Both
models are fit with `sklearn.utils.class_weight.compute_sample_weight
("balanced", y_train)` -- without it, a classifier converges to always
predicting "flat" and produces ~0 IC.

## Walk-forward retraining and train-set-only normalization

- **Same folds as the baselines**: `evaluate_model()` calls the identical
  `walk_forward_splits(len(df), n_splits, purge=max(horizons))` that
  `evaluate_baselines()` uses, so a model row and a baseline row for the
  same horizon are scored on identical test windows.
- **Retrained per fold**: `make_predict_fold_fn()`'s closure builds a
  *brand-new* model (`model_fn(config)`) for every `(horizon, fold)` pair
  and fits it only on that fold's purged train block -- there is no
  single model fit once on the whole day. This is more expensive
  (~180 total fits for the full comparison + ablation) but is the only
  way to keep every test fold's evaluation genuinely out-of-sample.
- **Train-set-only normalization**: `model.normalize()` z-scores both the
  train and test blocks using the *train* block's mean/std only (see
  `model.py`). Using test-set statistics (or the whole day's) would leak
  the test fold's distribution into the fit.

## Model IC vs. best baseline IC (full day, 5 walk-forward test folds)

```
              model  horizon     ic  accuracy  precision  recall    auc  best_baseline  best_baseline_ic  ic_delta_vs_best_baseline
logistic_regression        5 0.0425    0.0187     0.3689  0.3435 0.5739 imbalance_only            0.0622                    -0.0197
logistic_regression       10 0.0606    0.0317     0.3771  0.3548 0.5797 imbalance_only            0.0788                    -0.0182
logistic_regression      100 0.0472    0.0958     0.4415  0.3691 0.5694 imbalance_only            0.0512                    -0.0041
            xgboost        5 0.0633    0.0198     0.3737  0.4926 0.7805 imbalance_only            0.0622                     0.0011
            xgboost       10 0.0618    0.0324     0.4158  0.4894 0.7736 imbalance_only            0.0788                    -0.0169
            xgboost      100 0.0266    0.0835     0.4079  0.4173 0.7002 imbalance_only            0.0512                    -0.0246
```

Reproduce with `python model.py` (requires `data/features/features.parquet`
to already exist -- run `python features.py` first).

**Read, honestly**: neither fitted model beats `imbalance_only` (Stage
3's strongest baseline) at every horizon. XGBoost roughly ties it at
`h=5` (+0.0011 IC) but loses at `h=10` (-0.0169) and clearly loses at
`h=100` (-0.0246); logistic regression loses at every horizon. AUC is
well above the no-skill 0.5 baseline for both models (0.57-0.78),
confirming the models learned *something* real -- but that signal isn't
enough to out-rank a single well-chosen feature (`queue_imbalance_w5`)
used directly as a score. This is a genuine, unglamorous result, not a
bug: with only 22 features, no cross-sectional information, and heavy
class imbalance, a 22-feature model has limited room to beat one strong
univariate signal. It's reported here as-is, per this project's
correct-over-impressive principle.

## Ablation: XGBoost IC with one feature group removed

```
     removed_group  horizon     ic  ic_delta_vs_full_model
               ofi        5 0.0456                 -0.0177
               ofi       10 0.0538                 -0.0080
               ofi      100 0.0201                 -0.0066
   queue_imbalance        5 0.0566                 -0.0067
   queue_imbalance       10 0.0553                 -0.0065
   queue_imbalance      100 0.0242                 -0.0025
        microprice        5 0.0624                 -0.0009
        microprice       10 0.0507                 -0.0111
        microprice      100 0.0006                 -0.0260
            spread        5 0.0686                  0.0053
            spread       10 0.0756                  0.0137
            spread      100 0.0499                  0.0233
liquidity_pressure        5 0.0543                 -0.0090
liquidity_pressure       10 0.0634                  0.0016
liquidity_pressure      100 0.0134                 -0.0132
```

(Full-model XGBoost IC for reference: `h=5` 0.0633, `h=10` 0.0618, `h=100`
0.0266 -- the delta column above is `ablated_ic - full_model_ic`, so a
*negative* delta means removing that group hurt IC, and a *positive*
delta means removing it actually helped.)

**Which features actually mattered**:

- **`ofi`** matters most at the shortest horizon (`h=5`: removing it
  costs -0.0177 IC, the single largest drop in the table) and fades at
  longer horizons -- consistent with OFI being a fast, flow-based signal.
- **`microprice`** matters most at the longest horizon (`h=100`: removing
  it costs -0.0260 IC, the largest drop at that horizon, and collapses
  IC to nearly 0) -- the size-weighted fair-value price captures
  something aggregate/persistent that only shows up over more events.
- **`queue_imbalance`** contributes moderately and fairly evenly across
  all three horizons (-0.0067 to -0.0025) -- unsurprising, since it's
  also the strongest Stage 3 baseline on its own.
- **`liquidity_pressure`** has a mixed effect: it helps at `h=5` and
  `h=100` (removing it costs IC) but is neutral-to-slightly-helpful to
  remove at `h=10`.
- **`spread`** is the one group where removing it *improves* IC at every
  horizon (+0.0053 to +0.0233, the largest positive delta at `h=100`).
  This is consistent with Stage 3's finding that `spread_only` was the
  weakest baseline here (see
  [reference/03_labels_and_baselines.md](03_labels_and_baselines.md)):
  Level-10 depth truncation crosses the book on ~99% of events, so
  `spread` is frequently negative/degenerate on this dataset, and
  appears to add noise to the tree splits rather than signal.

Reproduce with `python model.py` (same run as above prints both tables).

## Sanity checks / invariants

- `checks/test_model.py`: `feature_columns()` covers all 5 groups with no
  duplicates and correctly excludes one group for ablation;
  `normalize()` is hand-checked to use train-set statistics only (a test
  block far outside the train distribution is transformed with the
  *train's* mean/std, not re-centered on itself) and to guard
  zero-variance train columns; `make_predict_fold_fn()`'s closure is
  checked to (a) return scores aligned exactly to the test fold's index,
  (b) recover a perfectly informative synthetic signal's sign >90% of the
  time, (c) leave rows with missing features as NaN rather than
  imputing, and (d) return all-NaN when a fold's train block has fewer
  than 2 label classes to fit against.
- `checks/test_evaluate.py`: `classifier_metrics()` is hand-computed on a
  small known (pred_class, proba, label) table, including the edge case
  where only one label class is present in a window (AUC is undefined
  there and correctly returns NaN); `evaluate_model()`'s output shape is
  checked end-to-end against a fake `predict_fold_fn`.
