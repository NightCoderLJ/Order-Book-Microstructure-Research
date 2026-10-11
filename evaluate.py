"""
evaluate.py -- walk-forward (purged) evaluation of the baselines in
baselines.py, and (Phase 4) the fitted models in model.py, against the
labels in labels.py.

Takes in: features.py's output table (data/features/features.parquet),
one row per event, in time/event order, with labels.py's `label_h{h}`
columns added.

Produces: for baselines (evaluate_baselines), one row per
(baseline, horizon) with the Spearman IC (rank correlation between the
predicted score and the realized forward return) and directional
accuracy (sign(score) vs label). For fitted models (evaluate_model), one
row per (model, horizon) with the same IC/accuracy plus precision,
recall, and AUC (macro-averaged over the 3 label classes) -- the extra
classifier metrics evaluate_baselines doesn't need, since the baselines
are closed-form formulas, not probabilistic classifiers. Every metric is
averaged across the walk-forward folds' test windows.

Why walk-forward + purging, not a random split: time-based splits only
(a random shuffle would let a baseline "predict" the past from the
future) and identical evaluation windows across baselines/model keep the
comparison fair. Purging drops the tail of each
fold's train block whose forward-looking label would otherwise read past
the train/test boundary into the test period -- see
reference/03_labels_and_baselines.md.
"""

from typing import Callable, List, Tuple

import pandas as pd
import yaml
from sklearn.metrics import precision_score, recall_score, roc_auc_score

from baselines import BASELINES
from labels import add_labels, forward_return


def load_config(path: str = "config.yaml") -> dict:
    """Loads the small set of named assumptions this pipeline depends on."""
    with open(path) as f:
        return yaml.safe_load(f)


def walk_forward_splits(n_rows: int, n_splits: int, purge: int) -> List[Tuple[range, range]]:
    """Expanding-window walk-forward split of a single chronological event
    sequence into `n_splits` folds -- no shuffling, no randomness.

    Divides [0, n_rows) into n_splits+1 equal-width contiguous blocks in
    time order. Fold i's test block is block i+1; its train block is every
    block up to and including block i, with the last `purge` rows dropped
    immediately before the test boundary (so no train-set label's forward
    window reads into the test set). The final fold's test block absorbs
    any remainder rows from the integer division, so every row is covered
    exactly once as a test row (except the first block, which is
    train-only across every fold, and the last `purge` rows of every
    train block, which are purged).

    Returns a list of (train_range, test_range) row-position pairs (not
    label values), one per fold, in fold order.
    """
    block = n_rows // (n_splits + 1)
    folds = []
    for i in range(1, n_splits + 1):
        train_end = i * block
        test_start = train_end
        test_end = (i + 1) * block if i < n_splits else n_rows
        train_range = range(0, max(0, train_end - purge))
        test_range = range(test_start, test_end)
        folds.append((train_range, test_range))
    return folds


def rank_ic(pred: pd.Series, target: pd.Series) -> float:
    """Spearman IC between a baseline's predicted score and a continuous
    target (the realized forward return), restricted to rows where both
    are defined.

    Computed as the Pearson correlation of ranks rather than via
    pandas' method='spearman', which requires scipy -- not a dependency
    this project otherwise needs. Ranking then Pearson-correlating is
    mathematically identical to Spearman's rho.
    """
    valid = pred.notna() & target.notna()
    if valid.sum() < 2:
        return float("nan")
    return pred[valid].rank().corr(target[valid].rank())


def directional_accuracy(pred: pd.Series, label: pd.Series) -> float:
    """Fraction of rows where sign(pred) matches the label, restricted to
    rows with a non-NaN label and a non-zero predicted sign.

    Rows where the baseline predicts exactly 0 (no directional call) are
    excluded rather than scored as wrong -- "no prediction" isn't the same
    claim as "predicted flat", and scoring it as an automatic miss would
    unfairly penalize baselines (e.g. spread_only when spread is exactly
    0) that sometimes emit a null signal.
    """
    pred_sign = pred.apply(lambda x: 1.0 if x > 0 else (-1.0 if x < 0 else 0.0))
    valid = label.notna() & (pred_sign != 0)
    if valid.sum() == 0:
        return float("nan")
    return (pred_sign[valid] == label[valid]).mean()


def evaluate_baselines(df: pd.DataFrame, horizons: List[int], n_splits: int) -> pd.DataFrame:
    """Runs every baseline in BASELINES against every horizon in
    `horizons`, scored on each walk-forward fold's test window, and
    returns one row per (baseline, horizon) with the mean IC and mean
    directional accuracy across folds.

    `df` must already have `label_h{h}` columns for every h in `horizons`
    (see labels.add_labels()). Purge length is derived here as
    max(horizons) rather than taken from config, so it can never fall out
    of sync with the labels it's protecting: the longest horizon's label
    is the one whose forward window reaches furthest past the boundary.
    """
    purge = max(horizons)
    folds = walk_forward_splits(len(df), n_splits, purge)

    rows = []
    for name, baseline_fn in BASELINES.items():
        for h in horizons:
            fwd_ret = forward_return(df["midprice"], h)
            label = df[f"label_h{h}"]
            pred = baseline_fn(df, h)

            fold_ics, fold_accs = [], []
            # train_range is unused: every baseline here is a closed-form
            # formula, not a fitted model, so there is nothing to train on
            # this fold's train block. The split still produces it so
            # model.py (Phase 4) can reuse walk_forward_splits() unchanged
            # for a fitted model evaluated on the identical folds.
            for _train_range, test_range in folds:
                test_slice = slice(test_range.start, test_range.stop)
                fold_ics.append(rank_ic(pred.iloc[test_slice], fwd_ret.iloc[test_slice]))
                fold_accs.append(directional_accuracy(pred.iloc[test_slice], label.iloc[test_slice]))

            rows.append({
                "baseline": name,
                "horizon": h,
                "ic": pd.Series(fold_ics, dtype=float).mean(),
                "accuracy": pd.Series(fold_accs, dtype=float).mean(),
                "n_folds": len(folds),
            })
    return pd.DataFrame(rows)


_CLASSES = (-1.0, 0.0, 1.0)


def classifier_metrics(pred_class: pd.Series, proba: pd.DataFrame, label: pd.Series) -> Tuple[float, float, float]:
    """Precision, recall (macro-averaged over the 3 label classes), and
    AUC (one-vs-rest, macro-averaged) for one fold's test window.

    Unlike rank_ic/directional_accuracy (which score a single continuous
    number and so also apply to the non-probabilistic baselines),
    precision/recall/AUC are classifier-specific: they need a discrete
    predicted class (precision/recall) or a full probability distribution
    (AUC), which only model.py's fitted models produce.

    Rows with a NaN label or NaN prediction are excluded (same NaN
    handling as rank_ic/directional_accuracy). Returns NaN for any metric
    that can't be computed on this window (e.g. fewer than 2 label
    classes present, so AUC is undefined).
    """
    class_valid = label.notna() & pred_class.notna()
    if class_valid.sum() == 0:
        precision = recall = float("nan")
    else:
        precision = precision_score(
            label[class_valid], pred_class[class_valid], labels=list(_CLASSES), average="macro", zero_division=0
        )
        recall = recall_score(
            label[class_valid], pred_class[class_valid], labels=list(_CLASSES), average="macro", zero_division=0
        )

    proba_valid = label.notna() & proba.notna().all(axis=1)
    if proba_valid.sum() < 2 or label[proba_valid].nunique() < 2:
        auc = float("nan")
    else:
        auc = roc_auc_score(
            label[proba_valid], proba.loc[proba_valid, list(_CLASSES)].values,
            multi_class="ovr", average="macro", labels=list(_CLASSES),
        )
    return precision, recall, auc


def evaluate_model(
    df: pd.DataFrame,
    horizons: List[int],
    n_splits: int,
    predict_fold_fn: Callable[[pd.DataFrame, range, range, str], Tuple[pd.Series, pd.Series, pd.DataFrame]],
    model_name: str,
) -> pd.DataFrame:
    """Runs one fitted model (via model.py's make_predict_fold_fn()
    closure) against every horizon in `horizons`, retraining fresh on
    each walk-forward fold's purged train block and scoring its test
    block, mirroring evaluate_baselines()'s fold/scoring loop exactly so
    the two are directly comparable.

    `predict_fold_fn(df, train_range, test_range, label_col)` must return
    (score, pred_class, proba) for the fold's test block -- see
    model.py's make_predict_fold_fn(). `df` must already have `label_h{h}`
    columns for every h in `horizons` (see labels.add_labels()).

    Returns one row per (model_name, horizon) with the mean IC, accuracy,
    precision, recall, and AUC across folds -- the same purge/fold
    structure as evaluate_baselines() (purge = max(horizons), n_splits
    folds), so a model row and a baseline row for the same horizon were
    scored on identical test windows.
    """
    purge = max(horizons)
    folds = walk_forward_splits(len(df), n_splits, purge)

    rows = []
    for h in horizons:
        label_col = f"label_h{h}"
        fwd_ret = forward_return(df["midprice"], h)

        fold_ics, fold_accs, fold_precisions, fold_recalls, fold_aucs = [], [], [], [], []
        for train_range, test_range in folds:
            score, pred_class, proba = predict_fold_fn(df, train_range, test_range, label_col)
            test_slice = slice(test_range.start, test_range.stop)
            label = df[label_col].iloc[test_slice]

            fold_ics.append(rank_ic(score, fwd_ret.iloc[test_slice]))
            fold_accs.append(directional_accuracy(score, label))
            precision, recall, auc = classifier_metrics(pred_class, proba, label)
            fold_precisions.append(precision)
            fold_recalls.append(recall)
            fold_aucs.append(auc)

        rows.append({
            "model": model_name,
            "horizon": h,
            "ic": pd.Series(fold_ics, dtype=float).mean(),
            "accuracy": pd.Series(fold_accs, dtype=float).mean(),
            "precision": pd.Series(fold_precisions, dtype=float).mean(),
            "recall": pd.Series(fold_recalls, dtype=float).mean(),
            "auc": pd.Series(fold_aucs, dtype=float).mean(),
            "n_folds": len(folds),
        })
    return pd.DataFrame(rows)


def main():
    config = load_config()
    df = pd.read_parquet(config["features"]["output_file"])
    df = add_labels(df, config["labels"]["horizons"], config["labels"]["epsilon"])

    results = evaluate_baselines(
        df,
        horizons=config["labels"]["horizons"],
        n_splits=config["evaluate"]["n_splits"],
    )
    pd.set_option("display.float_format", lambda x: f"{x:.4f}")
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
