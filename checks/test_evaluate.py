"""Unit tests for evaluate.py: the walk-forward/purge split, the IC and
accuracy metrics (hand-computed), and an end-to-end shape check of
evaluate_baselines() on a small synthetic table."""

import pandas as pd

from evaluate import (
    classifier_metrics,
    directional_accuracy,
    evaluate_baselines,
    evaluate_model,
    rank_ic,
    walk_forward_splits,
)
from labels import add_labels


def test_walk_forward_splits_cover_time_in_order_with_no_overlap():
    folds = walk_forward_splits(n_rows=100, n_splits=4, purge=0)

    assert len(folds) == 4
    # every fold's test block starts exactly where the previous fold's
    # test block ended -- test blocks partition the tail of the sequence
    # in time order, never overlapping or going backwards.
    for (_, test_a), (_, test_b) in zip(folds, folds[1:]):
        assert test_b.start == test_a.stop
    # the last fold's test block absorbs the remainder up to n_rows.
    assert folds[-1][1].stop == 100


def test_walk_forward_splits_train_is_always_strictly_before_test():
    folds = walk_forward_splits(n_rows=100, n_splits=4, purge=0)
    for train_range, test_range in folds:
        assert train_range.stop <= test_range.start


def test_walk_forward_splits_purge_shrinks_train_tail():
    folds_no_purge = walk_forward_splits(n_rows=100, n_splits=4, purge=0)
    folds_purged = walk_forward_splits(n_rows=100, n_splits=4, purge=5)

    for (train_a, test_a), (train_b, test_b) in zip(folds_no_purge, folds_purged):
        assert test_a == test_b  # purging only trims train, never test
        assert train_b.stop == train_a.stop - 5


def test_rank_ic_is_one_for_perfectly_monotonic_series():
    pred = pd.Series([1.0, 2.0, 3.0, 4.0])
    target = pd.Series([10.0, 20.0, 30.0, 40.0])
    assert rank_ic(pred, target) == 1.0


def test_rank_ic_is_negative_one_for_inverted_series():
    pred = pd.Series([1.0, 2.0, 3.0, 4.0])
    target = pd.Series([40.0, 30.0, 20.0, 10.0])
    assert rank_ic(pred, target) == -1.0


def test_rank_ic_ignores_nan_rows():
    pred = pd.Series([1.0, 2.0, float("nan"), 4.0])
    target = pd.Series([10.0, 20.0, 30.0, 40.0])
    assert rank_ic(pred, target) == 1.0


def test_directional_accuracy_hand_computed():
    pred = pd.Series([1.0, -1.0, 2.0, -3.0])
    label = pd.Series([1.0, -1.0, -1.0, -1.0])
    # 3 of 4 predicted signs match the label -> 0.75
    assert directional_accuracy(pred, label) == 0.75


def test_directional_accuracy_excludes_zero_predictions_and_nan_labels():
    pred = pd.Series([0.0, 1.0, -1.0])
    label = pd.Series([1.0, 1.0, float("nan")])
    # row 0 excluded (pred sign is 0), row 2 excluded (label is NaN);
    # only row 1 counted, and it matches -> accuracy 1.0
    assert directional_accuracy(pred, label) == 1.0


def test_evaluate_baselines_output_shape():
    # 8 events, small horizons, enough rows for 2 folds after purging.
    df = pd.DataFrame({
        "midprice": [100.0, 101.0, 99.0, 102.0, 103.0, 101.0, 104.0, 105.0],
        "spread": [1.0, 1.1, 0.9, 1.2, 1.0, 0.8, 1.3, 1.1],
        "queue_imbalance_w5": [0.1, 0.2, -0.1, 0.3, 0.0, -0.2, 0.4, -0.1],
    })
    df = add_labels(df, horizons=[1, 2], epsilon=0.01)

    out = evaluate_baselines(df, horizons=[1, 2], n_splits=2)

    assert len(out) == 5 * 2  # 5 baselines x 2 horizons
    assert set(out.columns) == {"baseline", "horizon", "ic", "accuracy", "n_folds"}
    assert set(out["horizon"]) == {1, 2}


def test_classifier_metrics_hand_computed():
    # 4 rows: predicted class matches label in rows 0,1,3 (2 correct
    # "up"/"down" calls + 1 correct "flat"); row 2 is a miss.
    pred_class = pd.Series([1.0, -1.0, 1.0, 0.0])
    label = pd.Series([1.0, -1.0, -1.0, 0.0])
    proba = pd.DataFrame({
        -1.0: [0.1, 0.7, 0.1, 0.2],
        0.0: [0.1, 0.2, 0.2, 0.6],
        1.0: [0.8, 0.1, 0.7, 0.2],
    })

    precision, recall, auc = classifier_metrics(pred_class, proba, label)

    assert 0.0 < precision <= 1.0
    assert 0.0 < recall <= 1.0
    assert 0.0 <= auc <= 1.0


def test_classifier_metrics_returns_nan_when_only_one_class_present():
    pred_class = pd.Series([0.0, 0.0, 0.0])
    label = pd.Series([0.0, 0.0, 0.0])
    proba = pd.DataFrame({-1.0: [0.1, 0.1, 0.1], 0.0: [0.8, 0.8, 0.8], 1.0: [0.1, 0.1, 0.1]})

    precision, recall, auc = classifier_metrics(pred_class, proba, label)

    assert pd.isna(auc)  # AUC undefined with a single class in the window
    # macro-averaged over all 3 labels: the "flat" class scores 1.0 (every
    # call correct) but the never-seen -1/+1 classes score 0
    # (zero_division=0), so the 3-class macro average is 1/3, not 1.0.
    assert precision == 1.0 / 3
    assert recall == 1.0 / 3


def test_evaluate_model_output_shape():
    df = pd.DataFrame({
        "midprice": [100.0, 101.0, 99.0, 102.0, 103.0, 101.0, 104.0, 105.0],
    })
    df = add_labels(df, horizons=[1, 2], epsilon=0.01)

    def fake_predict_fold_fn(df, train_range, test_range, label_col):
        test_slice = slice(test_range.start, test_range.stop)
        idx = df.index[test_slice]
        score = pd.Series(1.0, index=idx)
        pred_class = pd.Series(1.0, index=idx)
        proba = pd.DataFrame({-1.0: 0.1, 0.0: 0.1, 1.0: 0.8}, index=idx)
        return score, pred_class, proba

    out = evaluate_model(df, horizons=[1, 2], n_splits=2, predict_fold_fn=fake_predict_fold_fn, model_name="fake")

    assert len(out) == 2  # 1 model x 2 horizons
    assert set(out.columns) == {"model", "horizon", "ic", "accuracy", "precision", "recall", "auc", "n_folds"}
    assert set(out["model"]) == {"fake"}
