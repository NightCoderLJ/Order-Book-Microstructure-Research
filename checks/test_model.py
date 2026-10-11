"""Unit tests for model.py: feature-group bookkeeping, train-set-only
normalization, and per-fold fit/predict behavior on small synthetic
tables."""

import numpy as np
import pandas as pd

from model import (
    FEATURE_GROUPS,
    feature_columns,
    load_config,
    make_linear_model,
    make_predict_fold_fn,
    normalize,
)


def test_feature_columns_covers_all_five_groups_with_no_duplicates():
    cols = feature_columns()
    expected = [c for cols_ in FEATURE_GROUPS.values() for c in cols_]
    assert cols == expected
    assert len(cols) == len(set(cols))


def test_feature_columns_excludes_named_group():
    cols = feature_columns(exclude_group="spread")
    assert "spread" not in cols
    assert "relative_spread" not in cols
    # every other group's columns are still present
    for name, group_cols in FEATURE_GROUPS.items():
        if name != "spread":
            assert set(group_cols) <= set(cols)


def test_normalize_uses_train_set_statistics_only():
    train_x = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
    test_x = pd.DataFrame({"a": [100.0, 200.0]})  # very different distribution

    norm_train, norm_test = normalize(train_x, test_x)

    # train column is z-scored against its own mean(2.0)/std(1.0)
    assert norm_train["a"].tolist() == [(v - 2.0) / 1.0 for v in [1.0, 2.0, 3.0]]
    # test column uses the *train's* mean/std, not its own -- so it lands
    # far outside [-1, 1] rather than being re-centered on itself.
    assert norm_test["a"].tolist() == [(v - 2.0) / 1.0 for v in [100.0, 200.0]]


def test_normalize_guards_zero_variance_train_column():
    train_x = pd.DataFrame({"a": [5.0, 5.0, 5.0]})
    test_x = pd.DataFrame({"a": [5.0, 6.0]})

    norm_train, norm_test = normalize(train_x, test_x)

    assert (norm_train["a"] == 0.0).all()  # (5 - 5) / 1, not / 0
    assert norm_test["a"].tolist() == [0.0, 1.0]


def _synthetic_df(n: int) -> pd.DataFrame:
    """n rows with every FEATURE_GROUPS column present (zero-filled,
    irrelevant to the test) except queue_imbalance_w5, which is set equal
    to label_h1 -- a perfect, easy-to-learn directional signal so a fit
    model should score non-flat test rows with the correct sign."""
    rng = np.random.default_rng(0)
    df = pd.DataFrame({col: 0.0 for col in feature_columns()}, index=range(n))
    label = pd.Series(rng.choice([-1.0, 0.0, 1.0], size=n))
    df["queue_imbalance_w5"] = label
    df["label_h1"] = label
    return df


def test_predict_fold_fn_scores_test_range_and_learns_the_signal():
    df = _synthetic_df(60)
    fold_fn = make_predict_fold_fn(load_config(), make_linear_model, feature_columns(), uses_class_index=False)

    train_range, test_range = range(0, 40), range(40, 60)
    score, pred_class, proba = fold_fn(df, train_range, test_range, "label_h1")

    assert len(score) == len(test_range)
    assert list(score.index) == list(df.iloc[test_range.start:test_range.stop].index)
    # the model should recover the (perfectly informative) sign of the signal
    label = df["label_h1"].iloc[test_range.start:test_range.stop]
    directional = label != 0.0
    assert (np.sign(score[directional]) == label[directional]).mean() > 0.9


def test_predict_fold_fn_leaves_missing_feature_rows_as_nan():
    df = _synthetic_df(60)
    df.loc[df.index[45], feature_columns()[0]] = float("nan")
    fold_fn = make_predict_fold_fn(load_config(), make_linear_model, feature_columns(), uses_class_index=False)

    score, pred_class, proba = fold_fn(df, range(0, 40), range(40, 60), "label_h1")

    assert pd.isna(score.iloc[5])  # row 45 is the 6th row of the test block
    assert pd.isna(pred_class.iloc[5])


def test_predict_fold_fn_returns_all_nan_when_train_has_one_class():
    df = _synthetic_df(60)
    df["label_h1"] = 0.0  # only one class in the whole table
    fold_fn = make_predict_fold_fn(load_config(), make_linear_model, feature_columns(), uses_class_index=False)

    score, pred_class, proba = fold_fn(df, range(0, 40), range(40, 60), "label_h1")

    assert score.isna().all()
    assert pred_class.isna().all()
