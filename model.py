"""model.py -- Phase 4 forecasting models: a multinomial logistic
regression (linear baseline) and an XGBoost classifier, both trained on
features.py's microstructure feature table against labels.py's
label_h{h} classification labels.

Takes in: features.py's output table (data/features/features.parquet)
with labels.py's label_h{h} columns added, one row per event in event
order -- the same table baselines.py/evaluate.py score against.

Produces: no persisted model artifacts (this is a research comparison,
not a deployed system). model.py's job is to fit-and-score one
walk-forward fold at a time -- see make_predict_fold_fn() -- so
evaluate.py's evaluate_model() can retrain fresh on every fold's purged
train block and score its held-out test block, exactly like
evaluate_baselines() does for the non-fitted baselines. Running
`python model.py` prints: (1) model IC/accuracy/precision/recall/AUC per
horizon next to the best baseline's IC, and (2) an ablation table showing
each named feature group's IC contribution.

Why retrain per fold instead of fitting once on the whole day: walk-forward
retraining across the same splits the baselines use, and normalizing
features on train-set statistics only, both would leak future information
into the model if it were fit once on the entire day up front.
"""

from typing import Callable, Dict, List, Optional, Tuple

import pandas as pd
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier


def load_config(path: str = "config.yaml") -> dict:
    """Loads the small set of named assumptions this pipeline depends on."""
    with open(path) as f:
        return yaml.safe_load(f)


# -- feature groups --
#
# The 5 named microstructure feature groups from the project's scope
# (OFI, queue imbalance, microprice, spread, liquidity
# pressure), each expanded across features.py's 4 event-count windows.
# This is deliberately the model's *entire* input space: features.py also
# emits arrival_rate/cancel_rate/trade_imbalance and raw price levels
# (best_bid/best_ask/midprice), but the ablation spec names exactly these
# 5 groups, and raw price *levels* are excluded on purpose -- they're
# non-stationary across the day and would let the model key off "what
# price regime is this" instead of microstructure imbalance signals.
_WINDOWS = (5, 10, 20, 50)

FEATURE_GROUPS: Dict[str, List[str]] = {
    "ofi": [f"ofi_w{w}" for w in _WINDOWS],
    "queue_imbalance": [f"queue_imbalance_w{w}" for w in _WINDOWS],
    "microprice": [f"microprice_w{w}" for w in _WINDOWS] + [f"weighted_microprice_w{w}" for w in _WINDOWS],
    "spread": ["spread", "relative_spread"],
    "liquidity_pressure": [f"liquidity_pressure_w{w}" for w in _WINDOWS],
}


def feature_columns(exclude_group: Optional[str] = None) -> List[str]:
    """All model input columns across FEATURE_GROUPS, in a fixed order,
    optionally leaving one named group out entirely -- used both for the
    full model and for the Phase 4 ablation (retrain with one group
    removed, compare IC)."""
    return [col for name, cols in FEATURE_GROUPS.items() if name != exclude_group for col in cols]


def normalize(train_x: pd.DataFrame, test_x: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Z-scores both train_x and test_x using train_x's own mean/std only.

    Why train-set-only statistics: normalizing with statistics computed
    over train+test (or test alone) would leak the test fold's
    distribution into training.
    Zero-variance train columns divide by 1 instead of 0 (left unscaled)
    rather than producing inf/NaN; none of the current feature groups are
    constant in practice, this just guards the edge case.
    """
    mean = train_x.mean()
    std = train_x.std().replace(0.0, 1.0)
    return (train_x - mean) / std, (test_x - mean) / std


def make_linear_model(config: dict) -> LogisticRegression:
    """Multinomial logistic regression: the Phase 4 "linear baseline".
    class_weight='balanced' (via sample_weight at fit time, see
    make_predict_fold_fn) because label_h{h} is heavily skewed toward the
    flat (0) class (79-95%, see reference/03_labels_and_baselines.md) --
    unweighted, the model would trivially learn to always predict "flat"
    and produce ~0 IC."""
    p = config["model"]["logistic_regression"]
    return LogisticRegression(max_iter=p["max_iter"], C=p["C"])


def make_xgboost_model(config: dict) -> XGBClassifier:
    """Gradient-boosted tree classifier: the Phase 4 "XGBoost model".
    Modest depth/estimator count (config.yaml: model.xgboost) since it's
    refit fresh on every walk-forward fold, for every horizon, for the
    full model and every ablation -- not one model tuned and fit once."""
    p = config["model"]["xgboost"]
    return XGBClassifier(
        n_estimators=p["n_estimators"],
        max_depth=p["max_depth"],
        learning_rate=p["learning_rate"],
        subsample=p["subsample"],
        colsample_bytree=p["colsample_bytree"],
        objective="multi:softprob",
        num_class=3,
        tree_method="hist",
        eval_metric="mlogloss",
    )


_LABEL_TO_CLASS_IDX = {-1.0: 0, 0.0: 1, 1.0: 2}
_CLASS_IDX_TO_LABEL = {idx: label for label, idx in _LABEL_TO_CLASS_IDX.items()}
_CLASSES = (-1.0, 0.0, 1.0)


def make_predict_fold_fn(
    config: dict,
    model_fn: Callable[[dict], object],
    feature_cols: List[str],
    uses_class_index: bool,
) -> Callable[[pd.DataFrame, range, range, str], Tuple[pd.Series, pd.Series, pd.DataFrame]]:
    """Builds a `predict_fold_fn(df, train_range, test_range, label_col)`
    closure for evaluate.py's evaluate_model(): fits a *fresh* model_fn(config)
    on the fold's (already-purged) train block and scores the fold's test
    block, so evaluate_model() can call it once per (horizon, fold)
    without knowing anything about sklearn/xgboost/feature groups.

    uses_class_index: True for XGBoost, which requires 0/1/2 integer class
    labels rather than the {-1, 0, 1} labels.py uses natively (sklearn's
    LogisticRegression accepts the raw {-1, 0, 1} labels directly).

    Returns, per call: (score, pred_class, proba) all indexed like the
    fold's test block --
      - score: continuous P(label=+1) - P(label=-1), rankable for IC like
        baselines.py's outputs.
      - pred_class: the model's highest-probability class in {-1, 0, 1}.
      - proba: DataFrame with one column per class in {-1, 0, 1}.
    Rows whose features are NaN (or whose train block has fewer than 2
    label classes to fit against) score NaN/None rather than being
    imputed -- consistent with labels.py's own end-of-day NaN handling,
    and left for evaluate.py's existing NaN-aware metrics to drop.
    """

    def predict_fold(df: pd.DataFrame, train_range: range, test_range: range, label_col: str):
        train_df = df.iloc[train_range.start:train_range.stop]
        test_df = df.iloc[test_range.start:test_range.stop]

        train_valid = train_df[feature_cols].notna().all(axis=1) & train_df[label_col].notna()
        x_train_raw = train_df.loc[train_valid, feature_cols]
        y_train = train_df.loc[train_valid, label_col]

        test_valid = test_df[feature_cols].notna().all(axis=1)
        x_test_raw = test_df.loc[test_valid, feature_cols]

        score = pd.Series(float("nan"), index=test_df.index)
        pred_class = pd.Series(float("nan"), index=test_df.index)
        proba = pd.DataFrame(float("nan"), index=test_df.index, columns=list(_CLASSES))

        if len(x_train_raw) == 0 or len(x_test_raw) == 0 or y_train.nunique() < 2:
            return score, pred_class, proba

        x_train, x_test = normalize(x_train_raw, x_test_raw)
        weights = compute_sample_weight("balanced", y_train)

        model = model_fn(config)
        if uses_class_index:
            y_fit = y_train.map(_LABEL_TO_CLASS_IDX)
            model.fit(x_train, y_fit, sample_weight=weights)
            raw_proba = model.predict_proba(x_test)
            columns = [_CLASS_IDX_TO_LABEL[c] for c in model.classes_]
        else:
            model.fit(x_train, y_train, sample_weight=weights)
            raw_proba = model.predict_proba(x_test)
            columns = list(model.classes_)

        fold_proba = pd.DataFrame(raw_proba, columns=columns, index=x_test.index)
        for c in _CLASSES:
            proba.loc[x_test.index, c] = fold_proba[c] if c in fold_proba.columns else 0.0

        score.loc[x_test.index] = proba.loc[x_test.index, 1.0] - proba.loc[x_test.index, -1.0]
        pred_class.loc[x_test.index] = proba.loc[x_test.index, list(_CLASSES)].idxmax(axis=1)
        return score, pred_class, proba

    return predict_fold


def main():
    # Deferred import: evaluate.py imports model.py's predict-fold
    # closures, so importing evaluate.py back at module load time here
    # would be circular. Only main() (a script entry point, not something
    # evaluate.py calls into) needs it.
    from evaluate import evaluate_baselines, evaluate_model
    from labels import add_labels

    config = load_config()
    df = pd.read_parquet(config["features"]["output_file"])
    df = add_labels(df, config["labels"]["horizons"], config["labels"]["epsilon"])
    horizons = config["labels"]["horizons"]
    n_splits = config["evaluate"]["n_splits"]

    pd.set_option("display.float_format", lambda x: f"{x:.4f}")

    full_cols = feature_columns()
    baseline_results = evaluate_baselines(df, horizons, n_splits)
    linear_fn = make_predict_fold_fn(config, make_linear_model, full_cols, uses_class_index=False)
    xgb_fn = make_predict_fold_fn(config, make_xgboost_model, full_cols, uses_class_index=True)

    linear_results = evaluate_model(df, horizons, n_splits, linear_fn, "logistic_regression")
    xgb_results = evaluate_model(df, horizons, n_splits, xgb_fn, "xgboost")

    best_baseline = baseline_results.loc[baseline_results.groupby("horizon")["ic"].idxmax()]
    best_baseline = best_baseline[["horizon", "baseline", "ic"]].rename(
        columns={"baseline": "best_baseline", "ic": "best_baseline_ic"}
    )

    print("=== model vs. best baseline (IC) ===")
    comparison = pd.concat([linear_results, xgb_results], ignore_index=True)
    comparison = comparison.merge(best_baseline, on="horizon")
    comparison["ic_delta_vs_best_baseline"] = comparison["ic"] - comparison["best_baseline_ic"]
    print(comparison[[
        "model", "horizon", "ic", "accuracy", "precision", "recall", "auc",
        "best_baseline", "best_baseline_ic", "ic_delta_vs_best_baseline",
    ]].to_string(index=False))

    print()
    print("=== ablation: XGBoost IC with one feature group removed ===")
    full_ic = xgb_results.set_index("horizon")["ic"]
    ablation_rows = []
    for group in FEATURE_GROUPS:
        cols = feature_columns(exclude_group=group)
        fold_fn = make_predict_fold_fn(config, make_xgboost_model, cols, uses_class_index=True)
        result = evaluate_model(df, horizons, n_splits, fold_fn, f"xgboost_no_{group}")
        for _, row in result.iterrows():
            ablation_rows.append({
                "removed_group": group,
                "horizon": row["horizon"],
                "ic": row["ic"],
                "ic_delta_vs_full_model": row["ic"] - full_ic.loc[row["horizon"]],
            })
    ablation = pd.DataFrame(ablation_rows)
    print(ablation.to_string(index=False))


if __name__ == "__main__":
    main()
