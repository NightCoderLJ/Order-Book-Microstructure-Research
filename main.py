"""
main.py -- Phase 6 single-command driver for the full pipeline.

Takes in: nothing beyond config.yaml (each stage's own main() reads it
directly, exactly as it does when run standalone).

Produces: the full sequence of stage outputs, in the pipeline's file
order -- ingest+book replay throughput/reconciliation,
data/features/features.parquet, label class balance, baseline IC table,
model-vs-baseline IC and feature-group ablation, signal/paper-trading
backtest summaries, and viz.py's diagnostic plots under
config.yaml: viz.output_dir -- so a reviewer (or a fresh checkout with
data/raw/ populated per README.md) can reproduce every number and plot
in one command: `python main.py`.

Why call each stage's existing main() rather than re-implementing the
pipeline here: every stage module is already independently runnable and
unit-tested (checks/test_*.py). main.py's only job is running them in the
documented dependency order (each stage reads the previous stage's
output file, e.g. features.py -> data/features/features.parquet ->
labels.py/evaluate.py/model.py/backtest.py/viz.py) and reporting
per-stage wall-clock time -- re-deriving their logic here would just be a
second copy of the same code that could silently drift out of sync.
"""

import time

import backtest
import evaluate
import features
import labels
import model
import replay
import viz

STAGES = [
    ("1. ingest + book replay (throughput, reconciliation)", replay.main),
    ("2. feature extraction -> data/features/features.parquet", features.main),
    ("3a. label class balance", labels.main),
    ("3b. baseline IC / accuracy", evaluate.main),
    ("4. model vs. best baseline + feature-group ablation", model.main),
    ("5. backtest: signal replay vs. paper-trading", backtest.main),
    ("6. diagnostic plots", viz.main),
]


def main():
    total_start = time.perf_counter()
    for name, stage_fn in STAGES:
        header = f"=== {name} ==="
        print(f"\n{header}")
        stage_start = time.perf_counter()
        stage_fn()
        print(f"--- done in {time.perf_counter() - stage_start:.1f}s ---")

    print(f"\nfull pipeline done in {time.perf_counter() - total_start:.1f}s")


if __name__ == "__main__":
    main()
