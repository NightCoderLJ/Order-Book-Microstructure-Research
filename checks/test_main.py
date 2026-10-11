"""Unit test for main.py: checks the pipeline driver calls every stage's
main() exactly once, in the documented dependency order, without actually
running the (multi-minute) real pipeline."""

from unittest.mock import patch

import main


def test_main_runs_every_stage_once_in_order():
    call_order = []
    fakes = {name: (lambda n=name: call_order.append(n)) for name, _ in main.STAGES}

    with patch.object(main, "STAGES", [(name, fakes[name]) for name, _ in main.STAGES]):
        main.main()

    assert call_order == [name for name, _ in main.STAGES]


def test_stages_cover_every_pipeline_module_main():
    import backtest
    import evaluate
    import features
    import labels
    import model
    import replay
    import viz

    stage_fns = [fn for _, fn in main.STAGES]
    for module in (replay, features, labels, evaluate, model, backtest, viz):
        assert module.main in stage_fns
