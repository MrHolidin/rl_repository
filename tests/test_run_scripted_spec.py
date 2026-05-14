"""Resolution of opponent_sampler.params into ScriptedOpponentsSpec."""

from src.training.run import _build_scripted_spec


def test_minibg_legacy_bots_and_random_fraction():
    s = _build_scripted_spec(
        "minibg",
        {"bots": ["t1_random"], "random_fraction": 0.3},
    )
    assert s.mode == "minibg"
    assert abs(s.distribution["random"] - 0.3) < 1e-6
    assert abs(s.distribution["t1_random"] - 0.7) < 1e-6


def test_classic_heuristic_distribution_legacy():
    s = _build_scripted_spec(
        "connect4",
        {"heuristic_distribution": {"random": 2.0, "heuristic": 8.0}},
    )
    assert s.mode == "classic"
    assert abs(s.distribution["random"] - 0.2) < 1e-6
    assert abs(s.distribution["heuristic"] - 0.8) < 1e-6


def test_scripted_distribution_minibg():
    s = _build_scripted_spec(
        "minibg",
        {
            "scripted": {
                "distribution": {"random": 1.0, "t1_random": 2.0},
            }
        },
    )
    assert s.mode == "minibg"
    assert abs(s.distribution["random"] - 1.0 / 3.0) < 1e-5
    assert abs(s.distribution["t1_random"] - 2.0 / 3.0) < 1e-5
