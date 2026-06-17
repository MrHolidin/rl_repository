import csv
from pathlib import Path
from types import SimpleNamespace

from src.training.callbacks.metrics_file import MetricsFileCallback
from src.training.metric_groups import (
    DVD_IDENTITIES_GROUP,
    MISC_GROUP,
    header_for,
    parse_dvd_identity,
    route,
)


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------
def test_route_known_flat_keys():
    assert route("policy_loss") == "core"
    assert route("explained_variance") == "critic"
    assert route("rollout_size") == "rollout"
    assert route("avg_q") == "dqn"
    assert route("dvd_pop_diversity") == "dvd"


def test_route_slash_prefix_makes_dynamic_group():
    assert route("rnd/novelty_mean") == "rnd"
    assert route("loss/predictor") == "loss"


def test_route_dvd_identity_arrays_to_long_group():
    assert route("dvd_place_0") == DVD_IDENTITIES_GROUP
    assert route("dvd_tribe_7") == DVD_IDENTITIES_GROUP
    assert route("dvd_assigned_frac_3") == DVD_IDENTITIES_GROUP


def test_route_unknown_to_misc():
    assert route("something_new") == MISC_GROUP


def test_parse_dvd_identity():
    assert parse_dvd_identity("dvd_place_5") == ("place", 5)
    assert parse_dvd_identity("dvd_assigned_frac_12") == ("assigned_frac", 12)
    assert parse_dvd_identity("dvd_pop_diversity") is None


def test_header_for_groups():
    assert header_for("core")[:2] == ("step", "episode")
    assert "policy_loss" in header_for("core")
    assert header_for(DVD_IDENTITIES_GROUP) == (
        "step",
        "episode",
        "identity",
        "place",
        "tribe",
        "assigned_frac",
    )
    assert header_for(MISC_GROUP) == ("step", "episode", "metric", "value")


# ---------------------------------------------------------------------------
# Callback: grouped file emission
# ---------------------------------------------------------------------------
def _trainer():
    agent = SimpleNamespace(epsilon=0.5, learning_rate=1e-4)
    return SimpleNamespace(agent=agent, episode_index=7)


def _read(path: Path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def test_only_active_groups_get_files(tmp_path):
    cb = MetricsFileCallback(tmp_path, interval=1)
    cb.on_train_begin(_trainer())
    cb.on_step_end(_trainer(), 1, None, {"policy_loss": 0.1, "value_loss": 0.2})
    cb.on_train_end(_trainer())

    files = {p.name for p in tmp_path.glob("metrics_*.csv")}
    assert files == {"metrics_core.csv"}  # no dvd/critic/... files for a plain step


def test_core_file_has_grouped_columns_and_values(tmp_path):
    cb = MetricsFileCallback(tmp_path, interval=1)
    cb.on_train_begin(_trainer())
    cb.on_step_end(_trainer(), 1, None, {"policy_loss": -0.0067900134, "grad_norm": 1.5})
    cb.on_train_end(_trainer())

    rows = _read(tmp_path / "metrics_core.csv")
    assert len(rows) == 1
    assert rows[0]["step"] == "1"
    assert rows[0]["episode"] == "7"
    assert rows[0]["learning_rate"] == "0.0001"
    # 6 significant figures, not a noisy float repr.
    assert rows[0]["policy_loss"] == "-0.00679001"
    assert "policy_loss" in rows[0] and "dvd_pop_diversity" not in rows[0]


def test_dvd_identities_long_format(tmp_path):
    cb = MetricsFileCallback(tmp_path, interval=1)
    cb.on_train_begin(_trainer())
    cb.on_step_end(
        _trainer(),
        1,
        None,
        {
            "dvd_pop_diversity": 0.3,
            "dvd_place_0": 2.0,
            "dvd_tribe_0": 1.0,
            "dvd_assigned_frac_0": 0.5,
            "dvd_place_1": 3.0,
            "dvd_tribe_1": 4.0,
            "dvd_assigned_frac_1": 0.4,
        },
    )
    cb.on_train_end(_trainer())

    agg = _read(tmp_path / "metrics_dvd.csv")
    assert agg[0]["dvd_pop_diversity"] == "0.3"

    ident = _read(tmp_path / f"metrics_{DVD_IDENTITIES_GROUP}.csv")
    assert len(ident) == 2  # one row per identity, not 6 columns
    assert ident[0]["identity"] == "0"
    assert ident[0]["place"] == "2"
    assert ident[1]["identity"] == "1"
    assert ident[1]["tribe"] == "4"


def test_slash_namespaced_metric_lands_in_dynamic_long_file(tmp_path):
    cb = MetricsFileCallback(tmp_path, interval=1)
    cb.on_train_begin(_trainer())
    cb.on_step_end(_trainer(), 1, None, {"rnd/novelty_mean": 0.88})
    cb.on_train_end(_trainer())

    rows = _read(tmp_path / "metrics_rnd.csv")
    assert rows[0]["metric"] == "novelty_mean"  # slash prefix stripped
    assert rows[0]["value"] == "0.88"


def test_unknown_metric_captured_in_misc(tmp_path):
    cb = MetricsFileCallback(tmp_path, interval=1)
    cb.on_train_begin(_trainer())
    cb.on_step_end(_trainer(), 1, None, {"brand_new_metric": 1.23})
    cb.on_train_end(_trainer())

    rows = _read(tmp_path / "metrics_misc.csv")
    assert rows[0]["metric"] == "brand_new_metric"
    assert rows[0]["value"] == "1.23"


def test_stale_files_removed_on_train_begin(tmp_path):
    stale = tmp_path / "metrics_dvd.csv"
    stale.write_text("old,data\n1,2\n")
    cb = MetricsFileCallback(tmp_path, interval=1)
    cb.on_train_begin(_trainer())
    assert not stale.exists()
