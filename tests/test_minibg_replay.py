import json
from pathlib import Path
from unittest.mock import patch

import src.envs  # noqa: F401
from src.agents.random_agent import RandomAgent
from src.registry import make_game
from src.utils.match import play_single_game
from src.training.trainer import StartPolicy


def test_eval_checkpoints_vs_opponents_replay_dir(tmp_path: Path):
    from src.evaluation.eval_checkpoints import eval_checkpoints_vs_opponents

    ck = tmp_path / "dqn_1000.pt"
    ck.write_bytes(b"")
    rdir = tmp_path / "replays"
    with patch("src.evaluation.eval_checkpoints.load_training_agent_checkpoint") as ld:

        def _load(path, *, device=None, seed=42):
            return RandomAgent(seed=seed)

        ld.side_effect = _load
        df = eval_checkpoints_vs_opponents(
            [ck],
            opponent_names=["t1_random"],
            num_games=2,
            game_id="minibg",
            replay_dir=rdir,
            batch_size=0,
            seed=7,
            start_policy="agent_first",
        )
    assert len(df) == 1
    files = sorted(rdir.glob("*.jsonl"))
    assert len(files) == 2
    for f in files:
        assert f.stat().st_size > 50


def test_replay_jsonl_written(tmp_path: Path):
    rep = tmp_path / "t.jsonl"
    env = make_game(
        "minibg",
        seed=0,
        replay_path=rep,
        replay_meta={"test": True},
    )
    a, b = RandomAgent(seed=1), RandomAgent(seed=2)
    play_single_game(
        env,
        a,
        b,
        start_policy=StartPolicy.AGENT_FIRST,
        seed=3,
        deterministic_agent=False,
        deterministic_opponent=False,
    )
    env.close_replay()
    text = rep.read_text(encoding="utf-8")
    lines = text.strip().split("\n")
    assert len(lines) >= 2
    hdr = json.loads(lines[0])
    assert hdr["type"] == "header"
    assert hdr["test"] is True
    frames = []
    for L in lines[1:]:
        o = json.loads(L)
        if o["type"] == "frame":
            frames.append(o)
    assert frames, "expected at least one frame line"
    fr = frames[0]
    assert "state" in fr
    assert fr["state"]["p0"]["hp"] > 0


def test_replay_episode_break_on_second_reset(tmp_path: Path):
    rep = tmp_path / "m.jsonl"
    env = make_game("minibg", seed=1, replay_path=rep)
    env.reset(seed=0)
    env.reset(seed=1)
    env.close_replay()
    lines = rep.read_text(encoding="utf-8").strip().split("\n")
    kinds = [json.loads(L)["type"] for L in lines]
    assert "episode_break" in kinds
