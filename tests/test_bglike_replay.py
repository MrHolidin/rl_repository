import json
from pathlib import Path

from src.agents.random_agent import RandomAgent
from src.envs.bglike.action_map import A_FINISH
from src.envs.bglike.lobby_env import BGLobbyEnv
from src.envs.bglike.replay import attach_replay, close_replay
from src.envs.bglike.replay_render import decode_env_action_compact, render_jsonl_records
from src.envs.bglike.seat_config import lobby_from_learned_seats


def _load_frames(path: Path) -> list[dict]:
    frames = []
    for line in path.read_text(encoding="utf-8").strip().split("\n")[1:]:
        rec = json.loads(line)
        if rec.get("type") == "frame":
            frames.append(rec)
    return frames


def test_bglike_replay_jsonl_written(tmp_path: Path) -> None:
    import src.envs.bglike.lobby_env as le

    rep = tmp_path / "lobby.jsonl"
    agents = {s: RandomAgent(seed=10 + s) for s in range(8)}
    configs = lobby_from_learned_seats(tuple(range(8)), agent_by_seat=agents)
    env = BGLobbyEnv(
        configs,
        learned_seats=tuple(range(8)),
        training_seats=(0,),
        seed=0,
    )
    attach_replay(env, rep, {"test": True})
    old_cap = le.MAX_DRAIN_STEPS
    le.MAX_DRAIN_STEPS = 20_000
    try:
        env.reset(seed=0)
        env.drain_until_lobby_done(deterministic=False)
    finally:
        le.MAX_DRAIN_STEPS = old_cap
        close_replay(env)

    lines = rep.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) >= 2
    hdr = json.loads(lines[0])
    assert hdr["type"] == "header"
    assert hdr["game"] == "bglike"
    frames = _load_frames(rep)
    assert frames
    fr = frames[0]
    assert "seat" in fr
    assert "auto" in fr
    assert fr["state"]["players"]["0"]["hp"] > 0


def test_bglike_replay_record_seats_skips_other_finish_keeps_combat(tmp_path: Path) -> None:
    """Shop frames only for record_seats; combat/elimination frames always emitted."""
    rep = tmp_path / "record_seats.jsonl"
    agents = {s: RandomAgent(seed=20 + s) for s in range(8)}
    configs = lobby_from_learned_seats(tuple(range(8)), agent_by_seat=agents)
    env = BGLobbyEnv(
        configs,
        learned_seats=tuple(range(8)),
        training_seats=(0,),
        seed=1,
    )
    attach_replay(env, rep, {}, record_seats=frozenset({0}))
    try:
        env.reset(seed=1)
        env.drain_until_lobby_done(deterministic=False)
    finally:
        close_replay(env)

    frames = _load_frames(rep)
    assert frames
    shop_finish_frames = [
        fr
        for fr in frames
        if decode_env_action_compact(int(fr.get("a", -1))) == "FINISH"
        and not fr.get("info", {}).get("combat_advanced")
        and not fr.get("info", {}).get("eliminated_seats")
        and not fr.get("info", {}).get("lobby_done")
    ]
    assert {int(fr["seat"]) for fr in shop_finish_frames} <= {0}
    assert any(fr.get("info", {}).get("combat_advanced") for fr in frames)


def test_bglike_replay_empty_record_seats_skips_shop_frames(tmp_path: Path) -> None:
    rep = tmp_path / "no_shop.jsonl"
    agents = {s: RandomAgent(seed=25 + s) for s in range(8)}
    configs = lobby_from_learned_seats(tuple(range(8)), agent_by_seat=agents)
    env = BGLobbyEnv(
        configs,
        learned_seats=tuple(range(8)),
        training_seats=(0,),
        seed=1,
    )
    attach_replay(env, rep, {}, record_seats=frozenset())
    try:
        env.reset(seed=1)
        env.drain_until_lobby_done(deterministic=False)
    finally:
        close_replay(env)

    frames = _load_frames(rep)
    assert all(
        decode_env_action_compact(int(fr.get("a", -1))) != "FINISH"
        or fr.get("info", {}).get("combat_advanced")
        or fr.get("info", {}).get("eliminated_seats")
        or fr.get("info", {}).get("lobby_done")
        for fr in frames
    )


def test_bglike_render_normal_shows_board_on_finish() -> None:
    pl = (
        '"hp":30,"gold":0,"tier":2,"phase":"DONE","shop_done":true,"shop_acts":3,'
        '"board":[{"card_id":"a","name":"A","atk":2,"hp":3,"tier":1,"kw":[],"shield":false,'
        '"token":false,"abilities":[]}],"shop":[null,null,null,null,null,null],'
        '"hand":[null,null,null,null,null]'
    )
    frame = (
        '{"type":"frame","seat":2,"a":'
        + str(A_FINISH)
        + ',"auto":true,"illegal":false,'
        + '"state":{"round":3,"combat_round":1,"cur":2,"alive":[0,1,2,3,4,5,6,7],'
        + '"players":{"2":{' + pl + '}}},"info":{"eliminated_seats":[],"lobby_done":false,"combat_advanced":false}}'
    )
    text = render_jsonl_records(['{"type":"header","game":"bglike"}', frame])
    assert "[auto]" in text
    assert "S2 FINISH" in text
    assert "стол S2:" in text
    assert "[A 2/3]" in text


def test_bglike_decode_compact() -> None:
    from src.envs.bglike.action_map import A_FINISH, A_ROLL

    assert decode_env_action_compact(A_ROLL) == "ROLL"
    assert decode_env_action_compact(A_FINISH) == "FINISH"


def test_bglike_replay_structured_checkpoint(tmp_path: Path) -> None:
    """Structured PPO controllers must write replay frames (not header-only)."""
    from pathlib import Path as P

    import src.envs.bglike.lobby_env as le
    from src.evaluation.eval_checkpoints import load_training_agent_checkpoint

    ck = P("runs/bglike/dist_ppo_024/checkpoints/dist_bglike_ppo_955265.pt")
    if not ck.is_file():
        import pytest

        pytest.skip("checkpoint not available")

    rep = tmp_path / "structured_lobby.jsonl"
    try:
        agent = load_training_agent_checkpoint(ck, device="cpu", seed=0)
    except RuntimeError as exc:
        if "size mismatch" in str(exc):
            import pytest

            pytest.skip("checkpoint incompatible with current OBS_DIM")
        raise
    agent.eval()
    agents = {s: agent for s in range(8)}
    configs = lobby_from_learned_seats(tuple(range(8)), agent_by_seat=agents)
    env = BGLobbyEnv(
        configs,
        learned_seats=tuple(range(8)),
        training_seats=tuple(range(8)),
        seed=0,
    )
    attach_replay(env, rep, {"structured": True})
    old_cap = le.MAX_DRAIN_STEPS
    le.MAX_DRAIN_STEPS = 20_000
    try:
        env.reset(seed=0)
        env.drain_until_lobby_done(deterministic=True)
    finally:
        le.MAX_DRAIN_STEPS = old_cap
        close_replay(env)

    frames = _load_frames(rep)
    assert len(frames) > 10
    assert any(rec.get("state", {}).get("done") for rec in frames)
    rolls = sum(1 for rec in frames if decode_env_action_compact(int(rec.get("a", -1))) == "ROLL")
    assert rolls == 0


def test_bglike_replay_sparse_smaller_than_full(tmp_path: Path) -> None:
    """Sparse default skips micro shop actions (ROLL/BUY/...) between FINISH frames."""
    import src.envs.bglike.lobby_env as le

    agents = {s: RandomAgent(seed=30 + s) for s in range(8)}
    configs = lobby_from_learned_seats(tuple(range(8)), agent_by_seat=agents)

    def _frame_count(sparse: bool) -> int:
        rep = tmp_path / f"{'sparse' if sparse else 'full'}.jsonl"
        env = BGLobbyEnv(
            configs,
            learned_seats=tuple(range(8)),
            training_seats=(0,),
            seed=2,
        )
        attach_replay(env, rep, {}, sparse=sparse)
        old = le.MAX_DRAIN_STEPS
        le.MAX_DRAIN_STEPS = 20_000
        try:
            env.reset(seed=2)
            env.drain_until_lobby_done(deterministic=False)
        finally:
            le.MAX_DRAIN_STEPS = old
            close_replay(env)
        return len(_load_frames(rep))

    sparse_n = _frame_count(True)
    full_n = _frame_count(False)
    assert sparse_n < full_n // 2
    assert sparse_n >= 8


def test_bglike_replay_constructor_config(tmp_path: Path) -> None:
    from src.envs.bglike.replay_recorder import LobbyReplayConfig

    rep = tmp_path / "ctor.jsonl"
    agents = {s: RandomAgent(seed=40 + s) for s in range(8)}
    configs = lobby_from_learned_seats(tuple(range(8)), agent_by_seat=agents)
    config = LobbyReplayConfig(path=rep, header={"ctor": True}, record_seats=frozenset({0}))
    env = BGLobbyEnv(
        configs,
        learned_seats=tuple(range(8)),
        training_seats=(0,),
        seed=3,
        replay=config,
    )
    try:
        env.reset(seed=3)
        env.drain_until_lobby_done(deterministic=False)
    finally:
        close_replay(env)

    hdr = json.loads(rep.read_text(encoding="utf-8").strip().split("\n")[0])
    assert hdr["record_seats"] == [0]
    assert _load_frames(rep)
