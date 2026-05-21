import json
from pathlib import Path

from src.agents.random_agent import RandomAgent
from src.envs.bglike.lobby_env import BGLobbyEnv
from src.envs.bglike.replay import attach_replay, close_replay
from src.envs.bglike.replay_render import decode_env_action_compact, render_jsonl_records
from src.envs.bglike.seat_config import lobby_from_learned_seats


def test_bglike_replay_jsonl_written(tmp_path: Path) -> None:
    rep = tmp_path / "lobby.jsonl"
    agents = {s: RandomAgent(seed=10 + s) for s in range(8)}
    configs = lobby_from_learned_seats(tuple(range(8)), agent_by_seat=agents)
    env = BGLobbyEnv(
        configs,
        learned_seats=tuple(range(8)),
        training_seats=(0,),
        seed=0,
    )
    attach_replay(env, rep, {"test": True}, record_auto=True)
    try:
        env.reset(seed=0)
        env.drain_until_lobby_done(deterministic=False)
    finally:
        close_replay(env)

    lines = rep.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) >= 2
    hdr = json.loads(lines[0])
    assert hdr["type"] == "header"
    assert hdr["game"] == "bglike"
    frames = [json.loads(line) for line in lines[1:] if json.loads(line)["type"] == "frame"]
    assert frames
    fr = frames[0]
    assert "seat" in fr
    assert "auto" in fr
    assert fr["state"]["players"]["0"]["hp"] > 0


def test_bglike_replay_no_auto_skips_auto_frames(tmp_path: Path) -> None:
    rep = tmp_path / "no_auto.jsonl"
    agents = {s: RandomAgent(seed=20 + s) for s in range(8)}
    configs = lobby_from_learned_seats(tuple(range(8)), agent_by_seat=agents)
    env = BGLobbyEnv(
        configs,
        learned_seats=tuple(range(8)),
        training_seats=(0,),
        seed=1,
    )
    attach_replay(env, rep, {}, record_auto=False)
    try:
        env.reset(seed=1)
        env.drain_until_lobby_done(deterministic=False)
    finally:
        close_replay(env)

    frames = []
    for line in rep.read_text(encoding="utf-8").strip().split("\n")[1:]:
        rec = json.loads(line)
        if rec.get("type") == "frame":
            frames.append(rec)
    assert frames == []


def test_bglike_render_normal_shows_board_on_finish() -> None:
    from src.envs.bglike.action_map import A_FINISH

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
