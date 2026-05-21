"""Human-readable text from BGLike JSONL replays (no combat blow-by-blow)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, TextIO, Union

from src.envs.minibg.replay_render import (
    format_board_line,
    format_hand_line,
    format_shop_line,
)

from .action_map import (
    A_FINISH,
    A_LEVEL_UP,
    A_ROLL,
    NUM_ENV_ACTIONS,
    buy_slot,
    is_apply_effect_skip,
    is_buy,
    is_discover_pick,
    is_finish,
    is_finish_freeze_shop,
    is_magnet,
    is_place,
    is_sell,
    is_swap_board,
    is_target_board,
    magnet_hand_board,
    discover_pick_slot,
    place_slot,
    sell_pos,
    swap_adj_index_from_env_action,
    target_board_slot,
)


def decode_env_action(a: int) -> str:
    if a == A_ROLL:
        return "ROLL"
    if a == A_LEVEL_UP:
        return "LEVEL_UP"
    if is_buy(a):
        return f"BUY_SHOP_{buy_slot(a)}"
    if is_sell(a):
        return f"SELL_BOARD_{sell_pos(a)}"
    if is_place(a):
        return f"PLACE_HAND_{place_slot(a)}"
    if is_magnet(a):
        h, b = magnet_hand_board(a)
        return f"MAGNET_HAND_{h}_BOARD_{b}"
    if is_discover_pick(a):
        return f"DISCOVER_PICK_{discover_pick_slot(a)}"
    if is_target_board(a):
        return f"APPLY_EFFECT_{target_board_slot(a)}"
    if is_apply_effect_skip(a):
        return "APPLY_EFFECT_SKIP"
    if is_swap_board(a):
        i = swap_adj_index_from_env_action(a)
        return f"SWAP_BOARD_{i}_{i + 1}"
    if is_finish(a):
        return "FINISH"
    if is_finish_freeze_shop(a):
        return "FINISH_FREEZE_SHOP"
    if 0 <= a < NUM_ENV_ACTIONS:
        return f"UNKNOWN_{a}"
    return f"OUT_OF_RANGE_{a}"


def decode_env_action_compact(a: int) -> str:
    if a == A_ROLL:
        return "ROLL"
    if a == A_LEVEL_UP:
        return "LEVEL_UP"
    if is_buy(a):
        return "BUY"
    if is_sell(a):
        return "SELL"
    if is_place(a):
        return "PLACE"
    if is_magnet(a):
        return "MAGNET"
    if is_discover_pick(a):
        return "DISCOVER"
    if is_target_board(a):
        return "APPLY_EFFECT"
    if is_apply_effect_skip(a):
        return "APPLY_SKIP"
    if is_swap_board(a):
        return "SWAP"
    if is_finish(a):
        return "FINISH"
    if is_finish_freeze_shop(a):
        return "FINISH_FREEZE_SHOP"
    if 0 <= a < NUM_ENV_ACTIONS:
        return f"UNKNOWN_{a}"
    return f"OUT_OF_RANGE_{a}"


def _player(state: Dict[str, Any], seat: int) -> Dict[str, Any]:
    players = state.get("players") or {}
    return players.get(str(seat)) or players.get(seat) or {}


def _alive_seats(state: Dict[str, Any]) -> List[int]:
    alive = state.get("alive")
    if alive is not None:
        return [int(x) for x in alive]
    players = state.get("players") or {}
    return sorted(int(k) for k in players.keys())


def _seat_summary(state: Dict[str, Any], seat: int) -> str:
    p = _player(state, seat)
    return f"S{seat}(hp={p.get('hp', '?')}, gold={p.get('gold', '?')}, tier={p.get('tier', '?')})"


def _turn_ended(rec: Dict[str, Any], seat: int, player: Dict[str, Any]) -> bool:
    a = int(rec.get("a", -1))
    if is_finish(a) or is_finish_freeze_shop(a):
        return True
    info = rec.get("info") or {}
    if seat in info.get("eliminated_seats", []):
        return True
    return player.get("phase") == "DONE"


def _render_seat_details(out: List[str], state: Dict[str, Any], seat: int) -> None:
    p = _player(state, seat)
    board = list(p.get("board") or [])
    hand_raw = p.get("hand")
    shop_raw = p.get("shop")
    hand: Optional[List[Optional[Dict[str, Any]]]] = None if hand_raw is None else list(hand_raw)
    shop: Optional[List[Optional[Dict[str, Any]]]] = None if shop_raw is None else list(shop_raw)
    out.append(format_board_line(f"стол S{seat}", board))
    out.append(format_hand_line(f"рука S{seat}", hand))
    out.append(format_shop_line(f"лавка S{seat}", shop))


def render_jsonl_records(lines: Iterator[str], *, extended: bool = False) -> str:
    out: List[str] = []
    prev_combat_round: Optional[int] = None
    prev_hp: Optional[Dict[int, int]] = None
    prev_boards: Optional[Dict[int, List[Any]]] = None

    for raw in lines:
        raw = raw.strip()
        if not raw:
            continue
        rec = json.loads(raw)
        t = rec.get("type")
        if t == "header":
            meta = {k: v for k, v in rec.items() if k != "type"}
            out.append("[header] " + " ".join(f"{k}={v!r}" for k, v in sorted(meta.items())))
            out.append("")
            continue
        if t == "episode_break":
            out.append("")
            out.append(f"--- новый эпизод (break ep={rec.get('episode')}) ---")
            out.append("")
            prev_combat_round = None
            prev_hp = None
            prev_boards = None
            continue
        if t != "frame":
            continue

        st = rec.get("state") or {}
        rnd = int(st.get("round") or 0)
        combat = int(st.get("combat_round") or 0)
        seat = int(rec.get("seat", rec.get("p", -1)))
        a = int(rec.get("a", -1))
        illegal = bool(rec.get("illegal"))
        auto = bool(rec.get("auto"))
        player = _player(st, seat)

        tag = "ILLEGAL " if illegal else ""
        auto_tag = "[auto] " if auto else ""
        act = decode_env_action_compact(a)
        alive = _alive_seats(st)
        summary = ", ".join(_seat_summary(st, s) for s in alive)
        out.append(f"{tag}{auto_tag}R{rnd} C{combat} S{seat} {act}  ({summary})")

        if prev_combat_round is not None and combat > prev_combat_round:
            out.append("")
            out.append(
                f"  ┌─ Перед боем (состав досок после набора, combat {prev_combat_round}→{combat}) ─"
            )
            boards = prev_boards or {}
            for s in sorted(set(alive) | set(boards.keys())):
                if s not in boards:
                    continue
                out.append(format_board_line(f"S{s} стол", boards[s]))
            out.append("  └─")
            if prev_hp is not None:
                deltas = []
                for s in sorted(set(alive) | set(prev_hp.keys())):
                    cur_hp = int(_player(st, s).get("hp", 0))
                    d = cur_hp - prev_hp.get(s, cur_hp)
                    if d != 0:
                        deltas.append(f"S{s} {d:+d}")
                if deltas:
                    out.append(f"  ▸ Бой: Δhp {' / '.join(deltas)}")
            out.append("")

        show_details = extended or _turn_ended(rec, seat, player)
        if show_details:
            _render_seat_details(out, st, seat)

        if st.get("done"):
            w = st.get("winner")
            out.append(f"  ** конец лобби winner=S{w} **")

        prev_combat_round = combat
        prev_hp = {s: int(_player(st, s).get("hp", 0)) for s in range(8)}
        if prev_boards is None:
            prev_boards = {}
        for s in alive:
            prev_boards[s] = list(_player(st, s).get("board") or [])

    return "\n".join(out).rstrip() + "\n"


def render_jsonl_file(path: Union[str, Path], *, extended: bool = False) -> str:
    p = Path(path)
    return render_jsonl_records(p.read_text(encoding="utf-8").splitlines(), extended=extended)


def render_jsonl_to_stream(
    path: Union[str, Path],
    stream: TextIO,
    *,
    extended: bool = False,
) -> None:
    stream.write(render_jsonl_file(path, extended=extended))


__all__ = [
    "decode_env_action",
    "decode_env_action_compact",
    "render_jsonl_file",
    "render_jsonl_records",
    "render_jsonl_to_stream",
]
