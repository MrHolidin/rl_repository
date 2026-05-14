"""Human-readable text from MiniBG JSONL replays (no combat blow-by-blow)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, TextIO, Tuple, Union

from .action_map import (
    A_FINISH,
    A_LEVEL_UP,
    A_ROLL,
    NUM_ENV_ACTIONS,
    buy_slot,
    is_buy,
    is_discover_pick,
    is_finish,
    is_magnet,
    is_place,
    is_sell,
    is_swap_board,
    magnet_hand_board,
    discover_pick_slot,
    place_slot,
    sell_pos,
    swap_adj_index_from_env_action,
)


def _kw_short(kws: List[str]) -> str:
    if not kws:
        return ""
    abbr = {"TAUNT": "T", "SHIELD": "S"}
    return "".join(abbr.get(k, k[:1]) for k in kws)


def format_minion_slot(m: Optional[Dict[str, Any]]) -> str:
    if m is None:
        return "·"
    cid = m.get("card_id", "?")
    disp = m.get("name") or cid
    atk = m.get("atk", "?")
    hp = m.get("hp", "?")
    extra = _kw_short(list(m.get("kw") or []))
    suf = f" {extra}" if extra else ""
    return f"[{disp} {atk}/{hp}{suf}]"


def format_board_line(label: str, board: List[Optional[Dict[str, Any]]]) -> str:
    parts = [format_minion_slot(x) for x in board] if board else []
    body = " ".join(parts) if parts else "(пусто)"
    return f"  {label}: {body}"


def format_hand_line(label: str, hand: Optional[List[Optional[Dict[str, Any]]]]) -> str:
    """Pretty-print hand slots; ``hand is None`` means the replay record has no hand field."""
    if hand is None:
        return f"  {label}: (нет данных о руке в реплее)"
    parts = [format_minion_slot(x) for x in hand]
    any_card = any(x is not None for x in hand)
    body = " ".join(parts) if any_card else "(пусто)"
    return f"  {label}: {body}"


def format_shop_line(label: str, shop: Optional[List[Optional[Dict[str, Any]]]]) -> str:
    """Pretty-print tavern offers (fixed-width slots; empty slot is ``·``)."""
    if shop is None:
        return f"  {label}: (нет данных о лавке в реплее)"
    if not shop:
        return f"  {label}: (пусто)"
    parts = [format_minion_slot(x) for x in shop]
    return f"  {label}: {' '.join(parts)}"


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
    if is_finish(a):
        return "FINISH"
    if is_swap_board(a):
        j = swap_adj_index_from_env_action(a)
        return f"SWAP_BOARD_{j}_{j + 1}"
    if 0 <= a < NUM_ENV_ACTIONS:
        return f"UNKNOWN_{a}"
    return f"OUT_OF_RANGE_{a}"


def decode_env_action_compact(a: int) -> str:
    """Same actions as :func:`decode_env_action` but without shop/hand/board slot indices."""
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
    if is_finish(a):
        return "FINISH"
    if is_swap_board(a):
        return "SWAP_BOARD"
    if 0 <= a < NUM_ENV_ACTIONS:
        return f"UNKNOWN_{a}"
    return f"OUT_OF_RANGE_{a}"


def _player(state: Dict[str, Any], idx: int) -> Dict[str, Any]:
    return state.get(f"p{idx}", {}) or {}


def render_jsonl_records(lines: Iterator[str]) -> str:
    out: List[str] = []
    prev_round: Optional[int] = None
    prev_hp: Optional[Tuple[int, int]] = None
    prev_boards: Optional[Tuple[List[Any], List[Any]]] = None
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
            prev_round = None
            prev_hp = None
            prev_boards = None
            continue
        if t != "frame":
            continue

        st = rec.get("state") or {}
        rnd = int(st.get("round") or 0)
        p_act = int(rec.get("p", -1))
        a = int(rec.get("a", -1))
        illegal = bool(rec.get("illegal"))
        p0 = _player(st, 0)
        p1 = _player(st, 1)
        hp0 = int(p0.get("hp", 0))
        hp1 = int(p1.get("hp", 0))
        b0 = list(p0.get("board") or [])
        b1 = list(p1.get("board") or [])

        tag = "ILLEGAL " if illegal else ""
        act = decode_env_action_compact(a)
        out.append(
            f"{tag}R{rnd} P{p_act} {act}  "
            f"(gold {p0.get('gold')}/{p1.get('gold')}, tier {p0.get('tier')}/{p1.get('tier')}, "
            f"hp {hp0}/{hp1})"
        )
        h0_raw = p0.get("hand")
        h1_raw = p1.get("hand")
        h0: Optional[List[Optional[Dict[str, Any]]]] = (
            None if h0_raw is None else list(h0_raw)
        )
        h1: Optional[List[Optional[Dict[str, Any]]]] = (
            None if h1_raw is None else list(h1_raw)
        )
        s0_raw = p0.get("shop")
        s1_raw = p1.get("shop")
        s0: Optional[List[Optional[Dict[str, Any]]]] = (
            None if s0_raw is None else list(s0_raw)
        )
        s1: Optional[List[Optional[Dict[str, Any]]]] = (
            None if s1_raw is None else list(s1_raw)
        )
        out.append(format_board_line("стол P0", b0))
        out.append(format_board_line("стол P1", b1))
        out.append(format_hand_line("рука P0", h0))
        out.append(format_hand_line("рука P1", h1))
        out.append(format_shop_line("лавка P0", s0))
        out.append(format_shop_line("лавка P1", s1))

        if prev_round is not None and rnd > prev_round:
            out.append("")
            if prev_boards is not None:
                b0_pre, b1_pre = prev_boards
            else:
                b0_pre, b1_pre = b0, b1
            out.append(
                f"  ┌─ Перед боем (состав досок в конце раунда набора {prev_round}, "
                f"до разрешения боя) ─"
            )
            out.append(format_board_line("P0 стол", b0_pre))
            out.append(format_board_line("P1 стол", b1_pre))
            out.append("  └─")
            if prev_hp is not None:
                d0 = hp0 - prev_hp[0]
                d1 = hp1 - prev_hp[1]
                out.append(
                    f"  ▸ Бой: раунд {prev_round}→{rnd}, "
                    f"урон по игрокам (Δhp) P0 {d0:+d} / P1 {d1:+d} "
                    f"(hp теперь {hp0}/{hp1})"
                )
            out.append("")

        if st.get("done"):
            w = st.get("winner")
            out.append(f"  ** конец партии winner={w} **")

        prev_round = rnd
        prev_hp = (hp0, hp1)
        prev_boards = (list(b0), list(b1))

    return "\n".join(out).rstrip() + "\n"


def render_jsonl_file(path: Union[str, Path]) -> str:
    p = Path(path)
    return render_jsonl_records(p.read_text(encoding="utf-8").splitlines())


def render_jsonl_to_stream(path: Union[str, Path], stream: TextIO) -> None:
    stream.write(render_jsonl_file(path))


def board_compact_repr(board: List[Any]) -> str:
    parts = [format_minion_slot(x) for x in board] if board else []
    return " ".join(parts) if parts else "(пусто)"


def iter_pre_battle_rows(lines: Iterator[str]) -> Iterator[Dict[str, Any]]:
    """Yield one dict per combat, same ``prev_boards`` / ``prev_round`` logic as ``render_jsonl_records``.

    Rows describe boards at end of recruitment for ``shop_round_ended``, immediately before battle
    resolves and increments ``round`` to ``round_after``.
    """
    prev_round: Optional[int] = None
    prev_hp: Optional[Tuple[int, int]] = None
    prev_boards: Optional[Tuple[List[Any], List[Any]]] = None

    for raw in lines:
        raw = raw.strip()
        if not raw:
            continue
        rec = json.loads(raw)
        t = rec.get("type")
        if t == "episode_break":
            prev_round = None
            prev_hp = None
            prev_boards = None
            continue
        if t != "frame":
            continue

        st = rec.get("state") or {}
        rnd = int(st.get("round") or 0)
        p0 = _player(st, 0)
        p1 = _player(st, 1)
        hp0 = int(p0.get("hp", 0))
        hp1 = int(p1.get("hp", 0))
        b0 = list(p0.get("board") or [])
        b1 = list(p1.get("board") or [])

        if prev_round is not None and rnd > prev_round:
            if prev_boards is not None:
                b0_pre, b1_pre = prev_boards
            else:
                b0_pre, b1_pre = b0, b1
            d0 = d1 = 0
            if prev_hp is not None:
                d0 = hp0 - prev_hp[0]
                d1 = hp1 - prev_hp[1]
            yield {
                "shop_round_ended": prev_round,
                "round_after": rnd,
                "hp_p0_after": hp0,
                "hp_p1_after": hp1,
                "delta_hp_p0": d0,
                "delta_hp_p1": d1,
                "p0_table": board_compact_repr(b0_pre),
                "p1_table": board_compact_repr(b1_pre),
            }

        prev_round = rnd
        prev_hp = (hp0, hp1)
        prev_boards = (list(b0), list(b1))


__all__ = [
    "board_compact_repr",
    "decode_env_action",
    "decode_env_action_compact",
    "format_board_line",
    "format_hand_line",
    "format_shop_line",
    "format_minion_slot",
    "iter_pre_battle_rows",
    "render_jsonl_file",
    "render_jsonl_records",
    "render_jsonl_to_stream",
]
