"""Human-readable text from MiniBG JSONL replays (no combat blow-by-blow)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, TextIO, Tuple, Union

from .action_map import (
    A_BUY_BASE,
    A_FINISH,
    A_LEVEL_UP,
    A_PLACE_BASE,
    A_ROLL,
    A_SELECT_ORDER_BASE,
    A_SELL_BASE,
    NUM_ENV_ACTIONS,
    PERMUTATIONS_4,
    buy_slot,
    is_buy,
    is_finish,
    is_place,
    is_select_order,
    is_sell,
    order_index,
    place_slot,
    sell_pos,
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
    atk = m.get("atk", "?")
    hp = m.get("hp", "?")
    extra = _kw_short(list(m.get("kw") or []))
    suf = f" {extra}" if extra else ""
    return f"[{cid} {atk}/{hp}{suf}]"


def format_board_line(label: str, board: List[Optional[Dict[str, Any]]]) -> str:
    parts = [format_minion_slot(x) for x in board] if board else []
    body = " ".join(parts) if parts else "(пусто)"
    return f"  {label}: {body}"


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
    if is_finish(a):
        return "FINISH"
    if is_select_order(a):
        j = order_index(a)
        perm = PERMUTATIONS_4[j] if 0 <= j < len(PERMUTATIONS_4) else ()
        return f"SELECT_ORDER perm#{j} {perm}"
    if 0 <= a < NUM_ENV_ACTIONS:
        return f"UNKNOWN_{a}"
    return f"OUT_OF_RANGE_{a}"


def _player(state: Dict[str, Any], idx: int) -> Dict[str, Any]:
    return state.get(f"p{idx}", {}) or {}


def render_jsonl_records(lines: Iterator[str]) -> str:
    out: List[str] = []
    prev_round: Optional[int] = None
    prev_hp: Optional[Tuple[int, int]] = None

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
        act = decode_env_action(a)
        out.append(
            f"{tag}R{rnd} P{p_act} {act}  "
            f"(gold {p0.get('gold')}/{p1.get('gold')}, tier {p0.get('tier')}/{p1.get('tier')}, "
            f"hp {hp0}/{hp1})"
        )

        if prev_round is not None and rnd > prev_round:
            out.append("")
            out.append(
                f"  ┌─ Перед боем (состав досок после магазина раунда {prev_round}; "
                f"бой не меняет расстановку на доске) ─"
            )
            out.append(format_board_line("P0 стол", b0))
            out.append(format_board_line("P1 стол", b1))
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

    return "\n".join(out).rstrip() + "\n"


def render_jsonl_file(path: Union[str, Path]) -> str:
    p = Path(path)
    return render_jsonl_records(p.read_text(encoding="utf-8").splitlines())


def render_jsonl_to_stream(path: Union[str, Path], stream: TextIO) -> None:
    stream.write(render_jsonl_file(path))


__all__ = [
    "decode_env_action",
    "format_board_line",
    "format_minion_slot",
    "render_jsonl_file",
    "render_jsonl_records",
    "render_jsonl_to_stream",
]
