#!/usr/bin/env python3
"""Aggregate stats from rendered MiniBG replay .txt files.

Parses lines produced by src.envs.minibg.replay_render.render_jsonl_file.

Shop cards are absent from .txt; ``BUY_SHOP_*`` in the txt report is summarized by slot.
Exact **purchase counts per creature** come from sibling ``.jsonl`` (shop snapshot before
each BUY maps slot → ``card_id``).
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path
from statistics import mean, median
from typing import Dict, Iterable, List, NamedTuple, Optional, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.envs.minibg.action_map import buy_slot, is_buy

RE_HEADER = re.compile(r"game_index=(?P<gi>\d+).*?match_seed=(?P<ms>\d+)")
RE_ACTION = re.compile(r"^R(\d+)\s+P(\d)\s+(.+?)\s{2}\(gold\b")
RE_HAND = re.compile(r"^\s+P(\d) рука \(на конец хода\): (.+)\s*$")
RE_BOARD = re.compile(r"^\s+P(\d) стол:\s(.+)\s*$")
RE_WINNER = re.compile(r"winner=([\d\-]+)")
# [guard 1/3 T]   [shield_bot 2/2 S]   [big_guy 5/5]
RE_MINION = re.compile(r"\[([a-z_]+)\s+(\d+)/(\d+)(?:\s+([^\]]))?\]")


class BattleSnap(NamedTuple):
    rnd: Optional[int]
    boards: Tuple[Tuple[str, ...], Tuple[str, ...]]  # (p0 tuples str, p1 tuples str)


def _parse_minions(line: str) -> Tuple[str, ...]:
    sigs: List[str] = []
    for m in RE_MINION.finditer(line):
        cid, atk, hp, kw = m.group(1), m.group(2), m.group(3), (m.group(4) or "").strip()
        sigs.append(f"{cid}:{atk}/{hp}:{kw}")
    return tuple(sigs)


def _hand_cards(hand_rest: str) -> Tuple[str, ...]:
    hr = hand_rest.strip()
    if hr.startswith("(пусто)") or hr == "(нет данных":
        return tuple()
    return _parse_minions(hand_rest)


def _multiset_subtraction(c_new: Counter, c_old: Counter) -> Tuple[Counter, Counter]:
    pos = Counter()
    neg = Counter()
    keys = set(c_new) | set(c_old)
    for k in keys:
        d = int(c_new[k]) - int(c_old[k])
        if d > 0:
            pos[k] = d
        elif d < 0:
            neg[k] = -d
    return pos, neg


def agent_token(seed: int, game_index: int) -> int:
    rng = random.Random(seed + game_index)
    return 1 if rng.random() < 0.5 else -1


def parse_txt(path: Path) -> Dict[str, object]:
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()

    header_line = lines[0] if lines else ""
    hm = RE_HEADER.search(header_line)
    if not hm:
        raise ValueError(f"no header(game_index/match_seed) in {path}")
    game_index = int(hm.group("gi"))
    match_seed = int(hm.group("ms"))
    ag_tok = agent_token(match_seed, game_index)
    pi_ag = 0 if ag_tok == 1 else 1

    actions_agent: Counter[str] = Counter()
    actions_opp: Counter[str] = Counter()
    buy_slot_ag: Counter[int] = Counter()
    buy_slot_opp: Counter[int] = Counter()

    hand_samples_ag: List[int] = []
    hand_samples_opp: List[int] = []
    max_hand_ag = max_hand_apex = 0

    snaps: List[BattleSnap] = []
    rnd_for_snap: Optional[int] = None
    pending_p0_board: Optional[str] = None

    winner: Optional[int] = None

    for line in lines:
        wm = RE_WINNER.search(line)
        if wm:
            winner = int(wm.group(1))

        am = RE_ACTION.match(line)
        if am:
            rnd, pi_s, verb = am.group(1), am.group(2), am.group(3).strip()
            rnd_for_snap = int(rnd)
            pi = int(pi_s)
            is_ag = pi == pi_ag

            ctr = actions_agent if is_ag else actions_opp
            if verb.startswith("BUY_SHOP_"):
                ctr["BUY"] += 1
                slot = int(verb.replace("BUY_SHOP_", ""))
                if is_ag:
                    buy_slot_ag[slot] += 1
                else:
                    buy_slot_opp[slot] += 1
            elif verb == "ROLL":
                ctr["ROLL"] += 1
            elif verb == "LEVEL_UP":
                ctr["LEVEL_UP"] += 1
            elif verb.startswith("PLACE_HAND"):
                ctr["PLACE"] += 1
            elif verb.startswith("SELL_BOARD"):
                ctr["SELL"] += 1
            elif verb == "FINISH":
                ctr["FINISH"] += 1
            elif verb.startswith("SWAP_BOARD") or verb.startswith("SELECT_ORDER"):
                ctr["SWAP_BOARD"] += 1
            else:
                ctr[verb.split()[0]] += 1
            continue

        hm_l = RE_HAND.match(line)
        if hm_l:
            pi = int(hm_l.group(1))
            cnt = len(_hand_cards(hm_l.group(2)))
            if pi == pi_ag:
                hand_samples_ag.append(cnt)
                max_hand_ag = max(max_hand_ag, cnt)
            else:
                hand_samples_opp.append(cnt)
                max_hand_apex = max(max_hand_apex, cnt)

        bm = RE_BOARD.match(line)
        if bm:
            pi = int(bm.group(1))
            rest = bm.group(2)
            if pi == 0:
                pending_p0_board = rest
            elif pi == 1 and pending_p0_board is not None:
                b0 = _parse_minions(pending_p0_board)
                b1 = _parse_minions(rest)
                snaps.append(BattleSnap(rnd_for_snap, (b0, b1)))
                pending_p0_board = None

    # Net tuple flow between consecutive previews
    net_gain_ag_sig: Counter[str] = Counter()
    net_gain_apex_sig: Counter[str] = Counter()
    for i in range(len(snaps) - 1):
        o0 = Counter(snaps[i].boards[0])
        n0 = Counter(snaps[i + 1].boards[0])
        o1 = Counter(snaps[i].boards[1])
        n1 = Counter(snaps[i + 1].boards[1])

        ga, _ = _multiset_subtraction(n0, o0)
        gx, _ = _multiset_subtraction(n1, o1)
        net_gain_ag_sig.update(ga)
        net_gain_apex_sig.update(gx)

    # Final board card_ids only (collapse stats)
    def id_mult(tup: Iterable[str]) -> Counter:
        out = Counter()
        for sig in tup:
            cid = sig.split(":", 1)[0]
            out[cid] += 1
        return out

    final_board_ids_ag = Counter()
    final_board_ids_apex = Counter()
    final_sig_ag: Tuple[str, ...] = tuple()
    final_sig_apex: Tuple[str, ...] = tuple()
    if snaps:
        final_sig_ag = snaps[-1].boards[pi_ag]
        final_sig_apex = snaps[-1].boards[1 - pi_ag]
        final_board_ids_ag = id_mult(final_sig_ag)
        final_board_ids_apex = id_mult(final_sig_apex)

    agent_win = winner is not None and winner == ag_tok

    return {
        "path": path,
        "game_index": game_index,
        "agent_win": agent_win,
        "winner": winner,
        "pi_ag": pi_ag,
        "actions_agent": actions_agent,
        "actions_opp": actions_opp,
        "buy_slot_ag": buy_slot_ag,
        "buy_slot_opp": buy_slot_opp,
        "hand_ag": hand_samples_ag,
        "hand_opp": hand_samples_opp,
        "max_hand_ag": max_hand_ag,
        "max_hand_apex": max_hand_apex,
        "final_board_ag": final_board_ids_ag,
        "final_board_apex": final_board_ids_apex,
        "net_gain_between_battles_ag": net_gain_ag_sig,
        "net_gain_between_battles_apex": net_gain_apex_sig,
        "battle_count": len(snaps),
    }


def _pct_bar(c: Counter, topn: int = 12) -> str:
    total = sum(c.values())
    if not total:
        return "  (пусто)"
    lines_out = []
    for k, v in c.most_common(topn):
        lines_out.append(f"  {k:24s}  {v:4d}  ({100 * v / total:5.1f}%)")
    return "\n".join(lines_out)


def creature_purchases_from_jsonl(jsonl_path: Path) -> Tuple[Counter[str], Counter[str]]:
    """BUY counts by ``card_id`` (shop before the buy = previous frame state)."""
    purch_ag = Counter[str]()
    purch_apex = Counter[str]()
    with jsonl_path.open(encoding="utf-8") as f:
        hdr = json.loads(f.readline())
        gi = int(hdr["game_index"])
        ms = int(hdr["match_seed"])
        pi_ag = 0 if agent_token(ms, gi) == 1 else 1
        prev_st: Optional[dict] = None
        for line in f:
            rec = json.loads(line)
            if rec.get("type") != "frame":
                continue
            st = rec["state"]
            p = int(rec["p"])
            a = int(rec["a"])
            if prev_st is not None and not rec.get("illegal") and is_buy(a):
                shop = (prev_st.get(f"p{p}") or {}).get("shop") or []
                slot = buy_slot(a)
                if 0 <= slot < len(shop) and shop[slot] is not None:
                    cid = str(shop[slot].get("card_id", "?"))
                    if cid and cid != "?":
                        if p == pi_ag:
                            purch_ag[cid] += 1
                        else:
                            purch_apex[cid] += 1
            prev_st = st

    return purch_ag, purch_apex


def _print_buy_by_creature(agent: Counter[str], apex: Counter[str]) -> None:
    ta = sum(agent.values())
    tx = sum(apex.values())
    print("\n=== ПОКУПКИ по существу (card_id), данные из .jsonl рядом с .txt ===")
    print(f"    всего успешных BUY — PPO агент: {ta},  apex_improved: {tx}")
    ids = sorted(set(agent) | set(apex))
    scored: List[tuple[int, str]] = []
    for cid in ids:
        scored.append((max(agent[cid], apex[cid]), cid))
    scored.sort(reverse=True)

    hdr = (
        f"{'card_id':<18}"
        f"  {'агент шт':>8}  {'агент %':>8}"
        f"  {'apex шт':>8}  {'apex %':>8}"
        f"  {'разница (аг−ап)':>14}"
        f"  {'кто покупает больше':>20}"
    )
    print(hdr)
    print("-" * len(hdr))
    for _, cid in scored:
        va, vx = agent[cid], apex[cid]
        pa = 100 * va / ta if ta else 0.0
        px = 100 * vx / tx if tx else 0.0
        d = va - vx
        if vx == va == 0:
            who = "—"
        elif va > vx:
            who = "агент"
        elif vx > va:
            who = "apex"
        else:
            who = "ровно"
        print(
            f"{cid:<18}"
            f"  {va:8d}  {pa:7.1f}%"
            f"  {vx:8d}  {px:7.1f}%"
            f"  {d:+14d}"
            f"  {who:>20}"
        )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "replay_dir",
        type=Path,
        help="Directory with *.txt from eval_progress --replay-dir",
    )
    ap.add_argument(
        "--no-creature-jsonl",
        action="store_true",
        help="Do not read sibling *.jsonl for BUY→card_id (only txt slot/action stats)",
    )
    args = ap.parse_args()
    d = args.replay_dir.resolve()
    files = sorted(d.glob("*.txt"))
    if not files:
        raise SystemExit(f"No .txt in {d}")

    parsed = [parse_txt(p) for p in files]

    purch_tot_ag = Counter[str]()
    purch_tot_apex = Counter[str]()
    jp_missing = 0
    if not args.no_creature_jsonl:
        for txt_path in files:
            jp = txt_path.with_suffix(".jsonl")
            if jp.is_file():
                a, x = creature_purchases_from_jsonl(jp)
                purch_tot_ag.update(a)
                purch_tot_apex.update(x)
            else:
                jp_missing += 1

    sum_ag = Counter()
    sum_apex = Counter()
    sum_buy_ag = Counter()
    sum_buy_apex = Counter()
    fin_ag_all = Counter()
    fin_ax_all = Counter()
    net_b_ag_all = Counter()
    net_b_ax_all = Counter()
    fin_ag_w = Counter()
    fin_ag_l = Counter()
    fin_ax_w = Counter()
    fin_ax_l = Counter()

    all_h_ag: List[int] = []
    all_h_ax: List[int] = []

    print(f"Разобрано реплеев: {len(parsed)} .txt из {d}\n")

    if not args.no_creature_jsonl:
        if jp_missing:
            print(f"⚠ Нет парного jsonl для {jp_missing} .txt — часть партий выпала из суммы BUY.\n")
        if sum(purch_tot_ag.values()) or sum(purch_tot_apex.values()):
            _print_buy_by_creature(purch_tot_ag, purch_tot_apex)

    print("Файлы / исход (агент=PPO)")
    print("-" * 72)
    for r in sorted(parsed, key=lambda x: int(x["game_index"])):
        gi = r["game_index"]
        tag = "W" if r["agent_win"] else "L"
        print(
            f"  game {gi:04d}  {tag}  battles={r['battle_count']}  "
            f"max_hand ag/apex={r['max_hand_ag']}/{r['max_hand_apex']}  "
            f"median_hand@FINISH ag/apex={median(r['hand_ag']) if r['hand_ag'] else 0:.1f}/"
            f"{median(r['hand_opp']) if r['hand_opp'] else 0:.1f}"
        )
        sum_ag.update(r["actions_agent"])
        sum_apex.update(r["actions_opp"])
        sum_buy_ag.update(r["buy_slot_ag"])
        sum_buy_apex.update(r["buy_slot_opp"])
        fin_ag_all.update(r["final_board_ag"])
        fin_ax_all.update(r["final_board_apex"])

        cid_ag = "|".join(f"{k}×{r['final_board_ag'][k]}" for k in sorted(r["final_board_ag"].keys()))
        cid_ax = "|".join(f"{k}×{r['final_board_apex'][k]}" for k in sorted(r["final_board_apex"].keys()))
        print(f"         final стол (card_id×n)  agent: [{cid_ag}]")
        print(f"                                apex : [{cid_ax}]")

        net_b_ag_all.update(r["net_gain_between_battles_ag"])
        net_b_ax_all.update(r["net_gain_between_battles_apex"])
        all_h_ag.extend(r["hand_ag"])
        all_h_ax.extend(r["hand_opp"])

        if r["agent_win"]:
            fin_ag_w.update(r["final_board_ag"])
            fin_ax_w.update(r["final_board_apex"])
        else:
            fin_ag_l.update(r["final_board_ag"])
            fin_ax_l.update(r["final_board_apex"])

    print("\nРука: после каждого FINISH (по тексту строк «рука (на конец хода)»)")
    print(
        f"  агент  n={len(all_h_ag)}  mean={mean(all_h_ag) if all_h_ag else 0:.2f} "
        f"median={median(all_h_ag) if all_h_ag else 0:.2f}"
    )
    print(
        f"  apex   n={len(all_h_ax)} mean={mean(all_h_ax) if all_h_ax else 0:.2f} "
        f"median={median(all_h_ax) if all_h_ax else 0:.2f}"
    )

    print("\nПокупки: слот магазина (BUY_SHOP_0/_1/_2) — только .txt")
    print("  агент:" + "".join(f"  slot{s}:{sum_buy_ag.get(s, 0)}" for s in (0, 1, 2)))
    print("  apex : " + "".join(f"  slot{s}:{sum_buy_apex.get(s, 0)}" for s in (0, 1, 2)))

    def pct_actions(label: str, c: Counter) -> None:
        tot = sum(c.values())
        print(f"\nСмесь действий в логах — {label} (всего {tot})")
        if tot:
            for k, v in c.most_common(14):
                print(f"  {k:22s}  {v:4d}  ({100 * v / tot:5.1f}%)")

    pct_actions("агент", sum_ag)
    pct_actions("apex_improved", sum_apex)

    print("\nФинальный стол (card_id), все партии суммой")
    tot_ag_slots = sum(fin_ag_all.values())
    tot_ax_slots = sum(fin_ax_all.values())
    print(f"  агент (всего слотов миньонов {tot_ag_slots}):")
    print(_pct_bar(fin_ag_all))
    print("  apex:")
    print(_pct_bar(fin_ax_all))

    print("\nФинальный стол агента: победа vs поражение")
    print("  WIN:")
    print(_pct_bar(fin_ag_w))
    print("  LOSS:")
    print(_pct_bar(fin_ag_l))

    print(
        "\nМежду соседними блоками «Перед боём»: положительная дельта сигнатур миньонов "
        "(id:atk/hp:kw).\nНе равно «покупкам»: баффы видны как новая сигнатура..."
    )
    print("  агент:")
    sig_summary_ag = Counter()
    for sig, n in net_b_ag_all.items():
        cid = sig.split(":")[0]
        sig_summary_ag[cid] += n
    print(_pct_bar(sig_summary_ag, topn=14))
    print("  apex:")
    sig_summary_ax = Counter()
    for sig, n in net_b_ax_all.items():
        cid = sig.split(":")[0]
        sig_summary_ax[cid] += n
    print(_pct_bar(sig_summary_ax, topn=14))


if __name__ == "__main__":
    main()
