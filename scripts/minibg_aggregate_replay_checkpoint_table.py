#!/usr/bin/env python3
"""Сводная таблица по чекпоинтам из готовых MiniBG JSONL-реплеев (без доигрывания).

Читает каталог вида ``run_dir/replay_board_stats_selfplay/<stem>/*.jsonl`` (или
``--replay-root``). Для каждого реплея берёт последний frame с ``done`` и считает:
финальный раунд, число кадров (длина реплея), исключённая из магазина раса,
статы досок обоих игроков (Σatk+hp, число миньонов, тир, доминирующая раса на доске).

Пишет ``--out-csv`` и ``--out-md``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def _step_from_stem(stem: str) -> int:
    if stem.endswith("_final"):
        # Соответствует типичному train.total_steps=2e6 в конфиге ppo_structured.
        return 2_000_000
    m = re.search(r"_(\d+)$", stem)
    return int(m.group(1)) if m else -1


def _step_label(stem: str) -> str:
    return "final" if stem.endswith("_final") else str(_step_from_stem(stem))


def _dominant_board_race(board: List[Dict[str, Any]]) -> str:
    c = Counter()
    for m in board:
        if not isinstance(m, dict):
            continue
        r = m.get("race")
        if r is None:
            continue
        c[str(r)] += 1
    if not c:
        return "none"
    return max(c.items(), key=lambda kv: kv[1])[0]


def _terminal_stats(path: Path) -> Optional[Dict[str, Any]]:
    learned_idx: Optional[int] = None
    last: Optional[Dict[str, Any]] = None
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        t = rec.get("type")
        if t == "header":
            learned_idx = int(rec.get("learned_player_index", -1))
            continue
        if t != "frame":
            continue
        st = rec.get("state") or {}
        if st.get("done"):
            last = {
                "learned_player_index": learned_idx,
                "frame_i": int(rec.get("i") or 0),
                "round": int(st.get("round") or 0),
                "shop_excluded_race": str(st.get("shop_excluded_race") or "none"),
            }
            for pi in ("p0", "p1"):
                pl = st.get(pi) or {}
                board = pl.get("board") or []
                atk = hp = 0
                for m in board:
                    if isinstance(m, dict):
                        atk += int(m.get("atk") or 0)
                        hp += int(m.get("hp") or 0)
                last[f"{pi}_sum"] = atk + hp
                last[f"{pi}_nmin"] = len(board)
                last[f"{pi}_tier"] = int(pl.get("tier") or 0)
                last[f"{pi}_hero_hp"] = int(pl.get("hp") or 0)
                last[f"{pi}_dom"] = _dominant_board_race(board)
    if last is None:
        return None
    return last


def _mean_std(xs: List[float]) -> Tuple[float, float]:
    if not xs:
        return float("nan"), float("nan")
    m = sum(xs) / len(xs)
    if len(xs) < 2:
        return m, float("nan")
    v = sum((x - m) ** 2 for x in xs) / (len(xs) - 1)
    return m, math.sqrt(v)


def _pct_counts(counter: Counter, n: int) -> Dict[str, float]:
    if n <= 0:
        return {}
    return {k: 100.0 * counter[k] / n for k in counter}


def iter_jsonl_files(replay_root: Path) -> Iterable[Tuple[str, Path]]:
    for sub in sorted(replay_root.iterdir()):
        if not sub.is_dir():
            continue
        for jp in sorted(sub.glob("*.jsonl")):
            yield sub.name, jp


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--replay-root",
        type=Path,
        required=True,
        help="Например runs/.../replay_board_stats_selfplay",
    )
    ap.add_argument("--out-csv", type=Path, required=True)
    ap.add_argument("--out-md", type=Path, required=True)
    ap.add_argument(
        "--prefix",
        type=str,
        default="minibg_ppo_structured",
        help="Только подкаталоги, чьё имя начинается с prefix (после фильтра по stem).",
    )
    ap.add_argument(
        "--min-step",
        type=int,
        default=None,
        help="Оставить только чекпоинты с step >= этого (включая final как 2e6).",
    )
    ap.add_argument(
        "--max-step",
        type=int,
        default=None,
        help="Оставить только чекпоинты с step <= этого.",
    )
    args = ap.parse_args()
    root: Path = args.replay_root.resolve()

    by_stem: Dict[str, List[Dict[str, Any]]] = {}
    for stem, jp in iter_jsonl_files(root):
        if not stem.startswith(args.prefix):
            continue
        stats = _terminal_stats(jp)
        if stats is None:
            continue
        by_stem.setdefault(stem, []).append(stats)

    stems_sorted = sorted(by_stem.keys(), key=_step_from_stem)
    if args.min_step is not None:
        stems_sorted = [
            s for s in stems_sorted if _step_from_stem(s) >= args.min_step
        ]
    if args.max_step is not None:
        stems_sorted = [
            s for s in stems_sorted if _step_from_stem(s) <= args.max_step
        ]

    excl_keys = ["BEAST", "DEMON", "MECHANICAL", "MURLOC", "none"]
    dom_keys = ["BEAST", "DEMON", "MECHANICAL", "MURLOC", "ALL", "none"]

    md_rows: List[str] = []
    md_rows.append(
        "| step | n | round μ | round σ | frames μ | Σboard μ | n min μ | tier μ | "
        "excl B/D/Me/Mu/— % | dom (pooled) top |"
    )
    md_rows.append(
        "|---:|---:|---:|---:|---:|---:|---:|---:|---|---|"
    )

    csv_rows: List[Dict[str, Any]] = []

    for stem in stems_sorted:
        rows = by_stem[stem]
        step = _step_from_stem(stem)
        step_disp = _step_label(stem)
        n = len(rows)
        rounds = [float(r["round"]) for r in rows]
        frames = [float(r["frame_i"]) for r in rows]
        sums = []
        nmins = []
        tiers = []
        for r in rows:
            sums.append(float(r["p0_sum"] + r["p1_sum"]) / 2.0)
            nmins.append(float(r["p0_nmin"] + r["p1_nmin"]) / 2.0)
            tiers.append(float(r["p0_tier"] + r["p1_tier"]) / 2.0)

        r_m, r_s = _mean_std(rounds)
        f_m, _ = _mean_std(frames)
        s_m, _ = _mean_std(sums)
        nm_m, _ = _mean_std(nmins)
        t_m, _ = _mean_std(tiers)

        excl_ctr: Counter = Counter(r["shop_excluded_race"] for r in rows)
        dom_ctr: Counter = Counter()
        for r in rows:
            dom_ctr[r["p0_dom"]] += 1
            dom_ctr[r["p1_dom"]] += 1
        excl_pct = _pct_counts(excl_ctr, n)
        dom_pct = _pct_counts(dom_ctr, 2 * n)

        excl_abbr = {
            "BEAST": "B",
            "DEMON": "D",
            "MECHANICAL": "Me",
            "MURLOC": "Mu",
            "none": "—",
        }
        excl_compact = "/".join(
            f"{excl_abbr[k]}{int(excl_pct.get(k, 0))}" for k in excl_keys
        )
        dom_top = sorted(dom_pct.items(), key=lambda kv: -kv[1])[:3]
        dom_str = ", ".join(f"{k} {v:.0f}%" for k, v in dom_top)

        md_rows.append(
            f"| {step_disp} | {n} | {r_m:.1f} | {r_s:.1f} | {f_m:.0f} | {s_m:.1f} | "
            f"{nm_m:.2f} | {t_m:.2f} | {excl_compact} | {dom_str} |"
        )

        rec: Dict[str, Any] = {
            "step": step,
            "step_label": step_disp,
            "checkpoint_stem": stem,
            "n_replays": n,
            "round_mean": round(r_m, 3),
            "round_std": round(r_s, 3),
            "replay_frames_mean": round(f_m, 3),
            "board_sum_atk_hp_mean_both_avg": round(s_m, 3),
            "board_minions_mean_both_avg": round(nm_m, 3),
            "tavern_tier_mean_both_avg": round(t_m, 3),
        }
        for k in excl_keys:
            rec[f"shop_excl_{k}_pct"] = round(excl_pct.get(k, 0.0), 2)
        for k in dom_keys:
            rec[f"dominant_board_{k}_pct_pooled"] = round(dom_pct.get(k, 0.0), 2)
        csv_rows.append(rec)

    fieldnames = list(csv_rows[0].keys()) if csv_rows else []
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", newline="", encoding="utf-8") as f:
        if fieldnames:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(csv_rows)
        else:
            f.write("")

    args.out_md.write_text(
        "\n".join(md_rows)
        + "\n\n*Источник: последний `frame` с `done` в JSONL. Σboard — среднее (p0+p1)/2 по сумме atk+hp на доске. "
        "frames — индекс `i` терминального кадра. dom — доли доминирующей расы на доске, пул 2 досок на игру. "
        "excl — доли по `shop_excluded_race`: B=BEAST, D=DEMON, Me=MECHANICAL, Mu=MURLOC, —=все племена в пуле.*\n",
        encoding="utf-8",
    )
    print(f"Wrote {args.out_csv} ({len(csv_rows)} rows)")
    print(f"Wrote {args.out_md}")


if __name__ == "__main__":
    main()
