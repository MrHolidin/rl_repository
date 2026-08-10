#!/usr/bin/env python3
"""Play BGLike from the console against checkpoint bots (command REPL).

You sit at one of the 8 lobby seats; the other 7 are driven by a loaded
checkpoint. During your shop phase you type commands (one per line):

    roll              refresh the shop                 (alias: r)
    level             upgrade tavern tier              (alias: lvl, up)
    buy N             buy shop slot N                  (alias: b N)
    sell N            sell board minion at slot N      (alias: s N)
    play N            place hand minion N on board     (alias: p N)
    magnet H B        magnetize hand minion H onto board mech B
    swap I J          swap board positions I and J     (any time in shop)
    pick N            pick discover/triple option N (0-2)
    target N          choose battlecry target board slot N
    skip              skip the optional second target
    end               finish turn -> combat            (alias: e, done)
    end freeze        finish turn and freeze the shop
    help              reprint this help                (alias: h, ?)
    quit              abandon the game                 (alias: q)

Example (heroes checkpoint, the default):

    python scripts/play_bglike_human.py \
        --ckpt runs/bglike/v11_heroes_10M_w50_lr2p5e5_capture/checkpoints/dist_bglike_ppo_v11_heroes_7604118.pt \
        --patch-dir data/bgcore/19_6_0_74257 --obs-kind bglike_v5_heroes --with-heroes

For a no-hero checkpoint drop --with-heroes and pass --obs-kind bglike (or
bglike_v5), plus the matching --patch-dir if the checkpoint is patch-pinned.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Set, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

torch.set_num_threads(1)  # batch=1 forwards are faster single-threaded

import src.envs  # noqa: F401
from src.bg_core.effects import Keyword
from src.bg_recruitment.economy import (
    effective_buy_cost,
    effective_level_up_cost,
    effective_roll_cost,
)
from src.envs.bglike.action_map import A_SWAP_BOARD_0
from src.envs.bglike.actions import (
    MAX_TIER,
    NUM_PLAYERS,
    STARTING_HEALTH,
)
from src.envs.bglike.lobby_env import BGLobbyEnv
from src.envs.bglike.placement import is_seat_finished, placement_for_seat
from src.envs.bglike.seat_config import lobby_from_learned_seats
from src.envs.minibg.structured_actions import StructAction, StructActionType
from src.evaluation.eval_checkpoints import load_training_agent_checkpoint

BOARD_SIZE = 7

_KW_SHORT = {
    Keyword.TAUNT: "T",
    Keyword.SHIELD: "DS",
    Keyword.WINDFURY: "WF",
    Keyword.MEGA_WINDFURY: "MWF",
    Keyword.POISONOUS: "PSN",
    Keyword.CHARGE: "CH",
    Keyword.MAGNETIC: "MAG",
    Keyword.REBORN: "RB",
}


class BadCommand(Exception):
    """Raised for malformed / illegal console input (caught, reprompted)."""


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #
def card_name(env: BGLobbyEnv, card_id: str) -> str:
    """Resolve a raw card_id (discover/triple option) to its display name."""
    patch = env.game._patch
    try:
        return patch.describe(card_id).name
    except (KeyError, AttributeError):
        pass
    tpl = patch.templates.get(card_id) if hasattr(patch.templates, "get") else None
    if tpl is not None and tpl.name:
        return tpl.name
    return card_id


def fmt_minion(env: BGLobbyEnv, m) -> str:
    if m is None:
        return "--"
    star = "*" if getattr(m, "is_golden", False) else ""
    kws = sorted(_KW_SHORT[k] for k in m.all_keywords if k in _KW_SHORT)
    tag = (" " + ",".join(kws)) if kws else ""
    name = m.name or card_name(env, m.card_id)
    return f"{star}{name} {m.raw_attack}/{m.max_health}{tag}"


def _slot_line(env: BGLobbyEnv, label: str, items, costs=None, frozen=None) -> str:
    parts = []
    for i, m in enumerate(items):
        s = fmt_minion(env, m)
        if m is not None and costs is not None:
            s += f" ({costs}g)"
        if m is not None and frozen is not None and i < len(frozen) and frozen[i]:
            s += " [FROZEN]"
        parts.append(f"[{i}] {s}")
    body = "  ".join(parts) if parts else "(empty)"
    return f"{label}: {body}"


def render_state(env: BGLobbyEnv, seat: int) -> None:
    st = env.state
    p = st.players[seat]
    print()
    print("=" * 72)
    hero = f"  Hero: {p.hero.name}" if p.hero is not None else ""
    print(
        f"Round {st.round_number}   You(seat {seat})  "
        f"HP:{p.health}  Gold:{p.gold}  Tier:{p.tavern_tier}{hero}"
    )
    # Opponents
    opp = []
    for s in range(NUM_PLAYERS):
        if s == seat:
            continue
        op = st.players[s]
        alive = (s in st.alive) and not is_seat_finished(st, s)
        flag = "" if alive else " (out)"
        opp.append(f"s{s}:HP{op.health}/T{op.tavern_tier}{flag}")
    print("Opponents: " + "  ".join(opp))
    print("-" * 72)
    buy_c = effective_buy_cost(p)
    print(_slot_line(env, "Shop ", p.shop, costs=buy_c, frozen=p.shop_frozen))
    print(_slot_line(env, "Board", p.board))
    print(_slot_line(env, "Hand ", p.hand))
    roll_c = effective_roll_cost(p)
    lvl_c = effective_level_up_cost(p) if p.tavern_tier < MAX_TIER else None
    econ = f"roll={roll_c}g"
    econ += f"  level={lvl_c}g" if lvl_c is not None else "  level=MAX"
    print(econ)
    if p.pending_choice is not None:
        pc = p.pending_choice
        opts = "  ".join(
            f"[{i}] {card_name(env, o)}" for i, o in enumerate(pc.options)
        )
        print(f"CHOOSE ({pc.kind.name}) -> use `pick N`:  {opts}")
    plan = env.rl_pending_for_seat(seat)
    if plan is not None:
        print("BATTLECRY needs a target -> use `target N` (or `skip`).")
    print("-" * 72)


def render_combat(env: BGLobbyEnv, seat: int, prev_hp: int, prev_elim: Set[int]) -> None:
    st = env.state
    now_hp = st.players[seat].health
    delta = prev_hp - now_hp
    if delta > 0:
        outcome = f"you took {delta} damage (HP {prev_hp} -> {now_hp})"
    elif now_hp > prev_hp:
        outcome = f"you healed (HP {prev_hp} -> {now_hp})"
    else:
        outcome = f"no damage (HP {now_hp})"
    newly = sorted({snap.seat for snap in st.eliminated} - prev_elim)
    elim_txt = f"  eliminated: {newly}" if newly else ""
    print(f"\n*** COMBAT (round {st.combat_round}): {outcome}{elim_txt} ***")
    snaps = st.players[seat].last_battle_snapshots
    if snaps:
        opp = snaps[0].opp_board
        body = "  ".join(f"[{i}] {fmt_minion(env, m)}" for i, m in enumerate(opp))
        print("Opponent board: " + (body if body else "(empty)"))


HELP_TEXT = __doc__


# --------------------------------------------------------------------------- #
# Command handling
# --------------------------------------------------------------------------- #
def _legal_lookup(env: BGLobbyEnv, seat: int):
    legal = env.legal_structured_actions_for_seat(seat)
    return {(sa.type, tuple(sa.args)) for sa in legal}


def _require(legal_set, sa: StructAction) -> None:
    if (sa.type, tuple(sa.args)) not in legal_set:
        raise BadCommand(f"illegal move right now: {sa.type.name} {sa.args}")


def _parse_ints(rest: List[str], n: int, names: str) -> Tuple[int, ...]:
    if len(rest) != n:
        raise BadCommand(f"expected {n} number(s): {names}")
    try:
        return tuple(int(x) for x in rest)
    except ValueError:
        raise BadCommand(f"arguments must be integers: {names}")


def _apply_swap(env: BGLobbyEnv, seat: int, i: int, j: int) -> None:
    p = env.state.players[seat]
    if p.pending_choice is not None or env.rl_pending_for_seat(seat) is not None:
        raise BadCommand("cannot reorder the board mid-choice")
    k = len(p.board)
    if not (0 <= i < k and 0 <= j < k):
        raise BadCommand(f"board positions must be in [0,{k - 1}]")
    if i == j:
        raise BadCommand("positions are identical")
    a, b = (i, j) if i < j else (j, i)
    # Exact transposition (a,b) keeping all other positions fixed, expressed as
    # adjacent swaps. `A_SWAP_BOARD_0 + k` swaps positions (k, k+1).
    # Phase 1: bubble the element at b left to a -> swap (k-1,k) for k=b..a+1.
    for k in range(b, a, -1):
        env.step_action(seat, A_SWAP_BOARD_0 + (k - 1))
    # Phase 2: bubble the element now at a+1 right to b -> swap (k,k+1) for k=a+1..b-1.
    for k in range(a + 1, b):
        env.step_action(seat, A_SWAP_BOARD_0 + k)


def handle_command(env: BGLobbyEnv, seat: int, line: str) -> None:
    toks = line.split()
    if not toks:
        return
    cmd, rest = toks[0].lower(), toks[1:]
    legal = _legal_lookup(env, seat)
    identity = tuple(range(BOARD_SIZE))

    if cmd in ("help", "h", "?"):
        print(HELP_TEXT)
    elif cmd in ("roll", "r"):
        sa = StructAction(StructActionType.ROLL)
        _require(legal, sa)
        env.step_structured_for_seat(seat, sa)
    elif cmd in ("level", "lvl", "up"):
        sa = StructAction(StructActionType.LEVEL_UP)
        _require(legal, sa)
        env.step_structured_for_seat(seat, sa)
    elif cmd in ("buy", "b"):
        (n,) = _parse_ints(rest, 1, "shop slot")
        sa = StructAction(StructActionType.BUY, (n,))
        _require(legal, sa)
        env.step_structured_for_seat(seat, sa)
    elif cmd in ("sell", "s"):
        (n,) = _parse_ints(rest, 1, "board slot")
        sa = StructAction(StructActionType.SELL, (n,))
        _require(legal, sa)
        env.step_structured_for_seat(seat, sa)
    elif cmd in ("play", "p"):
        (n,) = _parse_ints(rest, 1, "hand slot")
        sa = StructAction(StructActionType.PLAY, (n,))
        _require(legal, sa)
        env.step_structured_for_seat(seat, sa)
    elif cmd in ("magnet", "m"):
        h, b = _parse_ints(rest, 2, "hand_slot board_pos")
        sa = StructAction(StructActionType.MAGNET, (h, b))
        _require(legal, sa)
        env.step_structured_for_seat(seat, sa)
    elif cmd == "swap":
        i, j = _parse_ints(rest, 2, "pos_i pos_j")
        _apply_swap(env, seat, i, j)
    elif cmd == "pick":
        (n,) = _parse_ints(rest, 1, "option 0-2")
        sa = StructAction(StructActionType.DISCOVER_PICK, (n,))
        _require(legal, sa)
        env.step_structured_for_seat(seat, sa)
    elif cmd == "target":
        (n,) = _parse_ints(rest, 1, "board slot")
        sa = StructAction(StructActionType.APPLY_EFFECT, (n,))
        _require(legal, sa)
        env.step_structured_for_seat(seat, sa)
    elif cmd == "skip":
        sa = StructAction(StructActionType.APPLY_EFFECT_SKIP)
        _require(legal, sa)
        env.step_structured_for_seat(seat, sa)
    elif cmd in ("end", "e", "done", "finish"):
        freeze = bool(rest) and rest[0].lower() in ("freeze", "f")
        sa = StructAction(
            StructActionType.COMPLETE_TURN_FREEZE_SHOP
            if freeze
            else StructActionType.COMPLETE_TURN
        )
        _require(legal, sa)
        env.step_structured_for_seat(seat, sa, board_perm=identity)
    elif cmd in ("quit", "q", "exit"):
        raise KeyboardInterrupt
    else:
        raise BadCommand(f"unknown command {cmd!r} (type `help`)")


def human_turn(env: BGLobbyEnv, seat: int) -> None:
    while (
        not env.lobby_done
        and env.current_seat() == seat
        and env._seat_can_act(seat)
    ):
        render_state(env, seat)
        try:
            line = input("> ").strip()
        except EOFError:
            raise KeyboardInterrupt
        try:
            handle_command(env, seat, line)
        except BadCommand as e:
            print(f"  ! {e}")


# --------------------------------------------------------------------------- #
# Main loop
# --------------------------------------------------------------------------- #
def play(
    *,
    ckpt: Path,
    seat: int,
    seed: int,
    device: str,
    patch_dir: Optional[str],
    obs_kind: str,
    with_heroes: bool,
    greedy: bool,
) -> None:
    agent = load_training_agent_checkpoint(ckpt, device=device, seed=seed)
    if hasattr(agent, "eval"):
        agent.eval()
    bot_seats = tuple(s for s in range(NUM_PLAYERS) if s != seat)
    configs = lobby_from_learned_seats(
        bot_seats, agent_by_seat={s: agent for s in bot_seats}, seed=seed
    )
    env = BGLobbyEnv(
        configs,
        learned_seats=bot_seats,
        seed=seed,
        patch_dir=patch_dir,
        obs_kind=obs_kind,
        with_heroes=with_heroes,
    )
    env.reset(seed=seed)

    print("=" * 72)
    print(f"BGLike console game -- you are seat {seat}, bots: {ckpt.name}")
    print("Type `help` for commands.")

    prev_cr = env.state.combat_round
    prev_hp = env.state.players[seat].health
    prev_elim: Set[int] = {snap.seat for snap in env.state.eliminated}

    while not env.lobby_done:
        cur = env.current_seat()
        if cur == seat and env._seat_can_act(seat):
            human_turn(env, seat)
        elif env._seat_can_act(cur):
            env.step_auto(cur, deterministic=greedy)
        else:
            break  # safety: no actor can move and lobby not done

        if env.state.combat_round != prev_cr:
            render_combat(env, seat, prev_hp, prev_elim)
            prev_cr = env.state.combat_round
            prev_hp = env.state.players[seat].health
            prev_elim = {snap.seat for snap in env.state.eliminated}

        if is_seat_finished(env.state, seat):
            break

    place = placement_for_seat(env.state, seat)
    print("\n" + "=" * 72)
    if env.lobby_done and env.state.winner == seat:
        print("You WON the lobby! 🏆")
    print(f"Final placement: {place} / {NUM_PLAYERS}  (1 = best)")


def main() -> None:
    ap = argparse.ArgumentParser(description="Play BGLike against checkpoint bots.")
    ap.add_argument(
        "--ckpt",
        type=Path,
        default=REPO_ROOT
        / "runs/bglike/v11_heroes_10M_w50_lr2p5e5_capture/checkpoints/dist_bglike_ppo_v11_heroes_7604118.pt",
        help="checkpoint to drive the 7 bots",
    )
    ap.add_argument("--seat", type=int, default=0, help="your seat (0-7)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument(
        "--patch-dir",
        type=str,
        default="data/bgcore/19_6_0_74257",
        help="patch package dir (required for patch-pinned checkpoints)",
    )
    ap.add_argument(
        "--obs-kind",
        type=str,
        default="bglike_v5_heroes",
        help="bglike / bglike_v5 / bglike_v5_heroes (must match the checkpoint)",
    )
    ap.add_argument(
        "--with-heroes",
        action="store_true",
        default=True,
        help="assign a random hero per seat (required for heroes nets)",
    )
    ap.add_argument(
        "--no-heroes",
        dest="with_heroes",
        action="store_false",
        help="disable heroes (use with non-hero checkpoints)",
    )
    ap.add_argument(
        "--greedy",
        action="store_true",
        help="bots play greedily (deterministic) instead of sampling",
    )
    args = ap.parse_args()

    if not 0 <= args.seat < NUM_PLAYERS:
        raise SystemExit(f"--seat must be in [0,{NUM_PLAYERS - 1}]")
    if not args.ckpt.exists():
        raise SystemExit(f"checkpoint not found: {args.ckpt}")

    try:
        play(
            ckpt=args.ckpt.resolve(),
            seat=args.seat,
            seed=args.seed,
            device=args.device,
            patch_dir=args.patch_dir,
            obs_kind=args.obs_kind,
            with_heroes=args.with_heroes,
            greedy=args.greedy,
        )
    except KeyboardInterrupt:
        print("\nAbandoned.")


if __name__ == "__main__":
    main()
