#!/usr/bin/env python3
"""Synthetic policy probes: low-tier elemental synergies (patch 74257).

Compares softmax preference for synergy-aligned vs neutral actions on crafted
shop states. Intended for quick checkpoint sanity checks (e.g. dist_ppo_056 @ 2M).
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import src.envs  # noqa: F401
from src.bg_catalog.patch_context import load_patch_context
from src.bg_core.minion import Race
from src.bg_lobby.player import PlayerPhase, PlayerState
from src.envs.bglike.actions import HAND_SIZE, MAX_SHOP_SLOTS, gold_for_round
from src.envs.bglike.game import BGLikeGame
from src.envs.bglike.lobby_env import BGLobbyEnv, OBS_KIND_BGLIKE_V5
from src.envs.bglike.seat_config import lobby_from_learned_seats
from src.envs.bglike.state import BGLikeState
from src.envs.minibg.structured_actions import StructAction, StructActionType
from src.evaluation.eval_checkpoints import load_training_agent_checkpoint
from src.utils import freeze_agent

PATCH_DIR = "data/bgcore/19_6_0_74257"

# Low-tier elemental synergy pieces (patch 74257)
SELLEMENTAL = "BGS_115"
PARTY_ELEMENTAL = "BGS_120"
MOLTEN_ROCK = "BGS_127"
ARCANE_ASSISTANT = "BGS_128"
REFRESHING_ANOMALY = "BGS_116"

# Neutral fillers (no elemental tribe)
ALLEYCAT = "CFM_315"  # tier 1 beast
PILOTED_SHREDDER = "BGS_023"  # tier 3 mechanical (in pool)


@dataclass(frozen=True)
class ProbeCase:
    name: str
    description: str
    good: StructAction
    bad: StructAction
    setup: Callable[[BGLikeGame, int], BGLikeState]


def _empty_hand() -> List[Optional["Minion"]]:
    from src.bg_core.minion import Minion

    return [None] * HAND_SIZE


def _hand(*cards: str, patch=None) -> List[Optional["Minion"]]:
    from src.bg_core.minion import Minion

    out: List[Optional[Minion]] = [None] * HAND_SIZE
    for i, cid in enumerate(cards):
        out[i] = patch.make_minion(cid)
    return out


def _shop(*cards: Optional[str], patch=None) -> List[Optional["Minion"]]:
    out: List[Optional["Minion"]] = [None] * MAX_SHOP_SLOTS
    for i, cid in enumerate(cards):
        if cid is not None:
            out[i] = patch.make_minion(cid)
    return out


def _board(*cards: str, patch=None) -> List:
    return [patch.make_minion(c) for c in cards]


def _base_state(
    game: BGLikeGame,
    seat: int,
    *,
    round_number: int = 3,
    tavern_tier: int = 2,
    gold: int = 10,
    board: Sequence[str] = (),
    shop: Sequence[Optional[str]] = (),
    hand: Sequence[str] = (),
    elementals_played: int = 0,
) -> BGLikeState:
    patch = game._patch
    st = game.initial_state()
    p0 = PlayerState(
        health=40,
        gold=gold,
        tavern_tier=tavern_tier,
        next_tier_up_cost=5 if tavern_tier < 6 else 0,
        board=_board(*board, patch=patch),
        shop=_shop(*shop, patch=patch),
        hand=_hand(*hand, patch=patch) if hand else _empty_hand(),
        phase=PlayerPhase.SHOP,
        shop_actions_used=0,
        elementals_played=elementals_played,
    )
    players = list(st.players)
    players[seat] = p0
    return replace(
        st,
        players=tuple(players),
        round_number=round_number,
        current_player_index=seat,
        shop_turn_order=(seat,),
        alive=(seat,),
        done=False,
        winner=None,
    )


def _buy(slot: int) -> StructAction:
    return StructAction(StructActionType.BUY, (slot,))


def _place(hand_slot: int) -> StructAction:
    return StructAction(StructActionType.PLACE, (hand_slot,))


def build_probes() -> List[ProbeCase]:
    return [
        ProbeCase(
            name="buy_sellemental_with_party_on_board",
            description="Party on board → buy Sellemental over Alleycat (tribe stack)",
            good=_buy(0),
            bad=_buy(1),
            setup=lambda g, s: _base_state(
                g,
                s,
                round_number=3,
                tavern_tier=2,
                gold=10,
                board=(PARTY_ELEMENTAL,),
                shop=(SELLEMENTAL, ALLEYCAT, None),
            ),
        ),
        ProbeCase(
            name="place_sellemental_triggers_party",
            description="Party on board → play Sellemental from hand over Alleycat",
            good=_place(0),
            bad=_place(1),
            setup=lambda g, s: _base_state(
                g,
                s,
                round_number=3,
                tavern_tier=2,
                gold=3,
                board=(PARTY_ELEMENTAL,),
                hand=(SELLEMENTAL, ALLEYCAT),
            ),
        ),
        ProbeCase(
            name="buy_arcane_assistant_with_two_elementals",
            description="Two elementals on board → buy Arcane Assistant over Piloted Shredder",
            good=_buy(0),
            bad=_buy(1),
            setup=lambda g, s: _base_state(
                g,
                s,
                round_number=5,
                tavern_tier=3,
                gold=10,
                board=(PARTY_ELEMENTAL, SELLEMENTAL),
                shop=(ARCANE_ASSISTANT, PILOTED_SHREDDER, None),
            ),
        ),
        ProbeCase(
            name="buy_party_over_neutral_empty_board",
            description="Empty board → buy Party Elemental over Alleycat (start tribe)",
            good=_buy(0),
            bad=_buy(1),
            setup=lambda g, s: _base_state(
                g,
                s,
                round_number=3,
                tavern_tier=2,
                gold=10,
                board=(),
                shop=(PARTY_ELEMENTAL, ALLEYCAT, None),
            ),
        ),
        ProbeCase(
            name="place_sellemental_grows_molten_rock",
            description="Molten Rock on board → play Sellemental over Alleycat (+HP synergy)",
            good=_place(0),
            bad=_place(1),
            setup=lambda g, s: _base_state(
                g,
                s,
                round_number=4,
                tavern_tier=2,
                gold=3,
                board=(MOLTEN_ROCK,),
                hand=(SELLEMENTAL, ALLEYCAT),
            ),
        ),
        ProbeCase(
            name="buy_molten_rock_over_alleycat_with_party",
            description="Party on board → buy Molten Rock over Alleycat",
            good=_buy(0),
            bad=_buy(1),
            setup=lambda g, s: _base_state(
                g,
                s,
                round_number=4,
                tavern_tier=2,
                gold=10,
                board=(PARTY_ELEMENTAL,),
                shop=(MOLTEN_ROCK, ALLEYCAT, None),
            ),
        ),
        ProbeCase(
            name="sell_alleycat_not_party",
            description=(
                "Party + Alleycat on board, Piloted Shredder in shop, gold=1 "
                "→ sell Alleycat over Party (free slot / gold for shop)"
            ),
            good=StructAction(StructActionType.SELL, (1,)),  # board idx 1 = alleycat
            bad=StructAction(StructActionType.SELL, (0,)),  # board idx 0 = party
            setup=lambda g, s: _base_state(
                g,
                s,
                round_number=4,
                tavern_tier=3,
                gold=1,
                board=(PARTY_ELEMENTAL, ALLEYCAT),
                shop=(PILOTED_SHREDDER, None, None),
            ),
        ),
        ProbeCase(
            name="buy_refreshing_anomaly_early",
            description="Early shop → buy Refreshing Anomaly (free roll synergy) over Alleycat",
            good=_buy(0),
            bad=_buy(1),
            setup=lambda g, s: _base_state(
                g,
                s,
                round_number=2,
                tavern_tier=1,
                gold=gold_for_round(2),
                board=(),
                shop=(REFRESHING_ANOMALY, ALLEYCAT, None),
            ),
        ),
    ]


def action_probabilities(
    agent,
    env: BGLobbyEnv,
    seat: int,
) -> Dict[StructAction, float]:
    legal = env.legal_structured_actions_for_seat(seat)
    if not legal:
        return {}
    obs = env.obs_for_seat(seat)
    obs_t = torch.as_tensor(obs, dtype=torch.float32, device=agent.device).unsqueeze(0)
    with torch.no_grad():
        logits, mask, _value, _cache = agent.policy_net.policy_logits_and_value(
            obs_t, [legal], return_cache=True
        )
        logits = logits.masked_fill(~mask, float("-inf"))
        probs = F.softmax(logits, dim=-1)[0].detach().cpu().numpy()
    return {legal[i]: float(probs[i]) for i in range(len(legal))}


def run_probe(
    agent,
    game: BGLikeGame,
    case: ProbeCase,
    seat: int = 0,
) -> dict:
    st = case.setup(game, seat)
    configs = lobby_from_learned_seats((seat,), agent_by_seat={seat: agent})
    env = BGLobbyEnv(
        configs,
        learned_seats=(seat,),
        training_seats=(seat,),
        seed=0,
        patch_dir=PATCH_DIR,
        obs_kind=OBS_KIND_BGLIKE_V5,
    )
    env._state = st
    env._game = game
    probs = action_probabilities(agent, env, seat)
    p_good = probs.get(case.good, 0.0)
    p_bad = probs.get(case.bad, 0.0)
    legal_good = case.good in probs
    legal_bad = case.bad in probs
    margin = p_good - p_bad
    logit_ratio = math.log((p_good + 1e-12) / (p_bad + 1e-12))
    return {
        "name": case.name,
        "description": case.description,
        "p_good": p_good,
        "p_bad": p_bad,
        "margin": margin,
        "log_odds": logit_ratio,
        "legal_good": legal_good,
        "legal_bad": legal_bad,
        "pass": legal_good and legal_bad and p_good > p_bad,
        "good_action": str(case.good),
        "bad_action": str(case.bad),
        "n_legal": len(probs),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--checkpoint",
        type=Path,
        default=REPO_ROOT / "runs/bglike/dist_ppo_056/checkpoints/dist_bglike_ppo_v6_74257_2007441.pt",
    )
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--min-margin", type=float, default=0.0, help="required p_good - p_bad")
    args = ap.parse_args()

    agent = load_training_agent_checkpoint(args.checkpoint, device=args.device, seed=77)
    if hasattr(agent, "eval"):
        agent.eval()
    freeze_agent(agent)

    game = BGLikeGame(seed=0, patch_dir=PATCH_DIR, shop_full_tribes=True)
    probes = build_probes()
    results = [run_probe(agent, game, c) for c in probes]

    passed = sum(1 for r in results if r["pass"] and r["margin"] >= args.min_margin)
    print(f"\nElemental synergy probe: {args.checkpoint.name}")
    print(f"Passed {passed}/{len(results)} (margin >= {args.min_margin})\n")
    print(f"{'probe':<42s} {'p_good':>7s} {'p_bad':>7s} {'margin':>8s}  ok")
    for r in results:
        ok = "✓" if r["pass"] and r["margin"] >= args.min_margin else "✗"
        if not r["legal_good"] or not r["legal_bad"]:
            ok = "?"
        print(
            f"{r['name']:<42s} {r['p_good']:7.3f} {r['p_bad']:7.3f} {r['margin']:+8.3f}  {ok}"
        )
        if not r["legal_good"] or not r["legal_bad"]:
            print(f"    legal: good={r['legal_good']} bad={r['legal_bad']}  ({r['description']})")

    print("\nDetails:")
    for r in results:
        status = "PASS" if r["pass"] and r["margin"] >= args.min_margin else "FAIL"
        print(f"  [{status}] {r['name']}: {r['description']}")
        print(f"         good={r['good_action']}  bad={r['bad_action']}")

    if passed < len(results):
        sys.exit(1)


if __name__ == "__main__":
    main()
