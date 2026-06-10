#!/usr/bin/env python3
"""Critic lookahead probe: when card X is offered, compare V(s) vs V(after BUY X) vs V(after ROLL).

If V(buy Brann) - V(s) is ~0 or negative while V(buy <mech>) - V(s) is positive,
the critic itself never learned the enabler's value — the policy can't be
expected to buy it (advantage would be flat).
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch

import src.envs  # noqa: F401
from src.envs.bglike.actions import Action as GameAction
from src.envs.bglike.lobby_env import BGLobbyEnv
from src.envs.bglike.obs_v5 import build_observation_v5
from src.envs.bglike.seat_config import lobby_from_learned_seats
from src.envs.minibg.structured_actions import StructActionType

SPECIALS = {
    "Brann Bronzebeard",
    "Nomi, Kitchen Nightmare",
    "Lightfang Enforcer",
    "Mythrax the Unraveler",
    "Baron Rivendare",
    "Khadgar",
    "Mama Bear",
    "Mal'Ganis",
    "Annoy-o-Module",
    "Drakonid Enforcer",
    "Menagerie Jug",
    "Bolvar, Fireblood",
}


def load_agent(path: Path, *, device: str, seed: int):
    from src.agents.ppo_dvd_agent import PPODvDAgent

    return PPODvDAgent.load(str(path), device=device, seed=seed)


class _StateView:
    def __init__(self, lobby: BGLobbyEnv) -> None:
        self.state = lobby.state


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", type=Path)
    ap.add_argument("--games", type=int, default=5)
    ap.add_argument("--seed", type=int, default=777)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--patch-dir", default="data/bgcore/19_6_0_74257")
    ap.add_argument("--num-identities", type=int, default=4)
    ap.add_argument("--pin-identity", type=int, default=None)
    args = ap.parse_args()

    agent = load_agent(args.checkpoint, device=args.device, seed=args.seed)
    agent.eval()
    agent.training = False

    agents = {s: agent for s in range(8)}
    configs = lobby_from_learned_seats(tuple(range(8)), agent_by_seat=agents)
    env = BGLobbyEnv(
        configs,
        learned_seats=tuple(range(8)),
        training_seats=(0,),
        seed=args.seed,
        patch_dir=args.patch_dir,
        obs_kind="bglike_v5",
    )
    patch = env._game._patch

    def value_of_state(state, seat) -> float:
        obs = build_observation_v5(
            state,
            seat,
            env.last_battle_signed(seat),
            is_my_turn=True,
            patch=patch,
        )
        obs_aug = agent._augment_obs_np(obs, None)
        obs_t = torch.as_tensor(
            obs_aug, dtype=torch.float32, device=agent.device
        ).unsqueeze(0)
        with torch.no_grad():
            _, cache = agent.policy_net.encode_state(obs_t)
            v = agent.policy_net.critic(cache["trunk"]).reshape(())
        return float(v.item())

    # name -> list of (V_buy - V_s, V_buy - V_roll)
    deltas = defaultdict(list)

    for g in range(args.games):
        agent.set_episode_identity(
            args.pin_identity if args.pin_identity is not None else g % args.num_identities
        )
        env.reset(seed=args.seed + 5000 * g)
        steps = 0
        while not env.lobby_done and steps < 20000:
            steps += 1
            cur = env.current_seat()
            if not env._seat_can_act(cur):
                if env.state.done:
                    break
                raise RuntimeError("stall")
            obs = env.obs_for_seat(cur)
            legal = env.legal_structured_actions_for_seat(cur)

            player = env.state.players[cur]
            buy_slots = {
                a.args[0]
                for a in legal
                if a.type == StructActionType.BUY
            }
            can_roll = any(a.type == StructActionType.ROLL for a in legal)
            if buy_slots:
                v_s = value_of_state(env.state, cur)
                v_roll = None
                if can_roll:
                    s_roll = env._game.apply_action(env.state, int(GameAction.ROLL))
                    v_roll = value_of_state(s_roll, cur)
                for slot in buy_slots:
                    m = player.shop[slot]
                    if m is None or (m.name or m.card_id) not in SPECIALS:
                        continue
                    s_buy = env._game.apply_action(
                        env.state, int(GameAction.BUY_SLOT_0) + slot
                    )
                    v_buy = value_of_state(s_buy, cur)
                    deltas[m.name].append(
                        (v_buy - v_s, (v_buy - v_roll) if v_roll is not None else np.nan)
                    )

            struct_act, board_perm, _ = agent.act_structured(
                obs, legal, _StateView(env), deterministic=True
            )
            env.step_structured_for_seat(cur, struct_act, board_perm=board_perm)

    print("critic one-step lookahead when card offered (mean over offers):")
    print(f"{'card':34s} {'n':>4s} {'V(buy)-V(s)':>12s} {'V(buy)-V(roll)':>14s}")
    for name in sorted(deltas, key=lambda n: -len(deltas[n])):
        arr = np.array(deltas[name], dtype=np.float64)
        d1 = arr[:, 0]
        d2 = arr[:, 1]
        d2m = np.nanmean(d2) if np.any(~np.isnan(d2)) else float("nan")
        print(f"{name:34s} {len(d1):4d} {np.mean(d1):+12.4f} {d2m:+14.4f}")


if __name__ == "__main__":
    main()
