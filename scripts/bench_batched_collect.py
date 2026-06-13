#!/usr/bin/env python3
"""Benchmark: rollout collection with cross-lobby batched CPU inference.

Runs M independent 8-seat lobbies; each tick gathers the one pending decision
per lobby, runs ONE batched policy forward (and one batched order-head forward
for the COMPLETE_TURN subset), applies actions, repeats. M=1 reproduces the
sequential per-decision path (training-like baseline). Buffer-equivalent
bookkeeping (obs/legal/action/logprob/value rows for learner seats 0-3) is done
in all modes so the compared work matches training collection.

Example:
    python scripts/bench_batched_collect.py --envs 1 --rows 3300
    python scripts/bench_batched_collect.py --envs 8 --rows 3300
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch

import src.envs  # noqa: F401
from src.bg_catalog.patch_context import PatchContext
from src.agents.random_agent import RandomAgent
from src.envs.bglike.lobby_env import BGLobbyEnv
from src.envs.bglike.seat_config import lobby_from_learned_seats
from src.envs.minibg.structured_actions import (
    StructActionType,
    slot_pick_sequence_to_perm,
)
from src.models.bglike_structured_v7 import BGLikeStructuredV7

LEARNER_SEATS = frozenset({0, 1, 2, 3})
NUM_IDENTITIES = 4
BOARD_SIZE = 7


def build_net(ctx: PatchContext) -> BGLikeStructuredV7:
    """083-config v7 architecture, fresh init."""
    return BGLikeStructuredV7(
        num_identities=NUM_IDENTITIES,
        ability_emb_dim=8,
        slot_hidden=64,
        state_dim=128,
        action_dim=64,
        region_conv2_kernel=1,
        card_emb_dim=16,
        entity_attention_layers=4,
        entity_attention_heads=4,
        entity_attention_ff_mult=4,
        entity_attention_init_scale=0.1,
        obs_layout="bglike",
        num_pool_indices=ctx.num_pool_indices,
        action_cross_attn_heads=4,
        action_cross_attn_ff_mult=2,
        action_cross_attn_init_scale=0.1,
    ).eval()


class LobbySlot:
    """One lobby + its per-episode identity one-hots and reset bookkeeping."""

    def __init__(self, idx: int, patch_dir: str, seed: int) -> None:
        cfgs = lobby_from_learned_seats(
            (0,), agent_by_seat={0: RandomAgent(seed=seed)}, seed=seed
        )
        self.env = BGLobbyEnv(
            cfgs,
            learned_seats=(0,),
            seed=seed,
            patch_dir=patch_dir,
            obs_kind="bglike_v5",
        )
        self.idx = idx
        self.games = 0
        self.seed = seed
        self.identity_by_seat: dict[int, int] = {}
        self.reset()

    def reset(self) -> None:
        self.env.reset(seed=self.seed + 7919 * self.games)
        self.games += 1
        perm = np.random.permutation(NUM_IDENTITIES)
        self.identity_by_seat = {s: int(perm[s % NUM_IDENTITIES]) for s in range(8)}

    def pending(self):
        """(obs_aug, legal, seat) for the current decision, or None if done."""
        env = self.env
        if env.lobby_done:
            self.reset()
        cur = env.current_seat()
        if not env._seat_can_act(cur):
            return None
        legal = env.legal_structured_actions_for_seat(cur)
        if not legal:
            return None
        obs = env.obs_for_seat(cur)
        onehot = np.zeros(NUM_IDENTITIES, dtype=np.float32)
        onehot[self.identity_by_seat[cur]] = 1.0
        return np.concatenate([obs.astype(np.float32), onehot]), legal, cur


def run(envs: int, rows_target: int, budget_s: float, patch_dir: str, seed: int):
    ctx = PatchContext.load(Path(patch_dir))
    net = build_net(ctx)
    slots = [LobbySlot(i, patch_dir, seed + 131 * i) for i in range(envs)]

    # buffer-equivalent storage (same fields the structured rollout buffer keeps)
    buf_obs, buf_legal, buf_act, buf_logp, buf_val = [], [], [], [], []

    n_dec = 0
    t0 = time.perf_counter()
    while len(buf_obs) < rows_target and time.perf_counter() - t0 < budget_s:
        batch = []
        for sl in slots:
            p = sl.pending()
            if p is None:
                # stalled lobby (shouldn't happen) — hard reset
                sl.reset()
                p = sl.pending()
                if p is None:
                    continue
            batch.append((sl, *p))
        if not batch:
            break

        obs_t = torch.from_numpy(np.stack([b[1] for b in batch]))
        legal_lists = [b[2] for b in batch]
        with torch.no_grad():
            logits, mask, values, cache = net.policy_logits_and_value(
                obs_t, legal_lists, return_cache=True
            )
            logits = logits.masked_fill(~mask, float("-inf"))
            dist = torch.distributions.Categorical(logits=logits)
            idx_t = dist.sample()
            logp_t = dist.log_prob(idx_t)

        # batched order head for the COMPLETE_TURN subset
        perms: dict[int, tuple] = {}
        complete_rows = [
            i
            for i, b in enumerate(batch)
            if legal_lists[i][int(idx_t[i])].type
            in (StructActionType.COMPLETE_TURN, StructActionType.COMPLETE_TURN_FREEZE_SHOP)
        ]
        if complete_rows:
            rows = torch.tensor(complete_rows, dtype=torch.long)
            occ = torch.zeros(len(complete_rows), BOARD_SIZE, dtype=torch.bool)
            for j, i in enumerate(complete_rows):
                sl, _, _, seat = batch[i]
                occ[j, : len(sl.env.state.players[seat].board)] = True
            with torch.no_grad():
                picked, order_logp, _ = net.sample_board_order(
                    cache["state_emb"][rows],
                    cache["E_own"][rows],
                    cache["g_full"][rows],
                    occ,
                )
            for j, i in enumerate(complete_rows):
                sl, _, _, seat = batch[i]
                k = len(sl.env.state.players[seat].board)
                seq = [int(x) for x in picked[j].numpy() if int(x) >= 0][:k]
                perms[i] = slot_pick_sequence_to_perm(seq, k, board_size=BOARD_SIZE)
                logp_t[i] = logp_t[i] + order_logp[j]

        for i, (sl, obs_aug, legal, seat) in enumerate(batch):
            a = int(idx_t[i])
            sl.env.step_structured_for_seat(seat, legal[a], board_perm=perms.get(i))
            n_dec += 1
            if seat in LEARNER_SEATS:
                buf_obs.append(obs_aug)
                buf_legal.append(legal)
                buf_act.append(a)
                buf_logp.append(float(logp_t[i]))
                buf_val.append(float(values[i]))

    dt = time.perf_counter() - t0
    return {
        "envs": envs,
        "decisions": n_dec,
        "learner_rows": len(buf_obs),
        "seconds": dt,
        "decisions_per_s": n_dec / dt,
        "rows_per_s": len(buf_obs) / dt,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--envs", type=int, default=8)
    ap.add_argument("--rows", type=int, default=3300, help="learner buffer rows to collect")
    ap.add_argument("--budget", type=float, default=110.0, help="wall-clock budget seconds")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--patch-dir", default="data/bgcore/19_6_0_74257")
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    np.random.seed(args.seed)
    r = run(args.envs, args.rows, args.budget, args.patch_dir, args.seed)
    print(
        f"envs={r['envs']:3d} threads={args.threads} | "
        f"{r['decisions']} decisions, {r['learner_rows']} learner rows in {r['seconds']:.1f}s | "
        f"{r['decisions_per_s']:.1f} dec/s, {r['rows_per_s']:.1f} rows/s"
    )


if __name__ == "__main__":
    main()
