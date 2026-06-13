#!/usr/bin/env python3
"""Late-training collection benchmark: 4 learner seats + 4 DISTINCT frozen opponents.

Single-threaded multiplexer over M lobbies (the architecture that won in
bench_batched_collect; threads+broker lose to the GIL). Per lobby: seats 0-3 =
live learner net (stochastic, batched across lobbies), seats 4-7 = four
distinct frozen snapshots (argmax, own weights). Frozen forwards run B=1 by
default (strict "all opponents different"); --group-frozen 1 batches frozen
requests across lobbies by snapshot id (realistic: the league pool holds at
most --frozen-slots snapshots, so seats repeat across lobbies).

Baseline = --envs 1 (everything B=1, same seat composition).
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import defaultdict
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


def build_net(ctx: PatchContext, seed: int) -> BGLikeStructuredV7:
    torch.manual_seed(seed)
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
    def __init__(self, idx: int, patch_dir: str, seed: int, n_frozen_slots: int) -> None:
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
        self.n_frozen_slots = n_frozen_slots
        self.rng = np.random.default_rng(seed)
        self.identity_by_seat: dict[int, int] = {}
        self.frozen_slot_by_seat: dict[int, int] = {}
        self.reset()

    def reset(self) -> None:
        self.env.reset(seed=self.seed + 7919 * self.games)
        self.games += 1
        perm = self.rng.permutation(NUM_IDENTITIES)
        self.identity_by_seat = {s: int(perm[s % NUM_IDENTITIES]) for s in range(8)}
        # 4 DISTINCT frozen snapshots per lobby (drawn from the league pool)
        picks = self.rng.choice(self.n_frozen_slots, size=4, replace=False)
        self.frozen_slot_by_seat = {4 + j: int(picks[j]) for j in range(4)}

    def pending(self):
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


def act_and_step(net, items, *, stochastic: bool) -> None:
    """One batched forward over items = [(slot, obs_aug, legal, seat)]; apply."""
    obs_t = torch.from_numpy(np.stack([it[1] for it in items]))
    legal_lists = [it[2] for it in items]
    with torch.no_grad():
        logits, mask, values, cache = net.policy_logits_and_value(
            obs_t, legal_lists, return_cache=True
        )
        logits = logits.masked_fill(~mask, float("-inf"))
        if stochastic:
            idx_t = torch.distributions.Categorical(logits=logits).sample()
        else:
            idx_t = logits.argmax(dim=-1)

    perms: dict[int, tuple] = {}
    complete_rows = [
        i
        for i in range(len(items))
        if legal_lists[i][int(idx_t[i])].type
        in (StructActionType.COMPLETE_TURN, StructActionType.COMPLETE_TURN_FREEZE_SHOP)
    ]
    if complete_rows:
        rows = torch.tensor(complete_rows, dtype=torch.long)
        occ = torch.zeros(len(complete_rows), BOARD_SIZE, dtype=torch.bool)
        for j, i in enumerate(complete_rows):
            sl, _, _, seat = items[i]
            occ[j, : len(sl.env.state.players[seat].board)] = True
        with torch.no_grad():
            picked, _, _ = net.sample_board_order(
                cache["state_emb"][rows],
                cache["E_own"][rows],
                cache["g_full"][rows],
                occ,
                deterministic=not stochastic,
            )
        for j, i in enumerate(complete_rows):
            sl, _, _, seat = items[i]
            k = len(sl.env.state.players[seat].board)
            seq = [int(x) for x in picked[j].numpy() if int(x) >= 0][:k]
            perms[i] = slot_pick_sequence_to_perm(seq, k, board_size=BOARD_SIZE)

    for i, (sl, _obs, legal, seat) in enumerate(items):
        sl.env.step_structured_for_seat(seat, legal[int(idx_t[i])], board_perm=perms.get(i))


def run(args) -> None:
    ctx = PatchContext.load(Path(args.patch_dir))
    live_net = build_net(ctx, args.seed)
    frozen_nets = [build_net(ctx, args.seed + 100 + i) for i in range(args.frozen_slots)]
    np.random.seed(args.seed)
    slots = [
        LobbySlot(i, args.patch_dir, args.seed + 131 * i, args.frozen_slots)
        for i in range(args.envs)
    ]

    learner_rows = 0
    n_dec = 0
    n_frozen_dec = 0
    frozen_batch_sizes = []
    t0 = time.perf_counter()
    while learner_rows < args.rows and time.perf_counter() - t0 < args.budget:
        learner_items = []
        frozen_by_slot: dict[int, list] = defaultdict(list)
        for sl in slots:
            p = sl.pending()
            if p is None:
                sl.reset()
                p = sl.pending()
                if p is None:
                    continue
            obs_aug, legal, seat = p
            if seat in LEARNER_SEATS:
                learner_items.append((sl, obs_aug, legal, seat))
            else:
                frozen_by_slot[sl.frozen_slot_by_seat[seat]].append(
                    (sl, obs_aug, legal, seat)
                )
        if not learner_items and not frozen_by_slot:
            break

        if learner_items:
            act_and_step(live_net, learner_items, stochastic=True)
            learner_rows += len(learner_items)
            n_dec += len(learner_items)

        for slot_id, items in frozen_by_slot.items():
            if args.group_frozen:
                act_and_step(frozen_nets[slot_id], items, stochastic=False)
                frozen_batch_sizes.append(len(items))
            else:
                for it in items:
                    act_and_step(frozen_nets[slot_id], [it], stochastic=False)
                    frozen_batch_sizes.append(1)
            n_dec += len(items)
            n_frozen_dec += len(items)

    dt = time.perf_counter() - t0
    fb = np.array(frozen_batch_sizes) if frozen_batch_sizes else np.array([0])
    print(
        f"envs={args.envs:3d} group_frozen={args.group_frozen} | "
        f"{n_dec} decisions ({learner_rows} learner, {n_frozen_dec} frozen) in {dt:.1f}s | "
        f"{learner_rows/dt:.1f} learner rows/s, {n_dec/dt:.1f} dec/s | "
        f"frozen batch mean={fb.mean():.2f}"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--envs", type=int, default=8)
    ap.add_argument("--rows", type=int, default=3300)
    ap.add_argument("--budget", type=float, default=110.0)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--frozen-slots", type=int, default=15)
    ap.add_argument("--group-frozen", type=int, default=0)
    ap.add_argument("--seed", type=int, default=23)
    ap.add_argument("--patch-dir", default="data/bgcore/19_6_0_74257")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    run(args)


if __name__ == "__main__":
    main()
