#!/usr/bin/env python3
"""Real-config collection benchmark: the trainer's actual worker loop, batched.

Baseline: one call to ``_collect_until_steps_structured`` (the literal worker
function: BGLikeAgentPerspectiveEnv, league opponent sampling with frozen
snapshots + scripted bots, real rollout buffer, segment closures, league
records) — timed to ``--rows`` buffer rows.

Batched: M threads each run the SAME function on their own env/agent/buffer,
all learner agents sharing one policy net behind a broker that batches
``policy_logits_and_value`` calls across threads (learner seats + sibling
seats). Frozen-snapshot opponents keep their own deepcopied nets and run
batch=1, exactly as in production. Order-head calls stay per-thread.

Example:
    python scripts/bench_real_collect_batched.py --mode baseline --rows 3300
    python scripts/bench_real_collect_batched.py --mode batched --threads-n 8 --rows 3300
"""

from __future__ import annotations

import argparse
import random as _random
import sys
import threading
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch

import src.envs  # noqa: F401
from src.bg_catalog.patch_context import PatchContext
from src.envs.bglike.actions import NUM_PLAYERS  # noqa: F401  (import side effects)
from src.envs.bglike.obs_v5 import OBS_DIM_V5
from src.models.bglike_structured_v7 import BGLikeStructuredV7
from src.training.selfplay.game_record import (
    build_scripted_slot_map,
    invert_scripted_slot_map,
)
from src.training.selfplay.league_config import LeagueSamplerConfig
from src.training.selfplay.league_sampler import LeagueSyncState
from src.training.distributed_trainer import (
    _collect_until_steps_structured,
    _state_dict_bytes,
)

PATCH_DIR = "data/bgcore/19_6_0_74257"
NUM_IDENTITIES = 4
SCRIPTED_DISTRIBUTION = {"t1_random": 1, "t_up_random": 1, "structured": 1, "random": 1}


def build_net(seed: int) -> BGLikeStructuredV7:
    torch.manual_seed(seed)
    ctx = PatchContext.load(Path(PATCH_DIR))
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
    )


def build_agent(net: BGLikeStructuredV7, seed: int):
    from src.agents.ppo_dvd_agent import PPODvDAgent
    from src.envs.bglike.action_map import NUM_ENV_ACTIONS

    return PPODvDAgent(
        observation_shape=(int(OBS_DIM_V5),),
        observation_type="vector",
        num_actions=int(NUM_ENV_ACTIONS),
        network=net,
        ppo_network_type="bglike_structured_v7",
        num_identities=NUM_IDENTITIES,
        diversity_coef=0.0,
        identity_seed=seed,
        rollout_steps=1 << 30,  # never trigger update() during the bench
        device="cpu",
        seed=seed,
    )


def make_mg(seed: int) -> dict:
    """game_params exactly as run_distributed builds them for the plain-v7 config."""
    return {
        "game_id": "bglike",
        "seed": seed,
        "use_structured": True,
        "battle_damage_shaping": 0.0,
        "num_current_seats": 4,
        "patch_dir": PATCH_DIR,
        "shop_excluded_count": 1,
        "obs_kind": "bglike_v5",
        "percent_high_game": 0.0,
        "dvd_network_type": "bglike_structured_v7",
        "dvd_num_identities": NUM_IDENTITIES,
        "dvd_diversity_coef": 0.0,
        "dvd_diversity_ema": 0.1,
        "dvd_identity_tribes": None,
        "dvd_reward_mode": "final",
        "dvd_sibling_fraction": 0.5,
    }


def make_league(n_frozen: int, seed: int):
    scripted_slot_ids = build_scripted_slot_map(SCRIPTED_DISTRIBUTION.keys())
    slot_id_to_key = invert_scripted_slot_map(scripted_slot_ids)
    frozen_pool = {}
    trueskill = {}
    win_rates = {}
    for i in range(n_frozen):
        snap = build_net(seed + 1000 + i)  # distinct weights per snapshot
        frozen_pool[100 + i] = _state_dict_bytes_from_net(snap)
        trueskill[100 + i] = (25.0, 8.333)
        win_rates[100 + i] = 0.5
    for sid in scripted_slot_ids.values():
        trueskill[sid] = (25.0, 8.333)
        win_rates[sid] = 0.5
    sync = LeagueSyncState(
        frozen_pool=frozen_pool,
        win_rates=win_rates,
        rating_kind="trueskill",
        trueskill=trueskill,
    )
    sampler = LeagueSamplerConfig(
        kind="trueskill_quality", current_self_fraction=0.0, past_self_fraction=0.4
    )
    return sync, sampler, scripted_slot_ids, slot_id_to_key


def _state_dict_bytes_from_net(net) -> bytes:
    class _Shim:
        policy_net = net

    return _state_dict_bytes(_Shim())


# ---------------------------------------------------------------------------
# Broker: batches B=1 policy_logits_and_value calls across collector threads.
# ---------------------------------------------------------------------------


class Broker:
    def __init__(self, net, max_wait_s: float = 0.004) -> None:
        self.net = net
        self.max_wait_s = float(max_wait_s)
        self.cond = threading.Condition()
        self.pending: list[dict] = []
        self.active = 0
        self.stop = False
        self.batch_sizes: list[int] = []
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def register(self) -> None:
        with self.cond:
            self.active += 1

    def unregister(self) -> None:
        with self.cond:
            self.active -= 1
            self.cond.notify_all()

    def submit(self, obs_t, legal_list):
        slot = {"obs": obs_t, "legal": legal_list, "ev": threading.Event(), "out": None}
        with self.cond:
            self.pending.append(slot)
            self.cond.notify_all()
        slot["ev"].wait()
        return slot["out"]

    def shutdown(self) -> None:
        with self.cond:
            self.stop = True
            self.cond.notify_all()
        self.thread.join(timeout=5)

    def _run(self) -> None:
        while True:
            with self.cond:
                while not self.pending and not self.stop:
                    self.cond.wait(timeout=0.05)
                if self.stop and not self.pending:
                    return
                deadline = time.monotonic() + self.max_wait_s
                while (
                    len(self.pending) < max(1, self.active)
                    and time.monotonic() < deadline
                    and not self.stop
                ):
                    self.cond.wait(timeout=max(0.0, deadline - time.monotonic()))
                batch = self.pending
                self.pending = []
            if not batch:
                continue
            self.batch_sizes.append(len(batch))
            obs = torch.cat([b["obs"] for b in batch], dim=0)
            legal = [b["legal"][0] for b in batch]
            with torch.no_grad():
                logits, mask, values, cache = self.net.policy_logits_and_value(
                    obs, legal, return_cache=True
                )
            for i, b in enumerate(batch):
                n = len(legal[i])
                b["out"] = (
                    logits[i : i + 1, :n],
                    mask[i : i + 1, :n],
                    values[i : i + 1],
                    {k: v[i : i + 1] for k, v in cache.items()},
                )
                b["ev"].set()


class NetProxy:
    """Routes B=1 ``policy_logits_and_value`` through the broker; everything
    else (order head, attributes, train/eval) goes straight to the real net."""

    def __init__(self, net, broker: Broker) -> None:
        object.__setattr__(self, "_net", net)
        object.__setattr__(self, "_broker", broker)

    def policy_logits_and_value(self, obs, legal_lists, *, return_cache=False):
        if obs.shape[0] != 1:
            return self._net.policy_logits_and_value(
                obs, legal_lists, return_cache=return_cache
            )
        logits, mask, values, cache = self._broker.submit(obs, legal_lists)
        if return_cache:
            return logits, mask, values, cache
        return logits, mask, values

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "_net"), name)


# ---------------------------------------------------------------------------


def run_one_collector(agent, ppo_opponent, *, rows: int, seed: int, thread_idx: int):
    mg = make_mg(seed)
    sync, sampler, scripted_slot_ids, slot_id_to_key = _LEAGUE
    return _collect_until_steps_structured(
        agent,
        ppo_opponent,
        min_steps=rows,
        mg=mg,
        current_sd=_CURRENT_SD,
        league_sync=sync,
        sampler=sampler,
        scripted_distribution=dict(SCRIPTED_DISTRIBUTION),
        scripted_slot_ids=scripted_slot_ids,
        slot_id_to_scripted_key=slot_id_to_key,
        game_rng=_random.Random(seed + 31 * thread_idx),
        seed=seed + 977 * thread_idx,
        round_idx=5,
        start_episode=0,
    )


_LEAGUE = None
_CURRENT_SD = None


def main() -> None:
    global _LEAGUE, _CURRENT_SD
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("baseline", "batched"), required=True)
    ap.add_argument("--rows", type=int, default=3300)
    ap.add_argument("--threads-n", type=int, default=8, help="collector threads (batched)")
    ap.add_argument("--frozen", type=int, default=3, help="frozen snapshots in league")
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--max-wait-ms", type=float, default=4.0)
    args = ap.parse_args()

    torch.set_num_threads(1)
    net = build_net(args.seed)
    _LEAGUE = make_league(args.frozen, args.seed)
    learner_for_sd = build_agent(net, args.seed)
    _CURRENT_SD = _state_dict_bytes_from_net(net)

    if args.mode == "baseline":
        agent = learner_for_sd
        agent.train()
        opp_template = build_agent(build_net(args.seed + 5), args.seed + 5)
        t0 = time.perf_counter()
        n_games, n_steps, buf, _ = run_one_collector(
            agent, opp_template, rows=args.rows, seed=args.seed, thread_idx=0
        )
        dt = time.perf_counter() - t0
        print(
            f"baseline: {n_steps} rows, {n_games} games in {dt:.1f}s "
            f"-> {n_steps/dt:.1f} rows/s"
        )
        return

    # --- batched mode ---
    broker = Broker(net, max_wait_s=args.max_wait_ms / 1e3)
    M = args.threads_n
    rows_per = (args.rows + M - 1) // M
    results = [None] * M
    errors: list[BaseException] = []

    def worker(i: int) -> None:
        try:
            agent = build_agent(net, args.seed + i)
            agent.policy_net = NetProxy(net, broker)
            agent.train()
            opp_template = build_agent(build_net(args.seed + 5), args.seed + 5 + i)
            broker.register()
            try:
                results[i] = run_one_collector(
                    agent, opp_template, rows=rows_per, seed=args.seed, thread_idx=i
                )
            finally:
                broker.unregister()
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(M)]
    t0 = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    dt = time.perf_counter() - t0
    broker.shutdown()
    if errors:
        raise errors[0]
    total_rows = sum(r[1] for r in results)
    total_games = sum(r[0] for r in results)
    bs = np.array(broker.batch_sizes) if broker.batch_sizes else np.array([0])
    print(
        f"batched(M={M}): {total_rows} rows, {total_games} games in {dt:.1f}s "
        f"-> {total_rows/dt:.1f} rows/s | broker batches: n={len(bs)}, "
        f"mean={bs.mean():.1f}, p50={np.percentile(bs, 50):.0f}, max={bs.max()}"
    )


if __name__ == "__main__":
    main()
