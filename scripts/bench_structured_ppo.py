"""Micro-benchmark: rollout act() and PPO update() for MiniBGStructuredActorCritic.

Runs on CPU (fastest local reproducer). On GPU expect a larger win because the original
code path was launching ~B*Lmax mini kernels per minibatch.
"""

from __future__ import annotations

import time

import numpy as np
import torch

from src.agents.ppo_structured_minibg_agent import (
    INFO_STRUCT_LEGAL,
    INFO_STRUCT_NEXT_LEGAL,
    MiniBGPPOStructuredAgent,
)
from src.envs.minibg import MiniBGEnv, NUM_ENV_ACTIONS
from src.envs.minibg.obs import OBS_DIM
from src.models.minibg_structured_ac import MiniBGStructuredActorCritic
from src.training.trainer import Transition


def build_agent(rollout: int, mb: int, epochs: int) -> MiniBGPPOStructuredAgent:
    net = MiniBGStructuredActorCritic(
        slot_hidden=16,
        trunk_hidden=256,
        state_dim=128,
        action_dim=64,
        order_hidden=64,
        order_pos_dim=16,
    )
    return MiniBGPPOStructuredAgent(
        observation_shape=(OBS_DIM,),
        observation_type="vector",
        num_actions=int(NUM_ENV_ACTIONS),
        network=net,
        rollout_steps=rollout,
        minibatch_size=mb,
        ppo_epochs=epochs,
        device="cpu",
        seed=0,
    )


def collect_rollout(agent: MiniBGPPOStructuredAgent, env: MiniBGEnv, n: int) -> float:
    obs = env.reset()
    t0 = time.perf_counter()
    steps = 0
    while steps < n:
        lg = env.legal_structured_actions()
        sa, perm, idx = agent.act_structured(np.asarray(obs, dtype=np.float32), lg, env)
        step = env.step_structured(sa, board_perm=perm)
        next_l = [] if step.done else env.legal_structured_actions()
        tr = Transition(
            obs=np.asarray(obs, dtype=np.float32),
            action=idx,
            reward=float(step.reward),
            next_obs=np.asarray(step.obs, dtype=np.float32),
            terminated=step.terminated,
            truncated=step.truncated,
            info={**step.info, INFO_STRUCT_LEGAL: lg, INFO_STRUCT_NEXT_LEGAL: next_l},
            legal_mask=None,
            next_legal_mask=None,
        )
        agent.observe(tr)
        obs = env.reset() if step.done else step.obs
        steps += 1
    return time.perf_counter() - t0


def bench_action_emb_paths() -> None:
    """Compare vectorized action embedding vs the old per-row/per-action Python loop."""
    from src.envs.minibg.structured_actions import StructAction, StructActionType
    from src.models.minibg_structured_ac import (
        _ROLE_NONE,
        _build_action_tokens,
        role_for_struct,
    )

    net = MiniBGStructuredActorCritic(
        slot_hidden=16,
        trunk_hidden=256,
        state_dim=128,
        action_dim=64,
        order_hidden=64,
        order_pos_dim=16,
    )
    net.eval()

    env = MiniBGEnv(seed=0)
    obs_list = []
    legal_list = []
    o = env.reset()
    for _ in range(256):
        legal = env.legal_structured_actions()
        if not legal:
            o = env.reset()
            continue
        obs_list.append(np.asarray(o, dtype=np.float32))
        legal_list.append(legal)
        a = legal[0]
        if a.type in (
            StructActionType.COMPLETE_TURN,
            StructActionType.COMPLETE_TURN_FREEZE_SHOP,
        ):
            step = env.step_structured(a, board_perm=(0, 1, 2, 3))
        else:
            step = env.step_structured(a)
        o = env.reset() if step.done else step.obs

    obs_np = np.stack(obs_list[:256], axis=0)
    legal_list = legal_list[:256]
    B = obs_np.shape[0]
    Lmax = max(len(r) for r in legal_list)
    obs_t = torch.from_numpy(obs_np)

    state_emb, cache = net.encode_state(obs_t)

    # Old-style: per-row, per-action Python loop with `torch.full(...)` per action.
    def old_action_emb_one(cache_row, actions):
        out = []
        for a in actions:
            te = net.type_emb(torch.full((1,), int(a.type), dtype=torch.long))
            re = net.role_emb(torch.full((1,), role_for_struct(a), dtype=torch.long))
            if a.type in {
                StructActionType.COMPLETE_TURN,
                StructActionType.COMPLETE_TURN_FREEZE_SHOP,
                StructActionType.ROLL,
                StructActionType.LEVEL_UP,
            }:
                ent = net.null_entity_action.unsqueeze(0)
            else:
                if a.type == StructActionType.BUY:
                    sf = cache_row["E_shop"][:, a.args[0]]
                elif a.type == StructActionType.SELL:
                    sf = cache_row["E_own"][:, a.args[0]]
                else:
                    sf = cache_row["E_hand"][:, a.args[0]]
                ent = net.entity_to_action(sf)
            out.append(te + re + ent)
        return torch.stack(out, dim=0)

    def old_full(obs, legal):
        se, c = net.encode_state(obs)
        max_l = max(len(r) for r in legal)
        ae_pad = torch.zeros(len(legal), max_l, net.action_dim, dtype=obs.dtype)
        mask = torch.zeros(len(legal), max_l, dtype=torch.bool)
        for b in range(len(legal)):
            acts = legal[b]
            ae = old_action_emb_one({k: v[b : b + 1] for k, v in c.items()}, acts)
            L = ae.size(0)
            ae_pad[b, :L] = ae.squeeze(1)
            mask[b, :L] = True
        s_exp = se.unsqueeze(1).expand(-1, max_l, -1)
        g_exp = c["g_full"].unsqueeze(1).expand(-1, max_l, -1)
        s_int = torch.tanh(net.state_to_interact(se))
        a_int = torch.tanh(net.action_to_interact(ae_pad))
        interaction = s_int.unsqueeze(1) * a_int
        h_all = torch.cat([s_exp, ae_pad, interaction, g_exp], dim=-1)
        logits = net.score_fc(h_all.reshape(-1, h_all.size(-1))).squeeze(-1).view(len(legal), max_l)
        return logits.masked_fill(~mask, float("-inf"))

    n_iter = 4
    with torch.no_grad():
        t0 = time.perf_counter()
        for _ in range(n_iter):
            _ = old_full(obs_t, legal_list)
        t_old = (time.perf_counter() - t0) / n_iter

        t_np, r_np, k_np, s_np, m_np = _build_action_tokens(legal_list, Lmax)
        tids = torch.from_numpy(t_np)
        rids = torch.from_numpy(r_np)
        kids = torch.from_numpy(k_np)
        sids = torch.from_numpy(s_np)
        mids = torch.from_numpy(m_np)
        t0 = time.perf_counter()
        for _ in range(n_iter):
            _ = net.policy_logits_value_from_tokens(obs_t, tids, rids, kids, sids, mids)
        t_new = (time.perf_counter() - t0) / n_iter

    print(
        f"per-minibatch policy_logits  (B={B}, Lmax={Lmax}):\n"
        f"  old (per-row + per-action loop):  {t_old * 1000:7.1f} ms\n"
        f"  new (vectorized tokens):          {t_new * 1000:7.1f} ms\n"
        f"  speedup:                          {t_old / max(t_new, 1e-9):.1f}x"
    )


def main() -> None:
    torch.manual_seed(0)
    np.random.seed(0)

    rollout = 1024
    mb = 256
    epochs = 4

    agent = build_agent(rollout, mb, epochs)
    env = MiniBGEnv(seed=0)

    print(f"Collecting {rollout} rollout steps (new code)...")
    t_roll = collect_rollout(agent, env, rollout)
    print(f"  rollout: {t_roll * 1000:.0f} ms  ({rollout / t_roll:.0f} steps/s)")

    print(f"Running PPO update (epochs={epochs}, mb={mb})...")
    t1 = time.perf_counter()
    metrics = agent.update()
    t_upd = time.perf_counter() - t1
    print(f"  update: {t_upd * 1000:.0f} ms")

    print()
    bench_action_emb_paths()


if __name__ == "__main__":
    main()
