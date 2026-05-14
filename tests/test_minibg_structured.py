"""MiniBG structured env + structured actor-critic tests."""

from __future__ import annotations

import numpy as np
import torch

from src.envs.minibg import MiniBGEnv
from src.envs.minibg.action_map import A_FINISH
from src.envs.minibg.actions import BOARD_SIZE
from src.envs.minibg.state import PlayerPhase
from src.envs.minibg.structured_actions import (
    StructAction,
    StructActionType,
    validate_board_perm,
)
from src.models.minibg_structured_ac import MiniBGStructuredActorCritic


def _shop_struct_to_env_int(a: StructAction) -> int:
    from src.envs.minibg.action_map import (
        A_BUY_BASE,
        A_DISCOVER_BASE,
        A_LEVEL_UP,
        A_MAGNET_BASE,
        A_PLACE_BASE,
        A_ROLL,
        A_SELL_BASE,
    )
    from src.envs.minibg.actions import BOARD_SIZE

    if a.type == StructActionType.ROLL:
        return int(A_ROLL)
    if a.type == StructActionType.LEVEL_UP:
        return int(A_LEVEL_UP)
    if a.type == StructActionType.BUY:
        return int(A_BUY_BASE + a.args[0])
    if a.type == StructActionType.SELL:
        return int(A_SELL_BASE + a.args[0])
    if a.type == StructActionType.PLACE:
        return int(A_PLACE_BASE + a.args[0])
    if a.type == StructActionType.MAGNET:
        return int(A_MAGNET_BASE + a.args[0] * BOARD_SIZE + a.args[1])
    if a.type == StructActionType.DISCOVER_PICK:
        return int(A_DISCOVER_BASE + a.args[0])
    raise ValueError(a)


def test_validate_board_perm_ok():
    validate_board_perm(tuple(range(BOARD_SIZE - 1, -1, -1)))


def test_structured_shop_moves_match_int_steps():
    env_s = MiniBGEnv(seed=9)
    env_i = MiniBGEnv(seed=9)
    env_s.reset()
    env_i.reset()
    for _ in range(3):
        ls = env_s.legal_structured_actions()
        li = env_i.get_legal_actions()
        pick = next(a for a in ls if a.type != StructActionType.COMPLETE_TURN)
        ai = _shop_struct_to_env_int(pick)
        assert ai in li
        env_s.step_structured(pick)
        env_i.step(ai)
        assert env_s.get_state_hash() == env_i.get_state_hash()


def test_structured_illegal_complete_turn_without_perm():
    env = MiniBGEnv(seed=1)
    env.reset()
    legal = env.legal_structured_actions()
    assert any(a.type == StructActionType.COMPLETE_TURN for a in legal)
    r = env.step_structured(StructAction(StructActionType.COMPLETE_TURN))
    assert r.info.get("invalid_action") is True


def test_structured_complete_turn_matches_finish_twice_in_order_phase():
    """Structured COMPLETE_TURN with perm matches shop FINISH + order FINISH for k=0."""
    env_s = MiniBGEnv(seed=42)
    env_i = MiniBGEnv(seed=42)
    env_s.reset()
    env_i.reset()

    perm = tuple(range(BOARD_SIZE))
    env_s.step_structured(
        StructAction(StructActionType.COMPLETE_TURN), board_perm=perm
    )
    env_i.step(A_FINISH)
    env_i.step(A_FINISH)
    assert env_s.get_state_hash() == env_i.get_state_hash()


def test_structured_order_phase_only_complete_turn():
    env = MiniBGEnv(seed=0)
    env.reset()
    from src.envs.minibg.actions import MAX_SHOP_ACTIONS

    for _ in range(MAX_SHOP_ACTIONS + 5):
        if env.done:
            break
        p = env.state.players[env.current_player()]
        if p.phase == PlayerPhase.ORDER:
            break
        legal = env.get_legal_actions()
        if not legal:
            break
        env.step(legal[0])

    if env.state.players[env.current_player()].phase != PlayerPhase.ORDER:
        return

    legal_s = env.legal_structured_actions()
    assert legal_s == [StructAction(StructActionType.COMPLETE_TURN)]
    h_before = env.get_state_hash()
    env.step_structured(
        StructAction(StructActionType.COMPLETE_TURN), board_perm=tuple(range(BOARD_SIZE))
    )
    assert env.get_state_hash() != h_before


def test_structured_actor_critic_forward_shapes():
    env = MiniBGEnv(seed=0)
    obs = env.reset()
    legal = [env.legal_structured_actions()]
    x = torch.from_numpy(obs).float().unsqueeze(0)
    m = MiniBGStructuredActorCritic(
        slot_hidden=8,
        trunk_hidden=32,
        state_dim=16,
        action_dim=12,
        order_hidden=16,
        order_pos_dim=8,
    )
    logits, mask, values = m.policy_logits_and_value(x, legal)
    assert logits.shape == (1, len(legal[0]))
    assert mask.shape == logits.shape
    assert values.shape == (1,)
    assert torch.isfinite(logits[mask]).all()


def test_structured_actor_critic_order_logprob_consistency():
    env = MiniBGEnv(seed=3)
    obs = env.reset()
    x = torch.from_numpy(obs).float().unsqueeze(0)
    m = MiniBGStructuredActorCritic(
        slot_hidden=8,
        trunk_hidden=32,
        state_dim=16,
        action_dim=12,
        order_hidden=16,
        order_pos_dim=8,
    )
    state_emb, cache = m.encode_state(x)
    E_own = cache["E_own"]
    g_full = cache["g_full"]
    occ = torch.tensor([[True] + [False] * (BOARD_SIZE - 1)], dtype=torch.bool)
    picked, lp_sample, _ = m.sample_board_order(state_emb, E_own, g_full, occ, deterministic=True)
    assert picked[0, 0].item() == 0
    lp_tf = m.order_logprob_given_sequence(state_emb, E_own, g_full, occ, picked[:, :1])
    assert torch.allclose(lp_sample, lp_tf)


def test_build_ppo_minibg_structured():
    from src.envs.minibg.obs import OBS_DIM
    from src.models.ppo_policy_factory import (
        PPO_NETWORK_MINIBG_STRUCTURED,
        build_ppo_actor_critic,
    )

    m = build_ppo_actor_critic(
        PPO_NETWORK_MINIBG_STRUCTURED, (OBS_DIM,), num_actions=1, slot_hidden_channels=4
    )
    assert isinstance(m, MiniBGStructuredActorCritic)


def test_structured_actor_critic_grad_flow():
    env = MiniBGEnv(seed=1)
    obs = env.reset()
    legal = [env.legal_structured_actions()]
    x = torch.from_numpy(obs).float().unsqueeze(0)
    m = MiniBGStructuredActorCritic(
        slot_hidden=8,
        trunk_hidden=32,
        state_dim=16,
        action_dim=12,
        order_hidden=16,
        order_pos_dim=8,
    )
    logits, mask, values = m.policy_logits_and_value(x, legal)
    probs = torch.softmax(logits.masked_fill(~mask, float("-inf")), dim=-1)
    loss = values.mean() + probs.sum() + m.null_entity_action.norm()
    state_emb, cache = m.encode_state(x)
    lp = m.order_logprob_given_sequence(
        state_emb,
        cache["E_own"],
        cache["g_full"],
        torch.ones(1, BOARD_SIZE, dtype=torch.bool),
        torch.tensor([[3, 2, 1, 0] + [-1] * (BOARD_SIZE - 4)], dtype=torch.long),
    )
    loss = loss + lp.mean()
    loss.backward()
    assert m.state_to_interact.weight.grad is not None
    assert m.action_to_interact.weight.grad is not None
    assert m.score_fc[0].weight.grad is not None
    assert m.score_fc[-1].weight.grad is not None
    assert m.order_gru.weight_hh.grad is not None
    assert m.order_init.weight.grad is not None


def test_order_logprob_teacher_padding_no_nan():
    """If picks truncate vs occupied (should not enter buffer after invariant); replay stays finite."""
    from src.envs.minibg.obs import OBS_DIM

    m = MiniBGStructuredActorCritic(
        slot_hidden=8,
        trunk_hidden=32,
        state_dim=16,
        action_dim=12,
        order_hidden=16,
        order_pos_dim=8,
    )
    x = torch.randn(1, OBS_DIM)
    state_emb, cache = m.encode_state(x)
    occ3 = torch.tensor(
        [[True, True, True] + [False] * (BOARD_SIZE - 3)], dtype=torch.bool
    )
    picks_short = torch.tensor(
        [[0, 1] + [-1] * (BOARD_SIZE - 2)], dtype=torch.long
    )
    lp = m.order_logprob_given_sequence(state_emb, cache["E_own"], cache["g_full"], occ3, picks_short)
    assert torch.isfinite(lp).all(), lp


def test_slot_pick_sequence_to_perm_shapes():
    from src.envs.minibg import slot_pick_sequence_to_perm

    assert slot_pick_sequence_to_perm([2, 0, 1], 3) == (2, 0, 1) + tuple(
        range(3, BOARD_SIZE)
    )
    assert slot_pick_sequence_to_perm([], 0) == tuple(range(BOARD_SIZE))


def test_structured_ppo_agent_rollout_update_smoke():
    from src.agents.ppo_structured_minibg_agent import (
        INFO_STRUCT_LEGAL,
        INFO_STRUCT_NEXT_LEGAL,
        MiniBGPPOStructuredAgent,
    )
    from src.envs.minibg import NUM_ENV_ACTIONS
    from src.envs.minibg.obs import OBS_DIM
    from src.training.trainer import Transition

    env = MiniBGEnv(seed=0)
    obs = env.reset()
    net = MiniBGStructuredActorCritic(
        slot_hidden=4,
        trunk_hidden=16,
        state_dim=8,
        action_dim=8,
        order_hidden=8,
        order_pos_dim=4,
    )
    agent = MiniBGPPOStructuredAgent(
        observation_shape=(OBS_DIM,),
        observation_type="vector",
        num_actions=int(NUM_ENV_ACTIONS),
        network=net,
        rollout_steps=2,
        minibatch_size=2,
        ppo_epochs=1,
        device="cpu",
    )

    steps = 0
    while len(agent.rollout_buffer) < 2 and steps < 40:
        o = np.asarray(obs, dtype=np.float32).copy()
        lg = env.legal_structured_actions()
        sa, perm, idx = agent.act_structured(o, lg, env)
        step = env.step_structured(sa, board_perm=perm)
        next_l = [] if step.done else env.legal_structured_actions()
        tr = Transition(
            obs=o,
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
        obs = step.obs
        steps += 1
        if step.done:
            obs = env.reset()

    assert len(agent.rollout_buffer) >= 2
    metrics = agent.update()
    assert "policy_loss" in metrics


def test_act_structured_eval_does_not_clear_rollout_cache():
    """Self-play drain uses training=False; learner rollout cache must survive until observe."""
    from src.agents.ppo_structured_minibg_agent import MiniBGPPOStructuredAgent
    from src.envs.minibg import NUM_ENV_ACTIONS
    from src.envs.minibg.obs import OBS_DIM

    env = MiniBGEnv(seed=1)
    obs = env.reset()
    net = MiniBGStructuredActorCritic(
        slot_hidden=4,
        trunk_hidden=16,
        state_dim=8,
        action_dim=8,
        order_hidden=8,
        order_pos_dim=4,
    )
    agent = MiniBGPPOStructuredAgent(
        observation_shape=(OBS_DIM,),
        observation_type="vector",
        num_actions=int(NUM_ENV_ACTIONS),
        network=net,
        rollout_steps=999,
        device="cpu",
    )
    agent.train()
    legal = env.legal_structured_actions()
    agent.act_structured(obs, legal, env)
    assert agent._cache is not None
    was = agent._cache["action_idx"]
    agent.eval()
    try:
        legal2 = env.legal_structured_actions()
        agent.act_structured(obs, legal2, env)
        assert agent._cache is not None
        assert agent._cache["action_idx"] == was
    finally:
        agent.train()


def test_masked_policy_entropy_backward_no_nan_grad():
    """Illegal slots get log_sm=-inf and p=0; p*log_sm is NaN unless masked before multiply (PPO entropy)."""
    import torch.nn.functional as F

    logits = torch.randn(2, 5, requires_grad=True)
    mask = torch.tensor([[True, True, False, False, False], [True] * 5])
    masked_logits = logits.masked_fill(~mask, float("-inf"))
    log_sm = F.log_softmax(masked_logits, dim=-1)
    p = log_sm.exp()
    log_sm_safe = log_sm.masked_fill(~mask, 0.0)
    p_safe = p.masked_fill(~mask, 0.0)
    ent_row = -(p_safe * log_sm_safe).sum(dim=-1)
    ent_row.mean().backward()
    assert torch.isfinite(logits.grad).all()
