"""Distributed trainer BGLike wiring."""

from __future__ import annotations


import numpy as np
import pytest

from src.agents.ppo_structured_minibg_agent import MiniBGPPOStructuredAgent
from src.envs.bglike.action_map import NUM_ENV_ACTIONS
from src.envs.bglike.obs import OBS_DIM
from src.envs.minibg.structured_actions import StructActionType
from src.features.action_space import DiscreteActionSpace
from src.models.minibg_structured_ac import MiniBGStructuredActorCritic
from src.models.ppo_policy_factory import (
    PPO_NETWORK_BGLIKE_STRUCTURED,
    build_ppo_actor_critic,
    ppo_network_type_for_save,
)
from src.training.bg_network_policy import reject_flat_bg_network
from src.training.distributed_trainer import (
    StructuredMiniBGRolloutBuffer,
    _merge_buffers,
    _payload_to_buffer,
    _use_structured_collect,
)
from tests.conftest import NUM_POOL_INDICES


def test_use_structured_collect_bglike_true():
    assert _use_structured_collect({"game_id": "bglike", "use_structured": True})


def test_use_structured_collect_bglike_false():
    assert not _use_structured_collect({"game_id": "bglike", "use_structured": False})


def test_reject_flat_bg_network():
    with pytest.raises(ValueError, match="Flat PPO/DQN is deprecated"):
        reject_flat_bg_network("bglike", "minibg_mlp")


def test_bglike_structured_ac_accepts_obs_dim():
    net = build_ppo_actor_critic(
        PPO_NETWORK_BGLIKE_STRUCTURED,
        (OBS_DIM,),
        NUM_ENV_ACTIONS,
        slot_hidden_channels=16,
        num_pool_indices=NUM_POOL_INDICES,
    )
    x = np.zeros(OBS_DIM, dtype=np.float32)
    import torch

    state_emb, cache = net.encode_state(torch.from_numpy(x).unsqueeze(0))
    assert state_emb.shape == (1, net.state_dim)
    assert cache["E_own"].shape[1] == 7


def test_structured_rollout_buffer_roundtrip():
    mg = {"game_id": "bglike", "use_structured": True}
    buf = StructuredMiniBGRolloutBuffer()
    buf.add(
        obs=np.zeros(OBS_DIM, dtype=np.float32),
        legal_list=[],
        action_index=0,
        complete_turn=False,
        occupied_mask=np.zeros(7, dtype=bool),
        order_pick_row=np.full(7, -1, dtype=np.int64),
        reward=0.0,
        done=False,
        value=0.0,
        log_prob=-1.0,
        next_obs=np.zeros(OBS_DIM, dtype=np.float32),
        next_legal_list=[],
    )
    payload = __import__("pickle").dumps(buf)
    restored = _payload_to_buffer(payload, mg)
    merged = _merge_buffers([restored], mg)
    assert len(merged) == 1


def test_ppo_structured_agent_save_load_bglike(tmp_path):
    net = MiniBGStructuredActorCritic(
        slot_hidden=16,
        trunk_hidden=32,
        obs_layout="bglike",
        num_pool_indices=NUM_POOL_INDICES,
    )
    agent = MiniBGPPOStructuredAgent(
        observation_shape=(OBS_DIM,),
        observation_type="vector",
        num_actions=NUM_ENV_ACTIONS,
        network=net,
        ppo_network_type=ppo_network_type_for_save(PPO_NETWORK_BGLIKE_STRUCTURED),
        ppo_network_kwargs=net.get_constructor_kwargs(),
        rollout_steps=64,
    )
    path = tmp_path / "agent.pt"
    agent.save(str(path))
    loaded = MiniBGPPOStructuredAgent.load(str(path), device="cpu", seed=1)
    assert loaded.policy_net.board_size == 7


def test_bglike_structured_legal_has_no_swap():
    from src.envs.bglike.lobby_env import make_bglike_training_env
    from src.agents import RandomAgent

    env = make_bglike_training_env(current_seats=(0,), seed=1)
    opponents = {i: RandomAgent(seed=i + 1) for i in range(1, 8)}
    env.set_agents(RandomAgent(seed=0), opponents)
    env.reset(seed=1)
    legal = env.legal_structured_actions()
    assert legal, "expected structured legal actions in shop"
    assert not any(a.type.name.startswith("SWAP") for a in legal)
    assert any(a.type == StructActionType.COMPLETE_TURN for a in legal)


def test_bglike_struct_action_accepts_large_hand_slot():
    from src.envs.bglike.actions import BOARD_SIZE, HAND_SIZE
    from src.envs.minibg.structured_actions import StructAction, validate_struct_action

    action = StructAction(StructActionType.PLACE, (5,))
    validate_struct_action(action, hand_size=HAND_SIZE, board_size=BOARD_SIZE)

    import pytest

    with pytest.raises(ValueError, match="hand_slot out of range"):
        validate_struct_action(action, hand_size=5, board_size=BOARD_SIZE)
