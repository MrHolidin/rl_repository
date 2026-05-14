"""Construct actor-critic modules for PPO by network type."""

from __future__ import annotations

from typing import Any, Dict, Tuple

import torch.nn as nn

from .author_critic_network import ActorCriticCNN
from .minibg_slot_ac import MiniBGSlotActorCritic, _OBS_DIM
from .minibg_structured_ac import MiniBGStructuredActorCritic

PPO_NETWORK_ACTOR_CRITIC_CNN = "actor_critic_cnn"
PPO_NETWORK_MINIBG_SLOT = "minibg_slot"
PPO_NETWORK_MINIBG_STRUCTURED = "minibg_structured"


def build_ppo_actor_critic(
    network_type: str,
    observation_shape: Tuple[int, ...],
    num_actions: int,
    *,
    slot_hidden_channels: int = 32,
    trunk_hidden_size: int = 256,
    region_conv2_kernel: int = 1,
    card_emb_dim: int = 16,
) -> nn.Module:
    nt = network_type.strip().lower()
    if nt in ("board_cnn", "actor_critic_cnn", PPO_NETWORK_ACTOR_CRITIC_CNN):
        if len(observation_shape) != 3:
            raise ValueError(
                f"board CNN PPO expects observation_shape (C,H,W), got {observation_shape!r}"
            )
        in_channels, rows, cols = observation_shape
        return ActorCriticCNN(
            rows=int(rows),
            cols=int(cols),
            in_channels=int(in_channels),
            num_actions=int(num_actions),
        )
    if nt == PPO_NETWORK_MINIBG_SLOT:
        if len(observation_shape) != 1 or int(observation_shape[0]) != _OBS_DIM:
            raise ValueError(
                f"network_type {PPO_NETWORK_MINIBG_SLOT!r} requires observation_shape [{_OBS_DIM}]"
            )
        return MiniBGSlotActorCritic(
            num_actions=int(num_actions),
            slot_hidden=int(slot_hidden_channels),
            trunk_hidden=int(trunk_hidden_size),
            region_conv2_kernel=int(region_conv2_kernel),
            card_emb_dim=int(card_emb_dim),
        )
    if nt == PPO_NETWORK_MINIBG_STRUCTURED:
        if len(observation_shape) != 1 or int(observation_shape[0]) != _OBS_DIM:
            raise ValueError(
                f"network_type {PPO_NETWORK_MINIBG_STRUCTURED!r} requires observation_shape [{_OBS_DIM}]"
            )
        return MiniBGStructuredActorCritic(
            slot_hidden=int(slot_hidden_channels),
            trunk_hidden=int(trunk_hidden_size),
            region_conv2_kernel=int(region_conv2_kernel),
            card_emb_dim=int(card_emb_dim),
        )
    raise ValueError(
        f"Unknown PPO network_type {network_type!r}. "
        f"Try {PPO_NETWORK_ACTOR_CRITIC_CNN!r}, 'board_cnn', {PPO_NETWORK_MINIBG_SLOT!r}, "
        f"or {PPO_NETWORK_MINIBG_STRUCTURED!r}."
    )


def restore_ppo_actor_critic(
    canonical_type: str,
    observation_shape: Tuple[int, ...],
    num_actions: int,
    kwargs_dict: Dict[str, Any],
) -> nn.Module:
    """Rebuild policy net from checkpoint ``ppo_network_type`` + ``ppo_network_kwargs``."""
    ct = canonical_type.strip().lower()
    kw = dict(kwargs_dict)
    if ct == PPO_NETWORK_ACTOR_CRITIC_CNN:
        if not kw and len(observation_shape) == 3:
            ic, r, c = observation_shape
            kw = {"in_channels": int(ic), "rows": int(r), "cols": int(c)}
        return ActorCriticCNN(
            rows=int(kw["rows"]),
            cols=int(kw["cols"]),
            in_channels=int(kw["in_channels"]),
            num_actions=int(num_actions),
        )
    if ct == PPO_NETWORK_MINIBG_SLOT:
        return MiniBGSlotActorCritic(
            num_actions=int(num_actions),
            slot_hidden=int(kw.get("slot_hidden", 32)),
            trunk_hidden=int(kw.get("trunk_hidden", 256)),
            region_conv2_kernel=int(kw.get("region_conv2_kernel", 1)),
            card_emb_dim=int(kw.get("card_emb_dim", 16)),
        )
    if ct == PPO_NETWORK_MINIBG_STRUCTURED:
        return MiniBGStructuredActorCritic(
            slot_hidden=int(kw.get("slot_hidden", 32)),
            trunk_hidden=int(kw.get("trunk_hidden", 256)),
            state_dim=int(kw.get("state_dim", 128)),
            action_dim=int(kw.get("action_dim", 64)),
            interaction_dim=int(kw.get("interaction_dim", 64)),
            order_hidden=int(kw.get("order_hidden", 64)),
            order_pos_dim=int(kw.get("order_pos_dim", 16)),
            score_hidden=int(kw.get("score_hidden", 128)),
            order_score_hidden=int(kw.get("order_score_hidden", 64)),
            critic_hidden=int(kw.get("critic_hidden", 128)),
            region_conv2_kernel=int(kw.get("region_conv2_kernel", 1)),
            card_emb_dim=int(kw.get("card_emb_dim", 16)),
        )
    raise ValueError(f"Unknown canonical PPO network type {canonical_type!r}")


def default_ppo_network_kwargs(network_type: str, module: nn.Module) -> Dict[str, Any]:
    """Serializer kwargs for checkpoint reload (excluding num_actions / obs shape)."""
    if isinstance(module, MiniBGSlotActorCritic):
        return {k: v for k, v in module.get_constructor_kwargs().items() if k != "num_actions"}
    if isinstance(module, MiniBGStructuredActorCritic):
        return dict(module.get_constructor_kwargs())
    if isinstance(module, ActorCriticCNN):
        return {
            "rows": int(module.rows),
            "cols": int(module.cols),
            "in_channels": int(module.in_channels),
        }
    _ = network_type
    return {}


def ppo_network_type_for_save(network_type: str) -> str:
    nt = network_type.strip().lower()
    if nt == "board_cnn":
        return PPO_NETWORK_ACTOR_CRITIC_CNN
    if nt == PPO_NETWORK_MINIBG_SLOT:
        return PPO_NETWORK_MINIBG_SLOT
    if nt == PPO_NETWORK_MINIBG_STRUCTURED:
        return PPO_NETWORK_MINIBG_STRUCTURED
    if nt == PPO_NETWORK_ACTOR_CRITIC_CNN:
        return PPO_NETWORK_ACTOR_CRITIC_CNN
    return nt


__all__ = [
    "build_ppo_actor_critic",
    "default_ppo_network_kwargs",
    "ppo_network_type_for_save",
    "restore_ppo_actor_critic",
    "PPO_NETWORK_ACTOR_CRITIC_CNN",
    "PPO_NETWORK_MINIBG_SLOT",
    "PPO_NETWORK_MINIBG_STRUCTURED",
]
