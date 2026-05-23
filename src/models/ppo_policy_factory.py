"""Construct actor-critic modules for PPO by network type.

Two-layer design:
  - ``build_ppo_actor_critic`` is the *fresh-agent* path. Accepts legacy positional
    kwargs (``slot_hidden_channels``, ``mlp_hidden_size``, …) coming from
    ``agent.params`` in YAML configs, translates to per-class constructor kwargs.
  - ``restore_ppo_actor_critic`` is the *checkpoint-reload* path. Uses the
    in-checkpoint ``ppo_network_type`` + ``ppo_network_kwargs`` blob — no name
    translation, just registry lookup + ``cls(**kwargs)``.

Adding a new architecture:
  1. Create the class in its own module (one file = one frozen architecture).
  2. Register it at the bottom of this file via ``register_ppo_network(...)``,
     with a canonical id and any aliases for back-compat.
  3. If the class needs game-specific defaults from ``build_ppo_actor_critic``,
     add a branch there. Otherwise the registry alone is enough for reload.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Tuple, Type

import torch.nn as nn

from .author_critic_network import ActorCriticCNN
from .minibg_slot_ac import MiniBGSlotActorCritic, _OBS_DIM
from .flat_mlp_ac import FlatMLPActorCritic
from .minibg_structured_ac import MiniBGStructuredActorCritic

# Canonical ``ppo_network_type`` ids. Keep stable: written into every checkpoint
# and read back at load time. Aliases (older names) are registered below.
PPO_NETWORK_ACTOR_CRITIC_CNN = "actor_critic_cnn"
PPO_NETWORK_MINIBG_SLOT = "minibg_slot"
PPO_NETWORK_MINIBG_STRUCTURED = "minibg_structured"
PPO_NETWORK_BGLIKE_STRUCTURED = "bglike_structured"
PPO_NETWORK_BGLIKE_STRUCTURED_V1 = "bglike_structured_v1"
PPO_NETWORK_BGLIKE_STRUCTURED_V2 = "bglike_structured_v2"
PPO_NETWORK_FLAT_MLP = "flat_mlp"


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

# A ``RestoreFn`` reconstructs a module from a checkpoint's
# ``ppo_network_kwargs`` dict. It receives ``(obs_shape, num_actions, kwargs)``
# so classes that need shape-derived constructor args (e.g. ActorCriticCNN's
# rows/cols when the checkpoint didn't store them) can still rebuild.
RestoreFn = Callable[[Tuple[int, ...], int, Dict[str, Any]], nn.Module]


@dataclass(frozen=True)
class _PPOModelSpec:
    canonical_name: str
    cls: Type[nn.Module]
    restore: RestoreFn
    aliases: Tuple[str, ...] = field(default_factory=tuple)


_REGISTRY: Dict[str, _PPOModelSpec] = {}


def register_ppo_network(
    canonical_name: str,
    cls: Type[nn.Module],
    *,
    restore: RestoreFn,
    aliases: Tuple[str, ...] = (),
) -> None:
    """Register a PPO actor-critic class under ``canonical_name`` + aliases.

    Adding an alias is the safe way to rename: old checkpoints with the old id
    keep loading, new checkpoints write the canonical name.
    """
    spec = _PPOModelSpec(
        canonical_name=canonical_name, cls=cls, restore=restore, aliases=tuple(aliases)
    )
    keys = (canonical_name,) + tuple(aliases)
    for k in keys:
        k_low = k.strip().lower()
        prev = _REGISTRY.get(k_low)
        if prev is not None and prev.canonical_name != canonical_name:
            raise ValueError(
                f"PPO registry conflict: id {k_low!r} already maps to "
                f"{prev.canonical_name!r}, cannot reuse for {canonical_name!r}"
            )
        _REGISTRY[k_low] = spec


def _lookup(network_type: str) -> _PPOModelSpec:
    nt = network_type.strip().lower()
    spec = _REGISTRY.get(nt)
    if spec is None:
        raise ValueError(
            f"Unknown PPO network type {network_type!r}. Registered ids: "
            f"{sorted(set(s.canonical_name for s in _REGISTRY.values()))}"
        )
    return spec


def registered_ppo_networks() -> Dict[str, Type[nn.Module]]:
    """Return ``{canonical_name: class}`` for introspection / debug."""
    return {s.canonical_name: s.cls for s in set(_REGISTRY.values())}


def ppo_network_type_for_save(network_type: str) -> str:
    """Canonicalize ``network_type`` for storage in checkpoints."""
    nt = network_type.strip().lower()
    # Common shorthand aliases used in configs that aren't full ids.
    if nt == "board_cnn":
        return PPO_NETWORK_ACTOR_CRITIC_CNN
    if nt in ("minibg_mlp", "mlp"):
        return PPO_NETWORK_FLAT_MLP
    spec = _REGISTRY.get(nt)
    if spec is None:
        return nt
    return spec.canonical_name


# ---------------------------------------------------------------------------
# Public factory APIs
# ---------------------------------------------------------------------------


def build_ppo_actor_critic(
    network_type: str,
    observation_shape: Tuple[int, ...],
    num_actions: int,
    *,
    slot_hidden_channels: int = 32,
    trunk_hidden_size: int = 256,
    region_conv2_kernel: int = 1,
    card_emb_dim: int = 16,
    mlp_hidden_size: int = 256,
    num_pool_indices: Optional[int] = None,
) -> nn.Module:
    """Build a fresh actor-critic from config-level kwargs.

    This is the path used by ``make_agent`` when a checkpoint is *not* being
    restored. Translates a small set of legacy kwarg names (``slot_hidden_channels``
    etc.) into each class's actual constructor parameters.
    """
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
            num_pool_indices=num_pool_indices,
        )
    if nt in (PPO_NETWORK_MINIBG_STRUCTURED, "minibg_structured_v1"):
        if len(observation_shape) != 1 or int(observation_shape[0]) != _OBS_DIM:
            raise ValueError(
                f"network_type {nt!r} requires observation_shape [{_OBS_DIM}]"
            )
        return MiniBGStructuredActorCritic(
            slot_hidden=int(slot_hidden_channels),
            trunk_hidden=int(trunk_hidden_size),
            region_conv2_kernel=int(region_conv2_kernel),
            card_emb_dim=int(card_emb_dim),
            obs_layout="minibg",
            num_pool_indices=num_pool_indices,
        )
    if nt in (PPO_NETWORK_BGLIKE_STRUCTURED, PPO_NETWORK_BGLIKE_STRUCTURED_V1):
        from src.envs.bglike.obs import OBS_DIM as _BG_OBS_DIM

        if len(observation_shape) != 1 or int(observation_shape[0]) != _BG_OBS_DIM:
            raise ValueError(
                f"network_type {nt!r} requires observation_shape [{_BG_OBS_DIM}]"
            )
        return MiniBGStructuredActorCritic(
            slot_hidden=int(slot_hidden_channels),
            trunk_hidden=int(trunk_hidden_size),
            region_conv2_kernel=int(region_conv2_kernel),
            card_emb_dim=int(card_emb_dim),
            obs_layout="bglike",
            num_pool_indices=num_pool_indices,
        )
    if nt == PPO_NETWORK_BGLIKE_STRUCTURED_V2:
        from src.envs.bglike.obs import OBS_DIM as _BG_OBS_DIM
        from .bglike_structured_v2 import BGLikeStructuredV2

        if len(observation_shape) != 1 or int(observation_shape[0]) != _BG_OBS_DIM:
            raise ValueError(
                f"network_type {nt!r} requires observation_shape [{_BG_OBS_DIM}]"
            )
        # V2 ignores trunk_hidden_size (no flatten trunk) and uses its own defaults
        # for state_dim / entity_attention_layers; pass only common knobs.
        return BGLikeStructuredV2(
            slot_hidden=int(slot_hidden_channels),
            region_conv2_kernel=int(region_conv2_kernel),
            card_emb_dim=int(card_emb_dim),
            obs_layout="bglike",
            num_pool_indices=num_pool_indices,
        )
    if nt in (PPO_NETWORK_FLAT_MLP, "minibg_mlp", "mlp"):
        if len(observation_shape) != 1:
            raise ValueError(
                f"network_type {network_type!r} requires observation_shape [D] (flat vector)"
            )
        return FlatMLPActorCritic(
            input_size=int(observation_shape[0]),
            num_actions=int(num_actions),
            hidden_size=int(mlp_hidden_size),
        )
    raise ValueError(
        f"Unknown PPO network_type {network_type!r}. "
        f"Registered: {sorted(set(s.canonical_name for s in _REGISTRY.values()))}"
    )


def restore_ppo_actor_critic(
    canonical_type: str,
    observation_shape: Tuple[int, ...],
    num_actions: int,
    kwargs_dict: Dict[str, Any],
) -> nn.Module:
    """Rebuild policy net from a checkpoint's ``ppo_network_type`` + ``ppo_network_kwargs``."""
    spec = _lookup(canonical_type)
    return spec.restore(observation_shape, num_actions, dict(kwargs_dict))


def default_ppo_network_kwargs(network_type: str, module: nn.Module) -> Dict[str, Any]:
    """Serializer kwargs for checkpoint reload (excluding num_actions / obs shape)."""
    if isinstance(module, FlatMLPActorCritic):
        return dict(module.get_constructor_kwargs())
    if isinstance(module, MiniBGSlotActorCritic):
        return {k: v for k, v in module.get_constructor_kwargs().items() if k != "num_actions"}
    if isinstance(module, MiniBGStructuredActorCritic):
        # V2 also subclasses nothing but exposes get_constructor_kwargs through Protocol;
        # handled by the generic branch below if isinstance(MiniBGStructuredActorCritic) fails.
        return dict(module.get_constructor_kwargs())
    if isinstance(module, ActorCriticCNN):
        return {
            "rows": int(module.rows),
            "cols": int(module.cols),
            "in_channels": int(module.in_channels),
        }
    # Generic fallback: any module that provides get_constructor_kwargs.
    if hasattr(module, "get_constructor_kwargs"):
        return dict(module.get_constructor_kwargs())
    _ = network_type
    return {}


# ---------------------------------------------------------------------------
# Restore-fn helpers + registrations
# ---------------------------------------------------------------------------


def _restore_actor_critic_cnn(
    obs_shape: Tuple[int, ...], num_actions: int, kw: Dict[str, Any]
) -> nn.Module:
    if not kw and len(obs_shape) == 3:
        ic, r, c = obs_shape
        kw = {"in_channels": int(ic), "rows": int(r), "cols": int(c)}
    return ActorCriticCNN(
        rows=int(kw["rows"]),
        cols=int(kw["cols"]),
        in_channels=int(kw["in_channels"]),
        num_actions=int(num_actions),
    )


def _default_num_pool_indices() -> int:
    from src.bg_catalog.patch_context import DEFAULT_PATCH_DIR, load_patch_context

    return load_patch_context(str(DEFAULT_PATCH_DIR)).num_pool_indices


def _restore_minibg_slot(
    obs_shape: Tuple[int, ...], num_actions: int, kw: Dict[str, Any]
) -> nn.Module:
    return MiniBGSlotActorCritic(
        num_actions=int(num_actions),
        slot_hidden=int(kw.get("slot_hidden", 32)),
        trunk_hidden=int(kw.get("trunk_hidden", 256)),
        region_conv2_kernel=int(kw.get("region_conv2_kernel", 1)),
        card_emb_dim=int(kw.get("card_emb_dim", 16)),
        num_pool_indices=kw.get("num_pool_indices") or _default_num_pool_indices(),
    )


def _restore_flat_mlp(
    obs_shape: Tuple[int, ...], num_actions: int, kw: Dict[str, Any]
) -> nn.Module:
    return FlatMLPActorCritic(
        input_size=int(obs_shape[0]),
        num_actions=int(num_actions),
        hidden_size=int(kw.get("hidden_size", 256)),
    )


def _restore_structured_v1(
    obs_shape: Tuple[int, ...], num_actions: int, kw: Dict[str, Any]
) -> nn.Module:
    # The same V1 class serves minibg and bglike observation layouts.
    # `obs_layout` is sometimes implied by the canonical_type at write-time —
    # restore-time defers to the stored kwarg, or to "minibg" as a last resort.
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
        entity_attention_layers=int(kw.get("entity_attention_layers", 0)),
        entity_attention_heads=int(kw.get("entity_attention_heads", 4)),
        entity_attention_ff_mult=int(kw.get("entity_attention_ff_mult", 2)),
        entity_attention_init_scale=float(kw.get("entity_attention_init_scale", 0.1)),
        use_global_entity_token=bool(kw.get("use_global_entity_token", True)),
        obs_layout=str(kw.get("obs_layout", "minibg")).strip().lower(),
        num_pool_indices=kw.get("num_pool_indices") or _default_num_pool_indices(),
    )


def _restore_structured_v1_bglike(
    obs_shape: Tuple[int, ...], num_actions: int, kw: Dict[str, Any]
) -> nn.Module:
    kw = dict(kw)
    kw["obs_layout"] = "bglike"
    return _restore_structured_v1(obs_shape, num_actions, kw)


def _restore_structured_v2(
    obs_shape: Tuple[int, ...], num_actions: int, kw: Dict[str, Any]
) -> nn.Module:
    from .bglike_structured_v2 import BGLikeStructuredV2

    kw = dict(kw)
    if kw.get("num_pool_indices") is None:
        kw["num_pool_indices"] = _default_num_pool_indices()
    return BGLikeStructuredV2(**kw)


register_ppo_network(
    PPO_NETWORK_ACTOR_CRITIC_CNN,
    ActorCriticCNN,
    restore=_restore_actor_critic_cnn,
)
register_ppo_network(
    PPO_NETWORK_MINIBG_SLOT,
    MiniBGSlotActorCritic,
    restore=_restore_minibg_slot,
)
register_ppo_network(
    PPO_NETWORK_FLAT_MLP,
    FlatMLPActorCritic,
    restore=_restore_flat_mlp,
)
register_ppo_network(
    PPO_NETWORK_MINIBG_STRUCTURED,
    MiniBGStructuredActorCritic,
    restore=_restore_structured_v1,
    aliases=("minibg_structured_v1",),
)
register_ppo_network(
    PPO_NETWORK_BGLIKE_STRUCTURED,
    MiniBGStructuredActorCritic,
    restore=_restore_structured_v1_bglike,
    aliases=(PPO_NETWORK_BGLIKE_STRUCTURED_V1,),
)


def _register_v2_lazy() -> None:
    """Import V2 at module-import time; isolated for clarity / circular-import safety."""
    from .bglike_structured_v2 import BGLikeStructuredV2

    register_ppo_network(
        PPO_NETWORK_BGLIKE_STRUCTURED_V2,
        BGLikeStructuredV2,
        restore=_restore_structured_v2,
    )


_register_v2_lazy()


__all__ = [
    "build_ppo_actor_critic",
    "default_ppo_network_kwargs",
    "ppo_network_type_for_save",
    "register_ppo_network",
    "registered_ppo_networks",
    "restore_ppo_actor_critic",
    "PPO_NETWORK_ACTOR_CRITIC_CNN",
    "PPO_NETWORK_MINIBG_SLOT",
    "PPO_NETWORK_MINIBG_STRUCTURED",
    "PPO_NETWORK_BGLIKE_STRUCTURED",
    "PPO_NETWORK_BGLIKE_STRUCTURED_V1",
    "PPO_NETWORK_BGLIKE_STRUCTURED_V2",
    "PPO_NETWORK_FLAT_MLP",
]
