"""v7 reproduces v6 (identity off / zero-init) and warm-starts from a v6 checkpoint.

v7 conditions on a population *identity* supplied as a one-hot tail appended to
the observation. Step-1 contract:
  * a v6 ``state_dict`` loads into v7 with ``strict=False`` adding only the new
    ``identity_proj`` module;
  * a core-width obs (no tail) is bit-identical to v6;
  * an obs with the identity tail is *also* identical to v6 while
    ``identity_proj`` is zero-init;
  * once ``identity_proj`` is perturbed, distinct identities change the output
    — proving the conditioning is wired through the CLS token;
  * the PPO factory can build / serialize / restore v7.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from src.bg_catalog.patch_context import load_patch_context
from src.envs.bglike.obs_v5 import OBS_DIM_V5
from src.models.bglike_structured_v6 import BGLikeStructuredV6
from src.models.bglike_structured_v7 import BGLikeStructuredV7
from src.models.ppo_policy_factory import (
    PPO_NETWORK_BGLIKE_STRUCTURED_V7,
    build_ppo_actor_critic,
    default_ppo_network_kwargs,
    ppo_network_type_for_save,
    restore_ppo_actor_critic,
)

_PATCH = "data/bgcore/15_6_2_36393"
_IDENTITY_KEYS = {
    "identity_proj.weight",
    "identity_proj.bias",
    "identity_emb_proj.weight",
    "identity_emb_proj.bias",
    "identity_slot_gate.weight",
    "identity_slot_gate.bias",
}
_N_ID = 4


def _base_kwargs(ctx):
    return dict(
        slot_hidden=48,
        state_dim=128,
        action_dim=64,
        num_pool_indices=ctx.num_pool_indices,
    )


def _v6_v7_with_shared_weights(num_identities: int = _N_ID):
    ctx = load_patch_context(_PATCH)
    kw = _base_kwargs(ctx)
    v6 = BGLikeStructuredV6(**kw).eval()
    v7 = BGLikeStructuredV7(num_identities=num_identities, **kw).eval()
    missing, unexpected = v7.load_state_dict(v6.state_dict(), strict=False)
    return v6, v7, set(missing), set(unexpected)


def _augment(x_core: torch.Tensor, identity: torch.Tensor, n: int) -> torch.Tensor:
    onehot = F.one_hot(identity, num_classes=n).to(x_core.dtype)
    return torch.cat([x_core, onehot], dim=1)


def test_v6_checkpoint_warm_starts_v7():
    _v6, v7, missing, unexpected = _v6_v7_with_shared_weights()
    assert missing == _IDENTITY_KEYS
    assert unexpected == set()
    assert v7.obs_dim == OBS_DIM_V5 + _N_ID


def test_v7_core_obs_matches_v6():
    v6, v7, _m, _u = _v6_v7_with_shared_weights()
    x = torch.zeros(3, OBS_DIM_V5, dtype=torch.float32)
    with torch.no_grad():
        s6, _ = v6.encode_state(x)
        s7, _ = v7.encode_state(x)  # core width → exact v6 path
    assert torch.allclose(s6, s7, atol=0.0, rtol=0.0)


def test_v7_zero_init_identity_tail_is_noop():
    v6, v7, _m, _u = _v6_v7_with_shared_weights(num_identities=_N_ID)
    x_core = torch.zeros(_N_ID, OBS_DIM_V5, dtype=torch.float32)
    x_aug = _augment(x_core, torch.arange(_N_ID), _N_ID)
    with torch.no_grad():
        s6, _ = v6.encode_state(x_core)
        s7, _ = v7.encode_state(x_aug)
    # identity_proj is zero-init → the tail contributes nothing yet.
    assert torch.allclose(s6, s7, atol=0.0, rtol=0.0)


def test_v7_identity_changes_output_once_wired():
    _v6, v7, _m, _u = _v6_v7_with_shared_weights(num_identities=_N_ID)
    with torch.no_grad():
        torch.nn.init.normal_(v7.identity_proj.weight, std=0.5)
    x_core = torch.zeros(_N_ID, OBS_DIM_V5, dtype=torch.float32)
    x_aug = _augment(x_core, torch.arange(_N_ID), _N_ID)
    with torch.no_grad():
        s_core, _ = v7.encode_state(x_core)
        s_id, _ = v7.encode_state(x_aug)
    assert not torch.allclose(s_core, s_id)
    assert not torch.allclose(s_id[0], s_id[1])  # distinct identities differ


def test_v7_identity_receives_gradient():
    ctx = load_patch_context(_PATCH)
    v7 = BGLikeStructuredV7(num_identities=_N_ID, **_base_kwargs(ctx)).train()
    x_core = torch.zeros(_N_ID, OBS_DIM_V5, dtype=torch.float32)
    x_aug = _augment(x_core, torch.arange(_N_ID), _N_ID)
    state_emb, _ = v7.encode_state(x_aug)
    state_emb.sum().backward()
    # Gradient flows into the identity projection even from zero-init.
    assert v7.identity_proj.weight.grad is not None
    assert torch.count_nonzero(v7.identity_proj.weight.grad) > 0


def test_factory_build_serialize_restore_v7():
    ctx = load_patch_context(_PATCH)
    net = build_ppo_actor_critic(
        PPO_NETWORK_BGLIKE_STRUCTURED_V7,
        (OBS_DIM_V5,),
        num_actions=128,
        slot_hidden_channels=48,
        card_emb_dim=16,
        num_pool_indices=ctx.num_pool_indices,
    )
    assert isinstance(net, BGLikeStructuredV7)

    kwargs = default_ppo_network_kwargs(PPO_NETWORK_BGLIKE_STRUCTURED_V7, net)
    assert kwargs["num_identities"] == net.num_identities

    restored = restore_ppo_actor_critic(
        ppo_network_type_for_save(PPO_NETWORK_BGLIKE_STRUCTURED_V7),
        (OBS_DIM_V5,),
        128,
        kwargs,
    )
    assert isinstance(restored, BGLikeStructuredV7)
    assert restored.num_identities == net.num_identities
    restored.load_state_dict(net.state_dict(), strict=True)
