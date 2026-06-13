"""v9 = v8 + economy-coloured action queries.

At zero-init the new ``action_econ_proj`` contributes nothing, so v9's policy
logits + value are bit-identical to v8 built from the same weights; gradient
still flows into the new projection on the first backward.
"""

from __future__ import annotations

import numpy as np
import torch

import src.envs  # noqa: F401  (break circular import)
from src.envs.bglike.obs_v5 import OBS_DIM_V5
from src.envs.minibg.structured_actions import StructAction, StructActionType
from src.models.bglike_structured_v8 import BGLikeStructuredV8
from src.models.bglike_structured_v9 import BGLikeStructuredV9

NUM_POOL = 160


def _kwargs():
    return dict(
        num_identities=4,
        ability_emb_dim=8,
        slot_hidden=64,
        state_dim=128,
        action_dim=64,
        region_conv2_kernel=1,
        card_emb_dim=16,
        entity_attention_layers=2,
        entity_attention_heads=4,
        entity_attention_ff_mult=2,
        entity_attention_init_scale=0.1,
        obs_layout="bglike",
        num_pool_indices=NUM_POOL,
        action_cross_attn_heads=4,
        action_cross_attn_ff_mult=2,
        action_cross_attn_init_scale=0.1,
    )


def _obs(batch=3, seed=0):
    rng = np.random.default_rng(seed)
    return torch.tensor(
        rng.standard_normal((batch, OBS_DIM_V5)).astype(np.float32)
    )


def _legal():
    return [
        StructAction(StructActionType.ROLL),
        StructAction(StructActionType.LEVEL_UP),
        StructAction(StructActionType.BUY, (0,)),
        StructAction(StructActionType.COMPLETE_TURN),
    ]


def test_v9_has_econ_proj_zero_init():
    net = BGLikeStructuredV9(**_kwargs())
    assert hasattr(net, "action_econ_proj")
    assert net._econ_dim == 25  # core(17) + rotation(8)
    assert torch.count_nonzero(net.action_econ_proj.weight) == 0
    assert torch.count_nonzero(net.action_econ_proj.bias) == 0


def test_v9_bit_exact_v8_at_init():
    kw = _kwargs()
    v9 = BGLikeStructuredV9(**kw).eval()
    v8 = BGLikeStructuredV8(**kw).eval()
    # Copy shared weights v9 -> v8 (v8 lacks action_econ_proj; everything else
    # is identical in name/shape). strict=False drops only the new key.
    missing, unexpected = v8.load_state_dict(v9.state_dict(), strict=False)
    assert unexpected == ["action_econ_proj.weight", "action_econ_proj.bias"]
    assert missing == []

    obs = _obs()
    legal = [_legal() for _ in range(obs.shape[0])]
    with torch.no_grad():
        l9, m9, v9v = v9.policy_logits_and_value(obs, legal)
        l8, m8, v8v = v8.policy_logits_and_value(obs, legal)
    assert torch.allclose(l9, l8, atol=1e-6, equal_nan=True)
    assert torch.allclose(v9v, v8v, atol=1e-6)
    # Distributional critic carried over from v8.
    assert hasattr(v9, "placement_logits") and hasattr(v9, "value_from_trunk")


def test_v9_econ_grad_flows():
    net = BGLikeStructuredV9(**_kwargs())
    obs = _obs(seed=1)
    legal = [_legal() for _ in range(obs.shape[0])]
    logits, mask, values = net.policy_logits_and_value(obs, legal)
    loss = logits.masked_fill(~mask, 0.0).sum() + values.sum()
    loss.backward()
    g = net.action_econ_proj.weight.grad
    assert g is not None
    assert torch.count_nonzero(g) > 0  # gradient reaches the new projection


def test_v9_econ_changes_logits_when_weights_nonzero():
    net = BGLikeStructuredV9(**_kwargs()).eval()
    obs = _obs(seed=2)
    legal = [_legal() for _ in range(obs.shape[0])]
    with torch.no_grad():
        base, mask, _ = net.policy_logits_and_value(obs, legal)
        torch.nn.init.normal_(net.action_econ_proj.weight, std=0.5)
        after, _, _ = net.policy_logits_and_value(obs, legal)
    # Non-zero econ projection must move the action logits.
    assert not torch.allclose(base, after, atol=1e-4)
