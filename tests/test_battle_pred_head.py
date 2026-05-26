"""Tests for the auxiliary battle-prediction head."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.bg_catalog.patch_context import load_patch_context
from src.envs.bglike.obs import OBS_DIM, encode_board_minions
from src.envs.minibg.actions import BOARD_SIZE
from src.envs.minibg.obs import SLOT_DIM
from src.envs.minibg.structured_actions import StructAction, StructActionType
from src.models.bglike_structured_v2 import BGLikeStructuredV2
from src.models.bglike_structured_v3 import BGLikeStructuredV3
from src.models.structured_common import BattlePredictionHead

_PATCH = "data/bgcore/15_6_2_36393"


def _make_v3(*, battle_pred=None):
    ctx = load_patch_context(_PATCH)
    return BGLikeStructuredV3(
        slot_hidden=48,
        state_dim=64,
        action_dim=32,
        obs_layout="bglike",
        num_pool_indices=ctx.num_pool_indices,
        battle_pred_config=battle_pred,
    )


def test_head_disabled_by_default():
    net = _make_v3()
    assert not net._battle_pred_enabled
    assert net.battle_head is None
    with pytest.raises(RuntimeError, match="disabled"):
        net.predict_battle(
            torch.zeros(1, BOARD_SIZE, SLOT_DIM),
            torch.zeros(1, BOARD_SIZE, SLOT_DIM),
            torch.tensor([0.0]),
        )


def test_head_enabled_forward_returns_scalar():
    net = _make_v3(battle_pred={"enabled": True, "head_hidden": 32})
    assert net._battle_pred_enabled
    assert isinstance(net.battle_head, BattlePredictionHead)
    B = 4
    own = torch.randn(B, BOARD_SIZE, SLOT_DIM)
    opp = torch.randn(B, BOARD_SIZE, SLOT_DIM)
    af = torch.tensor([1.0, 0.0, 1.0, 0.0])
    pred = net.predict_battle(own, opp, af)
    assert pred.shape == (B,)


def test_head_gradient_flows_to_backbone_when_joint():
    net = _make_v3(battle_pred={"enabled": True, "head_hidden": 16})
    torch.manual_seed(0)
    own = torch.randn(3, BOARD_SIZE, SLOT_DIM)
    opp = torch.randn(3, BOARD_SIZE, SLOT_DIM)
    af = torch.tensor([1.0, 0.0, 1.0])
    pred = net.predict_battle(own, opp, af, detach_features=False)
    pred.sum().backward()
    # Backbone conv received gradient via shared encoder.
    assert net.region_conv1.weight.grad is not None
    assert net.region_conv1.weight.grad.norm().item() > 0.0
    # Head's own MLP also got gradient.
    assert net.battle_head.proj[0].weight.grad is not None
    assert net.battle_head.proj[0].weight.grad.norm().item() > 0.0


def test_head_gradient_blocked_when_detached():
    net = _make_v3(battle_pred={"enabled": True, "head_hidden": 16})
    torch.manual_seed(0)
    own = torch.randn(3, BOARD_SIZE, SLOT_DIM)
    opp = torch.randn(3, BOARD_SIZE, SLOT_DIM)
    af = torch.tensor([1.0, 0.0, 1.0])
    pred = net.predict_battle(own, opp, af, detach_features=True)
    pred.sum().backward()
    # Backbone conv didn't see any gradient.
    assert net.region_conv1.weight.grad is None or net.region_conv1.weight.grad.norm().item() == 0.0
    # Head's own MLP STILL got gradient (its weights aren't detached).
    assert net.battle_head.proj[0].weight.grad is not None
    assert net.battle_head.proj[0].weight.grad.norm().item() > 0.0


def test_head_does_not_consume_state_emb():
    """Sanity: predict_battle takes ONLY board+attack_first, no state_emb."""
    net = _make_v3(battle_pred={"enabled": True, "head_hidden": 16})
    import inspect

    sig = inspect.signature(net.predict_battle)
    assert "state_emb" not in sig.parameters
    # Head module also has no cls_from_state projection from state_emb.
    assert not hasattr(net.battle_head, "cls_from_state")
    assert hasattr(net.battle_head, "cls_token")


def test_head_handles_empty_boards():
    """All-zero boards (no minions on either side) shouldn't blow up."""
    net = _make_v3(battle_pred={"enabled": True, "head_hidden": 16})
    own = torch.zeros(2, BOARD_SIZE, SLOT_DIM)
    opp = torch.zeros(2, BOARD_SIZE, SLOT_DIM)
    af = torch.tensor([1.0, 0.0])
    pred = net.predict_battle(own, opp, af)
    assert pred.shape == (2,)
    assert torch.isfinite(pred).all()


def test_constructor_roundtrip_with_battle_pred():
    ctx = load_patch_context(_PATCH)
    net = _make_v3(battle_pred={"enabled": True, "head_hidden": 32, "aux_coef": 0.2})
    kw = net.get_constructor_kwargs()
    assert kw["battle_pred_config"] == {"enabled": True, "head_hidden": 32, "aux_coef": 0.2}
    # Reconstruct: should rebuild head identically.
    net2 = BGLikeStructuredV3(**kw)
    assert net2._battle_pred_enabled
    assert isinstance(net2.battle_head, BattlePredictionHead)


def test_buffer_backfill_routes_to_correct_finish_row():
    from src.agents.ppo_structured_minibg_agent import (
        MiniBGPPOStructuredAgent,
        StructuredMiniBGRolloutBuffer,
    )

    net = _make_v3(battle_pred={"enabled": True, "head_hidden": 16})
    agent = MiniBGPPOStructuredAgent(
        observation_shape=(OBS_DIM,),
        observation_type="vector",
        num_actions=64,
        network=net,
        ppo_network_type="bglike_structured_v3",
        ppo_network_kwargs=dict(net.get_constructor_kwargs()),
        device="cpu",
        rollout_steps=16,
    )
    assert agent._battle_pred_enabled

    # Manually fill buffer with mixed COMPLETE_TURN and other rows for 2 seats.
    buf = agent.rollout_buffer
    rng = np.random.default_rng(0)
    legal = [StructAction(type=StructActionType.ROLL)]

    def _add(seat, complete):
        buf.add(
            obs=rng.standard_normal(OBS_DIM, dtype=np.float32),
            legal_list=legal,
            action_index=0,
            complete_turn=complete,
            occupied_mask=np.zeros(BOARD_SIZE, dtype=bool),
            order_pick_row=np.full(BOARD_SIZE, -1, dtype=np.int64),
            reward=0.0,
            done=False,
            value=0.0,
            log_prob=0.0,
            next_obs=rng.standard_normal(OBS_DIM, dtype=np.float32),
            next_legal_list=legal,
            seat_id=seat,
        )

    # Simulate the real ordering: combat backfill fires AFTER each combat,
    # BEFORE subsequent FINISHes are appended. So at backfill time the only
    # unfilled FINISH for each seat is the most recent one.
    own_a = np.random.RandomState(1).randn(BOARD_SIZE, SLOT_DIM).astype(np.float32)
    own_b = np.random.RandomState(2).randn(BOARD_SIZE, SLOT_DIM).astype(np.float32)
    opp_a = np.random.RandomState(3).randn(BOARD_SIZE, SLOT_DIM).astype(np.float32)
    opp_b = np.random.RandomState(4).randn(BOARD_SIZE, SLOT_DIM).astype(np.float32)

    # Round 1: seat0 acts, FINISHes; seat1 acts, FINISHes. Combat resolves.
    _add(0, False)  # 0
    _add(0, True)   # 1 — first FINISH for seat 0
    _add(1, False)  # 2
    _add(1, True)   # 3 — first FINISH for seat 1
    agent._backfill_battle_data({
        0: {
            "own_board_obs": own_a,
            "opp_board_obs": opp_a,
            "attack_first": 1.0,
            "damage_signed_uncapped": 7.5,
        },
        1: {
            "own_board_obs": own_b,
            "opp_board_obs": opp_b,
            "attack_first": 0.0,
            "damage_signed_uncapped": -3.0,
        },
    })
    assert buf.battle_target_valid[1] is True
    assert buf.battle_target_valid[3] is True
    assert buf.battle_target[1] == pytest.approx(7.5)
    assert buf.battle_target[3] == pytest.approx(-3.0)
    np.testing.assert_array_equal(buf.own_board_obs[1], own_a)
    np.testing.assert_array_equal(buf.opp_board_obs[3], opp_b)
    assert buf.battle_target_valid[0] is False
    assert buf.battle_target_valid[2] is False

    # Round 2: seat0 acts, FINISHes again. Backfill for the new FINISH only.
    _add(0, False)  # 4
    _add(0, True)   # 5 — second FINISH for seat 0
    agent._backfill_battle_data({
        0: {
            "own_board_obs": own_a * 2,
            "opp_board_obs": opp_a * 2,
            "attack_first": 1.0,
            "damage_signed_uncapped": 12.0,
        },
    })
    assert buf.battle_target_valid[5] is True
    assert buf.battle_target[5] == pytest.approx(12.0)
    # Row 1 must still hold ORIGINAL values — backfill must not overwrite.
    assert buf.battle_target[1] == pytest.approx(7.5)


def test_buffer_field_lengths_stay_in_sync():
    """Even when ``predict_battle`` is disabled, placeholders are appended so
    field arrays match the obs-row count. Otherwise downstream stacking would
    drift."""
    from src.agents.ppo_structured_minibg_agent import StructuredMiniBGRolloutBuffer

    buf = StructuredMiniBGRolloutBuffer()
    legal = [StructAction(type=StructActionType.ROLL)]
    for _ in range(5):
        buf.add(
            obs=np.zeros(OBS_DIM, dtype=np.float32),
            legal_list=legal,
            action_index=0,
            complete_turn=False,
            occupied_mask=np.zeros(BOARD_SIZE, dtype=bool),
            order_pick_row=np.full(BOARD_SIZE, -1, dtype=np.int64),
            reward=0.0,
            done=False,
            value=0.0,
            log_prob=0.0,
            next_obs=np.zeros(OBS_DIM, dtype=np.float32),
            next_legal_list=legal,
            seat_id=0,
        )
    assert len(buf) == 5
    for fld in ("own_board_obs", "opp_board_obs", "attack_first", "battle_target", "battle_target_valid"):
        assert len(getattr(buf, fld)) == 5
    assert all(v is False for v in buf.battle_target_valid)
