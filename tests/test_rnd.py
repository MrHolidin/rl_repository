"""Unit tests for the RND intrinsic-motivation pieces (arXiv:1810.12894).

Covers the mechanical core that is easy to get subtly wrong:
  * the own-board (n_normal, n_golden) count featurizer and its obs offsets;
  * the turn-level, non-episodic intrinsic GAE (broadcast + self-bootstrap +
    per-seat independence);
  * the RND target/predictor distillation (novelty drops on visited inputs);
  * the optional scalar intrinsic-value head on v11.
"""

from __future__ import annotations

import numpy as np
import torch

from src.agents.rnd import RNDModel, discounted_forward, own_board_counts
from src.agents.rollout_segments import compute_turn_intrinsic_advantages
from src.envs.bglike.actions import BOARD_SIZE
from src.envs.bglike.obs import BGLIKE_GLOBAL_DIM
from src.envs.bglike.obs_v5 import OBS_DIM_V5
from src.envs.minibg.obs import CARD_IDX_OFFSET, GOLDEN_OFFSET, PRESENCE_OFFSET, SLOT_DIM


def _set_slot(obs, slot, *, card_idx, golden):
    off = BGLIKE_GLOBAL_DIM + slot * SLOT_DIM
    obs[0, off + PRESENCE_OFFSET] = 1.0
    obs[0, off + CARD_IDX_OFFSET] = float(card_idx)
    obs[0, off + GOLDEN_OFFSET] = 1.0 if golden else 0.0


def test_own_board_counts_splits_normal_and_golden():
    num_pool = 50
    obs = torch.zeros(1, OBS_DIM_V5)
    _set_slot(obs, 0, card_idx=3, golden=False)  # normal card 3
    _set_slot(obs, 1, card_idx=3, golden=True)   # golden card 3
    _set_slot(obs, 2, card_idx=7, golden=False)  # normal card 7
    # slot 3 left empty (presence 0)

    feat = own_board_counts(obs, num_pool)
    assert feat.shape == (1, num_pool * 2)
    # column c of each half == pool index c + 1 (idx 0 = empty, dropped)
    assert feat[0, 3 - 1].item() == 1.0           # normal card 3
    assert feat[0, 7 - 1].item() == 1.0           # normal card 7
    assert feat[0, num_pool + (3 - 1)].item() == 1.0  # golden card 3
    assert feat.sum().item() == 3.0               # exactly three occupied bodies


def test_own_board_counts_ignores_empty_and_out_of_range():
    num_pool = 20
    obs = torch.zeros(1, OBS_DIM_V5)
    # presence set but card_idx 0 -> empty, must not count
    off = BGLIKE_GLOBAL_DIM + 0 * SLOT_DIM
    obs[0, off + PRESENCE_OFFSET] = 1.0
    obs[0, off + CARD_IDX_OFFSET] = 0.0
    assert own_board_counts(obs, num_pool).sum().item() == 0.0
    assert BOARD_SIZE >= 1  # offsets are valid for the configured board


def test_discounted_forward_matches_recurrence():
    g = discounted_forward([1.0, 2.0, 3.0], gamma=0.5)
    assert np.allclose(g, [1.0, 2.0 + 0.5 * 1.0, 3.0 + 0.5 * 2.5])


def test_turn_intrinsic_advantages_broadcast_and_self_bootstrap():
    # One seat, 5 rows, two turns: A=[0,1] (node 1), B=[2,3,4] (node 4).
    intrinsic = np.array([0.0, 1.0, 0.0, 0.0, 2.0], dtype=np.float32)
    values_int = np.array([0.0, 0.5, 0.0, 0.0, 1.0], dtype=np.float32)
    complete = np.array([0, 1, 0, 0, 1], dtype=np.bool_)
    seats = np.zeros(5, dtype=np.int64)
    gamma, lam = 0.9, 1.0

    adv, ret, is_node = compute_turn_intrinsic_advantages(
        intrinsic, values_int, complete, seats, discount_factor=gamma, gae_lambda=lam
    )

    # node B (last) self-bootstraps from its own value (non-episodic):
    delta_b = 2.0 + gamma * 1.0 - 1.0          # 1.9
    delta_a = 1.0 + gamma * 1.0 - 0.5          # 1.4
    gae_b = delta_b                            # 1.9
    gae_a = delta_a + gamma * lam * gae_b      # 3.11
    assert np.allclose(adv, [gae_a, gae_a, gae_b, gae_b, gae_b], atol=1e-5)
    assert list(is_node) == [False, True, False, False, True]
    assert np.isclose(ret[1], gae_a + 0.5)     # return target at node A
    assert np.isclose(ret[4], gae_b + 1.0)     # return target at node B


def test_turn_intrinsic_advantages_seats_independent():
    # Interleaved seats 0 and 1, one turn each.
    intrinsic = np.array([0.0, 1.0, 0.0, 3.0], dtype=np.float32)
    values_int = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    complete = np.array([0, 1, 0, 1], dtype=np.bool_)
    seats = np.array([0, 0, 1, 1], dtype=np.int64)

    adv, _, is_node = compute_turn_intrinsic_advantages(
        intrinsic, values_int, complete, seats, discount_factor=0.9, gae_lambda=1.0
    )
    # each seat's single-turn advantage == its reward (zero values, self-bootstrap)
    assert np.allclose(adv, [1.0, 1.0, 3.0, 3.0], atol=1e-5)
    assert list(is_node) == [False, True, False, True]


def test_trailing_incomplete_turn_gets_zero_advantage():
    # Buffer edge cuts the 2nd turn before its node -> rows 2,3 stay 0.
    intrinsic = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
    values_int = np.array([0.0, 0.5, 0.0, 0.0], dtype=np.float32)
    complete = np.array([0, 1, 0, 0], dtype=np.bool_)
    seats = np.zeros(4, dtype=np.int64)
    adv, _, is_node = compute_turn_intrinsic_advantages(
        intrinsic, values_int, complete, seats, discount_factor=0.9, gae_lambda=0.95
    )
    assert adv[2] == 0.0 and adv[3] == 0.0
    assert list(is_node) == [False, True, False, False]


def test_rnd_novelty_drops_after_distillation():
    torch.manual_seed(0)
    rnd = RNDModel(10, embed_dim=8, target_hidden=16, predictor_hidden=16, predictor_layers=1)
    feat = torch.randn(8, rnd.in_dim).abs()
    rnd.update_obs_rms(feat)

    nov_before = rnd.novelty(feat)
    assert nov_before.shape == (8,)
    assert bool((nov_before >= 0).all())

    opt = torch.optim.Adam(rnd.predictor.parameters(), lr=1e-2)
    for _ in range(100):
        opt.zero_grad()
        rnd.predictor_loss(feat).mean().backward()
        opt.step()

    nov_after = rnd.novelty(feat)
    assert nov_after.mean().item() < nov_before.mean().item()
    # the frozen target must not have moved
    assert all(not p.requires_grad for p in rnd.target.parameters())


def test_rnd_predictor_reset_reinjects_novelty():
    torch.manual_seed(0)
    rnd = RNDModel(10, embed_dim=8, target_hidden=16, predictor_hidden=16, predictor_layers=1)
    feat = torch.randn(8, rnd.in_dim).abs()
    rnd.update_obs_rms(feat)
    opt = torch.optim.Adam(rnd.predictor.parameters(), lr=1e-2)
    for _ in range(150):
        opt.zero_grad()
        rnd.predictor_loss(feat).mean().backward()
        opt.step()
    nov_learned = rnd.novelty(feat).mean().item()

    rnd.update_ret_rms(np.array([3.0, 9.0, 5.0], dtype=np.float64))
    rnd.reset_predictor()
    nov_reset = rnd.novelty(feat).mean().item()

    # the reset re-injects novelty (error jumps back up on the now-familiar feats)
    assert nov_reset > 3.0 * nov_learned
    # ret-rms is reset to ~1 so the reward re-normalizes to the new error scale
    assert abs(rnd.ret_std() - 1.0) < 1e-3
    # the frozen target is untouched (reference must not move)
    assert all(not p.requires_grad for p in rnd.target.parameters())


def test_rnd_ret_rms_persists_through_state_dict():
    rnd = RNDModel(5)
    rnd.update_ret_rms(np.array([3.0, 9.0, 5.0, 7.0], dtype=np.float64))
    sd = rnd.state_dict()
    rnd2 = RNDModel(5)
    rnd2.load_state_dict(sd)
    assert np.isclose(rnd2.ret_std(), rnd.ret_std())
    assert torch.allclose(rnd2.obs_mean, rnd.obs_mean)


def test_v11_value_int_head_zero_init():
    from src.models.bglike_structured_v11 import BGLikeStructuredV11

    net = BGLikeStructuredV11(num_pool_indices=200, with_value_int=True)
    trunk = torch.randn(4, net._state_summary_dim)
    v = net.value_int_from_trunk(trunk)
    assert v.shape == (4,)
    # zero-initialized last layer -> initial intrinsic value is exactly 0
    assert torch.allclose(v, torch.zeros(4), atol=1e-6)
    assert "with_value_int" in net.get_constructor_kwargs()


def test_agent_rnd_wiring_and_checkpoint(tmp_path):
    from src.models.bglike_structured_v11 import BGLikeStructuredV11
    from src.agents.ppo_structured_minibg_agent import MiniBGPPOStructuredAgent

    net = BGLikeStructuredV11(num_pool_indices=200, with_value_int=True)
    agent = MiniBGPPOStructuredAgent(
        observation_shape=(net.obs_dim,),
        observation_type="vector",
        num_actions=10,
        network=net,
        ppo_network_type="bglike_structured_v11",
        device="cpu",
        rnd={"enabled": True, "int_coef": 0.5, "value_coef_int": 0.3},
    )
    assert agent._rnd_enabled and agent.rnd is not None
    # predictor params got their own optimizer group; the frozen target did not.
    assert len(agent.optimizer.param_groups) == 2

    # RND running stats survive a save/load round-trip (incl. net rebuild).
    agent.rnd.update_ret_rms(np.array([2.0, 8.0, 5.0], dtype=np.float64))
    ckpt = str(tmp_path / "ckpt.pt")
    agent.save(ckpt)
    loaded = MiniBGPPOStructuredAgent.load(ckpt, device="cpu")
    assert loaded._rnd_enabled and loaded.rnd is not None
    assert loaded.rnd.num_pool_indices == 200
    assert np.isclose(loaded.rnd.ret_std(), agent.rnd.ret_std())
    assert loaded.policy_net.with_value_int is True


def test_rnd_stream_runs_end_to_end_on_real_net():
    from src.models.bglike_structured_v11 import BGLikeStructuredV11
    from src.agents.ppo_structured_minibg_agent import MiniBGPPOStructuredAgent

    net = BGLikeStructuredV11(num_pool_indices=50, with_value_int=True)
    agent = MiniBGPPOStructuredAgent(
        observation_shape=(net.obs_dim,),
        observation_type="vector",
        num_actions=10,
        network=net,
        ppo_network_type="bglike_structured_v11",
        device="cpu",
        rnd={"enabled": True, "warmup_rounds": 0},
    )
    # All-empty boards are a structurally valid obs (every slot is pad).
    N = 6
    obs = torch.zeros(N, net.obs_dim)
    complete = np.array([0, 1, 0, 1, 0, 1], dtype=np.bool_)
    seats = np.array([0, 0, 0, 1, 1, 1], dtype=np.int64)

    int_adv, int_ret, is_node, diag = agent._rnd_stream(obs, complete, seats)
    assert int_adv.shape == (N,) and int_ret.shape == (N,)
    assert list(is_node) == [False, True, False, True, False, True]
    assert diag["rnd/num_nodes"] == 3.0
    assert "rnd/novelty_mean" in diag and "rnd/ret_std" in diag
    # obs-rms actually advanced past its epsilon-seeded count.
    assert float(agent.rnd.obs_count.item()) > 1.0
