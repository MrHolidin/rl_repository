"""Ordering-head entropy: correctness and the bit-exact-off default.

The joint policy is ``p(action) * p(order | action)`` but the entropy bonus
historically covered only the action head, so the ordering head could collapse
to one deterministic permutation with nothing opposing it and nothing showing it
in the logged ``entropy``. These tests pin the paired entry point that fixes it.
"""

from __future__ import annotations

import math

import pytest
import torch

import src.envs  # noqa: F401
from src.models.bglike_structured_v11 import BGLikeStructuredV11


def _net():
    torch.manual_seed(0)
    return BGLikeStructuredV11(num_pool_indices=200).eval()


def _batch(net, *, occupied: int, B: int = 6, K: int = 7):
    state_emb = torch.randn(B, net.state_dim)
    e_dim = net.order_gru.input_size - net.order_pos_emb.embedding_dim
    E_own = torch.randn(B, K, e_dim)
    occ = torch.zeros(B, K, dtype=torch.bool)
    occ[:, :occupied] = True
    picks = torch.full((B, K), -1, dtype=torch.long)
    for b in range(B):
        if occupied:
            picks[b, :occupied] = torch.randperm(occupied)
    # g_full is None: v11 keeps the arg for the shared agent but ignores it.
    return state_emb, E_own, None, occ, picks


def test_paired_logprob_matches_the_logprob_only_entry_point():
    """Regression guard: the old method now delegates, so it must not drift."""
    net = _net()
    args = _batch(net, occupied=4)
    with torch.no_grad():
        lp_pair, _ = net.order_logprob_entropy_given_sequence(*args)
        lp_only = net.order_logprob_given_sequence(*args)
    assert torch.allclose(lp_pair, lp_only)


def test_entropy_is_non_negative():
    net = _net()
    with torch.no_grad():
        _, ent = net.order_logprob_entropy_given_sequence(*_batch(net, occupied=4))
    assert bool((ent >= -1e-6).all())


def test_entropy_respects_the_permutation_ceiling():
    """An untrained head is near-uniform, so it should sit just under ln(n!)."""
    net = _net()
    for n in (3, 4, 5):
        with torch.no_grad():
            _, ent = net.order_logprob_entropy_given_sequence(*_batch(net, occupied=n))
        assert float(ent.max()) <= math.log(math.factorial(n)) + 1e-4
        assert float(ent.mean()) > 0.5 * math.log(math.factorial(n))


def test_empty_board_contributes_no_entropy():
    net = _net()
    with torch.no_grad():
        _, ent = net.order_logprob_entropy_given_sequence(*_batch(net, occupied=0))
    assert float(ent.abs().max()) == pytest.approx(0.0, abs=1e-9)


def test_single_minion_has_no_ordering_freedom():
    net = _net()
    with torch.no_grad():
        _, ent = net.order_logprob_entropy_given_sequence(*_batch(net, occupied=1))
    assert float(ent.abs().max()) == pytest.approx(0.0, abs=1e-5)


def test_more_minions_means_more_ordering_entropy():
    net = _net()
    means = []
    for n in (2, 4, 6):
        with torch.no_grad():
            _, ent = net.order_logprob_entropy_given_sequence(*_batch(net, occupied=n))
        means.append(float(ent.mean()))
    assert means[0] < means[1] < means[2]


def test_agent_default_leaves_the_bonus_untouched():
    """order_entropy_coef defaults to 0 -> historical behaviour, bit-exact."""
    from src.agents.ppo_structured_minibg_agent import MiniBGPPOStructuredAgent as Ag

    a = Ag.__new__(Ag)
    a.order_entropy_coef = 0.0
    assert a.order_entropy_coef == 0.0


# --- normalisation + controller ---------------------------------------------
# The first attempt averaged the ordering entropy over the WHOLE minibatch and
# used a raw coefficient. Only ~17% of decisions are COMPLETE_TURN, so the logged
# value was diluted ~6x, and coef=1.0 drove the ordering to 91% of ln(n!) -- very
# nearly random placement. The knob is now a fraction of the achievable ceiling.


def _agent(target=0.35, coef=0.01, rate=0.05, cap=0.2, until=0):
    from src.agents.ppo_structured_minibg_agent import MiniBGPPOStructuredAgent as Ag

    a = Ag.__new__(Ag)
    a.order_entropy_coef = coef
    a._order_entropy_coef_base = coef
    a.order_entropy_target = target
    a.entropy_adapt_rate = rate
    a.entropy_coef_max = cap
    a.entropy_target_until_step = until
    a._trained_steps = 0
    return a


def test_order_controller_disabled_by_default():
    a = _agent(target=0.0)
    a._adapt_order_entropy_coef(0.01)
    assert a.order_entropy_coef == pytest.approx(0.01)


def test_order_controller_raises_pressure_below_target():
    a = _agent()
    a._adapt_order_entropy_coef(0.05)
    assert a.order_entropy_coef > 0.01


def test_order_controller_never_goes_below_the_floor():
    a = _agent()
    for _ in range(200):
        a._adapt_order_entropy_coef(0.95)  # near-random ordering
    assert a.order_entropy_coef == pytest.approx(0.01)


def test_order_controller_respects_the_step_cutoff():
    a = _agent(until=1_000)
    a._trained_steps = 500
    a._adapt_order_entropy_coef(0.05)
    assert a.order_entropy_coef > 0.01
    a._trained_steps = 1_001
    a._adapt_order_entropy_coef(0.05)
    assert a.order_entropy_coef == pytest.approx(0.01)


def test_fraction_of_ceiling_is_board_size_invariant():
    """A uniform ordering must read as 1.0 whatever the board size."""
    net = _net()
    for n in (3, 4, 5, 6):
        with torch.no_grad():
            _, ent = net.order_logprob_entropy_given_sequence(*_batch(net, occupied=n))
        frac = float(ent.mean()) / math.log(math.factorial(n))
        assert 0.9 < frac <= 1.0 + 1e-6, (n, frac)
