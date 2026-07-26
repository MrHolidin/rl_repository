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
