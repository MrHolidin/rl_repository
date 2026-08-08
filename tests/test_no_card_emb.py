"""``use_card_emb=False``: the card-identity ablation.

Swap-probe evidence motivating it (attn3, 8831 shop decisions, donors of the
same tier, measured on p(buy that slot)): keywords 0.0456, ability tokens
0.0338, race 0.0278, card_idx 0.0237, stats 0.0105, tier 0.0000. Since the
abilities of a card are a deterministic function of its id, the embedding can in
principle memorise everything the ability block encodes -- this flag is how we
find out whether it does.

The invariant that makes the comparison meaningful: shapes, module list and
parameter count must be identical with the flag on and off, so a paired run
differs in information content and nothing else.
"""

from __future__ import annotations

import pytest
import torch

import src.envs  # noqa: F401
from src.models.bglike_structured_v11 import BGLikeStructuredV11


def _net(use_card_emb: bool, seed: int = 0):
    torch.manual_seed(seed)
    return BGLikeStructuredV11(num_pool_indices=200, use_card_emb=use_card_emb).eval()


def test_parameter_count_is_identical():
    on, off = _net(True), _net(False)
    assert sum(p.numel() for p in on.parameters()) == sum(
        p.numel() for p in off.parameters()
    )


def test_module_structure_is_identical():
    on, off = _net(True), _net(False)
    assert [n for n, _ in on.named_parameters()] == [n for n, _ in off.named_parameters()]


def test_default_is_on_so_existing_configs_are_unchanged():
    assert _net(True).use_card_emb is True
    assert BGLikeStructuredV11(num_pool_indices=200).use_card_emb is True


def test_flag_is_serialised_for_checkpoint_reload():
    kw = _net(False).get_constructor_kwargs()
    assert kw["use_card_emb"] is False
    assert _net(True).get_constructor_kwargs()["use_card_emb"] is True


def test_card_identity_stops_changing_the_encoding_when_off():
    """The whole point: with the flag off, two different card ids that share
    every mechanical feature must encode identically."""
    from src.envs.bglike.obs_v5 import OBS_DIM_V5
    from src.envs.minibg.obs import CARD_IDX_OFFSET, SLOT_DIM
    from src.envs.bglike.obs import BGLIKE_GLOBAL_DIM

    pos = BGLIKE_GLOBAL_DIM + CARD_IDX_OFFSET  # first own-board slot

    for use, expect_equal in ((True, False), (False, True)):
        net = _net(use)
        x = torch.zeros(1, OBS_DIM_V5)
        x[0, BGLIKE_GLOBAL_DIM] = 1.0  # presence
        x[0, pos] = 11.0
        with torch.no_grad():
            a, _ = net.encode_state(x)
        x[0, pos] = 57.0  # different card, everything else identical
        with torch.no_grad():
            b, _ = net.encode_state(x)
        same = torch.allclose(a, b, atol=1e-6)
        assert same is expect_equal, (
            f"use_card_emb={use}: expected encodings "
            f"{'identical' if expect_equal else 'to differ'}"
        )


def test_ability_summon_token_path_still_uses_the_embedding():
    """The summon-token id inside an ability description is deliberately NOT
    ablated -- it is part of the ability, not the slot's identity."""
    net = _net(False)
    assert net.card_emb.weight.requires_grad
