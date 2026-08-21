"""The action space grows by appending, and a policy trained before the append
still reads every action it learned.

Two halves. The ids a trained policy knows must not move — which is why
HERO_POWER sits *above* the env-only actions rather than between them and the
game ones. And the tensors indexed by those ids are one row short in an old
checkpoint, which is a copy into the first rows rather than a mismatch.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn

import src.envs.minibg  # noqa: F401  (breaks a circular import at collection)
from src.agents.checkpoint_compat import current_action_space_size, grow_appended_rows
from src.envs.bglike import action_map as bglike_map
from src.envs.bglike import actions as bglike_actions
from src.envs.minibg import action_map as minibg_map
from src.envs.minibg import actions as minibg_actions


# --------------------------------------------------------------------------- #
# The ids themselves
# --------------------------------------------------------------------------- #


def test_the_env_ids_a_trained_policy_knows_have_not_moved():
    """Literals on purpose: these are what checkpoints on disk were trained on,
    and a change here is a silent relabelling of what a policy learned."""
    assert bglike_map.A_SWAP_BOARD_0 == 107
    assert bglike_map.A_SWAP_BOARD_LAST == 112
    assert bglike_map.A_APPLY_EFFECT_SKIP == 113
    assert minibg_map.A_SWAP_BOARD_0 == 73
    assert minibg_map.A_SWAP_BOARD_LAST == 78
    assert minibg_map.A_APPLY_EFFECT_SKIP == 79


@pytest.mark.parametrize(
    "actions, amap",
    [(bglike_actions, bglike_map), (minibg_actions, minibg_map)],
)
def test_the_env_stacks_its_own_actions_on_the_game_band(actions, amap):
    assert amap.A_SWAP_BOARD_0 == actions.NUM_CORE_ACTIONS
    assert amap.A_APPLY_EFFECT_SKIP == amap.A_SWAP_BOARD_LAST + 1


@pytest.mark.parametrize(
    "actions, amap",
    [(bglike_actions, bglike_map), (minibg_actions, minibg_map)],
)
def test_hero_power_was_appended_above_everything_that_existed(actions, amap):
    """Appending it at the game enum's end would have been an append for the
    enum and a *shift* for every env id above it."""
    hero_power = int(actions.Action.HERO_POWER)
    assert hero_power == amap.A_APPLY_EFFECT_SKIP + 1
    assert hero_power == amap.NUM_ENV_ACTIONS - 1
    assert hero_power == actions.NUM_ACTIONS - 1


@pytest.mark.parametrize("actions", [bglike_actions, minibg_actions])
def test_every_game_action_below_the_band_is_contiguous(actions):
    below = sorted(int(a) for a in actions.Action if int(a) < actions.NUM_CORE_ACTIONS)
    assert below == list(range(actions.NUM_CORE_ACTIONS))
    # ...and the reserved band holds no game action at all.
    band = range(actions.NUM_CORE_ACTIONS, int(actions.Action.HERO_POWER))
    assert not [a for a in actions.Action if int(a) in band]


# --------------------------------------------------------------------------- #
# Loading a checkpoint from a narrower space
# --------------------------------------------------------------------------- #


class _Net(nn.Module):
    def __init__(self, types: int, actions: int) -> None:
        super().__init__()
        self.type_emb = nn.Embedding(types, 4)
        self.head = nn.Linear(4, actions)


def test_a_narrower_checkpoint_becomes_a_prefix_of_the_wider_net():
    old, new = _Net(types=11, actions=114), _Net(types=12, actions=115)
    with torch.no_grad():
        old.type_emb.weight.fill_(1.0)
        old.head.weight.fill_(2.0)
        old.head.bias.fill_(3.0)
    state = {k: v.clone() for k, v in old.state_dict().items()}
    fresh_row = new.type_emb.weight[11].detach().clone()

    grown = grow_appended_rows(new, state)
    assert {g.split()[0] for g in grown} == {
        "type_emb.weight",
        "head.weight",
        "head.bias",
    }
    new.load_state_dict(state)
    assert torch.equal(new.type_emb.weight[:11], torch.ones(11, 4))
    assert torch.equal(new.type_emb.weight[11], fresh_row)
    assert torch.equal(new.head.bias[:114], torch.full((114,), 3.0))


def test_a_tensor_that_differs_any_other_way_is_left_to_fail():
    """Only a row-prefix is a grown action space. A different width, or a
    shorter model, is a real mismatch and must still raise."""
    net = _Net(types=12, actions=115)
    wrong_width = {"type_emb.weight": torch.zeros(11, 8)}
    assert grow_appended_rows(net, wrong_width) == []
    shrinking = {"type_emb.weight": torch.zeros(13, 4)}
    assert grow_appended_rows(net, shrinking) == []
    with pytest.raises(RuntimeError):
        net.load_state_dict({**net.state_dict(), **wrong_width})


def test_a_checkpoint_is_grown_to_the_space_the_env_offers_today():
    assert current_action_space_size("bglike_structured_v11", 114) == (
        bglike_map.NUM_ENV_ACTIONS
    )
    assert current_action_space_size("minibg_structured", 80) == (
        minibg_map.NUM_ENV_ACTIONS
    )
    # Already wide enough, or wider: left alone.
    assert current_action_space_size("bglike_structured_v11", 200) == 200
    # Narrower than the game band is not this env's net at all — a test stub or
    # a toy env — and growing it would invent a policy it never had.
    assert current_action_space_size("bglike_structured_v11", 11) == 11


def test_the_classic_pool_offers_nothing_to_press():
    """Which is what makes the appended row unreachable, and an old policy on
    the patch it was trained for exactly the policy it was."""
    from pathlib import Path

    from src.bg_catalog.patch_context import PatchContext

    for package in ("data/bgcore/19_6_0_74257", "data/bgcore/15_6_2_36393"):
        ctx = PatchContext.load(Path(package))
        assert not [h for h in ctx.heroes.values() if h.has_power()], package
