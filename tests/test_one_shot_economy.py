"""One-shot economy levers must apply exactly once — not zero times, not twice.

Before ``copy_player_state`` carried every field, the per-action state copy
cleared these levers, so a discount usually vanished before it could be spent.
Now that they survive, the opposite failure becomes possible: a lever that is
applied but never cleared would discount every roll or every upgrade.

Both directions are checked here because each is silent. Under-applying looks
like the card doing nothing; over-applying looks like the agent being good at
economy.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.bg_catalog.patch_context import PatchContext
from src.bg_lobby.player import PlayerPhase, PlayerState
from src.bg_recruitment import economy

PATCH_DIR = "data/bgcore/19_6_0_74257"


@pytest.fixture(scope="module")
def patch():
    return PatchContext.load(__import__("pathlib").Path(PATCH_DIR))


def _player(**kw):
    base = dict(
        health=40, gold=10, tavern_tier=1,         board=[], shop=[None] * 3, hand=[None] * 10,
        phase=PlayerPhase.SHOP, shop_actions_used=0,
    )
    base.update(kw)
    return PlayerState(**base)


def _roll(p, patch):
    economy.roll_shop(
        p, None, rng=np.random.default_rng(0),
        shared_pool=None, patch=patch,
    )


def test_refreshing_anomaly_makes_exactly_one_roll_free(patch):
    """next_roll_cost_override is set by the battlecry and spent by one roll."""
    p = _player(next_roll_cost_override=0)
    assert economy.effective_roll_cost(p) == 0

    gold_before = p.gold
    _roll(p, patch)
    assert p.gold == gold_before, "the discounted roll should have been free"
    assert p.next_roll_cost_override is None, "the override outlived its roll"
    assert economy.effective_roll_cost(p) == economy.ROLL_COST


def test_deck_swabbie_discounts_exactly_one_upgrade(patch):
    """upgrade_cost_delta is applied by one level-up and cleared by it."""
    p = _player(gold=20, upgrade_cost_delta=-1)
    discounted = economy.effective_level_up_cost(p)
    assert discounted == p.next_tier_up_cost - 1

    gold_before = p.gold
    economy.level_up_tavern(p, None, rng=np.random.default_rng(0),
                            shared_pool=None, patch=patch)
    assert p.gold == gold_before - discounted
    assert p.upgrade_cost_delta == 0, "the discount outlived its upgrade"
    # The next upgrade is full price for the new tier.
    assert economy.effective_level_up_cost(p) == p.next_tier_up_cost


def test_free_roll_charges_are_spent_one_per_roll(patch):
    """Multi-charge levers decrement rather than clearing outright."""
    p = _player(free_roll_charges=2, next_roll_cost_override=0)
    gold_before = p.gold

    _roll(p, patch)
    assert p.free_roll_charges == 1
    assert p.next_roll_cost_override == 0, "second charge should still be armed"

    _roll(p, patch)
    assert p.free_roll_charges == 0
    assert p.next_roll_cost_override is None

    assert p.gold == gold_before, "both charged rolls should have been free"
    _roll(p, patch)
    assert p.gold == gold_before - economy.ROLL_COST


def test_freeze_lifts_at_the_start_of_the_next_turn():
    """A freeze protects one shop, then lets go.

    The kept minions stay on the counter, but the pin is released — the next
    roll must be able to clear them. The whole-shop flag was already lifted at
    turn start; the per-slot tuple was never lifted anywhere, so a frozen slot
    survived every roll for the rest of the game. The per-action state copy hid
    it by clearing the field constantly.
    """
    from src.agents.random_agent import RandomAgent
    from src.envs.bglike.lobby_env import BGLobbyEnv
    from src.envs.bglike.seat_config import lobby_from_learned_seats

    agents = {s: RandomAgent(seed=s) for s in range(8)}
    env = BGLobbyEnv(
        lobby_from_learned_seats(tuple(range(8)), agent_by_seat=agents),
        learned_seats=tuple(range(8)), training_seats=(0,), seed=4,
        patch_dir=PATCH_DIR,
    )
    env.reset(seed=4)

    def step():
        seat = env.current_seat()
        if not env._seat_can_act(seat):
            return False
        env.step_action(seat, int(agents[seat].act(
            env.obs_for_seat(seat), legal_mask=env.legal_mask_for_seat(seat))))
        return True

    while env.state.combat_round < 4 and step():
        pass
    for p in env.state.players:
        p.shop_frozen = (True,) * len(p.shop_frozen)
    start = env.state.combat_round
    while env.state.combat_round == start and step():
        pass

    assert env.state.combat_round > start, "never crossed a turn boundary"
    for seat, p in enumerate(env.state.players):
        assert not any(p.shop_frozen), f"seat {seat} still frozen after turn start"
