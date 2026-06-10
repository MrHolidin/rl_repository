"""Step-3 DvD: board descriptor + behavioural repulsion bonus.

Covers the diversity machinery without a full rollout:
  * ``board_descriptor`` summarises a board (tribe mix / tier / fill / stats);
  * ``_diversity_bonus`` rewards an identity for differing from the others'
    running descriptors and folds the episode descriptor into its EMA;
  * with ``diversity_coef == 0`` no repulsion state is touched.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from src.agents import PPODvDAgent  # noqa: F401 (registers "ppo")
from src.bg_catalog.patch_context import load_patch_context
from src.bg_core.minion import Minion, Race
from src.envs.bglike.actions import NUM_ACTIONS
from src.envs.bglike.board_descriptor import BOARD_DESCRIPTOR_DIM, board_descriptor
from src.envs.bglike.obs_v5 import OBS_DIM_V5
from src.envs.minibg.obs import RACE_ONEHOT_DIM, _RACE_ORDER
from src.registry import make_agent

_PATCH = "data/bgcore/15_6_2_36393"
_N_ID = 4


def _m(race, atk=3, hp=4, tier=2, golden=False):
    label = race.name.lower() if race is not None else "none"
    return Minion(
        card_id=f"test_{label}",
        base_attack=atk,
        base_health=hp,
        tier=tier,
        race=race,
        is_golden=golden,
    )


def _state(board, tavern_tier=3):
    return SimpleNamespace(
        players=[SimpleNamespace(board=board, tavern_tier=tavern_tier)]
    )


# --------------------------------------------------------------------------- #
# board_descriptor
# --------------------------------------------------------------------------- #
def test_descriptor_dim_and_empty_board():
    v = board_descriptor(_state([], tavern_tier=2), 0)
    assert v.shape == (BOARD_DESCRIPTOR_DIM,)
    assert v[:RACE_ONEHOT_DIM].sum() == 0.0  # no minions → no tribe mass
    # tier still encoded; fill is zero.
    assert v[RACE_ONEHOT_DIM + 0] == 2.0 / 6.0
    assert v[RACE_ONEHOT_DIM + 1] == 0.0


def test_descriptor_pure_mech_board():
    board = [_m(Race.MECHANICAL) for _ in range(4)]
    v = board_descriptor(_state(board), 0, board_size=7)
    mech_idx = _RACE_ORDER.index(Race.MECHANICAL)
    assert v[mech_idx] == 1.0
    assert v[:RACE_ONEHOT_DIM].sum() == 1.0
    assert v[RACE_ONEHOT_DIM + 1] == 4.0 / 7.0  # fill fraction


def test_descriptor_distinguishes_tribes():
    mech = board_descriptor(_state([_m(Race.MECHANICAL)] * 5), 0)
    elem = board_descriptor(_state([_m(Race.ELEMENTAL)] * 5), 0)
    assert np.linalg.norm(mech - elem) > 0.0


# --------------------------------------------------------------------------- #
# repulsion bonus
# --------------------------------------------------------------------------- #
def _make_agent(diversity_coef=1.0, identity_tribes=None):
    ctx = load_patch_context(_PATCH)
    return make_agent(
        "ppo",
        network_type="bglike_structured_v7",
        num_identities=_N_ID,
        diversity_coef=diversity_coef,
        identity_tribes=identity_tribes,
        num_actions=NUM_ACTIONS,
        observation_shape=(OBS_DIM_V5,),
        observation_type="vector",
        num_pool_indices=ctx.num_pool_indices,
        device="cpu",
    )


def test_assigned_bonus_is_fraction_of_assigned_tribe():
    # identity 0 → MECHANICAL by the default auto-assignment override below.
    agent = _make_agent(identity_tribes=["MECHANICAL", "ELEMENTAL", "MURLOC", "NONE"])
    agent.set_episode_identity(0)  # pin identity 0 (assigned MECHANICAL)
    # 3 of 5 minions on-tribe → bonus should be 3/5.
    board = [_m(Race.MECHANICAL)] * 3 + [_m(Race.BEAST)] * 2
    agent._desc_by_seat[1] = board_descriptor(_state(board), 0)
    bonus = agent._diversity_bonus(seat=1)
    assert np.isclose(bonus, 0.6), bonus
    assert agent._phi_seen[0]


def test_assigned_bonus_zero_for_off_tribe_board():
    agent = _make_agent(identity_tribes=["MECHANICAL", "ELEMENTAL", "MURLOC", "NONE"])
    agent.set_episode_identity(1)  # assigned ELEMENTAL
    board = [_m(Race.MECHANICAL)] * 5  # no elementals
    agent._desc_by_seat[2] = board_descriptor(_state(board), 0)
    bonus = agent._diversity_bonus(seat=2)
    assert bonus == 0.0


def test_assigned_bonus_none_identity_rewards_tribeless():
    # identity 3 → NONE: rewarded for the tribeless fraction of the board.
    agent = _make_agent(identity_tribes=["MECHANICAL", "ELEMENTAL", "MURLOC", "NONE"])
    agent.set_episode_identity(3)
    # _m with race=None lands in the None bucket (index 0).
    board = [_m(None)] * 4 + [_m(Race.MECHANICAL)] * 1
    agent._desc_by_seat[0] = board_descriptor(_state(board), 0)
    bonus = agent._diversity_bonus(seat=0)
    assert np.isclose(bonus, 0.8), bonus


def test_diversity_coef_zero_adds_no_reward_but_still_tracks_metrics():
    # At coef=0 the descriptor/EMA are still tracked (so an A/B baseline logs
    # diversity), but the intrinsic *reward bonus* is exactly zero.
    agent = _make_agent(diversity_coef=0.0)
    # Per-seat mode: two seats with explicitly assigned distinct identities.
    agent._identity_by_seat = {0: 0, 1: 1}
    agent._desc_by_seat[0] = board_descriptor(_state([_m(Race.MECHANICAL)] * 4), 0)
    agent.close_segment(seat=0, terminal_reward=0.5)
    agent._desc_by_seat[1] = board_descriptor(_state([_m(Race.ELEMENTAL)] * 4), 0)
    agent.close_segment(seat=1, terminal_reward=-0.3)

    m = agent._dvd_metrics()
    assert agent._phi_seen.sum() == 2          # state tracked for metrics
    assert m["dvd_pop_diversity"] > 0.0
    assert m["dvd_mean_bonus"] == 0.0          # but no reward shaping at coef=0
    assert m["dvd_place_0"] == 0.5
    assert m["dvd_place_1"] == -0.3


# --------------------------------------------------------------------------- #
# Assigned-tribe bonus is non-zero from step 0 even at collapse (the property
# pairwise repulsion lacked): a single identity with no others seen still earns
# its on-tribe fraction.
# --------------------------------------------------------------------------- #
def test_assigned_bonus_nonzero_at_collapse():
    agent = _make_agent(
        diversity_coef=1.0, identity_tribes=["MECHANICAL", "ELEMENTAL", "MURLOC", "NONE"]
    )
    agent.set_episode_identity(1)  # ELEMENTAL, no other identity has data
    agent._desc_by_seat[0] = board_descriptor(_state([_m(Race.ELEMENTAL)] * 5), 0)
    bonus = agent._diversity_bonus(seat=0)
    # Unlike repulsion (which needs another identity to differ from), the
    # assigned bonus is paid immediately for committing to the target tribe.
    assert np.isclose(bonus, 1.0), bonus


# --------------------------------------------------------------------------- #
# diversity_reward_mode: "final" (default) rewards the final-board composition
# terminally; "acquisition" is behind the flag.
# --------------------------------------------------------------------------- #
def _append_seat_row(agent, seat):
    """Append one minimal rollout row for `seat` so close_segment can stamp it."""
    b = agent.rollout_buffer
    b.obs.append(np.zeros(OBS_DIM_V5 + 4, dtype=np.float32))
    b.seat_ids.append(seat); b.rewards.append(0.0); b.dones.append(False)
    b.values.append(0.0); b.log_probs.append(0.0); b.action_indices.append(0)
    b.complete_turn.append(False); b.occupied_masks.append(np.zeros(7, dtype=bool))
    b.order_picks.append(np.full(7, -1, np.int64)); b.legal_lists.append([])
    b.episode_ids.append(0); b.own_board_obs.append(np.zeros(0, np.float32))
    b.opp_board_obs.append(np.zeros(0, np.float32)); b.attack_first.append(0.0)
    b.battle_target.append(0.0); b.battle_target_valid.append(False)
    b.last_next_obs = b.obs[-1]


def test_default_mode_is_final():
    agent = _make_agent(diversity_coef=0.5,
                        identity_tribes=["MECHANICAL", "ELEMENTAL", "MURLOC", "NONE"])
    assert agent.diversity_reward_mode == "final"


def test_final_mode_terminal_bonus_is_coef_times_count():
    # "final" mode rewards coef * (# own-tribe minions on the final board), i.e.
    # COUNT, not fraction — one tribe minion is worth coef (not coef/board).
    agent = _make_agent(diversity_coef=0.5,
                        identity_tribes=["MECHANICAL", "ELEMENTAL", "MURLOC", "NONE"])
    agent.set_episode_identity(0)  # MECHANICAL
    _append_seat_row(agent, seat=0)
    # board = 3 mech + 2 beast → count(mech)=3 → bonus 0.5*3 = 1.5; placement -0.2
    agent._desc_by_seat[0] = board_descriptor(
        _state([_m(Race.MECHANICAL)] * 3 + [_m(Race.BEAST)] * 2), 0)
    agent.close_segment(seat=0, terminal_reward=-0.2)
    assert np.isclose(agent.rollout_buffer.rewards[-1], -0.2 + 0.5 * 3), agent.rollout_buffer.rewards[-1]


def test_potential_mode_telescopes_to_final_count():
    # Potential shaping (Φ=coef*count): the per-step signed deltas credited in
    # observe() PLUS the terminal residual stamped in close_segment must sum to
    # exactly coef * (# own-tribe minions on the final board) — the same total as
    # "final" mode, but spread across the episode so PPO can credit the buys.
    agent = _make_agent(diversity_coef=0.5,
                        identity_tribes=["MECHANICAL", "ELEMENTAL", "MURLOC", "NONE"])
    agent.diversity_reward_mode = "potential"
    agent.set_episode_identity(0)  # MECHANICAL

    # Emulate the per-step hook: by mid-episode the seat has 2 mechs, so observe()
    # has credited coef*(2-0) and left _prev_count_by_seat[0] == 2.
    step_credit = agent.diversity_coef * 2.0
    agent._prev_count_by_seat[0] = 2.0

    # Final board (before death) = 3 mech: the last buy was never re-observed as
    # an act row, so close_segment must stamp the residual coef*(3-2).
    _append_seat_row(agent, seat=0)
    agent._desc_by_seat[0] = board_descriptor(_state([_m(Race.MECHANICAL)] * 3), 0)
    agent.close_segment(seat=0, terminal_reward=-0.2)
    residual = agent.rollout_buffer.rewards[-1] - (-0.2)

    assert np.isclose(residual, agent.diversity_coef * 1.0), residual
    assert np.isclose(step_credit + residual, 0.5 * 3.0)  # == coef*final_count


def test_acquisition_mode_does_not_add_terminal_fraction():
    # In acquisition mode the terminal row carries only placement (per-step bonus
    # is added in observe(), not here).
    agent = _make_agent(diversity_coef=0.5,
                        identity_tribes=["MECHANICAL", "ELEMENTAL", "MURLOC", "NONE"])
    agent.diversity_reward_mode = "acquisition"
    agent.set_episode_identity(0)
    _append_seat_row(agent, seat=0)
    agent._desc_by_seat[0] = board_descriptor(_state([_m(Race.MECHANICAL)] * 5), 0)
    agent.close_segment(seat=0, terminal_reward=-0.2)
    assert np.isclose(agent.rollout_buffer.rewards[-1], -0.2), agent.rollout_buffer.rewards[-1]
