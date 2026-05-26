"""BGLike observation layout (no enemy board block)."""

from __future__ import annotations

import numpy as np

from src.bg_core.minion import Minion, Race
from src.envs.bglike.actions import BOARD_SIZE, HAND_SIZE, MAX_SHOP_SLOTS, NUM_PLAYERS
from tests.conftest import PATCH_CTX
from src.envs.bglike.game import BGLikeGame
from src.envs.bglike.obs import (
    BATTLE_HISTORY_OBS_DIM,
    BGLIKE_GLOBAL_CORE_DIM,
    BGLIKE_GLOBAL_DIM,
    LOBBY_PANEL_DIM,
    MAX_OPPS,
    OBS_DIM,
    SLOT_DIM,
    TRIBE_LOCK_THRESHOLD,
    build_observation,
)
from src.envs.minibg.obs import (
    LAST_BATTLE_DIM,
    PENDING_CHOICE_DIM,
    PHASE_DIM,
    RACE_ONEHOT_DIM,
    SHOP_ROTATION_OBS_DIM,
    _RACE_ORDER,
)


def test_obs_dim_excludes_enemy_board_block():
    expected = (
        BGLIKE_GLOBAL_DIM
        + BOARD_SIZE * SLOT_DIM
        + MAX_SHOP_SLOTS * SLOT_DIM
        + HAND_SIZE * SLOT_DIM
        + LAST_BATTLE_DIM
        + PHASE_DIM
        + PENDING_CHOICE_DIM
    )
    assert OBS_DIM == expected


def test_build_observation_shape():
    game = BGLikeGame(seed=0, patch_dir="data/bgcore/15_6_2_36393")
    state = game.initial_state()
    obs = build_observation(
        state,
        0,
        0.0,
        is_my_turn=True,
        patch=game._patch,
    )
    assert obs.shape == (OBS_DIM,)
    assert obs.dtype == np.float32


# ---------------------------------------------------------------------------
# Lobby panel + battle-history blocks (added with the 74257 obs extension).
# ---------------------------------------------------------------------------


def _globals_slice(obs: np.ndarray) -> np.ndarray:
    return obs[:BGLIKE_GLOBAL_DIM]


def _lobby_panel_slice(globals_arr: np.ndarray) -> np.ndarray:
    start = BGLIKE_GLOBAL_CORE_DIM + SHOP_ROTATION_OBS_DIM
    return globals_arr[start : start + LOBBY_PANEL_DIM]


def _battle_history_slice(globals_arr: np.ndarray) -> np.ndarray:
    start = BGLIKE_GLOBAL_CORE_DIM + SHOP_ROTATION_OBS_DIM + LOBBY_PANEL_DIM
    return globals_arr[start : start + BATTLE_HISTORY_OBS_DIM]


def test_lobby_panel_dim_matches_layout():
    assert LOBBY_PANEL_DIM == MAX_OPPS * (2 + RACE_ONEHOT_DIM)
    assert MAX_OPPS == NUM_PLAYERS - 1


def test_initial_state_opponent_panel_all_alive_no_lock():
    game = BGLikeGame(seed=0, patch_dir="data/bgcore/19_6_0_74257")
    state = game.initial_state()
    obs = build_observation(state, 0, 0.0, is_my_turn=True, patch=game._patch)
    panel = _lobby_panel_slice(_globals_slice(obs))
    hp = panel[:MAX_OPPS]
    alive = panel[MAX_OPPS : 2 * MAX_OPPS]
    tribes = panel[2 * MAX_OPPS :].reshape(MAX_OPPS, RACE_ONEHOT_DIM)
    # All opponents start at full HP, all alive, no tribe lock (no combat yet).
    assert np.allclose(hp[: NUM_PLAYERS - 1], 1.0)
    assert np.all(alive[: NUM_PLAYERS - 1] == 1.0)
    assert np.all(tribes == 0.0)


def test_opponent_panel_sorts_by_hp_descending():
    game = BGLikeGame(seed=0, patch_dir="data/bgcore/19_6_0_74257")
    state = game.initial_state()
    # Drop opponents to distinct HP values; seat 0 is the observer.
    state.players[1].health = 12
    state.players[2].health = 25
    state.players[3].health = 3
    obs = build_observation(state, 0, 0.0, is_my_turn=True, patch=game._patch)
    panel = _lobby_panel_slice(_globals_slice(obs))
    hp = panel[:MAX_OPPS]
    # First three observed opponents (in HP order) include the 25, 12, 3 we set
    # plus the others at full health. They must be monotonically non-increasing.
    nz = hp[hp > 0]
    assert list(nz) == sorted(nz, reverse=True)


def test_tribe_lock_triggers_after_combat():
    game = BGLikeGame(seed=0, patch_dir="data/bgcore/19_6_0_74257")
    state = game.initial_state()
    # Pre-populate the tribe-counts snapshot directly; we don't need a real
    # board to test the obs encoder (the snapshot is what build_observation reads).
    state.players[1].last_round_tribe_counts = {Race.MECHANICAL: TRIBE_LOCK_THRESHOLD}

    obs = build_observation(state, 0, 0.0, is_my_turn=True, patch=game._patch)
    panel = _lobby_panel_slice(_globals_slice(obs))
    tribes = panel[2 * MAX_OPPS :].reshape(MAX_OPPS, RACE_ONEHOT_DIM)
    mech_idx = _RACE_ORDER.index(Race.MECHANICAL)
    # The locked row must have exactly one bit set at MECHANICAL.
    locked_rows = np.where(tribes[:, mech_idx] == 1.0)[0]
    assert len(locked_rows) == 1, locked_rows
    assert tribes[locked_rows[0]].sum() == 1.0


def test_tribe_lock_below_threshold_stays_silent():
    game = BGLikeGame(seed=0, patch_dir="data/bgcore/19_6_0_74257")
    state = game.initial_state()
    state.players[1].last_round_tribe_counts = {Race.MECHANICAL: TRIBE_LOCK_THRESHOLD - 1}
    obs = build_observation(state, 0, 0.0, is_my_turn=True, patch=game._patch)
    panel = _lobby_panel_slice(_globals_slice(obs))
    tribes = panel[2 * MAX_OPPS :].reshape(MAX_OPPS, RACE_ONEHOT_DIM)
    assert np.all(tribes == 0.0)


def test_battle_history_empty_at_round_start():
    game = BGLikeGame(seed=0, patch_dir="data/bgcore/19_6_0_74257")
    state = game.initial_state()
    obs = build_observation(state, 0, 0.0, is_my_turn=True, patch=game._patch)
    bh = _battle_history_slice(_globals_slice(obs))
    # Layout: self_vals(3), self_mask(3), opp_vals(3), opp_mask(3), known(1).
    assert np.all(bh[:6] == 0.0)  # no history yet
    assert np.all(bh[6:12] == 0.0)
    # next_opp known is 1.0 (full lobby → deterministic).
    assert bh[12] == 1.0


def test_battle_history_records_last_three_deltas():
    """Append three deltas to a player and verify the mask + most-recent-last layout."""
    game = BGLikeGame(seed=0, patch_dir="data/bgcore/19_6_0_74257")
    state = game.initial_state()
    state.players[0].battle_history = (0.10, 0.20, 0.30)  # oldest-first

    obs = build_observation(state, 0, 0.0, is_my_turn=True, patch=game._patch)
    bh = _battle_history_slice(_globals_slice(obs))
    np.testing.assert_allclose(bh[:3], [0.10, 0.20, 0.30])
    assert np.all(bh[3:6] == 1.0)


def test_battle_history_left_pads_when_short():
    game = BGLikeGame(seed=0, patch_dir="data/bgcore/19_6_0_74257")
    state = game.initial_state()
    state.players[0].battle_history = (0.42,)
    obs = build_observation(state, 0, 0.0, is_my_turn=True, patch=game._patch)
    bh = _battle_history_slice(_globals_slice(obs))
    # Most-recent occupies the last slot; older slots are zero with mask=0.
    assert bh[2] == np.float32(0.42)
    assert bh[0] == 0.0 and bh[1] == 0.0
    assert bh[3] == 0.0 and bh[4] == 0.0 and bh[5] == 1.0
