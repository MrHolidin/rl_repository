"""BGLike observation layout (no enemy board block)."""

from __future__ import annotations

import numpy as np

from src.envs.bglike.actions import BOARD_SIZE, HAND_SIZE, MAX_SHOP_SLOTS
from src.envs.bglike.game import BGLikeGame
from src.envs.bglike.obs import OBS_DIM, SLOT_DIM, build_observation
from src.envs.minibg.obs import GLOBAL_DIM, LAST_BATTLE_DIM, PENDING_CHOICE_DIM, PHASE_DIM


def test_obs_dim_excludes_enemy_board_block():
    expected = (
        GLOBAL_DIM
        + BOARD_SIZE * SLOT_DIM
        + MAX_SHOP_SLOTS * SLOT_DIM
        + HAND_SIZE * SLOT_DIM
        + LAST_BATTLE_DIM
        + PHASE_DIM
        + PENDING_CHOICE_DIM
    )
    assert OBS_DIM == expected


def test_build_observation_shape():
    game = BGLikeGame(seed=0)
    state = game.initial_state()
    obs = build_observation(state, 0, 0.0, is_my_turn=True)
    assert obs.shape == (OBS_DIM,)
    assert obs.dtype == np.float32
