"""Shared pytest fixtures and patch constants for tests."""

from __future__ import annotations

from typing import List, Optional

import pytest

from src.bg_catalog.patch_context import PatchContext, load_patch_context

_PATCH_36393 = "data/bgcore/15_6_2_36393"
from src.envs.minibg.obs import (
    build_observation as _build_observation,
    encode_minion as _encode_minion,
    encode_pending_choice as _encode_pending_choice,
    encode_slots as _encode_slots,
)
from src.envs.minibg.state import MiniBGState, Minion

PATCH_CTX: PatchContext = load_patch_context(_PATCH_36393)
NUM_POOL_INDICES: int = PATCH_CTX.num_pool_indices


@pytest.fixture(scope="session")
def patch_ctx() -> PatchContext:
    return PATCH_CTX


def obs_encode_minion(minion: Optional[Minion], **kwargs):
    kwargs.setdefault("card_id_to_dense", PATCH_CTX.card_id_to_dense)
    return _encode_minion(minion, **kwargs)


def obs_encode_slots(minions, num_slots: int, **kwargs):
    kwargs.setdefault("card_id_to_dense", PATCH_CTX.card_id_to_dense)
    return _encode_slots(minions, num_slots, **kwargs)


def obs_encode_pending_choice(me, rl_pending=None, **kwargs):
    kwargs.setdefault("card_id_to_dense", PATCH_CTX.card_id_to_dense)
    return _encode_pending_choice(me, rl_pending=rl_pending, **kwargs)


def obs_build_observation(
    state: MiniBGState,
    player_idx: int,
    last_battle_signed: float,
    enemy_last_seen_board: Optional[List[Minion]],
    **kwargs,
):
    kwargs.setdefault("patch", PATCH_CTX)
    return _build_observation(
        state,
        player_idx,
        last_battle_signed,
        enemy_last_seen_board,
        **kwargs,
    )
