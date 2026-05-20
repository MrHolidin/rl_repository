"""Empty shop legal triggers RuntimeError and writes RL_RUN_DIR log."""

import os
from pathlib import Path

import pytest

from src.bg_catalog.cards import make_minion
from src.envs.minibg.invariants import assert_shop_has_legal_actions
from src.envs.minibg.state import (
    MiniBGState,
    PendingChoice,
    PendingChoiceKind,
    PlayerPhase,
    PlayerState,
)


def _shop_state(*, pending: bool, hand_full: bool) -> MiniBGState:
    hand = [make_minion("recruit") for _ in range(5)] if hand_full else [None] * 5
    p = PlayerState(
        health=40,
        gold=10,
        tavern_tier=3,
        next_tier_up_cost=5,
        board=[],
        shop=[None] * 6,
        hand=hand,
        phase=PlayerPhase.SHOP,
        shop_actions_used=0,
    )
    if pending:
        p.pending_choice = PendingChoice(
            PendingChoiceKind.DISCOVER_MURLOC,
            ("a", "b", "c"),
            0,
        )
    return MiniBGState(
        players=(p, p),
        round_number=1,
        current_player_index=0,
        initiative_player=0,
        winner=None,
        done=False,
    )


def test_assert_raises_and_logs(tmp_path, monkeypatch):
    monkeypatch.setenv("RL_RUN_DIR", str(tmp_path))
    state = _shop_state(pending=True, hand_full=True)
    with pytest.raises(RuntimeError, match="EMPTY_SHOP_LEGAL"):
        assert_shop_has_legal_actions(state, [], where="test")
    log = Path(tmp_path) / "empty_legal_assertion.log"
    assert log.is_file()
    text = log.read_text()
    assert "EMPTY_SHOP_LEGAL" in text
    assert "DISCOVER_MURLOC" in text
    assert "hand_discover_full_hand" in text or "test" in text


def test_assert_ok_when_legal_nonempty():
    state = _shop_state(pending=True, hand_full=True)
    assert_shop_has_legal_actions(state, [56, 57, 58], where="test")
