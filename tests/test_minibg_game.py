import pytest

from src.envs.minibg.actions import (
    Action,
    BUY_COST,
    MAX_SHOP_ACTIONS,
    SELL_REWARD,
    STARTING_HEALTH,
    gold_for_round,
)
from src.envs.minibg.cards import make_minion
from src.envs.minibg.game import MiniBGGame
from src.envs.minibg.state import MiniBGState


def _force_shop(state: MiniBGState, player_idx: int, *card_ids):
    p = state.players[player_idx]
    p.shop = [make_minion(cid) if cid is not None else None for cid in card_ids]
    while len(p.shop) < 3:
        p.shop.append(None)


def _make_game(seed=0):
    g = MiniBGGame(seed=seed)
    s = g.initial_state()
    return g, s


def test_initial_state_basic_invariants():
    g, s = _make_game()
    assert s.round_number == 1
    assert s.current_player_index == 0
    assert s.done is False
    assert s.winner is None
    for p in s.players:
        assert p.health == STARTING_HEALTH
        assert p.gold == gold_for_round(1)
        assert p.tavern_tier == 1
        assert p.board == []
        assert all(slot is not None for slot in p.shop)
        assert p.shopping_finished is False
        assert p.shop_actions_used == 0


def test_legal_actions_initial_state():
    g, s = _make_game()
    legal = set(g.legal_actions(s))
    assert legal == {
        int(Action.BUY_SLOT_0),
        int(Action.BUY_SLOT_1),
        int(Action.BUY_SLOT_2),
        int(Action.ROLL),
        int(Action.FINISH),
    }


def test_buy_action_places_minion_and_empties_slot():
    g, s = _make_game()
    _force_shop(s, 0, "recruit", "recruit", "recruit")
    s2 = g.apply_action(s, int(Action.BUY_SLOT_0))
    p0 = s2.players[0]
    assert len(p0.board) == 1
    assert p0.board[0].card_id == "recruit"
    assert p0.shop[0] is None
    assert p0.gold == gold_for_round(1) - BUY_COST
    assert p0.shop_actions_used == 1


def test_cannot_buy_when_board_full():
    g, s = _make_game()
    _force_shop(s, 0, "recruit", "recruit", "recruit")
    s.players[0].gold = 100
    s.players[0].board = [make_minion("recruit") for _ in range(4)]
    legal = set(g.legal_actions(s))
    for slot in (Action.BUY_SLOT_0, Action.BUY_SLOT_1, Action.BUY_SLOT_2):
        assert int(slot) not in legal


def test_cannot_buy_empty_slot():
    g, s = _make_game()
    _force_shop(s, 0, None, "recruit", "recruit")
    legal = set(g.legal_actions(s))
    assert int(Action.BUY_SLOT_0) not in legal
    assert int(Action.BUY_SLOT_1) in legal
    assert int(Action.BUY_SLOT_2) in legal


def test_sell_returns_one_gold_and_compacts_board():
    g, s = _make_game()
    s.players[0].board = [make_minion("recruit"), make_minion("guard"), make_minion("buffer")]
    s.players[0].gold = 0
    s2 = g.apply_action(s, int(Action.SELL_BOARD_1))
    p0 = s2.players[0]
    assert [m.card_id for m in p0.board] == ["recruit", "buffer"]
    assert p0.gold == SELL_REWARD


def test_roll_costs_one_gold_and_replaces_all_slots():
    g, s = _make_game()
    _force_shop(s, 0, "recruit", "recruit", "recruit")
    s.players[0].gold = 5
    s2 = g.apply_action(s, int(Action.ROLL))
    p0 = s2.players[0]
    assert p0.gold == 4
    assert all(slot is not None for slot in p0.shop)


def test_level_up_blocked_at_low_gold():
    g, s = _make_game()
    s.players[0].gold = 3
    legal = set(g.legal_actions(s))
    assert int(Action.LEVEL_UP) not in legal


def test_level_up_succeeds_at_correct_gold():
    g, s = _make_game()
    s.players[0].gold = 4
    legal = set(g.legal_actions(s))
    assert int(Action.LEVEL_UP) in legal
    s2 = g.apply_action(s, int(Action.LEVEL_UP))
    p0 = s2.players[0]
    assert p0.tavern_tier == 2
    assert p0.gold == 0


def test_level_up_blocked_at_max_tier():
    g, s = _make_game()
    s.players[0].gold = 100
    s.players[0].tavern_tier = 3
    legal = set(g.legal_actions(s))
    assert int(Action.LEVEL_UP) not in legal


def test_finish_switches_to_other_player():
    g, s = _make_game()
    s2 = g.apply_action(s, int(Action.FINISH))
    assert s2.current_player_index == 1
    assert s2.players[0].shopping_finished is True
    assert s2.players[1].shopping_finished is False


def test_both_finish_triggers_battle_and_advances_round():
    g, s = _make_game()
    s2 = g.apply_action(s, int(Action.FINISH))
    s3 = g.apply_action(s2, int(Action.FINISH))
    assert s3.round_number == 2
    assert s3.current_player_index == 0
    assert all(not p.shopping_finished for p in s3.players)
    assert all(p.gold == gold_for_round(2) for p in s3.players)
    assert all(p.shop_actions_used == 0 for p in s3.players)


def test_action_limit_forces_finish_after_10():
    g, s = _make_game()
    s.players[0].gold = 1000
    state = s
    for _ in range(MAX_SHOP_ACTIONS):
        state = g.apply_action(state, int(Action.ROLL))
    assert state.players[0].shopping_finished is True
    assert state.current_player_index == 1


def test_finish_does_not_count_against_action_limit():
    g, s = _make_game()
    s2 = g.apply_action(s, int(Action.FINISH))
    assert s2.players[0].shop_actions_used == 0


def test_buffer_buffs_existing_friendly():
    g = MiniBGGame(seed=1234)
    s = g.initial_state()
    s.players[0].board = [make_minion("recruit")]
    _force_shop(s, 0, "buffer", None, None)
    s.players[0].gold = 10
    s2 = g.apply_action(s, int(Action.BUY_SLOT_0))
    recruit = s2.players[0].board[0]
    assert recruit.card_id == "recruit"
    assert recruit.bonus_attack == 1
    assert recruit.bonus_health == 1


def test_mentor_fires_on_finish():
    g = MiniBGGame(seed=0)
    s = g.initial_state()
    s.players[0].board = [make_minion("mentor"), make_minion("recruit")]
    s2 = g.apply_action(s, int(Action.FINISH))
    recruit = s2.players[0].board[1]
    assert recruit.bonus_attack == 2
    assert recruit.bonus_health == 1


def test_mentor_fires_on_auto_finish_at_action_cap():
    g = MiniBGGame(seed=0)
    s = g.initial_state()
    s.players[0].board = [make_minion("mentor"), make_minion("recruit")]
    s.players[0].gold = 1000
    state = s
    for _ in range(MAX_SHOP_ACTIONS):
        state = g.apply_action(state, int(Action.ROLL))
    recruit = state.players[0].board[1]
    assert recruit.bonus_attack == 2
    assert recruit.bonus_health == 1


def test_buffer_no_others_is_noop():
    g = MiniBGGame(seed=1)
    s = g.initial_state()
    _force_shop(s, 0, "buffer", None, None)
    s.players[0].board = []
    s.players[0].gold = 10
    s2 = g.apply_action(s, int(Action.BUY_SLOT_0))
    buf = s2.players[0].board[0]
    assert buf.card_id == "buffer"
    assert buf.bonus_attack == 0
    assert buf.bonus_health == 0


def test_apply_action_does_not_mutate_input_state():
    g, s = _make_game()
    p0_gold_before = s.players[0].gold
    p0_shop_before = [None if m is None else m.card_id for m in s.players[0].shop]
    _ = g.apply_action(s, int(Action.FINISH))
    assert s.players[0].gold == p0_gold_before
    assert [None if m is None else m.card_id for m in s.players[0].shop] == p0_shop_before
    assert s.players[0].shopping_finished is False


def test_illegal_action_raises():
    g, s = _make_game()
    s.players[0].board = [make_minion("recruit") for _ in range(4)]
    s.players[0].gold = 100
    with pytest.raises(ValueError):
        g.apply_action(s, int(Action.BUY_SLOT_0))


def test_terminal_apply_raises():
    g, s = _make_game()
    s.done = True
    s.winner = 1
    with pytest.raises(ValueError):
        g.apply_action(s, int(Action.FINISH))


def test_game_runs_to_round_15_and_draws_when_no_damage():
    g = MiniBGGame(seed=42)
    s = g.initial_state()
    while not s.done:
        s.players[s.current_player_index].board = []
        s = g.apply_action(s, int(Action.FINISH))
    assert s.done is True
    assert s.winner == 0
    assert all(p.health > 0 for p in s.players)


def test_game_ends_when_player_dies():
    g = MiniBGGame(seed=0)
    s = g.initial_state()
    s.players[0].board = []
    s.players[0].health = 1
    s.players[1].board = [make_minion("big_guy")]
    s2 = g.apply_action(s, int(Action.FINISH))
    s3 = g.apply_action(s2, int(Action.FINISH))
    assert s3.done is True
    assert s3.winner == -1


def test_current_player_token_mapping():
    g, s = _make_game()
    assert g.current_player(s) == 1
    s2 = g.apply_action(s, int(Action.FINISH))
    assert g.current_player(s2) == -1
