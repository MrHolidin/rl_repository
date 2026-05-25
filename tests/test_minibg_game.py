import pytest

from src.envs.minibg.actions import (
    Action,
    BOARD_SIZE,
    BUY_COST,
    HAND_SIZE,
    MAX_SHOP_SLOTS,
    SELL_REWARD,
    STARTING_HEALTH,
    gold_for_round,
)
from tests.minibg_helpers import make_minion
from src.envs.minibg.game import MiniBGGame
from src.envs.minibg.state import MiniBGState, PlayerPhase

from tests.minibg_helpers import set_acting_player


def _force_shop(state: MiniBGState, player_idx: int, *card_ids):
    p = state.players[player_idx]
    p.shop = [make_minion(cid) if cid is not None else None for cid in card_ids]
    while len(p.shop) < MAX_SHOP_SLOTS:
        p.shop.append(None)


def _make_game(seed=0):
    g = MiniBGGame(seed=seed, patch_dir="data/bgcore/15_6_2_36393")
    return g, g.initial_state()


def _submit_order_identity(g: MiniBGGame, s: MiniBGState) -> MiniBGState:
    if s.players[s.current_player_index].phase == PlayerPhase.SHOP:
        return g.apply_action(s, int(Action.FINISH))
    return s


def test_initial_state_basic_invariants():
    g, s = _make_game()
    assert s.round_number == 1
    for p in s.players:
        assert p.health == STARTING_HEALTH
        assert p.gold == gold_for_round(1)
        assert p.board == []
        assert p.hand == [None] * HAND_SIZE
        assert p.phase == PlayerPhase.SHOP
        assert p.shopping_finished is False


def test_initial_next_tier_up_cost_is_base():
    g, s = _make_game()
    assert s.players[0].next_tier_up_cost == 5
    assert s.players[0].next_tier_up_cost == s.players[1].next_tier_up_cost


def test_tier_up_cost_discount_after_round_advance():
    g, s = _make_game()
    s = _submit_order_identity(g, s)
    s = _submit_order_identity(g, s)
    assert s.round_number == 2
    assert s.players[0].next_tier_up_cost == 4
    assert s.players[1].next_tier_up_cost == 4


def test_level_up_charges_discounted_price_and_resets_next_base():
    g, s = _make_game()
    s.players[0].gold = 10
    assert s.players[0].next_tier_up_cost == 5
    s2 = g.apply_action(s, int(Action.LEVEL_UP))
    p0 = s2.players[0]
    assert p0.tavern_tier == 2
    assert p0.gold == 5
    assert p0.next_tier_up_cost == 7
    assert sum(1 for x in p0.shop if x is not None) == 4


def test_legal_actions_initial_state():
    g, s = _make_game()
    assert set(g.legal_actions(s)) == {
        int(Action.BUY_SLOT_0),
        int(Action.BUY_SLOT_1),
        int(Action.BUY_SLOT_2),
        int(Action.ROLL),
        int(Action.FINISH),
        int(Action.FINISH_FREEZE_SHOP),
        int(Action.FREEZE_SHOP_SLOT_0),
        int(Action.FREEZE_SHOP_SLOT_1),
        int(Action.FREEZE_SHOP_SLOT_2),
    }


def test_buy_lands_in_hand_not_board():
    g, s = _make_game()
    _force_shop(s, 0, "recruit", "recruit", "recruit")
    s2 = g.apply_action(s, int(Action.BUY_SLOT_0))
    p0 = s2.players[0]
    assert p0.hand[0] is not None and p0.hand[0].card_id == "EX1_162"
    assert p0.board == []
    assert p0.shop[0] is None
    assert p0.gold == gold_for_round(1) - BUY_COST


def test_place_moves_hand_to_board():
    g, s = _make_game()
    _force_shop(s, 0, "recruit", "recruit", "recruit")
    s2 = g.apply_action(s, int(Action.BUY_SLOT_0))
    s3 = g.apply_action(s2, int(Action.PLACE_HAND_0))
    p0 = s3.players[0]
    assert p0.hand[0] is None
    assert [m.card_id for m in p0.board] == ["EX1_162"]


def test_buy_illegal_when_hand_full_but_legal_when_board_full():
    g, s = _make_game()
    _force_shop(s, 0, "recruit", "recruit", "recruit")
    s.players[0].gold = 100
    s.players[0].hand = [make_minion("guard") for _ in range(HAND_SIZE)]
    assert int(Action.BUY_SLOT_0) not in set(g.legal_actions(s))
    s.players[0].hand = [None] * HAND_SIZE
    s.players[0].board = [make_minion("recruit") for _ in range(BOARD_SIZE)]
    assert int(Action.BUY_SLOT_0) in set(g.legal_actions(s))


def test_place_illegal_when_board_full():
    g, s = _make_game()
    s.players[0].board = [make_minion("recruit") for _ in range(BOARD_SIZE)]
    s.players[0].hand[0] = make_minion("guard")
    assert int(Action.PLACE_HAND_0) not in set(g.legal_actions(s))


def test_sell_returns_one_gold_and_compacts_board():
    g, s = _make_game()
    s.players[0].board = [make_minion("recruit"), make_minion("guard"), make_minion("buffer")]
    s.players[0].gold = 0
    s2 = g.apply_action(s, int(Action.SELL_BOARD_1))
    p0 = s2.players[0]
    assert [m.card_id for m in p0.board] == ["EX1_162", "UNG_073"]
    assert p0.gold == SELL_REWARD


def test_finish_in_shop_submits_turn():
    g, s = _make_game()
    order = s.shop_turn_order
    assert s.current_player_index == order[0]
    s2 = g.apply_action(s, int(Action.FINISH))
    assert s2.players[order[0]].phase == PlayerPhase.DONE
    assert s2.players[order[0]].shop_freeze_next_round is False
    assert s2.current_player_index == order[1]
    assert g.legal_actions(s2)  # opponent's shop turn


def test_finish_freeze_shop_carry_offers_through_round_transition():
    g, s = _make_game(seed=11)
    _force_shop(s, 0, "recruit", "buffer", "guard")
    before = tuple(m.card_id if m else None for m in s.players[0].shop[:4])
    s = g.apply_action(s, int(Action.FINISH_FREEZE_SHOP))
    assert s.players[0].shop_freeze_next_round is True
    s = _submit_order_identity(g, s)

    ids_p0_now = tuple(m.card_id if m else None for m in s.players[0].shop[:4])
    assert ids_p0_now == before
    assert not s.players[0].shop_freeze_next_round


def test_round_advances_when_both_players_submit():
    g, s = _make_game()
    s = _submit_order_identity(g, s)
    s = _submit_order_identity(g, s)
    assert s.round_number == 2
    assert all(p.phase == PlayerPhase.SHOP for p in s.players)
    assert all(p.gold == gold_for_round(2) for p in s.players)


def test_hand_persists_across_rounds():
    g, s = _make_game()
    s.players[0].hand[0] = make_minion("guard")
    s = _submit_order_identity(g, s)
    s = _submit_order_identity(g, s)
    assert s.players[0].hand[0].card_id == "CS2_065"


def test_rockpool_on_place_buffs_murloc_on_board_not_hand():
    g = MiniBGGame(seed=1234, patch_dir="data/bgcore/15_6_2_36393")
    s = g.initial_state()
    set_acting_player(s, 0)
    s.players[0].board = [make_minion("murloc_warleader")]
    _force_shop(s, 0, "buffer", None, None)
    s.players[0].gold = 10
    s2 = g.apply_action(s, int(Action.BUY_SLOT_0))
    mur = s2.players[0].board[0]
    assert (mur.bonus_attack, mur.bonus_health) == (0, 0)
    s3 = g.apply_action(s2, int(Action.PLACE_HAND_0))
    mur_after = s3.players[0].board[0]
    assert mur_after.card_id == "EX1_507"
    assert (mur_after.bonus_attack, mur_after.bonus_health) == (1, 1)
    rp = s3.players[0].board[1]
    assert rp.card_id == "UNG_073"
    assert rp.bonus_attack == 0 and rp.bonus_health == 0

    g = MiniBGGame(seed=1234, patch_dir="data/bgcore/15_6_2_36393")
    s = g.initial_state()
    set_acting_player(s, 0)
    s.players[0].hand[0] = make_minion("recruit")
    _force_shop(s, 0, "buffer", None, None)
    s.players[0].gold = 10
    s2 = g.apply_action(s, int(Action.BUY_SLOT_0))
    assert s2.players[0].hand[1] is not None
    assert s2.players[0].hand[1].card_id == "UNG_073"
    assert s2.players[0].hand[0].bonus_attack == 0
    s3 = g.apply_action(s2, int(Action.PLACE_HAND_1))
    assert s3.players[0].board[0].card_id == "UNG_073"
    rec_hand = s3.players[0].hand[0]
    assert rec_hand is not None and rec_hand.card_id == "EX1_162"
    assert rec_hand.bonus_attack == 0


def test_crowd_favorite_buff_when_battlecry_minion_placed():
    g = MiniBGGame(seed=0, patch_dir="data/bgcore/15_6_2_36393")
    s = g.initial_state()
    s.players[0].board = [make_minion("AT_121")]
    s.players[0].hand[0] = make_minion("buffer")
    s2 = g.apply_action(s, int(Action.PLACE_HAND_0))
    cf = s2.players[0].board[0]
    assert cf.card_id == "AT_121"
    assert cf.bonus_attack == 1 and cf.bonus_health == 1


def test_vulgar_homunculus_damage_blocked_by_mal_ganis():
    g = MiniBGGame(seed=0, patch_dir="data/bgcore/15_6_2_36393")
    s = g.initial_state()
    s.players[0].board = [make_minion("mal_ganis")]
    s.players[0].hand[0] = make_minion("vulgar_homunculus")
    s2 = g.apply_action(s, int(Action.PLACE_HAND_0))
    assert s2.players[0].health == STARTING_HEALTH
    assert s2.players[0].hero_damage_taken_total == 0


def test_wrath_weaver_buffs_when_demon_placed_damage_not_blocked():
    g = MiniBGGame(seed=0, patch_dir="data/bgcore/15_6_2_36393")
    s = g.initial_state()
    s.players[0].board = [make_minion("wrath_weaver")]
    s.players[0].hand[0] = make_minion("imp_demon")
    s2 = g.apply_action(s, int(Action.PLACE_HAND_0))
    p0 = s2.players[0]
    assert p0.health == STARTING_HEALTH - 1
    assert p0.hero_damage_taken_total == 1
    ww = p0.board[0]
    assert ww.card_id == "BGS_004"
    assert ww.bonus_attack == 2 and ww.bonus_health == 2


def test_annihilan_gains_stats_from_cumulative_hero_damage():
    g = MiniBGGame(seed=0, patch_dir="data/bgcore/15_6_2_36393")
    s = g.initial_state()
    s.players[0].board = [make_minion("wrath_weaver")]
    s.players[0].hand[0] = make_minion("imp_demon")
    s2 = g.apply_action(s, int(Action.PLACE_HAND_0))
    s2.players[0].hand[0] = make_minion("annihilan")
    s3 = g.apply_action(s2, int(Action.PLACE_HAND_0))
    ann = s3.players[0].board[2]
    assert ann.card_id == "BGS_010"
    assert ann.bonus_attack == 0 and ann.bonus_health == 1

    g = MiniBGGame(seed=0, patch_dir="data/bgcore/15_6_2_36393")
    s = g.initial_state()
    s.players[0].board = [make_minion("mentor"), make_minion("toy_mech")]
    s_done = g.apply_action(s, int(Action.FINISH))
    assert s_done.players[0].board[1].card_id == "BOT_445"
    assert s_done.players[0].board[1].bonus_attack == 2
    assert s_done.players[0].board[1].bonus_health == 2


def test_terminal_apply_raises():
    g, s = _make_game()
    s.done = True
    s.winner = 1
    with pytest.raises(ValueError):
        g.apply_action(s, int(Action.FINISH))


def test_game_runs_to_max_rounds_and_draws_when_no_damage():
    g = MiniBGGame(seed=42, patch_dir="data/bgcore/15_6_2_36393")
    s = g.initial_state()
    while not s.done:
        s.players[s.current_player_index].board = []
        s = _submit_order_identity(g, s)
    assert s.done is True
    assert s.winner == 0


def test_game_ends_when_player_dies():
    g = MiniBGGame(seed=0, patch_dir="data/bgcore/15_6_2_36393")
    s = g.initial_state()
    s.players[0].board = []
    s.players[0].health = 1
    s.players[1].board = [make_minion("big_guy")]
    s = _submit_order_identity(g, s)
    s = _submit_order_identity(g, s)
    assert s.done is True
    assert s.winner == -1
