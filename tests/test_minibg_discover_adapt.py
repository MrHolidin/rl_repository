"""Discover (tier-weighted Murloc) and Gentle Megasaur Adapt."""

from src.envs.minibg.actions import Action, HAND_SIZE
from tests.conftest import PATCH_CTX
from tests.minibg_helpers import make_minion
from src.bg_recruitment.discover_pool import ADAPT_KEYS_ALL, murloc_discover_card_ids
from src.envs.minibg.game import MiniBGGame
from src.envs.minibg.state import PendingChoiceKind

from tests.minibg_helpers import set_acting_player


def test_primalfin_blocks_shop_until_discover_pick():
    g = MiniBGGame(seed=42, shop_full_tribes=True, patch_dir="data/bgcore/15_6_2_36393")
    s = g.initial_state()
    set_acting_player(s, 0)
    p = s.players[0]
    p.board = []
    p.hand[0] = make_minion("primalfin_lookout")
    assert p.pending_choice is None
    s2 = g.apply_action(s, int(Action.PLACE_HAND_0))
    p2 = s2.players[0]
    assert p2.pending_choice is not None
    assert p2.pending_choice.kind == PendingChoiceKind.DISCOVER_MURLOC
    assert len(set(p2.pending_choice.options)) == 3
    for cid in p2.pending_choice.options:
        assert cid in murloc_discover_card_ids(patch=PATCH_CTX)
    legal = set(g.legal_actions(s2))
    assert legal == {
        int(Action.DISCOVER_PICK_0),
        int(Action.DISCOVER_PICK_1),
        int(Action.DISCOVER_PICK_2),
    }
    picked = p2.pending_choice.options[1]
    s3 = g.apply_action(s2, int(Action.DISCOVER_PICK_1))
    p3 = s3.players[0]
    assert p3.pending_choice is None
    assert picked in {x.card_id for x in p3.hand if x is not None}
    assert int(Action.BUY_SLOT_0) in set(g.legal_actions(s3))


def test_gentle_megasaur_adapt_all_murlocs():
    g = MiniBGGame(seed=0, shop_full_tribes=True, patch_dir="data/bgcore/15_6_2_36393")
    s = g.initial_state()
    set_acting_player(s, 0)
    p = s.players[0]
    p.board = [make_minion("rockpool_hunter")]
    p.hand[0] = make_minion("gentle_megasaur")
    s2 = g.apply_action(s, int(Action.PLACE_HAND_0))
    pc = s2.players[0].pending_choice
    assert pc is not None and pc.kind == PendingChoiceKind.ADAPT
    for k in pc.options:
        assert k in ADAPT_KEYS_ALL
    m = s2.players[0].board[0]
    before = (m.raw_attack, m.max_health, frozenset(m.all_keywords), len(m.abilities))
    s3 = g.apply_action(s2, int(Action.DISCOVER_PICK_0))
    m = s3.players[0].board[0]
    after = (m.raw_attack, m.max_health, frozenset(m.all_keywords), len(m.abilities))
    assert before != after


def test_golden_megasaur_two_adapt_rounds():
    g = MiniBGGame(seed=1, shop_full_tribes=True, patch_dir="data/bgcore/15_6_2_36393")
    s = g.initial_state()
    set_acting_player(s, 0)
    p = s.players[0]
    p.board = [make_minion("rockpool_hunter")]
    p.hand[0] = make_minion("gentle_megasaur_golden")
    s2 = g.apply_action(s, int(Action.PLACE_HAND_0))
    assert s2.players[0].pending_choice.extra_modals_after >= 1
    s3 = g.apply_action(s2, int(Action.DISCOVER_PICK_0))
    assert s3.players[0].pending_choice is not None
    s4 = g.apply_action(s3, int(Action.DISCOVER_PICK_2))
    assert s4.players[0].pending_choice is None


def test_place_primalfin_illegal_if_discover_overflow_hand():
    g = MiniBGGame(seed=0, shop_full_tribes=True, patch_dir="data/bgcore/15_6_2_36393")
    s = g.initial_state()
    set_acting_player(s, 0)
    p = s.players[0]
    p.board = [make_minion("brann")]
    p.hand = [None] * HAND_SIZE
    p.hand[0] = make_minion("primalfin_lookout")
    for i in range(1, HAND_SIZE):
        p.hand[i] = make_minion("recruit")
    assert int(Action.PLACE_HAND_0) not in set(g.legal_actions(s))
    p.hand = [None] * HAND_SIZE
    p.hand[0] = make_minion("primalfin_lookout")
    p.hand[1] = make_minion("recruit")
    assert int(Action.PLACE_HAND_0) in set(g.legal_actions(s))

