"""Discover on place must stay legal when a triple merge would fill the hand."""

from src.bg_catalog.cards import make_minion
from src.bg_recruitment.place import place_from_hand
from src.envs.minibg.actions import BOARD_SIZE
from src.envs.minibg.game import MiniBGGame
from src.envs.minibg.state import MiniBGState, PendingChoiceKind, PlayerPhase, PlayerState


def test_discover_pending_stays_legal_when_triple_would_fill_hand():
    """Place discover murloc with 3 recruits on board: triple must not run before discover pick."""
    g = MiniBGGame(seed=3)
    triggers = g._shop_triggers
    rec = make_minion("recruit")
    disc = make_minion("BGS_020")
    p = PlayerState(
        health=40,
        gold=10,
        tavern_tier=3,
        next_tier_up_cost=5,
        board=[make_minion("recruit"), make_minion("recruit"), make_minion("recruit")],
        shop=[None] * 6,
        hand=[rec, rec, disc, make_minion("target_buffer"), make_minion("target_buffer")],
        phase=PlayerPhase.SHOP,
        shop_actions_used=0,
    )
    place_from_hand(
        p,
        2,
        None,
        board_size=BOARD_SIZE,
        triggers=triggers,
        rng=g._rng,
    )
    assert p.pending_choice is not None
    assert p.pending_choice.kind == PendingChoiceKind.DISCOVER_MURLOC
    s = MiniBGState(
        players=(p, p),
        round_number=1,
        current_player_index=0,
        initiative_player=0,
        winner=None,
        done=False,
    )
    legal = list(g.legal_actions(s))
    assert len(legal) == 3
    assert all(a in legal for a in (56, 57, 58))
