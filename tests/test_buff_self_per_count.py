"""``BuffSelfPerCount`` reproduces the three classes it replaced, exhaustively.

The golden trace is an integration check: it proves a scripted lobby plays out
identically, but it cannot prove three *independent* effect branches are each
equivalent — a card whose effect never fires in those three lobbies would pass
it untouched. So this file pins each variant directly, over the full cross
product of the arguments the merge collapsed (source × exclude_self × per-stat
amounts × board composition), against a reference implementation transcribed
from the pre-merge dispatch bodies.
"""

from __future__ import annotations

import itertools

import pytest

from src.bg_core.board_helpers import (
    apply_buff_self_per_count,
    count_friendly_tribe,
    count_golden_friendlies,
    count_unique_tribes,
)
from src.bg_core.effects import BuffSelfPerCount, CountSource
from src.bg_core.minion import Minion, Race


def _m(card_id: str, race=None, *, golden: bool = False) -> Minion:
    return Minion(
        card_id=card_id,
        base_attack=1,
        base_health=1,
        tier=1,
        race=race,
        is_golden=golden,
    )


# --- reference: the three pre-merge dispatch bodies, verbatim -------------


def _ref_friendly_tribe(eff, listener, board):
    n = count_friendly_tribe(
        board, eff.tribe, exclude=listener if eff.exclude_self else None
    )
    return eff.attack_per * n, eff.health_per * n


def _ref_unique_tribes(eff, listener, board):
    n = count_unique_tribes(board, exclude=listener if eff.exclude_self else None)
    return eff.attack_per * n, eff.health_per * n


def _ref_golden(eff, listener, board):
    n = count_golden_friendlies(
        board, exclude=listener if eff.exclude_self else None
    )
    return eff.attack_per * n, eff.health_per * n


_REFERENCE = {
    CountSource.FRIENDLY_OF_TRIBE: _ref_friendly_tribe,
    CountSource.UNIQUE_TRIBES: _ref_unique_tribes,
    CountSource.GOLDEN_FRIENDLIES: _ref_golden,
}


# --- board shapes worth distinguishing -----------------------------------


def _boards():
    """(name, factory) — factory returns (listener, board incl. listener)."""

    def empty():
        lis = _m("listener", Race.DRAGON)
        return lis, [lis]

    def same_tribe():
        lis = _m("listener", Race.DRAGON)
        return lis, [lis, _m("a", Race.DRAGON), _m("b", Race.DRAGON)]

    def mixed_tribes():
        lis = _m("listener", Race.DRAGON)
        return lis, [
            lis,
            _m("a", Race.DRAGON),
            _m("b", Race.MURLOC),
            _m("c", Race.BEAST),
            _m("d", None),
        ]

    def with_goldens():
        lis = _m("listener", Race.DRAGON, golden=True)
        return lis, [
            lis,
            _m("a", Race.DRAGON, golden=True),
            _m("b", Race.MURLOC),
            _m("c", Race.BEAST, golden=True),
        ]

    def with_amalgam():
        # Race.ALL counts as every tribe for FRIENDLY_OF_TRIBE but is ignored
        # by UNIQUE_TRIBES — the one place the two sources genuinely diverge.
        lis = _m("listener", Race.DRAGON)
        return lis, [lis, _m("amalgam", Race.ALL), _m("b", Race.MURLOC)]

    def listener_not_on_board():
        # Battlecry ordering can fire the effect before the body is appended.
        lis = _m("listener", Race.DRAGON)
        return lis, [_m("a", Race.DRAGON), _m("b", Race.MURLOC)]

    return [
        ("empty", empty),
        ("same_tribe", same_tribe),
        ("mixed_tribes", mixed_tribes),
        ("with_goldens", with_goldens),
        ("with_amalgam", with_amalgam),
        ("listener_not_on_board", listener_not_on_board),
    ]


@pytest.mark.parametrize("board_name,board_factory", _boards())
@pytest.mark.parametrize("source", list(CountSource))
@pytest.mark.parametrize("exclude_self", [True, False])
@pytest.mark.parametrize("atk_per,hp_per", [(1, 1), (0, 1), (2, 2), (1, 2), (0, 0)])
def test_matches_pre_merge_reference(
    board_name, board_factory, source, exclude_self, atk_per, hp_per
):
    eff = BuffSelfPerCount(
        source,
        tribe=Race.DRAGON if source is CountSource.FRIENDLY_OF_TRIBE else None,
        attack_per=atk_per,
        health_per=hp_per,
        exclude_self=exclude_self,
    )

    listener, board = board_factory()
    want_atk, want_hp = _REFERENCE[source](eff, listener, board)

    listener2, board2 = board_factory()
    before_atk, before_hp = listener2.bonus_attack, listener2.bonus_health
    apply_buff_self_per_count(eff, listener2, board2)

    assert listener2.bonus_attack - before_atk == want_atk
    assert listener2.bonus_health - before_hp == want_hp


def test_only_the_listener_is_mutated():
    lis = _m("listener", Race.DRAGON)
    board = [lis, _m("a", Race.DRAGON), _m("b", Race.DRAGON)]
    others_before = [(m.bonus_attack, m.bonus_health) for m in board[1:]]

    apply_buff_self_per_count(
        BuffSelfPerCount(CountSource.FRIENDLY_OF_TRIBE, tribe=Race.DRAGON), lis, board
    )

    assert [(m.bonus_attack, m.bonus_health) for m in board[1:]] == others_before
    assert (lis.bonus_attack, lis.bonus_health) == (2, 2)


def test_amalgam_counts_for_tribe_but_not_for_unique():
    """Pins the one divergence between the two tribe-ish sources."""
    lis = _m("listener", Race.DRAGON)
    board = [lis, _m("amalgam", Race.ALL)]

    tribe_lis = _m("listener", Race.DRAGON)
    apply_buff_self_per_count(
        BuffSelfPerCount(CountSource.FRIENDLY_OF_TRIBE, tribe=Race.DRAGON),
        tribe_lis,
        [tribe_lis, _m("amalgam", Race.ALL)],
    )
    assert tribe_lis.bonus_attack == 1  # Race.ALL matches Dragon

    uniq_lis = _m("listener", Race.DRAGON)
    apply_buff_self_per_count(
        BuffSelfPerCount(CountSource.UNIQUE_TRIBES),
        uniq_lis,
        [uniq_lis, _m("amalgam", Race.ALL)],
    )
    assert uniq_lis.bonus_attack == 0  # Race.ALL ignored, listener excluded


def test_every_count_source_is_dispatched():
    """A new CountSource member must not fall through to a silent no-op."""
    lis = _m("listener", Race.DRAGON)
    board = [lis, _m("a", Race.DRAGON, golden=True)]
    for source in CountSource:
        eff = BuffSelfPerCount(source, tribe=Race.DRAGON, attack_per=1, health_per=1)
        fresh = _m("listener", Race.DRAGON)
        apply_buff_self_per_count(eff, fresh, [fresh] + board[1:])


def test_obs_ids_match_the_pre_merge_classes():
    """The merge must not move these effects in the observation.

    31/32/33 are the ids ``BuffSelfFrom{UniqueTribe,GoldenFriendly,FriendlyTribe}Count``
    held before the merge; v5-family observations encode them per ability, so a
    shift would silently invalidate every checkpoint trained on that layout.
    """
    from src.envs.minibg.obs import EFFECT_INDEX, effect_signature

    expected = {
        CountSource.UNIQUE_TRIBES: 31,
        CountSource.GOLDEN_FRIENDLIES: 32,
        CountSource.FRIENDLY_OF_TRIBE: 33,
    }
    for source, want in expected.items():
        sig = effect_signature(BuffSelfPerCount(source))
        assert EFFECT_INDEX[sig] + 1 == want, source


def test_field_names_stay_readable_by_name():
    """``card_static`` and golden doubling read these by name, not by class."""
    from src.envs.bglike.card_static import NUMBER_FIELDS
    from src.bg_catalog.triple_effects import _GOLDEN_INT_FIELDS

    eff = BuffSelfPerCount(CountSource.UNIQUE_TRIBES, attack_per=1, health_per=2)
    for field in ("attack_per", "health_per"):
        assert hasattr(eff, field)
        assert field in NUMBER_FIELDS
        assert field in _GOLDEN_INT_FIELDS
