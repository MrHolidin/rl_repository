"""Sea Witch Zar'jira and Elemental of Surprise.

One was waiting for a spell that could name the counter, which the targeted-
spell change gave it. The other needed the triple rule itself to bend: a card
that "can triple with any Elemental" looks for a *pair* and joins it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.bg_catalog.cards import make_minion
from src.bg_catalog.patch_context import PatchContext
from src.bg_core.minion import Minion, Race
from src.bg_lobby.player import PlayerPhase, PlayerState
from src.bg_recruitment.shop_triggers import ShopTriggers
from src.bg_recruitment.spellcraft import (
    give_spellcraft_spell,
    play_spellcraft_spell_from_hand,
)
from src.bg_recruitment.triples import resolve_triples_loop

PATCH_DIR = Path("data/bgcore/36_2_0_248348")


@pytest.fixture(scope="module")
def patch() -> PatchContext:
    return PatchContext.load(PATCH_DIR)


def _card(patch, card_id):
    return make_minion(card_id, patch=patch)


def _player(patch, board=(), **kw) -> PlayerState:
    base = dict(
        health=30, gold=10, tavern_tier=6, board=list(board), shop=[None] * 7,
        hand=[None] * 10, phase=PlayerPhase.SHOP, shop_actions_used=0,
        ruleset=patch.meta.ruleset,
    )
    base.update(kw)
    return PlayerState(**base)


def _cast_sirens_song(patch, player, zarjira, *, shop_index):
    give_spellcraft_spell(player, zarjira.abilities[0].effect)
    slot = next(i for i, c in enumerate(player.hand) if c is not None)
    play_spellcraft_spell_from_hand(
        player, slot, shop_index=shop_index, patch=patch
    )


# --------------------------------------------------------------------------- #
# Sea Witch Zar'jira
# --------------------------------------------------------------------------- #


def test_sea_witch_copies_a_minion_off_the_counter(patch):
    zarjira = _card(patch, "BG27_514")
    player = _player(patch, board=[zarjira])
    offered = _card(patch, "BG25_008")  # an Eternal Knight
    player.shop[1] = offered
    _cast_sirens_song(patch, player, zarjira, shop_index=1)
    assert [c.card_id for c in player.hand if c is not None] == ["BG25_008"]


def test_the_copied_minion_stays_in_the_tavern(patch):
    """A copy, not a theft — the slot is untouched."""
    zarjira = _card(patch, "BG27_514")
    player = _player(patch, board=[zarjira])
    player.shop[0] = _card(patch, "BG25_008")
    _cast_sirens_song(patch, player, zarjira, shop_index=0)
    assert player.shop[0] is not None


def test_the_copy_is_the_printed_card(patch):
    """What the tavern body had gained does not come with it."""
    zarjira = _card(patch, "BG27_514")
    player = _player(patch, board=[zarjira])
    offered = _card(patch, "BG25_008")
    offered.bonus_attack += 10
    player.shop[0] = offered
    _cast_sirens_song(patch, player, zarjira, shop_index=0)
    copy = next(c for c in player.hand if c is not None)
    assert copy.bonus_attack == 0


def test_the_song_will_not_copy_zarjira_herself(patch):
    """"(except Sea Witch Zar'jira)" — the card says so in as many words."""
    zarjira = _card(patch, "BG27_514")
    player = _player(patch, board=[zarjira])
    player.shop[0] = _card(patch, "BG27_514")
    _cast_sirens_song(patch, player, zarjira, shop_index=0)
    assert all(c is None for c in player.hand)


def test_the_song_can_still_be_cast_at_the_board(patch):
    """A Spellcraft spell that names the counter has not lost the board."""
    zarjira = _card(patch, "BG27_514")
    naga = _card(patch, "BG23_000")
    player = _player(patch, board=[zarjira, naga])
    give_spellcraft_spell(player, naga.abilities[0].effect)
    slot = next(i for i, c in enumerate(player.hand) if c is not None)
    play_spellcraft_spell_from_hand(player, slot, 1, patch=patch)
    assert naga.raw_attack == naga.base_attack + 2


def test_golden_sea_witch_copies_twice(patch):
    (ability,) = patch.triple_merge_golden_abilities("BG27_514")
    assert ability.effect.buff.count == 2


# --------------------------------------------------------------------------- #
# Elemental of Surprise
# --------------------------------------------------------------------------- #


def test_elemental_of_surprise_completes_a_pair(patch):
    pair = [_card(patch, "BGS_115"), _card(patch, "BGS_115")]  # Sellemental
    wild = _card(patch, "BG26_175")
    player = _player(patch, board=pair + [wild])
    resolve_triples_loop(player, patch=patch)
    assert player.board == []
    merged = next(c for c in player.hand if c is not None and c.card_id == "BGS_115")
    # It joined the pair, so what comes out is the *pair's* Golden.
    assert merged.is_golden


def test_elemental_of_surprise_ignores_a_pair_of_another_tribe(patch):
    pair = [_card(patch, "BG25_008"), _card(patch, "BG25_008")]  # Undead
    wild = _card(patch, "BG26_175")
    player = _player(patch, board=pair + [wild])
    resolve_triples_loop(player, patch=patch)
    assert [m.card_id for m in player.board] == ["BG25_008", "BG25_008", "BG26_175"]


def test_a_lone_elemental_is_not_a_pair(patch):
    single = _card(patch, "BGS_115")
    wild = _card(patch, "BG26_175")
    player = _player(patch, board=[single, wild])
    resolve_triples_loop(player, patch=patch)
    assert len(player.board) == 2


def test_three_of_the_wildcard_still_triple_into_itself(patch):
    player = _player(patch, board=[_card(patch, "BG26_175") for _ in range(3)])
    resolve_triples_loop(player, patch=patch)
    merged = next(c for c in player.hand if c is not None and c.card_id == "BG26_175")
    assert merged.is_golden


def test_a_natural_triple_is_taken_before_the_wildcard_is_spent(patch):
    """Three of a kind is always the merge the seat meant."""
    natural = [_card(patch, "BGS_115") for _ in range(3)]
    wild = _card(patch, "BG26_175")
    player = _player(patch, board=natural + [wild])
    resolve_triples_loop(player, patch=patch)
    assert [m.card_id for m in player.board] == ["BG26_175"]  # not consumed


def test_the_wildcard_pairs_with_the_tavern_pool_intact(patch):
    """The pool takes back three copies of the Golden's card, not of the odd one."""
    from src.bg_lobby.shared_pool import build_initial_shared_pool

    pool = build_initial_shared_pool(patch=patch)
    pair = [_card(patch, "BGS_115"), _card(patch, "BGS_115")]
    wild = _card(patch, "BG26_175")
    before_wild = pool.remaining_copies("BG26_175")
    player = _player(patch, board=pair + [wild])
    resolve_triples_loop(player, shared_pool=pool, patch=patch)
    # The odd body went back and stayed back; nothing re-took it.
    assert pool.remaining_copies("BG26_175") == before_wild + 1


def test_a_wild_triple_carries_what_every_body_gained(patch):
    """Bonus stats are summed across all three, the odd body included."""
    pair = [_card(patch, "BGS_115"), _card(patch, "BGS_115")]
    pair[0].bonus_attack += 5
    wild = _card(patch, "BG26_175")
    wild.bonus_health += 7
    player = _player(patch, board=pair + [wild])
    resolve_triples_loop(player, patch=patch)
    merged = next(c for c in player.hand if c is not None and c.card_id == "BGS_115")
    tpl = patch.templates["BGS_115"]
    assert merged.raw_attack == tpl.base_attack * 2 + 5
    assert merged.max_health == tpl.base_health * 2 + 7


def test_a_wild_triple_does_not_carry_the_odd_bodys_printing(patch):
    """Elemental of Surprise is printed with Divine Shield; a golden
    Sellemental made with one has no business having it."""
    from src.bg_core.effects import Keyword

    pair = [_card(patch, "BGS_115"), _card(patch, "BGS_115")]
    wild = _card(patch, "BG26_175")
    assert Keyword.SHIELD in wild.all_keywords
    player = _player(patch, board=pair + [wild])
    resolve_triples_loop(player, patch=patch)
    merged = next(c for c in player.hand if c is not None and c.card_id == "BGS_115")
    assert Keyword.SHIELD not in merged.all_keywords
    assert not merged.has_shield


def test_a_wild_triple_does_carry_what_the_odd_body_was_granted(patch):
    """A keyword it picked up in play is a gain like any other."""
    from src.bg_core.effects import Keyword

    pair = [_card(patch, "BGS_115"), _card(patch, "BGS_115")]
    wild = _card(patch, "BG26_175")
    wild.granted_keywords = frozenset({Keyword.TAUNT})
    player = _player(patch, board=pair + [wild])
    resolve_triples_loop(player, patch=patch)
    merged = next(c for c in player.hand if c is not None and c.card_id == "BGS_115")
    assert Keyword.TAUNT in merged.all_keywords


def test_an_ordinary_triple_keeps_its_own_printed_keywords(patch):
    from src.bg_core.effects import Keyword

    player = _player(patch, board=[_card(patch, "BG26_175") for _ in range(3)])
    resolve_triples_loop(player, patch=patch)
    merged = next(c for c in player.hand if c is not None and c.card_id == "BG26_175")
    assert Keyword.SHIELD in merged.all_keywords and merged.has_shield


def test_an_ordinary_triple_still_carries_a_granted_keyword(patch):
    """A Taunt handed out by Houndmaster lands in ``keywords``, not
    ``granted_keywords`` — which is why the printing is subtracted rather than
    one field trusted to hold every gain."""
    from src.bg_core.effects import Keyword

    bodies = [_card(patch, "BGS_115") for _ in range(3)]
    bodies[1].keywords = frozenset(bodies[1].keywords | {Keyword.TAUNT})
    player = _player(patch, board=bodies)
    resolve_triples_loop(player, patch=patch)
    merged = next(c for c in player.hand if c is not None and c.card_id == "BGS_115")
    assert Keyword.TAUNT in merged.all_keywords


def test_a_golden_elemental_does_not_complete_a_pair(patch):
    """Only a body that could triple at all is a candidate."""
    pair = [_card(patch, "BGS_115"), _card(patch, "BGS_115")]
    wild = _card(patch, "BG26_175")
    wild.is_golden = True
    player = _player(patch, board=pair + [wild])
    resolve_triples_loop(player, patch=patch)
    assert len(player.board) == 3
