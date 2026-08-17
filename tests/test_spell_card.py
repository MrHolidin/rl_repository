"""SpellCard: construction, PatchContext loading, migrated triple-reward dispatch."""

from dataclasses import replace

from tests.conftest import PATCH_CTX

from src.bg_core.spell_card import SpellCard
from src.bg_recruitment.triples import (
    TRIPLE_REWARD_SPELL_CARD_ID,
    is_triple_reward_discover_spell,
    make_triple_reward_discover_spell,
)


def test_tavern_spell_is_not_a_minion():
    from src.bg_core.minion import Minion

    spell = SpellCard(card_id="x", name="X")
    assert not isinstance(spell, Minion)
    assert not hasattr(spell, "is_triple_reward_spell")
    assert not hasattr(spell, "base_attack")


def test_patch_context_carries_the_triple_reward_spell():
    spell = PATCH_CTX.tavern_spells[TRIPLE_REWARD_SPELL_CARD_ID]
    assert isinstance(spell, SpellCard)
    assert spell.card_id == TRIPLE_REWARD_SPELL_CARD_ID
    assert spell.triple_discover_tier == 0


def test_triple_reward_placeholder_minion_still_occupies_its_dense_index():
    # The Minion placeholder must still exist in templates/card_id_to_dense so
    # removing the hack fields didn't shift any other card's dense index.
    assert TRIPLE_REWARD_SPELL_CARD_ID in PATCH_CTX.templates
    tpl = PATCH_CTX.templates[TRIPLE_REWARD_SPELL_CARD_ID]
    assert tpl.is_token is True
    assert tpl.tier == 0
    assert TRIPLE_REWARD_SPELL_CARD_ID in PATCH_CTX.card_id_to_dense


def test_make_triple_reward_discover_spell_builds_a_tavern_spell():
    spell = make_triple_reward_discover_spell(discover_tier=3, patch=PATCH_CTX)
    assert isinstance(spell, SpellCard)
    assert spell.card_id == TRIPLE_REWARD_SPELL_CARD_ID
    assert spell.triple_discover_tier == 3
    assert is_triple_reward_discover_spell(spell)


def test_is_triple_reward_discover_spell_rejects_a_plain_minion():
    from tests.minibg_helpers import make_minion

    m = make_minion("recruit")
    assert not is_triple_reward_discover_spell(m)


def test_tavern_spell_is_frozen_and_replace_works():
    base = PATCH_CTX.tavern_spells[TRIPLE_REWARD_SPELL_CARD_ID]
    derived = replace(base, triple_discover_tier=5)
    assert base.triple_discover_tier == 0  # base untouched
    assert derived.triple_discover_tier == 5
