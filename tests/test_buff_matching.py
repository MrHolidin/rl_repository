"""``BuffMatching`` reproduces the four classes it replaced, exhaustively.

Same reasoning as ``test_buff_self_per_count``: the golden trace is an
integration check and cannot prove four independent branches are each
equivalent. This pins the predicate directly against a reference transcribed
from the pre-merge bodies, and separately pins the two behaviours that the
merge could most easily have "fixed" by accident:

* the deathrattle table never held ``BuffAllOtherOfTribe``, so an ON_DEATH
  ability carrying that target is a silent no-op in combat (BGS_030 ships
  one), and
* two combat call sites were reachable by exactly one of the four classes.
"""

from __future__ import annotations

import pytest

from src.bg_core.board_helpers import buff_matching_hits, minion_matches_tribe
from src.bg_core.effects import BuffMatching, BuffTarget, Keyword
from src.bg_core.minion import Minion, Race


def _m(card_id: str, race=None, *, keywords=frozenset()) -> Minion:
    return Minion(
        card_id=card_id,
        base_attack=1,
        base_health=1,
        tier=1,
        race=race,
        keywords=frozenset(keywords),
    )


# --- reference: the four pre-merge predicates, verbatim -------------------


def _ref_all_friendly(eff, cand, source):
    return True


def _ref_friendly_of_tribe(eff, cand, source):
    return minion_matches_tribe(cand, eff.tribe)


def _ref_other_of_tribe(eff, cand, source):
    if cand is source:
        return False
    return minion_matches_tribe(cand, eff.tribe)


def _ref_with_keyword(eff, cand, source):
    return eff.keyword in cand.all_keywords


_REFERENCE = {
    BuffTarget.ALL_FRIENDLY: _ref_all_friendly,
    BuffTarget.FRIENDLY_OF_TRIBE: _ref_friendly_of_tribe,
    BuffTarget.OTHER_OF_TRIBE: _ref_other_of_tribe,
    BuffTarget.FRIENDLY_WITH_KEYWORD: _ref_with_keyword,
}


def _candidates():
    return [
        ("no_race", _m("a")),
        ("dragon", _m("b", Race.DRAGON)),
        ("murloc", _m("c", Race.MURLOC)),
        ("amalgam", _m("d", Race.ALL)),
        ("taunt_dragon", _m("e", Race.DRAGON, keywords={Keyword.TAUNT})),
        ("taunt_no_race", _m("f", keywords={Keyword.TAUNT})),
        ("shield_murloc", _m("g", Race.MURLOC, keywords={Keyword.SHIELD})),
    ]


@pytest.mark.parametrize("target", list(BuffTarget))
@pytest.mark.parametrize("cand_name,candidate", _candidates())
@pytest.mark.parametrize("tribe", [Race.DRAGON, Race.MURLOC, Race.ALL])
@pytest.mark.parametrize("keyword", [Keyword.TAUNT, Keyword.SHIELD])
@pytest.mark.parametrize("is_source", [True, False])
def test_predicate_matches_pre_merge_reference(
    target, cand_name, candidate, tribe, keyword, is_source
):
    eff = BuffMatching(target, attack=1, health=1, tribe=tribe, keyword=keyword)
    source = candidate if is_source else _m("src", Race.BEAST)
    assert buff_matching_hits(eff, candidate, source) == _REFERENCE[target](
        eff, candidate, source
    )


def test_source_exclusion_only_applies_to_other_of_tribe():
    src = _m("src", Race.DRAGON)
    for target, expected_when_source in (
        (BuffTarget.ALL_FRIENDLY, True),
        (BuffTarget.FRIENDLY_OF_TRIBE, True),
        (BuffTarget.OTHER_OF_TRIBE, False),
    ):
        eff = BuffMatching(target, tribe=Race.DRAGON)
        assert buff_matching_hits(eff, src, src) is expected_when_source, target


def test_every_target_is_dispatched():
    """A new BuffTarget member must not silently fall through."""
    cand = _m("c", Race.DRAGON, keywords={Keyword.TAUNT})
    for target in BuffTarget:
        eff = BuffMatching(target, tribe=Race.DRAGON, keyword=Keyword.TAUNT)
        buff_matching_hits(eff, cand, None)


def test_king_bagurgle_deathrattle_fires():
    """BGS_030 = King Bagurgle, 'Battlecry **and Deathrattle**: +2/+2 to your
    other Murlocs'. The deathrattle half used to be silently dropped (its
    effect class had no ``_DEATHRATTLE_HANDLERS`` entry and the dispatch skipped
    misses without a word), so half the card did nothing."""
    import numpy as np

    from src.bg_catalog.cards import make_minion
    from src.bg_catalog.patch_context import load_patch_context
    from src.bg_combat.battle import simulate_battle

    ctx = load_patch_context("data/bgcore/19_6_0_74257")
    bagurgle = make_minion("BGS_030", patch=ctx)
    bagurgle.base_health = 1  # dies to the first swing
    allies = [make_minion("EX1_506", patch=ctx) for _ in range(2)]
    for m in allies:
        m.bonus_health += 60  # outlive the fight so the buff is observable
    before = [(m.raw_attack, m.max_health) for m in allies]

    enemy = [make_minion("EX1_506", patch=ctx)]
    enemy[0].bonus_attack += 5  # kills Bagurgle, cannot kill the others

    survivors: list = []
    simulate_battle(
        [bagurgle] + allies,
        enemy,
        p0_has_initiative=False,
        rng=np.random.default_rng(0),
        combat_board_max=7,
        damage_cap=15,
        max_board_slots=7,
        patch=ctx,
        p0_board_out=survivors,
    )

    assert before == [(2, 61), (2, 61)]
    assert [(m.raw_attack, m.max_health) for m in survivors] == [(4, 63), (4, 63)]


def test_obs_ids_match_the_pre_merge_classes():
    """43/44/45/46 are the ids the four classes held before the merge."""
    from src.envs.minibg.obs import EFFECT_INDEX, effect_signature

    expected = {
        BuffTarget.ALL_FRIENDLY: 43,
        BuffTarget.FRIENDLY_OF_TRIBE: 44,
        BuffTarget.OTHER_OF_TRIBE: 45,
        BuffTarget.FRIENDLY_WITH_KEYWORD: 46,
    }
    for target, want in expected.items():
        sig = effect_signature(BuffMatching(target))
        assert EFFECT_INDEX[sig] + 1 == want, target


def test_field_names_stay_readable_by_name():
    """card_static, golden doubling and the v5 encoder read these by name."""
    from src.envs.bglike.card_static import NUMBER_FIELDS
    from src.bg_catalog.triple_effects import _GOLDEN_INT_FIELDS

    eff = BuffMatching(BuffTarget.FRIENDLY_OF_TRIBE, attack=2, health=2, tribe=Race.DRAGON)
    for field in ("attack", "health"):
        assert hasattr(eff, field)
        assert field in NUMBER_FIELDS
        assert field in _GOLDEN_INT_FIELDS
    # probed by name by obs_v5's _effect_tribe_id / _keyword_id
    assert hasattr(eff, "tribe") and hasattr(eff, "keyword")


def test_unused_categorical_fields_read_as_absent():
    """A variant that doesn't use tribe/keyword must encode like the old class.

    The old classes simply had no such attribute; obs_v5 probes with getattr
    and treats a missing attribute as 0, so the merged default must be None
    (not, say, Race.BEAST) or the v5 obs would shift.
    """
    from src.envs.bglike.obs_v5 import _effect_tribe_id, _keyword_id

    all_friendly = BuffMatching(BuffTarget.ALL_FRIENDLY, attack=1, health=1)
    assert _effect_tribe_id(all_friendly) == 0
    assert _keyword_id(getattr(all_friendly, "keyword", None)) == 0
