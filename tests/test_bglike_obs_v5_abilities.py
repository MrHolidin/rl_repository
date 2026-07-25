"""Ability-token encoding in the v5 observation (`src/envs/bglike/obs_v5.py`).

Before the v5 redesign (64bff34) triggers and effects were one-hot channels in
every minibg slot, covered by `test_minibg_obs.py::test_encode_minion_ability_flags`
and a trigger assertion in `test_patch_74257.py`. That block was dropped from the
slot layout and the information moved here, as `+1`-shifted ids on a per-ability
token (0 stays free for padding). These tests carry that coverage over.

The `TRIGGER_INDEX` / `EFFECT_INDEX` registries still live in `minibg.obs`; this
module is now their only consumer.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.bg_core.effects import Ability, Trigger
from src.envs.bglike.obs_v5 import (
    ABIL_FEAT_DIM,
    ABIL_OFF_EFFECT,
    ABIL_OFF_TRIGGER,
    K_ABIL,
    NUM_EFFECT_IDS,
    NUM_TRIGGER_IDS,
    encode_ability_token,
    encode_minion_abilities,
)
from src.envs.minibg.obs import EFFECT_INDEX, TRIGGER_INDEX, _EFFECT_CLASSES
from tests.minibg_helpers import PATCH_CTX, make_minion


def _first_token(card_id: str) -> np.ndarray:
    m = make_minion(card_id)
    assert m.abilities, f"{card_id} has no abilities — bad fixture for this test"
    return encode_ability_token(m.abilities[0], PATCH_CTX)


# --- ported from test_minibg_obs.py::test_encode_minion_ability_flags ---------
# Same cards, same triggers; asserted by name instead of by one-hot position.
@pytest.mark.parametrize(
    "card_id,trigger",
    [
        ("buffer", Trigger.ON_PLACE),
        ("pack_rat", Trigger.ON_DEATH),
        ("commander", Trigger.AURA),
        ("mentor", Trigger.ON_TURN_END),
        ("wrath_weaver", Trigger.AFTER_FRIENDLY_MINION_PLACED),
        ("kangors_apprentice", Trigger.ON_DEATH),
    ],
)
def test_trigger_id_per_card(card_id: str, trigger: Trigger) -> None:
    tok = _first_token(card_id)
    assert tok[ABIL_OFF_TRIGGER] == TRIGGER_INDEX[trigger] + 1


def test_effect_id_is_registry_index_plus_one() -> None:
    m = make_minion("pack_rat")
    ab = m.abilities[0]
    tok = encode_ability_token(ab, PATCH_CTX)
    assert tok[ABIL_OFF_EFFECT] == EFFECT_INDEX[type(ab.effect)] + 1


def test_ids_are_never_the_padding_id() -> None:
    """A real ability must not encode as id 0 — padding_idx=0 in the embedding
    table returns zeros, so a collision silently masks the ability out."""
    for card_id in ("buffer", "pack_rat", "commander", "mentor", "wrath_weaver"):
        tok = _first_token(card_id)
        assert tok[ABIL_OFF_EFFECT] != 0.0
        assert tok[ABIL_OFF_TRIGGER] != 0.0


def test_every_registry_entry_fits_its_vocab() -> None:
    """Guards the embedding tables: every registered id must land in [1, vocab)."""
    for trigger, idx in TRIGGER_INDEX.items():
        assert 1 <= idx + 1 < NUM_TRIGGER_IDS, trigger
    for cls, idx in EFFECT_INDEX.items():
        assert 1 <= idx + 1 < NUM_EFFECT_IDS, cls
    assert len(EFFECT_INDEX) == len(_EFFECT_CLASSES)
    assert len(set(EFFECT_INDEX.values())) == len(EFFECT_INDEX)
    assert len(set(TRIGGER_INDEX.values())) == len(TRIGGER_INDEX)


def test_absent_minion_encodes_as_all_padding() -> None:
    block = encode_minion_abilities(None, PATCH_CTX)
    assert block.shape == (K_ABIL, ABIL_FEAT_DIM)
    assert not block.any()


def test_unregistered_effect_raises_instead_of_padding() -> None:
    """`_effect_id` fails loudly rather than collapsing to the padding id."""

    class _UnregisteredEffect:
        pass

    ab = Ability(Trigger.ON_DEATH, _UnregisteredEffect())
    with pytest.raises(KeyError, match="not in EFFECT_INDEX"):
        encode_ability_token(ab, PATCH_CTX)
