from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np

from .actions import (
    BOARD_SIZE,
    GOLD_AT_CAP,
    HAND_SIZE,
    MAX_ROUNDS,
    MAX_SHOP_ACTIONS,
    MAX_TIER,
    SHOP_SIZE,
    STARTING_HEALTH,
    gold_for_round,
)
from .cards import CARD_TEMPLATES
from .discover_pool import ADAPT_KEYS_ALL
from .effects import Keyword, Trigger
from .state import (
    MiniBGState,
    Minion,
    PendingChoiceKind,
    PlayerPhase,
    PlayerState,
    Race,
)


CARD_IDS_FOR_ENCODING: tuple[str, ...] = (
    "BOT_445",
    "CS2_065",
    "UNG_073",
    "EX1_507",
    "GVG_058",
    "CFM_316",
    "GVG_062",
    "GVG_027",
    "EX1_185",
    "BGS_004",
)
NUM_CARD_IDS = len(CARD_IDS_FOR_ENCODING)
CARD_ID_TO_INDEX = {cid: i for i, cid in enumerate(CARD_IDS_FOR_ENCODING)}

# Max tavern tier in patch catalog (BG); may exceed in-game MAX_TIER cap.
NUM_TIER_ONEHOT = 6

# Race one-hot: none, beast, demon, mech, murloc, all-tribes
_RACE_ORDER: tuple[Optional[Race], ...] = (
    None,
    Race.BEAST,
    Race.DEMON,
    Race.MECHANICAL,
    Race.MURLOC,
    Race.ALL,
)
RACE_ONEHOT_DIM = len(_RACE_ORDER)

# slot vector layout
#   [0]                         presence
#   [1 .. 1+NUM_CARD_IDS)       card_id one-hot (patch HS ids)
#   [1+C..1+C+NUM_TIER)         tavern tier one-hot (6 dims, tiers 1..6)
#   stats                       base/bonus atk hp (4 dims)
#   race                        one-hot (6)
#   keywords                    TAUNT, SHIELD, WINDFURY, POISONOUS, CHARGE, MAGNETIC (6)
#   runtime has_shield          (1)
#   triggers  ON_BUY..ON_TURN_START (8) + ON_FRIENDLY_SUMMON, ON_SELF_DAMAGED, ON_FRIENDLY_DIED (3)
_NUM_TRIG = 11
_C = NUM_CARD_IDS
_T0 = 1 + _C
_T1 = _T0 + NUM_TIER_ONEHOT
_S0 = _T1
_S1 = _S0 + 4
_R0 = _S1
_R1 = _R0 + RACE_ONEHOT_DIM
_K0 = _R1
_K1 = _K0 + 6
_SH = _K1
_TG0 = _SH + 1

SLOT_DIM = _TG0 + _NUM_TRIG

GLOBAL_DIM = 10
PENDING_CHOICE_DIM = 10
LAST_BATTLE_DIM = 1
HAND_LEN = HAND_SIZE
PHASE_DIM = 1
OBS_DIM = (
    GLOBAL_DIM
    + BOARD_SIZE * SLOT_DIM
    + SHOP_SIZE * SLOT_DIM
    + HAND_LEN * SLOT_DIM
    + BOARD_SIZE * SLOT_DIM
    + LAST_BATTLE_DIM
    + PHASE_DIM
    + PENDING_CHOICE_DIM
)

_STAT_NORM = 5.0


def _encode_race(m: Minion) -> np.ndarray:
    block = np.zeros(RACE_ONEHOT_DIM, dtype=np.float32)
    if m.race is None:
        block[0] = 1.0
    else:
        try:
            idx = _RACE_ORDER.index(m.race)
        except ValueError:
            block[0] = 1.0
        else:
            block[idx] = 1.0
    return block


def encode_pending_choice(me: PlayerState) -> np.ndarray:
    v = np.zeros(PENDING_CHOICE_DIM, dtype=np.float32)
    pc = me.pending_choice
    if pc is None:
        return v
    v[0] = 1.0
    v[1] = 1.0 if pc.kind == PendingChoiceKind.ADAPT else 0.0
    v[2] = min(1.0, pc.extra_modals_after / 5.0)
    for i, tok in enumerate(pc.options):
        if i >= 3:
            break
        if pc.kind == PendingChoiceKind.DISCOVER_MURLOC:
            tier = CARD_TEMPLATES[tok].tier
            v[3 + i] = tier / float(NUM_TIER_ONEHOT)
        else:
            v[3 + i] = float(ADAPT_KEYS_ALL.index(tok)) / 9.0
    return v


def encode_minion(minion: Optional[Minion]) -> np.ndarray:
    v = np.zeros(SLOT_DIM, dtype=np.float32)
    if minion is None:
        return v
    v[0] = 1.0
    cid_idx = CARD_ID_TO_INDEX.get(minion.card_id)
    if cid_idx is not None:
        v[1 + cid_idx] = 1.0
    tier = minion.tier
    if 1 <= tier <= NUM_TIER_ONEHOT:
        v[_T0 + tier - 1] = 1.0
    v[_S0] = minion.base_attack / _STAT_NORM
    v[_S0 + 1] = minion.base_health / _STAT_NORM
    v[_S0 + 2] = minion.bonus_attack / _STAT_NORM
    v[_S0 + 3] = minion.bonus_health / _STAT_NORM
    v[_R0 : _R1] = _encode_race(minion)
    kw = minion.all_keywords
    v[_K0] = 1.0 if Keyword.TAUNT in kw else 0.0
    v[_K0 + 1] = 1.0 if Keyword.SHIELD in kw else 0.0
    v[_K0 + 2] = 1.0 if Keyword.WINDFURY in kw else 0.0
    v[_K0 + 3] = 1.0 if Keyword.POISONOUS in kw else 0.0
    v[_K0 + 4] = 1.0 if Keyword.CHARGE in kw else 0.0
    v[_K0 + 5] = 1.0 if Keyword.MAGNETIC in kw else 0.0
    v[_SH] = 1.0 if minion.has_shield else 0.0
    triggers = {ab.trigger for ab in minion.abilities}
    v[_TG0] = 1.0 if Trigger.ON_BUY in triggers else 0.0
    v[_TG0 + 1] = 1.0 if Trigger.ON_DEATH in triggers else 0.0
    v[_TG0 + 2] = 1.0 if Trigger.AURA in triggers else 0.0
    v[_TG0 + 3] = 1.0 if Trigger.ON_TURN_END in triggers else 0.0
    v[_TG0 + 4] = 1.0 if Trigger.ON_PLACE in triggers else 0.0
    v[_TG0 + 5] = (
        1.0 if Trigger.AFTER_FRIENDLY_MINION_PLACED in triggers else 0.0
    )
    v[_TG0 + 6] = 1.0 if Trigger.ON_FRIENDLY_MECH_DIED in triggers else 0.0
    v[_TG0 + 8] = (
        1.0 if Trigger.ON_FRIENDLY_MINION_SUMMONED in triggers else 0.0
    )
    v[_TG0 + 9] = 1.0 if Trigger.ON_SELF_DAMAGED in triggers else 0.0
    v[_TG0 + 10] = 1.0 if Trigger.ON_FRIENDLY_MINION_DIED in triggers else 0.0
    return v


def encode_slots(
    minions: Sequence[Optional[Minion]], num_slots: int
) -> np.ndarray:
    out = np.zeros((num_slots, SLOT_DIM), dtype=np.float32)
    for i in range(min(num_slots, len(minions))):
        if minions[i] is not None:
            out[i] = encode_minion(minions[i])
    return out


def i_have_round_initiative(state: MiniBGState, player_idx: int) -> float:
    """Tie-breaker: who attacks first when boards have equal size."""
    is_odd_round = state.round_number % 2 == 1
    if is_odd_round:
        return 1.0 if state.initiative_player == player_idx else 0.0
    return 1.0 if state.initiative_player != player_idx else 0.0


def build_observation(
    state: MiniBGState,
    player_idx: int,
    last_battle_signed: float,
    enemy_last_seen_board: Optional[List[Minion]],
) -> np.ndarray:
    me = state.players[player_idx]
    enemy = state.players[1 - player_idx]

    actions_left = MAX_SHOP_ACTIONS - me.shop_actions_used

    globals_arr = np.array(
        [
            state.round_number / MAX_ROUNDS,
            me.health / STARTING_HEALTH,
            enemy.health / STARTING_HEALTH,
            me.gold / GOLD_AT_CAP,
            gold_for_round(state.round_number) / GOLD_AT_CAP,
            me.tavern_tier / MAX_TIER,
            enemy.tavern_tier / MAX_TIER,
            actions_left / MAX_SHOP_ACTIONS,
            len(me.board) / BOARD_SIZE,
            i_have_round_initiative(state, player_idx),
        ],
        dtype=np.float32,
    )

    own_board = encode_slots(list(me.board), BOARD_SIZE)
    shop = encode_slots(list(me.shop), SHOP_SIZE)
    hand = encode_slots(list(me.hand), HAND_LEN)
    enemy_board = encode_slots(
        list(enemy_last_seen_board) if enemy_last_seen_board else [],
        BOARD_SIZE,
    )
    last_battle = np.array([last_battle_signed], dtype=np.float32)
    phase_val = 1.0 if me.phase == PlayerPhase.ORDER else 0.0
    phase_arr = np.array([phase_val], dtype=np.float32)
    pending_arr = encode_pending_choice(me)

    return np.concatenate(
        [
            globals_arr,
            own_board.flatten(),
            shop.flatten(),
            hand.flatten(),
            enemy_board.flatten(),
            last_battle,
            phase_arr,
            pending_arr,
        ]
    )


__all__ = [
    "CARD_IDS_FOR_ENCODING",
    "NUM_CARD_IDS",
    "CARD_ID_TO_INDEX",
    "NUM_TIER_ONEHOT",
    "RACE_ONEHOT_DIM",
    "SLOT_DIM",
    "GLOBAL_DIM",
    "LAST_BATTLE_DIM",
    "HAND_LEN",
    "PHASE_DIM",
    "PENDING_CHOICE_DIM",
    "OBS_DIM",
    "encode_minion",
    "encode_slots",
    "i_have_round_initiative",
    "build_observation",
    "encode_pending_choice",
]
