from __future__ import annotations

from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .actions import (
    BOARD_SIZE,
    GOLD_AT_CAP,
    HAND_SIZE,
    LEVEL_UP_COST_MAX,
    MAX_ROUNDS,
    MAX_SHOP_ACTIONS,
    MAX_SHOP_SLOTS,
    MAX_TIER,
    STARTING_HEALTH,
    gold_for_round,
)
from src.bg_catalog.cards import normalize_shop_excluded_races
from src.bg_catalog.patch_context import PatchContext
from src.bg_core.effects import (
    AdaptAllMurlocsEffect,
    AdaptSelfRandomEffect,
    AdjacentStatAura,
    AddFromLastOpponentBoardEffect,
    AddRandomMinionToShopEffect,
    AddTokenToHandEffect,
    AttackBonusPerOtherMurlocGlobal,
    AttackImmediatelyAfterSurvivingEffect,
    BattlecryMultiplierAura,
    BuffAllShopOffersEffect,
    BuffRandomFriendlyFromPlacedTierEffect,
    BuffSelfFromFriendlyTribeCount,
    BuffSelfFromGoldenFriendlyCount,
    BuffSelfFromUniqueTribeCount,
    BuffSelfWhenFriendlyDeathrattlePlaced,
    BuffSummonedIfRace,
    BuffTargetFriendlyBattlecry,
    CleaveOnAttack,
    ConsumeFriendlyBattlecry,
    DealExcessDamageToAdjacentEffect,
    DeathrattleMultiplierAura,
    DiscoverMurlocEffect,
    GainGoldOnDeathEffect,
    GainGoldThisTurnEffect,
    GrantKeywordAllFriendlyOfTribe,
    HeroImmuneAura,
    IncrementShopTribeBonusEffect,
    Keyword,
    KeywordStatAura,
    MultiplySelfAttackEffect,
    PogoHopperBattlecry,
    ReduceUpgradeCostEffect,
    SetNextRollCostEffect,
    StartOfCombatDamagePerFriendlyTribe,
    SummonMultiplierAura,
    TransformIntoShopMinionEffect,
    TribalOtherStatAura,
    Trigger,
    TriggerRandomFriendlyDeathrattleEffect,
    ZappTargeting,
)
from src.bg_recruitment.discover_pool import ADAPT_KEYS_ALL
from .state import (
    MiniBGState,
    Minion,
    PendingChoiceKind,
    PlayerPhase,
    PlayerState,
    Race,
)


def _resolve_card_id_to_dense(
    card_id_to_dense: Optional[Mapping[str, int]] = None,
) -> Mapping[str, int]:
    if card_id_to_dense is None:
        raise ValueError(
            "card_id_to_dense is required (pass game._patch.card_id_to_dense)"
        )
    return card_id_to_dense


CARD_INDEX_EMPTY = 0

NUM_TIER_ONEHOT = 6

_RACE_ORDER: Tuple[Optional[Race], ...] = (
    None,
    Race.BEAST,
    Race.DEMON,
    Race.MECHANICAL,
    Race.MURLOC,
    Race.DRAGON,
    Race.PIRATE,
    Race.ELEMENTAL,
    Race.ALL,
)
RACE_ONEHOT_DIM = len(_RACE_ORDER)

NUM_KEYWORD_CHANNELS = 8  # TAUNT, SHIELD, WINDFURY, POISONOUS, CHARGE, MAGNETIC, REBORN, MEGA_WINDFURY
NUM_TRIGGER_CHANNELS = 21  # see TRIGGER_INDEX below
NUM_EFFECT_CHANNELS = 38  # see EFFECT_INDEX below

# Slot layout offsets (single source of truth — tests / nets pull these in directly).
PRESENCE_OFFSET = 0
CARD_IDX_OFFSET = 1
TIER_OFFSET = CARD_IDX_OFFSET + 1
STATS_OFFSET = TIER_OFFSET + NUM_TIER_ONEHOT
RACE_OFFSET = STATS_OFFSET + 4
KEYWORD_OFFSET = RACE_OFFSET + RACE_ONEHOT_DIM
SHIELD_OFFSET = KEYWORD_OFFSET + NUM_KEYWORD_CHANNELS
GOLDEN_OFFSET = SHIELD_OFFSET + 1
TRIGGER_OFFSET = GOLDEN_OFFSET + 1
EFFECT_OFFSET = TRIGGER_OFFSET + NUM_TRIGGER_CHANNELS
# Non-golden copies of the same ``card_id`` elsewhere (triple / buy planning); normalized.
HAND_SAME_CARD_COUNT_OFFSET = EFFECT_OFFSET + NUM_EFFECT_CHANNELS
BOARD_SAME_CARD_COUNT_OFFSET = HAND_SAME_CARD_COUNT_OFFSET + 1
TRIPLE_REWARD_SPELL_OFFSET = BOARD_SAME_CARD_COUNT_OFFSET + 1
TRIPLE_DISCOVER_TIER_OFFSET = TRIPLE_REWARD_SPELL_OFFSET + 1
FROZEN_OFFSET = TRIPLE_DISCOVER_TIER_OFFSET + 1
SLOT_DIM = FROZEN_OFFSET + 1

# Trigger channel mapping (preserve historical positions 0..6, 8..10; fill 7=ON_TURN_START, add ON_OVERKILL).
TRIGGER_INDEX: Dict[Trigger, int] = {
    Trigger.ON_BUY: 0,
    Trigger.ON_DEATH: 1,
    Trigger.AURA: 2,
    Trigger.ON_TURN_END: 3,
    Trigger.ON_PLACE: 4,
    Trigger.AFTER_FRIENDLY_MINION_PLACED: 5,
    Trigger.ON_FRIENDLY_MECH_DIED: 6,
    Trigger.ON_TURN_START: 7,
    Trigger.ON_FRIENDLY_MINION_SUMMONED: 8,
    Trigger.ON_SELF_DAMAGED: 9,
    Trigger.ON_FRIENDLY_MINION_DIED: 10,
    Trigger.ON_OVERKILL: 11,
    Trigger.ON_START_OF_COMBAT: 12,
    Trigger.ON_SELL: 13,
    Trigger.ON_FRIENDLY_BOUGHT: 14,
    Trigger.ON_AFTER_ATTACK: 15,
    Trigger.ON_FRIENDLY_ATTACK: 16,
    Trigger.ON_SURVIVED_ATTACK: 17,
    Trigger.ON_FRIENDLY_SHIELD_LOST: 18,
    Trigger.ON_WHEN_ATTACKED: 19,
    Trigger.ON_FRIENDLY_WHEN_ATTACKED: 20,
}
assert len(TRIGGER_INDEX) == NUM_TRIGGER_CHANNELS

# Effect markers — high-impact aura/multiplier/listener types the network should distinguish
# beyond the trigger fan-out (Brann/Khadgar/Baron etc. all share AURA trigger but differ wildly).
_EFFECT_CLASSES: Tuple[type, ...] = (
    # --- original 15 (preserved in order for checkpoint compat on minibg) ---
    BattlecryMultiplierAura,
    DeathrattleMultiplierAura,
    SummonMultiplierAura,
    HeroImmuneAura,
    CleaveOnAttack,
    ZappTargeting,
    AdjacentStatAura,
    TribalOtherStatAura,
    KeywordStatAura,
    AttackBonusPerOtherMurlocGlobal,
    BuffSummonedIfRace,
    PogoHopperBattlecry,
    DiscoverMurlocEffect,
    AdaptAllMurlocsEffect,
    BuffTargetFriendlyBattlecry,
    # --- patch 74257 additions ---
    # economy / shop
    GainGoldThisTurnEffect,
    GainGoldOnDeathEffect,
    SetNextRollCostEffect,
    ReduceUpgradeCostEffect,
    IncrementShopTribeBonusEffect,
    BuffAllShopOffersEffect,
    AddTokenToHandEffect,
    # shop manipulation
    TransformIntoShopMinionEffect,
    AddFromLastOpponentBoardEffect,
    AddRandomMinionToShopEffect,
    # combat unique mechanics
    MultiplySelfAttackEffect,
    TriggerRandomFriendlyDeathrattleEffect,
    AttackImmediatelyAfterSurvivingEffect,
    DealExcessDamageToAdjacentEffect,
    StartOfCombatDamagePerFriendlyTribe,
    # scaling / synergy
    BuffSelfFromUniqueTribeCount,
    BuffSelfFromGoldenFriendlyCount,
    BuffSelfFromFriendlyTribeCount,
    ConsumeFriendlyBattlecry,
    AdaptSelfRandomEffect,
    BuffRandomFriendlyFromPlacedTierEffect,
    # shop listeners
    BuffSelfWhenFriendlyDeathrattlePlaced,
    GrantKeywordAllFriendlyOfTribe,
)
assert len(_EFFECT_CLASSES) == NUM_EFFECT_CHANNELS
EFFECT_INDEX: Dict[type, int] = {cls: i for i, cls in enumerate(_EFFECT_CLASSES)}

GLOBAL_CORE_DIM = 11
# Max rotation tribes across supported patches (74257: 7 + 1 ratio slot).
SHOP_ROTATION_OBS_DIM = 8
GLOBAL_DIM = GLOBAL_CORE_DIM + SHOP_ROTATION_OBS_DIM

# Pending-choice layout (single block at obs tail):
#   [0]   has_pending
#   [1]   is_adapt (vs discover_murloc / transform)
#   [2]   extra_modals_after / 5
#   [3]   is_transform (TRANSFORM_SHOP_MINION modal; legal actions become BUY not DISCOVER_PICK)
#   [4..7) adapt option payload (key_idx/9 for adapt; 0 for discover/transform)
#   [7..10) discover option card_idx (dense pool index; 0 for adapt/transform)
# The 3 discover indices share the slot card embedding via the network's `self.card_emb`.
PENDING_HEADER_DIM = 4
PENDING_OPTIONS_DIM = 3
PENDING_DISCOVER_IDX_DIM = 3
PENDING_ELIGIBLE_DIM = BOARD_SIZE
PENDING_HEADER_OFFSET = 0
PENDING_OPTIONS_OFFSET = PENDING_HEADER_OFFSET + PENDING_HEADER_DIM
PENDING_DISCOVER_IDX_OFFSET = PENDING_OPTIONS_OFFSET + PENDING_OPTIONS_DIM
# RL pending apply-effect block (see ``MiniBGEnv._rl_pending``): kind + scalars + mask.
PENDING_IS_APPLY_OFFSET = PENDING_DISCOVER_IDX_OFFSET + PENDING_DISCOVER_IDX_DIM
PENDING_EFFECT_KIND_OFFSET = PENDING_IS_APPLY_OFFSET + 1
PENDING_EFFECT_ATK_OFFSET = PENDING_EFFECT_KIND_OFFSET + 1
PENDING_EFFECT_HP_OFFSET = PENDING_EFFECT_ATK_OFFSET + 1
PENDING_EFFECT_TAUNT_OFFSET = PENDING_EFFECT_HP_OFFSET + 1
PENDING_CAN_SKIP_OFFSET = PENDING_EFFECT_TAUNT_OFFSET + 1
PENDING_APPLY_REMAINING_OFFSET = PENDING_CAN_SKIP_OFFSET + 1
PENDING_PICKED_OFFSET = PENDING_APPLY_REMAINING_OFFSET + 1
PENDING_ELIGIBLE_OFFSET = PENDING_PICKED_OFFSET + PENDING_ELIGIBLE_DIM
PENDING_APPLY_SECTION_DIM = (
    1 + 5 + 1 + PENDING_ELIGIBLE_DIM + PENDING_ELIGIBLE_DIM
)
PENDING_CHOICE_DIM = (
    PENDING_HEADER_DIM
    + PENDING_OPTIONS_DIM
    + PENDING_DISCOVER_IDX_DIM
    + PENDING_APPLY_SECTION_DIM
)

LAST_BATTLE_DIM = 1
HAND_LEN = HAND_SIZE
PHASE_DIM = 1
OBS_DIM = (
    GLOBAL_DIM
    + BOARD_SIZE * SLOT_DIM
    + MAX_SHOP_SLOTS * SLOT_DIM
    + HAND_LEN * SLOT_DIM
    + BOARD_SIZE * SLOT_DIM
    + LAST_BATTLE_DIM
    + PHASE_DIM
    + PENDING_CHOICE_DIM
)

_STAT_NORM = 5.0


def _encode_shop_rotation_globals(
    shop_excluded_race: Optional[Race | tuple[Race, ...]],
    *,
    rotation_tribes: Sequence[Race],
    cnt_active_shop_tribes: int,
) -> np.ndarray:
    """Tribe one-hot in first ``SHOP_ROTATION_OBS_DIM - 1`` slots; last slot = active/total."""
    v = np.zeros(SHOP_ROTATION_OBS_DIM, dtype=np.float32)
    tribe_slots = SHOP_ROTATION_OBS_DIM - 1
    n_rot = len(rotation_tribes)
    excluded = normalize_shop_excluded_races(shop_excluded_race)
    if not excluded:
        v[tribe_slots] = 1.0
        return v
    for excluded_race in excluded:
        for i, r in enumerate(rotation_tribes):
            if i >= tribe_slots:
                break
            if r == excluded_race:
                v[i] = 1.0
                break
    if n_rot > 0:
        v[tribe_slots] = float(cnt_active_shop_tribes) / float(n_rot)
    else:
        v[tribe_slots] = 1.0
    return v


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


def encode_pending_choice(
    me: PlayerState,
    rl_pending: Optional["RlPendingEffect"] = None,
    *,
    card_id_to_dense: Optional[Mapping[str, int]] = None,
) -> np.ndarray:
    dense = _resolve_card_id_to_dense(card_id_to_dense)
    v = np.zeros(PENDING_CHOICE_DIM, dtype=np.float32)
    if rl_pending is not None:
        from src.bg_core.effects import (
            BuffAdjacentBattlecry,
            BuffTargetFriendlyBattlecry,
        )
        from src.envs.minibg.rl_effects import RlEffectKind

        v[PENDING_HEADER_OFFSET] = 1.0
        total = rl_pending.apply_steps_total
        rem_norm = float(rl_pending.remaining) / float(total) if total > 0 else 0.0
        v[PENDING_HEADER_OFFSET + 2] = rem_norm
        v[PENDING_IS_APPLY_OFFSET] = 1.0
        n_kinds = len(RlEffectKind)
        if n_kinds <= 1:
            v[PENDING_EFFECT_KIND_OFFSET] = 0.0
        else:
            v[PENDING_EFFECT_KIND_OFFSET] = float(rl_pending.kind.value) / float(
                n_kinds - 1
            )
        if isinstance(rl_pending.params, BuffTargetFriendlyBattlecry):
            v[PENDING_EFFECT_ATK_OFFSET] = float(rl_pending.params.attack) / _STAT_NORM
            v[PENDING_EFFECT_HP_OFFSET] = float(rl_pending.params.health) / _STAT_NORM
        elif isinstance(rl_pending.params, BuffAdjacentBattlecry):
            v[PENDING_EFFECT_ATK_OFFSET] = float(rl_pending.params.attack) / _STAT_NORM
            v[PENDING_EFFECT_HP_OFFSET] = float(rl_pending.params.health) / _STAT_NORM
            if rl_pending.params.grant_taunt:
                v[PENDING_EFFECT_TAUNT_OFFSET] = 1.0
        if rl_pending.can_skip_second_adjacent():
            v[PENDING_CAN_SKIP_OFFSET] = 1.0
        v[PENDING_APPLY_REMAINING_OFFSET] = rem_norm
        for i in rl_pending.picks:
            if 0 <= i < BOARD_SIZE:
                v[PENDING_PICKED_OFFSET + i] = 1.0
        for i in rl_pending.eligible_on_board_live(me.board):
            if 0 <= i < BOARD_SIZE:
                v[PENDING_ELIGIBLE_OFFSET + i] = 1.0
        return v

    pc = me.pending_choice
    if pc is None:
        return v
    is_adapt = pc.kind == PendingChoiceKind.ADAPT
    is_transform = pc.kind == PendingChoiceKind.TRANSFORM_SHOP_MINION
    v[PENDING_HEADER_OFFSET] = 1.0
    v[PENDING_HEADER_OFFSET + 1] = 1.0 if is_adapt else 0.0
    v[PENDING_HEADER_OFFSET + 2] = min(1.0, pc.extra_modals_after / 5.0)
    v[PENDING_HEADER_OFFSET + 3] = 1.0 if is_transform else 0.0
    for i, tok in enumerate(pc.options):
        if i >= 3:
            break
        if is_adapt:
            v[PENDING_OPTIONS_OFFSET + i] = float(ADAPT_KEYS_ALL.index(tok)) / 9.0
        else:
            v[PENDING_DISCOVER_IDX_OFFSET + i] = float(
                dense.get(tok, CARD_INDEX_EMPTY)
            )
    return v


def _count_non_golden_same_card_hand(
    player: PlayerState,
    card_id: str,
    *,
    exclude_hand_idx: Optional[int] = None,
) -> int:
    n = 0
    for i, hm in enumerate(player.hand):
        if hm is None:
            continue
        if exclude_hand_idx is not None and i == exclude_hand_idx:
            continue
        if not hm.is_golden and hm.card_id == card_id:
            n += 1
    return n


def _count_non_golden_same_card_board(
    player: PlayerState,
    card_id: str,
    *,
    exclude_board_idx: Optional[int] = None,
) -> int:
    n = 0
    for i, m in enumerate(player.board):
        if exclude_board_idx is not None and i == exclude_board_idx:
            continue
        if not m.is_golden and m.card_id == card_id:
            n += 1
    return n


def _count_non_golden_same_card_enemy_board(
    board: Sequence[Minion], idx: int, card_id: str
) -> int:
    return sum(
        1
        for j, m in enumerate(board)
        if j != idx and not m.is_golden and m.card_id == card_id
    )


def encode_minion(
    minion: Optional[Minion],
    *,
    same_non_golden_hand_elsewhere: int = 0,
    same_non_golden_board_elsewhere: int = 0,
    card_id_to_dense: Optional[Mapping[str, int]] = None,
    is_frozen: bool = False,
) -> np.ndarray:
    dense = _resolve_card_id_to_dense(card_id_to_dense)
    v = np.zeros(SLOT_DIM, dtype=np.float32)
    if minion is None:
        return v
    v[PRESENCE_OFFSET] = 1.0

    # Dense card index — networks gather an nn.Embedding row from this channel.
    # Unknown card_ids (e.g. test fixtures outside patch templates) collapse to 0.
    v[CARD_IDX_OFFSET] = float(dense.get(minion.card_id, CARD_INDEX_EMPTY))

    tier = minion.tier
    if 1 <= tier <= NUM_TIER_ONEHOT:
        v[TIER_OFFSET + tier - 1] = 1.0

    v[STATS_OFFSET] = minion.base_attack / _STAT_NORM
    v[STATS_OFFSET + 1] = minion.base_health / _STAT_NORM
    v[STATS_OFFSET + 2] = minion.bonus_attack / _STAT_NORM
    v[STATS_OFFSET + 3] = minion.bonus_health / _STAT_NORM

    v[RACE_OFFSET : RACE_OFFSET + RACE_ONEHOT_DIM] = _encode_race(minion)

    kw = minion.all_keywords
    v[KEYWORD_OFFSET] = 1.0 if Keyword.TAUNT in kw else 0.0
    v[KEYWORD_OFFSET + 1] = 1.0 if Keyword.SHIELD in kw else 0.0
    v[KEYWORD_OFFSET + 2] = 1.0 if Keyword.WINDFURY in kw else 0.0
    v[KEYWORD_OFFSET + 3] = 1.0 if Keyword.POISONOUS in kw else 0.0
    v[KEYWORD_OFFSET + 4] = 1.0 if Keyword.CHARGE in kw else 0.0
    v[KEYWORD_OFFSET + 5] = 1.0 if Keyword.MAGNETIC in kw else 0.0
    v[KEYWORD_OFFSET + 6] = 1.0 if Keyword.REBORN in kw else 0.0
    v[KEYWORD_OFFSET + 7] = 1.0 if Keyword.MEGA_WINDFURY in kw else 0.0

    v[SHIELD_OFFSET] = 1.0 if minion.has_shield else 0.0
    v[GOLDEN_OFFSET] = 1.0 if minion.is_golden else 0.0

    for ab in minion.abilities:
        ti = TRIGGER_INDEX.get(ab.trigger)
        if ti is not None:
            v[TRIGGER_OFFSET + ti] = 1.0
        ei = EFFECT_INDEX.get(type(ab.effect))
        if ei is not None:
            v[EFFECT_OFFSET + ei] = 1.0
    v[HAND_SAME_CARD_COUNT_OFFSET] = float(same_non_golden_hand_elsewhere) / float(
        HAND_SIZE
    )
    v[BOARD_SAME_CARD_COUNT_OFFSET] = float(same_non_golden_board_elsewhere) / float(
        BOARD_SIZE
    )
    if minion.is_triple_reward_spell:
        v[TRIPLE_REWARD_SPELL_OFFSET] = 1.0
        tier = minion.triple_discover_tier
        if tier > 0:
            v[TRIPLE_DISCOVER_TIER_OFFSET] = float(tier) / float(MAX_TIER)
    v[FROZEN_OFFSET] = 1.0 if is_frozen else 0.0
    return v


def encode_slots(
    minions: Sequence[Optional[Minion]],
    num_slots: int,
    *,
    card_id_to_dense: Optional[Mapping[str, int]] = None,
) -> np.ndarray:
    out = np.zeros((num_slots, SLOT_DIM), dtype=np.float32)
    for i in range(min(num_slots, len(minions))):
        if minions[i] is not None:
            out[i] = encode_minion(minions[i], card_id_to_dense=card_id_to_dense)
    return out


def _encode_own_board_with_pair_counts(
    player: PlayerState,
    *,
    card_id_to_dense: Optional[Mapping[str, int]] = None,
) -> np.ndarray:
    out = np.zeros((BOARD_SIZE, SLOT_DIM), dtype=np.float32)
    for i, m in enumerate(player.board):
        nh = _count_non_golden_same_card_hand(player, m.card_id)
        nb = _count_non_golden_same_card_board(
            player, m.card_id, exclude_board_idx=i
        )
        out[i] = encode_minion(
            m,
            same_non_golden_hand_elsewhere=nh,
            same_non_golden_board_elsewhere=nb,
            card_id_to_dense=card_id_to_dense,
        )
    return out


def _encode_hand_with_pair_counts(
    player: PlayerState,
    *,
    card_id_to_dense: Optional[Mapping[str, int]] = None,
) -> np.ndarray:
    out = np.zeros((HAND_LEN, SLOT_DIM), dtype=np.float32)
    for i, hm in enumerate(player.hand):
        if hm is None:
            continue
        nh = _count_non_golden_same_card_hand(
            player, hm.card_id, exclude_hand_idx=i
        )
        nb = _count_non_golden_same_card_board(player, hm.card_id)
        out[i] = encode_minion(
            hm,
            same_non_golden_hand_elsewhere=nh,
            same_non_golden_board_elsewhere=nb,
            card_id_to_dense=card_id_to_dense,
        )
    return out


def _encode_shop_with_pair_counts(
    player: PlayerState,
    *,
    card_id_to_dense: Optional[Mapping[str, int]] = None,
) -> np.ndarray:
    out = np.zeros((MAX_SHOP_SLOTS, SLOT_DIM), dtype=np.float32)
    for i, m in enumerate(player.shop):
        if m is None or i >= MAX_SHOP_SLOTS:
            continue
        nh = _count_non_golden_same_card_hand(player, m.card_id)
        nb = _count_non_golden_same_card_board(player, m.card_id)
        out[i] = encode_minion(
            m,
            same_non_golden_hand_elsewhere=nh,
            same_non_golden_board_elsewhere=nb,
            card_id_to_dense=card_id_to_dense,
        )
    return out


def _encode_enemy_board_with_pair_counts(
    board: List[Minion],
    *,
    card_id_to_dense: Optional[Mapping[str, int]] = None,
) -> np.ndarray:
    out = np.zeros((BOARD_SIZE, SLOT_DIM), dtype=np.float32)
    for i, m in enumerate(board):
        if i >= BOARD_SIZE:
            break
        n_same = _count_non_golden_same_card_enemy_board(board, i, m.card_id)
        out[i] = encode_minion(
            m,
            same_non_golden_hand_elsewhere=0,
            same_non_golden_board_elsewhere=n_same,
            card_id_to_dense=card_id_to_dense,
        )
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
    *,
    patch: PatchContext,
    rl_pending: Optional["RlPendingEffect"] = None,
) -> np.ndarray:
    me = state.players[player_idx]
    enemy = state.players[1 - player_idx]
    card_id_to_dense = patch.card_id_to_dense
    meta = patch.meta

    actions_left = MAX_SHOP_ACTIONS - me.shop_actions_used
    tier_up_cost = (
        0.0 if me.tavern_tier >= MAX_TIER else float(me.next_tier_up_cost)
    )

    globals_core = np.array(
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
            tier_up_cost / LEVEL_UP_COST_MAX,
        ],
        dtype=np.float32,
    )
    globals_arr = np.concatenate(
        [
            globals_core,
            _encode_shop_rotation_globals(
                state.shop_excluded_race,
                rotation_tribes=meta.rotation_tribes,
                cnt_active_shop_tribes=max(
                    0,
                    len(meta.rotation_tribes)
                    - len(normalize_shop_excluded_races(state.shop_excluded_race)),
                ),
            ),
        ]
    )

    own_board = _encode_own_board_with_pair_counts(me, card_id_to_dense=card_id_to_dense)
    shop = _encode_shop_with_pair_counts(me, card_id_to_dense=card_id_to_dense)
    hand = _encode_hand_with_pair_counts(me, card_id_to_dense=card_id_to_dense)
    enemy_board = (
        _encode_enemy_board_with_pair_counts(
            list(enemy_last_seen_board), card_id_to_dense=card_id_to_dense
        )
        if enemy_last_seen_board
        else np.zeros((BOARD_SIZE, SLOT_DIM), dtype=np.float32)
    )
    last_battle = np.array([last_battle_signed], dtype=np.float32)
    # 1.0 when shop action budget is exhausted but the player may still swap / finish.
    phase_val = (
        1.0
        if me.phase == PlayerPhase.SHOP and me.shop_actions_used >= MAX_SHOP_ACTIONS
        else 0.0
    )
    phase_arr = np.array([phase_val], dtype=np.float32)
    pending_arr = encode_pending_choice(
        me, rl_pending=rl_pending, card_id_to_dense=card_id_to_dense
    )

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
    "CARD_INDEX_EMPTY",
    "NUM_TIER_ONEHOT",
    "RACE_ONEHOT_DIM",
    "NUM_KEYWORD_CHANNELS",
    "NUM_TRIGGER_CHANNELS",
    "NUM_EFFECT_CHANNELS",
    "TRIGGER_INDEX",
    "EFFECT_INDEX",
    "PRESENCE_OFFSET",
    "CARD_IDX_OFFSET",
    "TIER_OFFSET",
    "STATS_OFFSET",
    "RACE_OFFSET",
    "KEYWORD_OFFSET",
    "SHIELD_OFFSET",
    "GOLDEN_OFFSET",
    "TRIGGER_OFFSET",
    "EFFECT_OFFSET",
    "HAND_SAME_CARD_COUNT_OFFSET",
    "BOARD_SAME_CARD_COUNT_OFFSET",
    "TRIPLE_REWARD_SPELL_OFFSET",
    "TRIPLE_DISCOVER_TIER_OFFSET",
    "FROZEN_OFFSET",
    "SLOT_DIM",
    "GLOBAL_DIM",
    "GLOBAL_CORE_DIM",
    "SHOP_ROTATION_OBS_DIM",
    "LAST_BATTLE_DIM",
    "HAND_LEN",
    "PHASE_DIM",
    "PENDING_CHOICE_DIM",
    "PENDING_HEADER_DIM",
    "PENDING_OPTIONS_DIM",
    "PENDING_DISCOVER_IDX_DIM",
    "PENDING_HEADER_OFFSET",
    "PENDING_OPTIONS_OFFSET",
    "PENDING_DISCOVER_IDX_OFFSET",
    "PENDING_IS_APPLY_OFFSET",
    "PENDING_EFFECT_KIND_OFFSET",
    "PENDING_EFFECT_ATK_OFFSET",
    "PENDING_EFFECT_HP_OFFSET",
    "PENDING_EFFECT_TAUNT_OFFSET",
    "PENDING_CAN_SKIP_OFFSET",
    "PENDING_APPLY_REMAINING_OFFSET",
    "PENDING_PICKED_OFFSET",
    "PENDING_ELIGIBLE_OFFSET",
    "PENDING_ELIGIBLE_DIM",
    "PENDING_APPLY_SECTION_DIM",
    "OBS_DIM",
    "encode_minion",
    "encode_slots",
    "i_have_round_initiative",
    "build_observation",
    "encode_pending_choice",
]
