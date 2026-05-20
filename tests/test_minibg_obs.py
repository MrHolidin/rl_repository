import numpy as np
import pytest

from src.envs.minibg.actions import (
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
from src.bg_catalog.cards import make_minion
from src.envs.minibg.game import MiniBGGame
from src.envs.minibg.obs import (
    BOARD_SAME_CARD_COUNT_OFFSET,
    TRIPLE_DISCOVER_TIER_OFFSET,
    TRIPLE_REWARD_SPELL_OFFSET,
    CARD_ID_TO_DENSE,
    CARD_IDX_OFFSET,
    EFFECT_OFFSET,
    GLOBAL_CORE_DIM,
    GLOBAL_DIM,
    HAND_LEN,
    HAND_SAME_CARD_COUNT_OFFSET,
    KEYWORD_OFFSET,
    LAST_BATTLE_DIM,
    NUM_TIER_ONEHOT,
    NUM_TRIGGER_CHANNELS,
    OBS_DIM,
    PHASE_DIM,
    PENDING_CHOICE_DIM,
    RACE_OFFSET,
    RACE_ONEHOT_DIM,
    GOLDEN_OFFSET,
    SHIELD_OFFSET,
    SLOT_DIM,
    STATS_OFFSET,
    TIER_OFFSET,
    TRIGGER_OFFSET,
    build_observation,
    encode_minion,
    encode_slots,
    i_have_round_initiative,
)
from src.envs.minibg.state import CNT_ACTIVE_SHOP_TRIBES, PlayerPhase, Race, ROTATION_SHOP_TRIBES

# Layout aliases preserved so legacy assertions read the same offsets.
_T0 = TIER_OFFSET
_S0 = STATS_OFFSET
_R0 = RACE_OFFSET
_K0 = KEYWORD_OFFSET
_SH = SHIELD_OFFSET
_TG0 = TRIGGER_OFFSET


def test_obs_dim_matches_layout():
    expected = (
        GLOBAL_DIM
        + BOARD_SIZE * SLOT_DIM
        + MAX_SHOP_SLOTS * SLOT_DIM
        + HAND_LEN * SLOT_DIM
        + BOARD_SIZE * SLOT_DIM
        + LAST_BATTLE_DIM
        + PHASE_DIM
        + PENDING_CHOICE_DIM
    )
    assert OBS_DIM == expected
    assert SLOT_DIM == 57
    assert GLOBAL_DIM == 16
    assert LAST_BATTLE_DIM == 1
    assert HAND_LEN == HAND_SIZE
    assert PHASE_DIM == 1


def test_encode_minion_none_is_zero_vector():
    v = encode_minion(None)
    assert v.shape == (SLOT_DIM,)
    assert np.all(v == 0.0)
    assert v.dtype == np.float32


def test_encode_minion_golden_flag():
    m = make_minion("recruit")
    m.is_golden = True
    v = encode_minion(m)
    assert v[GOLDEN_OFFSET] == 1.0


def test_encode_minion_triple_reward_spell_in_hand():
    from src.bg_recruitment.triples import make_triple_reward_discover_spell

    spell = make_triple_reward_discover_spell(discover_tier=3)
    v = encode_minion(spell)
    assert v[0] == 1.0
    assert v[TRIPLE_REWARD_SPELL_OFFSET] == 1.0
    assert v[TRIPLE_DISCOVER_TIER_OFFSET] == pytest.approx(3.0 / float(MAX_TIER))
    assert v[TIER_OFFSET : TIER_OFFSET + NUM_TIER_ONEHOT].sum() == 0.0


def test_build_observation_hand_encodes_triple_reward_spell():
    from src.bg_recruitment.triples import make_triple_reward_discover_spell
    from src.envs.minibg.state import MiniBGState, PlayerPhase, PlayerState

    spell = make_triple_reward_discover_spell(discover_tier=4)
    p0 = PlayerState(
        health=40,
        gold=10,
        tavern_tier=3,
        next_tier_up_cost=5,
        board=[],
        shop=[None] * MAX_SHOP_SLOTS,
        hand=[spell, None, None, None, None],
        phase=PlayerPhase.SHOP,
        shop_actions_used=0,
    )
    p1 = PlayerState(
        health=40,
        gold=10,
        tavern_tier=3,
        next_tier_up_cost=5,
        board=[],
        shop=[None] * MAX_SHOP_SLOTS,
        hand=[None] * HAND_SIZE,
        phase=PlayerPhase.SHOP,
        shop_actions_used=0,
    )
    s = MiniBGState(
        players=(p0, p1),
        round_number=1,
        current_player_index=0,
        initiative_player=0,
        winner=None,
        done=False,
    )
    obs = build_observation(s, 0, 0.0, [])
    hand_start = GLOBAL_DIM + BOARD_SIZE * SLOT_DIM + MAX_SHOP_SLOTS * SLOT_DIM
    hand_blk = obs[hand_start : hand_start + HAND_LEN * SLOT_DIM].reshape(
        HAND_LEN, SLOT_DIM
    )
    assert hand_blk[0][TRIPLE_REWARD_SPELL_OFFSET] == 1.0
    assert hand_blk[0][TRIPLE_DISCOVER_TIER_OFFSET] == pytest.approx(
        4.0 / float(MAX_TIER)
    )


def test_encode_minion_mecharoo_layout():
    m = make_minion("toy_mech")
    assert m.card_id == "BOT_445"
    v = encode_minion(m)
    assert v[0] == 1.0
    assert int(v[CARD_IDX_OFFSET]) == CARD_ID_TO_DENSE["BOT_445"]
    assert v[_T0] == 1.0 and v[_T0 + 1 : _T0 + NUM_TIER_ONEHOT].sum() == 0.0
    assert v[_S0] == 1.0 / 5.0
    assert v[_S0 + 1] == 1.0 / 5.0
    assert v[_S0 + 2] == 0.0
    assert v[_S0 + 3] == 0.0
    assert v[_R0 + 3] == 1.0
    assert v[_K0 : _SH].sum() == 0.0
    assert v[_SH] == 0.0
    assert v[_TG0 + 1] == 1.0
    assert v[_TG0] == 0.0
    assert v[_TG0 + 2 : _TG0 + 8].sum() == 0.0
    assert v[HAND_SAME_CARD_COUNT_OFFSET] == 0.0
    assert v[BOARD_SAME_CARD_COUNT_OFFSET] == 0.0
def test_build_observation_same_non_golden_copy_counts_own_board_and_hand():
    g = MiniBGGame(seed=0, shop_full_tribes=True)
    s = g.initial_state()
    p = s.players[0]
    p.board = [
        make_minion("recruit"),
        make_minion("recruit"),
        make_minion("guard"),
    ]
    p.hand[0] = make_minion("recruit")
    obs = build_observation(s, 0, 0.0, [])
    own = obs[GLOBAL_DIM : GLOBAL_DIM + BOARD_SIZE * SLOT_DIM].reshape(
        BOARD_SIZE, SLOT_DIM
    )
    assert float(own[0][HAND_SAME_CARD_COUNT_OFFSET]) == pytest.approx(1.0 / HAND_SIZE)
    assert float(own[0][BOARD_SAME_CARD_COUNT_OFFSET]) == pytest.approx(
        1.0 / BOARD_SIZE
    )
    assert float(own[2][HAND_SAME_CARD_COUNT_OFFSET]) == pytest.approx(0.0)
    assert float(own[2][BOARD_SAME_CARD_COUNT_OFFSET]) == pytest.approx(0.0)
    hs = GLOBAL_DIM + BOARD_SIZE * SLOT_DIM + MAX_SHOP_SLOTS * SLOT_DIM
    hand_blk = obs[hs : hs + HAND_LEN * SLOT_DIM].reshape(HAND_LEN, SLOT_DIM)
    assert float(hand_blk[0][HAND_SAME_CARD_COUNT_OFFSET]) == pytest.approx(0.0)
    assert float(hand_blk[0][BOARD_SAME_CARD_COUNT_OFFSET]) == pytest.approx(
        2.0 / BOARD_SIZE
    )


    m = make_minion("shield_bot")
    v = encode_minion(m)
    assert v[_K0 + 1] == 1.0
    assert v[_SH] == 1.0


def test_encode_minion_guard_taunt():
    m = make_minion("guard")
    v = encode_minion(m)
    assert v[_K0] == 1.0
    assert v[_K0 + 1] == 0.0


def test_encode_minion_ability_flags():
    buf = encode_minion(make_minion("buffer"))
    assert buf[_TG0 + 4] == 1.0
    assert buf[_TG0] == 0.0
    assert buf[_TG0 : _TG0 + 4].sum() == 0.0
    assert buf[_TG0 + 5] == 0.0
    assert buf[_TG0 + 6] == 0.0

    pr = encode_minion(make_minion("pack_rat"))
    assert pr[_TG0 + 1] == 1.0

    cmd = encode_minion(make_minion("commander"))
    assert cmd[_TG0 + 2] == 1.0

    mn = encode_minion(make_minion("mentor"))
    assert mn[_TG0 + 3] == 1.0

    ww = encode_minion(make_minion("wrath_weaver"))
    assert ww[_TG0 + 5] == 1.0

    kg = encode_minion(make_minion("kangors_apprentice"))
    assert kg[_TG0 + 1] == 1.0


def test_encode_minion_bonus_stats():
    m = make_minion("recruit")
    m.bonus_attack = 3
    m.bonus_health = 4
    v = encode_minion(m)
    assert v[_S0 + 2] == 3.0 / 5.0
    assert v[_S0 + 3] == 4.0 / 5.0


def test_encode_slots_pads_with_zero_rows():
    out = encode_slots([make_minion("recruit")], BOARD_SIZE)
    assert out.shape == (BOARD_SIZE, SLOT_DIM)
    assert out[0, 0] == 1.0
    for i in range(1, BOARD_SIZE):
        assert np.all(out[i] == 0.0)


def test_encode_slots_truncates_overflow():
    out = encode_slots([make_minion("recruit")] * 10, BOARD_SIZE)
    assert out.shape == (BOARD_SIZE, SLOT_DIM)
    assert np.all(out[:, 0] == 1.0)


def test_initiative_helper_self_centric():
    g = MiniBGGame(seed=0, shop_full_tribes=True)
    s = g.initial_state()
    s.initiative_player = 0
    s.round_number = 1
    assert i_have_round_initiative(s, 0) == 1.0
    assert i_have_round_initiative(s, 1) == 0.0
    s.round_number = 2
    assert i_have_round_initiative(s, 0) == 0.0
    assert i_have_round_initiative(s, 1) == 1.0


def test_build_observation_shape_and_dtype():
    g = MiniBGGame(seed=0, shop_full_tribes=True)
    s = g.initial_state()
    obs = build_observation(s, 0, 0.0, [])
    assert obs.shape == (OBS_DIM,)
    assert obs.dtype == np.float32


def test_build_observation_globals_match_state():
    g = MiniBGGame(seed=0, shop_full_tribes=True)
    s = g.initial_state()
    s.players[0].health = 10
    s.players[1].health = 7
    s.players[0].gold = 5
    s.players[0].tavern_tier = 2
    s.players[0].next_tier_up_cost = 4
    s.players[1].tavern_tier = 3
    s.players[0].shop_actions_used = 4
    s.players[0].board = [make_minion("recruit"), make_minion("guard")]
    s.round_number = 3
    s.initiative_player = 0

    obs = build_observation(s, 0, 0.0, [])
    g_arr = obs[:GLOBAL_DIM]
    assert g_arr[0] == 3 / MAX_ROUNDS
    assert g_arr[1] == 10 / STARTING_HEALTH
    assert g_arr[2] == 7 / STARTING_HEALTH
    assert g_arr[3] == 5 / GOLD_AT_CAP
    assert g_arr[4] == gold_for_round(3) / GOLD_AT_CAP
    assert g_arr[5] == 2 / MAX_TIER
    assert g_arr[6] == 3 / MAX_TIER
    assert g_arr[7] == (MAX_SHOP_ACTIONS - 4) / MAX_SHOP_ACTIONS
    assert g_arr[8] == 2 / BOARD_SIZE
    assert g_arr[9] == 1.0
    assert g_arr[10] == 4 / LEVEL_UP_COST_MAX
    assert tuple(g_arr[11:15]) == (0.0, 0.0, 0.0, 0.0)
    assert g_arr[15] == 1.0


def test_build_observation_self_centric():
    g = MiniBGGame(seed=0, shop_full_tribes=True)
    s = g.initial_state()
    s.players[0].health = 10
    s.players[1].health = 7
    s.players[0].board = [make_minion("recruit")]
    s.players[1].board = [make_minion("guard"), make_minion("bruiser")]

    obs0 = build_observation(s, 0, 0.5, [make_minion("guard")])
    obs1 = build_observation(s, 1, -0.3, [make_minion("recruit")])

    assert obs0[1] == 10 / STARTING_HEALTH
    assert obs0[2] == 7 / STARTING_HEALTH
    assert obs1[1] == 7 / STARTING_HEALTH
    assert obs1[2] == 10 / STARTING_HEALTH

    lb_off = PENDING_CHOICE_DIM + PHASE_DIM + LAST_BATTLE_DIM
    assert obs0[-lb_off] == np.float32(0.5)
    assert obs1[-lb_off] == np.float32(-0.3)


def test_build_observation_empty_enemy_board_zero_block():
    g = MiniBGGame(seed=0, shop_full_tribes=True)
    s = g.initial_state()
    obs = build_observation(s, 0, 0.0, [])
    enemy_block_start = (
        GLOBAL_DIM
        + BOARD_SIZE * SLOT_DIM
        + MAX_SHOP_SLOTS * SLOT_DIM
        + HAND_LEN * SLOT_DIM
    )
    enemy_block = obs[enemy_block_start : enemy_block_start + BOARD_SIZE * SLOT_DIM]
    assert np.all(enemy_block == 0.0)


def test_build_observation_hand_block_and_phase_indicator():
    g = MiniBGGame(seed=0, shop_full_tribes=True)
    s = g.initial_state()
    s.players[0].hand[1] = make_minion("guard")
    obs = build_observation(s, 0, 0.0, [])

    hand_start = GLOBAL_DIM + BOARD_SIZE * SLOT_DIM + MAX_SHOP_SLOTS * SLOT_DIM
    hand_block = obs[hand_start : hand_start + HAND_LEN * SLOT_DIM].reshape(
        HAND_LEN, SLOT_DIM
    )
    assert hand_block[0, 0] == 0.0 and hand_block[1, 0] == 1.0 and hand_block[2, 0] == 0.0

    phase_off = PENDING_CHOICE_DIM + PHASE_DIM
    assert obs[-phase_off] == 0.0
    from src.envs.minibg.actions import MAX_SHOP_ACTIONS

    s.players[0].shop_actions_used = MAX_SHOP_ACTIONS
    assert build_observation(s, 0, 0.0, [])[-phase_off] == 1.0


def test_build_observation_own_board_block_layout():
    g = MiniBGGame(seed=0, shop_full_tribes=True)
    s = g.initial_state()
    s.players[0].board = [make_minion("recruit")]
    obs = build_observation(s, 0, 0.0, [])
    own_start = GLOBAL_DIM
    first_slot = obs[own_start : own_start + SLOT_DIM]
    second_slot = obs[own_start + SLOT_DIM : own_start + 2 * SLOT_DIM]
    assert first_slot[0] == 1.0
    assert np.all(second_slot == 0.0)


def test_build_observation_encodes_shop_excluded_race():
    g = MiniBGGame(seed=123, shop_excluded_race=Race.MURLOC)
    s = g.initial_state()
    obs = build_observation(s, 0, 0.0, [])
    rot = obs[GLOBAL_CORE_DIM:GLOBAL_DIM]
    assert rot[3] == 1.0
    assert rot[0:3].sum() == 0.0
    assert rot[4] == CNT_ACTIVE_SHOP_TRIBES / len(ROTATION_SHOP_TRIBES)
