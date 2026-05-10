import numpy as np

from src.envs.minibg.actions import (
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
from src.envs.minibg.cards import make_minion
from src.envs.minibg.game import MiniBGGame
from src.envs.minibg.obs import (
    CARD_ID_TO_INDEX,
    GLOBAL_DIM,
    HAND_LEN,
    LAST_BATTLE_DIM,
    OBS_DIM,
    PHASE_DIM,
    SLOT_DIM,
    build_observation,
    encode_minion,
    encode_slots,
    i_have_round_initiative,
)
from src.envs.minibg.state import PlayerPhase


def test_obs_dim_matches_layout():
    expected = (
        GLOBAL_DIM
        + BOARD_SIZE * SLOT_DIM
        + SHOP_SIZE * SLOT_DIM
        + HAND_LEN * SLOT_DIM
        + BOARD_SIZE * SLOT_DIM
        + LAST_BATTLE_DIM
        + PHASE_DIM
    )
    assert OBS_DIM == expected
    assert SLOT_DIM == 25
    assert GLOBAL_DIM == 10
    assert LAST_BATTLE_DIM == 1
    assert HAND_LEN == HAND_SIZE
    assert PHASE_DIM == 1


def test_encode_minion_none_is_zero_vector():
    v = encode_minion(None)
    assert v.shape == (SLOT_DIM,)
    assert np.all(v == 0.0)
    assert v.dtype == np.float32


def test_encode_minion_recruit_layout():
    m = make_minion("recruit")
    v = encode_minion(m)
    assert v[0] == 1.0
    assert v[1 + CARD_ID_TO_INDEX["recruit"]] == 1.0
    # tier 1 → first tier slot
    assert v[11] == 1.0 and v[12] == 0.0 and v[13] == 0.0
    assert v[14] == 2.0 / 5.0  # base_attack
    assert v[15] == 2.0 / 5.0  # base_health
    assert v[16] == 0.0  # bonus_attack
    assert v[17] == 0.0  # bonus_health
    # no taunt / shield / abilities
    assert v[18] == 0.0
    assert v[19] == 0.0
    assert v[20] == 0.0
    assert v[21:25].sum() == 0.0


def test_encode_minion_shield_bot_runtime_shield_armed():
    m = make_minion("shield_bot")
    v = encode_minion(m)
    assert v[19] == 1.0  # shield keyword
    assert v[20] == 1.0  # runtime has_shield armed at construction


def test_encode_minion_guard_taunt():
    m = make_minion("guard")
    v = encode_minion(m)
    assert v[18] == 1.0
    assert v[19] == 0.0


def test_encode_minion_ability_flags():
    buf = encode_minion(make_minion("buffer"))
    assert buf[21] == 1.0  # ON_BUY
    assert buf[22] == 0.0
    assert buf[23] == 0.0
    assert buf[24] == 0.0

    pr = encode_minion(make_minion("pack_rat"))
    assert pr[22] == 1.0  # ON_DEATH

    cmd = encode_minion(make_minion("commander"))
    assert cmd[23] == 1.0  # AURA

    mn = encode_minion(make_minion("mentor"))
    assert mn[24] == 1.0  # ON_TURN_END


def test_encode_minion_bonus_stats():
    m = make_minion("recruit")
    m.bonus_attack = 3
    m.bonus_health = 4
    v = encode_minion(m)
    assert v[16] == 3.0 / 5.0
    assert v[17] == 4.0 / 5.0


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
    g = MiniBGGame(seed=0)
    s = g.initial_state()
    s.initiative_player = 0
    s.round_number = 1  # odd
    assert i_have_round_initiative(s, 0) == 1.0
    assert i_have_round_initiative(s, 1) == 0.0
    s.round_number = 2  # even → flips
    assert i_have_round_initiative(s, 0) == 0.0
    assert i_have_round_initiative(s, 1) == 1.0


def test_build_observation_shape_and_dtype():
    g = MiniBGGame(seed=0)
    s = g.initial_state()
    obs = build_observation(s, 0, 0.0, [])
    assert obs.shape == (OBS_DIM,)
    assert obs.dtype == np.float32


def test_build_observation_globals_match_state():
    g = MiniBGGame(seed=0)
    s = g.initial_state()
    s.players[0].health = 10
    s.players[1].health = 7
    s.players[0].gold = 5
    s.players[0].tavern_tier = 2
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
    assert g_arr[9] == 1.0  # initiative on odd round


def test_build_observation_self_centric():
    g = MiniBGGame(seed=0)
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

    # last_battle is 2nd-to-last; PHASE indicator is last.
    assert obs0[-2] == np.float32(0.5)
    assert obs1[-2] == np.float32(-0.3)


def test_build_observation_empty_enemy_board_zero_block():
    g = MiniBGGame(seed=0)
    s = g.initial_state()
    obs = build_observation(s, 0, 0.0, [])
    enemy_block_start = (
        GLOBAL_DIM
        + BOARD_SIZE * SLOT_DIM
        + SHOP_SIZE * SLOT_DIM
        + HAND_LEN * SLOT_DIM
    )
    enemy_block = obs[enemy_block_start : enemy_block_start + BOARD_SIZE * SLOT_DIM]
    assert np.all(enemy_block == 0.0)


def test_build_observation_hand_block_and_phase_indicator():
    g = MiniBGGame(seed=0)
    s = g.initial_state()
    s.players[0].hand[1] = make_minion("guard")
    obs = build_observation(s, 0, 0.0, [])

    hand_start = GLOBAL_DIM + BOARD_SIZE * SLOT_DIM + SHOP_SIZE * SLOT_DIM
    hand_block = obs[hand_start : hand_start + HAND_LEN * SLOT_DIM].reshape(
        HAND_LEN, SLOT_DIM
    )
    assert hand_block[0, 0] == 0.0 and hand_block[1, 0] == 1.0 and hand_block[2, 0] == 0.0

    assert obs[-1] == 0.0  # SHOP phase
    s.players[0].phase = PlayerPhase.ORDER
    assert build_observation(s, 0, 0.0, [])[-1] == 1.0


def test_build_observation_own_board_block_layout():
    g = MiniBGGame(seed=0)
    s = g.initial_state()
    s.players[0].board = [make_minion("recruit")]
    obs = build_observation(s, 0, 0.0, [])
    own_start = GLOBAL_DIM
    first_slot = obs[own_start : own_start + SLOT_DIM]
    second_slot = obs[own_start + SLOT_DIM : own_start + 2 * SLOT_DIM]
    assert first_slot[0] == 1.0
    assert np.all(second_slot == 0.0)
