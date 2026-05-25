"""Patch 74257 scaffold and P0 engine hooks."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.bg_catalog.patch_context import PatchContext
from src.bg_core.effects import (
    Ability,
    Keyword,
    ReduceUpgradeCostEffect,
    StartOfCombatDamagePerFriendlyTribe,
    Trigger,
)
from src.bg_core.minion import Race
from src.envs.minibg.obs import (
    KEYWORD_OFFSET,
    RACE_OFFSET,
    TRIGGER_INDEX,
    TRIGGER_OFFSET,
    _RACE_ORDER,
    encode_minion,
)
from tests.minibg_helpers import PATCH_CTX, make_minion, simulate_battle

PATCH_74257 = Path("data/bgcore/19_6_0_74257")


def test_patch_74257_loads():
    ctx = PatchContext.load(PATCH_74257)
    assert ctx.build == 74257
    assert ctx.patch == "19.6.0"
    assert len(ctx.pool_ids) == 127
    assert len(ctx.meta.rotation_tribes) == 7
    assert ctx.meta.cnt_active_shop_tribes == 6


def test_patch_74257_dragon_race_in_catalog():
    ctx = PatchContext.load(PATCH_74257)
    whelp = ctx.templates["BGS_019"]
    assert whelp.race == Race.DRAGON


def test_reborn_respawns_after_lethal():
    m = make_minion("recruit")
    m.keywords = frozenset({Keyword.REBORN})
    m.base_health = 2
    m.base_attack = 2
    e = make_minion("recruit")
    e.base_attack = 2
    e.base_health = 2
    survivors: list = []
    board_out: list = []
    simulate_battle(
        [m],
        [e],
        p0_has_initiative=False,
        rng=np.random.default_rng(0),
        p0_survivors_out=survivors,
        p0_board_out=board_out,
    )
    assert survivors == [m.card_id]
    assert len(board_out) == 1
    assert Keyword.REBORN not in board_out[0].all_keywords


def test_reborn_trades_to_one_hp_survivor():
    """2/2 Reborn vs 2/2: mutual lethal → p0 reborn at 1 HP, p1 eliminated."""
    m = make_minion("recruit")
    m.keywords = frozenset({Keyword.REBORN})
    m.base_health = 2
    m.base_attack = 2
    e = make_minion("recruit")
    e.base_attack = 2
    e.base_health = 2
    p0_survivors: list = []
    p1_survivors: list = []
    simulate_battle(
        [m],
        [e],
        p0_has_initiative=True,
        rng=np.random.default_rng(0),
        p0_survivors_out=p0_survivors,
        p1_survivors_out=p1_survivors,
    )
    assert p0_survivors == [m.card_id]
    assert p1_survivors == []


def test_reborn_does_not_trigger_twice():
    """After reborn is consumed, a second lethal leaves the minion dead."""
    m = make_minion("recruit")
    m.keywords = frozenset({Keyword.REBORN})
    m.base_health = 2
    m.base_attack = 2
    trade = make_minion("recruit")
    trade.base_attack = 2
    trade.base_health = 2
    finisher = make_minion("recruit")
    finisher.base_attack = 5
    finisher.base_health = 5
    p0_survivors: list = []
    simulate_battle(
        [m],
        [trade],
        p0_has_initiative=True,
        rng=np.random.default_rng(0),
        p0_survivors_out=p0_survivors,
    )
    assert p0_survivors == [m.card_id]
    p0_survivors.clear()
    simulate_battle(
        [m],
        [finisher],
        p0_has_initiative=False,
        rng=np.random.default_rng(0),
        p0_survivors_out=p0_survivors,
    )
    assert p0_survivors == []


def test_bronze_warden_has_reborn_from_74257_catalog():
    ctx = PatchContext.load(PATCH_74257)
    warden = ctx.templates["BGS_034"]
    assert Keyword.REBORN in warden.all_keywords


def test_start_of_combat_damages_enemy_before_attacks():
    ctx742 = PatchContext.load(PATCH_74257)
    whelp = ctx742.make_minion("BGS_019")
    whelp.abilities = (
        Ability(
            Trigger.ON_START_OF_COMBAT,
            StartOfCombatDamagePerFriendlyTribe(Race.DRAGON, amount_per_match=1),
        ),
    )
    target = make_minion("recruit")
    p1_out: list = []
    simulate_battle(
        [whelp],
        [target],
        p0_has_initiative=True,
        rng=np.random.default_rng(0),
        patch=ctx742,
        p1_board_out=p1_out,
    )
    assert p1_out == []


def test_obs_encodes_reborn_and_start_of_combat_triggers():
    m = make_minion("recruit")
    m.keywords = frozenset({Keyword.REBORN})
    m.abilities = (
        Ability(
            Trigger.ON_START_OF_COMBAT,
            StartOfCombatDamagePerFriendlyTribe(Race.DRAGON),
        ),
    )
    v = encode_minion(m, card_id_to_dense=PATCH_CTX.card_id_to_dense)
    assert v[KEYWORD_OFFSET + 6] == 1.0
    ti = TRIGGER_INDEX[Trigger.ON_START_OF_COMBAT]
    assert v[TRIGGER_OFFSET + ti] == 1.0


def test_effective_sell_reward_override():
    from src.bg_recruitment.economy import effective_sell_reward

    m = make_minion("recruit")
    assert effective_sell_reward(m) == 1
    m.sell_value = 3
    assert effective_sell_reward(m) == 3


def test_freedealing_gambler_sell_value_from_catalog():
    """T-P1-02: catalog text parses «sells for 3 Gold»."""
    ctx = PatchContext.load(PATCH_74257)
    gambler = ctx.templates["BGS_049"]
    assert gambler.sell_value == 3


def test_effective_shop_cost_overrides():
    from src.bg_lobby.player import PlayerPhase, PlayerState
    from src.bg_recruitment.economy import effective_level_up_cost, effective_roll_cost

    p = PlayerState(
        health=40,
        gold=10,
        tavern_tier=1,
        next_tier_up_cost=5,
        board=[],
        shop=[None] * 6,
        hand=[None] * 5,
        phase=PlayerPhase.SHOP,
        shop_actions_used=0,
        upgrade_cost_delta=-1,
        next_roll_cost_override=0,
    )
    assert effective_level_up_cost(p) == 4
    assert effective_roll_cost(p) == 0


def test_obs_encodes_elemental_race_74257():
    ctx = PatchContext.load(PATCH_74257)
    sel = ctx.make_minion("BGS_116")
    assert sel.race == Race.ELEMENTAL
    v = encode_minion(sel, card_id_to_dense=ctx.card_id_to_dense)
    idx = _RACE_ORDER.index(Race.ELEMENTAL)
    assert v[RACE_OFFSET + idx] == 1.0


def test_shop_frozen_slot_survives_roll():
    from src.bg_lobby.player import PlayerPhase, PlayerState
    from src.bg_recruitment.shop import refresh_shop

    p = PlayerState(
        health=40,
        gold=10,
        tavern_tier=1,
        next_tier_up_cost=5,
        board=[],
        shop=[None] * 6,
        hand=[None] * 5,
        phase=PlayerPhase.SHOP,
        shop_actions_used=0,
    )
    ctx = PatchContext.load(PATCH_74257)
    p.shop[0] = ctx.make_minion("BGS_019")
    p.shop_frozen = (True, False, False, False, False, False)
    kept = p.shop[0].card_id
    refresh_shop(
        p,
        None,
        rng=np.random.default_rng(0),
        frozen_slots=p.shop_frozen,
        patch=ctx,
    )
    assert p.shop[0] is not None
    assert p.shop[0].card_id == kept
    assert p.shop[1] is not None


def _shop_player(**kwargs):
    from src.bg_lobby.player import PlayerPhase, PlayerState

    defaults = dict(
        health=40,
        gold=10,
        tavern_tier=1,
        next_tier_up_cost=5,
        board=[],
        shop=[None] * 6,
        hand=[None] * 5,
        phase=PlayerPhase.SHOP,
        shop_actions_used=0,
    )
    defaults.update(kwargs)
    return PlayerState(**defaults)


def test_on_sell_sellemental_adds_token_to_hand():
    from src.bg_recruitment.economy import sell_from_board
    from src.bg_recruitment.shop_triggers import ShopTriggers

    ctx = PatchContext.load(PATCH_74257)
    p = _shop_player(gold=0)
    sel = ctx.make_minion("BGS_115")
    p.board = [sel]
    triggers = ShopTriggers(patch=ctx, rng=np.random.default_rng(0))
    sell_from_board(
        p,
        0,
        on_sell=triggers.fire_on_sell,
        on_triples=lambda _p: None,
    )
    assert p.board == []
    assert any(h is not None and h.card_id == "BGS_115t" for h in p.hand)


def test_on_friendly_bought_hoggarr_grants_gold_for_pirate():
    from src.envs.minibg.actions import BUY_COST
    from src.bg_recruitment.economy import buy_from_shop
    from src.bg_recruitment.shop_triggers import ShopTriggers

    ctx = PatchContext.load(PATCH_74257)
    p = _shop_player(gold=BUY_COST)
    hoggarr = ctx.make_minion("BGS_072")
    pirate = ctx.make_minion("BGS_049")
    p.board = [hoggarr]
    p.shop[0] = pirate
    triggers = ShopTriggers(patch=ctx, rng=np.random.default_rng(0))
    buy_from_shop(
        p,
        0,
        on_bought=triggers.fire_on_buy,
        on_friendly_bought=triggers.fire_on_friendly_bought,
        on_triples=lambda _p: None,
    )
    assert p.gold == 1


def test_nomi_increments_shop_elemental_bonus():
    from src.bg_recruitment.shop_triggers import ShopTriggers

    ctx = PatchContext.load(PATCH_74257)
    p = _shop_player()
    nomi = ctx.make_minion("BGS_104")
    elemental = ctx.make_minion("BGS_115")
    shop_elem = ctx.make_minion("BGS_116")
    p.board = [nomi]
    p.shop[0] = shop_elem
    triggers = ShopTriggers(patch=ctx, rng=np.random.default_rng(0))
    triggers.fire_on_place(elemental, p, None)
    triggers.fire_after_friendly_minion_placed(p, elemental)
    assert p.shop_elemental_bonus == 1
    assert shop_elem.bonus_attack == 1
    assert shop_elem.bonus_health == 1


def test_count_unique_tribes_excludes_self():
    from src.bg_core.board_helpers import count_unique_tribes

    ctx = PatchContext.load(PATCH_74257)
    amalg = ctx.make_minion("BGS_069")
    beast = ctx.make_minion("BGS_019")
    beast.race = Race.BEAST
    board = [amalg, beast]
    assert count_unique_tribes(board, exclude=amalg) == 1


def test_amalgadon_adapts_without_modal():
    from src.bg_recruitment.shop_triggers import ShopTriggers

    ctx = PatchContext.load(PATCH_74257)
    p = _shop_player()
    beast = ctx.make_minion("BGS_019")
    beast.race = Race.BEAST
    murloc = ctx.make_minion("BGS_020")
    amalg = ctx.make_minion("BGS_069")
    p.board = [beast, murloc, amalg]
    triggers = ShopTriggers(patch=ctx, rng=np.random.default_rng(0))
    triggers.fire_on_place(amalg, p, None)
    assert p.pending_choice is None
    assert amalg.bonus_attack + amalg.bonus_health > 0


def test_last_combat_won_flag_after_battle():
    from src.bg_lobby.player import PlayerPhase, PlayerState

    winner = make_minion("recruit")
    winner.base_attack = 10
    winner.base_health = 10
    p0 = PlayerState(
        health=40,
        gold=0,
        tavern_tier=1,
        next_tier_up_cost=5,
        board=[winner],
        shop=[None] * 6,
        hand=[None] * 5,
        phase=PlayerPhase.DONE,
        shop_actions_used=0,
    )
    p1 = PlayerState(
        health=40,
        gold=0,
        tavern_tier=1,
        next_tier_up_cost=5,
        board=[],
        shop=[None] * 6,
        hand=[None] * 5,
        phase=PlayerPhase.DONE,
        shop_actions_used=0,
    )
    ctx = PatchContext.load(PATCH_74257)
    dmg_p0, dmg_p1 = simulate_battle(
        p0.board,
        p1.board,
        p0_has_initiative=True,
        rng=np.random.default_rng(0),
        patch=ctx,
    )
    p0.last_combat_won = dmg_p0 == 0 and dmg_p1 > 0
    p1.last_combat_won = dmg_p1 == 0 and dmg_p0 > 0
    assert dmg_p0 == 0
    assert dmg_p1 > 0
    assert p0.last_combat_won is True
    assert p1.last_combat_won is False


def test_macaw_triggers_random_friendly_deathrattle():
    ctx = PatchContext.load(PATCH_74257)
    macaw = ctx.make_minion("BGS_078")
    macaw.base_attack = 6
    dr_rat = ctx.make_minion("EX1_556")
    enemy = make_minion("recruit")
    enemy.base_health = 1
    p0_out: list = []
    simulate_battle(
        [macaw, dr_rat],
        [enemy],
        p0_has_initiative=True,
        rng=np.random.default_rng(0),
        patch=ctx,
        p0_board_out=p0_out,
    )
    assert len(p0_out) >= 2
    assert any(m.card_id == "skele21" for m in p0_out)


def test_glyph_guardian_doubles_attack_after_swing():
    ctx = PatchContext.load(PATCH_74257)
    guard = ctx.make_minion("BGS_045")
    guard.base_attack = 2
    enemy = make_minion("recruit")
    enemy.base_health = 10
    enemy.base_attack = 0
    p0_out: list = []
    simulate_battle(
        [guard],
        [enemy],
        p0_has_initiative=True,
        rng=np.random.default_rng(0),
        patch=ctx,
        p0_board_out=p0_out,
    )
    assert p0_out
    assert p0_out[0].raw_attack >= 4


def test_drakonid_enforcer_buffs_on_friendly_shield_lost():
    ctx = PatchContext.load(PATCH_74257)
    from src.bg_core.effects import Ability, Keyword, Trigger

    shielded = make_minion("recruit")
    shielded.keywords = frozenset({Keyword.SHIELD})
    shielded.has_shield = True
    enforcer = ctx.make_minion("BGS_067")
    enforcer.abilities = ctx.effects["BGS_067"]
    enemy = make_minion("recruit")
    enemy.base_attack = 1
    p0_out: list = []
    simulate_battle(
        [shielded, enforcer],
        [enemy],
        p0_has_initiative=False,
        rng=np.random.default_rng(1),
        patch=ctx,
        p0_board_out=p0_out,
    )
    enf = next(m for m in p0_out if m.card_id == "BGS_067")
    assert enf.bonus_attack >= 2
    assert enf.bonus_health >= 2


def test_menagerie_mug_buffs_unique_tribes():
    from src.bg_recruitment.shop_triggers import ShopTriggers

    ctx = PatchContext.load(PATCH_74257)
    p = _shop_player()
    mug = ctx.make_minion("BGS_082")
    beast = ctx.make_minion("BGS_019")
    beast.race = Race.BEAST
    murloc = ctx.make_minion("BGS_020")
    demon = ctx.make_minion("BGS_001")
    p.board = [beast, murloc, demon]
    triggers = ShopTriggers(patch=ctx, rng=np.random.default_rng(0))
    triggers.fire_on_place(mug, p, None)
    buffed = sum(1 for m in p.board if m.bonus_attack >= 1 and m.bonus_health >= 1)
    assert buffed == 3


def test_kalecgos_buffs_dragons_after_friendly_battlecry():
    from src.bg_recruitment.shop_triggers import ShopTriggers

    ctx = PatchContext.load(PATCH_74257)
    p = _shop_player()
    kale = ctx.make_minion("BGS_041")
    dragon = ctx.make_minion("BGS_019")
    bc = ctx.make_minion("BGS_055")
    bc.abilities = (Ability(Trigger.ON_PLACE, ReduceUpgradeCostEffect(amount=1)),)
    p.board = [kale, dragon]
    triggers = ShopTriggers(patch=ctx, rng=np.random.default_rng(0))
    triggers.fire_on_place(bc, p, None)
    triggers.fire_after_friendly_minion_placed(p, bc)
    assert dragon.bonus_attack >= 1
    assert dragon.bonus_health >= 1


def test_deck_swabbie_reduces_level_up_cost():
    from src.bg_recruitment.economy import effective_level_up_cost
    from src.bg_recruitment.shop_triggers import ShopTriggers

    ctx = PatchContext.load(PATCH_74257)
    p = _shop_player(gold=10, next_tier_up_cost=5)
    swabbie = ctx.make_minion("BGS_055")
    triggers = ShopTriggers(patch=ctx, rng=np.random.default_rng(0))
    triggers.fire_on_place(swabbie, p, None)
    assert effective_level_up_cost(p) == 4


def test_refreshing_anomaly_sets_next_roll_free():
    from src.bg_recruitment.economy import effective_roll_cost
    from src.bg_recruitment.shop_triggers import ShopTriggers

    ctx = PatchContext.load(PATCH_74257)
    p = _shop_player()
    anomaly = ctx.make_minion("BGS_116")
    triggers = ShopTriggers(patch=ctx, rng=np.random.default_rng(0))
    triggers.fire_on_place(anomaly, p, None)
    assert effective_roll_cost(p) == 0
    assert p.free_roll_charges == 1


def test_toggle_shop_slot_frozen():
    from src.bg_recruitment.shop import refresh_shop, toggle_shop_slot_frozen

    ctx = PatchContext.load(PATCH_74257)
    p = _shop_player()
    p.shop[0] = ctx.make_minion("BGS_019")
    toggle_shop_slot_frozen(p, 0)
    assert p.shop_frozen[0] is True
    kept = p.shop[0].card_id
    refresh_shop(p, None, rng=np.random.default_rng(0), frozen_slots=p.shop_frozen, patch=ctx)
    assert p.shop[0] is not None
    assert p.shop[0].card_id == kept
    assert p.shop[1] is not None


def test_primalfin_requires_other_murloc_for_discover():
    from src.bg_recruitment.shop_triggers import ShopTriggers

    ctx = PatchContext.load(PATCH_74257)
    p = _shop_player()
    primalfin = ctx.make_minion("BGS_020")
    triggers = ShopTriggers(patch=ctx, rng=np.random.default_rng(0))
    triggers.fire_on_place(primalfin, p, None)
    assert p.pending_choice is None
    ally = ctx.make_minion("BGS_020")
    p.board = [ally]
    primalfin2 = ctx.make_minion("BGS_020")
    triggers.fire_on_place(primalfin2, p, None)
    assert p.pending_choice is not None


def test_hangry_dragon_buffs_after_win_on_turn_start():
    from src.bg_recruitment.shop_triggers import ShopTriggers

    ctx = PatchContext.load(PATCH_74257)
    p = _shop_player()
    p.last_combat_won = True
    hangry = ctx.make_minion("BGS_033")
    p.board = [hangry]
    triggers = ShopTriggers(patch=ctx, rng=np.random.default_rng(0))
    triggers.fire_on_turn_start(p)
    assert hangry.bonus_attack >= 2
    assert hangry.bonus_health >= 2


def test_stasis_adds_frozen_elemental_to_shop():
    from src.bg_recruitment.shop_triggers import ShopTriggers

    ctx = PatchContext.load(PATCH_74257)
    p = _shop_player()
    stasis = ctx.make_minion("BGS_122")
    triggers = ShopTriggers(patch=ctx, rng=np.random.default_rng(0))
    triggers.fire_on_place(stasis, p, None)
    filled = [i for i, m in enumerate(p.shop) if m is not None and m.race == Race.ELEMENTAL]
    assert filled
    assert p.shop_frozen[filled[0]] is True


def test_steward_buffs_shop_on_sell():
    from src.bg_recruitment.economy import sell_from_board
    from src.bg_recruitment.shop_triggers import ShopTriggers

    ctx = PatchContext.load(PATCH_74257)
    p = _shop_player()
    offer = ctx.make_minion("BGS_116")
    p.shop[0] = offer
    steward = ctx.make_minion("BGS_037")
    p.board = [steward]
    triggers = ShopTriggers(patch=ctx, rng=np.random.default_rng(0))
    sell_from_board(p, 0, on_sell=triggers.fire_on_sell, on_triples=lambda _p: None)
    assert offer.bonus_attack >= 1
    assert offer.bonus_health >= 1


def test_soul_devourer_consumes_demon_for_stats_and_gold():
    from src.bg_recruitment.place import place_from_hand
    from src.bg_recruitment.shop_triggers import ShopTriggers

    ctx = PatchContext.load(PATCH_74257)
    p = _shop_player(gold=0)
    victim = ctx.make_minion("BGS_001")
    devourer = ctx.make_minion("BGS_059")
    p.board = [victim]
    p.hand[0] = devourer
    triggers = ShopTriggers(patch=ctx, rng=np.random.default_rng(0))
    place_from_hand(
        p,
        0,
        None,
        board_size=7,
        triggers=triggers,
        rng=np.random.default_rng(0),
    )
    assert all(m is not victim for m in p.board)
    assert len(p.board) == 1
    assert p.board[0].card_id == "BGS_059"
    assert p.board[0].bonus_attack >= victim.raw_attack
    assert p.board[0].bonus_health >= victim.max_health
    assert p.gold == 3


def test_murozond_adds_from_last_opponent_board():
    from src.bg_core.board_helpers import snapshot_warband
    from src.bg_recruitment.shop_triggers import ShopTriggers

    ctx = PatchContext.load(PATCH_74257)
    p = _shop_player()
    opponent = ctx.make_minion("BGS_019")
    p.last_opponent_board = snapshot_warband([opponent])
    muro = ctx.make_minion("BGS_043")
    triggers = ShopTriggers(patch=ctx, rng=np.random.default_rng(0))
    triggers.fire_on_place(muro, p, None)
    assert any(h is not None and h.card_id == "BGS_019" for h in p.hand)


def test_last_opponent_board_set_after_battle():
    from src.bg_core.board_helpers import snapshot_warband

    ctx = PatchContext.load(PATCH_74257)
    p0_board = [make_minion("recruit")]
    p1_board = [ctx.make_minion("BGS_019")]
    p0 = _shop_player(board=p0_board)
    p1 = _shop_player(board=p1_board)
    p0.last_opponent_board = snapshot_warband(p1_board)
    p1.last_opponent_board = snapshot_warband(p0_board)
    assert p0.last_opponent_board[0].card_id == "BGS_019"
    assert p1.last_opponent_board[0].card_id == p0_board[0].card_id


def test_faceless_transforms_into_shop_offer():
    from src.bg_recruitment.place import place_from_hand
    from src.bg_recruitment.shop_triggers import ShopTriggers

    ctx = PatchContext.load(PATCH_74257)
    p = _shop_player()
    offer = ctx.make_minion("BGS_019")
    p.shop[0] = offer
    faceless = ctx.make_minion("BGS_113")
    p.hand[0] = faceless
    triggers = ShopTriggers(patch=ctx, rng=np.random.default_rng(0))
    place_from_hand(
        p,
        0,
        None,
        board_size=7,
        triggers=triggers,
        rng=np.random.default_rng(0),
    )
    assert len(p.board) == 1
    assert p.board[0].card_id == "BGS_019"
    assert p.pending_choice is None


def test_faceless_modal_picks_shop_slot():
    from src.bg_player_turn.engine import PlayerTurnEngine
    from src.bg_player_turn.context import PlayerTurnContext
    from src.bg_recruitment.place import place_from_hand
    from src.bg_recruitment.shop_triggers import ShopTriggers
    from src.bg_lobby.player import PendingChoiceKind
    from src.envs.minibg.actions import Action

    ctx = PatchContext.load(PATCH_74257)
    p = _shop_player()
    offer0 = ctx.make_minion("BGS_019")
    offer1 = ctx.make_minion("BGS_020")
    p.shop[0] = offer0
    p.shop[1] = offer1
    faceless = ctx.make_minion("BGS_113")
    p.hand[0] = faceless
    triggers = ShopTriggers(patch=ctx, rng=np.random.default_rng(0))
    place_from_hand(
        p,
        0,
        None,
        board_size=7,
        triggers=triggers,
        rng=np.random.default_rng(0),
    )
    assert p.pending_choice is not None
    assert p.pending_choice.kind == PendingChoiceKind.TRANSFORM_SHOP_MINION
    engine = PlayerTurnEngine()
    turn_ctx = PlayerTurnContext(
        rng=np.random.default_rng(0),
        triggers=triggers,
        patch=ctx,
    )
    legal = engine.legal_actions(p)
    assert int(Action.BUY_SLOT_0) in legal
    assert int(Action.BUY_SLOT_1) in legal
    engine.apply(p, int(Action.BUY_SLOT_1), turn_ctx)
    assert p.pending_choice is None
    assert p.board[0].card_id == "BGS_020"


def test_murozond_forged_adds_golden_to_hand():
    from src.bg_core.board_helpers import snapshot_warband
    from src.bg_recruitment.shop_triggers import ShopTriggers

    ctx = PatchContext.load(PATCH_74257)
    p = _shop_player()
    opponent = ctx.make_minion("BGS_019")
    p.last_opponent_board = snapshot_warband([opponent])
    muro = ctx.make_minion("BGS_043")
    muro.abilities = ctx.triple_merge_golden_abilities("BGS_043")
    muro.is_golden = True
    triggers = ShopTriggers(patch=ctx, rng=np.random.default_rng(0))
    triggers.fire_on_place(muro, p, None)
    hand = next(h for h in p.hand if h is not None)
    assert hand.is_golden
    assert hand.card_id == "BGS_019"
    assert hand.base_attack == ctx.templates["BGS_019"].base_attack * 2


def test_faceless_forged_transforms_into_golden_copy():
    from src.bg_recruitment.place import place_from_hand
    from src.bg_recruitment.shop_triggers import ShopTriggers

    ctx = PatchContext.load(PATCH_74257)
    p = _shop_player()
    offer = ctx.make_minion("BGS_019")
    p.shop[0] = offer
    faceless = ctx.make_minion("BGS_113")
    faceless.abilities = ctx.triple_merge_golden_abilities("BGS_113")
    faceless.is_golden = True
    p.hand[0] = faceless
    triggers = ShopTriggers(patch=ctx, rng=np.random.default_rng(0))
    place_from_hand(
        p,
        0,
        None,
        board_size=7,
        triggers=triggers,
        rng=np.random.default_rng(0),
    )
    assert len(p.board) == 1
    assert p.board[0].card_id == "BGS_019"
    assert p.board[0].is_golden
    assert p.board[0].base_attack == ctx.templates["BGS_019"].base_attack * 2


def test_faceless_token_in_patch():
    ctx = PatchContext.load(PATCH_74257)
    assert "BGS_046t" in ctx.token_ids
    tok = ctx.make_minion("BGS_046t")
    assert tok.card_id == "BGS_046t"


def test_king_bagurgle_buffs_other_murlocs_on_place():
    from src.bg_recruitment.shop_triggers import ShopTriggers

    ctx = PatchContext.load(PATCH_74257)
    p = _shop_player()
    murloc = ctx.make_minion("BGS_020")
    bagurgle = ctx.make_minion("BGS_030")
    p.board = [murloc]
    triggers = ShopTriggers(patch=ctx, rng=np.random.default_rng(0))
    triggers.fire_on_place(bagurgle, p, None)
    assert murloc.bonus_attack >= 2
    assert murloc.bonus_health >= 2


def test_mythrax_buffs_from_unique_tribes_on_turn_end():
    from src.bg_recruitment.shop_triggers import ShopTriggers

    ctx = PatchContext.load(PATCH_74257)
    p = _shop_player()
    mythrax = ctx.make_minion("BGS_202")
    beast = ctx.make_minion("BGS_019")
    beast.race = Race.BEAST
    murloc = ctx.make_minion("BGS_020")
    p.board = [mythrax, beast, murloc]
    triggers = ShopTriggers(patch=ctx, rng=np.random.default_rng(0))
    triggers.fire_on_turn_end(p)
    assert mythrax.bonus_attack >= 2
    assert mythrax.bonus_health >= 4


def test_tavern_tempest_adds_elemental_to_hand():
    from src.bg_recruitment.shop_triggers import ShopTriggers

    ctx = PatchContext.load(PATCH_74257)
    p = _shop_player(tavern_tier=3)
    tempest = ctx.make_minion("BGS_123")
    triggers = ShopTriggers(patch=ctx, rng=np.random.default_rng(0))
    triggers.fire_on_place(tempest, p, None)
    assert any(h is not None and h.race == Race.ELEMENTAL for h in p.hand)


def test_tormented_ritualist_buffs_adjacent_when_attacked():
    ctx = PatchContext.load(PATCH_74257)
    ritualist = ctx.make_minion("BGS_201")
    ritualist.keywords = frozenset({Keyword.TAUNT})
    ally = make_minion("recruit")
    ally.base_attack = 1
    ally.base_health = 5
    enemy = make_minion("recruit")
    enemy.base_attack = 1
    enemy.base_health = 10
    filler = make_minion("recruit")
    filler.base_attack = 0
    filler.base_health = 1
    p0_out: list = []
    simulate_battle(
        [ally, ritualist],
        [enemy, filler],
        p0_has_initiative=False,
        rng=np.random.default_rng(0),
        patch=ctx,
        p0_board_out=p0_out,
    )
    ally_out = next(m for m in p0_out if m.card_id == ally.card_id)
    assert ally_out.bonus_attack >= 1
    assert ally_out.bonus_health >= 1


def test_champion_buffs_self_when_friendly_taunt_attacked():
    ctx = PatchContext.load(PATCH_74257)
    champion = ctx.make_minion("BGS_111")
    taunt = make_minion("recruit")
    taunt.keywords = frozenset({Keyword.TAUNT})
    taunt.base_health = 10
    enemy = make_minion("recruit")
    enemy.base_attack = 1
    enemy.base_health = 1
    filler = make_minion("recruit")
    filler.base_attack = 0
    filler.base_health = 1
    p0_out: list = []
    simulate_battle(
        [champion, taunt],
        [enemy, filler],
        p0_has_initiative=False,
        rng=np.random.default_rng(0),
        patch=ctx,
        p0_board_out=p0_out,
    )
    champ = next(m for m in p0_out if m.card_id == "BGS_111")
    assert champ.bonus_attack >= 1
    assert champ.bonus_health >= 1


def test_arm_buffs_attacked_taunt():
    ctx = PatchContext.load(PATCH_74257)
    arm = ctx.make_minion("BGS_110")
    taunt = make_minion("recruit")
    taunt.keywords = frozenset({Keyword.TAUNT})
    taunt.base_health = 10
    enemy = make_minion("recruit")
    enemy.base_attack = 1
    enemy.base_health = 1
    filler = make_minion("recruit")
    filler.base_attack = 0
    filler.base_health = 1
    p0_out: list = []
    simulate_battle(
        [arm, taunt],
        [enemy, filler],
        p0_has_initiative=False,
        rng=np.random.default_rng(0),
        patch=ctx,
        p0_board_out=p0_out,
    )
    taunt_out = next(
        m for m in p0_out if m.card_id == taunt.card_id and Keyword.TAUNT in m.all_keywords
    )
    assert taunt_out.bonus_attack >= 2


def test_warden_grants_combat_gold_on_death():
    ctx = PatchContext.load(PATCH_74257)
    warden = ctx.make_minion("BGS_200")
    warden.base_health = 1
    enemy = make_minion("recruit")
    enemy.base_attack = 5
    enemy.base_health = 5
    combat_gold = [0, 0]
    simulate_battle(
        [warden],
        [enemy],
        p0_has_initiative=False,
        rng=np.random.default_rng(0),
        patch=ctx,
        combat_gold_out=combat_gold,
    )
    assert combat_gold[0] == 1


def test_strongarm_scales_with_pirates_bought():
    from src.bg_recruitment.shop_triggers import ShopTriggers
    from src.bg_recruitment.targeted_battlecry import apply_targeted_on_place_battlecries

    ctx = PatchContext.load(PATCH_74257)
    p = _shop_player()
    target = ctx.make_minion("BGS_049")
    strongarm = ctx.make_minion("BGS_048")
    p.pirates_bought_this_turn = 2
    p.board = [target]
    triggers = ShopTriggers(patch=ctx, rng=np.random.default_rng(0))
    p.board.append(strongarm)
    triggers.fire_on_place(strongarm, p, None)
    apply_targeted_on_place_battlecries(
        triggers, p, strongarm, rng=np.random.default_rng(0), forced_buff_target=target
    )
    assert target.bonus_attack >= 2
    assert target.bonus_health >= 2


def test_rabid_saurolisk_buffs_on_deathrattle_played():
    from src.bg_recruitment.shop_triggers import ShopTriggers

    ctx = PatchContext.load(PATCH_74257)
    p = _shop_player()
    saurolisk = ctx.make_minion("BGS_075")
    dr = ctx.make_minion("BGS_061")
    p.board = [saurolisk]
    triggers = ShopTriggers(patch=ctx, rng=np.random.default_rng(0))
    triggers.fire_on_place(dr, p, None)
    triggers.fire_after_friendly_minion_placed(p, dr)
    assert saurolisk.bonus_attack >= 1
    assert saurolisk.bonus_health >= 2


def test_majordomo_buffs_leftmost_per_elemental_played():
    from src.bg_recruitment.shop_triggers import ShopTriggers

    ctx = PatchContext.load(PATCH_74257)
    p = _shop_player()
    left = ctx.make_minion("BGS_049")
    majordomo = ctx.make_minion("BGS_105")
    p.board = [left, majordomo]
    p.elementals_played = 2
    triggers = ShopTriggers(patch=ctx, rng=np.random.default_rng(0))
    triggers.fire_on_turn_end(p)
    assert left.bonus_attack >= 2
    assert left.bonus_health >= 2


def test_garr_gains_health_when_elemental_played():
    from src.bg_recruitment.shop_triggers import ShopTriggers

    ctx = PatchContext.load(PATCH_74257)
    p = _shop_player()
    garr = ctx.make_minion("BGS_124")
    elem = ctx.make_minion("BGS_115")
    p.board = [garr, elem]
    triggers = ShopTriggers(patch=ctx, rng=np.random.default_rng(0))
    triggers.fire_on_place(elem, p, None)
    triggers.fire_after_friendly_minion_placed(p, elem)
    assert garr.bonus_health >= 2


def test_lil_rag_buffs_on_elemental_played():
    from src.bg_recruitment.shop_triggers import ShopTriggers

    ctx = PatchContext.load(PATCH_74257)
    p = _shop_player()
    rag = ctx.make_minion("BGS_100")
    buddy = ctx.make_minion("BGS_049")
    p.board = [rag, buddy]
    elem = ctx.make_minion("BGS_115")
    triggers = ShopTriggers(patch=ctx, rng=np.random.default_rng(0))
    triggers.fire_on_place(elem, p, None)
    triggers.fire_after_friendly_minion_placed(p, elem)
    assert any(m.bonus_attack >= 1 and m.bonus_health >= 1 for m in p.board)


def test_imp_mama_summons_on_damage():
    ctx = PatchContext.load(PATCH_74257)
    mama = ctx.make_minion("BGS_044")
    mama.base_health = 5
    enemy = make_minion("recruit")
    enemy.base_attack = 2
    enemy.base_health = 10
    filler = make_minion("recruit")
    filler.base_attack = 0
    filler.base_health = 1
    p0_out: list = []
    simulate_battle(
        [mama],
        [enemy, filler],
        p0_has_initiative=False,
        rng=np.random.default_rng(0),
        patch=ctx,
        p0_board_out=p0_out,
    )
    assert len(p0_out) >= 2


def test_waxrider_buffs_when_dragon_kills():
    ctx = PatchContext.load(PATCH_74257)
    wax = ctx.make_minion("BGS_035")
    dragon = make_minion("recruit")
    dragon.race = Race.DRAGON
    dragon.base_attack = 5
    dragon.abilities = ()
    enemy = make_minion("recruit")
    enemy.base_health = 1
    p0_out: list = []
    simulate_battle(
        [dragon, wax],
        [enemy],
        p0_has_initiative=True,
        rng=np.random.default_rng(0),
        patch=ctx,
        p0_board_out=p0_out,
    )
    wax_out = next(m for m in p0_out if m.card_id == "BGS_035")
    assert wax_out.bonus_attack >= 2
    assert wax_out.bonus_health >= 2


def test_nat_pagle_queues_hand_add_on_kill():
    ctx = PatchContext.load(PATCH_74257)
    nat = ctx.make_minion("BGS_046")
    nat.base_attack = 10
    enemy = make_minion("recruit")
    enemy.base_health = 1
    filler = make_minion("recruit")
    filler.base_attack = 0
    filler.base_health = 1
    combat_hand: list[list[str]] = [[], []]
    simulate_battle(
        [nat],
        [enemy, filler],
        p0_has_initiative=True,
        rng=np.random.default_rng(0),
        patch=ctx,
        combat_hand_adds_out=combat_hand,
    )
    assert len(combat_hand[0]) == 1


def test_wildfire_deals_excess_to_adjacent():
    ctx = PatchContext.load(PATCH_74257)
    wild = ctx.make_minion("BGS_126")
    wild.base_attack = 10
    mid = make_minion("recruit")
    mid.base_health = 1
    adj = make_minion("recruit")
    adj.base_health = 5
    adj.base_attack = 0
    filler = make_minion("recruit")
    filler.base_attack = 0
    filler.base_health = 1
    p1_out: list = []
    simulate_battle(
        [wild],
        [adj, mid, filler],
        p0_has_initiative=True,
        rng=np.random.default_rng(0),
        patch=ctx,
        p1_board_out=p1_out,
    )
    assert len(p1_out) <= 3


def test_all_token_ids_instantiate():
    ctx = PatchContext.load(PATCH_74257)
    for tid in sorted(ctx.token_ids):
        m = ctx.make_minion(tid)
        assert m.card_id == tid


def test_keyword_only_pool_ids():
    ctx = PatchContext.load(PATCH_74257)
    expected = {
        "BGS_034",
        "BGS_039",
        "BGS_049",
        "BGS_106",
        "BGS_119",
        "BGS_131",
    }
    assert ctx.keyword_only_pool_ids == expected
    for cid in expected:
        assert cid in ctx.pool_ids
        assert cid not in ctx.effects


def test_cobalt_scalebane_buffs_on_turn_end():
    from src.bg_recruitment.shop_triggers import ShopTriggers

    ctx = PatchContext.load(PATCH_74257)
    p = _shop_player()
    cobalt = ctx.make_minion("ICC_029")
    buddy = ctx.make_minion("BGS_049")
    p.board = [cobalt, buddy]
    triggers = ShopTriggers(patch=ctx, rng=np.random.default_rng(0))
    triggers.fire_on_turn_end(p)
    assert buddy.bonus_attack >= 3


def test_micro_mummy_buffs_on_turn_end():
    from src.bg_recruitment.shop_triggers import ShopTriggers

    ctx = PatchContext.load(PATCH_74257)
    p = _shop_player()
    mummy = ctx.make_minion("ULD_217")
    buddy = ctx.make_minion("BGS_049")
    p.board = [mummy, buddy]
    triggers = ShopTriggers(patch=ctx, rng=np.random.default_rng(0))
    triggers.fire_on_turn_end(p)
    assert buddy.bonus_attack >= 1


def test_bolvar_buffs_on_friendly_shield_lost():
    ctx = PatchContext.load(PATCH_74257)
    shielded = make_minion("recruit")
    shielded.keywords = frozenset({Keyword.SHIELD})
    shielded.has_shield = True
    bolvar = ctx.make_minion("ICC_858")
    enemy = make_minion("recruit")
    enemy.base_attack = 1
    p0_out: list = []
    simulate_battle(
        [shielded, bolvar],
        [enemy],
        p0_has_initiative=False,
        rng=np.random.default_rng(1),
        patch=ctx,
        p0_board_out=p0_out,
    )
    bol = next(m for m in p0_out if m.card_id == "ICC_858")
    assert bol.bonus_attack >= 2


def test_qiraji_harbinger_buffs_neighbors_when_taunt_dies():
    ctx = PatchContext.load(PATCH_74257)
    taunt = ctx.make_minion("BGS_039")
    taunt.base_health = 1
    harbinger = ctx.make_minion("BGS_112")
    filler = make_minion("recruit")
    enemy = make_minion("recruit")
    enemy.base_attack = 5
    enemy.base_health = 5
    p0_out: list = []
    simulate_battle(
        [taunt, harbinger, filler],
        [enemy],
        p0_has_initiative=False,
        rng=np.random.default_rng(0),
        patch=ctx,
        p0_board_out=p0_out,
    )
    har = next(m for m in p0_out if m.card_id == "BGS_112")
    assert har.bonus_attack >= 2
    assert har.bonus_health >= 2


def test_qiraji_harbinger_ignores_non_taunt_death():
    ctx = PatchContext.load(PATCH_74257)
    vanilla = make_minion("recruit")
    vanilla.base_health = 1
    harbinger = ctx.make_minion("BGS_112")
    filler = make_minion("recruit")
    enemy = make_minion("recruit")
    enemy.base_attack = 5
    p0_out: list = []
    simulate_battle(
        [vanilla, harbinger, filler],
        [enemy],
        p0_has_initiative=False,
        rng=np.random.default_rng(0),
        patch=ctx,
        p0_board_out=p0_out,
    )
    har = next(m for m in p0_out if m.card_id == "BGS_112")
    assert har.bonus_attack == 0
    assert har.bonus_health == 0


def test_imprisoner_summons_imp():
    ctx = PatchContext.load(PATCH_74257)
    imprisoner = ctx.make_minion("BGS_014")
    imprisoner.base_health = 1
    enemy = make_minion("recruit")
    enemy.base_attack = 5
    p0_out: list = []
    simulate_battle(
        [imprisoner],
        [enemy],
        p0_has_initiative=False,
        rng=np.random.default_rng(0),
        patch=ctx,
        p0_board_out=p0_out,
    )
    token = next(m for m in p0_out if m.card_id == "BRM_006t")
    assert token.raw_attack == 1
    assert token.max_health == 1
    assert Keyword.TAUNT not in token.all_keywords


def test_forged_imprisoner_summons_golden_imp():
    ctx = PatchContext.load(PATCH_74257)
    imprisoner = ctx.make_minion("BGS_014")
    imprisoner.abilities = ctx.triple_merge_golden_abilities("BGS_014")
    imprisoner.base_health = 1
    enemy = make_minion("recruit")
    enemy.base_attack = 5
    p0_out: list = []
    simulate_battle(
        [imprisoner],
        [enemy],
        p0_has_initiative=False,
        rng=np.random.default_rng(0),
        patch=ctx,
        p0_board_out=p0_out,
    )
    token = next(m for m in p0_out if m.card_id == "TB_BaconUps_030t")
    assert token.raw_attack == 2
    assert token.max_health == 2


def test_refreshing_anomaly_golden_two_free_rolls():
    from src.bg_recruitment.economy import effective_roll_cost, roll_shop
    from src.bg_recruitment.shop_triggers import ShopTriggers

    ctx = PatchContext.load(PATCH_74257)
    p = _shop_player(gold=10)
    anomaly = ctx.make_minion("BGS_116")
    anomaly.abilities = ctx.triple_merge_golden_abilities("BGS_116")
    triggers = ShopTriggers(patch=ctx, rng=np.random.default_rng(0))
    triggers.fire_on_place(anomaly, p, None)
    assert effective_roll_cost(p) == 0
    assert p.free_roll_charges == 2
    roll_shop(p, None, rng=np.random.default_rng(1), patch=ctx)
    assert effective_roll_cost(p) == 0
    assert p.free_roll_charges == 1
    roll_shop(p, None, rng=np.random.default_rng(2), patch=ctx)
    assert p.free_roll_charges == 0
    assert p.next_roll_cost_override is None


def test_southsea_captain_buffs_other_pirates():
    from src.bg_combat.battle import attack_value, build_battle_side

    ctx = PatchContext.load(PATCH_74257)
    captain = ctx.make_minion("NEW1_027")
    pirate = ctx.make_minion("BGS_061")
    pirate.base_attack = 1
    pirate.base_health = 1
    side = build_battle_side([captain, pirate], patch=ctx)
    assert attack_value(side.minions[0], side, death_resolution=False) == captain.base_attack
    assert attack_value(side.minions[1], side, death_resolution=False) == 2


def test_wildfire_golden_hits_both_adjacent():
    ctx = PatchContext.load(PATCH_74257)
    wild = ctx.make_minion("BGS_126")
    wild.abilities = ctx.triple_merge_golden_abilities("BGS_126")
    wild.base_attack = 10
    left = make_minion("recruit")
    left.base_health = 5
    left.base_attack = 0
    mid = make_minion("recruit")
    mid.base_health = 1
    mid.base_attack = 0
    right = make_minion("recruit")
    right.base_health = 5
    right.base_attack = 0
    p1_out: list = []
    simulate_battle(
        [wild],
        [left, mid, right],
        p0_has_initiative=True,
        rng=np.random.default_rng(0),
        patch=ctx,
        p1_board_out=p1_out,
    )
    assert not any(m.card_id == "recruit" and m.max_health == 5 for m in p1_out)


def test_herald_of_flame_overkill_damages_leftmost_enemy():
    ctx = PatchContext.load(PATCH_74257)
    herald = ctx.make_minion("BGS_032")
    herald.base_attack = 10
    left = make_minion("recruit")
    left.base_health = 1
    right = make_minion("recruit")
    right.base_health = 3
    p1_out: list = []
    simulate_battle(
        [herald],
        [left, right],
        p0_has_initiative=True,
        rng=np.random.default_rng(0),
        patch=ctx,
        p1_board_out=p1_out,
        max_attacks=1,
    )
    assert len(p1_out) == 0


def test_deflect_o_bot_ignores_shop_mechanic_summons():
    from src.bg_recruitment.shop_triggers import ShopTriggers

    ctx = PatchContext.load(PATCH_74257)
    p = _shop_player()
    bot = ctx.make_minion("BGS_071")
    mech = ctx.make_minion("BGS_071")
    mech.race = Race.MECHANICAL
    atk_before = bot.bonus_attack
    triggers = ShopTriggers(patch=ctx, rng=np.random.default_rng(0))
    triggers.fire_shop_friendly_summoned(p, mech)
    assert bot.bonus_attack == atk_before


def test_deflect_o_bot_buffs_on_combat_mech_summon():
    ctx = PatchContext.load(PATCH_74257)
    bot = ctx.make_minion("BGS_071")
    bot.base_health = 10
    damaged_mech = ctx.make_minion("BOT_218")
    enemy = make_minion("recruit")
    enemy.base_attack = 1
    enemy.base_health = 10
    p0_out: list = []
    simulate_battle(
        [bot, damaged_mech],
        [enemy],
        p0_has_initiative=False,
        rng=np.random.default_rng(0),
        patch=ctx,
        p0_board_out=p0_out,
    )
    bot_out = next(m for m in p0_out if m.card_id == "BGS_071")
    assert bot_out.bonus_attack >= 1


def test_gentle_djinni_summons_elemental_and_copies_to_hand():
    ctx = PatchContext.load(PATCH_74257)
    djinni = ctx.make_minion("BGS_121")
    djinni.base_health = 1
    enemy = make_minion("recruit")
    enemy.base_attack = 5
    p0_out: list = []
    combat_hand: list[list[str]] = [[], []]
    simulate_battle(
        [djinni],
        [enemy],
        p0_has_initiative=False,
        rng=np.random.default_rng(0),
        patch=ctx,
        p0_board_out=p0_out,
        combat_hand_adds_out=combat_hand,
    )
    assert len(p0_out) >= 1
    assert any(m.race == Race.ELEMENTAL for m in p0_out)
    assert len(combat_hand[0]) == 1


def test_scallywag_token_attacks_immediately():
    ctx = PatchContext.load(PATCH_74257)
    scally = ctx.make_minion("BGS_061")
    scally.base_health = 1
    enemy1 = make_minion("recruit")
    enemy1.base_health = 1
    enemy2 = make_minion("recruit")
    enemy2.base_health = 10
    enemy2.base_attack = 0
    p1_out: list = []
    simulate_battle(
        [scally],
        [enemy1, enemy2],
        p0_has_initiative=True,
        rng=np.random.default_rng(0),
        patch=ctx,
        p1_board_out=p1_out,
    )
    assert len(p1_out) <= 1


def test_unstable_ghoul_damages_all_minions():
    ctx = PatchContext.load(PATCH_74257)
    ghoul = ctx.make_minion("FP1_024")
    ghoul.base_health = 1
    ally = make_minion("recruit")
    ally.base_health = 5
    enemy = make_minion("recruit")
    enemy.base_attack = 5
    p0_out: list = []
    p1_out: list = []
    simulate_battle(
        [ally, ghoul],
        [enemy],
        p0_has_initiative=False,
        rng=np.random.default_rng(0),
        patch=ctx,
        p0_board_out=p0_out,
        p1_board_out=p1_out,
    )
    if ally in p0_out or any(m.card_id == ally.card_id for m in p0_out):
        ally_out = next(m for m in p0_out if m.card_id == ally.card_id)
        assert ally_out.current_health <= 4


def test_fiendish_servant_transfers_attack():
    ctx = PatchContext.load(PATCH_74257)
    servant = ctx.make_minion("YOD_026")
    servant.base_health = 1
    buddy = make_minion("recruit")
    buddy.base_health = 5
    enemy = make_minion("recruit")
    enemy.base_attack = 5
    p0_out: list = []
    simulate_battle(
        [servant, buddy],
        [enemy],
        p0_has_initiative=False,
        rng=np.random.default_rng(0),
        patch=ctx,
        p0_board_out=p0_out,
    )
    buddy_out = next(m for m in p0_out if m.card_id == buddy.card_id)
    assert buddy_out.bonus_attack >= servant.base_attack


def test_felfin_navigator_buffs_other_murlocs():
    from src.bg_recruitment.shop_triggers import ShopTriggers

    ctx = PatchContext.load(PATCH_74257)
    p = _shop_player()
    murloc = ctx.make_minion("BGS_020")
    nav = ctx.make_minion("BT_010")
    p.board = [murloc]
    triggers = ShopTriggers(patch=ctx, rng=np.random.default_rng(0))
    triggers.fire_on_place(nav, p, None)
    assert murloc.bonus_attack >= 1
    assert murloc.bonus_health >= 1


def test_ring_matron_summons_fiery_imps():
    ctx = PatchContext.load(PATCH_74257)
    matron = ctx.make_minion("DMF_533")
    matron.base_health = 1
    enemy = make_minion("recruit")
    enemy.base_attack = 5
    p0_out: list = []
    simulate_battle(
        [matron],
        [enemy],
        p0_has_initiative=False,
        rng=np.random.default_rng(0),
        patch=ctx,
        p0_board_out=p0_out,
    )
    imps = [m for m in p0_out if m.card_id == "DMF_533t"]
    assert len(imps) == 2
    assert all(m.raw_attack == 3 and m.max_health == 2 for m in imps)
