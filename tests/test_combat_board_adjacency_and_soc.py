"""Board-order rules that a corpse used to distort: cleave adjacency and the
Start-of-Combat firing order."""

from pathlib import Path

import numpy as np

from src.bg_catalog.cards import make_minion
from src.bg_catalog.patch_context import PatchContext
from src.bg_combat.battle.state import BattleSide, _CombatRuntime
from src.bg_combat.battle.sides import _build_side
from src.bg_combat.battle.simulate import simulate_battle
from src.bg_combat.battle.summon import _insert_idx_after, _summon_insert
from src.bg_combat.battle.targeting import _cleave_victim_ids_at_swing_start

PATCH = PatchContext.load(Path("data/bgcore/19_6_0_74257"))

BODY = "CS2_065"  # Voidwalker: a plain body, no triggers to muddy the setup
RED_WHELP = "BGS_019"  # Start of Combat: 1 damage per friendly Dragon
DRAGON = "BGS_038"  # Twilight Emissary, a second dragon for the whelp's count


def _runtime():
    return _CombatRuntime(
        sides=(BattleSide(), BattleSide()),
        rng=np.random.default_rng(0),
        combat_board_max=7,
        damage_cap=15,
        patch=PATCH,
    )


def _side(rt, n, card_id=BODY):
    side = _build_side([make_minion(card_id, patch=PATCH) for _ in range(n)], rt)
    rt.sides = (side, BattleSide())
    return side


def test_cleave_splashes_across_a_corpse():
    """A dead body between two minions is off the board, so they are neighbours."""
    rt = _runtime()
    side = _side(rt, 4)
    a, corpse, target, c = side.minions
    corpse.damage_taken = corpse.max_health + corpse.aura_health
    side.reap_dead()  # bodies leave the board where they die

    ids = _cleave_victim_ids_at_swing_start(side, target)

    assert [m.instance_id for m in side.alive_minions()] == [
        a.instance_id,
        target.instance_id,
        c.instance_id,
    ]
    assert ids == [a.instance_id, c.instance_id]


def test_cleave_splashes_left_of_a_deathrattle_token():
    """Tokens are inserted right behind the body that summoned them, so the
    corpse sat between the token and its left neighbour on every such board."""
    rt = _runtime()
    side = _side(rt, 3)
    a, summoner, c = side.minions
    summoner.damage_taken = summoner.max_health + summoner.aura_health
    token = _summon_insert(
        rt, 0, make_minion(BODY, patch=PATCH), _insert_idx_after(side, summoner)
    )

    ids = _cleave_victim_ids_at_swing_start(side, token)

    assert ids == [a.instance_id, c.instance_id]


def test_cleave_counts_a_summoned_token_as_a_neighbour():
    rt = _runtime()
    side = _side(rt, 3)
    _a, summoner, c = side.minions
    summoner.damage_taken = summoner.max_health + summoner.aura_health
    token = _summon_insert(
        rt, 0, make_minion(BODY, patch=PATCH), _insert_idx_after(side, summoner)
    )

    assert token.instance_id in _cleave_victim_ids_at_swing_start(side, c)


def _whelp_board():
    return [make_minion(RED_WHELP, patch=PATCH), make_minion(DRAGON, patch=PATCH)]


def _soc_fire_order(seed):
    """Which side each Start-of-Combat trigger fired from, in order."""
    import src.bg_combat.battle.engine as engine

    order = []
    original = engine._deal_random_enemy_minion_damage

    def spy(rt, from_side_idx, amount):
        order.append(from_side_idx)
        return original(rt, from_side_idx, amount)

    engine._deal_random_enemy_minion_damage = spy
    try:
        simulate_battle(
            _whelp_board(),
            _whelp_board(),
            p0_has_initiative=True,
            rng=np.random.default_rng(seed),
            combat_board_max=7,
            damage_cap=15,
            max_board_slots=7,
            patch=PATCH,
        )
    finally:
        engine._deal_random_enemy_minion_damage = original
    return order


def test_start_of_combat_dominant_side_is_drawn_not_fixed():
    """Side 0 is always the lower seat (pairings are emitted with a < b), so a
    fixed side-0-first order handed it every Start-of-Combat race."""
    firsts = {tuple(_soc_fire_order(s))[:1] for s in range(40) if _soc_fire_order(s)}
    assert firsts == {(0,), (1,)}


def _fire_soc_on(boards, seed=0):
    """Run only the Start-of-Combat phase and report which side each trigger came from."""
    import src.bg_combat.battle.engine as engine

    rt = _CombatRuntime(
        sides=(BattleSide(), BattleSide()),
        rng=np.random.default_rng(seed),
        combat_board_max=7,
        damage_cap=15,
        patch=PATCH,
    )
    rt.sides = (_build_side(boards[0], rt), _build_side(boards[1], rt))
    order = []
    original = engine._deal_random_enemy_minion_damage

    def spy(r, from_side_idx, amount):
        order.append(from_side_idx)
        return original(r, from_side_idx, amount)

    engine._deal_random_enemy_minion_damage = spy
    try:
        engine._fire_start_of_combat(rt)
    finally:
        engine._deal_random_enemy_minion_damage = original
    return order


def test_start_of_combat_alternates_between_sides():
    """One whelp a side, both survive (a lone dragon deals 1 into 2 health), so
    both fire and the two triggers must come from different sides."""
    board = lambda: [
        make_minion(RED_WHELP, patch=PATCH),
        make_minion(BODY, patch=PATCH),
    ]
    order = _fire_soc_on((board(), board()))

    assert len(order) == 2
    assert order in ([0, 1], [1, 0])


def test_start_of_combat_skips_a_trigger_killed_before_its_turn():
    """Deaths resolve between triggers, so a whelp that dies to the opposing
    whelp never fires. Side 1 holds a single body, so the target is forced."""
    side0 = [make_minion(RED_WHELP, patch=PATCH)] + [
        make_minion(DRAGON, patch=PATCH) for _ in range(3)
    ]
    side1 = [make_minion(RED_WHELP, patch=PATCH)]
    seen = set()
    for seed in range(30):
        order = _fire_soc_on(([make_minion(m.card_id, patch=PATCH) for m in side0],
                              [make_minion(m.card_id, patch=PATCH) for m in side1]),
                             seed=seed)
        seen.add(tuple(order))
        if order and order[0] == 0:
            # 4 dragons → 4 damage into the lone whelp: it dies before its turn.
            assert order == [0]
    assert (0,) in seen


def test_start_of_combat_mirror_is_not_a_side_0_sweep():
    """The seat asymmetry this produced end to end: a perfect whelp mirror used
    to be won by side 0 in 83.7% of combats and by side 1 in none."""
    wins = [0, 0]
    for seed in range(400):
        board = lambda: [make_minion(RED_WHELP, patch=PATCH) for _ in range(2)] + [
            make_minion(DRAGON, patch=PATCH) for _ in range(2)
        ]
        result = simulate_battle(
            board(),
            board(),
            p0_has_initiative=bool(seed % 2),
            rng=np.random.default_rng(seed),
            combat_board_max=7,
            damage_cap=15,
            max_board_slots=7,
            patch=PATCH,
        )
        if result.damage_p1 > 0:
            wins[0] += 1
        elif result.damage_p0 > 0:
            wins[1] += 1
    assert wins[1] > 0
    assert 0.4 < wins[0] / max(1, wins[0] + wins[1]) < 0.6


def test_tier_five_upgrade_costs_nine_on_this_patch():
    """11 gold arrived in patch 22.2.0 (Jan 2022); 19.6 is a year earlier."""
    assert PATCH.meta.ruleset.level_up_cost(4) == 9
    assert [PATCH.meta.ruleset.level_up_cost(t) for t in (1, 2, 3)] == [5, 7, 8]
