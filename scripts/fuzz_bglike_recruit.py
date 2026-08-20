"""Drive every card through every recruit-phase move and report what breaks.

The lobby fuzzer (``fuzz_bglike_games.py``) only reaches the cards a random
shop happened to deal, and only through the moves the *flat* action space can
express -- which leaves Activate, Tavern spells and the whole golden half of
the pool untouched, because those are engine API with no action index yet.

This harness goes the other way round: it takes each card in the package, in
its normal and its triple-forged golden printing, and forces it through buy /
play / sell / magnetize / Activate / tavern-death / turn-start / turn-end, plus
the "a friendly did X" watcher paths, on a seat built so every target a card
could name actually exists. Anything that raises is reported; so is anything
that reaches the shop dispatcher and is silently dropped, which is the other
half-working failure mode (``_HANDLED_ELSEWHERE`` is a ``return``, not a
handler).

    python scripts/fuzz_bglike_recruit.py [--patch <dir>] [--seed N] [--random N]
"""

from __future__ import annotations

import argparse
import random
import traceback
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import src.envs  # noqa: F401  (import for side effect: module init order)
from src.bg_catalog.patch_context import PatchContext
from src.bg_core.effects import Keyword, Trigger
from src.bg_core.minion import Minion, Race
from src.bg_lobby.player import PlayerPhase, PlayerState
from src.bg_lobby.shared_pool import build_initial_shared_pool
from src.bg_recruitment import economy, place as recruitment_place, triples
from src.bg_recruitment import shop_triggers as st_mod
from src.bg_recruitment.activate import activate_abilities, activate_minion
from src.bg_recruitment.shop_triggers import ShopTriggers
from src.bg_recruitment.tavern_spells import (
    buy_tavern_spell,
    offer_tavern_spells,
    play_tavern_spell_from_hand,
)

BOARD_SIZE = 7
SHOP_SLOTS = 7
HAND_SIZE = 10

# Effects the shop dispatcher takes an early ``return`` on. Reaching it from a
# path that does not apply them itself is a card that silently does nothing.
HANDLED_ELSEWHERE = st_mod._HANDLED_ELSEWHERE


class SkipRecorder:
    """Wrap ``apply_shop_effect`` and note every silent ``_HANDLED_ELSEWHERE``."""

    def __init__(self) -> None:
        self.hits: Counter = Counter()
        self.where: Dict[str, str] = {}
        self._orig = ShopTriggers.apply_shop_effect
        self.context: str = "?"

    def install(self) -> None:
        orig = self._orig
        rec = self

        def wrapper(self_st, player, source, effect, placed=None, **kw):  # noqa: ANN001
            if isinstance(effect, HANDLED_ELSEWHERE):
                import inspect

                caller = "?"
                for fr in inspect.stack()[1:6]:
                    if fr.function != "wrapper":
                        caller = f"{Path(fr.filename).name}:{fr.lineno}:{fr.function}"
                        break
                key = f"{type(effect).__name__} <- {caller}"
                rec.hits[key] += 1
                rec.where.setdefault(
                    key,
                    f"{rec.context} | source={getattr(source, 'card_id', None)}",
                )
            return orig(self_st, player, source, effect, placed=placed, **kw)

        ShopTriggers.apply_shop_effect = wrapper  # type: ignore[assignment]

    def uninstall(self) -> None:
        ShopTriggers.apply_shop_effect = self._orig  # type: ignore[assignment]


def _filler(patch: PatchContext, rng: random.Random, n: int) -> List[Minion]:
    """Bodies that make every "name a friendly" branch reachable."""
    want_races = [Race.MECHANICAL, Race.MURLOC, Race.BEAST, Race.DEMON, Race.UNDEAD]
    out: List[Minion] = []
    by_race: Dict[Race, List[str]] = defaultdict(list)
    for cid, tpl in patch.templates.items():
        if tpl.is_token or tpl.is_golden:
            continue
        if tpl.race is not None:
            by_race[tpl.race].append(cid)
    for race in want_races[:n]:
        pool = sorted(by_race.get(race, []))
        if pool:
            out.append(patch.make_minion(rng.choice(pool)))
    while len(out) < n:
        out.append(patch.make_minion(rng.choice(sorted(patch.pool_ids))))
    return out[:n]


def _seat(patch: PatchContext, rng: random.Random, *, board_n: int = 3) -> PlayerState:
    p = PlayerState(
        health=30,
        gold=10,
        tavern_tier=6,
        board=_filler(patch, rng, board_n),
        shop=[patch.make_minion(rng.choice(sorted(patch.pool_ids))) for _ in range(SHOP_SLOTS)],
        hand=[None] * HAND_SIZE,
        phase=PlayerPhase.SHOP,
        shop_actions_used=0,
        ruleset=patch.meta.ruleset,
    )
    return p


def _golden(patch: PatchContext, card_id: str) -> Minion:
    return triples.make_forged_golden_minion(card_id, patch=patch)


def _body(patch: PatchContext, card_id: str, golden: bool) -> Minion:
    return _golden(patch, card_id) if golden else patch.make_minion(card_id)


# ---------------------------------------------------------------- moves

def move_buy(patch, rng, card_id, golden, rec):
    p = _seat(patch, rng)
    p.shop[0] = _body(patch, card_id, golden)
    trg = ShopTriggers(np.random.default_rng(0), patch=patch)
    pool = build_initial_shared_pool(None, patch=patch)
    economy.buy_from_shop(
        p,
        0,
        patch=patch,
        on_bought=lambda m, pl: (trg.fire_on_buy(m, pl), trg.fire_on_bought(pl, m)),
        on_friendly_bought=trg.fire_on_friendly_bought,
        on_triples=lambda pl: triples.resolve_triples_loop(pl, shared_pool=pool, patch=patch),
        shared_pool=pool,
    )


def move_play(patch, rng, card_id, golden, rec):
    p = _seat(patch, rng)
    p.hand[0] = _body(patch, card_id, golden)
    trg = ShopTriggers(np.random.default_rng(0), patch=patch)
    pool = build_initial_shared_pool(None, patch=patch)
    recruitment_place.place_from_hand(
        p,
        0,
        None,
        board_size=BOARD_SIZE,
        triggers=trg,
        rng=np.random.default_rng(0),
        shared_pool=pool,
    )
    # A battlecry that opened a modal has to be closed by the seat; the pick
    # itself is exercised by the lobby fuzzer, so just settle the after-place.
    if p.pending_choice is None and p.placed_minion_pending_after is not None:
        trg.fire_after_friendly_minion_placed(p, p.placed_minion_pending_after)


def move_sell(patch, rng, card_id, golden, rec):
    p = _seat(patch, rng)
    p.board.insert(0, _body(patch, card_id, golden))
    trg = ShopTriggers(np.random.default_rng(0), patch=patch)
    pool = build_initial_shared_pool(None, patch=patch)
    economy.sell_from_board(
        p,
        0,
        on_sell=lambda m, pl: trg.fire_on_sell(m, pl, shared_pool=pool),
        on_triples=lambda pl: triples.resolve_triples_loop(pl, shared_pool=pool, patch=patch),
        shared_pool=pool,
    )


def move_activate(patch, rng, card_id, golden, rec):
    body = _body(patch, card_id, golden)
    if not activate_abilities(body):
        return "n/a"
    p = _seat(patch, rng)
    p.board.insert(0, body)
    p.gold = 10
    activate_minion(
        p,
        0,
        rng=np.random.default_rng(0),
        patch=patch,
        shared_pool=build_initial_shared_pool(None, patch=patch),
        buff_target=p.board[1] if len(p.board) > 1 else None,
        shop_target_index=0,
    )


def move_tavern_death(patch, rng, card_id, golden, rec):
    body = _body(patch, card_id, golden)
    if not any(ab.trigger is Trigger.ON_DEATH for ab in body.abilities):
        return "n/a"
    p = _seat(patch, rng)
    p.board.insert(0, body)
    ShopTriggers(np.random.default_rng(0), patch=patch).fire_tavern_deathrattle(body, p)


def move_turn_start(patch, rng, card_id, golden, rec):
    p = _seat(patch, rng)
    p.board.insert(0, _body(patch, card_id, golden))
    ShopTriggers(np.random.default_rng(0), patch=patch).fire_on_turn_start(p)


def move_turn_end(patch, rng, card_id, golden, rec):
    p = _seat(patch, rng)
    p.board.insert(0, _body(patch, card_id, golden))
    ShopTriggers(np.random.default_rng(0), patch=patch).fire_on_turn_end(p)


def move_watch_friendly(patch, rng, card_id, golden, rec):
    """The card sits on the board while another minion is bought, played, sold."""
    p = _seat(patch, rng, board_n=2)
    p.board.insert(0, _body(patch, card_id, golden))
    trg = ShopTriggers(np.random.default_rng(0), patch=patch)
    pool = build_initial_shared_pool(None, patch=patch)
    other = patch.make_minion(rng.choice(sorted(patch.pool_ids)))
    p.shop[0] = other
    economy.buy_from_shop(
        p,
        0,
        patch=patch,
        on_bought=lambda m, pl: (trg.fire_on_buy(m, pl), trg.fire_on_bought(pl, m)),
        on_friendly_bought=trg.fire_on_friendly_bought,
        on_triples=lambda pl: triples.resolve_triples_loop(pl, shared_pool=pool, patch=patch),
        shared_pool=pool,
    )
    h = next((i for i, c in enumerate(p.hand) if c is other), None)
    if h is not None and len(p.board) < BOARD_SIZE:
        recruitment_place.place_from_hand(
            p, h, None, board_size=BOARD_SIZE, triggers=trg,
            rng=np.random.default_rng(0), shared_pool=pool,
        )
    if len(p.board) > 1:
        economy.sell_from_board(
            p,
            len(p.board) - 1,
            on_sell=lambda m, pl: trg.fire_on_sell(m, pl, shared_pool=pool),
            on_triples=lambda pl: triples.resolve_triples_loop(pl, shared_pool=pool, patch=patch),
            shared_pool=pool,
        )


def move_magnetize(patch, rng, card_id, golden, rec):
    body = _body(patch, card_id, golden)
    if not recruitment_place.hand_minion_can_magnetize(body):
        return "n/a"
    p = _seat(patch, rng)
    target = next((i for i, m in enumerate(p.board) if recruitment_place.is_mech(m)), None)
    if target is None:
        p.board.insert(0, patch.make_minion(
            next(cid for cid, t in sorted(patch.templates.items())
                 if t.race == Race.MECHANICAL and not t.is_token)
        ))
        target = 0
    p.hand[0] = body
    recruitment_place.magnet_from_hand(p, 0, target, patch=patch)


def move_retrigger(patch, rng, card_id, golden, rec):
    """Brann-style "trigger a friendly minion's Battlecry" aimed at this card."""
    body = _body(patch, card_id, golden)
    if not any(ab.trigger is Trigger.ON_PLACE for ab in body.abilities):
        return "n/a"
    from src.bg_core.effects import RetriggerFriendlyAbilityEffect

    p = _seat(patch, rng)
    p.board.insert(0, body)
    trg = ShopTriggers(np.random.default_rng(0), patch=patch)
    trg.retrigger_friendly_ability(
        p, body, RetriggerFriendlyAbilityEffect(trigger=Trigger.ON_PLACE), shared_pool=None
    )


MOVES = {
    "buy": move_buy,
    "retrigger": move_retrigger,
    "play": move_play,
    "sell": move_sell,
    "activate": move_activate,
    "tavern_death": move_tavern_death,
    "turn_start": move_turn_start,
    "turn_end": move_turn_end,
    "watch_friendly": move_watch_friendly,
    "magnetize": move_magnetize,
}


# ------------------------------------------------------- tavern spells

def sweep_tavern_spells(patch, rng, failures, first, stats, rec):
    spells = getattr(patch, "tavern_spells", {}) or {}
    for card_id, spell in sorted(spells.items()):
        if not getattr(spell, "is_tavern_spell", False):
            continue
        for kind, kwargs in (
            ("no_target", {}),
            ("board_target", {"target_board_index": 0}),
            ("shop_target", {"target_shop_index": 0}),
            ("choose_one_1", {"choose_one_option": 1, "target_board_index": 0}),
        ):
            rec.context = f"tavern_spell:{card_id}:{kind}"
            stats["tavern_spell_casts"] += 1
            try:
                p = _seat(patch, rng)
                p.gold = 20
                offer_tavern_spells(p, rng=np.random.default_rng(0), patch=patch,
                                    card_ids=[card_id])
                buy_tavern_spell(p, 0, patch=patch)
                h = next(i for i, c in enumerate(p.hand) if c is not None)
                play_tavern_spell_from_hand(
                    p,
                    h,
                    rng=np.random.default_rng(0),
                    patch=patch,
                    shared_pool=build_initial_shared_pool(None, patch=patch),
                    **kwargs,
                )
            except Exception as exc:  # noqa: BLE001
                key = f"tavern_spell/{kind}: {type(exc).__name__}: {str(exc)[:160]}"
                failures[key] += 1
                first.setdefault(key, (card_id, traceback.format_exc()))


# ------------------------------------------------- random move sequences

def random_seat_game(patch: PatchContext, seed: int, stats: Counter, rec: SkipRecorder,
                     *, turns: int = 20) -> None:
    """One seat, many turns, moves drawn from the *whole* recruit-phase API.

    The flat action space is only part of it: Activate, buying and casting a
    Tavern spell, and naming a shop slot as a target all exist as engine calls
    with no action index, so a fuzzer that only walks the legal mask never
    reaches them.
    """
    from src.bg_lobby.eight_player import PlayerPhase as _PP  # noqa: F401
    from src.bg_player_turn import PlayerTurnContext, PlayerTurnEngine
    from src.bg_recruitment.activate import activate_minion, can_activate
    from src.bg_recruitment.economy import accrue_upgrade_discount, start_of_turn_gold
    from src.bg_recruitment.blood_gems import BLOOD_GEM_CARD_ID, play_blood_gem_on
    from src.bg_recruitment.choose_one import resolve_choose_one
    from src.bg_lobby.player import PendingChoiceKind
    from src.bg_recruitment.shop import refresh_shop
    from src.envs.bglike import actions as bglike_actions

    rng_py = random.Random(seed)
    rng = np.random.default_rng(seed)
    triggers = ShopTriggers(rng, patch=patch)
    pool = build_initial_shared_pool(None, patch=patch)
    engine = PlayerTurnEngine(bglike_actions)
    ruleset = patch.meta.ruleset

    p = PlayerState(
        health=ruleset.starting_health,
        gold=3,
        tavern_tier=1,
        board=[],
        shop=[None] * SHOP_SLOTS,
        hand=[None] * HAND_SIZE,
        phase=PlayerPhase.SHOP,
        shop_actions_used=0,
        ruleset=ruleset,
    )
    refresh_shop(p, None, rng=rng, shared_pool=pool, frozen_slots=p.shop_frozen, patch=patch)
    ctx = PlayerTurnContext(
        rng=rng, triggers=triggers, shop_excluded_race=None, shared_pool=pool, patch=patch
    )

    for turn in range(1, turns + 1):
        accrue_upgrade_discount(p)
        p.gold = start_of_turn_gold(p, turn)
        p.phase = PlayerPhase.SHOP
        p.shop_actions_used = 0
        p.pending_choice = None
        p.triple_reward_discover_pending = False
        p.triple_reward_spell_tier = 0
        p.placed_minion_board_index = None
        p.placed_minion_pending_after = None
        rec.context = f"random seat seed={seed} turn={turn} turn_start"
        triggers.fire_on_turn_start(p)
        refresh_shop(p, None, rng=rng, shared_pool=pool, frozen_slots=p.shop_frozen, patch=patch)
        stats["random_turns"] += 1

        for _ in range(40):
            flat = [
                a
                for a in engine.legal_actions(p, ruleset)
                if a not in (int(bglike_actions.Action.FINISH),
                             int(bglike_actions.Action.FINISH_FREEZE_SHOP))
            ]
            extra: List[Tuple[str, Any]] = []
            if p.pending_choice is None:
                for i, m in enumerate(p.board):
                    if can_activate(p, m):
                        extra.append(("activate", i))
                for i, sp in enumerate(p.tavern_spell_offers):
                    extra.append(("buy_spell", i))
                for h, c in enumerate(p.hand):
                    if c is None or isinstance(c, Minion):
                        continue
                    if getattr(c, "card_id", None) == BLOOD_GEM_CARD_ID:
                        extra.append(("blood_gem", h))
                    elif getattr(c, "is_tavern_spell", False):
                        extra.append(("cast_spell", h))
            else:
                pc = p.pending_choice
                if pc.kind == PendingChoiceKind.CHOOSE_ONE:
                    # The flat action space routes this into the Discover
                    # resolver, which is bug #1; resolve it the way the module
                    # that opened it means it to be resolved so the run goes on.
                    stats["known_choose_one_unresolvable"] += 1
                    resolve_choose_one(
                        p,
                        rng_py.randrange(len(pc.effects)),
                        apply_effect=lambda src, eff: triggers.apply_shop_effect(
                            p, src, eff, placed=None
                        ),
                    )
                    continue
            moves = [("flat", a) for a in flat] + extra
            if not moves or rng_py.random() < 0.06:
                break
            kind, arg = rng_py.choice(moves)
            rec.context = f"random seat seed={seed} turn={turn} {kind}:{arg}"
            stats[f"random_{kind}"] += 1
            if kind == "flat":
                if engine.apply(p, arg, ctx):
                    p.shop_actions_used += 1
            elif kind == "activate":
                activate_minion(
                    p, arg, rng=rng, patch=patch, shared_pool=pool,
                    buff_target=rng_py.choice(p.board) if p.board else None,
                    shop_target_index=rng_py.randrange(SHOP_SLOTS),
                )
            elif kind == "buy_spell":
                try:
                    buy_tavern_spell(p, arg, patch=patch)
                except Exception as exc:
                    if "costs" not in str(exc) and "hand is full" not in str(exc):
                        raise
            elif kind == "blood_gem":
                if p.board:
                    play_blood_gem_on(p, rng_py.choice(p.board), count=1, patch=patch)
                p.hand[arg] = None
            elif kind == "cast_spell":
                play_tavern_spell_from_hand(
                    p, arg, rng=rng, patch=patch,
                    target_board_index=(rng_py.randrange(len(p.board)) if p.board and rng_py.random() < 0.7 else None),
                    target_shop_index=(rng_py.randrange(SHOP_SLOTS) if rng_py.random() < 0.4 else None),
                    choose_one_option=rng_py.randrange(2),
                    shared_pool=pool,
                )
        rec.context = f"random seat seed={seed} turn={turn} turn_end"
        triggers.fire_on_turn_end(p)


# --------------------------------------------------------------- main

def sweep(patch, seed, failures, first, stats, rec, *, goldens: bool):
    rng = random.Random(seed)
    card_ids = sorted(patch.pool_ids)
    for card_id in card_ids:
        for golden in (False, True) if goldens else (False,):
            for name, fn in MOVES.items():
                rec.context = f"{card_id}{'_G' if golden else ''}:{name}"
                try:
                    out = fn(patch, rng, card_id, golden, rec)
                except Exception as exc:  # noqa: BLE001
                    key = (
                        f"{name}{'/golden' if golden else ''}: "
                        f"{type(exc).__name__}: {str(exc)[:160]}"
                    )
                    failures[key] += 1
                    first.setdefault(key, (card_id, traceback.format_exc()))
                    continue
                if out == "n/a":
                    continue
                stats[f"{name}{'_golden' if golden else ''}"] += 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--patch", default="data/bgcore/36_2_0_248348")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-goldens", action="store_true")
    ap.add_argument("--traceback", action="store_true")
    ap.add_argument("--max-report", type=int, default=40)
    ap.add_argument("--skips", action="store_true", help="list silent dispatcher skips")
    ap.add_argument("--random", type=int, default=0, help="also run N random seat-games")
    ap.add_argument("--no-sweep", action="store_true")
    args = ap.parse_args()

    patch = PatchContext.load(Path(args.patch))
    failures: Counter = Counter()
    first: Dict[str, Tuple[str, str]] = {}
    stats: Counter = Counter()
    rec = SkipRecorder()
    rec.install()
    try:
        if not args.no_sweep:
            sweep(patch, args.seed, failures, first, stats, rec, goldens=not args.no_goldens)
            sweep_tavern_spells(patch, random.Random(args.seed), failures, first, stats, rec)
        for i in range(args.random):
            seed = args.seed + i
            try:
                random_seat_game(patch, seed, stats, rec)
                stats["random_games"] += 1
            except Exception as exc:  # noqa: BLE001
                key = f"random_seat: {type(exc).__name__}: {str(exc)[:160]}"
                failures[key] += 1
                first.setdefault(key, (f"seed {seed} @ {rec.context}", traceback.format_exc()))
    finally:
        rec.uninstall()

    print(f"patch {args.patch}: {len(patch.pool_ids)} pool cards")
    print("  moves exercised: " + ", ".join(f"{k}={v}" for k, v in sorted(stats.items())))
    if args.skips:
        print("\nsilently skipped at the shop dispatcher (_HANDLED_ELSEWHERE early return):")
        for key, n in rec.hits.most_common():
            print(f"  {n:>5}x {key}   first: {rec.where[key]}")
    if not failures:
        print("\nno failures")
        return 0
    print(f"\n{sum(failures.values())} failures, {len(failures)} distinct:")
    for key, count in failures.most_common(args.max_report):
        card, tb = first[key]
        print(f"\n{count:>4}x  (first card {card})\n      {key}")
        if args.traceback:
            print(tb)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
