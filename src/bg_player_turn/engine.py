"""Single-player recruitment turn: legal actions, apply, end turn."""

from __future__ import annotations

from types import ModuleType
from typing import List, Optional, Sequence

from src.bg_catalog.ruleset import Ruleset
from src.bg_core.minion import Race
from src.bg_recruitment import discover as recruitment_discover
from src.bg_recruitment import economy as recruitment_economy
from src.bg_recruitment import hero_passives
from src.bg_recruitment import place as recruitment_place
from src.bg_recruitment import triples as recruitment_triples
from src.bg_recruitment.hand_slots import hand_has_free_slot, hand_size
from src.bg_recruitment.shop import effective_shop_offers_count, toggle_shop_slot_frozen
from src.bg_recruitment.shop_triggers import ShopTriggers
from src.bg_recruitment.spellcraft import flush_pending_spellcraft
from src.bg_core.minion import is_locked
from src.bg_lobby.player import PlayerPhase, PlayerState, PendingChoiceKind

from .context import PlayerTurnContext


class PlayerTurnEngine:
    """Shop-phase rules for one ``PlayerState``; does not advance lobby / match."""

    def __init__(self, actions_mod: Optional[ModuleType] = None) -> None:
        if actions_mod is None:
            from src.envs import minibg as _pkg  # noqa: F401

            from src.envs.minibg import actions as actions_mod
        self._a = actions_mod

    def is_active(self, player: PlayerState) -> bool:
        return player.phase == PlayerPhase.SHOP

    def is_finished(self, player: PlayerState) -> bool:
        return player.phase == PlayerPhase.DONE

    def legal_actions(self, player: PlayerState, ruleset: Ruleset) -> Sequence[int]:
        """Legal shop actions for ``player`` under the active patch's ``ruleset``.

        The ruleset is required rather than defaulted: the tier ceiling below is
        exactly the kind of number a patch changes (tier 7), and a default would
        silently keep the old one.
        """
        a = self._a
        if player.phase == PlayerPhase.DONE:
            return []

        if player.pending_choice is not None:
            pc = player.pending_choice
            if pc.kind == PendingChoiceKind.TRANSFORM_SHOP_MINION:
                n_offers = a.shop_offers_count(player.tavern_tier)
                return [
                    int(a.Action.BUY_SLOT_0) + slot
                    for slot in range(min(n_offers, a.MAX_SHOP_SLOTS))
                    if player.shop[slot] is not None
                ]
            # Choose One offers two, every other kind three. Offering a third
            # on a two-option choice is an action that can only crash.
            n_options = len(pc.options) if pc.options else 3
            return [
                int(a.Action.DISCOVER_PICK_0) + i for i in range(min(3, n_options))
            ]

        actions: List[int] = []
        can_act = player.shop_actions_used < a.MAX_SHOP_ACTIONS
        hsz = hand_size(player)
        hand_full = sum(1 for m in player.hand if m is not None) >= hsz
        board_full = len(player.board) >= a.BOARD_SIZE

        if can_act:
            if not hand_full:
                n_offers = effective_shop_offers_count(player)
                buy_cost = recruitment_economy.effective_buy_cost(player)
                for slot in range(min(n_offers, len(player.shop))):
                    if (
                        player.shop[slot] is not None
                        and player.gold >= buy_cost
                    ):
                        actions.append(int(a.Action.BUY_SLOT_0) + slot)

            for pos in range(a.BOARD_SIZE):
                if pos < len(player.board):
                    actions.append(int(a.Action.SELL_BOARD_0) + pos)

            for h in range(hsz):
                hm = player.hand[h]
                if hm is None or is_locked(hm):
                    continue
                if recruitment_triples.is_triple_reward_discover_spell(hm):
                    actions.append(int(a.Action.PLAY_HAND_0) + h)
                    continue
                if board_full:
                    continue
                bc_mult = ShopTriggers.battlecry_multiplier(player.board)
                needs = recruitment_discover.discover_cards_to_receive(hm, bc_mult)
                free = sum(1 for s in player.hand if s is None) + 1
                if needs > free:
                    continue
                actions.append(int(a.Action.PLAY_HAND_0) + h)

            for h in range(hsz):
                hm = player.hand[h]
                if (
                    hm is None
                    or is_locked(hm)
                    or not recruitment_place.hand_minion_can_magnetize(hm)
                ):
                    continue
                for b in range(len(player.board)):
                    if recruitment_place.is_mech(player.board[b]):
                        actions.append(
                            int(a.Action.MAGNET_HAND_0_BOARD_0) + h * a.BOARD_SIZE + b
                        )

            if player.gold >= recruitment_economy.effective_roll_cost(player):
                actions.append(int(a.Action.ROLL))

            if (
                player.tavern_tier < ruleset.max_tier
                and player.gold >= recruitment_economy.effective_level_up_cost(player)
            ):
                actions.append(int(a.Action.LEVEL_UP))

            n_offers = effective_shop_offers_count(player)
            for slot in range(min(n_offers, a.MAX_SHOP_SLOTS)):
                if player.shop[slot] is not None and hasattr(a.Action, "FREEZE_SHOP_SLOT_0"):
                    actions.append(int(a.Action.FREEZE_SHOP_SLOT_0) + slot)

        actions.append(int(a.Action.FINISH))
        actions.append(int(a.Action.FINISH_FREEZE_SHOP))
        return actions

    def apply(
        self,
        player: PlayerState,
        action: int,
        ctx: PlayerTurnContext,
        *,
        shop_excluded_race: Optional[Race] = None,
    ) -> bool:
        a = self._a
        race = (
            shop_excluded_race
            if shop_excluded_race is not None
            else ctx.shop_excluded_race
        )
        action_int = int(action)

        if action_int == int(a.Action.FINISH) or action_int == int(a.Action.FINISH_FREEZE_SHOP):
            raise ValueError("use end_turn() for FINISH / FINISH_FREEZE_SHOP")

        if player.pending_choice is not None:
            pc = player.pending_choice
            if pc.kind == PendingChoiceKind.TRANSFORM_SHOP_MINION:
                if not (
                    int(a.Action.BUY_SLOT_0)
                    <= action_int
                    < int(a.Action.BUY_SLOT_0) + a.MAX_SHOP_SLOTS
                ):
                    raise ValueError(
                        f"Expected BUY_SLOT_* while transform modal open, got {action_int}"
                    )
                from src.bg_recruitment.faceless import resolve_transform_shop_pick

                resolve_transform_shop_pick(
                    player,
                    action_int - int(a.Action.BUY_SLOT_0),
                    patch=ctx.patch,
                    on_after_placed=ctx.triggers.fire_after_friendly_minion_placed,
                )
                recruitment_triples.resolve_triples_loop(
                    player, shared_pool=ctx.shared_pool, patch=ctx.patch
                )
                recruitment_triples.flush_triple_reward_queue_if_idle(
                    player, race, rng=ctx.rng, patch=ctx.patch
                )
                flush_pending_spellcraft(player)
                return True
            if not a.is_discover_pick_game_action(action_int):
                raise ValueError(
                    f"Expected DISCOVER_PICK_* while pending_choice, got {action_int}"
                )
            if pc.kind is PendingChoiceKind.CHOOSE_ONE:
                # Its own resolver, which has existed and been tested since the
                # keyword was built — nothing ever routed a pick to it, so every
                # Choose One card fell through to the Discover path and either
                # dropped the chosen half or raised on an option that is not a
                # card id.
                from src.bg_recruitment.choose_one import (
                    fire_choose_one_played,
                    resolve_choose_one,
                )

                resolve_choose_one(
                    player,
                    a.discover_pick_index(action_int),
                    apply_effect=lambda src, eff: ctx.triggers.apply_shop_effect(
                        player,
                        src,
                        eff,
                        placed=src,
                        shop_excluded_race=race,
                        shared_pool=ctx.shared_pool,
                    ),
                    fire_played_listeners=lambda p: fire_choose_one_played(
                        p,
                        lambda src, eff: ctx.triggers.apply_shop_effect(
                            player, src, eff, placed=None
                        ),
                    ),
                )
                recruitment_triples.resolve_triples_loop(
                    player, shared_pool=ctx.shared_pool, patch=ctx.patch
                )
                return (
                    player.pending_choice is None
                    and player.placed_minion_pending_after is None
                )
            recruitment_discover.resolve_discover_pick(
                player,
                a.discover_pick_index(action_int),
                race,
                rng=ctx.rng,
                on_after_placed=ctx.triggers.fire_after_friendly_minion_placed,
                shared_pool=ctx.shared_pool,
                patch=ctx.patch,
            )
            return (
                player.pending_choice is None
                and player.placed_minion_pending_after is None
            )

        if int(a.Action.BUY_SLOT_0) <= action_int < int(a.Action.BUY_SLOT_0) + a.MAX_SHOP_SLOTS:
            def _on_bought(m, p):
                # Bookkeeping only (tribe-preference shaping reads the delta):
                # fires on the buyer, before any trigger can move the card.
                race = getattr(m, "race", None)
                p.bought_tribe_counts[race] = p.bought_tribe_counts.get(race, 0) + 1
                ctx.triggers.fire_on_buy(m, p)
                ctx.triggers.fire_on_bought(p, m)  # watchers that pay the card bought
                hero_passives.apply_hero_on_bought(m, p)  # Kael'thas / Rat King

            recruitment_economy.buy_from_shop(
                player,
                action_int - int(a.Action.BUY_SLOT_0),
                patch=ctx.patch,
                on_bought=_on_bought,
                on_friendly_bought=ctx.triggers.fire_on_friendly_bought,
                on_triples=lambda p: recruitment_triples.resolve_triples_loop(
                    p, shared_pool=ctx.shared_pool, patch=ctx.patch
                ),
                shared_pool=ctx.shared_pool,
            )
            return True

        if int(a.Action.SELL_BOARD_0) <= action_int < int(a.Action.SELL_BOARD_0) + a.BOARD_SIZE:
            def _on_sell(m, p):
                ctx.triggers.fire_on_sell(
                    m, p, shop_excluded_race=race, shared_pool=ctx.shared_pool
                )
                hero_passives.apply_hero_on_sell(  # Dancin' Deryl / Flurgl
                    m,
                    p,
                    rng=ctx.rng,
                    patch=ctx.patch,
                    shared_pool=ctx.shared_pool,
                    shop_excluded_race=race,
                )

            recruitment_economy.sell_from_board(
                player,
                action_int - int(a.Action.SELL_BOARD_0),
                on_sell=_on_sell,
                on_triples=lambda p: recruitment_triples.resolve_triples_loop(
                    p, shared_pool=ctx.shared_pool, patch=ctx.patch
                ),
                shared_pool=ctx.shared_pool,
            )
            return True

        if action_int == int(a.Action.ROLL):
            recruitment_economy.roll_shop(
                player, race, rng=ctx.rng, shared_pool=ctx.shared_pool, patch=ctx.patch
            )
            return True

        if action_int == int(a.Action.LEVEL_UP):
            recruitment_economy.level_up_tavern(
                player, race, rng=ctx.rng, shared_pool=ctx.shared_pool, patch=ctx.patch
            )
            hero_passives.apply_hero_on_level_up(player)  # Forest Warden Omu
            return True

        if hasattr(a.Action, "FREEZE_SHOP_SLOT_0"):
            if int(a.Action.FREEZE_SHOP_SLOT_0) <= action_int <= int(
                a.Action.FREEZE_SHOP_SLOT_5
            ):
                toggle_shop_slot_frozen(player, action_int - int(a.Action.FREEZE_SHOP_SLOT_0))
                return True

        hsz = hand_size(player)
        if int(a.Action.PLAY_HAND_0) <= action_int < int(a.Action.PLAY_HAND_0) + hsz:
            recruitment_place.place_from_hand(
                player,
                action_int - int(a.Action.PLAY_HAND_0),
                race,
                board_size=a.BOARD_SIZE,
                triggers=ctx.triggers,
                rng=ctx.rng,
                shared_pool=ctx.shared_pool,
            )
            return (
                player.pending_choice is None
                and player.placed_minion_pending_after is None
            )

        if a.is_magnet_game_action(action_int):
            h, b = a.magnet_hand_board_from_game_action(action_int)
            recruitment_place.magnet_from_hand(
                player,
                h,
                b,
                patch=ctx.patch,
                triggers=ctx.triggers,
                shared_pool=ctx.shared_pool,
            )
            return True

        raise ValueError(f"Unknown action {action_int}")

    def end_turn(self, player: PlayerState, *, freeze_shop: bool = False) -> None:
        player.shop_freeze_next_round = freeze_shop
        player.phase = PlayerPhase.DONE


__all__ = ["PlayerTurnEngine"]
