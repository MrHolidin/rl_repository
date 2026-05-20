"""Shop-phase trigger dispatch (ON_BUY, ON_PLACE, turn hooks, battlecries)."""

from __future__ import annotations

from typing import Any, Callable, List, Optional

import numpy as np

from src.bg_catalog.cards import make_minion
from src.bg_core.effects import (
    Ability,
    AdaptAllMurlocsEffect,
    BattlecryMultiplierAura,
    BuffAdjacentBattlecry,
    BuffAllFriendlyOfTribe,
    BuffAllOtherOfTribe,
    BuffAllWithKeyword,
    BuffListenerIfSummonedMatches,
    BuffOnePerListedTribeFriendly,
    BuffRandomFriendly,
    BuffSelf,
    BuffSelfFromHeroDamageTaken,
    BuffSelfWhenFriendlyBattlecryPlaced,
    BuffSummonedIfRace,
    BuffTargetFriendlyBattlecry,
    DealHeroDamage,
    DiscoverMurlocEffect,
    Effect,
    GrantKeywordRandomFriendly,
    GrantListenerKeywordIfSummonedMatches,
    HeroImmuneAura,
    Keyword,
    PogoHopperBattlecry,
    SummonEffect,
    Trigger,
)
from src.bg_core.minion import Minion, Race
from src.envs.minibg.actions import BOARD_SIZE
from src.bg_recruitment.discover_pool import roll_adapt_triple, roll_discover_murloc_triple
from src.envs.minibg.state import PendingChoice, PendingChoiceKind, PlayerState


class ShopTriggers:
    def __init__(
        self,
        rng: np.random.Generator,
        *,
        on_triples: Callable[[PlayerState], None],
    ) -> None:
        self._rng = rng
        self._on_triples = on_triples

    @staticmethod
    def minion_matches_tribe(m: Minion, tribe: Any) -> bool:
        if m.race is None:
            return False
        if tribe == Race.ALL or m.race == Race.ALL:
            return True
        return m.race == tribe

    @staticmethod
    def player_has_hero_immune(player: PlayerState) -> bool:
        for m in player.board:
            for ab in m.abilities:
                if ab.trigger == Trigger.AURA and isinstance(ab.effect, HeroImmuneAura):
                    return True
        return False

    def damage_hero(self, player: PlayerState, amount: int) -> None:
        if amount <= 0 or self.player_has_hero_immune(player):
            return
        player.health -= amount
        player.hero_damage_taken_total += amount

    def apply_buff_adjacent(
        self,
        player: PlayerState,
        source: Minion,
        effect: BuffAdjacentBattlecry,
    ) -> None:
        board = player.board
        try:
            idx = board.index(source)
        except ValueError:
            return
        for j in (idx - 1, idx + 1):
            if 0 <= j < len(board):
                tgt = board[j]
                tgt.bonus_attack += effect.attack
                tgt.bonus_health += effect.health
                if effect.grant_taunt:
                    tgt.keywords = frozenset(tgt.keywords | {Keyword.TAUNT})

    def apply_buff_random(
        self,
        source: Minion,
        effect: BuffRandomFriendly,
        board: List[Minion],
    ) -> None:
        for _ in range(max(1, effect.repeats)):
            pool = (
                [m for m in board if m is not source]
                if effect.exclude_self
                else list(board)
            )
            if effect.filter_race is not None:
                pool = [
                    m for m in pool if self.minion_matches_tribe(m, effect.filter_race)
                ]
            if not pool:
                return
            target = pool[int(self._rng.integers(0, len(pool)))]
            target.bonus_attack += effect.attack
            target.bonus_health += effect.health
            if effect.grant_taunt:
                target.keywords = frozenset(target.keywords | {Keyword.TAUNT})

    def apply_buff_one_per_listed_tribe(
        self,
        source: Minion,
        effect: BuffOnePerListedTribeFriendly,
        board: List[Minion],
    ) -> None:
        for tribe in effect.tribes:
            pool = (
                [m for m in board if m is not source]
                if effect.exclude_self
                else list(board)
            )
            pool = [m for m in pool if self.minion_matches_tribe(m, tribe)]
            if not pool:
                continue
            target = pool[int(self._rng.integers(0, len(pool)))]
            target.bonus_attack += effect.attack
            target.bonus_health += effect.health

    def apply_buff_all_other_tribe(
        self,
        player: PlayerState,
        source: Minion,
        effect: BuffAllOtherOfTribe,
    ) -> None:
        for m in player.board:
            if m is source or not self.minion_matches_tribe(m, effect.tribe):
                continue
            m.bonus_attack += effect.attack
            m.bonus_health += effect.health

    def apply_buff_all_friendly_tribe(
        self, player: PlayerState, effect: BuffAllFriendlyOfTribe
    ) -> None:
        for m in player.board:
            if not self.minion_matches_tribe(m, effect.tribe):
                continue
            m.bonus_attack += effect.attack
            m.bonus_health += effect.health

    def apply_buff_all_keyword(
        self, player: PlayerState, effect: BuffAllWithKeyword
    ) -> None:
        for m in player.board:
            if effect.keyword not in m.all_keywords:
                continue
            m.bonus_attack += effect.attack
            m.bonus_health += effect.health

    def apply_grant_keyword_random(
        self,
        player: PlayerState,
        source: Minion,
        effect: GrantKeywordRandomFriendly,
    ) -> None:
        pool = list(player.board)
        if effect.exclude_self:
            pool = [m for m in pool if m is not source]
        if effect.filter_race is not None:
            pool = [m for m in pool if self.minion_matches_tribe(m, effect.filter_race)]
        for _ in range(max(1, effect.repeats)):
            if not pool:
                return
            tgt = pool[int(self._rng.integers(0, len(pool)))]
            tgt.keywords = frozenset(tgt.keywords | {effect.keyword})
            if effect.keyword == Keyword.SHIELD:
                tgt.has_shield = True

    def apply_summon_from_place(
        self, player: PlayerState, source: Minion, effect: SummonEffect
    ) -> None:
        if effect.for_opponent or effect.count_from_source_attack:
            return
        try:
            idx = player.board.index(source)
        except ValueError:
            return
        insert_at = idx + 1
        for _ in range(max(0, effect.count)):
            if len(player.board) >= BOARD_SIZE:
                break
            tok = make_minion(effect.token_id)
            player.board.insert(insert_at, tok)
            self.fire_shop_friendly_summoned(player, tok)
            insert_at += 1
        if player.pending_choice is None:
            self._on_triples(player)

    def fire_shop_friendly_summoned(self, player: PlayerState, summoned: Minion) -> None:
        for m in player.board:
            if m is summoned:
                continue
            for ab in m.abilities:
                if ab.trigger != Trigger.ON_FRIENDLY_MINION_SUMMONED:
                    continue
                eff = ab.effect
                if isinstance(eff, BuffSummonedIfRace):
                    if self.minion_matches_tribe(summoned, eff.tribe):
                        summoned.bonus_attack += eff.attack
                        summoned.bonus_health += eff.health
                elif isinstance(eff, GrantListenerKeywordIfSummonedMatches):
                    if self.minion_matches_tribe(summoned, eff.tribe):
                        m.keywords = frozenset(m.keywords | {eff.keyword})
                        if eff.keyword == Keyword.SHIELD:
                            m.has_shield = True
                elif isinstance(eff, BuffListenerIfSummonedMatches):
                    if self.minion_matches_tribe(summoned, eff.tribe):
                        m.bonus_attack += eff.attack
                        m.bonus_health += eff.health

    def apply_shop_effect(
        self,
        player: PlayerState,
        source: Minion,
        effect: Effect,
        placed: Optional[Minion],
    ) -> None:
        if isinstance(effect, BuffRandomFriendly):
            self.apply_buff_random(source, effect, player.board)
        elif isinstance(effect, BuffOnePerListedTribeFriendly):
            self.apply_buff_one_per_listed_tribe(source, effect, player.board)
        elif isinstance(effect, DealHeroDamage):
            self.damage_hero(player, effect.amount)
        elif isinstance(effect, BuffSelf):
            source.bonus_attack += effect.attack
            source.bonus_health += effect.health
        elif isinstance(effect, BuffSelfFromHeroDamageTaken):
            source.bonus_health += player.hero_damage_taken_total
        elif isinstance(effect, BuffAllOtherOfTribe):
            self.apply_buff_all_other_tribe(player, source, effect)
        elif isinstance(effect, BuffAllFriendlyOfTribe):
            self.apply_buff_all_friendly_tribe(player, effect)
        elif isinstance(effect, BuffAllWithKeyword):
            self.apply_buff_all_keyword(player, effect)
        elif isinstance(effect, GrantKeywordRandomFriendly):
            self.apply_grant_keyword_random(player, source, effect)
        elif isinstance(effect, SummonEffect):
            self.apply_summon_from_place(player, source, effect)

    @staticmethod
    def battlecry_multiplier(board: List[Minion]) -> int:
        p = 1
        for m in board:
            for ab in m.abilities:
                if ab.trigger == Trigger.AURA and isinstance(
                    ab.effect, BattlecryMultiplierAura
                ):
                    p *= ab.effect.factor
        return p

    def fire_on_buy(self, minion: Minion, player: PlayerState) -> None:
        for ab in minion.abilities:
            if ab.trigger == Trigger.ON_BUY and isinstance(ab.effect, BuffRandomFriendly):
                self.apply_buff_random(minion, ab.effect, player.board)

    def fire_on_place(
        self,
        placed: Minion,
        player: PlayerState,
        shop_excluded_race: Optional[Race],
    ) -> None:
        mult = self.battlecry_multiplier(player.board)
        for ab in placed.abilities:
            if ab.trigger != Trigger.ON_PLACE:
                continue
            e = ab.effect
            if isinstance(e, DiscoverMurlocEffect):
                free_slots = sum(1 for s in player.hand if s is None)
                total = min(mult * e.repeats, free_slots)
                if total <= 0:
                    return
                opts = roll_discover_murloc_triple(
                    self._rng, player.tavern_tier, shop_excluded_race
                )
                from src.bg_recruitment.discover import try_open_hand_discover_modal

                if try_open_hand_discover_modal(
                    player,
                    PendingChoiceKind.DISCOVER_MURLOC,
                    opts,
                    total - 1,
                ):
                    return
                return
            if isinstance(e, AdaptAllMurlocsEffect):
                total = mult * e.repeats
                opts = roll_adapt_triple(self._rng)
                player.pending_choice = PendingChoice(
                    PendingChoiceKind.ADAPT, opts, total - 1
                )
                return
        for ab in placed.abilities:
            if ab.trigger != Trigger.ON_PLACE:
                continue
            if isinstance(ab.effect, PogoHopperBattlecry):
                n = player.pogo_hoppers_played
                e = ab.effect
                placed.bonus_attack += e.attack_each * n * mult
                placed.bonus_health += e.health_each * n * mult
                player.pogo_hoppers_played += 1
                break
        i = 0
        while i < mult:
            i += 1
            for ab in placed.abilities:
                if ab.trigger != Trigger.ON_PLACE:
                    continue
                if isinstance(ab.effect, PogoHopperBattlecry):
                    continue
                self.apply_shop_effect(player, placed, ab.effect, placed)

    def fire_after_friendly_minion_placed(
        self, player: PlayerState, placed: Minion
    ) -> None:
        for m in list(player.board):
            for ab in m.abilities:
                if ab.trigger != Trigger.AFTER_FRIENDLY_MINION_PLACED:
                    continue
                if ab.filter_race is not None and placed.race != ab.filter_race:
                    continue
                eff = ab.effect
                if isinstance(eff, BuffSelfWhenFriendlyBattlecryPlaced):
                    if m is placed:
                        continue
                    if not any(x.trigger == Trigger.ON_PLACE for x in placed.abilities):
                        continue
                    m.bonus_attack += eff.attack
                    m.bonus_health += eff.health
                    continue
                self.apply_shop_effect(player, m, ab.effect, placed)

    def fire_on_turn_end(self, player: PlayerState) -> None:
        for source in list(player.board):
            for ab in source.abilities:
                if ab.trigger != Trigger.ON_TURN_END:
                    continue
                if isinstance(ab.effect, BuffRandomFriendly):
                    self.apply_buff_random(source, ab.effect, player.board)
                elif isinstance(ab.effect, BuffOnePerListedTribeFriendly):
                    self.apply_buff_one_per_listed_tribe(
                        source, ab.effect, player.board
                    )

    def fire_on_turn_start(self, player: PlayerState) -> None:
        """After round increment, before shop reroll: board L→R, then hand slots."""
        for source in list(player.board):
            for ab in source.abilities:
                if ab.trigger != Trigger.ON_TURN_START:
                    continue
                e = ab.effect
                if isinstance(e, BuffRandomFriendly):
                    self.apply_buff_random(source, e, player.board)
                elif isinstance(e, BuffOnePerListedTribeFriendly):
                    self.apply_buff_one_per_listed_tribe(source, e, player.board)
                elif isinstance(e, BuffSelf):
                    self.apply_shop_effect(player, source, e, None)
        for source in list(player.hand):
            if source is None:
                continue
            for ab in source.abilities:
                if ab.trigger != Trigger.ON_TURN_START:
                    continue
                e = ab.effect
                if isinstance(e, BuffRandomFriendly):
                    self.apply_buff_random(source, e, player.board)
                elif isinstance(e, BuffOnePerListedTribeFriendly):
                    self.apply_buff_one_per_listed_tribe(source, e, player.board)
                elif isinstance(e, BuffSelf):
                    self.apply_shop_effect(player, source, e, None)
