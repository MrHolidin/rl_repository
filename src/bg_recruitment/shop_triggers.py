"""Shop-phase trigger dispatch (ON_BUY, ON_PLACE, turn hooks, battlecries)."""

from __future__ import annotations

from typing import Any, Callable, List, Optional

import numpy as np

from src.bg_catalog.cards import make_minion
from src.bg_catalog.patch_context import PatchContext
from src.bg_core.conditions import ability_condition_met
from src.bg_core.effects import (
    Ability,
    AdaptAllMurlocsEffect,
    AdaptSelfRandomEffect,
    AddRandomMinionToShopEffect,
    AddFromLastOpponentBoardEffect,
    AddRandomMinionToHandEffect,
    AddTokenToHandEffect,
    BuffSelfFromFriendlyTribeCount,
    BuffSelfFromGoldenFriendlyCount,
    BuffSelfFromUniqueTribeCount,
    TransformIntoShopMinionEffect,
    BattlecryMultiplierAura,
    BuffAdjacentBattlecry,
    BuffAllFriendlyOfTribe,
    BuffAllOtherOfTribe,
    BuffAllShopOffersEffect,
    BuffAllWithKeyword,
    BuffListenerIfSummonedMatches,
    BuffOnePerListedTribeFriendly,
    BuffRandomFriendly,
    BuffRandomUniqueTribeFriendlies,
    BuffSelf,
    BuffSelfFromHeroDamageTaken,
    BuffSelfWhenFriendlyBattlecryPlaced,
    BuffSelfWhenFriendlyDeathrattlePlaced,
    BuffLeftmostRepeatedEffect,
    BuffRandomFriendlyFromPlacedTierEffect,
    BuffSummonedIfRace,
    BuffTargetFriendlyBattlecry,
    DealHeroDamage,
    DiscoverMurlocEffect,
    Effect,
    GainGoldThisTurnEffect,
    GrantKeywordRandomFriendly,
    GrantListenerKeywordIfSummonedMatches,
    HeroImmuneAura,
    IncrementShopTribeBonusEffect,
    Keyword,
    PogoHopperBattlecry,
    ReduceUpgradeCostEffect,
    SetNextRollCostEffect,
    SummonEffect,
    Trigger,
)
from src.bg_recruitment.hand_slots import first_free_hand_slot
from src.bg_core.board_helpers import (
    count_friendly_tribe,
    count_golden_friendlies,
    count_unique_tribes,
)
from src.bg_core.minion import Minion, Race
from src.envs.minibg.actions import BOARD_SIZE
from src.bg_recruitment.discover_pool import (
    ADAPT_KEYS_ALL,
    apply_adapt_key_to_minion,
    roll_adapt_triple,
    roll_discover_murloc_triple,
)
from src.bg_recruitment.shop import (
    add_random_minion_to_hand,
    add_random_minion_to_shop,
    apply_shop_tribe_bonus_to_minion,
    buff_all_shop_offers,
    buff_shop_minions_of_tribe,
)
from src.bg_lobby.player import PendingChoice, PendingChoiceKind, PlayerState


class ShopTriggers:
    def __init__(
        self,
        rng: np.random.Generator,
        *,
        on_triples: Optional[Callable[[PlayerState], None]] = None,
        patch: PatchContext,
    ) -> None:
        from src.bg_catalog.patch_context import require_patch

        self._rng = rng
        self._on_triples = on_triples
        self._patch = require_patch(patch, where="ShopTriggers.__init__")

    def _resolve_triples(
        self, player: PlayerState, *, shared_pool: Any = None
    ) -> None:
        if self._on_triples is not None:
            self._on_triples(player)
            return
        from src.bg_recruitment import triples as recruitment_triples

        recruitment_triples.resolve_triples_loop(
            player, shared_pool=shared_pool, patch=self._patch
        )

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

    def apply_buff_random_unique_tribe(
        self,
        source: Minion,
        effect: BuffRandomUniqueTribeFriendlies,
        board: List[Minion],
    ) -> None:
        pool = (
            [m for m in board if m is not source]
            if effect.exclude_self
            else list(board)
        )
        by_tribe: dict[Race, List[Minion]] = {}
        for m in pool:
            if m.race is None or m.race == Race.ALL:
                continue
            by_tribe.setdefault(m.race, []).append(m)
        tribes = list(by_tribe.keys())
        if not tribes:
            return
        order = tribes.copy()
        for i in range(len(order) - 1, 0, -1):
            j = int(self._rng.integers(0, i + 1))
            order[i], order[j] = order[j], order[i]
        for tribe in order[: max(0, effect.count)]:
            candidates = by_tribe[tribe]
            target = candidates[int(self._rng.integers(0, len(candidates)))]
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
            tok = make_minion(effect.token_id, patch=self._patch)
            player.board.insert(insert_at, tok)
            self.fire_shop_friendly_summoned(player, tok)
            insert_at += 1
        if player.pending_choice is None:
            self._resolve_triples(player)

    def fire_shop_friendly_summoned(self, player: PlayerState, summoned: Minion) -> None:
        for m in player.board:
            if m is summoned:
                continue
            for ab in m.abilities:
                if ab.trigger != Trigger.ON_FRIENDLY_MINION_SUMMONED:
                    continue
                if ab.combat_only:
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
        *,
        shop_excluded_race: Optional[Race] = None,
        shared_pool=None,
    ) -> None:
        if isinstance(effect, BuffRandomFriendly):
            self.apply_buff_random(source, effect, player.board)
        elif isinstance(effect, BuffOnePerListedTribeFriendly):
            self.apply_buff_one_per_listed_tribe(source, effect, player.board)
        elif isinstance(effect, BuffRandomUniqueTribeFriendlies):
            self.apply_buff_random_unique_tribe(source, effect, player.board)
        elif isinstance(effect, BuffAllShopOffersEffect):
            buff_all_shop_offers(player, attack=effect.attack, health=effect.health)
        elif isinstance(effect, AddRandomMinionToShopEffect):
            add_random_minion_to_shop(
                player,
                effect.tribe,
                shop_excluded_race,
                rng=self._rng,
                shared_pool=shared_pool,
                patch=self._patch,
                freeze_slot=effect.freeze_slot,
            )
        elif isinstance(effect, DealHeroDamage):
            self.damage_hero(player, effect.amount)
        elif isinstance(effect, BuffSelf):
            source.bonus_attack += effect.attack
            source.bonus_health += effect.health
        elif isinstance(effect, BuffSelfFromHeroDamageTaken):
            source.bonus_health += (
                player.hero_damage_taken_total * effect.health_per_damage
            )
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
        elif isinstance(effect, ReduceUpgradeCostEffect):
            player.upgrade_cost_delta -= effect.amount
        elif isinstance(effect, SetNextRollCostEffect):
            player.next_roll_cost_override = effect.cost
            player.free_roll_charges = effect.uses
        elif isinstance(effect, GainGoldThisTurnEffect):
            if effect.filter_race is None or (
                placed is not None and self.minion_matches_tribe(placed, effect.filter_race)
            ):
                player.gold += effect.amount
        elif isinstance(effect, AddTokenToHandEffect):
            for _ in range(max(0, effect.count)):
                slot = first_free_hand_slot(player)
                if slot is None:
                    break
                player.hand[slot] = make_minion(effect.token_id, patch=self._patch)
        elif isinstance(effect, IncrementShopTribeBonusEffect):
            if effect.tribe == Race.ELEMENTAL:
                player.shop_elemental_bonus += effect.attack
            buff_shop_minions_of_tribe(
                player, effect.tribe, attack=effect.attack, health=effect.health
            )
        elif isinstance(effect, AddFromLastOpponentBoardEffect):
            if not player.last_opponent_board:
                return
            pick = player.last_opponent_board[
                int(self._rng.integers(0, len(player.last_opponent_board)))
            ]
            slot = first_free_hand_slot(player)
            if slot is None:
                return
            if effect.make_golden:
                from src.bg_recruitment.triples import make_forged_golden_minion

                player.hand[slot] = make_forged_golden_minion(
                    pick.card_id, patch=self._patch
                )
            else:
                player.hand[slot] = make_minion(pick.card_id, patch=self._patch)
        elif isinstance(effect, TransformIntoShopMinionEffect):
            try:
                idx = player.board.index(source)
            except ValueError:
                return
            from src.bg_recruitment.faceless import apply_transform_into_shop_minion

            slots = [i for i, m in enumerate(player.shop) if m is not None]
            if not slots:
                return
            pick = slots[int(self._rng.integers(0, len(slots)))]
            apply_transform_into_shop_minion(
                player, idx, pick, patch=self._patch, copy_golden=effect.copy_golden
            )
        elif isinstance(effect, AddRandomMinionToHandEffect):
            add_random_minion_to_hand(
                player,
                effect.tribe,
                shop_excluded_race,
                rng=self._rng,
                patch=self._patch,
            )

    @staticmethod
    def _has_battlecry(minion: Minion) -> bool:
        return any(ab.trigger == Trigger.ON_PLACE for ab in minion.abilities)

    @staticmethod
    def _has_deathrattle(minion: Minion) -> bool:
        return any(ab.trigger == Trigger.ON_DEATH for ab in minion.abilities)

    def fire_on_sell(self, sold: Minion, player: PlayerState) -> None:
        for ab in sold.abilities:
            if ab.trigger != Trigger.ON_SELL:
                continue
            self.apply_shop_effect(player, sold, ab.effect, sold)

    def fire_on_friendly_bought(self, bought: Minion, player: PlayerState) -> None:
        if bought.race == Race.PIRATE:
            player.pirates_bought_this_turn += 1
        for m in list(player.board):
            if m is bought:
                continue
            for ab in m.abilities:
                if ab.trigger != Trigger.ON_FRIENDLY_BOUGHT:
                    continue
                if ab.filter_race is not None and not self.minion_matches_tribe(
                    bought, ab.filter_race
                ):
                    continue
                self.apply_shop_effect(player, m, ab.effect, bought)

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
        *,
        shared_pool=None,
    ) -> None:
        mult = self.battlecry_multiplier(player.board)
        for ab in placed.abilities:
            if ab.trigger != Trigger.ON_PLACE:
                continue
            if not ability_condition_met(ab, player, player.board, placed=placed):
                continue
            e = ab.effect
            if isinstance(e, DiscoverMurlocEffect):
                free_slots = sum(1 for s in player.hand if s is None)
                total = min(mult * e.repeats, free_slots)
                if total <= 0:
                    return
                opts = roll_discover_murloc_triple(
                    self._rng,
                    player.tavern_tier,
                    shop_excluded_race,
                    shared_pool=shared_pool,
                    patch=self._patch,
                )
                if opts is None:
                    return
                from src.bg_recruitment.discover import try_open_hand_discover_modal

                if try_open_hand_discover_modal(
                    player,
                    PendingChoiceKind.DISCOVER_MURLOC,
                    opts,
                    total - 1,
                    shared_pool=shared_pool,
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
            if isinstance(e, AdaptSelfRandomEffect):
                if e.count_from_unique_other_tribes:
                    n = count_unique_tribes(player.board, exclude=placed) * mult * e.repeats
                else:
                    n = mult * e.repeats
                for _ in range(n):
                    key = ADAPT_KEYS_ALL[int(self._rng.integers(0, len(ADAPT_KEYS_ALL)))]
                    apply_adapt_key_to_minion(placed, key)
                continue
            if isinstance(e, TransformIntoShopMinionEffect):
                from src.bg_recruitment.faceless import try_open_transform_shop_modal

                idx = player.board.index(placed)
                try_open_transform_shop_modal(
                    player,
                    idx,
                    patch=self._patch,
                    rng=self._rng,
                    copy_golden=e.copy_golden,
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
                if not ability_condition_met(ab, player, player.board, placed=placed):
                    continue
                if isinstance(ab.effect, PogoHopperBattlecry):
                    continue
                if isinstance(ab.effect, AdaptSelfRandomEffect):
                    continue
                if isinstance(ab.effect, TransformIntoShopMinionEffect):
                    continue
                self.apply_shop_effect(
                    player, placed, ab.effect, placed,
                    shop_excluded_race=shop_excluded_race,
                    shared_pool=shared_pool,
                )
        if placed.race == Race.ELEMENTAL:
            player.elementals_played += 1
            from src.bg_recruitment import hero_passives

            hero_passives.apply_hero_on_elemental_played(player)  # Chenvaala

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
                    if not self._has_battlecry(placed):
                        continue
                    m.bonus_attack += eff.attack
                    m.bonus_health += eff.health
                    continue
                if isinstance(eff, BuffSelfWhenFriendlyDeathrattlePlaced):
                    if m is placed:
                        continue
                    if not self._has_deathrattle(placed):
                        continue
                    m.bonus_attack += eff.attack
                    m.bonus_health += eff.health
                    continue
                if isinstance(eff, BuffSelfFromFriendlyTribeCount):
                    e = eff
                    n = count_friendly_tribe(
                        player.board,
                        e.tribe,
                        exclude=m if e.exclude_self else None,
                    )
                    m.bonus_attack += e.attack_per * n
                    m.bonus_health += e.health_per * n
                    continue
                if isinstance(eff, BuffRandomFriendlyFromPlacedTierEffect):
                    e = eff
                    tier = max(0, placed.tier)
                    atk = tier if e.attack_per_tier else 0
                    hp = tier if e.health_per_tier else 0
                    if atk == 0 and hp == 0:
                        continue
                    eligible = [
                        x
                        for x in player.board
                        if not (e.exclude_self and x is m)
                    ]
                    if not eligible:
                        continue
                    pick = eligible[int(self._rng.integers(0, len(eligible)))]
                    pick.bonus_attack += atk
                    pick.bonus_health += hp
                    continue
                if isinstance(eff, BuffAllFriendlyOfTribe):
                    if not self._has_battlecry(placed):
                        continue
                if isinstance(eff, IncrementShopTribeBonusEffect):
                    pass
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
                elif isinstance(ab.effect, BuffSelfFromFriendlyTribeCount):
                    e = ab.effect
                    n = count_friendly_tribe(
                        player.board,
                        e.tribe,
                        exclude=source if e.exclude_self else None,
                    )
                    source.bonus_attack += e.attack_per * n
                    source.bonus_health += e.health_per * n
                elif isinstance(ab.effect, BuffSelfFromUniqueTribeCount):
                    e = ab.effect
                    n = count_unique_tribes(
                        player.board, exclude=source if e.exclude_self else None
                    )
                    source.bonus_attack += e.attack_per * n
                    source.bonus_health += e.health_per * n
                elif isinstance(ab.effect, BuffSelfFromGoldenFriendlyCount):
                    e = ab.effect
                    n = count_golden_friendlies(
                        player.board, exclude=source if e.exclude_self else None
                    )
                    source.bonus_attack += e.attack_per * n
                    source.bonus_health += e.health_per * n
                elif isinstance(ab.effect, BuffLeftmostRepeatedEffect):
                    e = ab.effect
                    n = int(getattr(player, e.counter, 0))
                    if n > 0 and player.board:
                        left = player.board[0]
                        for _ in range(n):
                            left.bonus_attack += e.attack
                            left.bonus_health += e.health

    def fire_on_turn_start(self, player: PlayerState) -> None:
        """After round increment, before shop reroll: board L→R, then hand slots."""
        player.pirates_bought_this_turn = 0
        player.elementals_played = 0
        for source in list(player.board):
            for ab in source.abilities:
                if ab.trigger != Trigger.ON_TURN_START:
                    continue
                if not ability_condition_met(ab, player, player.board, placed=source):
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
