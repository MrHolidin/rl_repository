"""Shop-phase trigger dispatch (ON_BUY, ON_PLACE, turn hooks, battlecries)."""

from __future__ import annotations

from typing import Any, Callable, List, Optional

import numpy as np

from src.bg_catalog.cards import make_minion
from src.bg_catalog.patch_context import PatchContext
from src.bg_core.conditions import ability_condition_met
from src.bg_core.effects import (
    AdaptAllMurlocsEffect,
    AdaptSelfRandomEffect,
    AddRandomMinionToShopEffect,
    AddFromLastOpponentBoardEffect,
    AddRandomMinionToHandEffect,
    CreateSpellcraftSpellEffect,
    GainBloodGemsEffect,
    GrantTemporaryBuffEffect,
    IncreaseBloodGemBonusEffect,
    PlayBloodGemsEffect,
    AddTokenToHandEffect,
    BuffSelfPerCount,
    TransformIntoShopMinionEffect,
    MultiplierKind,
    BuffAdjacentBattlecry,
    BuffMatching,
    BuffTarget,
    BuffAllShopOffersEffect,
    BuffOnePerListedTribeFriendly,
    BuffRandomFriendly,
    BuffRandomUniqueTribeFriendlies,
    BuffSelf,
    BuffSelfFromHeroDamageTaken,
    BuffTargetFriendlyBattlecry,
    BuffTargetFromPiratesBoughtBattlecry,
    ChooseOneEffect,
    ConsumeFriendlyBattlecry,
    ConsumeTavernMinionEffect,
    BuffSelfWhenFriendlyBattlecryPlaced,
    BuffSelfWhenFriendlyDeathrattlePlaced,
    BuffLeftmostRepeatedEffect,
    BuffRandomFriendlyFromPlacedTierEffect,
    DealHeroDamage,
    DiscoverMurlocEffect,
    Effect,
    GainGoldThisTurnEffect,
    GainGoldNextTurnEffect,
    BuffPlacedMinionEffect,
    BumpSeatCounterEffect,
    AddCardToNextRefreshesEffect,
    AddRandomTavernSpellToHandEffect,
    CastRandomTavernSpellEffect,
    CopyLastTavernSpellEffect,
    DiscoverMinionAtTierEffect,
    DiscoverTavernSpellEffect,
    IncreaseTavernSpellBonusEffect,
    RaiseStandingBonusEffect,
    RepeatPerCountEffect,
    StealTavernMinionEffect,
    GiveLockboxEffect,
    AddTavernSpellToHandEffect,
    ReduceTavernSpellCostEffect,
    GrantKeywordRandomFriendly,
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
from src.bg_recruitment.shop_auras import refresh_attack_thresholds
from src.bg_recruitment.choose_one import open_choose_one
from src.bg_recruitment.lockbox import give_lockbox, tick_lockboxes
from src.bg_recruitment.game_counts import (
    bump_seat_counter,
    bump_summoned,
    improve_level,
)
from src.bg_recruitment.standing_bonuses import (
    BonusScope,
    ScopeKind,
    raise_standing_bonus,
    settle_standing_bonuses,
)
from src.bg_recruitment.tavern_spells import steal_tavern_minion
from src.bg_recruitment.spellcraft import (
    discard_spellcraft_spells,
    expire_temporary_buffs,
    give_spellcraft_spell,
)
from src.bg_recruitment.activate import reset_activations
from src.bg_recruitment.blood_gems import (
    blood_gem_targets,
    give_blood_gems,
    play_blood_gem_on,
)
from src.bg_core.board_helpers import (
    count_for_source,
    apply_buff_matching,
    apply_summoned_listener,
    grant_keyword,
    grant_keyword_random,
    apply_buff_self_per_count,
    multiplier_for,
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
    buff_all_shop_offers,
    buff_shop_minions_of_tribe,
)
from src.bg_lobby.player import PendingChoice, PendingChoiceKind, PlayerState


class UnhandledShopEffect(RuntimeError):
    """An effect reached the shop dispatcher and nothing knew what to do with it.

    This used to be a silent fall-through, which made "does nothing in the
    tavern, by design" and "nobody ever implemented this" look identical from
    the outside -- including to the card that quietly stopped working.
    """


#: Effects that legitimately reach ``apply_shop_effect`` and must do nothing
#: there, because something upstream already applied them. Each entry is a
#: claim that can be checked, which is the point of writing them down.
#: Effects the Tavern-spell module owns end to end — including the tier
#: Discover, which is printed on a spell and on a minion alike.
_TAVERN_SPELL_EFFECTS = (
    DiscoverMinionAtTierEffect,
    IncreaseTavernSpellBonusEffect,
    AddRandomTavernSpellToHandEffect,
    DiscoverTavernSpellEffect,
    CastRandomTavernSpellEffect,
    CopyLastTavernSpellEffect,
)

_HANDLED_ELSEWHERE = (
    # Battlecries needing a target the player picked: applied by
    # bg_recruitment/targeted_battlecry.py off the placement action.
    BuffAdjacentBattlecry,
    BuffTargetFriendlyBattlecry,
    BuffTargetFromPiratesBoughtBattlecry,
    ChooseOneEffect,
    ConsumeFriendlyBattlecry,
    ConsumeTavernMinionEffect,
    # fire_on_place applies these itself and then skips them here: they read
    # and write per-turn counters that Brann must not multiply.
    PogoHopperBattlecry,
    AdaptSelfRandomEffect,
    # fire_after_friendly_minion_placed handles these before it delegates.
    BuffPlacedMinionEffect,
    BuffSelfPerCount,
    BuffSelfWhenFriendlyBattlecryPlaced,
    BuffSelfWhenFriendlyDeathrattlePlaced,
    BuffRandomFriendlyFromPlacedTierEffect,
    # Open a discover rather than resolve: fire_on_place sets pending_choice.
    AdaptAllMurlocsEffect,
    DiscoverMurlocEffect,
)


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

    def apply_buff_matching(
        self,
        player: PlayerState,
        source: Minion,
        effect: BuffMatching,
    ) -> None:
        apply_buff_matching(effect, player.board, source)

    def apply_grant_keyword_random(
        self,
        player: PlayerState,
        source: Minion,
        effect: GrantKeywordRandomFriendly,
    ) -> None:
        grant_keyword_random(
            effect,
            player.board,
            source,
            rng=self._rng,
            grant=grant_keyword,
        )

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
        # Khadgar multiplies every summon, in the tavern as well as in combat
        # (combat does the same via auras._summon_multiplier).
        n_sum = self.summon_multiplier(player.board)
        for _ in range(max(0, effect.count) * n_sum):
            if len(player.board) >= BOARD_SIZE:
                break
            tok = make_minion(effect.token_id, patch=self._patch)
            player.board.insert(insert_at, tok)
            self.fire_shop_friendly_summoned(player, tok)
            insert_at += 1
        if player.pending_choice is None:
            self._resolve_triples(player)

    def fire_shop_friendly_summoned(self, player: PlayerState, summoned: Minion) -> None:
        # Every arrival is a summon as the cards use the word — played from
        # hand, summoned by a deathrattle, opened out of a Lockbox — and this
        # is the one place all of them come through.
        bump_summoned(player, summoned)
        for m in player.board:
            if m is summoned:
                continue
            for ab in m.abilities:
                if ab.trigger != Trigger.ON_FRIENDLY_MINION_SUMMONED:
                    continue
                # The one flag no other shop site honours, because BGS_071 is
                # the only ability carrying it and this is its trigger.
                if ab.combat_only:
                    continue
                apply_summoned_listener(
                    ab.effect, m, summoned, grant_keyword=grant_keyword
                )

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
        if isinstance(effect, _HANDLED_ELSEWHERE):
            return
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
        elif isinstance(effect, BuffMatching):
            self.apply_buff_matching(player, source, effect)
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
        elif isinstance(effect, StealTavernMinionEffect):
            steal_tavern_minion(
                player,
                rng=self._rng,
                shared_pool=shared_pool,
                highest_attack=effect.highest_attack,
            )
        elif isinstance(effect, BumpSeatCounterEffect):
            bump_seat_counter(player, effect.counter)
        elif isinstance(effect, RepeatPerCountEffect):
            if effect.counter:
                # An "improves" card: worth its printed value once, plus once
                # more per `per` events counted.
                repeats = improve_level(player, effect.counter, effect.per)
            else:
                repeats = int(effect.base_repeats) + count_for_source(
                    effect.source, player.board, tribe=effect.tribe
                )
            inner = effect.effect
            for _ in range(max(0, repeats)):
                if isinstance(inner, BuffAdjacentBattlecry):
                    # Adjacency is positional, and the dispatcher hands
                    # BuffAdjacentBattlecry to the placement path (it is in
                    # _HANDLED_ELSEWHERE) because at placement the slot is not
                    # settled yet. Here the source is already standing, so the
                    # neighbours are simply known.
                    self.apply_buff_adjacent(player, source, inner)
                    continue
                self.apply_shop_effect(
                    player,
                    source,
                    inner,
                    placed,
                    shop_excluded_race=shop_excluded_race,
                    shared_pool=shared_pool,
                )
        elif isinstance(effect, _TAVERN_SPELL_EFFECTS):
            # Owned by bg_recruitment.tavern_spells: they read and write the
            # spell counter, the spell bonus and the seat's spell memory, none
            # of which this dispatcher knows about.
            from src.bg_recruitment.tavern_spells import apply_tavern_spell_effect

            apply_tavern_spell_effect(
                player,
                effect,
                rng=self._rng,
                patch=self._patch,
                source=source,
                shop_excluded_race=shop_excluded_race,
                shared_pool=shared_pool,
            )
        elif isinstance(effect, RaiseStandingBonusEffect):
            key = effect.scope_key
            if effect.scope_kind is ScopeKind.CARD and key is None:
                # "for each other <me>" — the card scopes the bonus to itself.
                key = source.card_id if source is not None else None
            if not (effect.scope_kind is ScopeKind.CARD and key is None):
                raise_standing_bonus(
                    player,
                    BonusScope(effect.scope_kind, key, effect.scope_max_tier),
                    effect.attack,
                    effect.health,
                )
        elif isinstance(effect, AddCardToNextRefreshesEffect):
            have = player.refresh_promises.get(effect.card_id, 0)
            player.refresh_promises[effect.card_id] = have + int(effect.refreshes)
        elif isinstance(effect, GiveLockboxEffect):
            give_lockbox(player, sooner=int(effect.sooner))
        elif isinstance(effect, AddTavernSpellToHandEffect):
            spell = self._patch.tavern_spells.get(effect.card_id)
            for _ in range(max(1, int(effect.count))):
                slot = first_free_hand_slot(player) if spell is not None else None
                if slot is None:
                    break
                player.hand[slot] = spell
        elif isinstance(effect, ReduceTavernSpellCostEffect):
            player.tavern_spell_cost_delta -= int(effect.amount)
        elif isinstance(effect, GainGoldNextTurnEffect):
            player.gold_next_turn += int(effect.amount)
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
        elif isinstance(effect, CreateSpellcraftSpellEffect):
            # Playing the Naga hands you its spell straight away; the start of
            # every later turn hands you another (fire_on_turn_start).
            give_spellcraft_spell(player, effect)
        elif isinstance(effect, AddRandomMinionToHandEffect):
            add_random_minion_to_hand(
                player,
                effect.tribe,
                shop_excluded_race,
                rng=self._rng,
                patch=self._patch,
                tier=effect.tier,
            )
        elif isinstance(effect, GainBloodGemsEffect):
            give_blood_gems(
                player, effect.count, quilboar_keyword=effect.quilboar_keyword
            )
        elif isinstance(effect, PlayBloodGemsEffect):
            for target in blood_gem_targets(player, source, effect.target):
                play_blood_gem_on(player, target, count=effect.count)
        elif isinstance(effect, IncreaseBloodGemBonusEffect):
            player.blood_gem_bonus_attack += effect.attack
            player.blood_gem_bonus_health += effect.health
        else:
            raise UnhandledShopEffect(
                f"{type(effect).__name__} reached the shop dispatcher with no "
                f"handler and is not listed in _HANDLED_ELSEWHERE. Either give "
                f"it a branch or say where it is handled instead."
            )
        settle_standing_bonuses(player)
        refresh_attack_thresholds(player.board)

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
        return multiplier_for(board, MultiplierKind.BATTLECRY)

    @staticmethod
    def summon_multiplier(board: List[Minion]) -> int:
        """Khadgar reads "your cards that summon minions summon twice as many"
        -- no phase qualifier -- so a tavern battlecry summon doubles exactly
        like a deathrattle summon in combat."""
        return multiplier_for(board, MultiplierKind.SUMMON)

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
            if isinstance(e, ChooseOneEffect):
                # Parks the two options and stops here; the pick applies them,
                # multiplied battlecries included (Brann re-opens the choice).
                open_choose_one(player, e, source=placed)
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
                if isinstance(eff, BuffPlacedMinionEffect):
                    # The one listener on this trigger that pays the newcomer
                    # rather than the watcher.
                    if m is placed:
                        continue
                    placed.bonus_attack += eff.attack
                    placed.bonus_health += eff.health
                    continue
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
                if isinstance(eff, BuffSelfPerCount):
                    apply_buff_self_per_count(eff, m, player.board)
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
                # Gate is specific to the FRIENDLY_OF_TRIBE target — before the
                # merge only ``BuffAllFriendlyOfTribe`` reached this branch, so
                # widening it to every BuffMatching variant would change them.
                if (
                    isinstance(eff, BuffMatching)
                    and eff.target is BuffTarget.FRIENDLY_OF_TRIBE
                ):
                    if not self._has_battlecry(placed):
                        continue
                self.apply_shop_effect(player, m, ab.effect, placed)

    def fire_on_turn_end(self, player: PlayerState) -> None:
        # A Spellcraft spell is worth exactly the turn it was made for.
        discard_spellcraft_spells(player)
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
                elif isinstance(ab.effect, BuffSelf):
                    # "At the end of your turn, gain +1 Health" (Lullabot, and
                    # every card printed on that pattern since). fire_on_turn_start
                    # already delegates its BuffSelf the same way.
                    self.apply_shop_effect(player, source, ab.effect, None)
                elif isinstance(ab.effect, BuffSelfPerCount):
                    apply_buff_self_per_count(ab.effect, source, player.board)
                elif isinstance(ab.effect, BuffLeftmostRepeatedEffect):
                    e = ab.effect
                    n = int(getattr(player, e.counter, 0))
                    if n > 0 and player.board:
                        left = player.board[0]
                        for _ in range(n):
                            left.bonus_attack += e.attack
                            left.bonus_health += e.health
                else:
                    # Anything else this trigger can carry goes to the shop
                    # dispatcher, which raises on an effect nobody handles.
                    # The branches above stay because they read the board in
                    # ways the dispatcher's single-source signature cannot.
                    self.apply_shop_effect(player, source, ab.effect, None)

    def fire_on_turn_start(self, player: PlayerState) -> None:
        """After round increment, before shop reroll: board L→R, then hand slots."""
        player.pirates_bought_this_turn = 0
        player.elementals_played = 0
        reset_activations(player)
        # Last turn's "until next turn" buffs end here — after the combat they
        # were cast for, before the seat acts again.
        expire_temporary_buffs(player)
        # A Lockbox counts down in the seat's own turns. It only draws from the
        # generator on the turn it opens, and no shipped package can make one,
        # so the random stream on 36393 / 74257 is untouched.
        tick_lockboxes(player, rng=self._rng, patch=self._patch)
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
                elif isinstance(e, CreateSpellcraftSpellEffect):
                    give_spellcraft_spell(player, e)
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
