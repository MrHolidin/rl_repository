from __future__ import annotations

from copy import copy
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.games.turn_based_game import Action as ActionType
from src.games.turn_based_game import TurnBasedGame

from .actions import (
    BOARD_SIZE,
    BUY_COST,
    HAND_SIZE,
    LEVEL_UP_COSTS,
    LEVEL_UP_DISCOUNT_PER_ROUND,
    MAX_ROUNDS,
    MAX_SHOP_ACTIONS,
    MAX_SHOP_SLOTS,
    MAX_TIER,
    ROLL_COST,
    SELL_REWARD,
    STARTING_HEALTH,
    STARTING_TIER,
    Action,
    discover_pick_index,
    gold_for_round,
    is_discover_pick_game_action,
    is_magnet_game_action,
    magnet_hand_board_from_game_action,
    shop_offers_count,
)
from .battle import simulate_battle
from .card_pool import triple_merge_golden_abilities
from .cards import CARD_TEMPLATES, make_minion, shop_pool_for_tier
from .discover_pool import (
    apply_adapt_key_to_minion,
    is_murloc_board_minion,
    roll_adapt_triple,
    roll_discover_murloc_triple,
    roll_triple_reward_discover_triple,
)
from .effects import (
    Ability,
    AdjacentStatAura,
    BuffAdjacentBattlecry,
    BuffAllFriendlyOfTribe,
    BuffAllOtherOfTribe,
    BuffAllWithKeyword,
    BuffOnePerListedTribeFriendly,
    BuffRandomFriendly,
    BuffSelf,
    BuffSelfFromHeroDamageTaken,
    BuffSelfWhenFriendlyBattlecryPlaced,
    BuffSummonedIfRace,
    BuffListenerIfSummonedMatches,
    BattlecryMultiplierAura,
    DealHeroDamage,
    AdaptAllMurlocsEffect,
    DiscoverMurlocEffect,
    Effect,
    GrantKeywordRandomFriendly,
    GrantListenerKeywordIfSummonedMatches,
    HeroImmuneAura,
    KeywordStatAura,
    Keyword,
    StatAura,
    SummonEffect,
    SummonFirstDeadFriendlyMechsThisCombat,
    SummonRandomMinionEffect,
    SummonMultiplierAura,
    DeathrattleMultiplierAura,
    CleaveOnAttack,
    TribalOtherStatAura,
    Trigger,
    ZappTargeting,
    PogoHopperBattlecry,
)
from .state import (
    MiniBGState,
    Minion,
    PendingChoice,
    PendingChoiceKind,
    PlayerPhase,
    PlayerState,
    Race,
    ROTATION_SHOP_TRIBES,
)


PLAYER_TOKENS = (1, -1)


class MiniBGGame(TurnBasedGame[MiniBGState]):
    def __init__(
        self,
        seed: Optional[int] = None,
        *,
        shop_excluded_race: Optional[Race] = None,
        shop_full_tribes: bool = False,
    ) -> None:
        self._rng = np.random.default_rng(seed)
        self._shop_full_tribes = shop_full_tribes
        self._shop_excluded_race_fixed = shop_excluded_race

    def _pick_shop_excluded_race(self) -> Optional[Race]:
        if self._shop_excluded_race_fixed is not None:
            return self._shop_excluded_race_fixed
        if self._shop_full_tribes:
            return None
        i = int(self._rng.integers(0, len(ROTATION_SHOP_TRIBES)))
        return ROTATION_SHOP_TRIBES[i]

    def initial_state(self) -> MiniBGState:
        initiative_player = int(self._rng.integers(0, 2))
        shop_excluded = self._pick_shop_excluded_race()
        players = (
            self._fresh_player(round_number=1, shop_excluded_race=shop_excluded),
            self._fresh_player(round_number=1, shop_excluded_race=shop_excluded),
        )
        return MiniBGState(
            players=players,
            round_number=1,
            current_player_index=0,
            initiative_player=initiative_player,
            winner=None,
            done=False,
            shop_excluded_race=shop_excluded,
        )

    def current_player(self, state: MiniBGState) -> int:
        return PLAYER_TOKENS[state.current_player_index]

    def is_terminal(self, state: MiniBGState) -> bool:
        return state.done

    def winner(self, state: MiniBGState) -> Optional[int]:
        return state.winner

    def legal_actions(self, state: MiniBGState) -> Sequence[ActionType]:
        if state.done:
            return []
        player = state.players[state.current_player_index]
        if player.phase == PlayerPhase.DONE:
            return []

        if player.phase == PlayerPhase.ORDER:
            # Order phase: a permutation must be submitted. The reorder is
            # produced by the env layer (`reorder_board(...)`) and immediately
            # followed by ``apply_action(FINISH)``; the only "game-level"
            # action emitted from the order phase is ``FINISH``.
            return [int(Action.FINISH)]

        # Shop phase — pending Discover / Adapt blocks other actions.
        if player.pending_choice is not None:
            pc = player.pending_choice
            if pc.kind in (
                PendingChoiceKind.DISCOVER_MURLOC,
                PendingChoiceKind.TRIPLE_REWARD_DISCOVER,
            ):
                if not MiniBGGame._hand_has_free_slot(player):
                    return []
            return [
                int(Action.DISCOVER_PICK_0),
                int(Action.DISCOVER_PICK_1),
                int(Action.DISCOVER_PICK_2),
            ]

        actions: List[int] = []

        # Shop phase.
        can_act = player.shop_actions_used < MAX_SHOP_ACTIONS
        hand_full = sum(1 for m in player.hand if m is not None) >= HAND_SIZE
        board_full = len(player.board) >= BOARD_SIZE

        if can_act:
            if not hand_full:
                n_offers = shop_offers_count(player.tavern_tier)
                for slot in range(min(n_offers, len(player.shop))):
                    if (
                        player.shop[slot] is not None
                        and player.gold >= BUY_COST
                    ):
                        actions.append(int(Action.BUY_SLOT_0) + slot)

            for pos in range(BOARD_SIZE):
                if pos < len(player.board):
                    actions.append(int(Action.SELL_BOARD_0) + pos)

            if not board_full:
                bc_mult = self._battlecry_multiplier(player.board)
                for h in range(HAND_SIZE):
                    hm = player.hand[h]
                    if hm is None:
                        continue
                    needs = self._discover_cards_to_receive(hm, bc_mult)
                    free = sum(1 for s in player.hand if s is None) + 1
                    if needs > free:
                        continue
                    actions.append(int(Action.PLACE_HAND_0) + h)

            for h in range(HAND_SIZE):
                hm = player.hand[h]
                if hm is None or not self._hand_minion_can_magnetize(hm):
                    continue
                for b in range(len(player.board)):
                    if self._is_mech(player.board[b]):
                        actions.append(
                            int(Action.MAGNET_HAND_0_BOARD_0) + h * BOARD_SIZE + b
                        )

            if player.gold >= ROLL_COST:
                actions.append(int(Action.ROLL))

            if (
                player.tavern_tier < MAX_TIER
                and player.gold >= player.next_tier_up_cost
            ):
                actions.append(int(Action.LEVEL_UP))

        actions.append(int(Action.FINISH))
        actions.append(int(Action.FINISH_FREEZE_SHOP))
        return actions

    def apply_action(self, state: MiniBGState, action: ActionType) -> MiniBGState:
        if state.done:
            raise ValueError("Cannot apply action in terminal state")

        action_int = int(action)
        legal = self.legal_actions(state)
        if action_int not in legal:
            raise ValueError(
                f"Illegal action {action_int} in current state "
                f"(player={state.current_player_index}, "
                f"phase={state.players[state.current_player_index].phase.name}, "
                f"gold={state.players[state.current_player_index].gold})"
            )

        new_state = self._copy_state(state)
        idx = new_state.current_player_index
        player = new_state.players[idx]

        if player.phase == PlayerPhase.ORDER:
            # FINISH from order phase finalizes the round for this player.
            assert action_int == int(Action.FINISH)
            self._submit_order(player)
            self._after_player_finished(new_state, idx)
            return new_state

        # Shop phase.
        if action_int == int(Action.FINISH):
            # Phase flip only — same player keeps the turn to submit order.
            self._enter_order_phase(player)
            player.shop_freeze_next_round = False
            return new_state

        if action_int == int(Action.FINISH_FREEZE_SHOP):
            self._enter_order_phase(player)
            player.shop_freeze_next_round = True
            return new_state

        consumes_budget = False

        if player.pending_choice is not None:
            if not is_discover_pick_game_action(action_int):
                raise ValueError(
                    f"Expected DISCOVER_PICK_* while pending_choice, got {action_int}"
                )
            self._resolve_discover_pick(
                player,
                discover_pick_index(action_int),
                new_state.shop_excluded_race,
            )
            consumes_budget = (
                player.pending_choice is None
                and player.placed_minion_pending_after is None
            )
        elif int(Action.BUY_SLOT_0) <= action_int < int(Action.BUY_SLOT_0) + MAX_SHOP_SLOTS:
            self._do_buy(player, action_int - int(Action.BUY_SLOT_0))
            consumes_budget = True
        elif int(Action.SELL_BOARD_0) <= action_int < int(Action.SELL_BOARD_0) + BOARD_SIZE:
            self._do_sell(player, action_int - int(Action.SELL_BOARD_0))
            consumes_budget = True
        elif action_int == int(Action.ROLL):
            self._do_roll(player, new_state.shop_excluded_race)
            consumes_budget = True
        elif action_int == int(Action.LEVEL_UP):
            self._do_level_up(player, new_state.shop_excluded_race)
            consumes_budget = True
        elif int(Action.PLACE_HAND_0) <= action_int < int(Action.PLACE_HAND_0) + HAND_SIZE:
            self._do_place(
                player,
                action_int - int(Action.PLACE_HAND_0),
                new_state.shop_excluded_race,
            )
            consumes_budget = (
                player.pending_choice is None
                and player.placed_minion_pending_after is None
            )
        elif is_magnet_game_action(action_int):
            h, b = magnet_hand_board_from_game_action(action_int)
            self._do_magnet(player, h, b)
            consumes_budget = True
        else:
            raise ValueError(f"Unknown action {action_int}")

        if consumes_budget:
            player.shop_actions_used += 1
            if player.shop_actions_used >= MAX_SHOP_ACTIONS:
                # Budget exhausted -> auto-flip to order phase. Player still
                # owes order-phase moves (swap / finish) on their next turn.
                self._enter_order_phase(player)
                player.shop_freeze_next_round = False

        return new_state

    def reorder_board(
        self,
        state: MiniBGState,
        player_idx: int,
        perm: Sequence[int],
    ) -> MiniBGState:
        if state.done:
            raise ValueError("Cannot reorder in terminal state")
        if len(perm) != BOARD_SIZE:
            raise ValueError(f"perm must have length {BOARD_SIZE}, got {len(perm)}")
        if sorted(perm) != list(range(BOARD_SIZE)):
            raise ValueError(
                f"perm must be a permutation of 0..{BOARD_SIZE - 1}: got {tuple(perm)}"
            )

        new_state = self._copy_state(state)
        player = new_state.players[player_idx]
        k = len(player.board)
        # Compact-after-permute: only old positions < k correspond to real
        # minions. Take their target positions in order, drop empties.
        # For k = BOARD_SIZE this is a normal reorder; for k < BOARD_SIZE
        # all 24 perms collapse into k! distinct outcomes.
        player.board = [player.board[p] for p in perm if p < k]
        return new_state

    def swap_board_adjacent(
        self,
        state: MiniBGState,
        player_idx: int,
        i: int,
    ) -> MiniBGState:
        if state.done:
            raise ValueError("Cannot swap board in terminal state")
        new_state = self._copy_state(state)
        player = new_state.players[player_idx]
        if player.phase != PlayerPhase.ORDER:
            raise ValueError("swap_board_adjacent only valid in ORDER phase")
        b = player.board
        if not (0 <= i < len(b) - 1):
            raise ValueError(
                f"swap index {i} invalid for board length {len(b)}"
            )
        b[i], b[i + 1] = b[i + 1], b[i]
        return new_state

    # ------------------------------------------------------------------
    # Phase helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _enter_order_phase(player: PlayerState) -> None:
        if player.phase == PlayerPhase.SHOP:
            player.phase = PlayerPhase.ORDER

    @staticmethod
    def _submit_order(player: PlayerState) -> None:
        player.phase = PlayerPhase.DONE

    def _after_player_finished(self, state: MiniBGState, idx: int) -> None:
        """Called after a player transitions to DONE."""
        self._fire_on_turn_end(state.players[idx])
        other_idx = 1 - idx
        if state.players[other_idx].phase != PlayerPhase.DONE:
            state.current_player_index = other_idx
        else:
            self._resolve_battle_and_advance(state)

    # ------------------------------------------------------------------
    # Shop primitives
    # ------------------------------------------------------------------

    def _do_buy(self, player: PlayerState, slot: int) -> None:
        minion = player.shop[slot]
        assert minion is not None
        player.gold -= BUY_COST
        player.shop[slot] = None
        # First empty hand slot.
        h = next(
            (i for i in range(HAND_SIZE) if player.hand[i] is None), None
        )
        assert h is not None, "BUY illegal when hand is full (legal mask bug)"
        player.hand[h] = minion
        self._fire_on_buy(minion, player)
        self._try_resolve_triples_loop(player)

    def _do_place(
        self,
        player: PlayerState,
        hand_slot: int,
        shop_excluded_race: Optional[Race],
    ) -> None:
        minion = player.hand[hand_slot]
        assert minion is not None
        assert len(player.board) < BOARD_SIZE
        queued_triple_reward = minion.is_golden and minion.from_triple_merge
        player.hand[hand_slot] = None
        player.board.append(minion)
        self._fire_shop_friendly_summoned(player, minion)
        player.placed_minion_board_index = len(player.board) - 1
        player.placed_minion_pending_after = minion
        self._fire_on_place(minion, player, shop_excluded_race)
        if player.pending_choice is None:
            try:
                idx = player.board.index(minion)
            except ValueError:
                pass
            else:
                self._fire_after_friendly_minion_placed(player, player.board[idx])
            player.placed_minion_board_index = None
            player.placed_minion_pending_after = None

        self._try_resolve_triples_loop(player)

        if queued_triple_reward and minion in player.board and minion.is_golden:
            if player.pending_choice is None:
                self._try_open_triple_reward_discover(player, shop_excluded_race)
            else:
                player.triple_reward_discover_pending = True
            minion.from_triple_merge = False

        self._flush_triple_reward_queue_if_idle(player, shop_excluded_race)

    def _resolve_discover_pick(
        self,
        player: PlayerState,
        pick_slot: int,
        shop_excluded_race: Optional[Race],
    ) -> None:
        pc = player.pending_choice
        assert pc is not None
        assert 0 <= pick_slot <= 2
        choice_token = pc.options[pick_slot]
        extra = pc.extra_modals_after
        hand_discover = pc.kind in (
            PendingChoiceKind.DISCOVER_MURLOC,
            PendingChoiceKind.TRIPLE_REWARD_DISCOVER,
        )
        if hand_discover:
            h = next((i for i in range(HAND_SIZE) if player.hand[i] is None), None)
            if h is None:
                raise ValueError(
                    "DISCOVER pick with full hand; legal mask must require a free hand slot"
                )
            player.hand[h] = make_minion(choice_token)
            self._try_resolve_triples_loop(player)
        else:
            for m in player.board:
                if is_murloc_board_minion(m):
                    apply_adapt_key_to_minion(m, choice_token)

        chain_next = extra > 0
        if chain_next and hand_discover and not MiniBGGame._hand_has_free_slot(player):
            chain_next = False

        if chain_next:
            player.pending_choice = self._roll_pending_modal(
                player, pc.kind, extra - 1, shop_excluded_race
            )
        else:
            player.pending_choice = None
            ref = player.placed_minion_pending_after
            if ref is not None:
                if ref in player.board:
                    self._fire_after_friendly_minion_placed(player, ref)
                player.placed_minion_pending_after = None
                player.placed_minion_board_index = None
            self._flush_triple_reward_queue_if_idle(player, shop_excluded_race)

    def _roll_pending_modal(
        self,
        player: PlayerState,
        kind: PendingChoiceKind,
        remaining_after: int,
        shop_excluded_race: Optional[Race],
    ) -> PendingChoice:
        if kind == PendingChoiceKind.DISCOVER_MURLOC:
            opts = roll_discover_murloc_triple(
                self._rng, player.tavern_tier, shop_excluded_race
            )
        elif kind == PendingChoiceKind.TRIPLE_REWARD_DISCOVER:
            opts = roll_triple_reward_discover_triple(
                self._rng, player.tavern_tier, shop_excluded_race
            )
        else:
            opts = roll_adapt_triple(self._rng)
        return PendingChoice(kind, opts, remaining_after)

    @staticmethod
    def _discover_cards_to_receive(placed: Minion, battlecry_mult: int) -> int:
        n_need = 0
        for ab in placed.abilities:
            if ab.trigger != Trigger.ON_PLACE:
                continue
            if isinstance(ab.effect, DiscoverMurlocEffect):
                n_need = max(n_need, battlecry_mult * ab.effect.repeats)
        return n_need

    @staticmethod
    def _is_mech(m: Minion) -> bool:
        return m.race in (Race.MECHANICAL, Race.ALL)

    @staticmethod
    def _hand_minion_can_magnetize(m: Minion) -> bool:
        return Keyword.MAGNETIC in m.all_keywords and MiniBGGame._is_mech(m)

    @staticmethod
    def merge_magnetic_inplace(target: Minion, magnet: Minion) -> None:
        """HS-style magnetic: keep target identity/buffs; add magnet stats/keywords/DRs."""
        target.base_attack += magnet.raw_attack
        target.base_health += magnet.max_health
        combined_kw = (
            target.keywords
            | target.granted_keywords
            | magnet.keywords
            | magnet.granted_keywords
        ) - {Keyword.MAGNETIC}
        target.keywords = combined_kw
        target.granted_keywords = frozenset()
        target.has_shield = target.has_shield or magnet.has_shield

        nt = [ab for ab in target.abilities if ab.trigger != Trigger.ON_DEATH]
        dt = [ab for ab in target.abilities if ab.trigger == Trigger.ON_DEATH]
        nm = [ab for ab in magnet.abilities if ab.trigger != Trigger.ON_DEATH]
        dm = [ab for ab in magnet.abilities if ab.trigger == Trigger.ON_DEATH]
        target.abilities = tuple(nt + nm + dt + dm)

    def _do_magnet(self, player: PlayerState, hand_slot: int, board_pos: int) -> None:
        magnet = player.hand[hand_slot]
        assert magnet is not None
        assert board_pos < len(player.board)
        target = player.board[board_pos]
        assert self._is_mech(target)
        assert self._hand_minion_can_magnetize(magnet)
        player.hand[hand_slot] = None
        self.merge_magnetic_inplace(target, magnet)
        self._try_resolve_triples_loop(player)

    @staticmethod
    def _merge_three_non_golden_into_golden(card_id: str, a: Minion, b: Minion, c: Minion) -> Minion:
        tpl = CARD_TEMPLATES[card_id]
        merged_kw = (
            a.keywords
            | a.granted_keywords
            | b.keywords
            | b.granted_keywords
            | c.keywords
            | c.granted_keywords
        )
        shield = a.has_shield or b.has_shield or c.has_shield or (
            Keyword.SHIELD in merged_kw
        )
        return Minion(
            card_id=card_id,
            base_attack=tpl.base_attack * 2,
            base_health=tpl.base_health * 2,
            tier=tpl.tier,
            name=tpl.name,
            bonus_attack=a.bonus_attack + b.bonus_attack + c.bonus_attack,
            bonus_health=a.bonus_health + b.bonus_health + c.bonus_health,
            race=tpl.race,
            keywords=frozenset(merged_kw),
            granted_keywords=frozenset(),
            abilities=triple_merge_golden_abilities(card_id),
            has_shield=shield,
            is_token=tpl.is_token,
            is_golden=True,
            from_triple_merge=True,
            dbf_id=tpl.dbf_id,
        )

    def _try_resolve_triples_loop(self, player: PlayerState) -> None:
        for _ in range(24):
            if not self._resolve_one_triple(player):
                break

    def _resolve_one_triple(self, player: PlayerState) -> bool:
        groups: Dict[str, List[Tuple[str, int, Minion]]] = {}
        for i, m in enumerate(player.board):
            if not m.is_golden:
                groups.setdefault(m.card_id, []).append(("b", i, m))
        for i, hm in enumerate(player.hand):
            if hm is not None and not hm.is_golden:
                groups.setdefault(hm.card_id, []).append(("h", i, hm))
        candidate: Optional[str] = None
        for cid in sorted(groups.keys()):
            if len(groups[cid]) >= 3:
                candidate = cid
                break
        if candidate is None:
            return False
        ordered = sorted(
            groups[candidate], key=lambda t: (0 if t[0] == "b" else 1, t[1])
        )[:3]
        m0, m1, m2 = ordered[0][2], ordered[1][2], ordered[2][2]
        merged = MiniBGGame._merge_three_non_golden_into_golden(candidate, m0, m1, m2)
        for _, idx, _ in sorted(
            (t for t in ordered if t[0] == "b"), key=lambda t: -t[1]
        ):
            del player.board[idx]
        for _, idx, _ in sorted(
            (t for t in ordered if t[0] == "h"), key=lambda t: -t[1]
        ):
            player.hand[idx] = None
        hslot = next((i for i in range(HAND_SIZE) if player.hand[i] is None), None)
        assert hslot is not None, "triple merge with full hand (bug)"
        player.hand[hslot] = merged
        return True

    @staticmethod
    def _hand_has_free_slot(player: PlayerState) -> bool:
        return any(s is None for s in player.hand)

    def _try_open_triple_reward_discover(
        self,
        player: PlayerState,
        shop_excluded_race: Optional[Race],
    ) -> None:
        """Open triple-reward discover only if hand has room for the received minion."""

        if not MiniBGGame._hand_has_free_slot(player):
            player.triple_reward_discover_pending = True
            return
        player.triple_reward_discover_pending = False
        opts = roll_triple_reward_discover_triple(
            self._rng, player.tavern_tier, shop_excluded_race
        )
        player.pending_choice = PendingChoice(
            PendingChoiceKind.TRIPLE_REWARD_DISCOVER, opts, 0
        )

    def _flush_triple_reward_queue_if_idle(
        self, player: PlayerState, shop_excluded_race: Optional[Race]
    ) -> None:
        if player.pending_choice is None and player.triple_reward_discover_pending:
            self._try_open_triple_reward_discover(player, shop_excluded_race)

    @staticmethod
    def _player_has_hero_immune(player: PlayerState) -> bool:
        for m in player.board:
            for ab in m.abilities:
                if ab.trigger == Trigger.AURA and isinstance(ab.effect, HeroImmuneAura):
                    return True
        return False

    @staticmethod
    def _minion_matches_tribe(m: Minion, tribe: Any) -> bool:
        if m.race is None:
            return False
        if tribe == Race.ALL or m.race == Race.ALL:
            return True
        return m.race == tribe

    def _apply_buff_all_other_tribe(
        self,
        player: PlayerState,
        source: Minion,
        effect: BuffAllOtherOfTribe,
    ) -> None:
        for m in player.board:
            if m is source or not self._minion_matches_tribe(m, effect.tribe):
                continue
            m.bonus_attack += effect.attack
            m.bonus_health += effect.health

    def _apply_buff_all_friendly_tribe(
        self, player: PlayerState, effect: BuffAllFriendlyOfTribe
    ) -> None:
        for m in player.board:
            if not self._minion_matches_tribe(m, effect.tribe):
                continue
            m.bonus_attack += effect.attack
            m.bonus_health += effect.health

    def _apply_buff_all_keyword(
        self, player: PlayerState, effect: BuffAllWithKeyword
    ) -> None:
        for m in player.board:
            if effect.keyword not in m.all_keywords:
                continue
            m.bonus_attack += effect.attack
            m.bonus_health += effect.health

    def _apply_grant_keyword_random(
        self,
        player: PlayerState,
        source: Minion,
        effect: GrantKeywordRandomFriendly,
    ) -> None:
        pool = list(player.board)
        if effect.exclude_self:
            pool = [m for m in pool if m is not source]
        if effect.filter_race is not None:
            pool = [m for m in pool if self._minion_matches_tribe(m, effect.filter_race)]
        for _ in range(max(1, effect.repeats)):
            if not pool:
                return
            tgt = pool[int(self._rng.integers(0, len(pool)))]
            tgt.keywords = frozenset(tgt.keywords | {effect.keyword})
            if effect.keyword == Keyword.TAUNT:
                pass
            if effect.keyword == Keyword.SHIELD:
                tgt.has_shield = True

    def _apply_summon_from_place(
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
            self._fire_shop_friendly_summoned(player, tok)
            insert_at += 1
        self._try_resolve_triples_loop(player)

    def _fire_shop_friendly_summoned(self, player: PlayerState, summoned: Minion) -> None:
        for m in player.board:
            if m is summoned:
                continue
            for ab in m.abilities:
                if ab.trigger != Trigger.ON_FRIENDLY_MINION_SUMMONED:
                    continue
                eff = ab.effect
                if isinstance(eff, BuffSummonedIfRace):
                    if self._minion_matches_tribe(summoned, eff.tribe):
                        summoned.bonus_attack += eff.attack
                        summoned.bonus_health += eff.health
                elif isinstance(eff, GrantListenerKeywordIfSummonedMatches):
                    if self._minion_matches_tribe(summoned, eff.tribe):
                        m.keywords = frozenset(m.keywords | {eff.keyword})
                        if eff.keyword == Keyword.SHIELD:
                            m.has_shield = True
                elif isinstance(eff, BuffListenerIfSummonedMatches):
                    if self._minion_matches_tribe(summoned, eff.tribe):
                        m.bonus_attack += eff.attack
                        m.bonus_health += eff.health

    def _damage_hero(self, player: PlayerState, amount: int) -> None:
        if amount <= 0 or self._player_has_hero_immune(player):
            return
        player.health -= amount
        player.hero_damage_taken_total += amount

    def _apply_shop_effect(
        self,
        player: PlayerState,
        source: Minion,
        effect: Effect,
        placed: Optional[Minion],
    ) -> None:
        if isinstance(effect, BuffRandomFriendly):
            self._apply_buff_random(source, effect, player.board)
        elif isinstance(effect, BuffOnePerListedTribeFriendly):
            self._apply_buff_one_per_listed_tribe(source, effect, player.board)
        elif isinstance(effect, DealHeroDamage):
            self._damage_hero(player, effect.amount)
        elif isinstance(effect, BuffSelf):
            source.bonus_attack += effect.attack
            source.bonus_health += effect.health
        elif isinstance(effect, BuffSelfFromHeroDamageTaken):
            x = player.hero_damage_taken_total
            source.bonus_health += x
        elif isinstance(effect, BuffAdjacentBattlecry):
            self._apply_buff_adjacent(player, source, effect)
        elif isinstance(effect, BuffAllOtherOfTribe):
            self._apply_buff_all_other_tribe(player, source, effect)
        elif isinstance(effect, BuffAllFriendlyOfTribe):
            self._apply_buff_all_friendly_tribe(player, effect)
        elif isinstance(effect, BuffAllWithKeyword):
            self._apply_buff_all_keyword(player, effect)
        elif isinstance(effect, GrantKeywordRandomFriendly):
            self._apply_grant_keyword_random(player, source, effect)
        elif isinstance(effect, SummonEffect):
            self._apply_summon_from_place(player, source, effect)
        elif isinstance(
            effect,
            (
                SummonRandomMinionEffect,
                StatAura,
                TribalOtherStatAura,
                KeywordStatAura,
                AdjacentStatAura,
                HeroImmuneAura,
                SummonFirstDeadFriendlyMechsThisCombat,
                BattlecryMultiplierAura,
                DeathrattleMultiplierAura,
                SummonMultiplierAura,
                ZappTargeting,
                CleaveOnAttack,
                DiscoverMurlocEffect,
                AdaptAllMurlocsEffect,
                PogoHopperBattlecry,
            ),
        ):
            pass

    @staticmethod
    def _battlecry_multiplier(board: List[Minion]) -> int:
        p = 1
        for m in board:
            for ab in m.abilities:
                if ab.trigger == Trigger.AURA and isinstance(
                    ab.effect, BattlecryMultiplierAura
                ):
                    p *= ab.effect.factor
        return p

    def _fire_on_place(
        self,
        placed: Minion,
        player: PlayerState,
        shop_excluded_race: Optional[Race],
    ) -> None:
        mult = self._battlecry_multiplier(player.board)
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
                player.pending_choice = PendingChoice(
                    PendingChoiceKind.DISCOVER_MURLOC, opts, total - 1
                )
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
                self._apply_shop_effect(player, placed, ab.effect, placed)

    def _fire_after_friendly_minion_placed(
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
                    if not any(
                        x.trigger == Trigger.ON_PLACE for x in placed.abilities
                    ):
                        continue
                    m.bonus_attack += eff.attack
                    m.bonus_health += eff.health
                    continue
                self._apply_shop_effect(player, m, ab.effect, placed)

    def _do_sell(self, player: PlayerState, pos: int) -> None:
        del player.board[pos]
        player.gold += SELL_REWARD
        self._try_resolve_triples_loop(player)

    def _do_roll(
        self, player: PlayerState, shop_excluded_race: Optional[Race]
    ) -> None:
        player.gold -= ROLL_COST
        self._refresh_shop(player, shop_excluded_race)

    def _do_level_up(
        self, player: PlayerState, shop_excluded_race: Optional[Race]
    ) -> None:
        cost = player.next_tier_up_cost
        player.gold -= cost
        old_tier = player.tavern_tier
        player.tavern_tier += 1
        if player.tavern_tier < MAX_TIER:
            player.next_tier_up_cost = LEVEL_UP_COSTS[player.tavern_tier]
        old_n = shop_offers_count(old_tier)
        new_n = shop_offers_count(player.tavern_tier)
        pool = self._tavern_card_pool(player.tavern_tier, shop_excluded_race)
        while len(player.shop) < MAX_SHOP_SLOTS:
            player.shop.append(None)
        for i in range(old_n, new_n):
            card_id = pool[int(self._rng.integers(0, len(pool)))]
            player.shop[i] = make_minion(card_id)

    # ------------------------------------------------------------------
    # Effects
    # ------------------------------------------------------------------

    @staticmethod
    def _apply_buff_adjacent(
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

    def _apply_buff_random(
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
                pool = [m for m in pool if self._minion_matches_tribe(m, effect.filter_race)]
            if not pool:
                return
            target = pool[int(self._rng.integers(0, len(pool)))]
            target.bonus_attack += effect.attack
            target.bonus_health += effect.health
            if effect.grant_taunt:
                target.keywords = frozenset(target.keywords | {Keyword.TAUNT})

    def _apply_buff_one_per_listed_tribe(
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
            pool = [m for m in pool if self._minion_matches_tribe(m, tribe)]
            if not pool:
                continue
            target = pool[int(self._rng.integers(0, len(pool)))]
            target.bonus_attack += effect.attack
            target.bonus_health += effect.health

    def _fire_on_buy(self, minion: Minion, player: PlayerState) -> None:
        for ab in minion.abilities:
            if ab.trigger == Trigger.ON_BUY and isinstance(ab.effect, BuffRandomFriendly):
                self._apply_buff_random(minion, ab.effect, player.board)

    def _fire_on_turn_end(self, player: PlayerState) -> None:
        for source in list(player.board):
            for ab in source.abilities:
                if ab.trigger != Trigger.ON_TURN_END:
                    continue
                if isinstance(ab.effect, BuffRandomFriendly):
                    self._apply_buff_random(source, ab.effect, player.board)
                elif isinstance(ab.effect, BuffOnePerListedTribeFriendly):
                    self._apply_buff_one_per_listed_tribe(
                        source, ab.effect, player.board
                    )

    def _fire_on_turn_start(self, player: PlayerState) -> None:
        """After round increment, before shop reroll: board left-to-right, then hand slots."""
        for source in list(player.board):
            for ab in source.abilities:
                if ab.trigger != Trigger.ON_TURN_START:
                    continue
                e = ab.effect
                if isinstance(e, BuffRandomFriendly):
                    self._apply_buff_random(source, e, player.board)
                elif isinstance(e, BuffOnePerListedTribeFriendly):
                    self._apply_buff_one_per_listed_tribe(source, e, player.board)
                elif isinstance(e, BuffSelf):
                    self._apply_shop_effect(player, source, e, None)
        for source in list(player.hand):
            if source is None:
                continue
            for ab in source.abilities:
                if ab.trigger != Trigger.ON_TURN_START:
                    continue
                e = ab.effect
                if isinstance(e, BuffRandomFriendly):
                    self._apply_buff_random(source, e, player.board)
                elif isinstance(e, BuffOnePerListedTribeFriendly):
                    self._apply_buff_one_per_listed_tribe(source, e, player.board)
                elif isinstance(e, BuffSelf):
                    self._apply_shop_effect(player, source, e, None)

    @staticmethod
    def _tavern_card_pool(
        tavern_tier: int,
        shop_excluded_race: Optional[Race],
    ) -> List[str]:
        pool = shop_pool_for_tier(
            tavern_tier, shop_excluded_race=shop_excluded_race
        )
        if not pool:
            pool = shop_pool_for_tier(tavern_tier, shop_excluded_race=None)
        return pool

    def _refresh_shop(
        self, player: PlayerState, shop_excluded_race: Optional[Race]
    ) -> None:
        n = shop_offers_count(player.tavern_tier)
        pool = self._tavern_card_pool(player.tavern_tier, shop_excluded_race)
        new_shop: List[Optional[Minion]] = [None] * MAX_SHOP_SLOTS
        for i in range(n):
            card_id = pool[int(self._rng.integers(0, len(pool)))]
            new_shop[i] = make_minion(card_id)
        player.shop = new_shop

    def _refresh_shop_fill_empty_slots(
        self, player: PlayerState, shop_excluded_race: Optional[Race]
    ) -> None:
        """Keep existing offers in active slots; reroll only empty offers; clear inactive tiers."""
        n = shop_offers_count(player.tavern_tier)
        pool = self._tavern_card_pool(player.tavern_tier, shop_excluded_race)
        while len(player.shop) < MAX_SHOP_SLOTS:
            player.shop.append(None)
        for i in range(MAX_SHOP_SLOTS):
            if i >= n:
                player.shop[i] = None
            elif player.shop[i] is None:
                card_id = pool[int(self._rng.integers(0, len(pool)))]
                player.shop[i] = make_minion(card_id)

    # ------------------------------------------------------------------
    # Round resolution

    def _resolve_battle_and_advance(self, state: MiniBGState) -> None:
        p0_has_initiative = (state.round_number % 2 == 1) == (state.initiative_player == 0)
        dmg_p0, dmg_p1 = simulate_battle(
            state.players[0].board,
            state.players[1].board,
            p0_has_initiative=p0_has_initiative,
            rng=self._rng,
            p0_tavern_tier=state.players[0].tavern_tier,
            p1_tavern_tier=state.players[1].tavern_tier,
        )
        # Retail BG: combat only determines hero damage; recruitment boards are unchanged.
        state.players[0].health -= dmg_p0
        state.players[1].health -= dmg_p1

        p0_dead = state.players[0].health <= 0
        p1_dead = state.players[1].health <= 0

        if p0_dead or p1_dead:
            state.done = True
            if p0_dead and p1_dead:
                state.winner = 0
            elif p0_dead:
                state.winner = -1
            else:
                state.winner = 1
            return

        if state.round_number >= MAX_ROUNDS:
            state.done = True
            state.winner = 0
            return

        state.round_number += 1
        for p in state.players:
            if p.tavern_tier < MAX_TIER:
                p.next_tier_up_cost = max(
                    0, p.next_tier_up_cost - LEVEL_UP_DISCOUNT_PER_ROUND
                )
            p.gold = gold_for_round(state.round_number)
            p.phase = PlayerPhase.SHOP
            p.shop_actions_used = 0
            p.pending_choice = None
            p.triple_reward_discover_pending = False
            p.placed_minion_board_index = None
            p.placed_minion_pending_after = None
            # Start-of-recruitment: board (L→R) then hand; then shop (full reroll or
            # reuse frozen offers and fill empties).
            self._fire_on_turn_start(p)
            if p.shop_freeze_next_round:
                self._refresh_shop_fill_empty_slots(p, state.shop_excluded_race)
                p.shop_freeze_next_round = False
            else:
                self._refresh_shop(p, state.shop_excluded_race)
        state.current_player_index = 0

    def _fresh_player(
        self, round_number: int, shop_excluded_race: Optional[Race]
    ) -> PlayerState:
        player = PlayerState(
            health=STARTING_HEALTH,
            hero_damage_taken_total=0,
            gold=gold_for_round(round_number),
            tavern_tier=STARTING_TIER,
            next_tier_up_cost=LEVEL_UP_COSTS[STARTING_TIER],
            board=[],
            shop=[None for _ in range(MAX_SHOP_SLOTS)],
            hand=[None for _ in range(HAND_SIZE)],
            phase=PlayerPhase.SHOP,
            shop_actions_used=0,
            pending_choice=None,
            placed_minion_board_index=None,
            placed_minion_pending_after=None,
        )
        self._refresh_shop(player, shop_excluded_race)
        return player

    @staticmethod
    def _copy_state(state: MiniBGState) -> MiniBGState:
        new_players = (
            MiniBGGame._copy_player(state.players[0]),
            MiniBGGame._copy_player(state.players[1]),
        )
        return MiniBGState(
            players=new_players,
            round_number=state.round_number,
            current_player_index=state.current_player_index,
            initiative_player=state.initiative_player,
            winner=state.winner,
            done=state.done,
            shop_excluded_race=state.shop_excluded_race,
        )

    @staticmethod
    def _copy_player(p: PlayerState) -> PlayerState:
        new_board = [copy(m) for m in p.board]
        remapped_pending: Optional[Minion] = None
        pend = p.placed_minion_pending_after
        if pend is not None:
            try:
                i = p.board.index(pend)
            except ValueError:
                pass
            else:
                if 0 <= i < len(new_board):
                    remapped_pending = new_board[i]
        placed_idx: Optional[int] = None
        if remapped_pending is not None:
            try:
                placed_idx = new_board.index(remapped_pending)
            except ValueError:
                placed_idx = None
        return PlayerState(
            health=p.health,
            hero_damage_taken_total=p.hero_damage_taken_total,
            gold=p.gold,
            tavern_tier=p.tavern_tier,
            next_tier_up_cost=p.next_tier_up_cost,
            board=new_board,
            shop=[copy(m) if m is not None else None for m in p.shop],
            hand=[copy(m) if m is not None else None for m in p.hand],
            phase=p.phase,
            shop_actions_used=p.shop_actions_used,
            shop_freeze_next_round=p.shop_freeze_next_round,
            pending_choice=(
                PendingChoice(
                    p.pending_choice.kind,
                    p.pending_choice.options,
                    p.pending_choice.extra_modals_after,
                )
                if p.pending_choice is not None
                else None
            ),
            placed_minion_board_index=placed_idx,
            placed_minion_pending_after=remapped_pending,
            triple_reward_discover_pending=p.triple_reward_discover_pending,
            pogo_hoppers_played=p.pogo_hoppers_played,
        )


__all__ = ["MiniBGGame", "PLAYER_TOKENS"]
