from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np

from src.bg_lobby import eight_player as bg_lobby_eight
from src.bg_lobby.player import copy_player_state
from src.bg_lobby.shared_pool import SharedCardPool, build_initial_shared_pool
from src.bg_lobby.shop_order import sample_shop_turn_order
from src.bg_player_turn import PlayerTurnContext, PlayerTurnEngine
from src.games.turn_based_game import Action as ActionType
from src.games.turn_based_game import TurnBasedGame

from src.bg_recruitment import hero_passives
from src.bg_recruitment import place as recruitment_place
from src.bg_recruitment import shop as recruitment_shop
from src.bg_recruitment import triples as recruitment_triples
from src.bg_recruitment import discover as recruitment_discover
from src.bg_recruitment.shop_triggers import ShopTriggers

from src.bg_catalog.cards import (
    make_minion,
    normalize_shop_excluded_races,
    shop_minion_allowed_with_exclusion,
)
from src.bg_catalog.patch_context import PatchContext, load_patch_context

from . import actions as bglike_actions
from .tribe_pref import draw_tribe_pref
from .state import (
    BGLikeState,
    Minion,
    PendingChoice,
    PlayerPhase,
    PlayerState,
    Race,
)


def _resolve_patch(
    patch: Optional[PatchContext],
    patch_dir: Optional[str],
) -> PatchContext:
    if patch is not None and patch_dir is not None:
        raise ValueError("pass patch or patch_dir, not both")
    if patch_dir is not None:
        return load_patch_context(patch_dir)
    if patch is not None:
        return patch
    raise ValueError("BGLikeGame requires patch or patch_dir")


class BGLikeGame(TurnBasedGame[BGLikeState]):
    def __init__(
        self,
        seed: Optional[int] = None,
        *,
        shop_excluded_race: Optional[Race | Tuple[Race, ...]] = None,
        shop_excluded_count: Optional[int] = None,
        shop_full_tribes: bool = False,
        high_mode: bool = False,
        with_heroes: bool = False,
        with_tribe_pref: bool = False,
        patch: Optional[PatchContext] = None,
        patch_dir: Optional[str] = None,
    ) -> None:
        self._rng = np.random.default_rng(seed)
        self._shop_full_tribes = shop_full_tribes
        # When set, each seat is assigned a random hero (passive power) at game
        # start. Off ⇒ classic no-hero seats (obs/actions unchanged).
        self._with_heroes = bool(with_heroes)
        # When set, each seat draws a tribe-preference vector at game start.
        # Off ⇒ every seat carries an empty vector, which every consumer reads
        # as "no preference" (obs block absent, shaping term zero).
        self._with_tribe_pref = bool(with_tribe_pref)
        # Deterministic flag: when set, ``initial_state`` builds a "high mode"
        # start (all players at tier 5 / 10 gold / round 8 with a random tier-5
        # + tier-6 board). The *decision* of which games are high mode belongs
        # to the trainer (curriculum + its RNG); the game is a pure mechanism.
        self._high_mode = bool(high_mode)
        self._patch = _resolve_patch(patch, patch_dir)
        self._shop_excluded_race_fixed = (
            tuple(normalize_shop_excluded_races(shop_excluded_race))
            if shop_excluded_race is not None
            else None
        )
        self._shop_excluded_count = (
            self._patch.meta.rotation_excluded_count
            if shop_excluded_count is None
            else int(shop_excluded_count)
        )
        self._shop_triggers = ShopTriggers(self._rng, patch=self._patch)
        self._player_turn = PlayerTurnEngine(bglike_actions)

    def _turn_ctx(self, state: BGLikeState) -> PlayerTurnContext:
        return PlayerTurnContext(
            rng=self._rng,
            triggers=self._shop_triggers,
            shop_excluded_race=state.shop_excluded_race,
            shared_pool=state.shared_pool,
            patch=self._patch,
        )

    def _pick_shop_excluded_race(self) -> Optional[Tuple[Race, ...]]:
        if self._shop_excluded_race_fixed is not None:
            return self._shop_excluded_race_fixed
        if self._shop_full_tribes:
            return None
        tribes = self._patch.meta.rotation_tribes
        max_excluded = max(0, len(tribes) - 1)
        excluded_count = max(0, min(int(self._shop_excluded_count), max_excluded))
        if excluded_count <= 0:
            return None
        picks = self._rng.choice(len(tribes), size=excluded_count, replace=False)
        return tuple(tribes[int(i)] for i in picks)

    def set_high_mode(self, flag: bool) -> None:
        """Trainer-controlled switch for the next ``initial_state`` build."""
        self._high_mode = bool(flag)

    def initial_state(self) -> BGLikeState:
        n = bglike_actions.NUM_PLAYERS
        high_mode = self._high_mode
        round_number = (
            bglike_actions.HIGH_MODE_START_ROUND if high_mode else 1
        )
        shop_excluded = self._pick_shop_excluded_race()
        shared_pool = build_initial_shared_pool(
            shop_excluded,
            patch=self._patch,
        )
        make_player = self._fresh_player_high if high_mode else self._fresh_player
        players = tuple(
            make_player(
                round_number=round_number,
                shop_excluded_race=shop_excluded,
                shared_pool=shared_pool,
            )
            for _ in range(n)
        )
        alive = tuple(range(n))
        order = sample_shop_turn_order(self._rng, len(alive))
        state = BGLikeState(
            players=players,
            alive=alive,
            round_number=round_number,
            combat_round=0,
            full_lobby_cycle_round=0,
            current_player_index=order[0],
            shop_turn_order=order,
            recent_opponents=tuple(() for _ in range(n)),
            eliminated=(),
            pairings=(),
            initiative_player=int(self._rng.integers(0, n)),
            winner=None,
            done=False,
            shop_excluded_race=shop_excluded,
            shared_pool=shared_pool,
            patch_build=self._patch.build,
        )
        # Round-1 pairings are observable during the first shop phase.
        bg_lobby_eight.draw_combat_pairings(state, rng=self._rng)
        return state

    def current_player(self, state: BGLikeState) -> int:
        return state.current_player_index

    def is_terminal(self, state: BGLikeState) -> bool:
        return state.done

    def winner(self, state: BGLikeState) -> Optional[int]:
        return state.winner

    def legal_actions(self, state: BGLikeState) -> Sequence[ActionType]:
        if state.done:
            return []
        player = state.players[state.current_player_index]
        if player.pending_choice is not None:
            pc = player.pending_choice
            if recruitment_discover.is_hand_discover_kind(pc.kind):
                if not recruitment_triples.hand_has_free_slot(player):
                    raise RuntimeError(
                        "hand discover with full hand (legal mask bug)"
                    )
        return self._player_turn.legal_actions(player, self._patch.meta.ruleset)

    def apply_action(self, state: BGLikeState, action: ActionType) -> BGLikeState:
        if state.done:
            raise ValueError("Cannot apply action in terminal state")

        action_int = int(action)
        legal = self.legal_actions(state)
        if action_int not in legal:
            raise ValueError(
                f"Illegal action {action_int} "
                f"(player={state.current_player_index}, "
                f"phase={state.players[state.current_player_index].phase.name})"
            )

        new_state = self._copy_state(state)
        idx = new_state.current_player_index
        player = new_state.players[idx]
        ctx = self._turn_ctx(new_state)

        if action_int == int(bglike_actions.Action.FINISH):
            self._player_turn.end_turn(player, freeze_shop=False)
            self._after_player_finished(new_state, idx)
            return new_state

        if action_int == int(bglike_actions.Action.FINISH_FREEZE_SHOP):
            self._player_turn.end_turn(player, freeze_shop=True)
            self._after_player_finished(new_state, idx)
            return new_state

        consumes_budget = self._player_turn.apply(
            player,
            action_int,
            ctx,
            shop_excluded_race=new_state.shop_excluded_race,
        )
        if consumes_budget:
            player.shop_actions_used += 1

        return new_state

    def reorder_board(
        self,
        state: BGLikeState,
        player_idx: int,
        perm: Sequence[int],
    ) -> BGLikeState:
        from src.envs.minibg.board_order import reorder_board

        return reorder_board(
            state,
            player_idx,
            perm,
            board_size=bglike_actions.BOARD_SIZE,
            copy_state=self._copy_state,
        )

    def swap_board_adjacent(
        self,
        state: BGLikeState,
        player_idx: int,
        i: int,
    ) -> BGLikeState:
        from src.envs.minibg.board_order import swap_board_adjacent

        return swap_board_adjacent(
            state,
            player_idx,
            i,
            copy_state=self._copy_state,
        )

    def _after_player_finished(self, state: BGLikeState, idx: int) -> None:
        bg_lobby_eight.after_player_finished(
            state,
            idx,
            fire_on_turn_end=self._shop_triggers.fire_on_turn_end,
            resolve_combat_round=self._resolve_combat_round,
        )

    def _refresh_shop(
        self,
        player: PlayerState,
        shop_excluded_race: Optional[Race],
        *,
        shared_pool: Optional[SharedCardPool],
    ) -> None:
        recruitment_shop.refresh_shop(
            player,
            shop_excluded_race,
            rng=self._rng,
            shared_pool=shared_pool,
            frozen_slots=player.shop_frozen,
            patch=self._patch,
        )

    def _refresh_shop_fill_empty_slots(
        self,
        player: PlayerState,
        shop_excluded_race: Optional[Race],
        *,
        shared_pool: Optional[SharedCardPool],
    ) -> None:
        recruitment_shop.refresh_shop_fill_empty_slots(
            player,
            shop_excluded_race,
            rng=self._rng,
            shared_pool=shared_pool,
            frozen_slots=player.shop_frozen,
            patch=self._patch,
        )

    def _resolve_combat_round(self, state: BGLikeState) -> None:
        pool = state.shared_pool

        def refresh_shop(p: PlayerState, exc: Optional[Race]) -> None:
            self._refresh_shop(p, exc, shared_pool=pool)

        def refresh_fill(p: PlayerState, exc: Optional[Race]) -> None:
            self._refresh_shop_fill_empty_slots(p, exc, shared_pool=pool)

        bg_lobby_eight.resolve_combat_round(
            state,
            rng=self._rng,
            combat_board_max=bglike_actions.COMBAT_BOARD_MAX,
            board_size=bglike_actions.BOARD_SIZE,
            ruleset=self._patch.meta.ruleset,
            fire_on_turn_start=self._shop_triggers.fire_on_turn_start,
            refresh_shop=refresh_shop,
            refresh_shop_fill_empty_slots=refresh_fill,
        )

    def _assign_tribe_pref(self, player: PlayerState) -> None:
        """Draw this seat's tribe-preference vector, once, at construction.

        Drawn from the game RNG so a seeded game reproduces its preferences
        along with everything else.
        """
        if not self._with_tribe_pref:
            return
        player.tribe_pref = draw_tribe_pref(self._rng)

    def _fresh_player(
        self,
        round_number: int,
        shop_excluded_race: Optional[Race],
        *,
        shared_pool: SharedCardPool,
    ) -> PlayerState:
        ruleset = self._patch.meta.ruleset
        player = PlayerState(
            health=ruleset.starting_health,
            hero_damage_taken_total=0,
            gold=ruleset.gold_for_round(round_number),
            tavern_tier=bglike_actions.STARTING_TIER,
            ruleset=ruleset,
            board=[],
            shop=[None for _ in range(bglike_actions.MAX_SHOP_SLOTS)],
            hand=[None for _ in range(bglike_actions.HAND_SIZE)],
            phase=PlayerPhase.SHOP,
            shop_actions_used=0,
            pending_choice=None,
            placed_minion_board_index=None,
            placed_minion_pending_after=None,
        )
        self._assign_tribe_pref(player)
        if self._with_heroes:
            # Assign before the opening shop fill so Millificent/Ysera shape it.
            hero_passives.assign_random_hero(player, patch=self._patch, rng=self._rng)
        self._refresh_shop(player, shop_excluded_race, shared_pool=shared_pool)
        if self._with_heroes:
            hero_passives.apply_hero_on_game_start(
                player,
                round_number,
                patch=self._patch,
                rng=self._rng,
                shared_pool=shared_pool,
                shop_excluded_race=shop_excluded_race,
            )
        return player

    def _random_minion_of_tier(
        self,
        tier: int,
        shop_excluded_race: Optional[Race],
        shared_pool: SharedCardPool,
    ):
        """A fresh ``Minion`` of exactly ``tier``. Prefers shop-eligible cards
        (respecting excluded races); if the exclusion empties the tier it falls
        back to the full tier so a high-mode board is *always* seeded. Returns
        ``None`` only if the patch has no card of that tier at all (unreachable
        in practice). Reserves a pool copy best-effort for shared-pool
        accounting."""
        tpl = self._patch.templates

        def _candidates(respect_exclusion: bool) -> List[str]:
            return [
                cid
                for cid, t in tpl.items()
                if not t.is_token
                and not t.is_golden
                and t.tier == tier
                and (
                    shop_minion_allowed_with_exclusion(t, shop_excluded_race)
                    if respect_exclusion
                    else True
                )
            ]

        cands = _candidates(respect_exclusion=True) or _candidates(respect_exclusion=False)
        if not cands:
            return None
        cid = cands[int(self._rng.integers(0, len(cands)))]
        if shared_pool is not None:
            shared_pool.acquire_new(cid)  # best-effort; synthetic board start
        return make_minion(cid, patch=self._patch)

    def _fresh_player_high(
        self,
        round_number: int,
        shop_excluded_race: Optional[Race],
        *,
        shared_pool: SharedCardPool,
    ) -> PlayerState:
        """High-mode player: tier 5, gold for ``round_number`` (10 at round 8),
        and a board seeded with one random tier-5 and one random tier-6 minion."""
        ruleset = self._patch.meta.ruleset
        tier = bglike_actions.HIGH_MODE_START_TIER
        player = PlayerState(
            health=ruleset.starting_health,
            hero_damage_taken_total=0,
            gold=ruleset.gold_for_round(round_number),
            tavern_tier=tier,
            ruleset=ruleset,
            board=[],
            shop=[None for _ in range(bglike_actions.MAX_SHOP_SLOTS)],
            hand=[None for _ in range(bglike_actions.HAND_SIZE)],
            phase=PlayerPhase.SHOP,
            shop_actions_used=0,
            pending_choice=None,
            placed_minion_board_index=None,
            placed_minion_pending_after=None,
        )
        self._assign_tribe_pref(player)
        if self._with_heroes:
            hero_passives.assign_random_hero(player, patch=self._patch, rng=self._rng)
        for seed_tier in (5, 6):
            m = self._random_minion_of_tier(seed_tier, shop_excluded_race, shared_pool)
            if m is not None:
                player.board.append(m)
        self._refresh_shop(player, shop_excluded_race, shared_pool=shared_pool)
        if self._with_heroes:
            hero_passives.apply_hero_on_game_start(
                player,
                round_number,
                patch=self._patch,
                rng=self._rng,
                shared_pool=shared_pool,
                shop_excluded_race=shop_excluded_race,
            )
        return player

    def _copy_state(self, state: BGLikeState) -> BGLikeState:
        new_players = tuple(
            BGLikeGame._copy_player(p) for p in state.players
        )
        return BGLikeState(
            players=new_players,
            alive=state.alive,
            round_number=state.round_number,
            combat_round=state.combat_round,
            full_lobby_cycle_round=state.full_lobby_cycle_round,
            current_player_index=state.current_player_index,
            shop_turn_order=state.shop_turn_order,
            recent_opponents=state.recent_opponents,
            eliminated=state.eliminated,
            pairings=state.pairings,
            initiative_player=state.initiative_player,
            winner=state.winner,
            done=state.done,
            shop_excluded_race=state.shop_excluded_race,
            shared_pool=state.shared_pool.copy() if state.shared_pool else None,
            patch_build=state.patch_build,
        )

    @staticmethod
    def _copy_player(p: PlayerState) -> PlayerState:
        # Single implementation lives next to the dataclass, so the field
        # list cannot drift out of step with it again.
        return copy_player_state(p)

