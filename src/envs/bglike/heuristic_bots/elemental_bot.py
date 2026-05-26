"""Elemental (Nomi-force) heuristic bot for BGLike patch 74257."""

from __future__ import annotations

from typing import Optional

import numpy as np

from ..action_map import (
    A_BUY_BASE,
    A_DISCOVER_BASE,
    A_FINISH,
    A_LEVEL_UP,
    A_PLACE_BASE,
    A_ROLL,
    A_SELL_BASE,
    buy_slot,
    is_buy,
    is_discover_pick,
    is_magnet,
    is_sell,
    magnet_hand_board,
    place_slot,
    sell_pos,
)
from ..actions import BOARD_SIZE, HAND_SIZE, MAX_SHOP_SLOTS
from src.bg_catalog.cards import make_minion
from src.bg_core.minion import Race
from src.bg_lobby.player import PendingChoiceKind, PlayerState
from src.envs.minibg.heuristic_bots.value_model import (
    adapt_choice_score,
    board_power,
    dominant_race,
    minion_shop_value,
    roll_threshold_adjusted,
    rounds_left_estimate,
    triple_cluster_keep_bonus_board,
    triple_progress_buy_bonus,
    triple_progress_place_bonus,
)

from .bots import (
    HeuristicBot,
    HeuristicEnv,
    _mandatory_place_actions,
    _mask,
    _me,
    _prefer_growth_then_finish,
    _replacement_sell_actions,
)
from .common import (
    legal_env_indices,
    masked_finish,
    pick_rl_apply_action,
)

# BGS card IDs for key elemental pieces (patch 74257)
_NOMI_IDS = frozenset({"BGS_104", "TB_BaconUps_201"})
_LIL_RAG_IDS = frozenset({"BGS_100", "TB_BaconUps_200"})
_PARTY_ELEMENTAL_IDS = frozenset({"BGS_120", "TB_BaconUps_160"})
_MOLTEN_ROCK_IDS = frozenset({"BGS_127", "TB_Baconups_202"})
_CRACKLING_CYCLONE_IDS = frozenset({"BGS_119", "TB_BaconUps_159"})
_ARCANE_ASSISTANT_IDS = frozenset({"BGS_128", "TB_Baconups_203"})
_STASIS_ELEMENTAL_IDS = frozenset({"BGS_122", "TB_BaconUps_161"})
_SELLEMENTAL_IDS = frozenset({"BGS_115", "TB_BaconUps_156"})
_TAVERN_TEMPEST_IDS = frozenset({"BGS_123", "TB_BaconUps_162"})
_GENTLE_DJINNI_IDS = frozenset({"BGS_121", "TB_BaconUps_165"})
_WILDFIRE_ELEMENTAL_IDS = frozenset({"BGS_126", "TB_BaconUps_166"})
_LIEUTENANT_GARR_IDS = frozenset({"BGS_125", "TB_BaconUps_164"})

# "Keeper" card IDs — worth holding long-term; never sell post-Nomi unless desperate
_KEEPER_IDS: frozenset[str] = (
    _NOMI_IDS
    | _LIL_RAG_IDS
    | _CRACKLING_CYCLONE_IDS
    | _PARTY_ELEMENTAL_IDS
    | _GENTLE_DJINNI_IDS
    | _MOLTEN_ROCK_IDS
    | _WILDFIRE_ELEMENTAL_IDS
    | _LIEUTENANT_GARR_IDS
)

# Extra flat bonuses added on top of normal minion_shop_value (pre-Nomi)
_ELEMENTAL_CARD_BONUS: dict[frozenset, float] = {
    _NOMI_IDS: 22.0,
    _LIL_RAG_IDS: 10.0,
    _CRACKLING_CYCLONE_IDS: 6.0,
    _PARTY_ELEMENTAL_IDS: 5.5,
    _ARCANE_ASSISTANT_IDS: 4.0,
    _STASIS_ELEMENTAL_IDS: 3.5,
    _TAVERN_TEMPEST_IDS: 3.5,
    _GENTLE_DJINNI_IDS: 3.0,
    _WILDFIRE_ELEMENTAL_IDS: 3.0,
    _MOLTEN_ROCK_IDS: 3.0,
    _SELLEMENTAL_IDS: 2.5,
}

# Flattened lookup: card_id -> bonus
_CARD_BONUS_LOOKUP: dict[str, float] = {}
for _ids, _bonus in _ELEMENTAL_CARD_BONUS.items():
    for _cid in _ids:
        _CARD_BONUS_LOOKUP[_cid] = _bonus


def _has_nomi(p: PlayerState) -> bool:
    return any(m.card_id in _NOMI_IDS for m in p.board)


def _is_elemental(m) -> bool:
    return m.race == Race.ELEMENTAL


def _is_keeper(m) -> bool:
    """Keeper = worth holding long-term; don't cycle-sell post-Nomi."""
    return m.card_id in _KEEPER_IDS


_KEEPER_HOLD_PREMIUM = 10.0  # how much better the shop must be to displace a keeper

def _cycle_board_value(m, rl: int, rn: int, board_len: int) -> float:
    """Intrinsic value of keeping this minion on the board."""
    base = minion_shop_value(
        m,
        rounds_left=rl,
        dominant=Race.ELEMENTAL,
        board_len=board_len,
        round_number=rn,
    )
    if _is_keeper(m):
        base += _KEEPER_HOLD_PREMIUM
    return base


def _strongest_alive_opponent_board_power(env: HeuristicEnv, seat: int, rl: int, rn: int) -> float:
    best = 0.0
    for o in env.state.alive:
        if o == seat:
            continue
        bp = board_power(env.state.players[o].board, rounds_left=rl, round_number=rn)
        best = max(best, bp)
    return best


class ElementalHeuristicBot(HeuristicBot):
    """Elemental-forcing bot: levels T2 on round 2, T3 on round 5, caps at T5.
    No rolling before T3. After finding Nomi, only buys elementals.
    """

    name = "elemental"

    def __init__(self, seed: Optional[int] = None) -> None:
        self._rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # Scoring helpers
    # ------------------------------------------------------------------

    def _elemental_buy_score(self, p: PlayerState, rl: int, rn: int, slot: int) -> float:
        m = p.shop[slot]
        if m is None:
            return -1e18

        has_nomi = _has_nomi(p)

        # After Nomi: only buy elementals
        if has_nomi and not _is_elemental(m):
            return -1e18

        bl = len(p.board) + (1 if len(p.board) < BOARD_SIZE else 0)
        sc = minion_shop_value(
            m,
            rounds_left=rl,
            dominant=Race.ELEMENTAL,  # always treat elemental as dominant
            board_len=max(bl, 1),
            round_number=rn,
            tavern_tier_cap=p.tavern_tier,
        ) + triple_progress_buy_bonus(p, m.card_id)

        sc += _CARD_BONUS_LOOKUP.get(m.card_id, 0.0)

        # Generic elemental bonus (for cards not in the lookup)
        if _is_elemental(m) and m.card_id not in _CARD_BONUS_LOOKUP:
            sc += 2.5

        return sc

    # ------------------------------------------------------------------
    # Leveling logic
    # ------------------------------------------------------------------

    def _should_level_up(self, env: HeuristicEnv, p: PlayerState, rl: int) -> bool:
        if not bool(_mask(env)[A_LEVEL_UP]):
            return False
        if p.gold < p.next_tier_up_cost:
            return False

        tier = p.tavern_tier
        rn = env.state.round_number

        # Never upgrade to T6
        if tier >= 5:
            return False

        # Always upgrade T1→T2 on round 2 (first time we can afford it)
        if tier == 1 and rn >= 2:
            return True

        # Always upgrade T2→T3 starting round 5 (when we have 7 gold)
        if tier == 2 and rn >= 5:
            return True

        # T3→T4 and T4→T5: aggressively push toward T5 for Nomi
        if tier >= 3:
            gold_left = p.gold - p.next_tier_up_cost
            seat = env.current_player()
            mine_bp = board_power(p.board, rounds_left=rl, round_number=rn)
            opp_bp = _strongest_alive_opponent_board_power(env, seat, rl, rn)
            opp_safe = max(opp_bp, 1e-6)
            ratio = mine_bp / opp_safe
            # Level whenever we can afford it and leave at least 2 gold for buys
            if gold_left >= 2:
                return True
            # If behind on power, still level if we have any gold left (T5 is the win condition)
            if gold_left >= 0 and rn >= 7:
                return True
            # Even desperate level-up if really late
            if gold_left >= 0 and ratio < 0.65 and rn >= 9:
                return False  # too far behind, don't waste gold on leveling if dying
            if gold_left >= 0 and rn >= 9:
                return True

        return False

    # ------------------------------------------------------------------
    # Discover pick
    # ------------------------------------------------------------------

    def _pick_discover(self, env: HeuristicEnv, mask: np.ndarray, p: PlayerState) -> int:
        pc = p.pending_choice
        assert pc is not None
        rl = rounds_left_estimate(env.state.round_number)
        rn = env.state.round_number
        cap = p.tavern_tier
        best_a = -1
        best_s = -1e18
        for i in range(3):
            a = A_DISCOVER_BASE + i
            if not bool(mask[a]):
                continue
            tok = pc.options[i]
            if pc.kind in (
                PendingChoiceKind.DISCOVER_MURLOC,
                PendingChoiceKind.TRIPLE_REWARD_DISCOVER,
            ):
                patch = env._game._patch if hasattr(env, "_game") else env.patch
                m = make_minion(tok, patch=patch)
                bl = len(p.board) + 1
                sc = minion_shop_value(
                    m,
                    rounds_left=rl,
                    dominant=Race.ELEMENTAL,
                    board_len=bl,
                    round_number=rn,
                    tavern_tier_cap=cap,
                ) + _CARD_BONUS_LOOKUP.get(m.card_id, 0.0)
            else:
                sc = adapt_choice_score(tok)
            if sc > best_s:
                best_s = sc
                best_a = a
        assert best_a >= 0
        return int(best_a)

    # ------------------------------------------------------------------
    # Place pick
    # ------------------------------------------------------------------

    def _pick_place(self, mask: np.ndarray, p: PlayerState, rl: int, rn: int) -> int:
        actions = _mandatory_place_actions(mask, p)
        assert actions
        cap = p.tavern_tier
        best_a = actions[0]
        best_sc = -1e18
        for a in actions:
            slot = place_slot(a)
            hm = p.hand[slot]
            assert hm is not None
            bl = len(p.board) + 1
            sc = minion_shop_value(
                hm,
                rounds_left=rl,
                dominant=Race.ELEMENTAL,
                board_len=bl,
                round_number=rn,
                tavern_tier_cap=cap,
            ) + triple_progress_place_bonus(p, hm.card_id, slot) + _CARD_BONUS_LOOKUP.get(hm.card_id, 0.0)
            if sc > best_sc:
                best_sc = sc
                best_a = a
        return int(best_a)

    # ------------------------------------------------------------------
    # Magnet delta
    # ------------------------------------------------------------------

    def _magnet_delta(self, p: PlayerState, rl: int, rn: int, env_action: int) -> float:
        from copy import copy
        hi, bi = magnet_hand_board(env_action)
        hm = p.hand[hi]
        bm = p.board[bi]
        if hm is None or bm is None:
            return -1e18
        bl = len(p.board)
        cap = p.tavern_tier
        v_before = minion_shop_value(
            bm, rounds_left=rl, dominant=Race.ELEMENTAL, board_len=bl, round_number=rn, tavern_tier_cap=cap
        )
        v_hand = minion_shop_value(
            hm, rounds_left=rl, dominant=Race.ELEMENTAL, board_len=bl, round_number=rn, tavern_tier_cap=cap
        )
        t = copy(bm)
        mg = copy(hm)
        from src.bg_recruitment.place import merge_magnetic_inplace
        merge_magnetic_inplace(t, mg)
        v_after = minion_shop_value(
            t, rounds_left=rl, dominant=Race.ELEMENTAL, board_len=bl, round_number=rn, tavern_tier_cap=cap
        )
        return v_after - v_before - v_hand

    # ------------------------------------------------------------------
    # Finish fallback
    # ------------------------------------------------------------------

    def _finish(self, env: HeuristicEnv) -> int:
        mask = _mask(env)
        p = _me(env)
        if p.pending_choice is not None and p.pending_choice.kind == PendingChoiceKind.TRANSFORM_SHOP_MINION:
            buys = [A_BUY_BASE + s for s in range(MAX_SHOP_SLOTS) if bool(mask[A_BUY_BASE + s])]
            if buys:
                return int(buys[0])
        for i in range(3):
            if bool(mask[A_DISCOVER_BASE + i]):
                return A_DISCOVER_BASE + i
        for i in range(HAND_SIZE):
            if i < len(p.hand) and bool(mask[A_PLACE_BASE + i]):
                return A_PLACE_BASE + i
        return masked_finish(mask)

    # ------------------------------------------------------------------
    # Main decision
    # ------------------------------------------------------------------

    def _choose_action(self, env: HeuristicEnv) -> int:
        mask = _mask(env)
        p = _me(env)

        rl_apply = pick_rl_apply_action(env, mask)
        if rl_apply is not None:
            return rl_apply

        legal = legal_env_indices(mask)
        disc = [a for a in legal if is_discover_pick(a)]
        if disc:
            return self._pick_discover(env, mask, p)

        rl = rounds_left_estimate(env.state.round_number)
        rn = env.state.round_number

        places = _mandatory_place_actions(mask, p)
        if places:
            return self._pick_place(mask, p, rl, rn)

        magnets = [a for a in legal if is_magnet(a)]
        if magnets:
            ranked = sorted(magnets, key=lambda a: self._magnet_delta(p, rl, rn, a), reverse=True)
            if self._magnet_delta(p, rl, rn, ranked[0]) >= -1.15:
                return int(ranked[0])

        if bool(mask[A_LEVEL_UP]) and self._should_level_up(env, p, rl):
            return A_LEVEL_UP

        tier = p.tavern_tier
        ok_sell = set(_replacement_sell_actions(mask, p, tier_up_style=True))

        def pick_best_buy() -> Optional[int]:
            tier_pairs: list[tuple[float, int]] = []
            any_pairs: list[tuple[float, int]] = []
            for a in legal:
                if not is_buy(a):
                    continue
                s = buy_slot(a)
                m = p.shop[s]
                if m is None:
                    continue
                sc = self._elemental_buy_score(p, rl, rn, s)
                if sc <= -1e17:
                    continue
                any_pairs.append((sc, a))
                if m.tier == tier:
                    tier_pairs.append((sc, a))
            if tier_pairs:
                return int(max(tier_pairs)[1])
            if any_pairs:
                return int(max(any_pairs)[1])
            return None

        def max_shop_offer_score() -> float:
            mx = -1e18
            for s in range(MAX_SHOP_SLOTS):
                if p.shop[s] is None:
                    continue
                mx = max(mx, self._elemental_buy_score(p, rl, rn, s))
            return mx

        has_nomi = _has_nomi(p)

        # ---------------------------------------------------------------
        # POST-NOMI: cycle elementals through the board to stack Nomi buffs
        # Priority:
        #   1. Buy any elemental if we can afford it and have board space
        #   2. If board full: sell cheapest non-keeper elemental, then buy
        #   3. Roll if shop has no elementals left
        # ---------------------------------------------------------------
        if has_nomi:
            best_buy = pick_best_buy()
            board_full = len(p.board) >= BOARD_SIZE

            if best_buy is not None and not board_full:
                return int(best_buy)

            if board_full:
                # Rank board minions by how much we want to keep them
                sell_candidates = [
                    (A_SELL_BASE + i, _cycle_board_value(p.board[i], rl, rn, len(p.board)))
                    for i in range(len(p.board))
                    if bool(mask[A_SELL_BASE + i])
                ]
                if sell_candidates and best_buy is not None:
                    worst_action, worst_val = min(sell_candidates, key=lambda x: x[1])
                    best_shop_score = max_shop_offer_score()
                    # Sell if the shop elemental is worth more than keeping the worst board minion
                    # (the keeper premium is already baked into worst_val)
                    if best_shop_score > worst_val:
                        return int(worst_action)

                # Board full but nothing worth selling for — still buy if a slot opens
                if best_buy is not None:
                    return int(best_buy)

            # Roll to find more elementals
            if bool(mask[A_ROLL]) and A_ROLL in legal and p.shop_actions_used < 19:
                if p.free_roll_charges > 0 or p.next_roll_cost_override == 0 or p.gold >= 1:
                    return A_ROLL

            return self._finish(env)

        # No rolling before T3
        def _should_roll() -> bool:
            if not bool(mask[A_ROLL]) or A_ROLL not in legal:
                return False
            if p.tavern_tier < 3:
                return False
            if p.shop_actions_used >= 19:
                return False
            if p.free_roll_charges > 0 or p.next_roll_cost_override == 0:
                return True
            seat = env.current_player()
            mine_bp = board_power(p.board, rounds_left=rl, round_number=rn)
            opp_bp = _strongest_alive_opponent_board_power(env, seat, rl, rn)
            thr = roll_threshold_adjusted(
                tavern_tier=p.tavern_tier,
                round_number=rn,
                mine_board_power=mine_bp,
                opp_board_power=opp_bp,
            )
            return max_shop_offer_score() < thr

        best_buy = pick_best_buy()

        if len(p.board) < BOARD_SIZE:
            if best_buy is not None:
                return int(best_buy)
            if _should_roll():
                return A_ROLL
            pool = [a for a in legal if not is_sell(a) or a in ok_sell]
            if not pool:
                return self._finish(env)
            return _prefer_growth_then_finish(mask, p, pool, self._rng)

        if best_buy is not None:
            return int(best_buy)

        sell_legal = [a for a in legal if is_sell(a) and a in ok_sell]
        if sell_legal:
            bl = len(p.board)
            def sell_key(a: int) -> float:
                bi = sell_pos(a)
                m = p.board[bi]
                return minion_shop_value(
                    m,
                    rounds_left=rl,
                    dominant=Race.ELEMENTAL,
                    board_len=bl,
                    round_number=rn,
                    tavern_tier_cap=p.tavern_tier,
                ) + triple_cluster_keep_bonus_board(p, bi)
            return int(min(sell_legal, key=sell_key))

        if _should_roll():
            return A_ROLL

        pool = [a for a in legal if not is_sell(a) or a in ok_sell]
        if not pool:
            return self._finish(env)
        return _prefer_growth_then_finish(mask, p, pool, self._rng)


__all__ = ["ElementalHeuristicBot"]
