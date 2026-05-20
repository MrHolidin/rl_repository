"""Greedy heuristic bot driven by structural minion/board value (no card_id hacks)."""

from __future__ import annotations

from copy import copy
from typing import Optional

import numpy as np

from ..action_map import (
    A_DISCOVER_BASE,
    A_FINISH,
    A_LEVEL_UP,
    A_PLACE_BASE,
    A_ROLL,
    buy_slot,
    is_buy,
    is_discover_pick,
    is_magnet,
    is_sell,
    magnet_hand_board,
    place_slot,
    sell_pos,
)
from ..actions import BOARD_SIZE, HAND_SIZE, MAX_SHOP_SLOTS, MAX_TIER
from src.bg_catalog.cards import make_minion
from ..env import MiniBGEnv
from ..game import MiniBGGame
from ..state import PendingChoiceKind, PlayerState

from .bots import (
    HeuristicBot,
    _mandatory_place_actions,
    _mask,
    _me,
    _prefer_growth_then_finish,
    _replacement_sell_actions,
)
from .common import choose_final_order, legal_env_indices, masked_finish
from .value_model import (
    adapt_choice_score,
    board_power,
    dominant_race,
    minion_shop_value,
    order_key_structured,
    roll_threshold_adjusted,
    rounds_left_estimate,
    triple_cluster_keep_bonus_board,
    triple_progress_buy_bonus,
    triple_progress_place_bonus,
)


class StructuredHeuristicBot(HeuristicBot):
    """Scores buys/sells/magnet/placement/discover from ``value_model``; positions via triggers/keywords."""

    name = "structured"

    def __init__(self, seed: Optional[int] = None) -> None:
        self._rng = np.random.default_rng(seed)

    def _finish(self, env: MiniBGEnv) -> int:
        mask = _mask(env)
        p = _me(env)
        for i in range(3):
            if bool(mask[A_DISCOVER_BASE + i]):
                return A_DISCOVER_BASE + i
        swap = choose_final_order(p.board, mask, order_key_structured)
        if swap != A_FINISH and bool(mask[swap]):
            return swap
        for i in range(HAND_SIZE):
            if bool(mask[A_PLACE_BASE + i]):
                return A_PLACE_BASE + i
        return masked_finish(mask)

    def _ctx(self, env: MiniBGEnv) -> tuple[int, PlayerState]:
        p = _me(env)
        rl = rounds_left_estimate(env.state.round_number)
        return rl, p

    def _pick_discover(self, env: MiniBGEnv, mask: np.ndarray, p: PlayerState) -> int:
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
                m = make_minion(tok)
                dom = dominant_race(p.board)
                bl = len(p.board) + 1
                sc = minion_shop_value(
                    m,
                    rounds_left=rl,
                    dominant=dom,
                    board_len=bl,
                    round_number=rn,
                    tavern_tier_cap=cap,
                )
            else:
                sc = adapt_choice_score(tok)
            if sc > best_s:
                best_s = sc
                best_a = a
        assert best_a >= 0
        return int(best_a)

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
            counts: dict = {}
            for m in p.board:
                if m.race is not None:
                    counts[m.race] = counts.get(m.race, 0) + 1
            if hm.race is not None:
                counts[hm.race] = counts.get(hm.race, 0) + 1
            dominant = max(counts, key=counts.get) if counts else None
            bl = len(p.board) + 1
            sc = minion_shop_value(
                hm,
                rounds_left=rl,
                dominant=dominant,
                board_len=bl,
                round_number=rn,
                tavern_tier_cap=cap,
            ) + triple_progress_place_bonus(p, hm.card_id, slot)
            if sc > best_sc:
                best_sc = sc
                best_a = a
        return int(best_a)

    def _magnet_delta(self, p: PlayerState, rl: int, rn: int, env_action: int) -> float:
        hi, bi = magnet_hand_board(env_action)
        hm = p.hand[hi]
        bm = p.board[bi]
        if hm is None or bm is None:
            return -1e18
        dom = dominant_race(p.board)
        bl = len(p.board)
        cap = p.tavern_tier
        v_before = minion_shop_value(
            bm, rounds_left=rl, dominant=dom, board_len=bl, round_number=rn, tavern_tier_cap=cap
        )
        v_hand = minion_shop_value(
            hm, rounds_left=rl, dominant=dom, board_len=bl, round_number=rn, tavern_tier_cap=cap
        )
        t = copy(bm)
        mg = copy(hm)
        from src.bg_recruitment.place import merge_magnetic_inplace

        merge_magnetic_inplace(t, mg)
        v_after = minion_shop_value(
            t, rounds_left=rl, dominant=dom, board_len=bl, round_number=rn, tavern_tier_cap=cap
        )
        return v_after - v_before - v_hand

    def _buy_score(self, p: PlayerState, rl: int, rn: int, slot: int) -> float:
        m = p.shop[slot]
        if m is None:
            return -1e18
        bl = len(p.board) + (1 if len(p.board) < BOARD_SIZE else 0)
        return minion_shop_value(
            m,
            rounds_left=rl,
            dominant=dominant_race(p.board),
            board_len=max(bl, 1),
            round_number=rn,
            tavern_tier_cap=p.tavern_tier,
        ) + triple_progress_buy_bonus(p, m.card_id)

    def _should_level_up(self, env: MiniBGEnv, p: PlayerState, rl: int) -> bool:
        if p.tavern_tier >= MAX_TIER:
            return False
        if p.gold < p.next_tier_up_cost:
            return False
        rn = env.state.round_number
        opp = env.state.players[1 - env.current_player()].board
        mine_bp = board_power(p.board, rounds_left=rl, round_number=rn)
        opp_bp = board_power(opp, rounds_left=rl, round_number=rn)
        gold_left = p.gold - p.next_tier_up_cost
        cost = p.next_tier_up_cost

        opp_safe = max(opp_bp, 1e-6)
        ratio = mine_bp / opp_safe

        if rn <= 10 and ratio < 0.62:
            return False

        if ratio >= 1.10 and gold_left >= 1:
            return True

        if cost <= 4 and rn >= 5 and ratio >= 0.70 and gold_left >= 2:
            return True

        if rn >= 7 and ratio >= 0.85 and gold_left >= 3:
            return True

        if rn >= 11 and ratio >= 0.76 and gold_left >= 2:
            return True

        if gold_left >= 4 and ratio >= 0.72:
            return True

        return False

    def _choose_action(self, env: MiniBGEnv) -> int:
        mask = _mask(env)
        p = _me(env)
        legal = legal_env_indices(mask)
        disc = [a for a in legal if is_discover_pick(a)]
        if disc:
            return self._pick_discover(env, mask, p)

        rl, _ = self._ctx(env)
        rn = env.state.round_number

        places = _mandatory_place_actions(mask, p)
        if places:
            return self._pick_place(mask, p, rl, rn)

        magnets = [a for a in legal if is_magnet(a)]
        if magnets:
            ranked = sorted(
                magnets,
                key=lambda a: self._magnet_delta(p, rl, rn, a),
                reverse=True,
            )
            best_d = self._magnet_delta(p, rl, rn, ranked[0])
            if best_d >= -1.15:
                return int(ranked[0])

        if bool(mask[A_LEVEL_UP]) and self._should_level_up(env, p, rl):
            return A_LEVEL_UP

        tier = p.tavern_tier
        ok_sell = set(_replacement_sell_actions(mask, p, tier_up_style=True))

        mine_bp = board_power(p.board, rounds_left=rl, round_number=rn)
        opp_bp = board_power(
            env.state.players[1 - env.current_player()].board,
            rounds_left=rl,
            round_number=rn,
        )

        def max_shop_offer_score() -> float:
            mx = -1e18
            for s in range(MAX_SHOP_SLOTS):
                if p.shop[s] is None:
                    continue
                mx = max(mx, self._buy_score(p, rl, rn, s))
            return mx

        thr = roll_threshold_adjusted(
            tavern_tier=p.tavern_tier,
            round_number=rn,
            mine_board_power=mine_bp,
            opp_board_power=opp_bp,
        )

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
                sc = self._buy_score(p, rl, rn, s)
                any_pairs.append((sc, a))
                if m.tier == tier:
                    tier_pairs.append((sc, a))
            if tier_pairs:
                return int(max(tier_pairs)[1])
            if any_pairs:
                return int(max(any_pairs)[1])
            return None

        best_buy = pick_best_buy()

        if len(p.board) < BOARD_SIZE:
            if best_buy is not None:
                return int(best_buy)
            if (
                bool(mask[A_ROLL])
                and A_ROLL in legal
                and max_shop_offer_score() < thr
                and p.shop_actions_used < 19
            ):
                return A_ROLL
            pool = [a for a in legal if not is_sell(a) or a in ok_sell]
            if not pool:
                return self._finish(env)
            return _prefer_growth_then_finish(mask, p, pool, self._rng)

        if best_buy is not None:
            return int(best_buy)

        sell_legal = [a for a in legal if is_sell(a) and a in ok_sell]
        if sell_legal:
            dom = dominant_race(p.board)
            bl = len(p.board)

            def sell_key(a: int) -> float:
                bi = sell_pos(a)
                m = p.board[bi]
                return minion_shop_value(
                    m,
                    rounds_left=rl,
                    dominant=dom,
                    board_len=bl,
                    round_number=rn,
                    tavern_tier_cap=p.tavern_tier,
                ) + triple_cluster_keep_bonus_board(p, bi)

            return int(min(sell_legal, key=sell_key))

        if (
            bool(mask[A_ROLL])
            and A_ROLL in legal
            and max_shop_offer_score() < thr
            and p.shop_actions_used < 19
        ):
            return A_ROLL

        pool = [a for a in legal if not is_sell(a) or a in ok_sell]
        if not pool:
            return self._finish(env)
        return _prefer_growth_then_finish(mask, p, pool, self._rng)
