"""AgentPerspectiveEnv factory for 8-player BGLike lobbies.

Each learner seat is an independent decision segment (shared weights, no merged
credit assignment). Segment boundaries: learner elimination or switch to another
learner seat. League outcomes are recorded once per lobby at episode end.
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Sequence, Tuple

from src.agents.base_agent import BaseAgent
from src.envs.base import StepResult
from src.envs.bglike.action_map import NUM_ENV_ACTIONS
from src.envs.bglike.actions import NUM_PLAYERS
from src.envs.bglike.board_shaping import parse_minions_shaping
from src.envs.bglike.obs import OBS_DIM
from src.envs.bglike.lobby_env import BGLobbyMultiCurrentEnv, make_bglike_training_env
from src.envs.bglike.placement import placement_reward
from src.envs.bglike.tribe_pref import pref_reward_for_counts, pref_stack_reward, pref_value
from src.training.selfplay.game_record import (
    GameRecord,
    game_record_for_lobby_end,
)
from src.envs.reward_config import RewardConfig
from src.training.agent_perspective_env import AgentPerspectiveEnv, ShapingFn
from src.training.opponent_sampler import OpponentSampler


def make_bglike_shaping_fn(scale: float) -> Optional[ShapingFn]:
    """Shaping from ``info['battle_signed_seat']`` (normalized health lost last combat)."""
    if not scale:
        return None

    def _fn(info: Dict[str, Any], _agent_token: int) -> float:
        v = info.get("battle_signed_seat")
        if v is None:
            return 0.0
        return float(scale) * float(v)

    return _fn


def read_opponent_slot_by_seat(opponent_sampler: Any) -> Dict[int, int]:
    """Return opponent league slot ids keyed by lobby seat (distributed + pool samplers)."""
    for attr in ("_slot_by_seat", "_episode_slot_by_seat"):
        mapping = getattr(opponent_sampler, attr, None)
        if isinstance(mapping, dict) and mapping:
            return {int(k): int(v) for k, v in mapping.items()}
    return {}


def submit_game_records_to_sampler(
    opponent_sampler: Any,
    records: Sequence[GameRecord],
) -> None:
    if not records or opponent_sampler is None:
        return
    pool = getattr(opponent_sampler, "opponent_pool", None)
    if pool is not None and hasattr(pool, "submit"):
        for record in records:
            pool.submit(record)  # type: ignore[operator]


def _assert_minion_names_in_patch(base_env: Any, bonuses: Dict[str, float]) -> None:
    """Fail loudly on a display name the patch does not contain.

    A misspelled name pays nothing and reads exactly like "the shaping did
    nothing", which is the hypothesis the run is testing. Same rule as
    ``parse_tier_milestones``: an unusable spec is an error, not a silent zero.
    """
    # The lobby is built lazily on reset, so at construction time the patch is
    # only reachable through the dir the env was configured with (the loader is
    # lru_cached, so this costs nothing per worker).
    known: set = set()
    try:
        known = {str(t.name) for t in base_env.lobby._game._patch.templates.values()}
    except (AttributeError, AssertionError):
        patch_dir = getattr(base_env, "_patch_dir", None)
        if patch_dir is None:
            return  # test fixtures without a real lobby: nothing to check against
        from src.bg_catalog.patch_context import load_patch_context

        known = {str(t.name) for t in load_patch_context(str(patch_dir)).templates.values()}
    unknown = sorted(n for n in bonuses if n not in known)
    if unknown:
        raise ValueError(
            f"minion_play_shaping names not in this patch: {unknown}. "
            "Use the card's display name, e.g. 'Nomi, Kitchen Nightmare'."
        )


def parse_tier_milestones(raw: Any) -> Dict[int, Tuple[float, int]]:
    """YAML ``{5: {base: 0.2, target_round: 8}, ...}`` -> ``{5: (0.2, 8)}``.

    Fails loudly on an unusable spec: a silently-dropped milestone would look
    exactly like "the shaping did nothing", which is the hypothesis under test.
    """
    if not raw:
        return {}
    out: Dict[int, Tuple[float, int]] = {}
    for tier_raw, spec in dict(raw).items():
        tier = int(tier_raw)
        if not 2 <= tier <= 6:
            raise ValueError(f"tier_milestones: tier must be 2..6, got {tier}")
        if isinstance(spec, dict):
            base = float(spec["base"])
            target = int(spec["target_round"])
        else:
            base, target = float(spec[0]), int(spec[1])
        if target < 1:
            raise ValueError(f"tier_milestones[{tier}]: target_round must be >= 1")
        out[tier] = (base, target)
    return out


class BGLikeAgentPerspectiveEnv(AgentPerspectiveEnv):
    """Reuses ``AgentPerspectiveEnv`` drain/placement; per-seat trajectory segments."""

    def __init__(
        self,
        base_env: BGLobbyMultiCurrentEnv,
        opponent_sampler: OpponentSampler,
        *,
        num_current_seats: Optional[int] = None,
        rng: Optional[random.Random] = None,
        shaping_fn: Optional[ShapingFn] = None,
        reward_config: Optional[RewardConfig] = None,
        percent_high_game: float = 0.0,
        tier_milestones: Optional[Dict[int, Tuple[float, int]]] = None,
        tier_milestone_decay: float = 0.8,
        elemental_shop_bonus_shaping: float = 0.0,
        elemental_shop_bonus_cap: int = 10,
        minion_play_shaping: Optional[Any] = None,
        tribe_pref_shaping: float = 0.0,
        tribe_pref_board_shaping: float = 0.0,
    ) -> None:
        super().__init__(
            base_env,
            opponent_sampler,
            agent_first_probability=0.5,
            rng=rng,
            shaping_fn=shaping_fn,
            reward_config=reward_config,
        )
        self._bg_base = base_env
        self._num_current = num_current_seats
        self._learner: Optional[BaseAgent] = None
        self._opponent_slot_by_seat: Dict[int, int] = {}
        self._lobby_league_recorded: bool = False
        # Curriculum: fraction of games started in high mode. The decision lives
        # here in the training harness (with this env's RNG), not in the game.
        self._percent_high_game = max(0.0, min(1.0, float(percent_high_game)))
        # Tier-milestone shaping: {tier: (base, target_round)}. Paid ONCE per seat
        # per lobby, the first time that seat is observed at >= tier, scaled by
        # ``decay ** rounds_late``. Deliberately NOT potential-based — there is no
        # terminal correction, so it does bias the optimum toward a faster curve.
        # That is the point (training wheels); validate by zeroing it afterwards
        # and re-measuring the curve.
        self._tier_milestones: Dict[int, Tuple[float, int]] = dict(tier_milestones or {})
        self._tier_milestone_decay = float(tier_milestone_decay)
        self._tier_paid: Dict[int, set] = {}
        # Nomi shaping: pay per point the acting seat adds to its own tavern
        # elemental bonus, up to ``cap`` points a lobby. Nomi only ever fires for
        # the player who played the elemental, so watching that seat's own
        # counter is already "the model's own action did this". Not
        # potential-based either — same caveat as the tier milestones above.
        self._elem_bonus_coef = float(elemental_shop_bonus_shaping)
        self._elem_bonus_cap = int(elemental_shop_bonus_cap)
        self._elem_bonus_seen: Dict[int, int] = {}
        self._elem_bonus_paid: Dict[int, int] = {}
        # Play shaping: {minion display name: bonus}, paid once per copy the
        # acting seat puts on its board. Not potential-based — same caveat as
        # the two above.
        self._play_shaping: Dict[str, float] = parse_minions_shaping(
            minion_play_shaping
        )
        self._play_paid: Dict[int, Dict[str, int]] = {}
        # Tribe-preference shaping: coef * v[tribe] per bought minion, where v
        # is the seat's own vector drawn by the game at start. Not
        # potential-based either — same caveat as the terms above.
        self._tribe_pref_coef = float(tribe_pref_shaping or 0.0)
        self._tribe_pref_seen: Dict[int, Dict[Any, int]] = {}
        # Per-round board form of the same preference: paid for what stands at
        # the end of every round, not for the act of buying.
        self._tribe_pref_board_coef = float(tribe_pref_board_shaping or 0.0)
        self._tribe_pref_round_paid: Dict[int, int] = {}
        if self._play_shaping:
            _assert_minion_names_in_patch(base_env, self._play_shaping)

    @property
    def supports_seat_segments(self) -> bool:
        return bool(getattr(self._bg_base, "uses_seat_segments", False))

    def set_learner_agent(self, agent: BaseAgent) -> None:
        self._learner = agent

    def set_high_mode(self, flag: bool) -> None:
        """Forward an explicit per-game high-mode decision to the lobby."""
        self._bg_base.set_high_mode(flag)

    def reset(self):
        for _ in range(self.MAX_RESET_RETRIES):
            episode_seed = self._rng.randrange(self._SEED_SPACE)
            # Roll this game's high-mode flag (trainer-side decision, this env's
            # RNG) and push it to the lobby before it builds the initial state.
            if self._percent_high_game > 0.0:
                self._bg_base.set_high_mode(
                    self._rng.random() < self._percent_high_game
                )
            n = self._num_current or len(self._bg_base.current_seats)
            n = max(1, min(int(n), NUM_PLAYERS))
            self._bg_base.set_current_seats(
                tuple(sorted(self._rng.sample(range(NUM_PLAYERS), n)))
            )

            self.opponent_sampler.prepare(self._episode_index)
            if self._learner is None:
                raise RuntimeError("set_learner_agent() required before reset")
            current = set(self._bg_base.current_seats)
            opponent_seats = [s for s in range(NUM_PLAYERS) if s not in current]
            opponents_by_seat = self.opponent_sampler.sample_for_seats(opponent_seats)
            self._opponent_slot_by_seat = read_opponent_slot_by_seat(self.opponent_sampler)
            self._bg_base._opponent_slot_by_seat = dict(self._opponent_slot_by_seat)
            self._bg_base.set_agents(self._learner, opponents_by_seat)
            seen_env: set[int] = set()
            for opp in opponents_by_seat.values():
                oid = id(opp)
                if oid in seen_env:
                    continue
                seen_env.add(oid)
                if hasattr(opp, "set_env"):
                    opp.set_env(self._bg_base)
                if hasattr(opp, "epsilon"):
                    setattr(opp, "epsilon", 0.0)

            obs = self._bg_base.reset(seed=episode_seed)
            if self._bg_base.done:
                self._episode_index += 1
                continue

            self._agent_token = 1
            self._done = False
            self._lobby_league_recorded = False
            return obs
        raise RuntimeError(
            "BGLikeAgentPerspectiveEnv: could not obtain a non-terminal initial state."
        )

    def apply_pending_segment_closures(self, info: Dict[str, Any]) -> None:
        """Close learner segments after ``observe()`` (all seats, including acting)."""
        closures = info.get("segment_closures") or ()
        if not closures:
            return
        learner = self._learner
        if learner is None:
            return
        close = getattr(learner, "close_segment", None)
        if close is None:
            return
        for item in closures:
            seat = int(item["seat"])
            rew = float(item["placement_reward"])
            place = item.get("placement")
            if not close(seat, rew, placement=place):
                raise AssertionError(
                    "segment_closures: seat "
                    f"{seat} has no prior rollout step to close "
                    f"(placement={item.get('placement')}, placement_reward={rew}). "
                    "Current-seat shop turns must go through the learner act/observe path."
                )

    def finish_lobby_to_end(self) -> Dict[str, Any]:
        """Auto-play opponents until lobby completes; apply pending segment closures."""
        info = self._bg_base.finish_lobby_to_end()
        self.apply_pending_segment_closures(info)
        self._done = bool(self._bg_base.done)
        return info

    def step(self, action: int) -> StepResult:
        if self._done:
            raise RuntimeError("Episode is done; call reset() first.")

        base_step = self._bg_base.step(action)
        info = dict(base_step.info) if isinstance(base_step.info, dict) else {}

        lobby_done = bool(self._bg_base.done)
        if lobby_done:
            reward = self._final_reward_for_agent(info)
        else:
            reward = self._reward_in_agent_perspective(base_step, agent_acted=True)
        if lobby_done:
            self._done = True

        return StepResult(
            obs=base_step.obs,
            reward=reward,
            terminated=lobby_done,
            truncated=False,
            info=info,
        )

    def step_structured(
        self,
        action,
        *,
        board_perm=None,
    ) -> StepResult:
        if self._done:
            raise RuntimeError("Episode is done; call reset() first.")

        base_step = self._bg_base.step_structured(action, board_perm=board_perm)
        info = dict(base_step.info) if isinstance(base_step.info, dict) else {}

        # On combat-resolution steps the env now carries per-seat snapshots of
        # the just-resolved combat (own/opp boards in their final pre-combat
        # order + signed uncapped damage + attack_first). The PPO agent reads
        # this in observe() and back-fills the last FINISH-row for each seat
        # with the battle-prediction head's target. Computed for *all* training
        # seats (alive at start of combat), so we don't miss any battles.
        if info.get("combat_advanced"):
            info["battle_data_per_seat"] = self._collect_battle_data_per_seat()

        lobby_done = bool(self._bg_base.done)
        if lobby_done:
            reward = self._final_reward_for_agent(info)
        else:
            reward = self._reward_in_agent_perspective(base_step, agent_acted=True)
        if lobby_done:
            self._done = True

        return StepResult(
            obs=base_step.obs,
            reward=reward,
            terminated=lobby_done,
            truncated=False,
            info=info,
        )

    def _collect_battle_data_per_seat(self) -> Dict[int, Dict[str, Any]]:
        """Snapshot per-seat battle data on a combat-resolution step.

        Returns a dict from training-seat to dict with keys
        ``own_board_obs``, ``opp_board_obs`` (np.ndarray (BOARD_SIZE, SLOT_DIM)),
        ``attack_first`` (float 0/1), ``damage_signed_uncapped`` (float).
        """
        from src.envs.bglike.board_strength import board_strength
        from src.envs.bglike.obs import encode_board_minions

        out: Dict[int, Dict[str, Any]] = {}
        state = self._bg_base.state
        lobby = self._bg_base.lobby
        patch = lobby._game._patch
        card_id_to_dense = patch.card_id_to_dense
        for seat in self._bg_base.current_seats:
            player = state.players[seat]
            if not player.last_battle_snapshots:
                continue
            snap = player.last_battle_snapshots[0]
            out[int(seat)] = {
                "own_board_obs": encode_board_minions(
                    snap.own_board, card_id_to_dense=card_id_to_dense
                ),
                "opp_board_obs": encode_board_minions(
                    snap.opp_board, card_id_to_dense=card_id_to_dense
                ),
                "attack_first": float(player.last_attack_first),
                "damage_signed_uncapped": float(player.last_battle_raw_signed),
                # Strength of the board that just fought. The relative-strength
                # head turns a sequence of these into ratios; taking it from the
                # same pre-combat snapshot means one consistent measurement
                # point per round and no extra plumbing through the engine.
                "board_strength": board_strength(snap.own_board),
            }
        return out

    def _final_reward_for_agent(self, info: Dict[str, Any]) -> float:
        if isinstance(info, dict) and info.get("placement_reward") is not None:
            return float(info["placement_reward"])
        return super()._final_reward_for_agent(info)

    def _battle_shaping_for_acting_seat(self, info: Dict[str, Any]) -> float:
        if self.shaping_fn is None:
            return 0.0
        seat = info.get("acting_seat")
        if seat is None:
            return 0.0
        signed = -self._bg_base.lobby.last_battle_signed(int(seat))
        return float(
            self.shaping_fn({"battle_signed_seat": signed}, self._agent_token)
        )

    def _tier_milestone_reward(self, info: Dict[str, Any]) -> float:
        """One-off, time-decayed bonus the first time the acting seat holds a tier.

        Attribution follows the same rule as the battle shaping: the reward lands
        on ``info['acting_seat']``, whose segment is the one being stepped. Paying
        on ``tier >= T`` rather than on the LEVEL_UP action itself keeps it robust
        to a seat reaching the tier by any route.
        """
        # getattr: an env built without __init__ (test fixtures) or predating this
        # feature must read as "shaping disabled", not raise.
        milestones = getattr(self, "_tier_milestones", None)
        if not milestones:
            return 0.0
        seat = info.get("acting_seat")
        if seat is None:
            return 0.0
        seat = int(seat)
        state = getattr(self._bg_base, "state", None)
        if state is None:
            return 0.0
        try:
            tier = int(state.players[seat].tavern_tier)
        except (IndexError, AttributeError):
            return 0.0
        rnd = int(getattr(state, "round_number", 0))
        paid = self._tier_paid.setdefault(seat, set())
        total = 0.0
        for milestone_tier, (base, target_round) in milestones.items():
            if milestone_tier in paid or tier < milestone_tier:
                continue
            paid.add(milestone_tier)
            late = max(0, rnd - int(target_round))
            total += float(base) * (getattr(self, "_tier_milestone_decay", 0.8) ** late)
        return total

    def _elemental_bonus_reward(self, info: Dict[str, Any]) -> float:
        """Pay for each point the acting seat adds to its tavern's elemental buff.

        Nomi raises this counter only for the player who played the Elemental, so
        the seat's own counter is the attribution. Credit is capped per lobby, and
        the cap is on what has been *paid*, not on the counter — the engine leaves
        the buff itself unbounded.
        """
        # getattr for the same reason as the tier milestones: an env built
        # without __init__ (test fixtures) reads as "shaping disabled", not raise.
        coef = float(getattr(self, "_elem_bonus_coef", 0.0) or 0.0)
        if not coef:
            return 0.0
        seat = info.get("acting_seat")
        if seat is None:
            return 0.0
        seat = int(seat)
        state = getattr(self._bg_base, "state", None)
        if state is None:
            return 0.0
        try:
            current = int(state.players[seat].shop_elemental_bonus)
        except (IndexError, AttributeError):
            return 0.0
        seen = self._elem_bonus_seen
        gained = current - seen.get(seat, 0)
        seen[seat] = current
        if gained <= 0:
            return 0.0
        paid = self._elem_bonus_paid.get(seat, 0)
        creditable = min(gained, max(0, int(self._elem_bonus_cap) - paid))
        if creditable <= 0:
            return 0.0
        self._elem_bonus_paid[seat] = paid + creditable
        return coef * float(creditable)

    def _minion_play_reward(self, info: Dict[str, Any]) -> float:
        """Pay once for each copy of a configured minion the acting seat PLAYS.

        Board only: buying is not the step that matters. Paying on ownership
        instead bought a card that then sat in hand — 23 of 26 seat-games at 3M
        held Nomi without ever putting it down, and a Nomi in hand triggers
        nothing. Counting board copies rather than intercepting the PLAY action
        keeps the credit robust to the route (played, tripled, summoned), the
        same rule the tier milestones use. The paid count is a high-water mark
        per seat, so selling and replaying the same minion cannot farm it.
        """
        # getattr for the same reason as the other two shaping terms: an env
        # built without __init__ (test fixtures) reads as "shaping disabled".
        bonuses = getattr(self, "_play_shaping", None)
        if not bonuses:
            return 0.0
        seat = info.get("acting_seat")
        if seat is None:
            return 0.0
        seat = int(seat)
        state = getattr(self._bg_base, "state", None)
        if state is None:
            return 0.0
        try:
            player = state.players[seat]
        except (IndexError, AttributeError):
            return 0.0
        owned: Dict[str, int] = {}
        for card in player.board:
            name = getattr(card, "name", None)
            if name in bonuses:
                owned[name] = owned.get(name, 0) + 1
        if not owned:
            return 0.0
        paid = self._play_paid.setdefault(seat, {})
        total = 0.0
        for name, count in owned.items():
            new_copies = count - paid.get(name, 0)
            if new_copies <= 0:
                continue
            paid[name] = count
            total += float(bonuses[name]) * new_copies
        return total

    def _tribe_pref_reward(self, info: Dict[str, Any]) -> float:
        """Pay ``coef * v[tribe]`` for every minion the acting seat has bought.

        Reads the engine's cumulative per-tribe purchase counter and pays on the
        delta, so a purchase is credited exactly once however the card is used
        afterwards — played, tripled, or sold the same turn. Half the vector is
        negative by construction, so this is a preference, not a buy-more bonus.
        """
        # getattr for the same reason as the other shaping terms: an env built
        # without __init__ (test fixtures) reads as "shaping disabled".
        coef = float(getattr(self, "_tribe_pref_coef", 0.0) or 0.0)
        if not coef:
            return 0.0
        seat = info.get("acting_seat")
        if seat is None:
            return 0.0
        seat = int(seat)
        state = getattr(self._bg_base, "state", None)
        if state is None:
            return 0.0
        try:
            player = state.players[seat]
        except (IndexError, AttributeError):
            return 0.0
        pref = getattr(player, "tribe_pref", ()) or ()
        if not pref:
            return 0.0
        counts = dict(getattr(player, "bought_tribe_counts", {}) or {})
        seen = self._tribe_pref_seen.setdefault(seat, {})
        delta = {
            race: n - seen.get(race, 0)
            for race, n in counts.items()
            if n - seen.get(race, 0) > 0
        }
        if not delta:
            return 0.0
        self._tribe_pref_seen[seat] = counts
        return coef * pref_reward_for_counts(pref, delta)

    def _tribe_pref_board_reward(self, info: Dict[str, Any]) -> float:
        """Pay per round for the tribe composition standing at the round's end.

        Score is ``sum_x min(5, n_x) ** 1.5 * v[x]`` over tribes, so the reward
        is superlinear in how concentrated the board is: three of one tribe pay
        5.20 where three separate tribes pay 3.00. The per-purchase form paid
        once and then stopped caring (a seat could collect and immediately
        sell); this pays for what is actually kept, every round it is kept.

        Accrual is keyed on (seat, round) and checked on EVERY step, not on
        ``combat_advanced``: a seat does not necessarily act on the step that
        carries the combat flag, and binding to it paid only 4/6/3/1 of a
        15-round lobby's rounds — a quarter of the intended coefficient, with
        the rest silently skipped. Paying on the seat's first step of a new
        round instead credits exactly the board it finished the previous round
        with (combat does not alter the board) and fires once per round.

        The seat's last round is not paid: the lobby ends or the seat is
        eliminated before it acts again. One round in ~19.
        """
        coef = float(getattr(self, "_tribe_pref_board_coef", 0.0) or 0.0)
        if not coef:
            return 0.0
        seat = info.get("acting_seat")
        if seat is None:
            return 0.0
        seat = int(seat)
        state = getattr(self._bg_base, "state", None)
        if state is None:
            return 0.0
        rnd = int(getattr(state, "round_number", 0))
        paid = getattr(self, "_tribe_pref_round_paid", None)
        if paid is None:
            return 0.0
        if paid.get(seat) == rnd:
            return 0.0
        paid[seat] = rnd
        try:
            player = state.players[seat]
        except (IndexError, AttributeError):
            return 0.0
        pref = getattr(player, "tribe_pref", ()) or ()
        if not pref:
            return 0.0
        counts: Dict[Any, int] = {}
        for minion in player.board:
            if minion is None:
                continue
            race = getattr(minion, "race", None)
            counts[race] = counts.get(race, 0) + 1
        return coef * pref_stack_reward(pref, counts)

    def _reward_in_agent_perspective(self, step, agent_acted: bool) -> float:
        info = step.info if isinstance(step.info, dict) else {}
        reward = self._tier_milestone_reward(info)
        reward += self._elemental_bonus_reward(info)
        reward += self._minion_play_reward(info)
        reward += self._tribe_pref_reward(info)
        reward += self._tribe_pref_board_reward(info)
        if info.get("combat_advanced"):
            reward += self._battle_shaping_for_acting_seat(info)
        return reward

    def notify_episode_end(self, info: Dict[str, Any]) -> None:
        # Reliable per-lobby boundary for the learner: in bglike the learner
        # never sees a transition with terminated/truncated=True (it finishes
        # via segment closures / lobby end), so any per-episode agent state must
        # be reset here, not on a transition flag. DvD uses this to hand out a
        # fresh, collision-free seat→identity assignment each lobby.
        self._tier_paid = {}
        self._elem_bonus_seen = {}
        self._elem_bonus_paid = {}
        self._play_paid = {}
        self._tribe_pref_seen = {}
        self._tribe_pref_round_paid = {}
        learner = self._learner
        if learner is not None:
            hook = getattr(learner, "on_episode_boundary", None)
            if hook is not None:
                hook()

        if self.opponent_sampler is None:
            self._episode_index += 1
            return

        placements = info.get("placements_current") or {}
        self.opponent_sampler.on_episode_end(
            self._episode_index,
            {
                "agent_token": self._agent_token,
                "placements_current": placements,
                "info": info,
                "skip_league_record": True,
            },
        )
        self._episode_index += 1


def make_bglike_agent_perspective_env(
    opponent_sampler: OpponentSampler,
    *,
    current_seats: Optional[Sequence[int]] = None,
    num_current_seats: Optional[int] = None,
    seed: Optional[int] = None,
    shaping_fn: Optional[ShapingFn] = None,
    reward_config: Optional[RewardConfig] = None,
    rng: Optional[random.Random] = None,
    percent_high_game: float = 0.0,
    tier_milestones: Optional[Dict[Any, Any]] = None,
    tier_milestone_decay: float = 0.8,
    elemental_shop_bonus_shaping: float = 0.0,
    elemental_shop_bonus_cap: int = 10,
    minion_play_shaping: Optional[Any] = None,
    tribe_pref_shaping: float = 0.0,
    tribe_pref_board_shaping: float = 0.0,
    **lobby_kwargs: Any,
) -> BGLikeAgentPerspectiveEnv:
    # ``percent_high_game`` and the tier-milestone knobs are consumed by the
    # perspective wrapper (trainer-side shaping/curriculum), never forwarded to
    # the inner lobby/game constructors.
    base = make_bglike_training_env(
        current_seats=current_seats or (0,),
        seed=seed,
        reward_config=reward_config,
        **lobby_kwargs,
    )
    return BGLikeAgentPerspectiveEnv(
        base,
        opponent_sampler,
        num_current_seats=num_current_seats,
        rng=rng,
        shaping_fn=shaping_fn,
        reward_config=reward_config,
        percent_high_game=percent_high_game,
        tier_milestones=parse_tier_milestones(tier_milestones),
        tier_milestone_decay=tier_milestone_decay,
        elemental_shop_bonus_shaping=elemental_shop_bonus_shaping,
        elemental_shop_bonus_cap=elemental_shop_bonus_cap,
        minion_play_shaping=minion_play_shaping,
        tribe_pref_shaping=tribe_pref_shaping,
        tribe_pref_board_shaping=tribe_pref_board_shaping,
    )


__all__ = [
    "BGLikeAgentPerspectiveEnv",
    "apply_bglike_segment_closures_after_observe",
    "collect_bglike_lobby_league_outcome",
    "finalize_bglike_lobby_league_record",
    "make_bglike_agent_perspective_env",
    "make_bglike_shaping_fn",
    "read_opponent_slot_by_seat",
    "submit_game_records_to_sampler",
]


def apply_bglike_segment_closures_after_observe(
    env: Any, info: Any
) -> None:
    """After ``observe()``: close learner segments (training only, no league update)."""
    if not isinstance(info, dict):
        return
    apply = getattr(env, "apply_pending_segment_closures", None)
    if apply is not None:
        apply(info)


def finalize_bglike_lobby_league_record(env: Any, info: Any) -> Optional[GameRecord]:
    """Build one lobby-end ``GameRecord`` after the lobby has finished."""
    if not isinstance(info, dict):
        return None
    placements_full = info.get("placements_full") or {}
    slot_by_seat = getattr(env, "_opponent_slot_by_seat", None) or {}
    if not slot_by_seat and hasattr(env, "opponent_sampler"):
        slot_by_seat = read_opponent_slot_by_seat(env.opponent_sampler)
    current_seats = info.get("current_seats")
    if current_seats is None:
        bg = getattr(env, "_bg_base", None)
        current_seats = getattr(bg, "current_seats", ()) if bg is not None else ()
    key_map = getattr(env, "_slot_id_to_scripted_key", None)
    if not key_map and hasattr(env, "opponent_sampler"):
        key_map = getattr(env.opponent_sampler, "_slot_id_to_scripted_key", None)
    record = game_record_for_lobby_end(
        current_seats=current_seats,
        slot_by_seat=slot_by_seat,
        placements_full=placements_full,
        slot_id_to_scripted_key=key_map or {},
    )
    if record is not None and hasattr(env, "opponent_sampler"):
        submit_game_records_to_sampler(env.opponent_sampler, records=[record])
    return record


def collect_bglike_lobby_league_outcome(
    env: Any, last_info: Dict[str, Any]
) -> Tuple[Dict[str, Any], Optional[GameRecord]]:
    """Finish lobby if needed and return one league record for the full lobby."""
    if getattr(env, "_lobby_league_recorded", False):
        return dict(last_info or {}), None
    info = dict(last_info or {})
    if not getattr(env, "done", False):
        finish = getattr(env, "finish_lobby_to_end", None)
        if finish is not None:
            info = finish()
    if not getattr(env, "done", False):
        return info, None
    record = finalize_bglike_lobby_league_record(env, info)
    if record is not None:
        env._lobby_league_recorded = True
    return info, record
