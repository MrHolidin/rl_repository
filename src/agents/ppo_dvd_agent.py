"""PPO agent for diverse-population (DvD) training over a v7 identity-conditioned net.

Step-2 scope: identity *rotation* + observation augmentation only.

Each episode the learner adopts one population identity ``i``; its one-hot is
appended to every observation the v7 network sees — both the rollout
observations (via :meth:`act_structured`, which also feeds opponent forwards
through ``opponent_step``) and the GAE bootstrap observation (via
:meth:`_value_only`). Because the structured rollout buffer and PPO update
carry the observation tensor unchanged, the identity tail rides through the
whole update for free; no buffer/update surgery.

Step-3 adds behavioural repulsion (DvD): each acting step records a board
descriptor for the seat; at segment close the active identity's terminal
(placement) reward gains ``diversity_coef * novelty``, where novelty is the
mean distance of this episode's board descriptor from the *other* identities'
running (EMA) descriptors. This pushes identities apart in build-space while
placement keeps each competent. ``diversity_coef == 0`` recovers step-2
behaviour exactly (no bonus, no EMA effect on reward).

Note: opponents are still sampled from the existing pool (scripted / frozen);
identity *co-play* (sibling opponents) is a later refinement — the repulsion
does not require it, since each identity's EMA accumulates across the episodes
in which it is active.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, List, Optional, Tuple

import numpy as np
import torch

from src.envs.bglike.actions import BOARD_SIZE as _BG_BOARD_SIZE
from src.envs.bglike.board_descriptor import BOARD_DESCRIPTOR_DIM, board_descriptor
from src.envs.minibg.obs import RACE_ONEHOT_DIM, _RACE_ORDER

from .ppo_structured_minibg_agent import MiniBGPPOStructuredAgent


class PPODvDAgent(MiniBGPPOStructuredAgent):
    """Structured PPO agent that conditions a v7 net on a rotating identity."""

    def __init__(
        self,
        *,
        num_identities: Optional[int] = None,
        diversity_coef: float = 0.0,
        diversity_ema: float = 0.1,
        identity_seed: int = 0,
        identity_tribes: Optional[List[Any]] = None,
        identity_init_std: float = 0.0,
        diversity_reward_mode: str = "final",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        # Break identity symmetry at the start. v7 zero-inits the identity
        # projections (so a fresh net == v6), but that means all identities are
        # behaviourally identical at step 0 and the gradient to differentiate
        # them is tiny + normalization-bounded → they never separate (placement,
        # which is identity-agnostic, learns fine; the identity-dependent tribe
        # bonus does not). A non-zero init makes identities act differently from
        # the first step, giving the bonus real behavioural variance to reinforce.
        if identity_init_std and identity_init_std > 0.0:
            import torch as _t
            for _name in ("identity_proj", "identity_emb_proj", "identity_slot_gate"):
                proj = getattr(self.policy_net, _name, None)
                if proj is not None:
                    with _t.no_grad():
                        _t.nn.init.normal_(proj.weight, mean=0.0, std=float(identity_init_std))
                        _t.nn.init.zeros_(proj.bias)
        self.identity_init_std = float(identity_init_std)

        # Prefer the network's own width (recovered on checkpoint reload, where
        # the agent-level kwarg is not re-supplied); fall back to the kwarg.
        net_n = getattr(self.policy_net, "num_identities", None)
        if net_n is None:
            if num_identities is None:
                raise ValueError(
                    "PPODvDAgent requires a v7 network exposing num_identities, "
                    "or an explicit num_identities kwarg."
                )
            net_n = int(num_identities)
        self.num_identities = int(net_n)
        self.diversity_coef = float(diversity_coef)
        self.diversity_ema = float(diversity_ema)
        # Diversity reward shape:
        #   "final"       — terminal bonus = coef * (assigned-tribe fraction of the
        #                   final board). One reward per segment, clean, exactly
        #                   "reward the final composition" (the default / requested).
        #   "acquisition" — per-step +coef per own-tribe minion acquired (count
        #                   delta). Kept behind this flag; note it over-counts
        #                   combat/token churn and drowns out placement.
        mode = str(diversity_reward_mode).strip().lower()
        if mode not in ("final", "acquisition", "potential"):
            raise ValueError(
                "diversity_reward_mode must be 'final', 'potential' or "
                f"'acquisition', got {mode!r}"
            )
        self.diversity_reward_mode = mode
        # The descriptor's first RACE_ONEHOT_DIM dims are tribe fractions; we
        # read each identity's assigned-tribe fraction straight out of that block.
        self._tribe_slice = slice(0, RACE_ONEHOT_DIM)

        # --- Assigned-tribe diversity -------------------------------------
        # Each identity is permanently assigned one tribe; its intrinsic bonus
        # is ``coef * (fraction of that tribe on the final board)``. This is a
        # fixed (non-learned) per-identity reward, so symmetry is broken from
        # step 0 (unlike pairwise repulsion, which is zero at collapse) and it
        # parallelises trivially across workers (pure function, no shared
        # learned state — the reason we dropped the DIAYN discriminator).
        # Bonus uses the board *fraction*, not minion count, so an identity is
        # not rewarded for stuffing weak on-tribe minions at the cost of placement.
        self._identity_tribe_idx = self._resolve_identity_tribes(identity_tribes)

        self._id_rng = np.random.default_rng(identity_seed)
        self._active_identity = int(self._id_rng.integers(self.num_identities))
        # Single-identity mode (frozen opponents / explicit pin): when set, every
        # seat this agent drives uses ``_active_identity`` and the internal
        # per-seat assignment is bypassed. Learners leave this False and run in
        # per-seat mode (each current seat gets its own identity → several
        # co-learners share one net, the canonical DvD setup).
        self._identity_externally_set = False
        # Transient per-forward override used by sibling opponents that share
        # this net but must act under a *different* identity. ``None`` → fall
        # back to the seat/active identity. Set/cleared around a single forward;
        # safe because the lobby drain processes seats sequentially.
        self._forced_identity: Optional[int] = None
        # Per-seat identity for learner (per-seat mode). Reset each episode; a
        # fresh shuffled permutation spreads identities across the learner's
        # current seats so distinct seats train distinct styles on-policy.
        self._identity_by_seat: dict = {}
        self._seat_assign_count = 0
        self._episode_identity_perm = self._fresh_identity_perm()
        self._last_acted_seat: Optional[int] = None

        # Per-identity EMA of board descriptors + which identities have data.
        self._phi = np.zeros((self.num_identities, BOARD_DESCRIPTOR_DIM), dtype=np.float32)
        self._phi_seen = np.zeros(self.num_identities, dtype=bool)
        # Per-identity placement (terminal reward) EMA — quality per identity.
        self._placement_ema = np.full(self.num_identities, np.nan, dtype=np.float64)
        # Latest board descriptor per seat (set on each acting step, consumed at
        # segment close). Belongs to the seat's current-episode identity.
        self._desc_by_seat: dict = {}
        # --- Dense potential-based tribe shaping (replaces the terminal-only
        # bonus, which failed: a terminal reward proportional to final tribe
        # fraction is smeared by GAE over all ~90 segment steps, so the specific
        # "buy my tribe" action gets no more credit than a reroll. Even coef=1000
        # did not move behaviour. Here each step is rewarded for the *increase*
        # in own-tribe fraction it caused: F = coef*(Phi(s') - Phi(s)). This
        # telescopes to the same episode total (coef*Phi_final) but credits the
        # exact buy action, and is potential-based ⇒ optimum-preserving.) ---
        self._prev_frac_by_seat: dict = {}   # (legacy, unused by acquisition reward)
        self._last_row_by_seat: dict = {}    # (legacy)
        self._prev_count_by_seat: dict = {}  # seat -> last own-tribe minion count
        # Since-last-emit accumulators (reset each time metrics are emitted).
        self._acc_n = 0
        self._acc_bonus = 0.0       # sum of assigned-tribe FRACTION (commitment)
        self._acc_count = 0.0       # sum of assigned-tribe COUNT (actual reward magnitude)
        self._acc_abs_place = 0.0

    def _resolve_identity_tribes(self, identity_tribes: Optional[List[Any]]) -> np.ndarray:
        """Map each identity → its assigned tribe's index in ``_RACE_ORDER``.

        ``identity_tribes`` is a list of tribe names (e.g. ``["MECHANICAL",
        "ELEMENTAL", "MURLOC", "NONE"]``); ``"NONE"``/``None`` legitimately
        assigns the tribeless / good-stuff niche (index 0). If omitted, tribes
        are auto-assigned by walking ``_RACE_ORDER`` (skipping ALL), so a config
        that only sets ``num_identities`` still gets distinct assignments.
        """
        from src.bg_core.minion import Race

        order = list(_RACE_ORDER)  # index 0 = None, then the tribes, last = ALL
        if identity_tribes is None:
            # Auto: None, BEAST, DEMON, MECH, MURLOC, DRAGON, PIRATE, ELEMENTAL...
            auto = [i for i, r in enumerate(order) if r is not Race.ALL]
            return np.array(
                [auto[k % len(auto)] for k in range(self.num_identities)],
                dtype=np.int64,
            )
        if len(identity_tribes) != self.num_identities:
            raise ValueError(
                f"identity_tribes has {len(identity_tribes)} entries but "
                f"num_identities={self.num_identities}"
            )
        idx = []
        for name in identity_tribes:
            if name is None or str(name).strip().upper() == "NONE":
                idx.append(0)
                continue
            race = Race[str(name).strip().upper()]
            if race not in order:
                raise ValueError(f"identity_tribes: unknown / unsupported tribe {name!r}")
            idx.append(order.index(race))
        return np.array(idx, dtype=np.int64)

    # ------------------------------------------------------------------
    # Observation augmentation: append the active identity one-hot.
    # ------------------------------------------------------------------
    def _fresh_identity_perm(self) -> list:
        """A shuffled identity order to hand out to successive learner seats."""
        perm = list(range(self.num_identities))
        self._id_rng.shuffle(perm)
        return perm

    def _identity_for_seat(self, seat: int) -> int:
        """Identity driving ``seat`` in per-seat (learner) mode.

        Assigned lazily the first time a seat acts this episode, walking a
        shuffled permutation so the learner's current seats get distinct
        identities (wrapping if there are more seats than identities).
        """
        s = int(seat)
        ident = self._identity_by_seat.get(s)
        if ident is None:
            ident = self._episode_identity_perm[
                self._seat_assign_count % self.num_identities
            ]
            self._seat_assign_count += 1
            self._identity_by_seat[s] = int(ident)
        return int(ident)

    def _identity_for_forward(self, seat: Optional[int] = None) -> int:
        # 1) explicit per-forward override (sibling opponents) wins;
        # 2) single-identity mode (frozen / pinned) uses _active_identity;
        # 3) otherwise per-seat learner mode.
        if self._forced_identity is not None:
            return self._forced_identity
        if self._identity_externally_set or seat is None:
            return self._active_identity
        return self._identity_for_seat(seat)

    def _augment_obs_np(self, obs: np.ndarray, seat: Optional[int] = None) -> np.ndarray:
        obs = np.asarray(obs, dtype=np.float32)
        onehot = np.zeros(self.num_identities, dtype=np.float32)
        onehot[self._identity_for_forward(seat)] = 1.0
        return np.concatenate([obs, onehot], axis=-1)

    def _augment_obs_tensor(
        self, obs: torch.Tensor, seat: Optional[int] = None
    ) -> torch.Tensor:
        onehot = torch.zeros(
            (obs.shape[0], self.num_identities), dtype=obs.dtype, device=obs.device
        )
        onehot[:, self._identity_for_forward(seat)] = 1.0
        return torch.cat([obs, onehot], dim=-1)

    # ------------------------------------------------------------------
    # Population co-play hooks (used by the trainer / sibling opponents).
    # ------------------------------------------------------------------
    def set_episode_identity(self, identity: int) -> None:
        """Pin the learner's identity for the whole upcoming episode.

        Called by the trainer before ``reset`` so the learner and its sibling
        opponents agree on who is who. Disables the internal rotate-on-done
        fallback (the trainer now owns identity assignment).
        """
        self._active_identity = int(identity) % self.num_identities
        self._identity_externally_set = True
        self._identity_by_seat = {}
        self._prev_frac_by_seat = {}
        self._last_row_by_seat = {}
        self._prev_count_by_seat = {}

    def _reset_seat_identities(self) -> None:
        """Begin a new episode in per-seat (learner) mode."""
        self._identity_by_seat = {}
        self._seat_assign_count = 0
        self._episode_identity_perm = self._fresh_identity_perm()
        # Dense-shaping tracking is per-lobby: stale row indices from a previous
        # lobby must not be credited (buffer isn't cleared between lobbies).
        self._prev_frac_by_seat = {}
        self._last_row_by_seat = {}
        self._prev_count_by_seat = {}

    @contextmanager
    def _forced_identity_ctx(self, identity: int):
        """Temporarily force the identity used for obs augmentation.

        Used by :class:`SiblingOpponent`, which shares this net but must act as
        a *different* population member. The lobby drain is sequential, so a
        single forward under the override is race-free.
        """
        prev = self._forced_identity
        self._forced_identity = int(identity) % self.num_identities
        try:
            yield
        finally:
            self._forced_identity = prev

    # ------------------------------------------------------------------
    # Entry points that introduce observations to the net / buffer.
    # ------------------------------------------------------------------
    def act_structured(
        self,
        obs: np.ndarray,
        legal_list: List[Any],
        env: Any,
        *,
        deterministic: bool = False,
    ) -> Tuple[Any, Optional[Tuple[int, ...]], int]:
        seat = None
        try:
            seat = int(env.state.current_player_index)
        except Exception:
            seat = None
        # Record this seat's board descriptor; the latest before segment close
        # ≈ the final board. Recorded whenever training (even at coef=0) so the
        # diversity metrics are available for an A/B against the baseline.
        if self.training and seat is not None:
            try:
                self._desc_by_seat[seat] = board_descriptor(env.state, seat)
            except Exception:
                pass
        if seat is not None:
            self._last_acted_seat = seat
        return super().act_structured(
            self._augment_obs_np(obs, seat), legal_list, env, deterministic=deterministic
        )

    def _assigned_frac_for_seat(self, seat: int, desc) -> float:
        i = self._identity_of_seat(seat)
        return float(np.asarray(desc, dtype=np.float64)[self._identity_tribe_idx[i]])

    def _assigned_count_for_seat(self, seat: int, desc) -> float:
        """Number of own-tribe minions on board = tribe_fraction * board_fill.

        The descriptor stores tribe *fractions* and ``fill`` = n / BOARD_SIZE at
        index RACE_ONEHOT_DIM+1, so count = frac * fill * BOARD_SIZE."""
        d = np.asarray(desc, dtype=np.float64)
        i = self._identity_of_seat(seat)
        n = float(d[RACE_ONEHOT_DIM + 1]) * float(_BG_BOARD_SIZE)
        return float(d[self._identity_tribe_idx[i]]) * n

    def on_episode_boundary(self) -> None:
        """Reset per-seat identity assignment for the next lobby.

        Called by the perspective env's ``notify_episode_end`` — the *only*
        reliable per-lobby boundary for a bglike learner (it never sees a
        transition flagged terminated/truncated; it finishes via segment
        closures). Previously this reset was gated on that nonexistent flag, so
        ``_identity_by_seat`` accumulated across lobbies and handed out
        duplicate identities once the seat set changed (the dist_ppo_078 bug:
        4 learner seats collapsing onto 2 identities). In pinned / single-
        identity mode (frozen / sibling opponents) there is nothing to reset.
        """
        if not self._identity_externally_set:
            self._reset_seat_identities()

    def _value_only(self, obs_1: torch.Tensor) -> torch.Tensor:
        # Bootstrap obs is the raw env observation (core width); augment with the
        # identity of the last seat that acted (the buffer's final row), so the
        # baseline is consistent with that row's rollout identity. GAE zeroes the
        # bootstrap across segment boundaries, so only the last segment's seat
        # matters here.
        seat = getattr(self, "_last_acted_seat", None)
        return super()._value_only(self._augment_obs_tensor(obs_1, seat))

    # ------------------------------------------------------------------
    # DvD behavioural repulsion.
    # ------------------------------------------------------------------
    def _identity_of_seat(self, seat: int) -> int:
        """The identity that drove ``seat`` this episode (per-seat or pinned)."""
        if self._identity_externally_set:
            return self._active_identity
        return self._identity_for_seat(seat)

    def _update_identity_ema(self, seat: int):
        """Fold ``seat``'s episode board into its identity EMA (always tracked,
        regardless of diversity_mode, so φ-based metrics stay valid).

        Returns ``(identity, desc)``; ``desc`` is ``None`` if nothing recorded.
        """
        desc = self._desc_by_seat.get(int(seat))
        i = self._identity_of_seat(seat)
        if desc is None:
            return i, None
        if self._phi_seen[i]:
            self._phi[i] = (1.0 - self.diversity_ema) * self._phi[i] + self.diversity_ema * desc
        else:
            self._phi[i] = np.asarray(desc, dtype=self._phi.dtype)
            self._phi_seen[i] = True
        return i, desc

    def _assigned_tribe_bonus(self, i: int, desc: np.ndarray) -> float:
        """Fraction of identity ``i``'s assigned tribe on this episode's board.

        The descriptor's tribe block already stores per-tribe board fractions,
        so this is a direct read: ``desc[assigned_tribe_idx] ∈ [0, 1]``. Fixed
        target (no learned state), non-zero from step 0 even at collapse, and a
        pure function → safe to compute identically on every worker.
        """
        return float(np.asarray(desc, dtype=np.float64)[self._identity_tribe_idx[i]])

    # Back-compat alias: some tests / callers refer to ``_diversity_bonus``.
    def _diversity_bonus(self, seat: int) -> float:
        i, desc = self._update_identity_ema(seat)
        if desc is None:
            return 0.0
        return self._assigned_tribe_bonus(i, desc)

    def close_segment(
        self, seat: int, terminal_reward: float, placement: Optional[int] = None
    ) -> bool:
        if self.training:
            raw = float(terminal_reward)
            # Always track the identity EMA so φ-metrics stay valid.
            i, desc = self._update_identity_ema(seat)

            # Assigned-tribe presence on the final board.
            frac = 0.0 if desc is None else self._assigned_tribe_bonus(i, desc)   # fraction (metric)
            count = 0.0 if desc is None else self._assigned_count_for_seat(seat, desc)  # # of minions

            # "final" mode: terminal diversity reward = coef * (# own-tribe minions
            # on the final board). COUNT (not fraction): one tribe minion is worth
            # coef, not coef/board_size≈coef/7. The fraction form (coef*frac) made
            # a single minion worth only ~0.03 at coef 0.2 → too weak to flip any
            # buy decision (dist_ppo_083: zero causal differentiation, all
            # identities built the same meta despite the gate partially learning).
            # Count gives a per-minion incentive strong enough to move actions,
            # while staying a clean terminal reward for the final composition.
            # "acquisition" mode shapes per-step in observe(); close_segment then
            # carries only placement.
            if self.diversity_reward_mode == "final":
                terminal_reward = raw + self.diversity_coef * count
            elif self.diversity_reward_mode == "potential":
                # Per-step potential shaping (observe) telescopes to coef*count,
                # but the final board can change after the seat's last shop act
                # (a last buy whose post-state was never re-observed). Stamp the
                # residual here so the per-episode sum is EXACTLY coef*final_count
                # — equal to the terminal "final" reward, with per-step credit.
                prev_count = self._prev_count_by_seat.get(seat, 0.0)
                terminal_reward = raw + self.diversity_coef * (count - prev_count)
                self._prev_count_by_seat[seat] = count

            # Per-identity placement EMA (quality), accumulators for emit window.
            if np.isnan(self._placement_ema[i]):
                self._placement_ema[i] = raw
            else:
                self._placement_ema[i] = (
                    (1.0 - self.diversity_ema) * self._placement_ema[i]
                    + self.diversity_ema * raw
                )
            self._acc_n += 1
            self._acc_bonus += frac
            self._acc_count += count
            self._acc_abs_place += abs(raw)
        return super().close_segment(seat, terminal_reward, placement=placement)

    def observe(self, transition: Any, is_augmented: bool = False) -> dict:
        n_before = len(self.rollout_buffer)
        out = super().observe(transition, is_augmented=is_augmented)
        # Optional per-step ACQUISITION reward (behind diversity_reward_mode=
        # "acquisition"): +coef per own-tribe minion acquired (count delta). NOT
        # the default — it over-counts combat/token churn (~36 "acquisitions"/game
        # for a 7-board) and drowns placement (≈91% of the reward signal even at
        # coef 0.2). The default "final" mode rewards the final-board composition
        # terminally in close_segment instead.
        if (
            self.diversity_reward_mode in ("acquisition", "potential")
            and self.training
            and not is_augmented
            and self.diversity_coef != 0.0
            and len(self.rollout_buffer) > n_before
        ):
            seat = getattr(self, "_last_acted_seat", None)
            desc = self._desc_by_seat.get(seat) if seat is not None else None
            if desc is not None:
                cur_count = self._assigned_count_for_seat(seat, desc)
                prev_count = self._prev_count_by_seat.get(seat)
                if prev_count is not None:
                    delta = cur_count - prev_count
                    # "acquisition": +coef per minion ACQUIRED (positive only) —
                    #   over-counts combat/token churn. "potential": SIGNED delta
                    #   (Ng potential shaping, Φ=coef*count). Signed deltas
                    #   telescope to coef*final_count (a token that appears then
                    #   dies nets to 0; a sold minion refunds -coef) → identical
                    #   total to the terminal "final" reward, but credited at the
                    #   step the board composition changes so PPO can follow it.
                    if self.diversity_reward_mode == "acquisition":
                        gained = max(0.0, delta)
                    else:  # potential
                        gained = delta
                    if gained != 0.0:
                        self.rollout_buffer.rewards[-1] += self.diversity_coef * gained
                self._prev_count_by_seat[seat] = cur_count
        return out

    def _tribe_dist(self, a: np.ndarray, b: np.ndarray) -> float:
        """Euclidean distance over the tribe block only (composition distance)."""
        da = np.asarray(a, dtype=np.float64)[self._tribe_slice]
        db = np.asarray(b, dtype=np.float64)[self._tribe_slice]
        return float(np.linalg.norm(da - db))

    def population_diversity(self) -> float:
        """Mean pairwise tribe-composition distance between identities' EMA descriptors.

        Uses the same tribe-only distance as the repulsion bonus so the metric
        measures exactly what the bonus optimises."""
        seen = np.flatnonzero(self._phi_seen)
        if seen.size < 2:
            return 0.0
        dists = [
            self._tribe_dist(self._phi[a], self._phi[b])
            for ia, a in enumerate(seen)
            for b in seen[ia + 1 :]
        ]
        return float(np.mean(dists)) if dists else 0.0

    # ------------------------------------------------------------------
    # Distributed DvD-state transport: workers accumulate φ / placement EMA /
    # novelty during rollout collection, but the *host* runs ``update()`` and
    # emits metrics. The host therefore has no DvD state of its own — workers
    # ship a snapshot, the host merges them before computing ``_dvd_metrics``.
    # ------------------------------------------------------------------
    def dvd_state_snapshot(self) -> dict:
        """Picklable DvD state for shipping a worker's accumulators to the host."""
        return {
            "phi": self._phi.copy(),
            "phi_seen": self._phi_seen.copy(),
            "placement_ema": self._placement_ema.copy(),
            "acc_n": int(self._acc_n),
            "acc_bonus": float(self._acc_bonus),
            "acc_count": float(self._acc_count),
            "acc_abs_place": float(self._acc_abs_place),
        }

    def merge_dvd_state(self, snaps: "list") -> None:
        """Aggregate worker snapshots into this (host) agent's DvD state.

        Per identity: average the workers' EMA descriptors / placement over the
        workers that have data for it (unweighted — workers run equal rollout
        budgets). Window accumulators (novelty / |place| / n) sum across workers
        so the emit-window means are over the whole round.
        """
        if not snaps:
            return
        n_id = self.num_identities
        phi_sum = np.zeros_like(self._phi)
        phi_cnt = np.zeros(n_id, dtype=np.float64)
        pl_sum = np.zeros(n_id, dtype=np.float64)
        pl_cnt = np.zeros(n_id, dtype=np.float64)
        acc_n = 0
        acc_bonus = 0.0
        acc_count = 0.0
        acc_abs = 0.0
        for s in snaps:
            seen = np.asarray(s["phi_seen"], dtype=bool)
            phi_sum[seen] += np.asarray(s["phi"])[seen]
            phi_cnt[seen] += 1.0
            pl = np.asarray(s["placement_ema"], dtype=np.float64)
            pl_valid = ~np.isnan(pl)
            pl_sum[pl_valid] += pl[pl_valid]
            pl_cnt[pl_valid] += 1.0
            acc_n += int(s.get("acc_n", 0))
            acc_bonus += float(s.get("acc_bonus", 0.0))
            acc_count += float(s.get("acc_count", 0.0))
            acc_abs += float(s.get("acc_abs_place", 0.0))
        has_phi = phi_cnt > 0
        self._phi[has_phi] = phi_sum[has_phi] / phi_cnt[has_phi, None]
        self._phi_seen = has_phi
        self._placement_ema = np.where(
            pl_cnt > 0, pl_sum / np.where(pl_cnt > 0, pl_cnt, 1.0), np.nan
        )
        self._acc_n = acc_n
        self._acc_bonus = acc_bonus
        self._acc_count = acc_count
        self._acc_abs_place = acc_abs

    # ------------------------------------------------------------------
    # Metrics: emitted on PPO-update steps (same cadence as the rest).
    # ------------------------------------------------------------------
    def _dvd_metrics(self) -> dict:
        m: dict = {}
        seen_idx = np.flatnonzero(self._phi_seen)
        m["dvd_identity_coverage"] = float(self._phi_seen.mean())
        m["dvd_pop_diversity"] = self.population_diversity()

        doms = []
        for i in seen_idx:
            tribe_block = self._phi[i, :RACE_ONEHOT_DIM]
            if float(tribe_block.sum()) > 0.0:
                doms.append(int(np.argmax(tribe_block)))
        m["dvd_distinct_tribes"] = float(len(set(doms)))

        pl = self._placement_ema[seen_idx]
        pl = pl[~np.isnan(pl)]
        if pl.size:
            m["dvd_placement_best"] = float(pl.max())
            m["dvd_placement_worst"] = float(pl.min())
            m["dvd_placement_spread"] = float(pl.max() - pl.min())

        mean_frac = self._acc_bonus / self._acc_n if self._acc_n else 0.0
        mean_count = self._acc_count / self._acc_n if self._acc_n else 0.0
        mean_abs = self._acc_abs_place / self._acc_n if self._acc_n else 0.0
        # dvd_mean_assigned_frac = mean fraction of the assigned tribe on the
        # board (∈[0,1]) — the "are identities committing" signal. dvd_mean_bonus
        # is the ACTUAL per-episode diversity reward = coef * COUNT (frac*fill*7),
        # NOT coef*frac — the earlier metric undercounted it by the fill*7 factor.
        m["dvd_mean_assigned_frac"] = float(mean_frac)
        m["dvd_mean_bonus"] = float(self.diversity_coef * mean_count)
        m["dvd_bonus_place_ratio"] = float(
            self.diversity_coef * mean_count / (mean_abs + 1e-8)
        )

        proj = getattr(self.policy_net, "identity_proj", None)
        if proj is not None:
            with torch.no_grad():
                W = proj.weight.detach()  # (slot_hidden, num_identities)
                b = proj.bias.detach()  # (slot_hidden,)
                contribs = [
                    float((W[:, i] + b).norm().item())
                    for i in range(self.num_identities)
                ]
            m["dvd_identity_contrib_norm"] = float(np.mean(contribs))

        # Per-identity detail (read which identity built what, and how it places).
        for i in range(min(self.num_identities, 8)):
            if not self._phi_seen[i]:
                continue
            if not np.isnan(self._placement_ema[i]):
                m[f"dvd_place_{i}"] = float(self._placement_ema[i])
            tb = self._phi[i, :RACE_ONEHOT_DIM]
            if float(tb.sum()) > 0.0:
                m[f"dvd_tribe_{i}"] = float(int(np.argmax(tb)))
                # Fraction of this identity's *assigned* tribe on its EMA board:
                # the direct "did identity i commit to its target tribe" readout.
                m[f"dvd_assigned_frac_{i}"] = float(tb[self._identity_tribe_idx[i]])

        # Reset the emit-window accumulators.
        self._acc_n = 0
        self._acc_bonus = 0.0       # sum of assigned-tribe FRACTION (commitment)
        self._acc_count = 0.0       # sum of assigned-tribe COUNT (actual reward magnitude)
        self._acc_abs_place = 0.0
        return m

    def update(self) -> dict:
        metrics = super().update()
        # Only augment on steps where a real PPO update ran (buffer was full).
        if metrics and "policy_loss" in metrics:
            metrics.update(self._dvd_metrics())
        return metrics

    @classmethod
    def load(cls, path: str, *, device=None, **overrides):
        """Load like the structured parent, but consume DvD-only kwargs here.

        ``num_identities`` is recovered from the v7 network; ``diversity_coef`` /
        ``diversity_ema`` / ``identity_seed`` / ``sibling_fraction`` are agent-
        level and not stored in the checkpoint, so accept them as overrides
        (the parent's ``load`` would reject these unknown keys).
        """
        dvd_kw = {
            k: overrides.pop(k)
            for k in (
                "diversity_coef", "diversity_ema", "identity_seed",
                "identity_tribes", "diversity_reward_mode",
            )
            if k in overrides
        }
        overrides.pop("sibling_fraction", None)  # sampler-level, ignore here
        agent = super().load(path, device=device, **overrides)
        if "diversity_coef" in dvd_kw:
            agent.diversity_coef = float(dvd_kw["diversity_coef"])
        if "diversity_ema" in dvd_kw:
            agent.diversity_ema = float(dvd_kw["diversity_ema"])
        if "diversity_reward_mode" in dvd_kw:
            m = str(dvd_kw["diversity_reward_mode"]).strip().lower()
            if m in ("final", "acquisition"):
                agent.diversity_reward_mode = m
        if "identity_tribes" in dvd_kw:
            agent._identity_tribe_idx = agent._resolve_identity_tribes(
                dvd_kw["identity_tribes"]
            )
        return agent


class SiblingOpponent:
    """Opponent that is the *same* live v7 net acting under a fixed identity ``j``.

    The lobby drives structured opponents via ``act_structured`` (see
    ``controller_step.lobby_seat_step``), toggling ``agent.training`` off itself.
    This wrapper exposes the structured interface and forwards to the shared
    base agent under a forced identity, so one network can staff several seats
    with different build styles. The base's ``act_structured`` gates buffer
    writes / descriptor recording behind ``self.training`` (forced off by the
    lobby), so a sibling forward has no rollout side effects.
    """

    def __init__(self, base_agent: "PPODvDAgent", identity: int, *, greedy: bool = True) -> None:
        self._base = base_agent
        self.identity = int(identity) % int(base_agent.num_identities)
        self._greedy = greedy

    def act_structured(
        self,
        obs: Any,
        legal_list: List[Any],
        env: Any,
        *,
        deterministic: bool = False,
    ) -> Tuple[Any, Optional[Tuple[int, ...]], int]:
        with self._base._forced_identity_ctx(self.identity):
            return self._base.act_structured(
                obs, legal_list, env, deterministic=deterministic or self._greedy
            )

    @property
    def training(self) -> bool:
        return self._base.training

    @training.setter
    def training(self, value: bool) -> None:
        # The lobby toggles ``agent.training`` around the opponent forward;
        # delegate to the shared base so its rollout gating stays consistent.
        self._base.training = value

    def act(self, obs, legal_mask=None, deterministic: bool = False) -> int:
        raise RuntimeError("SiblingOpponent drives lobby via act_structured (structured path).")

    def eval(self) -> None:  # pragma: no cover - frozen wrapper
        return None

    def train(self) -> None:  # pragma: no cover
        return None

    def save(self, path: str) -> None:  # pragma: no cover
        return None


__all__ = ["PPODvDAgent", "SiblingOpponent"]
