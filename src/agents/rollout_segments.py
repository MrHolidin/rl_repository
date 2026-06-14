"""Multi-seat rollout segment logic for PPO with interleaved learner trajectories.

The rollout buffer mixes transitions from multiple lobby seats (controlled by
the same policy). Each seat is its own RL trajectory: actions of one seat do
not appear in another seat's return. ``compute_gae_advantages`` reflects this
by running GAE per seat — bootstrapping from the next same-seat value, treating
``dones[t]=True`` as a terminal (set by ``close_rollout_segment`` when the seat
is eliminated/finished).

For single-seat callers (uniform ``seat_ids``, e.g. all ``-1`` from MiniBG 2p /
Connect4 / Othello) the function reduces to the standard linear GAE.
"""

from __future__ import annotations

from typing import Any, Dict, List, Protocol, Tuple, runtime_checkable

import numpy as np


@runtime_checkable
class SegmentRolloutBuffer(Protocol):
    rewards: List[float]
    dones: List[bool]
    obs: List[np.ndarray]
    seat_ids: List[int]
    # ``next_obs`` (per-step list) is optional: the flat PPO buffer keeps it; the
    # structured buffer retains only a single ``last_next_obs``. The terminal
    # next_obs written below is never read anyway (``dones[idx]`` is set True →
    # GAE skips its bootstrap), so we only write it when the list exists.


def acting_seat_from_info(info: Any) -> int:
    if isinstance(info, dict):
        acting = info.get("acting_seat")
        if acting is not None:
            return int(acting)
    return -1


def close_rollout_segment(
    buf: SegmentRolloutBuffer,
    seat: int,
    terminal_reward: float,
    placement: "int | None" = None,
) -> bool:
    seat = int(seat)
    next_obs_list = getattr(buf, "next_obs", None)
    labels = getattr(buf, "placement_label", None)
    for idx in range(len(buf.seat_ids) - 1, -1, -1):
        if buf.seat_ids[idx] == seat:
            buf.rewards[idx] = float(terminal_reward)
            buf.dones[idx] = True
            if next_obs_list is not None and idx < len(next_obs_list):
                next_obs_list[idx] = np.asarray(buf.obs[idx], dtype=np.float32)
            if labels is not None and placement is not None:
                # Stamp the final placement on EVERY row of this seat's segment
                # (distributional critic CE target). Walk back over this seat's
                # rows until the previous segment's terminal row (dones=True
                # before the one just set) or an already-labeled row.
                p = int(placement)
                j = idx
                while j >= 0:
                    if buf.seat_ids[j] != seat:
                        j -= 1
                        continue
                    if labels[j] >= 1 or (j < idx and buf.dones[j]):
                        break
                    labels[j] = p
                    j -= 1
            return True
    return False


def seat_ids_array(seat_ids: List[int], n: int) -> np.ndarray:
    if seat_ids:
        return np.array(seat_ids, dtype=np.int64)
    return np.full(n, -1, dtype=np.int64)


def compute_gae_advantages(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    seat_ids: np.ndarray,
    *,
    discount_factor: float,
    gae_lambda: float,
    last_next_value: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Per-seat GAE: each seat is an independent trajectory interleaved with others.

    Within each seat's index list (preserving buffer order):
      - ``dones[t]=True`` → terminal: bootstrap=0, GAE accumulator resets.
      - Else if the seat has a later in-buffer step → bootstrap from
        ``values[t_next_same_seat]``.
      - Else (seat's last in-buffer step is also the buffer's last and not done)
        → bootstrap from ``last_next_value``.
      - Other seats whose last in-buffer step is mid-trajectory: zero bootstrap.
        In practice the collector closes every active seat at game end, so this
        branch is dead; if reached, the resulting bias is bounded by one step.

    For uniform ``seat_ids`` (e.g. all ``-1``) reduces to standard linear GAE
    bit-identically.
    """
    n = int(len(rewards))
    advantages = np.zeros(n, dtype=np.float32)
    if n == 0:
        return advantages, advantages.copy()

    seat_to_idxs: Dict[int, List[int]] = {}
    for t in range(n):
        seat_to_idxs.setdefault(int(seat_ids[t]), []).append(t)

    last_idx = n - 1
    last_done = bool(dones[last_idx])

    for _seat, idxs in seat_to_idxs.items():
        gae = 0.0
        for j in reversed(range(len(idxs))):
            t = idxs[j]
            if bool(dones[t]):
                next_value = 0.0
                cont = 0.0
            elif j < len(idxs) - 1:
                next_value = float(values[idxs[j + 1]])
                cont = 1.0
            elif t == last_idx and not last_done:
                next_value = float(last_next_value)
                cont = 1.0
            else:
                next_value = 0.0
                cont = 1.0
            delta = float(rewards[t]) + discount_factor * next_value * cont - float(values[t])
            gae = delta + discount_factor * gae_lambda * cont * gae
            advantages[t] = gae

    returns = advantages + values.astype(np.float32)
    return advantages, returns


def compute_turn_intrinsic_advantages(
    intrinsic_rewards: np.ndarray,
    values_int: np.ndarray,
    complete_turn: np.ndarray,
    seat_ids: np.ndarray,
    *,
    discount_factor: float,
    gae_lambda: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Turn-level, non-episodic GAE for the RND intrinsic stream.

    The intrinsic reward is a property of a whole turn's settled board (credited
    on ``complete_turn`` rows), so the intrinsic MDP runs one node PER TURN —
    ``discount_factor`` / ``gae_lambda`` are therefore per-turn, not per-micro-step.
    Each seat's turn nodes form an independent trajectory (buffer order).

    Returns ``(advantages, returns, is_node)`` all shaped ``(N,)``:
      * ``advantages`` — the turn's intrinsic advantage BROADCAST to every row of
        that turn (every buy/play that built the board shares the credit). Rows
        after a seat's last completed turn (a buffer-edge partial turn) stay 0.
      * ``returns`` — intrinsic value target, meaningful only on node rows.
      * ``is_node`` — True on ``complete_turn`` rows; the intrinsic value head is
        trained (MSE) only there.

    Non-episodic: a seat's final turn bootstraps from its OWN ``V_int`` (a
    continuing-task stand-in for the unobserved next game) instead of 0, so a
    novel board reached just before elimination is not treated as worthless.
    """
    n = int(len(intrinsic_rewards))
    advantages = np.zeros(n, dtype=np.float32)
    returns = np.zeros(n, dtype=np.float32)
    is_node = np.zeros(n, dtype=np.bool_)
    if n == 0:
        return advantages, returns, is_node

    seat_to_idxs: Dict[int, List[int]] = {}
    for t in range(n):
        seat_to_idxs.setdefault(int(seat_ids[t]), []).append(t)

    for _seat, idxs in seat_to_idxs.items():
        # Segment this seat's rows into turns: a turn is the run of rows ending
        # at a ``complete_turn`` node. ``members`` are broadcast targets.
        turns: List[Tuple[int, List[int]]] = []
        members: List[int] = []
        for t in idxs:
            members.append(t)
            if bool(complete_turn[t]):
                turns.append((t, members))
                members = []
        # Trailing ``members`` (a turn cut by the buffer edge) have no node → 0.

        gae = 0.0
        for k in reversed(range(len(turns))):
            node, member_rows = turns[k]
            if k < len(turns) - 1:
                next_value = float(values_int[turns[k + 1][0]])
            else:
                next_value = float(values_int[node])  # non-episodic self-bootstrap
            delta = (
                float(intrinsic_rewards[node])
                + discount_factor * next_value
                - float(values_int[node])
            )
            gae = delta + discount_factor * gae_lambda * gae
            for r in member_rows:
                advantages[r] = gae
            returns[node] = gae + float(values_int[node])
            is_node[node] = True

    return advantages, returns, is_node


__all__ = [
    "SegmentRolloutBuffer",
    "acting_seat_from_info",
    "close_rollout_segment",
    "compute_gae_advantages",
    "compute_turn_intrinsic_advantages",
    "seat_ids_array",
]
