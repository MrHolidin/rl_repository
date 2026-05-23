"""Compact JSONL replays for BGLike (8-player lobby, state snapshot per step)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, TextIO, Union

from src.bg_lobby.player import PlayerState

from src.envs.minibg.replay import minion_to_dict

from .actions import NUM_PLAYERS
from .state import BGLikeState

if TYPE_CHECKING:
    from .lobby_env import BGLobbyEnv, LobbyStepInfo
    from .replay_recorder import LobbyReplayConfig


def _placed_minion_idx_for_replay(p: PlayerState) -> Optional[int]:
    ref = p.placed_minion_pending_after
    if ref is not None and ref in p.board:
        return p.board.index(ref)
    return None


def player_to_dict(p: PlayerState) -> Dict[str, Any]:
    pend = None
    if p.pending_choice is not None:
        pc = p.pending_choice
        pend = {
            "kind": pc.kind.name,
            "options": list(pc.options),
            "extra_after": pc.extra_modals_after,
        }
    return {
        "hp": p.health,
        "hero_dmg_taken": p.hero_damage_taken_total,
        "gold": p.gold,
        "tier": p.tavern_tier,
        "tier_up_cost": p.next_tier_up_cost,
        "phase": p.phase.name,
        "shop_done": p.shopping_finished,
        "shop_acts": p.shop_actions_used,
        "shop_freeze_next_round": p.shop_freeze_next_round,
        "pending": pend,
        "triple_reward_pending": p.triple_reward_discover_pending,
        "placed_idx": _placed_minion_idx_for_replay(p),
        "board": [minion_to_dict(m) for m in p.board],
        "shop": [None if x is None else minion_to_dict(x) for x in p.shop],
        "hand": [None if x is None else minion_to_dict(x) for x in p.hand],
    }


def state_to_dict(state: BGLikeState) -> Dict[str, Any]:
    return {
        "round": state.round_number,
        "combat_round": state.combat_round,
        "cur": state.current_player_index,
        "init": state.initiative_player,
        "done": state.done,
        "winner": state.winner,
        "alive": list(state.alive),
        "eliminated": [
            {
                "seat": snap.seat,
                "combat_round": snap.eliminated_combat_round,
                "tier": snap.tavern_tier,
            }
            for snap in state.eliminated
        ],
        "shop_excluded_race": (
            None if state.shop_excluded_race is None else state.shop_excluded_race.name
        ),
        "players": {str(i): player_to_dict(state.players[i]) for i in range(NUM_PLAYERS)},
    }


class ReplayJsonlSink:
    """Append-only JSONL; first line is header, then frames and optional episode breaks."""

    def __init__(self, path: Union[str, Path], header: Dict[str, Any]) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fp: TextIO = self.path.open("w", encoding="utf-8")
        self._fp.write(json.dumps({"type": "header", **header}, separators=(",", ":")) + "\n")

    def episode_break(self, episode_index: int) -> None:
        if episode_index > 0:
            self._fp.write(
                json.dumps({"type": "episode_break", "episode": episode_index}, separators=(",", ":"))
                + "\n"
            )

    def frame(
        self,
        *,
        episode: int,
        frame: int,
        seat: int,
        action: int,
        auto: bool,
        illegal: bool,
        state: BGLikeState,
        info: Dict[str, Any],
    ) -> None:
        rec: Dict[str, Any] = {
            "type": "frame",
            "ep": episode,
            "i": frame,
            "seat": seat,
            "a": action,
            "auto": auto,
            "illegal": illegal,
            "state": state_to_dict(state),
            "info": info,
        }
        self._fp.write(json.dumps(rec, separators=(",", ":")) + "\n")

    def close(self) -> None:
        if self._fp is not None:
            self._fp.close()
            self._fp = None  # type: ignore[assignment]

    def __enter__(self) -> "ReplayJsonlSink":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


class ReplayHeuristicEnvBridge:
    """Minimal ``BGLobbyMultiCurrentEnv`` stand-in for heuristic ``set_env`` during replay capture."""

    def __init__(self, lobby: BGLobbyEnv) -> None:
        self._lobby = lobby

    @property
    def lobby(self) -> BGLobbyEnv:
        return self._lobby

    @property
    def acting_seat(self) -> None:
        return None

    @property
    def state(self) -> BGLikeState:
        return self._lobby.state


def _attach_recorder(env: BGLobbyEnv, config: LobbyReplayConfig) -> None:
    from .replay_recorder import LobbyReplayRecorder

    if env._replay is not None:
        env._replay.close()
    env._replay = LobbyReplayRecorder(config)


def attach_replay(
    env: BGLobbyEnv,
    path: Union[str, Path],
    header: Dict[str, Any],
    *,
    record_seats: frozenset[int] | None = None,
    sparse: bool = True,
) -> ReplayJsonlSink:
    """Attach JSONL replay capture to a lobby session.

    ``record_seats``: shop frames only for these acting seats (``None`` = all 8).
    Global events (combat, elimination, lobby end) are always recorded.
    ``sparse`` (default): milestone shop frames — FINISH / FINISH_FREEZE_SHOP only.
    """
    from .replay_recorder import LobbyReplayConfig

    config = LobbyReplayConfig(
        path=Path(path),
        header=header,
        record_seats=record_seats,
        sparse=sparse,
    )
    _attach_recorder(env, config)
    assert env._replay is not None
    return env._replay._sink


def attach_replay_config(env: BGLobbyEnv, config: LobbyReplayConfig) -> ReplayJsonlSink:
    """Attach replay from a ``LobbyReplayConfig`` (constructor / programmatic use)."""
    _attach_recorder(env, config)
    assert env._replay is not None
    return env._replay._sink


def close_replay(env: BGLobbyEnv) -> None:
    if env._replay is not None:
        env._replay.close()
        env._replay = None


def lobby_step_info_to_replay_info(
    info: LobbyStepInfo,
    *,
    combat_advanced: bool,
) -> Dict[str, Any]:
    return {
        "eliminated_seats": list(info.eliminated_seats),
        "lobby_done": info.lobby_done,
        "placements": dict(info.placements),
        "combat_advanced": combat_advanced,
    }


__all__ = [
    "ReplayHeuristicEnvBridge",
    "ReplayJsonlSink",
    "attach_replay",
    "attach_replay_config",
    "close_replay",
    "lobby_step_info_to_replay_info",
    "player_to_dict",
    "state_to_dict",
]
