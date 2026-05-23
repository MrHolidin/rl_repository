"""Lobby-level replay recording (sparse milestones, seat filtering)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Protocol, runtime_checkable

from .action_map import is_finish, is_finish_freeze_shop
from .replay import ReplayJsonlSink, lobby_step_info_to_replay_info
from .state import BGLikeState


@runtime_checkable
class _ReplayStepInfo(Protocol):
    eliminated_seats: tuple[int, ...]
    lobby_done: bool


@dataclass(frozen=True)
class LobbyReplayConfig:
    path: Path
    header: dict
    record_seats: frozenset[int] | None = None  # None = all 8 seats
    sparse: bool = True


class LobbyReplayRecorder:
    """Sink, frame counter, episode breaks, and emit policy for BGLobbyEnv."""

    def __init__(self, config: LobbyReplayConfig) -> None:
        header: Dict[str, Any] = {"format": 1, "game": "bglike", **config.header}
        if config.record_seats is not None:
            header["record_seats"] = sorted(config.record_seats)
        self._config = config
        self._sink = ReplayJsonlSink(config.path, header)
        self._episode = -1
        self._frame = 0

    @property
    def config(self) -> LobbyReplayConfig:
        return self._config

    def on_reset(self) -> None:
        self._episode += 1
        self._sink.episode_break(self._episode)
        self._frame = 0

    def close(self) -> None:
        self._sink.close()

    @staticmethod
    def is_global_event(
        info: _ReplayStepInfo,
        *,
        state: BGLikeState,
        prev_combat_round: int,
    ) -> bool:
        if info.eliminated_seats:
            return True
        if state.combat_round > prev_combat_round:
            return True
        if info.lobby_done:
            return True
        return False

    @staticmethod
    def is_sparse_milestone(action_int: int) -> bool:
        return is_finish(action_int) or is_finish_freeze_shop(action_int)

    def should_emit_frame(
        self,
        acting_seat: int,
        action_int: int,
        info: _ReplayStepInfo,
        *,
        state: BGLikeState,
        prev_combat_round: int,
    ) -> bool:
        if self.is_global_event(info, state=state, prev_combat_round=prev_combat_round):
            return True
        record_seats = self._config.record_seats
        if record_seats is not None and acting_seat not in record_seats:
            return False
        if self._config.sparse and not self.is_sparse_milestone(action_int):
            return False
        return True

    def maybe_record(
        self,
        acting_seat: int,
        action_int: int,
        info: _ReplayStepInfo,
        *,
        state: BGLikeState,
        prev_combat_round: int,
        auto: bool,
        illegal: bool = False,
    ) -> None:
        if not self.should_emit_frame(
            acting_seat,
            action_int,
            info,
            state=state,
            prev_combat_round=prev_combat_round,
        ):
            return
        self._frame += 1
        self._sink.frame(
            episode=self._episode,
            frame=self._frame,
            seat=acting_seat,
            action=int(action_int),
            auto=auto,
            illegal=illegal,
            state=state,
            info=lobby_step_info_to_replay_info(
                info,
                combat_advanced=state.combat_round > prev_combat_round,
            ),
        )


__all__ = [
    "LobbyReplayConfig",
    "LobbyReplayRecorder",
]
