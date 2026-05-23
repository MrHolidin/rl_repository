"""MiniBG-shaped env view for heuristic bots over ``BGLobbyMultiCurrentEnv``."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import numpy as np

if TYPE_CHECKING:
    from src.envs.bglike.lobby_env import BGLobbyMultiCurrentEnv


class BGLikeHeuristicEnvView:
    def __init__(self, lobby: BGLobbyMultiCurrentEnv) -> None:
        self._multi = lobby
        self._mask_override: Optional[np.ndarray] = None

    @property
    def state(self):
        return self._multi.state

    def _inner(self):
        return self._multi.lobby

    def _seat(self) -> int:
        inner = self._inner()
        ctrl = inner._heuristic_control_seat
        if ctrl is not None:
            return int(ctrl)
        acting = self._multi.acting_seat
        if acting is not None:
            return int(acting)
        return int(inner.current_seat())

    def current_player(self) -> int:
        return self._seat()

    def set_mask_override(self, mask: Optional[np.ndarray]) -> None:
        self._mask_override = None if mask is None else np.asarray(mask, dtype=bool)

    @property
    def legal_actions_mask(self) -> np.ndarray:
        if self._mask_override is not None:
            return self._mask_override
        return self._inner().legal_mask_for_seat(self._seat())

    @property
    def patch(self):
        return self._inner()._game._patch

    @property
    def rl_pending(self):
        return self._inner().rl_pending_for_seat(self._seat())


__all__ = ["BGLikeHeuristicEnvView"]
