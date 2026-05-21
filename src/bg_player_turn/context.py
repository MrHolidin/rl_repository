"""Context for one player's recruitment turn (shop phase)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from src.bg_core.minion import Race
from src.bg_recruitment.shop_triggers import ShopTriggers
from src.bg_lobby.shared_pool import SharedCardPool


@dataclass
class PlayerTurnContext:
    rng: np.random.Generator
    triggers: ShopTriggers
    shop_excluded_race: Optional[Race] = None
    shared_pool: Optional[SharedCardPool] = None


__all__ = ["PlayerTurnContext"]
