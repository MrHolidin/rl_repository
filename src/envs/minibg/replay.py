"""Compact JSONL replays for MiniBG (state snapshots per env step)."""

from __future__ import annotations

import json
from dataclasses import asdict
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, TextIO, Union

from .effects import Ability, Effect, Keyword
from .state import Minion, MiniBGState, PlayerState


def _keyword_names(kw: frozenset[Keyword]) -> List[str]:
    return sorted(k.name for k in kw)


def _jsonify_for_replay(x: Any) -> Any:
    """Recursively make values JSON-serializable (Enums → name; containers preserved as list/dict)."""
    if isinstance(x, Enum):
        return x.name
    if isinstance(x, dict):
        return {str(k): _jsonify_for_replay(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonify_for_replay(v) for v in x]
    return x


def _effect_dict(eff: Effect) -> Dict[str, Any]:
    d = asdict(eff)
    return _jsonify_for_replay(d)  # nested tribe tuples / filter_race etc.


def _ability_dict(ab: Ability) -> Dict[str, Any]:
    race = None
    if ab.filter_race is not None:
        race = ab.filter_race.name
    return {
        "trigger": ab.trigger.name,
        "effect_type": type(ab.effect).__name__,
        "effect": _effect_dict(ab.effect),
        **({"filter_race": race} if race is not None else {}),
    }


def minion_to_dict(m: Minion) -> Dict[str, Any]:
    race_name = None if m.race is None else m.race.name
    return {
        "card_id": m.card_id,
        "name": m.name,
        "dbf_id": m.dbf_id,
        "atk": m.raw_attack,
        "hp": m.max_health,
        "tier": m.tier,
        "race": race_name,
        "kw": _keyword_names(m.keywords),
        "granted_kw": _keyword_names(m.granted_keywords),
        "shield": m.has_shield,
        "token": m.is_token,
        "golden": m.is_golden,
        "from_triple_merge": m.from_triple_merge,
        "abilities": [_ability_dict(a) for a in m.abilities],
    }


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
        "shop": [
            None if x is None else minion_to_dict(x) for x in p.shop
        ],
        "hand": [
            None if x is None else minion_to_dict(x) for x in p.hand
        ],
    }


def state_to_dict(state: MiniBGState) -> Dict[str, Any]:
    return {
        "round": state.round_number,
        "cur": state.current_player_index,
        "init": state.initiative_player,
        "done": state.done,
        "winner": state.winner,
        "shop_excluded_race": (
            None if state.shop_excluded_race is None else state.shop_excluded_race.name
        ),
        "p0": player_to_dict(state.players[0]),
        "p1": player_to_dict(state.players[1]),
    }


class ReplayJsonlSink:
    """Append-only JSONL; first line is header, then frames and optional episode breaks."""

    def __init__(self, path: Union[str, Path], header: Dict[str, Any]) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fp: TextIO = self.path.open("w", encoding="utf-8")
        self._fp.write(json.dumps({"type": "header", **header}, separators=(",", ":")) + "\n")
        self._frame_i = 0

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
        acting_idx: int,
        action: int,
        illegal: bool,
        state: MiniBGState,
        info: Dict[str, Any],
    ) -> None:
        self._frame_i += 1
        rec: Dict[str, Any] = {
            "type": "frame",
            "ep": episode,
            "i": frame,
            "p": acting_idx,
            "a": action,
            "illegal": illegal,
            "state": state_to_dict(state),
            "info": {k: info[k] for k in ("winner", "termination_reason", "invalid_action", "battle_damage_shaping") if k in info},
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


__all__ = [
    "ReplayJsonlSink",
    "state_to_dict",
    "minion_to_dict",
    "player_to_dict",
]
