"""Parse league rating / sampler settings from YAML opponent_sampler params."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class LeagueRatingConfig:
    kind: str = "ema"
    ema_beta: float = 0.05
    trueskill: Optional[Dict[str, Any]] = None


@dataclass(frozen=True)
class LeagueSamplerConfig:
    kind: str = "fractional"
    current_self_fraction: float = 0.4
    past_self_fraction: float = 0.4

    def uses_match_quality(self, rating_kind: str) -> bool:
        kind = (self.kind or "fractional").strip().lower()
        if kind == "trueskill_quality":
            return True
        if kind == "pfsp_unified" and rating_kind.strip().lower() == "trueskill":
            return True
        return False

    def is_unified_pool(self) -> bool:
        kind = (self.kind or "fractional").strip().lower()
        return kind in ("pfsp_unified", "trueskill_quality")


@dataclass(frozen=True)
class LeagueSettings:
    rating: LeagueRatingConfig = field(default_factory=LeagueRatingConfig)
    sampler: LeagueSamplerConfig = field(default_factory=LeagueSamplerConfig)


def parse_league_settings(opponent_sampler_params: Dict[str, Any]) -> LeagueSettings:
    """Read ``league:`` block with fallback to legacy ``self_play`` keys."""
    params = dict(opponent_sampler_params or {})
    sp = dict(params.get("self_play") or {})
    league_raw = dict(params.get("league") or {})
    rating_raw = dict(league_raw.get("rating") or {})
    sampler_raw = dict(league_raw.get("sampler") or {})

    rating_kind = str(rating_raw.get("kind") or sp.get("rating") or "ema")
    ema_beta = float(
        rating_raw.get("ema_beta")
        or sp.get("frozen_ema_beta")
        or 0.05
    )
    ts_raw = rating_raw.get("trueskill") or sp.get("trueskill")
    trueskill = dict(ts_raw) if ts_raw else None

    sampler_kind = str(sampler_raw.get("kind") or "fractional")
    if sampler_kind.strip().lower() == "trueskill_quality" and rating_kind.strip().lower() != "trueskill":
        raise ValueError("league.sampler.kind=trueskill_quality requires league.rating.kind=trueskill")
    current_fraction = float(
        sampler_raw.get("current_self_fraction")
        if "current_self_fraction" in sampler_raw
        else sp.get("current_self_fraction", 0.4)
    )
    past_fraction = float(
        sampler_raw.get("past_self_fraction")
        if "past_self_fraction" in sampler_raw
        else sp.get("past_self_fraction", 0.4)
    )

    return LeagueSettings(
        rating=LeagueRatingConfig(
            kind=rating_kind,
            ema_beta=ema_beta,
            trueskill=trueskill,
        ),
        sampler=LeagueSamplerConfig(
            kind=sampler_kind,
            current_self_fraction=current_fraction,
            past_self_fraction=past_fraction,
        ),
    )


__all__ = [
    "LeagueRatingConfig",
    "LeagueSamplerConfig",
    "LeagueSettings",
    "parse_league_settings",
]
