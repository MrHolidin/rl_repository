"""Placement EMA tracker for BGLike league."""

from __future__ import annotations

from src.training.selfplay.placement_ema import PlacementEmaTracker


def test_placement_ema_window():
    tracker = PlacementEmaTracker(window=20)
    for place in [8, 7, 6]:
        tracker.record(0, place)
    assert tracker.get(0) is not None
    assert tracker.games_in_window(0) == 3
    assert 6.0 <= tracker.get(0) <= 8.0


def test_placement_ema_resets_on_remove():
    tracker = PlacementEmaTracker(window=20)
    tracker.record(1, 3)
    tracker.remove(1)
    assert tracker.get(1) is None
