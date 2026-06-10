"""Final-board per-minion / per-tribe terminal reward shaping."""

from __future__ import annotations

import pytest

from src.bg_core.minion import Minion, Race
from src.bg_lobby.match_types import EliminatedSnapshot
from src.envs.bglike.board_shaping import (
    BoardShapingConfig,
    final_board_for_seat,
    minions_shaping_total,
    parse_minions_shaping,
    parse_tribes_shaping,
    terminal_reward_for_seat,
    tribes_shaping_total,
)
from src.envs.bglike.game import BGLikeGame
from src.envs.bglike.placement import placement_reward


def _minion(
    *,
    name: str = "test",
    attack: int = 1,
    health: int = 1,
    race: Race | None = None,
) -> Minion:
    return Minion(
        card_id="test",
        name=name,
        base_attack=attack,
        base_health=health,
        tier=1,
        race=race,
    )


def _state_with_eliminated_board(seat: int, board: tuple[Minion, ...]):
    game = BGLikeGame(seed=0, patch_dir="data/bgcore/15_6_2_36393")
    state = game.initial_state()
    state.eliminated = (
        EliminatedSnapshot(
            seat=seat,
            last_board=board,
            tavern_tier=4,
            eliminated_combat_round=1,
        ),
    )
    state.alive = tuple(s for s in state.alive if s != seat)
    return state


def test_minions_shaping_sums_per_matching_name():
    board = (
        _minion(name="Rockpool Hunter"),
        _minion(name="Rockpool Hunter"),
        _minion(name="Alleycat"),
    )
    bonuses = parse_minions_shaping({"Rockpool Hunter": 0.03, "Alleycat": 0.01})
    assert minions_shaping_total(board, bonuses) == pytest.approx(0.07)


def test_tribes_shaping_sums_per_matching_race():
    board = (
        _minion(race=Race.MURLOC),
        _minion(race=Race.MURLOC),
        _minion(race=Race.BEAST),
    )
    bonuses = parse_tribes_shaping({"MURLOC": 0.02, "BEAST": 0.05})
    assert tribes_shaping_total(board, bonuses) == pytest.approx(0.09)


def test_terminal_reward_adds_configured_bonuses():
    board = (
        _minion(name="Hero Minion", race=Race.ELEMENTAL),
        _minion(name="Hero Minion", race=Race.ELEMENTAL),
    )
    state = _state_with_eliminated_board(2, board)
    cfg = BoardShapingConfig.from_params(
        minions_shaping={"Hero Minion": 0.1},
        tribes_shaping={"ELEMENTAL": 0.02},
    )
    place = 4
    base = placement_reward(place)
    total = terminal_reward_for_seat(state, 2, place, cfg)
    assert total == pytest.approx(base + 0.1 * 2 + 0.02 * 2)


def test_empty_config_preserves_placement_reward():
    board = (_minion(name="X", race=Race.MURLOC),)
    state = _state_with_eliminated_board(1, board)
    place = 3
    assert terminal_reward_for_seat(state, 1, place) == placement_reward(place)


def test_final_board_for_seat_reads_eliminated_snapshot():
    board = (_minion(name="A", race=Race.BEAST),)
    state = _state_with_eliminated_board(2, board)
    assert final_board_for_seat(state, 2) == board


def test_parse_tribes_shaping_rejects_unknown_race():
    with pytest.raises(KeyError):
        parse_tribes_shaping({"NOT_A_TRIBE": 1.0})


def test_v6_config_shaping_matches_catalog_and_reward_math():
    """Regression: ppo_structured_v6_74257 minion names must match patch catalog."""
    from src.bg_catalog.cards import make_minion
    from src.bg_catalog.patch_context import load_patch_context
    from src.config import load_config
    from src.envs.bglike.board_shaping import terminal_reward_breakdown
    from src.training.bglike_perspective import make_bglike_training_env

    app = load_config("configs/bglike/ppo_structured_v6_74257.yaml")
    gp = dict(app.game.params)
    patch_dir = gp["patch_dir"]
    patch = load_patch_context(patch_dir)

    configured_names = set(gp["minions_shaping"].keys())
    catalog_names = {m.name for m in patch.templates.values()}
    assert configured_names <= catalog_names

    lobby_kw = {
        k: v
        for k, v in gp.items()
        if k not in ("num_current_seats", "battle_damage_shaping", "seed", "percent_high_game")
    }
    env = make_bglike_training_env((0,), seed=13, **lobby_kw)
    cfg = env._board_shaping

    board = (
        make_minion("BGS_104", patch=patch),
        make_minion("BGS_023", patch=patch),
        make_minion("BGS_023", patch=patch),
        make_minion("BGS_202", patch=patch),
        make_minion("BGS_009", patch=patch),
    )
    game = BGLikeGame(seed=0, patch_dir=patch_dir)
    state = game.initial_state()
    state.eliminated = (
        EliminatedSnapshot(
            seat=0,
            last_board=board,
            tavern_tier=5,
            eliminated_combat_round=10,
        ),
    )
    state.alive = tuple(s for s in state.alive if s != 0)

    place = 4
    bd = terminal_reward_breakdown(state, 0, place, cfg)
    assert bd["minions_shaping"] == pytest.approx(0.1 - 0.1 - 0.1)
    assert bd["tribes_shaping"] == pytest.approx(2 * -0.03)
    assert bd["placement_reward"] == pytest.approx(
        placement_reward(place) + bd["board_shaping_total"]
    )
