"""Tests for environment and agent registries."""

from __future__ import annotations

from uuid import uuid4

import pytest

from src.registry import (
    get_agent_entry,
    get_game_entry,
    list_agents,
    list_games,
    make_agent,
    make_game,
    register_agent,
    register_game,
)
import src.agents  # noqa: F401 - ensures default agents are registered
import src.envs  # noqa: F401 - ensures default games are registered


class _StubGame:
    def __init__(self, size: int, reward: float = 0.0) -> None:
        self.size = size
        self.reward = reward


class _StubAgent:
    def __init__(self, name: str, epsilon: float) -> None:
        self.name = name
        self.epsilon = epsilon


def test_register_and_make_game():
    game_id = f"stub_game_{uuid4().hex}"
    register_game(game_id, _StubGame, size=4, reward=1.0)

    instance = make_game(game_id, reward=2.5)
    assert isinstance(instance, _StubGame)
    assert instance.size == 4
    assert instance.reward == 2.5

    factory, defaults = get_game_entry(game_id)
    assert factory is _StubGame
    assert defaults == {"size": 4, "reward": 1.0}

    with pytest.raises(ValueError):
        register_game(game_id, _StubGame)


def test_register_and_make_agent():
    agent_id = f"stub_agent_{uuid4().hex}"
    register_agent(agent_id, _StubAgent)

    instance = make_agent(agent_id, name="test", epsilon=0.1)
    assert isinstance(instance, _StubAgent)
    assert instance.name == "test"
    assert instance.epsilon == 0.1

    ctor = get_agent_entry(agent_id)
    assert ctor is _StubAgent

    with pytest.raises(ValueError):
        register_agent(agent_id, _StubAgent)


def test_registry_lists_include_defaults():
    games = list_games()
    agents = list_agents()

    assert "connect4" in games
    assert {"random", "heuristic", "smart_heuristic", "qlearning", "dqn"}.issubset(set(agents))


def test_registry_make_missing_entries():
    with pytest.raises(KeyError):
        make_game("missing_game")
    with pytest.raises(KeyError):
        make_agent("missing_agent")

