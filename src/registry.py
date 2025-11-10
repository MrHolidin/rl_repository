"""Central registries for environments and agents."""

from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, Tuple


GameFactory = Callable[..., Any]
AgentFactory = Callable[..., Any]

_GAME_REGISTRY: Dict[str, Tuple[GameFactory, Dict[str, Any]]] = {}
_AGENT_REGISTRY: Dict[str, AgentFactory] = {}


def register_game(game_id: str, entry_point: GameFactory, **default_kwargs: Any) -> None:
    """Register a turn-based environment constructor."""
    if game_id in _GAME_REGISTRY:
        raise ValueError(f"Game id '{game_id}' is already registered.")
    _GAME_REGISTRY[game_id] = (entry_point, dict(default_kwargs))


def make_game(game_id: str, **overrides: Any) -> Any:
    """Instantiate a registered game using optional parameter overrides."""
    if game_id not in _GAME_REGISTRY:
        raise KeyError(f"Game id '{game_id}' is not registered.")

    entry_point, defaults = _GAME_REGISTRY[game_id]
    params = {**defaults, **overrides}
    return entry_point(**params)


def list_games() -> Iterable[str]:
    """Return iterable of registered game identifiers."""
    return tuple(_GAME_REGISTRY.keys())


def get_game_entry(game_id: str) -> Tuple[GameFactory, Dict[str, Any]]:
    """Retrieve the raw entry point and defaults for a game."""
    if game_id not in _GAME_REGISTRY:
        raise KeyError(f"Game id '{game_id}' is not registered.")
    entry_point, defaults = _GAME_REGISTRY[game_id]
    return entry_point, dict(defaults)


def register_agent(agent_id: str, ctor: AgentFactory) -> None:
    """Register an agent constructor."""
    if agent_id in _AGENT_REGISTRY:
        raise ValueError(f"Agent id '{agent_id}' is already registered.")
    _AGENT_REGISTRY[agent_id] = ctor


def make_agent(agent_id: str, **kwargs: Any) -> Any:
    """Instantiate a registered agent."""
    if agent_id not in _AGENT_REGISTRY:
        raise KeyError(f"Agent id '{agent_id}' is not registered.")
    return _AGENT_REGISTRY[agent_id](**kwargs)


def list_agents() -> Iterable[str]:
    """Return iterable of registered agent identifiers."""
    return tuple(_AGENT_REGISTRY.keys())


def get_agent_entry(agent_id: str) -> AgentFactory:
    """Retrieve the raw constructor for an agent."""
    if agent_id not in _AGENT_REGISTRY:
        raise KeyError(f"Agent id '{agent_id}' is not registered.")
    return _AGENT_REGISTRY[agent_id]

