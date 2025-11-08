"""Serialization utilities for agents."""

import pickle
import os
from typing import Any


def save_agent(agent: Any, path: str) -> None:
    """
    Save an agent to file.
    
    Args:
        agent: Agent instance to save
        path: Path to save file
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    if hasattr(agent, "save"):
        agent.save(path)
    else:
        # Fallback: pickle
        with open(path, "wb") as f:
            pickle.dump(agent, f)


def load_agent(agent: Any, path: str) -> Any:
    """
    Load an agent from file.
    
    Args:
        agent: Agent instance or class to load into
        path: Path to load file from
        
    Returns:
        Loaded agent
    """
    if hasattr(agent, "load"):
        agent.load(path)
        return agent
    else:
        # Fallback: pickle
        with open(path, "rb") as f:
            return pickle.load(f)

