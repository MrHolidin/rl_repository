"""Agent utility functions."""


def freeze_agent(agent) -> None:
    """Put agent into eval mode with pure exploitation (epsilon=0 if applicable)."""
    agent.eval()
    if hasattr(agent, "epsilon"):
        agent.epsilon = 0.0
