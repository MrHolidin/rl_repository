"""MCTS configuration."""

from dataclasses import dataclass


@dataclass
class MCTSConfig:
    """Configuration for Monte Carlo Tree Search."""

    num_simulations: int = 800
    c_puct: float = 1.4

    dirichlet_alpha: float = 0.3
    dirichlet_frac: float = 0.25

    temperature: float = 1.0
    temperature_threshold: int = 15

    def get_temperature(self, move_number: int) -> float:
        if move_number < self.temperature_threshold:
            return self.temperature
        return 0.0
