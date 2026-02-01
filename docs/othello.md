# Othello Environment

## Overview

The Othello (Reversi) environment implements a standard 8x8 board game where two players place discs and flip opponent pieces.

## Game Rules

- **Board**: 8x8 grid
- **Starting Position**: 4 pieces in the center (2 black, 2 white) in a diagonal pattern
- **Players**: Two players (tokens: 1 and -1)
- **Objective**: Have the most pieces when no more moves are available

### Move Rules

1. Players must place a piece that flips at least one opponent piece
2. Flipping occurs in straight lines (horizontal, vertical, diagonal)
3. If a player has no legal moves, their turn is skipped
4. Game ends when neither player can move
5. Winner is determined by piece count

## Environment Structure

### Files

- `env.py`: Main environment class (`OthelloEnv`)
- `game.py`: Pure game rules (`OthelloGame`)
- `state.py`: State dataclass (`OthelloState`)
- `utils.py`: Helper functions (flipping, validation, etc.)
- `eval.py`: Evaluation function for search algorithms

### Key Features

- **Action Space**: 64 positions (8x8 board), represented as integers 0-63
  - Action = row × 8 + col
- **Observation**: Board state with current player perspective
- **Legal Actions**: Only valid moves that flip at least one opponent piece
- **Rewards**: Configurable via `RewardConfig`
  - Win: +1.0 (default)
  - Loss: -1.0 (default)
  - Draw: 0.0 (default)
  - Invalid action: -1.0 (default)

## Usage

### Basic Usage

```python
from src.envs.othello import OthelloEnv

env = OthelloEnv()
obs = env.reset()

while not env.done:
    legal_actions = env.get_legal_actions()
    action = # ... select action ...
    result = env.step(action)
    obs = result.obs
```

### Using the Registry

```python
import src.envs
from src.registry import make_game

env = make_game('othello')
```

### With Agents

```python
from src.envs.othello import OthelloEnv
from src.agents.random_agent import RandomAgent
from src.agents.othello import OthelloHeuristicAgent

env = OthelloEnv()
agent = OthelloHeuristicAgent()

obs = env.reset()
while not env.done:
    action = agent.act(obs, legal_mask=env.legal_actions_mask)
    result = env.step(action)
    obs = result.obs
```

## Agents

### OthelloHeuristicAgent

A rule-based agent using positional heuristics:

- **Corner positions**: Highest value (stable, cannot be flipped)
- **Edge positions**: Medium value
- **Adjacent to corners**: Negative value (dangerous)
- **Disc count**: Maximizes flips

Performance: ~85% win rate vs random agent

## Testing

Run tests with:

```bash
python3 -m pytest tests/test_othello_env.py -v
```

Test scripts:

- `scripts/test_othello.py`: Play random vs random game
- `scripts/test_othello_heuristic.py`: Benchmark heuristic vs random

## Implementation Notes

### Pass Moves

The environment automatically handles pass moves when a player has no legal actions. The turn switches to the opponent, and if the opponent also has no moves, the game ends.

### State Management

- `get_state()`: Returns a copy of the current state
- `set_state()`: Restores a previous state (useful for search algorithms)
- `get_state_hash()`: Returns a string representation for caching

### Rendering

The `render()` method displays the board in ASCII format with:
- Row/column indices
- X for player 1, O for player -1
- Piece counts
- Current player indicator

## Future Enhancements

Possible improvements:

1. **Mobility heuristic**: Count legal moves for each player
2. **Stability analysis**: Identify stable discs that cannot be flipped
3. **Parity**: Consider odd/even move counts in endgame
4. **Pattern recognition**: Detect common board patterns
5. **Opening book**: Pre-computed optimal opening moves
