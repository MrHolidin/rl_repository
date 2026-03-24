"""AlphaZero (MCTS) vs DQN checkpoints — non-deterministic evaluation."""
import numpy as np
from src.envs.connect4 import Connect4Game, build_state_dict, CONNECT4_ROWS, CONNECT4_COLS
from src.features.observation_builder import BoardChannels
from src.features.action_space import DiscreteActionSpace
from src.agents.alphazero.agent import AlphaZeroAgent
from src.agents.dqn.agent import DQNAgent
from src.search.mcts import MCTSConfig, OptimizedMCTS, make_batched_evaluator

N_GAMES = 200
MCTS_SIMS = 200
TEMPERATURE = 0.3

AZ_CKPT = "runs/alphazero/connect4/test/alphazero_iter_000020.pt"

DQN_CKPTS = [
    ("rainbow_selfplay_500k",  "runs/connect4/test_improvements/rainbow_selfplay/checkpoints/dqn_500000.pt"),
    ("rainbow_selfplay_650k",  "runs/connect4/test_improvements/rainbow_selfplay/checkpoints/dqn_650000.pt"),
    ("rainbow_selfplay_750k",  "runs/connect4/test_improvements/rainbow_selfplay/checkpoints/dqn_750000.pt"),
    ("rainbow_selfplay_900k",  "runs/connect4/test_improvements/rainbow_selfplay/checkpoints/dqn_900000.pt"),
    ("rainbow_selfplay_1M",    "runs/connect4/test_improvements/rainbow_selfplay/checkpoints/dqn_1000000.pt"),
]


def play_match(az_agent, evaluator, opponent_dqn, game, obs_builder, num_games, seed=0):
    config = MCTSConfig(num_simulations=MCTS_SIMS, c_puct=1.4)
    az_agent.eval()
    wins, draws, losses = 0, 0, 0

    for game_idx in range(num_games):
        rng = np.random.default_rng(seed + game_idx)
        mcts = OptimizedMCTS(game, evaluator, config, rng=rng, batch_size=48)

        az_player_index = game_idx % 2
        az_token = [1, -1][az_player_index]
        state = game.initial_state()

        while not game.is_terminal(state):
            legal = list(game.legal_actions(state))
            legal_mask = np.zeros(CONNECT4_COLS, dtype=bool)
            for a in legal:
                legal_mask[a] = True

            state_dict = build_state_dict(state, game, legal_mask=legal_mask)
            obs = obs_builder.build(state_dict)

            if state.current_player_index == az_player_index:
                root = mcts.search(state, add_dirichlet_noise=False)
                action, _ = mcts.get_action_probs(root, temperature=TEMPERATURE)
            else:
                action = opponent_dqn.act(obs, legal_mask=legal_mask, deterministic=False)

            state = game.apply_action(state, action)

        winner = game.winner(state)
        if winner == az_token:
            wins += 1
        elif winner == 0:
            draws += 1
        else:
            losses += 1

    return wins, draws, losses


print(f"AlphaZero checkpoint : {AZ_CKPT}")
print(f"MCTS sims: {MCTS_SIMS}  temperature: {TEMPERATURE}  games per matchup: {N_GAMES}")
print()

game = Connect4Game()
obs_builder = BoardChannels(board_shape=(CONNECT4_ROWS, CONNECT4_COLS))

az_agent = AlphaZeroAgent.load(AZ_CKPT)
az_agent.eval()
evaluator = make_batched_evaluator(az_agent, game, build_state_dict, obs_builder)

# GPU warmup
import torch
state0 = game.initial_state()
w_cfg = MCTSConfig(num_simulations=10, c_puct=1.4)
w_mcts = OptimizedMCTS(game, evaluator, w_cfg, batch_size=48)
w_mcts.search(state0, add_dirichlet_noise=False)
print("GPU warmed up.\n")

action_space = DiscreteActionSpace(n=7)
print(f"{'opponent':<28}  {'W':>4}  {'D':>4}  {'L':>4}    {'WR':>7}")
print("-" * 58)

for name, path in DQN_CKPTS:
    print(f"  running {name}...", flush=True, end="\r")
    dqn = DQNAgent.load(path, action_space=action_space)
    dqn.eval()
    w, d, l = play_match(az_agent, evaluator, dqn, game, obs_builder, N_GAMES, seed=42)
    print(f"{name:<28}  {w:>4}  {d:>4}  {l:>4}    {w/N_GAMES:>6.1%}")
