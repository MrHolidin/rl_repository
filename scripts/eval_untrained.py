import numpy as np
from src.envs.connect4 import Connect4Game, build_state_dict, CONNECT4_ROWS, CONNECT4_COLS
from src.features.observation_builder import BoardChannels
from src.models.alphazero import Connect4AlphaZeroNetwork
from src.agents.alphazero import AlphaZeroAgent
from src.agents.connect4.heuristic_agent import HeuristicAgent
from src.search.mcts import MCTSConfig, OptimizedMCTS, make_batched_evaluator

game        = Connect4Game()
obs_builder = BoardChannels(board_shape=(CONNECT4_ROWS, CONNECT4_COLS))
network     = Connect4AlphaZeroNetwork(rows=CONNECT4_ROWS, cols=CONNECT4_COLS,
    in_channels=2, num_actions=CONNECT4_COLS, trunk_channels=64, num_res_blocks=4)
agent = AlphaZeroAgent(network=network, num_actions=CONNECT4_COLS)
agent.eval()

evaluator = make_batched_evaluator(agent, game, build_state_dict, obs_builder)
heuristic = HeuristicAgent(seed=0)

for sims in [50, 200, 800]:
    config = MCTSConfig(num_simulations=sims, c_puct=1.4)
    wins, draws, losses = 0, 0, 0
    N = 200
    for i in range(N):
        rng = np.random.default_rng(i)
        mcts = OptimizedMCTS(game, evaluator, config, rng=rng, batch_size=16)
        az_idx = i % 2
        az_token = [1, -1][az_idx]
        state = game.initial_state()
        while not game.is_terminal(state):
            legal = list(game.legal_actions(state))
            legal_mask = np.zeros(CONNECT4_COLS, dtype=bool)
            for a in legal: legal_mask[a] = True
            if state.current_player_index == az_idx:
                root = mcts.search(state, add_dirichlet_noise=False)
                action, _ = mcts.get_action_probs(root, temperature=0.0)
            else:
                state_dict = build_state_dict(state, game, legal_mask=legal_mask)
                obs = obs_builder.build(state_dict)
                action = heuristic.act(obs, legal_mask=legal_mask, deterministic=False)
            state = game.apply_action(state, action)
        winner = game.winner(state)
        if winner == az_token: wins += 1
        elif winner == 0: draws += 1
        else: losses += 1
    print(f"sims={sims:4d}: {wins}W {draws}D {losses}L  wr={wins/N:.1%}", flush=True)
