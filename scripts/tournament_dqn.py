"""Round-robin tournament between all DQN 100k checkpoints."""
import numpy as np
import torch
from src.envs.connect4 import Connect4Game, build_state_dict, CONNECT4_ROWS, CONNECT4_COLS
from src.features.observation_builder import BoardChannels
from src.features.action_space import DiscreteActionSpace
from src.agents.dqn.agent import DQNAgent
from src.agents.dqn import action_scores

N_GAMES = 100
TEMPERATURE = 0.1  # softmax temperature over Q-values; T=0.5 was too flat (~2x best/worst ratio given Q-spread ~0.3)
BASE = "runs/connect4/test_improvements"

AGENTS = [
    ("baseline",            f"{BASE}/baseline/checkpoints/dqn_100000.pt"),
    ("3_nstep",             f"{BASE}/3_nstep/checkpoints/dqn_100000.pt"),
    ("5_nstep",             f"{BASE}/5_nstep/checkpoints/dqn_100000.pt"),
    ("with_noise",          f"{BASE}/with_noise/checkpoints/dqn_100000.pt"),
    ("with_noise_dist",     f"{BASE}/with_noise_dist/checkpoints/dqn_100000.pt"),
    ("with_dueling",        f"{BASE}/with_dueling/checkpoints/dqn_100000.pt"),
    ("nd_dueling",          f"{BASE}/with_noise_dist_dueling/checkpoints/dqn_100000.pt"),
    ("nd_dueling_aug",      f"{BASE}/with_noise_dist_dueling_aug/checkpoints/dqn_100000.pt"),
    ("full_rainbow",        f"{BASE}/with_noise_dist_dueling_aug_per/checkpoints/dqn_100000.pt"),
    ("no_per",              f"{BASE}/no_per/checkpoints/dqn_100000.pt"),
    ("per_fixed",           f"{BASE}/per_fixed/checkpoints/dqn_100000.pt"),
    ("perlr",               f"{BASE}/perlr/checkpoints/dqn_100000.pt"),
    ("rainbow_old",         f"{BASE}/rainbow_old/checkpoints/dqn_100000.pt"),
    ("rainbow_selfplay",    f"{BASE}/rainbow_selfplay/checkpoints/dqn_100000.pt"),
    ("rbow_wo_per",         f"{BASE}/rainbow_without_per/checkpoints/dqn_100000.pt"),
    ("rbow_wo_per_lr",      f"{BASE}/rainbow_without_per_lr/checkpoints/dqn_100000.pt"),
    ("better_params",       f"{BASE}/better_params_multi/checkpoints/dqn_100000.pt"),
]

game = Connect4Game()
obs_builder = BoardChannels(board_shape=(CONNECT4_ROWS, CONNECT4_COLS))
action_space = DiscreteActionSpace(n=7)

print(f"Loading {len(AGENTS)} agents...", flush=True)
loaded = {}
for name, path in AGENTS:
    agent = DQNAgent.load(path, action_space=action_space)
    agent.eval()
    loaded[name] = agent
print("Done.\n", flush=True)

# wins[a][b] = wins of a against b
wins = {n: {m: 0 for m, _ in AGENTS} for n, _ in AGENTS}
draws = {n: {m: 0 for m, _ in AGENTS} for n, _ in AGENTS}

names = [n for n, _ in AGENTS]
total_matchups = len(names) * (len(names) - 1) // 2

def act_softmax(agent: DQNAgent, obs: np.ndarray, legal_mask: np.ndarray, rng: np.random.Generator) -> int:
    """Sample action from softmax(Q / T) over legal actions."""
    with torch.no_grad():
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=agent.device).unsqueeze(0)
        lm_t = torch.as_tensor(legal_mask, dtype=torch.bool, device=agent.device).unsqueeze(0)
        out = agent.q_network(obs_t, legal_mask=lm_t)
        scores = action_scores(out)[0].cpu().numpy()
    scores[~legal_mask] = -np.inf
    scores = scores - scores[legal_mask].max()  # numerical stability
    probs = np.exp(scores / TEMPERATURE)
    probs[~legal_mask] = 0.0
    probs /= probs.sum()
    return int(rng.choice(len(probs), p=probs))


def play_games(agent_a, agent_b, n_games, seed=0):
    w_a = w_b = d = 0
    for i in range(n_games):
        a_idx = i % 2
        state = game.initial_state()
        rng = np.random.default_rng(seed + i)
        while not game.is_terminal(state):
            legal = list(game.legal_actions(state))
            lm = np.zeros(CONNECT4_COLS, dtype=bool)
            for ac in legal: lm[ac] = True
            obs = obs_builder.build(build_state_dict(state, game, legal_mask=lm))
            if state.current_player_index == a_idx:
                action = act_softmax(agent_a, obs, lm, rng)
            else:
                action = act_softmax(agent_b, obs, lm, rng)
            state = game.apply_action(state, action)
        w = game.winner(state)
        a_token = [1, -1][a_idx]
        if w == a_token:       w_a += 1
        elif w == -a_token:    w_b += 1
        else:                  d += 1
    return w_a, w_b, d

done = 0
for i, name_a in enumerate(names):
    for name_b in names[i+1:]:
        w_a, w_b, d = play_games(loaded[name_a], loaded[name_b], N_GAMES, seed=42)
        wins[name_a][name_b] = w_a
        wins[name_b][name_a] = w_b
        draws[name_a][name_b] = d
        draws[name_b][name_a] = d
        done += 1
        print(f"[{done:3d}/{total_matchups}] {name_a:<20} vs {name_b:<20}  {w_a}-{d}-{w_b}", flush=True)

# Scoring: win=2, draw=1, loss=0
scores = {}
for name in names:
    s = 0
    for opp in names:
        if opp == name: continue
        s += wins[name][opp] * 2 + draws[name][opp]
    scores[name] = s

ranked = sorted(names, key=lambda n: scores[n], reverse=True)

print("\n" + "=" * 55)
print(f"{'Rank':<5} {'Name':<22} {'Score':>7}  {'W':>5} {'D':>5} {'L':>5}  {'WR':>7}")
print("-" * 55)
for rank, name in enumerate(ranked, 1):
    total_w = sum(wins[name][o] for o in names if o != name)
    total_d = sum(draws[name][o] for o in names if o != name)
    total_l = sum(wins[o][name] for o in names if o != name)
    total_g = total_w + total_d + total_l
    wr = total_w / total_g if total_g else 0
    print(f"#{rank:<4} {name:<22} {scores[name]:>7}  {total_w:>5} {total_d:>5} {total_l:>5}  {wr:>6.1%}")
