"""Canonical Connect4 opponent set and single-agent evaluation table."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from src.agents import HeuristicAgent, RandomAgent, SmartHeuristicAgent
from src.agents.base_agent import BaseAgent
from src.envs import RewardConfig
from src.envs.connect4 import Connect4Game
from src.search.connect4.heuristic_minimax import make_connect4_heuristic_minimax_policy
from src.search.connect4.minimax_env_adapter import Connect4MinimaxEnvAdapter
from src.utils import freeze_agent
from src.utils.match import play_match_batched

# Best available DQN checkpoint (~1 M env steps).
# The run dqn_20251204_202348 reached 975 k steps and uses the legacy
# checkpoint format (network_type / observation_shape instead of network_class).
DEFAULT_DQN_1M_PATH: Path = (
    Path(__file__).resolve().parent.parent.parent
    / "data/checkpoints/dqn_20251204_202348/dqn_975000.pt"
)


class _ObsPaddedAgent(BaseAgent):
    """
    Wraps an agent to pad the observation channel dim from ``src_ch`` to
    ``tgt_ch`` (zeros-fill trailing channels) before forwarding.

    Needed when a checkpoint was trained with more input channels than the
    env currently produces (e.g. legacy 3-channel DQN vs current 2-channel env).
    """

    def __init__(self, agent: BaseAgent, src_channels: int, tgt_channels: int) -> None:
        self._agent = agent
        self._src = src_channels
        self._tgt = tgt_channels

    def act(
        self,
        obs: np.ndarray,
        legal_mask: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> int:
        if obs.shape[0] < self._tgt:
            pad = np.zeros((self._tgt - obs.shape[0],) + obs.shape[1:], dtype=obs.dtype)
            obs = np.concatenate([obs, pad], axis=0)
        return self._agent.act(obs, legal_mask=legal_mask, deterministic=deterministic)

    def act_batch(
        self,
        obs_batch: np.ndarray,
        legal_batch: np.ndarray,
        deterministic: bool = False,
    ) -> np.ndarray:
        if obs_batch.shape[1] < self._tgt:
            pad = np.zeros(
                (obs_batch.shape[0], self._tgt - obs_batch.shape[1]) + obs_batch.shape[2:],
                dtype=obs_batch.dtype,
            )
            obs_batch = np.concatenate([obs_batch, pad], axis=1)
        return self._agent.act_batch(obs_batch, legal_batch, deterministic=deterministic)

    def eval(self) -> None:
        if hasattr(self._agent, "eval"):
            self._agent.eval()

    def save(self, path: str) -> None:
        self._agent.save(path)

    @classmethod
    def load(cls, path: str, **kwargs: object) -> "BaseAgent":
        raise NotImplementedError


def _load_dqn_any_format(path: Path, *, seed: int = 42) -> BaseAgent:
    """
    Load a DQNAgent from either the current checkpoint format
    (network_class / network_kwargs) or the legacy format
    (network_type / observation_shape / q_network_state_dict).

    If the checkpoint's input channel count differs from the env's default
    (2 channels), wraps the result in ``_ObsPaddedAgent``.
    """
    from src.agents.dqn.agent import DQNAgent
    from src.utils import freeze_agent as _freeze

    map_loc = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(str(path), map_location=map_loc, weights_only=False)

    if "network_class" in ck:
        # Current format.
        in_ch = ck["network_kwargs"].get("in_channels", 2)
        agent = DQNAgent.load(str(path), device=map_loc, seed=seed, load_optimizer=False)
    else:
        # Legacy format (pre-refactor): uses ConvQNetwork.
        from src.models.q_network_factory import ConvQNetwork
        from src.features.action_space import DiscreteActionSpace
        obs_shape = ck["observation_shape"]  # e.g. (3, 6, 7)
        in_ch = obs_shape[0]
        num_actions = ck["num_actions"]
        dueling = ck.get("network_type", "dqn") == "dueling_dqn"
        q_sd = ck["q_network_state_dict"]
        fc_hidden = q_sd["value_stream.0.weight"].shape[0]
        network = ConvQNetwork(
            observation_shape=tuple(obs_shape),
            num_actions=num_actions,
            conv_layers_config=[
                {"out_channels": 64, "kernel_size": 3, "stride": 1, "padding": 1},
                {"out_channels": 128, "kernel_size": 3, "stride": 1, "padding": 1},
            ],
            fc_layers=[fc_hidden],
            dueling=dueling,
        )
        agent = DQNAgent(
            network=network,
            num_actions=num_actions,
            learning_rate=ck.get("learning_rate", 1e-4),
            discount_factor=ck.get("discount_factor", 0.99),
            epsilon=0.0,
            batch_size=ck.get("batch_size", 32),
            replay_buffer_size=ck.get("replay_buffer_capacity", 10_000),
            target_update_freq=ck.get("target_update_freq", 100),
            soft_update=ck.get("soft_update", False),
            tau=ck.get("tau", 0.01),
            n_step=ck.get("n_step", 1),
            device=map_loc,
            seed=seed,
            action_space=DiscreteActionSpace(num_actions),
        )
        agent.q_network.load_state_dict(q_sd)
        agent.target_network.load_state_dict(ck["target_network_state_dict"])
        agent.step_count = ck.get("step_count", 0)

    _freeze(agent)

    # Env produces 2-channel obs by default; pad if checkpoint expects more.
    env_channels = 2
    if in_ch > env_channels:
        return _ObsPaddedAgent(agent, src_channels=env_channels, tgt_channels=in_ch)
    return agent

# Ordered list of canonical opponent names.
CANONICAL_OPPONENT_NAMES: List[str] = [
    "random",
    "heuristic",
    "smart_heuristic",
    "minimax_2",
    "minimax_2_smart",
    "minimax_4",
    "minimax_3_smart",
    "dqn_1m",
]


def _make_minimax(
    depth: int,
    heuristic: str,
    seed: int,
) -> Connect4MinimaxEnvAdapter:
    game = Connect4Game(rows=6, cols=7)
    policy = make_connect4_heuristic_minimax_policy(
        game,
        depth=depth,
        heuristic=heuristic,  # type: ignore[arg-type]
        rng=np.random.default_rng(seed),
    )
    return Connect4MinimaxEnvAdapter(game=game, policy=policy, get_state=None)


def build_canonical_opponents(
    *,
    dqn_1m_path: Optional[Path] = None,
    include_dqn: bool = True,
    seed: int = 42,
) -> Dict[str, BaseAgent]:
    """
    Build the canonical Connect4 opponent set.

    Args:
        dqn_1m_path: Path to the ~1 M-step DQN checkpoint.
                     Defaults to DEFAULT_DQN_1M_PATH.
                     If the file does not exist and include_dqn is True, a
                     RuntimeError is raised.
        include_dqn: Whether to include the DQN checkpoint opponent.
        seed: Base RNG seed; each opponent gets seed+offset for independence.

    Returns:
        Ordered dict of {name: agent}.
    """
    opponents: Dict[str, BaseAgent] = {}

    opponents["random"] = RandomAgent(seed=seed + 1)
    opponents["heuristic"] = HeuristicAgent(seed=seed + 2)
    opponents["smart_heuristic"] = SmartHeuristicAgent(seed=seed + 3)
    opponents["minimax_2"] = _make_minimax(depth=2, heuristic="trivial", seed=seed + 4)
    opponents["minimax_2_smart"] = _make_minimax(depth=2, heuristic="smart", seed=seed + 5)
    opponents["minimax_4"] = _make_minimax(depth=4, heuristic="trivial", seed=seed + 6)
    opponents["minimax_3_smart"] = _make_minimax(depth=3, heuristic="smart", seed=seed + 7)

    if include_dqn:
        path = Path(dqn_1m_path) if dqn_1m_path is not None else DEFAULT_DQN_1M_PATH
        if not path.exists():
            raise RuntimeError(
                f"DQN checkpoint not found: {path}\n"
                "Pass dqn_1m_path= to point to an existing checkpoint, "
                "or set include_dqn=False."
            )
        dqn_agent = _load_dqn_any_format(path, seed=seed + 8)
        opponents["dqn_1m"] = dqn_agent

    for agent in opponents.values():
        freeze_agent(agent)

    return opponents


def _play_both_sides(
    agent: BaseAgent,
    opponent: BaseAgent,
    *,
    num_games_per_side: int,
    seed: int,
    reward_config: RewardConfig,
    batch_size: int,
) -> Tuple[int, int, int, int, int, int]:
    """
    Play agent vs opponent with agent as P1 and P2, return:
        (wins_p1, draws_p1, losses_p1, wins_p2, draws_p2, losses_p2)
    """
    # Agent goes first (P1)
    w1, d1, l1 = play_match_batched(
        agent, opponent,
        num_games=num_games_per_side,
        batch_size=min(batch_size, num_games_per_side),
        seed=seed,
        randomize_first_player=False,
        reward_config=reward_config,
    )
    # Agent goes second (P2)
    l2, d2, w2 = play_match_batched(
        opponent, agent,
        num_games=num_games_per_side,
        batch_size=min(batch_size, num_games_per_side),
        seed=seed + 100_000,
        randomize_first_player=False,
        reward_config=reward_config,
    )
    return w1, d1, l1, w2, d2, l2


def evaluate_agent(
    agent: BaseAgent,
    *,
    opponents: Optional[Dict[str, BaseAgent]] = None,
    num_games_per_side: int = 100,
    seed: int = 42,
    batch_size: int = 64,
    reward_config: Optional[RewardConfig] = None,
    dqn_1m_path: Optional[Path] = None,
    include_dqn: bool = True,
) -> pd.DataFrame:
    """
    Evaluate *agent* against the canonical Connect4 opponent set.

    Each opponent is played ``num_games_per_side`` games with agent as player 1
    and ``num_games_per_side`` games with agent as player 2.

    Args:
        agent: The agent under evaluation.
        opponents: Pre-built opponent dict (skips build_canonical_opponents).
        num_games_per_side: Games per side per opponent.
        seed: Base seed for reproducibility.
        batch_size: Parallel envs for batched play.
        reward_config: Env reward config (default: RewardConfig()).
        dqn_1m_path: Path to the ~1 M-step DQN checkpoint.
        include_dqn: Include DQN checkpoint opponent (ignored if opponents= given).

    Returns:
        DataFrame indexed by opponent name with columns:
            win_rate, draw_rate, loss_rate,
            win_rate_p1, draw_rate_p1, loss_rate_p1,
            win_rate_p2, draw_rate_p2, loss_rate_p2,
            wins, draws, losses, games.
    """
    if reward_config is None:
        reward_config = RewardConfig()

    if opponents is None:
        opponents = build_canonical_opponents(
            dqn_1m_path=dqn_1m_path,
            include_dqn=include_dqn,
            seed=seed,
        )

    freeze_agent(agent)

    rows: List[dict] = []
    for opp_name, opponent in opponents.items():
        opp_seed = seed + abs(hash(opp_name)) % (2**16)
        w1, d1, l1, w2, d2, l2 = _play_both_sides(
            agent, opponent,
            num_games_per_side=num_games_per_side,
            seed=opp_seed,
            reward_config=reward_config,
            batch_size=batch_size,
        )
        n1 = w1 + d1 + l1
        n2 = w2 + d2 + l2
        total_w = w1 + w2
        total_d = d1 + d2
        total_l = l1 + l2
        total = n1 + n2

        rows.append({
            "opponent": opp_name,
            "win_rate":    total_w / total if total else 0.0,
            "draw_rate":   total_d / total if total else 0.0,
            "loss_rate":   total_l / total if total else 0.0,
            "win_rate_p1": w1 / n1 if n1 else 0.0,
            "draw_rate_p1": d1 / n1 if n1 else 0.0,
            "loss_rate_p1": l1 / n1 if n1 else 0.0,
            "win_rate_p2": w2 / n2 if n2 else 0.0,
            "draw_rate_p2": d2 / n2 if n2 else 0.0,
            "loss_rate_p2": l2 / n2 if n2 else 0.0,
            "wins":  total_w,
            "draws": total_d,
            "losses": total_l,
            "games": total,
        })

    df = pd.DataFrame(rows).set_index("opponent")
    return df


class AlphaZeroMCTSAgent(BaseAgent):
    """
    Wraps AlphaZeroAgent + OptimizedMCTS as a drop-in BaseAgent.

    Requires ``set_env(env)`` to be called before each ``act`` so the
    agent can read the current game state from the environment.
    """

    def __init__(
        self,
        az_agent: "AlphaZeroAgent",
        game: Connect4Game,
        evaluator,
        mcts_sims: int = 200,
        temperature: float = 0.1,
        c_puct: float = 1.4,
        batch_size: int = 48,
        seed: int = 42,
    ) -> None:
        from src.search.mcts import MCTSConfig
        self._az = az_agent
        self._game = game
        self._evaluator = evaluator
        self._config = MCTSConfig(num_simulations=mcts_sims, c_puct=c_puct)
        self._temperature = temperature
        self._batch_size = batch_size
        self._rng = np.random.default_rng(seed)
        self._env = None

    def set_env(self, env) -> None:
        self._env = env

    def act(
        self,
        obs: np.ndarray,
        legal_mask: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> int:
        from src.search.mcts import OptimizedMCTS
        if self._env is None:
            raise RuntimeError("AlphaZeroMCTSAgent: call set_env(env) before act()")
        state = self._env.get_state()
        mcts = OptimizedMCTS(
            self._game, self._evaluator, self._config,
            rng=self._rng, batch_size=self._batch_size,
        )
        root = mcts.search(state, add_dirichlet_noise=False)
        action, _ = mcts.get_action_probs(root, temperature=self._temperature)
        return int(action)

    def eval(self) -> None:
        self._az.eval()

    def save(self, path: str) -> None:
        self._az.save(path)

    @classmethod
    def load(cls, path: str, **kwargs: object) -> "BaseAgent":
        raise NotImplementedError


def load_az_mcts_agent(
    checkpoint_path: Path,
    *,
    mcts_sims: int = 200,
    temperature: float = 0.1,
    c_puct: float = 1.4,
    mcts_batch_size: int = 48,
    device: Optional[str] = None,
    seed: int = 42,
) -> AlphaZeroMCTSAgent:
    """
    Load AlphaZero from checkpoint and wrap with MCTS.

    Args:
        checkpoint_path: Path to the .pt checkpoint.
        mcts_sims: MCTS simulations per move.
        temperature: Action sampling temperature (0 = greedy).
        c_puct: PUCT exploration constant.
        mcts_batch_size: Leaf evaluation batch size for MCTS.
        device: Torch device.
        seed: RNG seed for MCTS.

    Returns:
        AlphaZeroMCTSAgent ready to use as a BaseAgent.
    """
    from src.agents.alphazero.agent import AlphaZeroAgent
    from src.envs.connect4 import Connect4Game, build_state_dict
    from src.features.observation_builder import BoardChannels
    from src.search.mcts import make_batched_evaluator

    az = AlphaZeroAgent.load(str(checkpoint_path), device=device)
    az.eval()

    game = Connect4Game()
    obs_builder = BoardChannels(board_shape=(6, 7))
    evaluator = make_batched_evaluator(az, game, build_state_dict, obs_builder)

    return AlphaZeroMCTSAgent(
        az_agent=az,
        game=game,
        evaluator=evaluator,
        mcts_sims=mcts_sims,
        temperature=temperature,
        c_puct=c_puct,
        batch_size=mcts_batch_size,
        seed=seed,
    )


def print_eval_table(df: pd.DataFrame) -> None:
    """Pretty-print the evaluation DataFrame."""
    display_cols = ["win_rate", "draw_rate", "loss_rate", "win_rate_p1", "win_rate_p2", "games"]
    available = [c for c in display_cols if c in df.columns]
    fmt = {c: "{:.1%}".format for c in available if c != "games"}
    formatted = df[available].copy()
    for col in fmt:
        if col in formatted:
            formatted[col] = formatted[col].map(fmt[col])
    print(formatted.to_string())
