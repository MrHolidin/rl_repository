import numpy as np

import src.envs  # noqa: F401
from src.agents.random_agent import RandomAgent
from src.utils.match import _default_env_factory, play_match, play_single_game
from src.envs.reward_config import RewardConfig
from src.training.trainer import StartPolicy


def test_default_env_factory_minibg():
    from src.envs.minibg import OBS_DIM

    env = _default_env_factory("minibg", RewardConfig())
    obs = env.reset(seed=0)
    assert obs.shape == (OBS_DIM,)
    assert obs.dtype == np.float32


def test_play_single_game_minibg_two_random_agents():
    env = _default_env_factory("minibg", RewardConfig())
    a, b = RandomAgent(seed=1), RandomAgent(seed=2)
    out = play_single_game(
        env,
        a,
        b,
        start_policy=StartPolicy.AGENT_FIRST,
        seed=3,
        deterministic_agent=False,
        deterministic_opponent=False,
    )
    assert out["winner"] in (-1, 0, 1)
    assert out["agent_token"] == 1
    assert out["steps"] > 0


def test_play_match_minibg_keyword_game_id():
    w1, d, w2 = play_match(
        RandomAgent(seed=0),
        RandomAgent(seed=1),
        num_games=4,
        seed=99,
        randomize_first_player=True,
        env=None,
        game_id="minibg",
    )
    assert w1 + d + w2 == 4


def test_eval_build_opponents_minibg():
    from src.evaluation.eval_checkpoints import build_opponents_from_names

    op = build_opponents_from_names(["random", "tempo"], seed=5, game_id="minibg")
    assert set(op) == {"random", "tempo"}
    env = _default_env_factory("minibg", RewardConfig())
    for bot in op.values():
        if hasattr(bot, "set_env"):
            bot.set_env(env)
    m = env.legal_actions_mask
    _ = op["random"].act(env._get_obs(), legal_mask=m, deterministic=False)
