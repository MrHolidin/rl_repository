import numpy as np
import pytest

from src.envs.minibg.heuristic_bots.bots import default_bot_constructors
from src.envs.minibg.heuristic_bots.common import legal_env_indices
from src.envs.minibg.heuristic_bots.tournament import make_bot, play_game, run_tournament


@pytest.mark.parametrize("name", sorted(default_bot_constructors().keys()))
def test_bot_always_chooses_legal_action(name: str) -> None:
    from src.envs.minibg.env import MiniBGEnv

    for seed in range(5):
        env = MiniBGEnv(seed=seed + 9000)
        bot = make_bot(name, seed + 11)
        for _ in range(200):
            if env.done:
                break
            mask = env.legal_actions_mask
            a = bot.choose_action(env)
            assert 0 <= a < len(mask), (name, a)
            assert bool(mask[a]), (name, seed, a, np.where(mask)[0].tolist())
            env.step(a)


def test_play_game_terminates() -> None:
    b0 = make_bot("t1_random", 1)
    b1 = make_bot("t_up_random", 2)
    r = play_game(b0, b1, seed=42)
    assert r in ("bot0", "bot1", "draw")


def test_tournament_smoke() -> None:
    names = ["t1_random", "t_up_random"]
    res = run_tournament(names, games_per_pair=1, base_seed=7)
    assert len(res) == 1
    for wa, wb, d, to in res.values():
        assert wa + wb + d + to == 2
