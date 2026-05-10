from src.agents.random_agent import RandomAgent
from src.envs.minibg import MiniBGEnv
from src.registry import make_game
from src.training.control_flow import (
    AlternatingDriver,
    MiniBGDriver,
    make_control_driver,
)
from src.training.trainer import Trainer


def test_make_control_driver_defaults():
    assert isinstance(make_control_driver("connect4", None), AlternatingDriver)
    assert isinstance(make_control_driver("minibg", None), MiniBGDriver)
    assert isinstance(make_control_driver("connect4", "alternating"), AlternatingDriver)
    assert isinstance(make_control_driver("minibg", "minibg"), MiniBGDriver)


def test_minibg_immediate_trainer_smoke():
    env = MiniBGEnv(seed=0)
    agent = RandomAgent(seed=1)
    trainer = Trainer(
        env,
        agent,
        control_driver=MiniBGDriver(),
        opponent_sampler=None,
    )
    trainer.train(total_steps=8)
    assert trainer.global_step == 8


def test_minibg_make_game_registered():
    env = make_game("minibg", seed=0)
    assert isinstance(env, MiniBGEnv)
    assert len(env.legal_actions_mask) == 33
