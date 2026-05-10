from src.agents.random_agent import RandomAgent
from src.envs.minibg import MiniBGEnv
from src.registry import make_game
from src.training.agent_perspective_env import AgentPerspectiveEnv
from src.training.control_flow import active_role
from src.training.opponent_sampler import RandomOpponentSampler
from src.training.trainer import Trainer


def test_active_role_predicate_minibg():
    env = MiniBGEnv(seed=0)
    env.reset(seed=0)
    role_p1 = active_role(env, agent_token=1)
    role_p_neg1 = active_role(env, agent_token=-1)
    assert {role_p1, role_p_neg1} == {"agent", "opponent"}


def test_minibg_trainer_smoke_with_perspective_env():
    base = MiniBGEnv(seed=0)
    sampler = RandomOpponentSampler(seed=2)
    env = AgentPerspectiveEnv(base, sampler, agent_first_probability=0.5)
    trainer = Trainer(env, RandomAgent(seed=1), opponent_sampler=sampler)
    trainer.train(total_steps=8)
    assert trainer.global_step == 8


def test_minibg_make_game_registered():
    from src.envs.minibg import NUM_ENV_ACTIONS

    env = make_game("minibg", seed=0)
    assert isinstance(env, MiniBGEnv)
    assert len(env.legal_actions_mask) == NUM_ENV_ACTIONS
