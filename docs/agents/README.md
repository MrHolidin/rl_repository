# Agents

- **Available agents:** `random`, `heuristic`, `smart_heuristic`, `qlearning`, `dqn`. (PPO is deprecated and not supported.)
- **Creating an agent:** `make_agent(agent_id, **kwargs)`; `kwargs` usually come from config (e.g. `configs/agent/<id>.yaml`) plus env/action_space when starting training.
- **Before implementing or relying on agent behaviour,** read [Invariants and conventions](invariants.md).

## DQN

- Preferred way to construct: pass a ready `network` in kwargs; training hyperparameters (learning rate, gamma, epsilon, batch size, replay size, target update, tau, etc.) go in config.
- Loading a checkpoint: `DQNAgent.load(path, device=...)`.
