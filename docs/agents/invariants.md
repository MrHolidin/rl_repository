# Agent invariants and conventions

Read this before implementing or extending agents. These rules keep training and evaluation consistent.

---

## Observation

- **Perspective:** Observation is always from the **current player's** point of view (the player whose turn it is).
- **BoardChannels layout:**  
  - `obs[0]` — current player's pieces  
  - `obs[1]` — opponent's pieces  
  - `obs[2]` — legacy / optional channel (current-player token indicator). **Agents must not use it to determine player sign.** It is redundant and may be removed; treat it as debug/optional only.
- **Invariant:** In this representation, "current player" is effectively +1 when reconstructing the board. Agents that parse the board (e.g. heuristics, value functions) should set: my pieces from `obs[0]`, opponent from `obs[1]`, and `current_player = 1`.
- **next_obs:** In a transition, `next_obs` is the state **where the agent is to move again** (or a terminal state). If the environment returned `next_obs` right after the agent's move (before the opponent's response), the trainer must finish the opponent's move before recording the transition; otherwise this invariant is broken.

---

## Rewards

- **Agent-centric:** Reward is always in the **agent's** coordinate system, regardless of which token the agent plays.
- **Timing:** Reward for the agent's action is credited **after the opponent's move** (except when the game ends on the agent's move). If the game terminates on the opponent's move, the reward is attributed to the **previous** agent action.

---

## legal_mask and actions

- **legal_mask:** Boolean array of shape `(num_actions,)`; `True` means the action is legal. **Type:** `np.ndarray` with dtype **bool** (not float 0/1).
- In a transition: `legal_mask` applies to `obs`; `next_legal_mask` applies to `next_obs`.
- **Terminal:** When the episode is done, `next_legal_mask` is either an array of the same shape as `legal_mask` with **all False**, or `None` if `legal_mask` was not provided.

---

## Transition and done

- **Fields:** `obs`, `action`, `reward`, `next_obs`, `terminated`, `truncated`, `info`, `legal_mask`, `next_legal_mask`.
- **done:** `done = terminated or truncated`.
- **TD bootstrap:** Use **not terminated** (not `not done`) when deciding whether to bootstrap, if `truncated` means "step limit" (episode cut by length). If `truncated` means "timeout = draw/terminal", treat it like a terminal: no bootstrap.  
  The current DQN may use `not done`; align with the rule above when needed.

---

## RNG

- Agents that take a `seed` must use **only a local RNG** (e.g. `random.Random(seed)` or `np.random.default_rng(seed)`). Do **not** call `random.seed(seed)` or `np.random.seed(seed)` in the constructor — that breaks training when opponents are created with different seeds each episode.
- Do **not** use global `np.random` (e.g. without `default_rng`) inside `act()`; use only the agent's local RNG so behaviour does not leak through third-party code.
- Converting `legal_mask` to indices (e.g. `np.flatnonzero(legal_mask)`) is deterministic; the only source of randomness in action choice should be the agent's own RNG.

---

## BaseAgent contract

- **act(obs, legal_mask=None, deterministic=False) -> int** — choose an action. Prefer calling `act` with `legal_mask`; `select_action(obs, legal_actions)` is a backward-compat wrapper.
- **observe(transition) -> dict** — optional; may be a no-op. Returned metrics are for **logging only** and must not affect training (training step is separate, e.g. in `update()` or inside the trainer).
- **save(path)**, **load(path, **kwargs)** — persist / restore agent state.
- **train()** / **eval()** — switch mode (e.g. epsilon, dropout).
