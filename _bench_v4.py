"""At iteration-0 minibatch-0 of fresh v4 agent, replay should match collection.
ratio = 1 → policy_loss = -mean(adv_normalized) ≈ 0. If not — bug."""
import numpy as np
import torch

import src.agents
from src.bg_catalog.patch_context import load_patch_context
from src.envs.bglike.actions import BOARD_SIZE
from src.envs.bglike.obs import OBS_DIM
from src.envs.minibg.structured_actions import StructAction, StructActionType
from src.agents.ppo_structured_minibg_agent import MiniBGPPOStructuredAgent
from src.models.bglike_structured_v4 import BGLikeStructuredV4

ctx = load_patch_context("data/bgcore/19_6_0_74257")
DEV = "cpu"

torch.manual_seed(0)
np.random.seed(0)
net = BGLikeStructuredV4(
    slot_hidden=64, state_dim=128, action_dim=64,
    region_conv2_kernel=1, card_emb_dim=16,
    entity_attention_layers=2, entity_attention_heads=4,
    entity_attention_ff_mult=2, entity_attention_init_scale=0.1,
    action_cross_attn_heads=4, action_cross_attn_ff_mult=2,
    action_cross_attn_init_scale=0.1,
    recurrent_hidden_dim=128,
    obs_layout="bglike", num_pool_indices=ctx.num_pool_indices,
).to(DEV)
# Freeze grads to make sure even if optimizer does something we can detect.
agent = MiniBGPPOStructuredAgent(
    observation_shape=(OBS_DIM,),
    observation_type="vector",
    num_actions=512,
    network=net,
    ppo_network_type="bglike_structured_v4",
    ppo_network_kwargs=dict(net.get_constructor_kwargs()),
    device=DEV,
    rollout_steps=8,
    ppo_epochs=1,
    minibatch_size=64,
)

# Build a small synthetic buffer with REAL log_probs from the same network
# (so old_log_prob = forward log_prob exactly). Then a single PPO update
# should produce ratio=1 on FIRST mb, policy_loss = -mean(adv_normalized).
N_SEQ, SEQ_LEN, LMAX = 2, 4, 6
H = 128
types = [StructActionType.ROLL, StructActionType.LEVEL_UP, StructActionType.COMPLETE_TURN]
def make_legal(n):
    return [StructAction(type=types[i % 3]) for i in range(n)]

# Synthesize obs + ACT through the agent (no env) to fill buffer with correct log_probs.
rng = np.random.default_rng(0)
class FakeEnv:
    def __init__(self):
        self.state = type("S", (), {})()
        self.state.players = [type("P", (), {"board": []})() for _ in range(8)]
        self.state.current_player_index = 0

env = FakeEnv()
agent.train()

buf = agent.rollout_buffer
for ep in range(N_SEQ):
    agent._hidden_by_seat.clear()  # mimic episode reset on host
    for t in range(SEQ_LEN):
        seat = ep % 2
        env.state.current_player_index = seat
        obs = rng.standard_normal(OBS_DIM, dtype=np.float32)
        is_complete = (t == SEQ_LEN - 1)
        legal = make_legal(LMAX)
        if is_complete:
            legal[-1] = StructAction(type=StructActionType.COMPLETE_TURN)
            # Force the chosen action to be COMPLETE_TURN, indexed last.
        # Sample via agent.act_structured (this populates cache too).
        chosen, board_perm, idx = agent.act_structured(obs, legal, env, deterministic=False)
        # Build a fake transition for observe.
        from src.envs.base import StepResult
        next_legal = make_legal(LMAX)
        info = {
            "minibg_struct_legal": legal,
            "minibg_struct_next_legal": next_legal,
            "acting_seat": seat,
            "combat_advanced": is_complete,
        }
        tr = type("T", (), {})()
        tr.obs = obs
        tr.action = idx
        tr.reward = float(rng.normal())
        tr.next_obs = rng.standard_normal(OBS_DIM, dtype=np.float32)
        tr.terminated = bool(is_complete and ep == N_SEQ - 1)
        tr.truncated = False
        tr.info = info
        agent.observe(tr)

print(f"buffer size: {len(buf)}")
print(f"log_probs (stored): {[f'{x:.4f}' for x in buf.log_probs]}")

# Now do one PPO update. With unchanged model and same h chain, on first mb
# the recomputed log_prob should equal stored log_prob. ratio = 1.
# After advantage normalization to mean 0, policy_loss should be near 0.
m = agent.update()
print(f"\npolicy_loss after 1 update: {m.get('policy_loss')}")
print(f"value_loss: {m.get('value_loss')}")
print(f"entropy: {m.get('entropy')}")
print(f"approx_kl (should be ~0 on first iter): {m.get('approx_kl')}")
print(f"clip_frac (should be 0): {m.get('clip_frac')}")
