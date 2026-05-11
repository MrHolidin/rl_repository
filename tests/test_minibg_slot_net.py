import torch

import src.models  # noqa: F401 — triggers network registry
from src.envs.minibg.cards import make_minion
from src.envs.minibg.game import MiniBGGame
from src.envs.minibg.obs import (
    GLOBAL_DIM,
    SHOP_SIZE,
    SLOT_DIM,
    build_observation,
)
from src.models.minibg_slot_net import MiniBGSlotEncoderNet, _OBS_DIM


def test_minibg_slot_forward_shape():
    from src.envs.minibg import NUM_ENV_ACTIONS

    net = MiniBGSlotEncoderNet(
        num_actions=NUM_ENV_ACTIONS, slot_hidden=32, trunk_hidden=128, dueling=True
    )
    x = torch.randn(5, _OBS_DIM)
    q = net(x)
    assert q.shape == (5, NUM_ENV_ACTIONS)


def test_minibg_slot_noisy_forward_and_reset_noise():
    from src.envs.minibg import NUM_ENV_ACTIONS

    net = MiniBGSlotEncoderNet(
        num_actions=NUM_ENV_ACTIONS,
        slot_hidden=16,
        trunk_hidden=64,
        dueling=True,
        use_noisy=True,
        noisy_sigma=0.5,
    )
    x = torch.randn(3, _OBS_DIM)
    net.train()
    net.reset_noise()
    q0 = net(x)
    net.reset_noise()
    q1 = net(x)
    assert q0.shape == (3, NUM_ENV_ACTIONS)
    assert not torch.allclose(q0, q1)
    net.eval()
    qa = net(x)
    qb = net(x)
    assert torch.allclose(qa, qb)


def test_minibg_slot_distinguishes_slot_permutations():
    """Same regional contents in different slot order must produce
    different Q-values. With mean-pool this would silently pass; we want
    a permutation-sensitive encoder so that slot-indexed action heads can
    read which card sits in slot i.
    """
    from src.envs.minibg import NUM_ENV_ACTIONS

    g = MiniBGGame(seed=0)
    s = g.initial_state()
    # Force a deterministic, distinguishable shop (recruit / big_guy / None).
    s.players[0].shop = [
        make_minion("recruit"),
        make_minion("big_guy"),
        None,
    ]
    obs_a = build_observation(s, 0, 0.0, [])

    # Same minions, swapped slots.
    s.players[0].shop = [
        make_minion("big_guy"),
        make_minion("recruit"),
        None,
    ]
    obs_b = build_observation(s, 0, 0.0, [])

    # Sanity: regional aggregates (mean / sum) should be identical.
    shop_start = GLOBAL_DIM + 4 * SLOT_DIM  # globals + own_board (4 slots)
    shop_end = shop_start + SHOP_SIZE * SLOT_DIM
    region_a = obs_a[shop_start:shop_end].reshape(SHOP_SIZE, SLOT_DIM)
    region_b = obs_b[shop_start:shop_end].reshape(SHOP_SIZE, SLOT_DIM)
    assert (region_a.sum(axis=0) == region_b.sum(axis=0)).all()
    assert not (region_a == region_b).all()  # actually permuted

    net = MiniBGSlotEncoderNet(
        num_actions=NUM_ENV_ACTIONS, slot_hidden=16, trunk_hidden=64, dueling=True
    )
    net.eval()
    with torch.no_grad():
        q_a = net(torch.from_numpy(obs_a).unsqueeze(0))
        q_b = net(torch.from_numpy(obs_b).unsqueeze(0))

    # The two Q-vectors must NOT collapse to the same value.
    assert not torch.allclose(q_a, q_b, atol=1e-6)
