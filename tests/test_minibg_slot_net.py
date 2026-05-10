import torch

import src.models  # noqa: F401 — triggers network registry
from src.models.minibg_slot_net import MiniBGSlotEncoderNet, _OBS_DIM


def test_minibg_slot_forward_shape():
    net = MiniBGSlotEncoderNet(num_actions=33, slot_hidden=32, trunk_hidden=128, dueling=True)
    x = torch.randn(5, _OBS_DIM)
    q = net(x)
    assert q.shape == (5, 33)
