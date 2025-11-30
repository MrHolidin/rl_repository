"""Tests for observation builders and model factory."""

from __future__ import annotations

import numpy as np
import torch

from src.features.observation_builder import BoardChannels
from src.models.dueling_utils import dueling_aggregate
from src.models.q_network_factory import build_q_network


def test_board_channels_basic():
    builder = BoardChannels(board_shape=(6, 7))
    board = np.zeros((6, 7), dtype=np.int8)
    board[5, 0] = 1
    board[4, 3] = -1

    raw_state = {
        "board": board,
        "current_player_token": 1,
        "last_move": None,
        "legal_actions_mask": np.ones(7, dtype=bool),
    }

    obs = builder.build(raw_state)
    assert obs.shape == builder.observation_shape
    assert obs.dtype == np.float32
    # Current player's piece channel should mark player 1 token
    assert obs[0, 5, 0] == 1.0
    # Opponent channel should mark player -1 token
    assert obs[1, 4, 3] == 1.0
    # Turn channel should be full of ones because current player token is 1
    assert np.all(obs[2] == 1.0)


def test_board_channels_with_extras():
    builder = BoardChannels(board_shape=(6, 7), include_last_move=True, include_legal_moves=True)
    board = np.zeros((6, 7), dtype=np.int8)
    board[5, 0] = -1

    last_move = (5, 0)
    legal_mask = np.array([False, True, True, False, True, True, True], dtype=bool)

    raw_state = {
        "board": board,
        "current_player_token": -1,
        "last_move": last_move,
        "legal_actions_mask": legal_mask,
    }

    obs = builder.build(raw_state)
    assert obs.shape == builder.observation_shape
    # Turn channel should be zeros because current player token is -1
    assert np.all(obs[2] == 0.0)
    # Last move channel should highlight the last move
    assert obs[3, last_move[0], last_move[1]] == 1.0
    # Legal moves channel should contain mask on the top row
    assert np.all(obs[4, 0, :] == legal_mask.astype(np.float32))


def test_build_q_network_board_and_vector():
    board_shape = (3, 6, 7)
    num_actions = 7
    board_model = build_q_network(
        observation_type="board",
        observation_shape=board_shape,
        num_actions=num_actions,
        dueling=False,
        model_config={
            "conv_layers": [
                {"out_channels": 32, "kernel_size": 3, "stride": 1, "padding": 1},
            ],
            "fc_layers": [128],
        },
    )
    sample_input = torch.zeros((2, *board_shape), dtype=torch.float32)
    output = board_model(sample_input)
    assert output.shape == (2, num_actions)

    vector_shape = (20,)
    vector_model = build_q_network(
        observation_type="vector",
        observation_shape=vector_shape,
        num_actions=5,
        dueling=True,
        model_config={"fc_layers": [64, 64]},
    )
    vector_input = torch.zeros((3, *vector_shape), dtype=torch.float32)
    vector_output = vector_model(vector_input)
    assert vector_output.shape == (3, 5)


def test_dueling_aggregate_respects_mask():
    value = torch.tensor([[1.0]])
    advantage = torch.tensor([[1.0, -1.0, 2.0]])
    legal_mask = torch.tensor([[True, False, True]])

    q_values = dueling_aggregate(value, advantage, legal_mask)
    expected = torch.tensor([[0.5, -0.5, 1.5]])
    assert torch.allclose(q_values, expected)

    all_illegal_mask = torch.zeros_like(legal_mask, dtype=torch.bool)
    q_all_illegal = dueling_aggregate(value, advantage, all_illegal_mask)
    assert torch.allclose(q_all_illegal, torch.full_like(advantage, 1.0))

