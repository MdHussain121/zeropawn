import pytest
import numpy as np
import zeropawn_cpp


def test_connect4_basics():
    c4 = zeropawn_cpp.Connect4()
    assert len(c4.legal_actions()) == 7
    assert not c4.is_terminal()
    assert c4.current_player() == 1
    assert c4.winner() == 0

    state = c4.encode_state()
    assert state.shape == (3, 6, 7)
    assert state[0].sum() == 0.0  # No player 1 pieces yet
    assert state[1].sum() == 0.0  # No player 2 pieces yet
    assert state[2].sum() == 42.0  # Turn plane


def test_connect4_vertical_win():
    c4 = zeropawn_cpp.Connect4()
    # P1: 0, P2: 1, P1: 0, P2: 1, P1: 0, P2: 1, P1: 0 (win!)
    for _ in range(3):
        c4.apply_action(0)
        c4.apply_action(1)
    assert not c4.is_terminal()
    c4.apply_action(0)  # P1 drops 4th piece in col 0
    assert c4.is_terminal()
    assert c4.winner() == 1
    assert c4.terminal_value() == -1.0  # Current player (P2) lost


def test_connect4_mcts_tactical():
    c4 = zeropawn_cpp.Connect4()
    for _ in range(3):
        c4.apply_action(0)  # P1
        c4.apply_action(1)  # P2

    # P1 to move, col 0 is immediate win
    def eval_fn(game):
        return [1.0 / 7.0] * 7, 0.0

    mcts = zeropawn_cpp.MCTS()
    res = mcts.search(c4, 50, eval_fn)
    assert res.best_action == 0
