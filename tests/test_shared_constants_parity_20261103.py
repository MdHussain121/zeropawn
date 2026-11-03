"""Shared-constants parity test (S7-01 remediation, 2026-11-03).

``python/constants.py`` is the Python-side contract mirroring
``cpp/include/zeropawn/encoder.hpp``. Before this test existed the module had
zero importers and every consumer hardcoded 19/1216/4672, so drift between the
Python and C++ constants was undetectable. This test pins the constants to the
live C++ binding surfaces so numeric drift fails loudly:

- ``NUM_ACTIONS`` == ``zeropawn_cpp.ChessGame().num_actions()``
- ``STATE_SIZE``  == ``zeropawn_cpp.ChessGame().state_size()``
- ``NUM_PLANES`` / ``PLANES_SHAPE`` == the real ``zeropawn_cpp.encode_state``
  output shape
- ``ChessNet`` (whose defaults are now sourced from ``python.constants``)
  produces exactly ``NUM_ACTIONS`` policy logits and a 3-way WDL head.
"""

from __future__ import annotations

import numpy as np
import torch

import python.constants as constants
import zeropawn_cpp
from python.model.chess_net import ChessNet


def _fresh_game() -> "zeropawn_cpp.ChessGame":
    return zeropawn_cpp.ChessGame()


def test_num_actions_matches_cpp_game() -> None:
    assert constants.NUM_ACTIONS == _fresh_game().num_actions()
    # Pin the literal too: a change to either side must fail this test, not
    # silently pass because both sides drifted together.
    assert constants.NUM_ACTIONS == 4672


def test_state_size_matches_cpp_game() -> None:
    assert constants.STATE_SIZE == _fresh_game().state_size()
    assert constants.STATE_SIZE == constants.NUM_PLANES * constants.NUM_SQUARES


def test_encoder_output_shape_matches_constants() -> None:
    planes = zeropawn_cpp.encode_state(_fresh_game())
    assert planes.dtype == np.float32
    assert planes.shape == constants.PLANES_SHAPE
    assert planes.shape == (constants.NUM_PLANES, constants.BOARD_SIZE, constants.BOARD_SIZE)
    assert planes.size == constants.STATE_SIZE


def test_chessnet_heads_match_constants() -> None:
    model = ChessNet()
    t = torch.from_numpy(zeropawn_cpp.encode_state(_fresh_game())).unsqueeze(0)
    with torch.no_grad():
        policy_logits, wdl = model(t)
    assert policy_logits.shape[-1] == constants.NUM_ACTIONS
    assert wdl.shape[-1] == 3


def test_wdl_convention_is_internally_consistent() -> None:
    # Value-head convention documented in constants.py: win > draw > loss on
    # the scalar axis, three WDL labels.
    assert constants.WDL_WIN > constants.WDL_DRAW > constants.WDL_LOSS
    assert len(constants.WDL_LABELS) == 3
