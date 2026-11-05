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


def test_full_search_prob_is_the_live_worker_default() -> None:
    # SA1-05: the old FULL_P=0.35 constant was dead (zero references) and
    # divergent from the live 0.25 worker default. FULL_SEARCH_PROB now pins
    # the live value and the worker classes derive their defaults from it.
    import inspect

    from python.selfplay.worker import SelfPlayGame, SelfPlayWorker

    assert constants.FULL_SEARCH_PROB == 0.25
    assert constants.RESIGN_THRESHOLD == -0.95
    for cls in (SelfPlayGame, SelfPlayWorker):
        params = inspect.signature(cls.__init__).parameters
        assert params["full_search_prob"].default == constants.FULL_SEARCH_PROB
        assert params["resign_threshold"].default == constants.RESIGN_THRESHOLD


def test_dead_divergent_full_p_is_not_reintroduced() -> None:
    # SA1-05 regression pin: FULL_P (0.35) diverged from the live 0.25
    # default while being referenced by nothing. If this fails, the dead
    # name is back — route any new constant through FULL_SEARCH_PROB.
    assert not hasattr(constants, "FULL_P")


def test_sim_counts_match_config_registry_defaults() -> None:
    # SA1-05: FULL_SIMS / FAST_SIMS were defined-but-unreferenced; they are
    # now the source of the canonical selfplay config defaults (800 / 150).
    from python.cli.config import CONFIG_REGISTRY

    assert constants.FULL_SIMS == 800
    assert constants.FAST_SIMS == 150
    assert CONFIG_REGISTRY["selfplay.full_search_sims"].default == constants.FULL_SIMS
    assert CONFIG_REGISTRY["selfplay.fast_search_sims"].default == constants.FAST_SIMS
