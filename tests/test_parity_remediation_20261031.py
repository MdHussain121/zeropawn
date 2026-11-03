"""Regression tests for the C++/Python parity remediation (2026-10-31).

Covers every crash-class and validation defect fixed by the A-09 remediation:

* F-01: ``Connect4.cell`` bounds validation (was: OOB read / access violation).
* F-02: ``ChessGame.perft`` depth validation (was: stack overflow, process death).
* F-05: ``ChessGame.apply_action`` legality validation (was: silent state
  corruption from in-range illegal actions, e.g. a pawn teleporting a2->a5).
* F-11: ``MCTSConfig`` semantic validation (was: negative c_puct etc. accepted).
* F-12: ``MCTS.search_batched`` exact evaluator-output validation (was: silent
  truncation/zero-fill) and int-like batch sizes (np.int64 was rejected).
* F-04: GIL discipline in the ``search`` binding — stability stress test with
  shared evaluator/abort objects across concurrent searches.
* REM-01/REM-02: Connect4 ``string_to_action`` / ``apply_action`` validation.

Crash-class cases are additionally exercised in isolated subprocesses so a
regression to process death is observable as a nonzero exit instead of killing
the test runner.
"""

import os
import subprocess
import sys
import threading
from pathlib import Path

import numpy as np
import pytest

import zeropawn_cpp


def uniform_eval(game):
    n = game.num_actions()
    return ([1.0 / n] * n, 0.0)


def uniform_batch_planes(batch_planes):
    n = int(batch_planes.shape[0])
    return (
        np.full((n, 4672), 1.0 / 4672, dtype=np.float32),
        np.zeros((n,), dtype=np.float32),
    )


def _run_in_subprocess(code: str) -> subprocess.CompletedProcess:
    """Run a probe in a fresh interpreter importing the same pyd."""
    pyd_dir = Path(zeropawn_cpp.__file__).parent
    env = dict(os.environ)
    env["PYTHONPATH"] = str(pyd_dir) + os.pathsep + env.get("PYTHONPATH", "")
    return subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=300,
        env=env,
    )


# ==============================================================================
# F-01: Connect4.cell bounds validation
# ==============================================================================


class TestConnect4CellBounds:
    @pytest.mark.parametrize(
        "r,c", [(6, 0), (-1, 0), (0, 7), (0, -1), (10**7, 0), (-1, -1), (100, 100)]
    )
    def test_out_of_range_raises_valueerror(self, r, c):
        c4 = zeropawn_cpp.Connect4()
        with pytest.raises(ValueError):
            c4.cell(r, c)

    def test_valid_cells_unchanged(self):
        c4 = zeropawn_cpp.Connect4()
        assert c4.cell(0, 0) == 0
        c4.apply_action(3)  # legal: column 3 not full
        assert c4.cell(0, 3) == 1
        assert c4.cell(5, 6) == 0

    def test_huge_coordinate_no_access_violation_subprocess(self):
        # The audit's exact repro: cell(10**7, 0) killed the interpreter with
        # an access violation (exit 3221225477). It must now raise ValueError.
        p = _run_in_subprocess(
            "import zeropawn_cpp as z\n"
            "c4 = z.Connect4()\n"
            "try:\n"
            "    c4.cell(10**7, 0)\n"
            "    raise SystemExit(2)\n"
            "except ValueError:\n"
            "    raise SystemExit(0)\n"
        )
        assert p.returncode == 0, p.stderr


# ==============================================================================
# F-02: perft depth validation
# ==============================================================================


class TestPerftDepthLimits:
    @pytest.mark.parametrize("depth", [-1, -100, 33, 20000, 10**9])
    def test_invalid_depth_raises_valueerror(self, depth):
        g = zeropawn_cpp.ChessGame()
        with pytest.raises(ValueError):
            g.perft(depth)

    def test_perft_zero_convention(self):
        assert zeropawn_cpp.ChessGame().perft(0) == 1

    def test_perft_anchors_unaffected(self):
        g = zeropawn_cpp.ChessGame()
        assert g.perft(4) == 197281
        kiwi = zeropawn_cpp.ChessGame(
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"
        )
        assert kiwi.perft(3) == 97862

    def test_deep_perft_no_stack_overflow_subprocess(self):
        # perft(20000) used to die with STATUS_STACK_OVERFLOW (0xC00000FD).
        p = _run_in_subprocess(
            "import zeropawn_cpp as z\n"
            "g = z.ChessGame()\n"
            "assert g.perft(6) == 119060324\n"
            "try:\n"
            "    g.perft(20000)\n"
            "    raise SystemExit(2)\n"
            "except ValueError:\n"
            "    raise SystemExit(0)\n"
        )
        assert p.returncode == 0, p.stderr


# ==============================================================================
# F-05: apply_action legality validation
# ==============================================================================


class TestApplyActionLegality:
    def test_illegal_in_range_action_raises_and_preserves_state(self):
        g = zeropawn_cpp.ChessGame()
        idx = g.string_to_action("a2a5")  # decodable, in range, illegal
        assert idx >= 0
        fen_before, hash_before = g.get_fen(), g.get_hash()
        with pytest.raises(ValueError):
            g.apply_action(idx)
        assert g.get_fen() == fen_before
        assert g.get_hash() == hash_before
        assert g.history_len() == 0

    def test_garbage_indices_still_raise(self):
        g = zeropawn_cpp.ChessGame()
        for bad in (99999, -1, 4672):
            with pytest.raises(ValueError):
                g.apply_action(bad)

    def test_all_legal_actions_apply(self):
        g = zeropawn_cpp.ChessGame()
        for action in g.legal_actions():
            clone = g.clone()
            clone.apply_action(action)
            assert clone.history_len() == 1

    def test_special_moves_apply_via_index(self):
        # Castling, en passant, and promotion must remain reachable by index.
        castle = zeropawn_cpp.ChessGame("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1")
        for uci in ("e1g1", "e1c1"):
            idx = castle.string_to_action(uci)
            assert idx >= 0
            clone = castle.clone()
            clone.apply_action(idx)
            assert clone.history_len() == 1

        ep = zeropawn_cpp.ChessGame("rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3")
        idx = ep.string_to_action("e5f6")
        assert idx >= 0
        ep.apply_action(idx)
        assert ep.history_len() == 1

        promo = zeropawn_cpp.ChessGame("8/P7/8/8/8/8/k6K/8 w - - 0 1")
        for uci in ("a7a8q", "a7a8n", "a7a8r", "a7a8b"):
            idx = promo.string_to_action(uci)
            assert idx >= 0
            clone = promo.clone()
            clone.apply_action(idx)
            assert clone.history_len() == 1


# ==============================================================================
# F-11: MCTSConfig semantic validation
# ==============================================================================


class TestMCTSConfigValidation:
    @pytest.mark.parametrize(
        "attr,value",
        [
            ("c_puct", -1.0),
            ("dirichlet_alpha", 0.0),
            ("dirichlet_alpha", -0.3),
            ("dirichlet_weight", -0.1),
            ("fpu_reduction", -0.5),
            ("virtual_loss", -1),
            ("batch_size", 0),
            ("batch_size", -3),
            ("max_nodes", 0),
        ],
    )
    def test_invalid_config_raises(self, attr, value):
        cfg = zeropawn_cpp.MCTSConfig()
        setattr(cfg, attr, value)
        with pytest.raises(ValueError):
            zeropawn_cpp.MCTS(cfg)

    def test_valid_config_constructs(self):
        cfg = zeropawn_cpp.MCTSConfig()
        cfg.c_puct = 0.0  # zero allowed
        cfg.batch_size = 64
        cfg.max_nodes = 1
        mcts = zeropawn_cpp.MCTS(cfg)
        assert mcts.cache_size() == 0

    def test_defaults_still_construct(self):
        zeropawn_cpp.MCTS(zeropawn_cpp.MCTSConfig())


# ==============================================================================
# F-12: search_batched evaluator-output validation + int-like batch sizes
# ==============================================================================


class TestSearchBatchedValidation:
    def _game_and_mcts(self):
        return zeropawn_cpp.ChessGame(), zeropawn_cpp.MCTS()

    def test_short_policy_raises(self):
        g, m = self._game_and_mcts()

        def bad(planes):
            n = planes.shape[0]
            return (np.zeros((n, 100), dtype=np.float32), np.zeros((n,), dtype=np.float32))

        with pytest.raises(RuntimeError):
            m.search_batched(g, 8, 4, bad)

    def test_short_values_raises(self):
        g, m = self._game_and_mcts()

        def bad(planes):
            n = planes.shape[0]
            return (
                np.full((n, 4672), 1.0 / 4672, dtype=np.float32),
                np.zeros((1,), dtype=np.float32),
            )

        with pytest.raises(RuntimeError):
            m.search_batched(g, 8, 4, bad)

    def test_oversized_output_raises(self):
        g, m = self._game_and_mcts()

        def bad(planes):
            n = planes.shape[0]
            return (
                np.full((n + 1, 4672), 1.0 / 4672, dtype=np.float32),
                np.zeros((n + 1,), dtype=np.float32),
            )

        with pytest.raises(RuntimeError):
            m.search_batched(g, 8, 4, bad)

    @pytest.mark.parametrize("batch_size", [2, np.int64(2), np.int32(2)])
    def test_int_like_batch_sizes_accepted(self, batch_size):
        g, m = self._game_and_mcts()
        res = m.search_batched(g, 4, batch_size, uniform_batch_planes)
        assert res.best_action >= 0
        assert res.new_simulations == 4

    def test_search_evaluator_validation_unaffected(self):
        g, m = self._game_and_mcts()
        with pytest.raises(RuntimeError):
            m.search(g, 8, lambda game: ([0.0] * 10, 0.0))


# ==============================================================================
# REM-01 / REM-02: Connect4 input validation
# ==============================================================================


class TestConnect4Validation:
    @pytest.mark.parametrize(
        "s,expected",
        [("0", 0), ("6", 6), ("9", -1), ("abc", -1), ("", -1), ("10", -1), ("x", -1)],
    )
    def test_string_to_action_validation(self, s, expected):
        assert zeropawn_cpp.Connect4().string_to_action(s) == expected

    def test_full_column_raises(self):
        c4 = zeropawn_cpp.Connect4()
        for _ in range(6):
            c4.apply_action(0)  # alternating owners: no vertical win
        hash_before = c4.get_hash()
        with pytest.raises(ValueError):
            c4.apply_action(0)
        assert c4.get_hash() == hash_before

    def test_out_of_range_column_raises(self):
        c4 = zeropawn_cpp.Connect4()
        for bad in (-1, 7, 100):
            with pytest.raises(ValueError):
                c4.apply_action(bad)

    def test_finished_game_raises(self):
        c4 = zeropawn_cpp.Connect4()
        # P1 wins vertically in column 0.
        for i in range(6):
            if i < 3:
                c4.apply_action(0)
                c4.apply_action(1)
        c4.apply_action(0)  # 4th P1 piece: win
        assert c4.is_terminal()
        with pytest.raises(ValueError):
            c4.apply_action(6)

    def test_mcts_training_path_unaffected(self):
        # train_connect4 applies legal_actions()/best_action only; a full
        # 42-move game must still play out without raising.
        c4 = zeropawn_cpp.Connect4()
        while not c4.is_terminal():
            c4.apply_action(c4.legal_actions()[0])
        assert c4.is_terminal()


# ==============================================================================
# F-04: GIL discipline stress — shared callbacks across concurrent searches
# ==============================================================================


class TestSearchGILDiscipline:
    def test_concurrent_searches_share_evaluator_and_abort(self):
        # Four threads × 20 searches, all sharing ONE evaluator function and
        # ONE abort_fn object: exercises the incref/decref paths of the search
        # binding under contention. The F-04 fix performs every py::object
        # refcount operation with the GIL held.
        shared_eval = uniform_eval
        shared_abort = lambda: False  # noqa: E731
        errors = []
        barrier = threading.Barrier(4)

        def worker():
            try:
                barrier.wait(timeout=30)
                for _ in range(20):
                    mcts = zeropawn_cpp.MCTS()
                    mcts.set_seed(1)
                    res = mcts.search(zeropawn_cpp.ChessGame(), 64, shared_eval, None, shared_abort)
                    assert res.best_action >= 0
            except Exception as e:  # pragma: no cover - failure path
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=120)
        assert not any(t.is_alive() for t in threads)
        assert errors == []

    def test_repeated_create_search_destroy_stable(self):
        allocated_before = zeropawn_cpp.MCTS.total_allocated_nodes()
        freed_before = zeropawn_cpp.MCTS.total_freed_nodes()
        for _ in range(50):
            mcts = zeropawn_cpp.MCTS()
            mcts.search(zeropawn_cpp.ChessGame(), 16, uniform_eval)
            del mcts
        assert (
            zeropawn_cpp.MCTS.total_allocated_nodes() - allocated_before
            == zeropawn_cpp.MCTS.total_freed_nodes() - freed_before
        ), "every node allocated across create/search/destroy cycles must be freed"
