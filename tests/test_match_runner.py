"""Unit tests for MatchRunner and StockfishPlayer."""

from __future__ import annotations

import inspect
from pathlib import Path
import pytest
import torch

import zeropawn_cpp
from python.model.chess_net import ChessNet
from python.eval.match_runner import (
    StockfishPlayer,
    MatchRunner,
    check_search_mode,
    require_serial_search,
    engine_outcome,
    score_board_results,
)


def test_stockfish_player_basic():
    print("\n  [Op 1/2] Connecting to Stockfish engine (nodes=100)...")
    sf = StockfishPlayer(stockfish_path="tools/stockfish.exe", nodes=100)
    try:
        start_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        print("  [Op 2/2] Querying best move for standard starting position...")
        move = sf.get_move(start_fen)
        assert move != ""
        assert len(move) in (4, 5)
        print(f"    [OK] Stockfish responded with valid move: '{move}'")
    finally:
        sf.close()


def test_match_runner_single_pair():
    print(
        "\n  [Op 1/2] Initializing MatchRunner (ChessNet dummy model, Stockfish at 50 nodes, 10 MCTS sims)..."
    )
    model = ChessNet(num_blocks=2, num_filters=32)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    runner = MatchRunner(
        model=model,
        device=device,
        stockfish_path="tools/stockfish.exe",
        stockfish_nodes=50,
        mcts_sims=10,
        openings_path="data/openings/eval_openings_50.epd",
    )
    try:
        # Run match with 1 opening (2 games total: 1 White, 1 Black)
        print("  [Op 2/2] Executing 1 opening pair match (2 games: White & Black)...")
        results = runner.run_match(num_openings=1)
        assert results["total_games"] == 2
        assert results["engine_wins"] + results["engine_draws"] + results["engine_losses"] == 2
        assert 0.0 <= results["engine_score"] <= 1.0
        assert 0.0 <= results["draw_rate"] <= 1.0
        print(
            f"    [OK] Match Completed: {results['total_games']} games | Wins: {results['engine_wins']} | Draws: {results['engine_draws']} | Losses: {results['engine_losses']} | Score: {results['engine_score']:.2f}"
        )
    finally:
        runner.close()


def test_elo_diff_ci_shape_and_edges():
    from python.cli.commands_match import _elo_diff_ci

    mid = _elo_diff_ci(3, 4, 3)
    assert "95% CI" in mid and "N=10" in mid
    # Even score centers near zero
    assert "+0.0" in _elo_diff_ci(1, 2, 1) or "-0.0" in _elo_diff_ci(1, 2, 1)
    # Edges never invent numbers
    assert _elo_diff_ci(0, 0, 2).startswith("n/a")
    assert _elo_diff_ci(2, 0, 0).startswith("n/a")
    assert _elo_diff_ci(0, 0, 0).startswith("n/a")


# ---------------------------------------------------------------------------
# Stage 1 Item 3b: board-relative results (no Stockfish binary / model needed)
# ---------------------------------------------------------------------------

WHITE_MATED_FEN = "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 0 1"
BLACK_MATED_FEN = "r1bqkbnr/pppp1Qpp/2n5/4p3/2B1P3/8/PPPP1PPP/RNB1K1NR b KQkq - 0 1"
STALEMATE_FEN = "7k/5Q2/6K1/8/8/8/8/8 b - - 0 1"


class _FakeSearchResult:
    def __init__(self, best_action: int, root_value: float) -> None:
        self.best_action = best_action
        self.root_value = root_value


class _FakeMCTS:
    """Stub MCTS: scripted root value; best action picked by callback."""

    def __init__(self, root_value: float = 0.0, move_fn=None) -> None:
        self.root_value = root_value
        self.move_fn = move_fn
        self.seen_sides: list[bool] = []

    def clear(self) -> None:
        pass

    def reset_root(self) -> None:
        pass

    def search(self, game, sims, cb, batch_cb=None):
        assert callable(cb), "eval callback must be callable"
        if batch_cb is not None:
            assert callable(batch_cb), "batch eval callback must be callable"
        self.seen_sides.append(bool(game.side_to_move()))
        if self.move_fn is not None:
            best = self.move_fn(game)
        else:
            legal = game.legal_actions()
            best = legal[0]
        return _FakeSearchResult(best, self.root_value)


class _FakeStockfish:
    def __init__(self, move_fn) -> None:
        self.move_fn = move_fn

    def reset_game(self) -> None:
        pass

    def get_move(self, fen: str) -> str:
        return self.move_fn(fen)


class _FakeCtx:
    def __init__(self, mcts, stockfish) -> None:
        self.mcts = mcts
        self.stockfish = stockfish


def _bare_runner(mcts_sims: int = 10) -> MatchRunner:
    runner = MatchRunner.__new__(MatchRunner)
    runner.mcts_sims = mcts_sims
    runner.search_mode = "serial"
    return runner


def _legal_reply(fen: str) -> str:
    import chess

    moves = list(chess.Board(fen).legal_moves)
    assert moves, f"no legal moves in {fen}"
    return moves[0].uci()


def test_engine_outcome_table():
    assert engine_outcome("1-0", True) == "win"
    assert engine_outcome("1-0", False) == "loss"
    assert engine_outcome("0-1", True) == "loss"
    assert engine_outcome("0-1", False) == "win"
    assert engine_outcome("1/2-1/2", True) == "draw"
    assert engine_outcome("1/2-1/2", False) == "draw"


def test_score_board_results_split_colors():
    tally = score_board_results(["1-0", "0-1", "1/2-1/2"], ["1-0", "0-1"])
    # As White: 1W/1D/1L; as Black: "1-0" is a loss, "0-1" is a win.
    assert tally == {"engine_wins": 2, "engine_draws": 1, "engine_losses": 2}


@pytest.mark.parametrize(
    ("fen", "engine_is_white", "expected"),
    [
        (WHITE_MATED_FEN, True, "0-1"),  # engine (White) mated
        (WHITE_MATED_FEN, False, "0-1"),  # board-relative: same string
        (BLACK_MATED_FEN, True, "1-0"),  # board-relative: Black mated
        (BLACK_MATED_FEN, False, "1-0"),  # engine (Black) mated
        (STALEMATE_FEN, True, "1/2-1/2"),
        (STALEMATE_FEN, False, "1/2-1/2"),
    ],
)
def test_worker_terminal_results_board_relative(fen, engine_is_white, expected):
    runner = _bare_runner()
    ctx = _FakeCtx(_FakeMCTS(), _FakeStockfish(_legal_reply))
    assert runner._play_game_worker(fen, engine_is_white, ctx) == expected


@pytest.mark.parametrize(
    ("engine_is_white", "expected"),
    [(True, "0-1"), (False, "1-0")],
)
def test_worker_resign_adjudication_board_relative(engine_is_white, expected):
    # Engine losing on its own turns 4x -> engine forfeits, board-relative.
    runner = _bare_runner()
    ctx = _FakeCtx(_FakeMCTS(root_value=-1.0), _FakeStockfish(_legal_reply))
    start = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    assert runner._play_game_worker(start, engine_is_white, ctx, max_plies=150) == expected


@pytest.mark.parametrize(
    ("engine_is_white", "expected"),
    [(True, "1-0"), (False, "0-1")],
)
def test_worker_winning_adjudication_board_relative(engine_is_white, expected):
    runner = _bare_runner()
    ctx = _FakeCtx(_FakeMCTS(root_value=1.0), _FakeStockfish(_legal_reply))
    start = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    assert runner._play_game_worker(start, engine_is_white, ctx, max_plies=150) == expected


def test_worker_stockfish_illegal_move_forfeit_board_relative():
    # Stockfish to move replies illegally -> mover (NOT engine) forfeits.
    runner = _bare_runner()
    start = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    ctx = _FakeCtx(_FakeMCTS(), _FakeStockfish(lambda fen: "0000"))
    # Engine White: stockfish (Black) never moves; engine moves e2e4-like first
    # legal move, then stockfish replies illegally on its turn.
    assert runner._play_game_worker(start, True, ctx) == "1-0"
    assert runner._play_game_worker(start, False, ctx) == "0-1"


def test_worker_engine_illegal_move_forfeit_board_relative():
    # Engine to move returns an illegal action -> engine forfeits.
    runner = _bare_runner()
    fen = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"  # after 1.e4
    probe = zeropawn_cpp.ChessGame(fen)
    illegal_idx = zeropawn_cpp.move_to_index(probe, "e2e4")
    assert not probe.is_legal("e2e4")
    ctx = _FakeCtx(_FakeMCTS(move_fn=lambda game: illegal_idx), _FakeStockfish(_legal_reply))
    # Engine Black to move plays illegally -> Black loses -> White wins -> "1-0".
    assert runner._play_game_worker(fen, False, ctx) == "1-0"


@pytest.mark.parametrize("engine_is_white", [True, False])
def test_worker_engine_moves_own_color(engine_is_white):
    """Regression: side_to_move() is bool; engine must move its own pieces."""
    runner = _bare_runner()
    mcts = _FakeMCTS()
    ctx = _FakeCtx(mcts, _FakeStockfish(_legal_reply))
    start = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    runner._play_game_worker(start, engine_is_white, ctx, max_plies=6)
    assert mcts.seen_sides, "MCTS was never asked to move"
    assert all(s == engine_is_white for s in mcts.seen_sides)


def test_search_mode_plumbing():
    assert check_search_mode("serial") == "serial"
    assert check_search_mode("batched") == "batched"
    with pytest.raises(ValueError):
        check_search_mode("hyper")
    require_serial_search("serial", "test")
    # With search_batched implemented in zeropawn_cpp, batched passes
    require_serial_search("batched", "test")
    # If search_batched were absent, require_serial_search raises RuntimeError
    orig = getattr(zeropawn_cpp.MCTS, "search_batched", None)
    try:
        delattr(zeropawn_cpp.MCTS, "search_batched")
        with pytest.raises(RuntimeError):
            require_serial_search("batched", "test")
    finally:
        if orig is not None:
            setattr(zeropawn_cpp.MCTS, "search_batched", orig)
    # Default is serial (contract: behavior == today's serial path).
    sig = inspect.signature(MatchRunner.__init__)
    assert sig.parameters["search_mode"].default == "serial"
    # NOTE: no clock/time-loss path exists in this runner (no chess clocks),
    # so resignation/checkmate/illegal-move cover all terminal adjudications.


def test_elo_interval_shape():
    from python.eval.match_runner import elo_interval

    even = elo_interval(2, 4, 2)
    assert even["elo_diff"] == 0.0
    assert even["lo"] < 0.0 < even["hi"]
    assert even["score"] == 0.5
    strong = elo_interval(60, 8, 32)
    assert 90.0 < strong["elo_diff"] < 110.0  # s=0.64 -> ~+100
    assert strong["lo"] < strong["elo_diff"] < strong["hi"]
    empty = elo_interval(0, 0, 0)
    assert empty == {"elo_diff": None, "lo": None, "hi": None, "score": 0.0}
    shutout = elo_interval(0, 0, 5)
    assert shutout["elo_diff"] is None and shutout["score"] == 0.0


def test_pgn_write_and_rescore_roundtrip(tmp_path):
    import chess
    from python.eval.match_runner import write_match_pgn, rescore_pgn

    # Fool's mate: black mates white.
    start = chess.Board()
    for mv in ["f2f3", "e7e5", "g2g4", "d8h4"]:
        start.push_uci(mv)
    assert start.is_checkmate()
    games = [
        {
            "opening_idx": 0,
            "opening_fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "engine_is_white": False,
            "result": "0-1",
            "moves": ["f2f3", "e7e5", "g2g4", "d8h4"],
            "seed": 42,
        },
        {
            "opening_idx": 0,
            "opening_fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "engine_is_white": True,
            "result": "1/2-1/2",
            "moves": ["e2e4", "e7e5"],
            "seed": 43,
        },
    ]
    pgn = write_match_pgn(games, tmp_path / "m.pgn")
    resc = rescore_pgn(pgn)
    assert resc["games"] == 2 and resc["clean_games"] == 2
    assert resc["problem_rows"] == []
    # Engine black won game 0, drew game 1 as white.
    assert (resc["engine_wins"], resc["engine_draws"], resc["engine_losses"]) == (1, 1, 0)

    # A contradictory record is flagged.
    bad = [dict(games[0], result="1-0")]
    resc2 = rescore_pgn(write_match_pgn(bad, tmp_path / "b.pgn"))
    assert len(resc2["problem_rows"]) == 1


def test_stockfish_elo_validation():
    import inspect
    from python.eval.match_runner import StockfishPlayer

    assert "set_elo" in dir(StockfishPlayer)
    sig = inspect.signature(MatchRunner.__init__)
    assert sig.parameters["seed"].default == 42
    assert sig.parameters["stockfish_elo"].default is None


def test_match_runner_batched_mode():
    model = ChessNet(num_blocks=2, num_filters=32)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    runner = MatchRunner(
        model=model,
        device=device,
        stockfish_path="tools/stockfish.exe",
        stockfish_nodes=50,
        mcts_sims=10,
        search_mode="batched",
        batch_size=8,
        openings_path="data/openings/eval_openings_50.epd",
    )
    assert runner.search_mode == "batched"
    assert runner.batch_size == 8
    results = runner.run_match(num_openings=1)
    assert results["total_games"] == 2
    assert results["search_mode"] == "batched"
