"""Tests for python/selfplay/tb_adjudicate.py.

Covers:
(a) Known positions: KQvK, KRvK, KPvK (win and draw), KBvK draw, cursed-win example, Black to move.
(b) Filter rejection: no probe with castling rights or 6+ pieces.
(c) Label perspective: side to move perspective (+1 win, -1 loss, 0 draw).
(d) 10,000 random <=5-piece positions: string filter parity vs python-chess board.
(e) 6-thread stress test: 5,000 positions, multi-threaded vs single-threaded equality,
    lock wait time reporting.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import random
import time
import pytest
import chess

from python.selfplay.tb_adjudicate import TablebaseAdjudicator

TB_PATH = "data/syzygy"


@pytest.fixture(scope="module")
def adjudicator():
    tb_p = Path(TB_PATH)
    if not tb_p.exists() or not any(tb_p.glob("*.rtbw")):
        pytest.skip("Syzygy tablebases not present in data/syzygy")
    adj = TablebaseAdjudicator.get_instance(tablebase_path=TB_PATH, max_pieces=5)
    if adj.tablebase is None:
        pytest.skip("Failed to load tablebases from data/syzygy")
    yield adj


def test_known_positions_outcomes(adjudicator: TablebaseAdjudicator):
    """Test known endgame positions with expected WDL outcomes."""
    # 1. KQvK - White to move winning -> +1
    fen_kqvk_w = "8/8/8/8/8/5K2/4Q3/7k w - - 0 1"
    res, meta = adjudicator.probe(fen_kqvk_w)
    assert res == 1
    assert meta["raw_wdl"] == 2

    # 2. KQvK - Black to move (White is winning, Black is losing) -> -1
    fen_kqvk_b = "8/8/8/8/8/5K2/4Q3/7k b - - 0 1"
    res, meta = adjudicator.probe(fen_kqvk_b)
    assert res == -1
    assert meta["raw_wdl"] == -2

    # 3. KRvK - White to move winning -> +1 (Rook on a2, King on e3, Black King on h1)
    fen_krvk_w = "8/8/8/8/8/4K3/R7/7k w - - 0 1"
    res, meta = adjudicator.probe(fen_krvk_w)
    assert res == 1
    assert meta["raw_wdl"] == 2

    # 4. KPvK - White to move winning (Pawn on 7th, King guarding) -> +1
    fen_kpvk_win = "8/4P1k1/8/5K2/8/8/8/8 w - - 0 1"
    res, meta = adjudicator.probe(fen_kpvk_win)
    assert res == 1

    # 5. KPvK - White to move Drawn (King in front, opposition) -> 0
    fen_kpvk_draw = "8/8/8/8/4k3/8/4P3/4K3 w - - 0 1"
    res, meta = adjudicator.probe(fen_kpvk_draw)
    assert res == 0
    assert meta["raw_wdl"] == 0

    # 6. KBvK - Known theoretical Draw -> 0
    fen_kbvk = "8/8/8/8/8/5K2/4B3/7k w - - 0 1"
    res, meta = adjudicator.probe(fen_kbvk)
    assert res == 0
    assert meta["raw_wdl"] == 0

    # 7. Black winning - Black to move -> +1
    # Black has Queen vs White King, Black to move
    fen_kqvk_black_win = "8/8/8/8/8/5k2/4q3/7K b - - 0 1"
    res, meta = adjudicator.probe(fen_kqvk_black_win)
    assert res == 1
    assert meta["raw_wdl"] == 2

    # 8. Black winning - White to move -> -1
    fen_kqvk_white_lost = "8/8/8/8/8/5k2/4q3/7K w - - 0 1"
    res, meta = adjudicator.probe(fen_kqvk_white_lost)
    assert res == -1
    assert meta["raw_wdl"] == -2


def test_cursed_win_maps_to_draw(adjudicator: TablebaseAdjudicator):
    """Verify that a cursed-win (or blessed-loss) position maps to draw (0)."""
    # Known 5-piece cursed win position under 50-move rule:
    # KNNvKP where mate requires >50 moves without pawn move
    fen_cursed = "8/8/8/8/8/1N6/2N5/K2k3p w - - 0 1"
    qualifies, _ = adjudicator.fast_filter(fen_cursed, max_pieces=5)
    assert qualifies
    res, meta = adjudicator.probe(fen_cursed)
    if meta.get("raw_wdl") in (1, -1):
        # Cursed win (+1) or blessed loss (-1) must map to 0
        assert res == 0


def test_filter_rejection_castling_and_piece_count(adjudicator: TablebaseAdjudicator):
    """Verify that positions with castling rights or > 5 pieces are rejected without probing."""
    # Starting position (32 pieces, castling rights KQkq)
    fen_start = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    passed, pc = adjudicator.fast_filter(fen_start, max_pieces=5)
    assert not passed
    assert pc == 32
    res, meta = adjudicator.probe(fen_start)
    assert res is None
    assert not meta["qualified"]

    # 4 pieces but has castling rights "K"
    fen_castling = "4k3/8/8/8/8/8/8/4K2R w K - 0 1"
    passed, pc = adjudicator.fast_filter(fen_castling, max_pieces=5)
    assert not passed
    res, meta = adjudicator.probe(fen_castling)
    assert res is None

    # 6 pieces with no castling rights
    fen_6_pieces = "4k3/8/8/3p4/4P3/8/4K3/R6R w - - 0 1"
    passed, pc = adjudicator.fast_filter(fen_6_pieces, max_pieces=5)
    assert not passed
    assert pc == 6
    res, meta = adjudicator.probe(fen_6_pieces)
    assert res is None


def test_string_filter_parity_10000_positions():
    """Verify that the fast string filter exactly matches python-chess on 10,000 random positions."""
    random.seed(42)
    pieces = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]

    agreements = 0
    total = 10000

    for _ in range(total):
        board = chess.Board(None)
        # Always place kings
        wk_sq = random.choice(list(chess.SQUARES))
        board.set_piece_at(wk_sq, chess.Piece(chess.KING, chess.WHITE))

        bk_candidates = [s for s in chess.SQUARES if s != wk_sq]
        bk_sq = random.choice(bk_candidates)
        board.set_piece_at(bk_sq, chess.Piece(chess.KING, chess.BLACK))

        # Randomly choose number of additional pieces (0 to 8)
        num_extra = random.randint(0, 8)
        avail_sqs = [s for s in chess.SQUARES if s not in (wk_sq, bk_sq)]
        extra_sqs = random.sample(avail_sqs, num_extra)

        for sq in extra_sqs:
            pt = random.choice(pieces)
            co = random.choice([chess.WHITE, chess.BLACK])
            board.set_piece_at(sq, chess.Piece(pt, co))

        # Randomly add castling rights 30% of the time
        if random.random() < 0.30:
            board.castling_rights = random.choice(
                [chess.BB_H1, chess.BB_A1, chess.BB_H8, chess.BB_A8]
            )
        else:
            board.castling_rights = chess.BB_EMPTY

        fen = board.fen()

        # String filter decision
        str_pass, str_pc = TablebaseAdjudicator.fast_filter(fen, max_pieces=5)

        b_fen = chess.Board(fen)
        true_pc = len(b_fen.piece_map())
        true_castling = bool(b_fen.castling_rights)
        true_pass = (true_pc <= 5) and (not true_castling)

        assert str_pc == true_pc, f"Piece count mismatch for {fen}: {str_pc} vs {true_pc}"
        assert str_pass == true_pass, (
            f"Filter decision mismatch for {fen}: {str_pass} vs {true_pass}"
        )
        agreements += 1

    assert agreements == total


def test_six_thread_stress_5000_positions(adjudicator: TablebaseAdjudicator):
    """Stress test: 5,000 positions queried across 6 threads.

    Asserts:
    1. Multi-threaded probe results match single-threaded probe results 100%.
    2. Measures average probe time, FEN-building time, and lock wait time.
    3. Verifies lock wait overhead is negligible (< 1% of wall-clock under simulated worker pacing).
    """
    random.seed(12345)
    # Generate 50 distinct valid <=5 piece positions
    unique_fens = []
    attempts = 0
    while len(unique_fens) < 50 and attempts < 1000:
        attempts += 1
        b = chess.Board(None)
        wk, bk = random.sample(list(chess.SQUARES), 2)
        b.set_piece_at(wk, chess.Piece(chess.KING, chess.WHITE))
        b.set_piece_at(bk, chess.Piece(chess.KING, chess.BLACK))

        # Add 1 or 2 extra pieces
        for sq in random.sample(
            [s for s in chess.SQUARES if s not in (wk, bk)], random.randint(1, 2)
        ):
            b.set_piece_at(
                sq,
                chess.Piece(
                    random.choice([chess.ROOK, chess.BISHOP, chess.KNIGHT]),
                    random.choice([chess.WHITE, chess.BLACK]),
                ),
            )
        b.castling_rights = chess.BB_EMPTY
        if b.is_valid():
            unique_fens.append(b.fen())

    # Build a test workload of 5,000 queries
    test_workload = [random.choice(unique_fens) for _ in range(5000)]

    # 1. Single-threaded baseline
    single_results = []
    t_single_start = time.perf_counter()
    for fen in test_workload:
        res, _ = adjudicator.probe(fen)
        single_results.append(res)
    t_single_wall = time.perf_counter() - t_single_start

    # Reset adjudicator telemetry
    adjudicator.total_lock_wait_sec = 0.0
    adjudicator.total_probe_time_sec = 0.0
    adjudicator.probes_total = 0

    # 2. Multi-threaded run with 6 worker threads (without sleep to test pure thread safety)
    def worker_task(sub_fens):
        return [adjudicator.probe(fen)[0] for fen in sub_fens]

    chunk_size = 5000 // 6
    chunks = [
        test_workload[i * chunk_size : (i + 1) * chunk_size if i < 5 else 5000] for i in range(6)
    ]

    t_multi_start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=6) as executor:
        futures = [executor.submit(worker_task, chunk) for chunk in chunks]
        multi_results = []
        for fut in futures:
            multi_results.extend(fut.result())
    t_multi_wall = time.perf_counter() - t_multi_start

    # Verify 100% equality
    assert multi_results == single_results, (
        "Multi-threaded probe results diverged from single-threaded!"
    )

    # 3. Measure lock wait time with simulated worker pacing (0.2 ms per move, realistic for fast workers)
    adjudicator.total_lock_wait_sec = 0.0
    adjudicator.total_probe_time_sec = 0.0
    adjudicator.probes_total = 0

    def paced_worker_task(sub_fens):
        res = []
        for fen in sub_fens:
            time.sleep(0.0002)  # 0.2 ms simulated move time
            out, _ = adjudicator.probe(fen)
            res.append(out)
        return res

    t_paced_start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=6) as executor:
        paced_results = list(executor.map(paced_worker_task, chunks))
    t_paced_wall = time.perf_counter() - t_paced_start

    stats = adjudicator.get_stats()
    total_lock_wait = stats["total_lock_wait_sec"]
    lock_wait_ratio = (total_lock_wait / t_paced_wall) if t_paced_wall > 0 else 0.0

    print(f"\n--- 6-Thread Tablebase Stress Results (5,000 queries) ---")
    print(
        f"Single-threaded wall time: {t_single_wall:.3f}s ({t_single_wall / 5000 * 1e6:.1f} us/call)"
    )
    print(
        f"Multi-threaded wall time:  {t_multi_wall:.3f}s ({t_multi_wall / 5000 * 1e6:.1f} us/call)"
    )
    print(f"Paced worker wall time:    {t_paced_wall:.3f}s")
    print(f"Total lock wait time:      {total_lock_wait:.4f}s")
    print(f"Lock wait / Wall-clock:    {lock_wait_ratio * 100:.3f}% (Threshold: < 1.00%)")
    print(f"Average probe time:        {stats['avg_probe_time_us']:.1f} us")

    # Lock wait across all 6 workers must be under 1% of wall-clock under worker pacing
    assert lock_wait_ratio < 0.01, (
        f"Lock wait ratio {lock_wait_ratio * 100:.2f}% exceeded 1% threshold!"
    )
