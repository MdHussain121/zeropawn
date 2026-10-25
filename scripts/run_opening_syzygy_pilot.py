"""Phase 8: Bounded End-to-End Pilot Execution Script.

Executes a bounded, fully verified pipeline demonstrating:
1. Unified opening ingestion & canonical deduplication.
2. Frequency thresholding & Stockfish soundness evaluation.
3. Curated opening database export & runtime book verification.
4. Opening supervised dataset construction (memmap format).
5. Offline Syzygy 3-4-5 piece endgame dataset generation.
6. Target shapes, types, and mover-perspective verification.
7. Bounded supervised opening policy training smoke test.
8. Checkpoint saving, reloading, and inference verification.
9. Verification that baseline checkpoint models/checkpoints_scaled/best_model.pt is preserved.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import time
from pathlib import Path

import chess
import numpy as np
import torch

_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

import python
import zeropawn_cpp
from python.cli.common import load_engine_model, setup_environment
from python.data.build_opening_dataset import build_opening_dataset
from python.data.opening_book import OpeningBook
from python.data.opening_corpus import OpeningCorpus
from python.data.stockfish_soundness import StockfishSoundnessEvaluator
from python.data.syzygy_dataset import SyzygySupervisionGenerator
from python.train.train_opening_policy import train_opening_policy

logger = logging.getLogger(__name__)


def run_bounded_pilot() -> int:
    setup_environment()
    t0 = time.time()
    print("=" * 80)
    print("ZEROPAWN: BOUNDED OPENING, POLICY TRAINING & SYZYGY PILOT")
    print("=" * 80)

    baseline_path = Path("models/checkpoints_scaled/best_model.pt")
    if not baseline_path.is_file():
        print(f"ERROR: Baseline checkpoint not found at {baseline_path}")
        return 1

    baseline_hash_before = hashlib.sha256(baseline_path.read_bytes()).hexdigest()
    print(
        f"[BASELINE] Verified baseline model: {baseline_path} (SHA256: {baseline_hash_before[:16]}...)"
    )

    # --------------------------------------------------------------------------
    # Step 1 & 2: Ingest & Deduplicate Opening Data
    # --------------------------------------------------------------------------
    print("\n--- STEP 1 & 2: INGESTING AND DEDUPLICATING OPENING DATA ---")
    corpus = OpeningCorpus(min_game_frequency=5, sf_soundness_threshold_cp=-30.0)

    # Ingest 50 canonical positions from EPD
    epd_path = Path("data/openings/eval_openings_50.epd")
    n_epd = corpus.ingest_epd(epd_path, source_name="eval_50_epd")
    print(f"  • Ingested {n_epd} benchmark positions from {epd_path}")

    # Ingest seed master lines from JSON catalog
    json_path = Path("data/openings/book_selfplay.json")
    n_json = corpus.ingest_json_catalog(json_path, source_name="book_selfplay_json")
    print(f"  • Ingested {n_json} opening lines from {json_path}")

    # Ingest bounded master games from PGN
    pgn_path = Path("data/elite_2500.pgn")
    n_pgn = corpus.ingest_pgn(pgn_path, max_games=50, max_plies=16, source_name="elite_2500_sample")
    print(f"  • Ingested {n_pgn} games from {pgn_path}")

    stats = corpus.get_summary_stats()
    print(f"[AUDIT] Unique Canonical Positions: {stats['unique_canonical_positions']:,}")
    print(f"[AUDIT] Unique Position-Move Pairs: {stats['unique_position_move_pairs']:,}")
    print(f"[AUDIT] Distinct Games Indexed: {stats['distinct_games_indexed']:,}")

    # --------------------------------------------------------------------------
    # Step 3: Bounded Stockfish Soundness Evaluation
    # --------------------------------------------------------------------------
    print("\n--- STEP 3: BOUNDED STOCKFISH SOUNDNESS EVALUATION ---")
    evaluator = StockfishSoundnessEvaluator(nodes=5000)
    print("  • Evaluating candidate moves on 5 positions (5,000 nodes/move)...")
    eval_res = evaluator.evaluate_corpus_positions(
        corpus, max_positions=5, max_moves_per_pos=3, nodes=5000
    )
    evaluator.close()
    print(
        f"[STOCKFISH] Moves evaluated: {eval_res['moves_evaluated']}, Sound moves (>= -0.30 pawns): {eval_res['sound_moves_found']}"
    )

    # --------------------------------------------------------------------------
    # Step 4: Export Curated Opening Database & Verify Runtime Book
    # --------------------------------------------------------------------------
    print("\n--- STEP 4: CURATED OPENING DATABASE & RUNTIME BOOK ---")
    curated_book_path = Path("data/openings/curated_book.json")
    corpus.export_curated_db(curated_book_path, include_all=True)
    print(f"  • Exported curated opening book to {curated_book_path}")

    book = OpeningBook(book_path=curated_book_path, policy="best")
    startpos = chess.Board()
    book_lookup = book.lookup(startpos)
    assert book_lookup is not None, "Starting position must have book moves"
    best_move, meta = book_lookup
    print(
        f"[BOOK LOOKUP] Starting Position best move: '{best_move}' (candidate moves: {meta['candidate_count']})"
    )
    assert best_move in [m.uci() for m in startpos.legal_moves], (
        "Selected move must be strictly legal"
    )

    # --------------------------------------------------------------------------
    # Step 5 & 6: Build Opening Supervised Dataset
    # --------------------------------------------------------------------------
    print("\n--- STEP 5 & 6: BUILDING OPENING SUPERVISED DATASET ---")
    opening_ds_dir = Path("data/dataset_openings_pilot")
    op_summary = build_opening_dataset(corpus, output_dir=opening_ds_dir, val_ratio=0.1, seed=42)
    print(
        f"[DATASET OPENING] Total: {op_summary['total_samples']} samples (Train: {op_summary['train_samples']}, Val: {op_summary['val_samples']})"
    )

    # --------------------------------------------------------------------------
    # Step 7 & 8: Generate Syzygy Endgame Dataset & Verify Perspective
    # --------------------------------------------------------------------------
    print("\n--- STEP 7 & 8: GENERATING SYZYGY ENDGAME SUPERVISION DATASET ---")
    tb_gen = SyzygySupervisionGenerator(tablebase_path="data/syzygy", max_pieces=5)
    assert tb_gen.tb is not None, "Syzygy tablebase must be accessible"

    syzygy_ds_dir = Path("data/dataset_syzygy_pilot")
    tb_summary = tb_gen.build_endgame_dataset(
        output_dir=syzygy_ds_dir, synthetic_count=40, val_ratio=0.2, seed=42
    )
    tb_gen.close()
    print(
        f"[DATASET SYZYGY] Total: {tb_summary['total_samples']} samples (Train: {tb_summary['train_samples']}, Val: {tb_summary['val_samples']})"
    )
    print(
        f"  • Breakdown: Wins={tb_summary['wins']}, Draws={tb_summary['draws']}, Losses={tb_summary['losses']}"
    )

    # Verify perspective invariants on White and Black
    w_kqk = chess.Board("8/8/4k3/8/8/3K4/2Q5/8 w - - 0 1")
    b_kqk = chess.Board("8/8/4k3/8/8/3K4/2Q5/8 b - - 0 1")
    tb_probe = SyzygySupervisionGenerator(tablebase_path="data/syzygy")
    rw_w = tb_probe.probe_wdl(w_kqk)
    rw_b = tb_probe.probe_wdl(b_kqk)
    tb_probe.close()
    assert rw_w == 2, f"White KQvK must be +2 (Win), got {rw_w}"
    assert rw_b == -2, f"Black KQvK must be -2 (Loss), got {rw_b}"
    print(
        "[PERSPECTIVE] Verified mover-relative perspective: White KQvK = +2 (Win), Black KQvK = -2 (Loss)"
    )

    # --------------------------------------------------------------------------
    # Step 9 & 10: Bounded Supervised Opening Policy Training Smoke Test
    # --------------------------------------------------------------------------
    print("\n--- STEP 9 & 10: BOUNDED SUPERVISED OPENING POLICY TRAINING ---")
    pilot_ckpt_dir = Path("models/checkpoints_opening_pilot")
    pilot_ckpt_dir.mkdir(parents=True, exist_ok=True)

    train_res = train_opening_policy(
        opening_data_dir=opening_ds_dir,
        resume_checkpoint=baseline_path,
        output_checkpoint_dir=pilot_ckpt_dir,
        epochs=1,
        batch_size=32,
        lr=0.0001,
        max_steps=25,
        num_blocks=12,
        num_filters=128,
        seed=42,
    )
    print(
        f"[TRAINING] Finished {train_res['final_step']} steps. Best Val Loss: {train_res['best_val_loss']:.4f}"
    )
    new_model_path = Path(train_res["best_checkpoint"])
    assert new_model_path.is_file(), f"Trained checkpoint must exist at {new_model_path}"

    # Verify reloading new checkpoint and running inference
    print("\n--- STEP 11: VERIFYING INFERENCE ON NEW CHECKPOINT ---")
    model, info = load_engine_model(new_model_path, device="cpu")
    model.eval()

    cpp_game = zeropawn_cpp.ChessGame()
    planes = zeropawn_cpp.encode_state(cpp_game)
    t = torch.from_numpy(planes).unsqueeze(0)
    with torch.no_grad():
        p_logits, wdl_logits = model(t)
    p_probs = torch.softmax(p_logits.float(), dim=-1).squeeze(0).numpy()
    v_probs = torch.softmax(wdl_logits.float(), dim=-1).squeeze(0).numpy()

    best_action_idx = int(np.argmax(p_probs))
    best_move_str = zeropawn_cpp.index_to_move(cpp_game, best_action_idx)
    expected_val = float(v_probs[0] - v_probs[2])
    print(
        f"[INFERENCE] New Checkpoint Move: '{best_move_str}' (p={p_probs[best_action_idx]:.3f}, value={expected_val:+.3f})"
    )
    assert best_move_str in cpp_game.legal_moves_uci(), "Inference move must be strictly legal"

    # --------------------------------------------------------------------------
    # Step 12: Verify Original Baseline Checkpoint Untouched
    # --------------------------------------------------------------------------
    print("\n--- STEP 12: VERIFYING BASELINE PRESERVATION ---")
    baseline_hash_after = hashlib.sha256(baseline_path.read_bytes()).hexdigest()
    assert baseline_hash_before == baseline_hash_after, "Baseline model was modified! VIOLATION!"
    print(
        f"[PRESERVATION] Verified models/checkpoints_scaled/best_model.pt is identical and unmodified."
    )

    elapsed = time.time() - t0
    print("\n" + "=" * 80)
    print(f"PILOT TEST COMPLETED SUCCESSFULLY IN {elapsed:.2f}s")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    sys.exit(run_bounded_pilot())
