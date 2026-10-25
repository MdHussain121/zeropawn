"""Book subcommands for ZeroPawn: build, evaluate, inspect, and train opening book."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import chess

from python.cli.common import (
    console,
    print_banner,
    print_error,
    print_info,
    print_success,
    setup_environment,
)
from python.data.opening_book import OpeningBook
from python.data.opening_corpus import OpeningCorpus
from python.data.stockfish_soundness import StockfishSoundnessEvaluator

logger = logging.getLogger(__name__)


def register_parser(subparsers: argparse._SubParsersAction) -> None:
    book_parser = subparsers.add_parser(
        "book",
        help="Opening book tools: build, evaluate, inspect, and policy fine-tuning.",
        description="Comprehensive opening book and theory management utilities.",
    )
    sub = book_parser.add_subparsers(dest="book_subcommand", help="Opening book actions")

    # 1. Build
    p_build = sub.add_parser(
        "build", help="Build curated opening database from EPD, JSON, and PGN sources."
    )
    p_build.add_argument(
        "--epd", type=str, default="data/openings/eval_openings_50.epd", help="Input EPD suite"
    )
    p_build.add_argument(
        "--catalog", type=str, default="data/openings/book_selfplay.json", help="Input JSON catalog"
    )
    p_build.add_argument(
        "--pgn", type=str, default="data/elite_2500.pgn", help="Input master PGN file"
    )
    p_build.add_argument("--max-pgn-games", type=int, default=500, help="Max PGN games to ingest")
    p_build.add_argument(
        "--output",
        type=str,
        default="data/openings/curated_book.json",
        help="Output curated book JSON",
    )
    p_build.add_argument(
        "--min-frequency", type=int, default=5, help="Master-game frequency threshold"
    )
    p_build.set_defaults(func=run_book_build)

    # 2. Evaluate
    p_eval = sub.add_parser(
        "evaluate", help="Run Stockfish soundness evaluations on candidate moves in the book."
    )
    p_eval.add_argument(
        "--book", type=str, default="data/openings/curated_book.json", help="Curated book path"
    )
    p_eval.add_argument("--max-positions", type=int, default=20, help="Max positions to evaluate")
    p_eval.add_argument(
        "--nodes", type=int, default=10000, help="Stockfish node budget per candidate move"
    )
    p_eval.set_defaults(func=run_book_evaluate)

    # 3. Inspect
    p_insp = sub.add_parser("inspect", help="Inspect book moves and statistics for a FEN position.")
    p_insp.add_argument(
        "fen",
        type=str,
        nargs="?",
        default="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        help="FEN string to inspect",
    )
    p_insp.add_argument(
        "--book", type=str, default="data/openings/curated_book.json", help="Curated book path"
    )
    p_insp.add_argument("--policy", type=str, default="frequency", help="Selection policy")
    p_insp.set_defaults(func=run_book_inspect)

    # 4. Train
    p_train = sub.add_parser(
        "train", help="Train / fine-tune opening policy on curated opening dataset."
    )
    p_train.add_argument(
        "--dataset", type=str, default="data/dataset_openings", help="Opening dataset directory"
    )
    p_train.add_argument(
        "--puzzle-data",
        "--puzzles",
        type=str,
        default=None,
        help="Path to compiled tactical puzzle dataset",
    )
    p_train.add_argument(
        "--syzygy-data", type=str, default=None, help="Path to compiled Syzygy endgame dataset"
    )
    p_train.add_argument(
        "--general-data",
        type=str,
        default=None,
        help="Path to general self-play replay buffer for anti-forgetting mixing",
    )
    p_train.add_argument(
        "--resume",
        type=str,
        default="models/checkpoints_scaled/best_model.pt",
        help="Baseline checkpoint to load",
    )
    p_train.add_argument(
        "--checkpoint-dir",
        type=str,
        default="models/checkpoints_opening",
        help="Output checkpoint directory",
    )
    p_train.add_argument("--epochs", type=int, default=2, help="Training epochs")
    p_train.add_argument("--batch-size", type=int, default=64, help="Batch size")
    p_train.add_argument("--lr", type=float, default=0.0001, help="Learning rate")
    p_train.add_argument(
        "--max-steps", type=int, default=100, help="Max training steps (for bounded runs)"
    )
    p_train.set_defaults(func=run_book_train)

    # 5. Build Puzzles Dataset
    p_puz = sub.add_parser("build-puzzles", help="Compile tactical puzzle dataset from CSV.")
    p_puz.add_argument(
        "--csv",
        type=str,
        default="data/puzzles/lichess_puzzles_5000_1800_2800.csv",
        help="Input puzzle CSV",
    )
    p_puz.add_argument(
        "--output", type=str, default="data/dataset_puzzles", help="Output puzzle dataset dir"
    )
    p_puz.add_argument("--min-rating", type=int, default=2000, help="Minimum puzzle rating filter")
    p_puz.add_argument("--max-positions", type=int, default=5000, help="Maximum puzzle positions")
    p_puz.set_defaults(func=run_book_build_puzzles)


def run_book_build(args: argparse.Namespace) -> int:
    setup_environment()
    print_banner("Building Curated Opening Database")
    from python.cli.common import REPO_ROOT

    corpus = OpeningCorpus(min_game_frequency=args.min_frequency)

    # Resolve EPD
    epd_path = Path(args.epd) if args.epd else None
    if epd_path and not epd_path.is_file() and (REPO_ROOT / epd_path).is_file():
        epd_path = REPO_ROOT / epd_path

    # Resolve Catalog with fallback to known candidates
    catalog_path = Path(args.catalog) if args.catalog else None
    if catalog_path and not catalog_path.is_file() and (REPO_ROOT / catalog_path).is_file():
        catalog_path = REPO_ROOT / catalog_path
    elif catalog_path and not catalog_path.is_file():
        for cand in [
            REPO_ROOT / "data/openings/opening_book.bin.json",
            Path("data/openings/opening_book.bin.json"),
            REPO_ROOT / "data/openings/curated_book.json",
            Path("data/openings/curated_book.json"),
            REPO_ROOT / "data/openings/eco_openings.json",
        ]:
            if cand.is_file():
                catalog_path = cand
                break

    # Resolve PGN with fallback
    pgn_path = Path(args.pgn) if args.pgn else None
    if pgn_path and not pgn_path.is_file() and (REPO_ROOT / pgn_path).is_file():
        pgn_path = REPO_ROOT / pgn_path
    elif pgn_path and not pgn_path.is_file():
        for cand in [
            REPO_ROOT / "data/elite_2500.pgn",
            Path("data/elite_2500.pgn"),
            REPO_ROOT / "data/filtered.pgn",
            Path("data/filtered.pgn"),
        ]:
            if cand.is_file():
                pgn_path = cand
                break

    # Ingest EPD
    if epd_path and epd_path.is_file():
        print_info(f"Ingesting EPD: {epd_path}")
        n_epd = corpus.ingest_epd(epd_path, source_name="eval_50_epd")
        print_info(f"Ingested {n_epd} positions from EPD.")

    # Ingest JSON catalog
    if catalog_path and catalog_path.is_file():
        print_info(f"Ingesting JSON catalog: {catalog_path}")
        n_cat = corpus.ingest_json_catalog(catalog_path, source_name="catalog_json")
        print_info(f"Ingested {n_cat} lines from JSON catalog.")

    # Ingest PGN master games
    if pgn_path and pgn_path.is_file():
        print_info(f"Ingesting PGN master games: {pgn_path} (max={args.max_pgn_games})")
        n_pgn = corpus.ingest_pgn(
            pgn_path, max_games=args.max_pgn_games, source_name="elite_2500_pgn"
        )
        print_info(f"Ingested {n_pgn} games from PGN.")

    stats = corpus.get_summary_stats()
    print_success(
        f"Ingestion complete: {stats['unique_canonical_positions']} unique positions, {stats['unique_position_move_pairs']} unique (position, move) pairs."
    )

    if stats["unique_canonical_positions"] == 0:
        print_error(
            "No opening positions were ingested! Check that input files (EPD, catalog, or PGN) exist."
        )
        return 1

    out_path = Path(args.output)
    if not out_path.is_absolute():
        out_path = REPO_ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    exported = corpus.export_curated_db(out_path, include_all=False)
    with open(exported, "r", encoding="utf-8") as f:
        data = json.load(f)
    if len(data.get("positions", {})) == 0:
        print_info(
            "Zero positions met strict union filter; exporting full corpus with include_all=True..."
        )
        exported = corpus.export_curated_db(out_path, include_all=True)

    print_success(f"Exported curated opening book to {exported}")

    # Synchronize to CWD and REPO_ROOT if they differ
    try:
        locations = [
            REPO_ROOT / "data/openings/curated_book.json",
            Path("data/openings/curated_book.json"),
        ]
        content = exported.read_text(encoding="utf-8")
        for loc in locations:
            try:
                if not loc.resolve().samefile(exported.resolve()):
                    loc.parent.mkdir(parents=True, exist_ok=True)
                    loc.write_text(content, encoding="utf-8")
            except Exception:
                pass
    except Exception:
        pass

    return 0


def run_book_evaluate(args: argparse.Namespace) -> int:
    setup_environment()
    print_banner("Running Stockfish Opening Soundness Analysis")
    from python.cli.common import REPO_ROOT

    book_path = Path(args.book)
    if not book_path.is_file() and (REPO_ROOT / book_path).is_file():
        book_path = REPO_ROOT / book_path
    if not book_path.is_file():
        print_error(f"Opening book not found at {book_path}")
        return 1

    with open(book_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    corpus = OpeningCorpus()
    # Reconstruct positions from book JSON
    for key, pos_dict in data.get("positions", {}).items():
        board = chess.Board(pos_dict["fen"])
        corpus.get_or_create_position(
            board,
            ply=pos_dict.get("ply", 0),
            eco=pos_dict.get("eco"),
            name=pos_dict.get("opening_name"),
        )
        for uci, ev_dict in pos_dict.get("moves", {}).items():
            if chess.Move.from_uci(uci) in board.legal_moves:
                corpus.add_position_move(board, chess.Move.from_uci(uci), source="curated_book")
                # Preserve original frequency weights from the curated book.
                ev = corpus.get_or_create_position(board).moves.get(uci)
                if ev is not None and ev_dict is not None:
                    original_freq = ev_dict.get("frequency")
                    if original_freq is not None:
                        ev.frequency = max(ev.frequency, int(original_freq))
                        # Approximate distinct game count to match frequency.
                        needed = max(0, int(original_freq) - len(ev.distinct_game_ids))
                        for i in range(needed):
                            ev.distinct_game_ids.add(f"reconstructed_{uci}_{i}")

    evaluator = StockfishSoundnessEvaluator(nodes=args.nodes)
    print_info(
        f"Evaluating candidate moves with Stockfish (max_positions={args.max_positions}, nodes={args.nodes})..."
    )
    res = evaluator.evaluate_corpus_positions(
        corpus, max_positions=args.max_positions, nodes=args.nodes
    )
    evaluator.close()

    print_success(
        f"Stockfish evaluation finished: {res['moves_evaluated']} moves evaluated, {res['sound_moves_found']} sound moves found (cache size={res['cache_size']})."
    )

    corpus.export_curated_db(args.book, include_all=False)
    print_success(f"Updated curated opening book at {args.book}")
    return 0


def run_book_inspect(args: argparse.Namespace) -> int:
    setup_environment()
    from python.cli.common import REPO_ROOT

    book_path = Path(args.book)
    if not book_path.is_file() and (REPO_ROOT / book_path).is_file():
        book_path = REPO_ROOT / book_path
    if not book_path.is_file():
        print_error(f"Book file not found: {book_path}")
        return 1

    book = OpeningBook(book_path=book_path, policy=args.policy)
    board = chess.Board(args.fen)
    res = book.lookup(board)

    console.print(f"\n[bold green]Inspecting Position:[/bold green] {board.fen()}")
    key = board.epd(en_passant="pawns")
    in_book = key in book.positions

    if not in_book:
        print_info("Position is NOT in opening book. Engine will fall back to MCTS search.")
        return 0

    pos_data = book.positions[key]
    console.print(
        f"Opening: [bold cyan]{pos_data.get('opening_name', 'Unknown')}[/bold cyan] ({pos_data.get('eco', 'N/A')})"
    )
    console.print(f"Qualified Moves in Book: {len(pos_data.get('moves', {}))}")

    for uci, ev in pos_data.get("moves", {}).items():
        cp_str = (
            f"{ev.get('stockfish_cp'):+.1f} cp"
            if ev.get("stockfish_cp") is not None
            else "unevaluated"
        )
        console.print(
            f"  • [bold yellow]{uci}[/bold yellow] ({ev.get('san', uci)}): freq={ev.get('frequency', 0)}, eval={cp_str}, qual={ev.get('qualification')}"
        )

    if res:
        chosen_move, meta = res
        console.print(
            f"\nPolicy [bold cyan]{args.policy}[/bold cyan] selected: [bold green]{chosen_move}[/bold green]\n"
        )
    return 0


def run_book_train(args: argparse.Namespace) -> int:
    setup_environment()
    from python.train.train_opening_policy import train_opening_policy

    print_banner("Training Supervised Policy (Openings + Endgames + Tactics)")
    dataset_dir = Path(args.dataset)
    if not (dataset_dir / "train").is_dir():
        print_error(f"Opening dataset directory {dataset_dir} does not exist. Build it first.")
        return 1

    extra_dirs = [
        d for d in [getattr(args, "puzzle_data", None), getattr(args, "syzygy_data", None)] if d
    ]

    res = train_opening_policy(
        opening_data_dir=dataset_dir,
        general_data_dir=getattr(args, "general_data", None),
        extra_data_dirs=extra_dirs,
        resume_checkpoint=args.resume,
        output_checkpoint_dir=args.checkpoint_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        max_steps=args.max_steps,
    )
    print_success(
        f"Supervised training complete! Best checkpoint: {res['best_checkpoint']} (val_loss={res['best_val_loss']:.4f})"
    )
    return 0


def run_book_build_puzzles(args: argparse.Namespace) -> int:
    setup_environment()
    from python.data.build_puzzle_dataset import build_puzzle_dataset

    print_banner("Compiling Tactical Puzzle Dataset")
    csv_path = Path(args.csv)
    if not csv_path.is_file():
        print_error(f"Puzzle CSV file not found at {csv_path}")
        return 1

    summary = build_puzzle_dataset(
        csv_path=csv_path,
        output_dir=args.output,
        min_rating=args.min_rating,
        max_positions=args.max_positions,
    )
    print_success(
        f"Built puzzle dataset: {summary['total_samples']} samples ({summary['train_samples']} train, {summary['val_samples']} val) saved to {args.output}"
    )
    return 0
