"""Terminal parity gate (Task 7b).

Runs each operation directly and through chess.cmd, normalizes volatile
tokens (timestamps, durations, absolute paths, live counters, machine-state
readings), and FAILS if the normalized outputs differ, a required section is
missing, or exit codes disagree.

Usage: python scripts/check_terminal_parity.py [--op NAME] [--keep]
Exits 0 when parity holds, 1 with a diff report otherwise.
"""

from __future__ import annotations

import argparse
import difflib
import os
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CHESS_CMD = str(REPO_ROOT / "chess.cmd")
PY = sys.executable

# (pattern, replacement) applied in order to both transcripts.
NORMALIZE = [
    (r"\[\d{2}:\d{2}:\d{2}\]", "[TIME]"),
    (r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}", "DATE TIME"),
    (r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}", "DATE TIME"),
    (r"DURATION: \d+\.\d+s", "DURATION: Ns"),
    (r"in \d+\.\d+s \(M\)", "in Ns (M)"),
    (r"= \d+\.\d+s \(M\)", "= Ns (M)"),
    (r"\d+\.\d+s \(\d+\.\d+s/game\)", "Ns (Ns/game)"),
    (r"t=\d+/\d+s", "t=N/Ns"),
    (r"\b\d+\.\d+s", "Ns"),
    (r"sha=[0-9a-f]{8}", "sha=HASH"),
    (r"buffer=\d+", "buffer=N"),
    (r"\(have \d+\)", "(have N)"),
    (r"games=\d+", "games=N"),
    (r"steps=\d+", "steps=N"),
    (r"positions=\d+", "positions=N"),
    (r"evals=\d+", "evals=N"),
    (r"sims_sec=[\d.]+", "sims_sec=N"),
    (r"games_hr=[\d.]+", "games_hr=N"),
    (r"pos_hr=[\d.]+", "pos_hr=N"),
    (r"batch_avg=[\d.]+", "batch_avg=N"),
    (r"lat_ms=[\d.]+", "lat_ms=N"),
    (r"train_steps_sec=[\d.]+", "train_steps_sec=N"),
    (r"gpu=[\d.]+%", "gpu=N%"),
    (r"vram_mb=\d+", "vram_mb=N"),
    (r"ram_free_gb=[\d.]+", "ram_free_gb=N"),
    (r"rss_mb=\d+", "rss_mb=N"),
    (r"lag_min=[\d.]+", "lag_min=N"),
    (r"next_eval_s=[\d.]+", "next_eval_s=N"),
    (r"elapsed=\d+s", "elapsed=Ns"),
    (r"duty=[\d.]+", "duty=N"),
    (r"avg_plies=[\d.]+", "avg_plies=N"),
    (r"draw_rate=[\d.]+", "draw_rate=N"),
    (r"adj_frac=[\d.]+", "adj_frac=N"),
    (r"would_resign=[\d.]+", "would_resign=N"),
    (r"files_done=\d+/\d+", "files_done=N/N"),
    (r"mb=[\d.]+", "mb=N"),
    (r"\d[\d,]*(?:\.\d+)? (?:GB|MB|KB)", "N UNIT"),
    (r"\(\d+\.\d+% used\)", "(N% used)"),
    (r"g\+\+\S* [^\n]*? \d+\.\d+\.\d+", "g++ COMPILER"),
    (r"MinGW[^\n]*", "MINGW"),
    (r"[A-Z]:\\[^\s\"']*", "ABS_PATH"),
    (r"/[^\s\"']*chess-rl[^\s\"']*", "ABS_PATH"),
    (r"tmp[\\/][^\s\"']*", "TMP_PATH"),
    (r"seed=\d+", "seed=N"),
]

OPS = [
    # name, direct argv, cmd argv (None = no cmd path), stdin, timeout, sections
    (
        "download-license",
        [PY, "python/data/download.py", "--check-license"],
        None,
        None,
        120,
        ["OPERATION:", "START:", "COMMAND:", "PARAMETERS", "STATUS:", "EXIT:"],
    ),
    (
        "download-info",
        [PY, "python/data/download.py", "--month", "2026-09", "--info"],
        None,
        None,
        120,
        ["OPERATION:", "size_bytes", "STATUS:", "EXIT:"],
    ),
    (
        "filter",
        [
            PY,
            "python/data/filter.py",
            "--input",
            "data/filtered_2200.pgn",
            "--output",
            "tmp/parity_filter.pgn",
            "--max-kept",
            "5",
            "--log-interval",
            "5",
        ],
        None,
        None,
        300,
        ["OPERATION:", "PARAMETERS", "kept=5", "STATUS:", "EXIT:"],
    ),
    (
        "build",
        [
            PY,
            "python/data/build_dataset.py",
            "--pgn",
            "tmp/parity_filter.pgn",
            "--output-dir",
            "tmp/parity_dataset",
            "--max-games",
            "5",
        ],
        None,
        None,
        300,
        ["OPERATION:", "Training positions", "STATUS:", "EXIT:"],
    ),
    (
        "supervised-script",
        [
            PY,
            "-u",
            "python/train/train_supervised.py",
            "--train-dir",
            "tmp/terminal_golden_dataset/train",
            "--val-dir",
            "tmp/terminal_golden_dataset/train",
            "--max-steps",
            "2",
            "--batch-size",
            "8",
            "--checkpoint-dir",
            "tmp/parity_ckpt",
            "--log-dir",
            "tmp/parity_runs",
            "--no-gpu-resident",
            "--workers",
            "0",
            "--seed",
            "0",
        ],
        None,
        None,
        600,
        ["OPERATION:", "val_loss=9.0158", "STATUS:", "EXIT:"],
    ),
    (
        "supervised-cli",
        [
            PY,
            "-u",
            "cli.py",
            "supervised",
            "--train-dir",
            "tmp/terminal_golden_dataset/train",
            "--val-dir",
            "tmp/terminal_golden_dataset/train",
            "--steps",
            "2",
            "--batch-size",
            "8",
            "--checkpoint-dir",
            "tmp/parity_ckpt_cli",
            "--log-dir",
            "tmp/parity_runs_cli",
            "--no-gpu-resident",
            "--workers",
            "0",
            "--seed",
            "0",
        ],
        [
            "supervised",
            "--train-dir",
            "tmp/terminal_golden_dataset/train",
            "--val-dir",
            "tmp/terminal_golden_dataset/train",
            "--steps",
            "2",
            "--batch-size",
            "8",
            "--checkpoint-dir",
            "tmp/parity_ckpt_cli2",
            "--log-dir",
            "tmp/parity_runs_cli2",
            "--no-gpu-resident",
            "--workers",
            "0",
            "--seed",
            "0",
        ],
        None,
        600,
        ["OPERATION:", "val_loss=9.0158", "STATUS:", "EXIT:"],
    ),
    (
        "puzzles-eval",
        [PY, "-u", "cli.py", "puzzles", "eval", "-n", "5", "--mcts-sims", "10"],
        ["puzzles", "eval", "-n", "5", "--mcts-sims", "10"],
        None,
        600,
        ["OPERATION:", "2/5 = 40.00%", "STATUS:", "EXIT:"],
    ),
    (
        "eval-script",
        [PY, "-u", "scripts/eval_puzzles_now.py", "--mcts-puzzles", "2"],
        None,
        None,
        600,
        ["OPERATION:", "343/1000 = 34.30%", "STATUS:", "EXIT:"],
    ),
    (
        "match",
        [PY, "-u", "cli.py", "match", "run", "-n", "2", "--mcts-sims", "10", "--workers", "1"],
        ["match", "run", "-n", "2", "--mcts-sims", "10", "--workers", "1"],
        None,
        600,
        ["OPERATION:", "engine_losses", "elo_diff_95ci", "STATUS:", "EXIT:"],
    ),
    (
        "status",
        [PY, "-u", "cli.py", "status"],
        ["status"],
        None,
        300,
        ["OPERATION:", "checkpoints_found", "STATUS:", "EXIT:"],
    ),
    (
        "addpuzzles-list",
        [PY, "-u", "cli.py", "add-puzzles", "list"],
        ["add-puzzles", "list"],
        None,
        180,
        ["OPERATION:", "1,000 puzzles", "STATUS:", "EXIT:"],
    ),
    (
        "fetch",
        [PY, "scripts/fetch_puzzles.py"],
        ["add-puzzles", "fetch"],
        None,
        180,
        None,
        ["1000 puzzles"],
    ),  # different programs: fact parity only
    (
        "test-fast",
        [PY, "-u", "cli.py", "test", "--suite", "fast", "-k", "test_import"],
        ["test", "--suite", "fast", "-k", "test_import"],
        None,
        300,
        ["OPERATION:", "1 passed", "STATUS:", "EXIT:"],
    ),
    (
        "cleanup",
        [PY, "-u", "cli.py", "cleanup"],
        ["cleanup"],
        None,
        180,
        ["OPERATION:", "DRYRUN", "STATUS:", "EXIT:"],
    ),
    (
        "uci",
        [PY, "-u", "cli.py", "uci", "--sims", "5"],
        ["uci", "--sims", "5"],
        b"uci\nquit\n",
        300,
        ["id name ZeroPawn", "uciok"],
    ),
    (
        "train-verify",
        [PY, "-u", "cli.py", "train", "--verify"],
        ["train", "--verify"],
        None,
        900,
        ["OPERATION:", "EVAL:", "STATUS:", "EXIT:"],
    ),
    (
        "syzygy",
        [PY, "scripts/download_syzygy.py"],
        None,
        None,
        300,
        ["OPERATION:", "files_verified", "STATUS:", "EXIT:"],
    ),
    (
        "bench-tiny",
        [
            PY,
            "scripts/bench_selfplay.py",
            "--mode",
            "selfplay",
            "--warmup-sec",
            "1",
            "--counted-sec",
            "2",
            "--sample-interval",
            "1.0",
            "--trials",
            "1",
            "--concurrency",
            "2",
            "--full-sims",
            "5",
            "--fast-sims",
            "5",
            "--output-dir",
            "tmp/parity_bench",
            "--checkpoint-path",
            "models/checkpoints_scaled/best_model.pt",
            "--base-seed",
            "0",
        ],
        None,
        None,
        600,
        ["OPERATION:", "BENCHMARK SUMMARY", "STATUS:", "EXIT:"],
    ),
    (
        "tb-feasibility",
        [PY, "scripts/tb_feasibility.py", "--max-games", "2"],
        None,
        None,
        600,
        ["OPERATION:", "subset 2 of", "STATUS:", "EXIT:"],
    ),
    (
        "tb-validate",
        [PY, "scripts/validate_tb_offline.py", "--max-games", "2", "--sf-nodes", "100"],
        None,
        None,
        600,
        ["OPERATION:", "subset 2 of", "STATUS:", "EXIT:"],
    ),
]


def normalize(text: str) -> str:
    out = text.replace(str(REPO_ROOT), "ABS_PATH")
    for pattern, replacement in NORMALIZE:
        out = re.sub(pattern, replacement, out)
    return out


def run_capture(argv, stdin_bytes=None, timeout=300):
    """Run direct (list argv) or via chess.cmd (argv = cmd args)."""
    if argv and argv[0] == "CHESS":
        cmd = ["cmd", "/c", CHESS_CMD] + argv[1:]
        shell_cwd = str(REPO_ROOT)
    else:
        cmd = argv
        shell_cwd = str(REPO_ROOT)
    try:
        proc = subprocess.run(
            cmd,
            input=stdin_bytes,
            capture_output=True,
            timeout=timeout,
            cwd=shell_cwd,
        )
    except subprocess.TimeoutExpired:
        return None, "TIMEOUT", b"", b""
    out = (proc.stdout or b"").decode("utf-8", errors="replace")
    err = (proc.stderr or b"").decode("utf-8", errors="replace")
    return proc.returncode, "OK", out, err


def check_op(op, keep=False):
    name, direct, cmd_args, stdin, timeout = op[0], op[1], op[2], op[3], op[4]
    sections = op[5] if len(op) > 5 else None
    loose_facts = op[6] if len(op) > 6 else None
    print(f"--- {name} ---")
    failures = []
    rc_d, st_d, out_d, err_d = run_capture(direct, stdin, timeout)
    if st_d == "TIMEOUT":
        return [f"{name}: direct run timed out"]
    if rc_d != 0:
        failures.append(f"{name}: direct exit={rc_d} (expected 0)")
    if sections:
        for section in sections:
            if section not in out_d:
                failures.append(f"{name}: direct output missing required {section!r}")
    if cmd_args is None:
        print(f"    direct exit={rc_d} (no cmd path; sections checked)")
        return failures
    rc_c, st_c, out_c, err_c = run_capture(["CHESS"] + cmd_args, stdin, timeout)
    if st_c == "TIMEOUT":
        return failures + [f"{name}: cmd run timed out"]
    if rc_c != rc_d:
        failures.append(f"{name}: exit mismatch direct={rc_d} cmd={rc_c}")
    if loose_facts is not None:
        # Known-divergent implementations: same exit + same facts required.
        for fact in loose_facts:
            if fact not in out_d:
                failures.append(f"{name}: direct output missing fact {fact!r}")
            if fact not in out_c:
                failures.append(f"{name}: cmd output missing fact {fact!r}")
        if not [f for f in failures if f.startswith(name + ":")]:
            print(f"    fact parity OK (exit={rc_d})")
        return failures
    norm_d = normalize(out_d)
    norm_c = normalize(out_c)
    if norm_d != norm_c:
        diff = list(
            difflib.unified_diff(
                norm_d.splitlines(), norm_c.splitlines(), "direct", "chess.cmd", lineterm=""
            )
        )
        diff_path = REPO_ROOT / "tmp" / f"parity_{name}.diff"
        try:
            diff_path.parent.mkdir(parents=True, exist_ok=True)
            diff_path.write_text("\n".join(diff[:80]), encoding="utf-8")
        except OSError:
            pass
        failures.append(
            f"{name}: normalized outputs differ ({len(diff)} diff lines; see {diff_path})"
        )
        for line in diff[:12]:
            print("    " + line)
    else:
        print(f"    parity OK (exit={rc_d})")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description="Terminal parity gate: direct vs chess.cmd.")
    parser.add_argument("--op", type=str, default=None, help="Run a single op by name.")
    parser.add_argument("--keep", action="store_true", help="Keep tmp artifacts.")
    args = parser.parse_args()

    ops = [op for op in OPS if args.op is None or op[0] == args.op]
    if args.op is not None and not ops:
        print(f"unknown op: {args.op}")
        return 1
    failures: list[str] = []
    for op in ops:
        failures.extend(check_op(op, keep=args.keep))
    if failures:
        print("\nPARITY FAILURES:")
        for failure in failures:
            print("  FAIL " + failure)
        return 1
    print("\nPARITY: all checked operations identical after normalization.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
