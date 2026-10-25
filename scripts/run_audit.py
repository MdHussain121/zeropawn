"""Audit script to run every operation directly and via chess.cmd, capturing output and exit codes."""

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
AUDIT_DIR = ROOT / "docs" / "output_audit"
AUDIT_DIR.mkdir(parents=True, exist_ok=True)

# Define operations to test
# tuple: (op_key, direct_cmd, chesscmd_cmd)
OPERATIONS = [
    (
        "status",
        [sys.executable, "cli.py", "status"],
        ["cmd.exe", "/c", "chess.cmd", "status"],
    ),
    (
        "puzzles_eval",
        [sys.executable, "cli.py", "puzzles", "eval", "-n", "5", "--mcts-sims", "10"],
        ["cmd.exe", "/c", "chess.cmd", "puzzles", "eval", "-n", "5", "--mcts-sims", "10"],
    ),
    (
        "match_run",
        [
            sys.executable,
            "cli.py",
            "match",
            "run",
            "-n",
            "2",
            "--mcts-sims",
            "10",
            "--workers",
            "1",
        ],
        [
            "cmd.exe",
            "/c",
            "chess.cmd",
            "match",
            "run",
            "-n",
            "2",
            "--mcts-sims",
            "10",
            "--workers",
            "1",
        ],
    ),
    (
        "add_puzzles_list",
        [sys.executable, "cli.py", "add-puzzles", "list"],
        ["cmd.exe", "/c", "chess.cmd", "add-puzzles", "list"],
    ),
    (
        "test_fast",
        [sys.executable, "cli.py", "test", "--suite", "fast", "-k", "test_import"],
        ["cmd.exe", "/c", "chess.cmd", "test", "--suite", "fast", "-k", "test_import"],
    ),
    (
        "eval_puzzles_script_vs_cli",
        [sys.executable, "scripts/eval_puzzles_now.py", "--mcts-puzzles", "2"],
        ["cmd.exe", "/c", "chess.cmd", "puzzles", "eval", "-n", "2", "--mcts-sims", "50"],
    ),
    (
        "fetch_puzzles_script_vs_cli",
        [sys.executable, "scripts/fetch_puzzles.py"],
        ["cmd.exe", "/c", "chess.cmd", "add-puzzles", "fetch", "-n", "1000"],
    ),
    (
        "data_download_license",
        [sys.executable, "python/data/download.py", "--check-license"],
        ["cmd.exe", "/c", "chess.cmd", "download", "--check-license"],
    ),
    (
        "data_filter",
        [
            sys.executable,
            "python/data/filter.py",
            "--input",
            "data/filtered_2200.pgn",
            "--output",
            "tmp/golden_filter.pgn",
            "--max-kept",
            "5",
            "--log-interval",
            "5",
        ],
        [
            "cmd.exe",
            "/c",
            "chess.cmd",
            "filter",
            "--input",
            "data/filtered_2200.pgn",
            "--output",
            "tmp/golden_filter.pgn",
            "--max-kept",
            "5",
        ],
    ),
    (
        "data_build_dataset",
        [
            sys.executable,
            "python/data/build_dataset.py",
            "--pgn",
            "tmp/golden_filter.pgn",
            "--output-dir",
            "tmp/golden_dataset",
            "--max-games",
            "5",
        ],
        ["cmd.exe", "/c", "chess.cmd", "build-dataset", "--pgn", "tmp/golden_filter.pgn"],
    ),
    (
        "train_supervised",
        [
            sys.executable,
            "python/train/train_supervised.py",
            "--train-dir",
            "tmp/golden_dataset/train",
            "--val-dir",
            "tmp/golden_dataset/val",
            "--max-steps",
            "2",
            "--batch-size",
            "8",
            "--checkpoint-dir",
            "tmp/golden_ckpt",
            "--log-dir",
            "tmp/golden_runs_sup",
            "--no-gpu-resident",
            "--workers",
            "0",
        ],
        ["cmd.exe", "/c", "chess.cmd", "train-supervised", "--max-steps", "2"],
    ),
]


def run_and_save(key: str, mode: str, cmd: list[str]) -> tuple[int, str]:
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    proc = subprocess.run(
        cmd,
        cwd=str(ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
    )
    combined = f"COMMAND: {' '.join(cmd)}\nEXIT_CODE: {proc.returncode}\n\n=== STDOUT ===\n{proc.stdout}\n=== STDERR ===\n{proc.stderr}\n"
    out_file = AUDIT_DIR / f"{key}_{mode}.txt"
    out_file.write_text(combined, encoding="utf-8")
    return proc.returncode, combined


def main():
    print("Running audit captures...")
    summary = []
    for key, direct_cmd, chess_cmd in OPERATIONS:
        print(f"  Testing {key}...")
        rc_dir, out_dir = run_and_save(key, "direct", direct_cmd)
        rc_cmd, out_cmd = run_and_save(key, "chesscmd", chess_cmd)
        summary.append(
            {
                "key": key,
                "rc_dir": rc_dir,
                "rc_cmd": rc_cmd,
                "match": out_dir == out_cmd,
            }
        )
    print("\nAudit Capture Complete. Summary:")
    for s in summary:
        print(
            f"  {s['key']:30s} | Direct RC: {s['rc_dir']:2d} | Cmd RC: {s['rc_cmd']:2d} | Match: {s['match']}"
        )


if __name__ == "__main__":
    main()
