"""`chess finetune` — fine-tune an existing checkpoint on a picked dataset.

Datasets are listed with numbers (scanning data/dataset_*), checkpoints too.
Enter one number for a single dataset, or comma-separated numbers (e.g. "1,2")
to train on a merged mix (bins are concatenated into data/_mix_tmp).
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Optional

import numpy as np

from python.cli.common import (
    REPO_ROOT,
    console,
    print_banner,
    print_error,
    print_info,
    print_success,
)

_BIN_FILES = ("planes.bin", "moves.bin", "outcomes.bin", "game_ids.bin", "plies.bin")


def scan_datasets() -> list[dict]:
    """All data/dataset_* dirs that look like valid train/val bin sets."""
    out = []
    data_root = REPO_ROOT / "data"
    if not data_root.is_dir():
        return out
    for d in sorted(data_root.glob("dataset_*")):
        tr, va = d / "train", d / "val"
        if not (tr / "metadata.json").is_file() or not (va / "metadata.json").is_file():
            continue
        if not all((tr / f).is_file() for f in _BIN_FILES):
            continue
        try:
            tr_meta = json.loads((tr / "metadata.json").read_text(encoding="utf-8"))
            va_meta = json.loads((va / "metadata.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        out.append(
            {
                "dir": d,
                "name": d.name,
                "train_n": int(tr_meta.get("num_samples", 0)),
                "val_n": int(va_meta.get("num_samples", 0)),
                "kind": tr_meta.get("kind", "games"),
            }
        )
    return out


def scan_checkpoints() -> list[dict]:
    from python.cli.common import get_available_checkpoints

    ckpts = get_available_checkpoints()
    ckpts.sort(key=lambda c: (c["dir"], str(c["path"].name)))
    return ckpts


def merge_datasets(selection: list[dict], merge_dir: Path) -> tuple[Path, Path]:
    """Concatenate bins of the selected datasets into merge_dir (train/val)."""
    if merge_dir.exists():
        shutil.rmtree(merge_dir)
    for split in ("train", "val"):
        (merge_dir / split).mkdir(parents=True, exist_ok=True)
        total = 0
        for ds in selection:
            src = ds["dir"] / split
            meta = json.loads((src / "metadata.json").read_text(encoding="utf-8"))
            total += int(meta.get("num_samples", 0))
        # stream-copy each bin file
        for fname in _BIN_FILES:
            dtypes = {
                "planes.bin": np.float32,
                "moves.bin": np.int16,
                "outcomes.bin": np.uint8,
                "game_ids.bin": np.uint64,
                "plies.bin": np.int16,
            }
            dtype = dtypes[fname]
            if fname == "planes.bin":
                out_mm = np.memmap(
                    merge_dir / split / fname, dtype=dtype, mode="w+", shape=(total, 19, 8, 8)
                )
            else:
                out_mm = np.memmap(
                    merge_dir / split / fname, dtype=dtype, mode="w+", shape=(total,)
                )
            offset = 0
            for ds in selection:
                src = ds["dir"] / split / fname
                meta = json.loads((ds["dir"] / split / "metadata.json").read_text(encoding="utf-8"))
                n = int(meta.get("num_samples", 0))
                if fname == "planes.bin":
                    in_mm = np.memmap(src, dtype=dtype, mode="r", shape=(n, 19, 8, 8))
                else:
                    in_mm = np.memmap(src, dtype=dtype, mode="r", shape=(n,))
                out_mm[offset : offset + n] = in_mm[:]
                offset += n
                del in_mm
            assert offset == total, f"bin size mismatch in {fname}: {offset} != {total}"
            out_mm.flush()
            del out_mm
        meta = {
            "split": split,
            "num_samples": total,
            "planes_shape": [19, 8, 8],
            "dtype_planes": "float32",
            "dtype_moves": "int16",
            "dtype_outcomes": "uint8",
            "dtype_game_ids": "uint64",
            "dtype_plies": "int16",
            "merged_from": [ds["name"] for ds in selection],
        }
        (merge_dir / split / "metadata.json").write_text(
            json.dumps(meta, indent=2), encoding="utf-8"
        )
        print(f"  merged {split}: {total:,} positions from {[ds['name'] for ds in selection]}")
    return merge_dir / "train", merge_dir / "val"


def pick_numbered(
    items: list[dict], label: str, prompt_default: str = "", multi: bool = False
) -> list[dict] | None:
    """Numbered picker. Returns selected items or None on cancel."""
    if not items:
        print_error(f"No {label} found.")
        return None
    console.print(f"\n[bold cyan]Available {label}:[/bold cyan]")
    for i, it in enumerate(items, 1):
        if label == "datasets":
            console.print(
                f"  [{i}] {it['name']:<22} train {it['train_n']:>10,} · "
                f"val {it['val_n']:>9,} · {it['kind']}"
            )
        else:
            console.print(
                f"  [{i}] {it['name']:<40} ({it['dir']}, step {it['step']}, {it['size_mb']} MB)"
            )
    mode = "one or more, comma-separated (e.g. 1,2)" if multi else "number"
    raw = console.input(
        f"[bold yellow]Select {label} ({mode}) [default {prompt_default or 1}]: [/bold yellow]"
    ).strip()
    raw = raw or (prompt_default or "1")
    picked: list[dict] = []
    for part in raw.split(","):
        part = part.strip()
        try:
            idx = int(part)
        except ValueError:
            print_error(f"Not a number: {part!r}")
            return None
        if not (1 <= idx <= len(items)):
            print_error(f"Out of range: {idx}")
            return None
        if items[idx - 1] not in picked:
            picked.append(items[idx - 1])
    if not picked:
        return None
    return picked


def run_finetune(args: argparse.Namespace) -> int:
    print_banner("ZEROPAWN FINE-TUNE", "Warm-start a checkpoint on a selected dataset")

    datasets = scan_datasets()
    ckpts = scan_checkpoints()

    # --- dataset selection (flag or interactive) ---
    if args.dataset:
        picked_ds: list[dict] | None = []
        for part in str(args.dataset).split(","):
            part_str = part.strip()
            if part_str.isdigit():
                idx = int(part_str)
                if not (1 <= idx <= len(datasets)):
                    print_error(f"--dataset {idx} out of range 1-{len(datasets)}")
                    return 1
                picked_ds.append(datasets[idx - 1])
            else:
                matched = next(
                    (
                        d
                        for d in datasets
                        if d["name"].lower() == part_str.lower()
                        or part_str.lower() in d["name"].lower()
                    ),
                    None,
                )
                if matched is None:
                    avail = ", ".join(f"'{d['name']}'" for d in datasets)
                    print_error(f"Unknown dataset '{part_str}'. Available: {avail}")
                    return 1
                if matched not in picked_ds:
                    picked_ds.append(matched)
        if picked_ds:
            console.print(
                "[bold cyan]Selected Dataset(s):[/bold cyan] "
                + ", ".join(
                    f"{d['name']} ({d['train_n']:,} samples, {d['kind']})" for d in picked_ds
                )
            )
    else:
        picked_ds = pick_numbered(datasets, "datasets", multi=True)
        if picked_ds is None:
            return 1

    # --- checkpoint selection (flag or interactive) ---
    resume_path: Optional[Path] = None
    if args.resume is not None:
        if str(args.resume).isdigit():
            idx = int(args.resume)
            if not (1 <= idx <= len(ckpts)):
                print_error(f"--resume {idx} out of range 1-{len(ckpts)}")
                return 1
            resume_path = ckpts[idx - 1]["path"]
        else:
            p = Path(args.resume)
            if not p.is_file() and (REPO_ROOT / args.resume).is_file():
                p = REPO_ROOT / args.resume
            if not p.is_file():
                print_error(f"--resume not found: {args.resume}")
                return 1
            resume_path = p
    elif args.no_resume:
        print_info("Fine-tuning from RANDOM weights (--no-resume).")
    else:
        picked_ck = pick_numbered(ckpts, "checkpoints")
        if picked_ck is None:
            return 1
        resume_path = picked_ck[0]["path"]
    if resume_path is not None:
        console.print(f"[bold cyan]Checkpoint:[/bold cyan] {resume_path}")

    # --- merge if multiple datasets ---
    if len(picked_ds) > 1:
        print_info("Multiple datasets selected - merging bins (one-time copy)...")
        tr_dir, va_dir = merge_datasets(picked_ds, REPO_ROOT / "data" / "_mix_tmp")
    else:
        tr_dir = picked_ds[0]["dir"] / "train"
        va_dir = picked_ds[0]["dir"] / "val"

    # --- train ---
    from python.train.train_supervised import train_supervised

    ckpt_dir = REPO_ROOT / "models" / "checkpoints_finetune"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    metrics = train_supervised(
        train_dir=str(tr_dir),
        val_dir=str(va_dir),
        batch_size=int(args.batch_size),
        base_lr=float(args.lr),
        warmup_steps=int(args.warmup_steps),
        epochs=int(args.epochs),
        max_steps=args.steps,
        checkpoint_dir=str(ckpt_dir),
        log_dir=str(REPO_ROOT / "runs" / "finetune"),
        resume_path=str(resume_path) if resume_path else None,
        resume_weights_only=True,
        gpu_budget_gb=args.gpu_budget,
        num_workers=int(args.workers),
        gpu_resident=not args.no_gpu_resident,
    )
    print_success(
        f"Fine-tune done. val_loss={metrics.get('total_loss', float('nan')):.4f} "
        f"-> checkpoints in {ckpt_dir}"
    )
    if len(picked_ds) > 1:
        shutil.rmtree(REPO_ROOT / "data" / "_mix_tmp", ignore_errors=True)
    return 0


def register_parser(subparsers) -> None:
    parser = subparsers.add_parser(
        "finetune",
        aliases=["ft"],
        help="Fine-tune a checkpoint on a numbered dataset (tactics / elite2500 / mix).",
    )
    parser.add_argument(
        "--dataset",
        "-d",
        type=str,
        default=None,
        help="Dataset number(s), comma-separated (e.g. 2 or 1,2). Omit to pick interactively.",
    )
    parser.add_argument(
        "--resume",
        "-r",
        type=str,
        default=None,
        help="Checkpoint number (from the numbered list) or path.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        default=False,
        help="Skip checkpoint selection: fine-tune from random weights.",
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--steps", "--max-steps", dest="steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Peak LR (default 1e-4 = 1/10 of base supervised LR).",
    )
    parser.add_argument("--warmup-steps", type=int, default=200)
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="DataLoader workers for CPU loading (default 0: "
        "Windows-safe; >0 often breaks on spawn).",
    )
    parser.add_argument(
        "--gpu-budget",
        type=float,
        default=None,
        help="Cap GPU-resident dataset cache in GB (rest spills to RAM). "
        "Set ~5.0 on 8GB cards for large mixes. Default: auto.",
    )
    parser.add_argument("--no-gpu-resident", action="store_true", default=False)
    parser.set_defaults(func=run_finetune)


def main() -> int:
    parser = argparse.ArgumentParser(prog="finetune")
    register_parser(parser)
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
