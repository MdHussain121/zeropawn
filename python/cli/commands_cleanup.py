"""Cleanup, quarantine, and recovery CLI commands for ZeroPawn."""

from __future__ import annotations

import argparse
import logging
import time

from rich import box
from rich.table import Table

from python.cli.common import REPO_ROOT
from python.cli.ui import COLOR_BORDER, console, is_interactive
from python.common.quarantine import QuarantineManager

logger = logging.getLogger(__name__)


def register_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "cleanup",
        help="Safe cleanup & quarantine: dry-run manifest, transactional quarantine, and recovery.",
        description=(
            "Post-run janitor and quarantine manager. In dry-run mode (default), "
            "inventories all obsolete buffers, events, and temporary outputs, producing "
            "a detailed manifest without touching real data. Destructive moves are transactional, "
            "timestamped into quarantine/, and strictly protected against modifying models/checkpoints_scaled/best_model.pt "
            "or preserved source datasets."
        ),
    )

    # Sub-actions
    sub = parser.add_subparsers(
        dest="cleanup_action", help="Cleanup action (default: dry-run / interactive)"
    )

    # 1. Dry run
    p_dry = sub.add_parser(
        "dry-run", help="Generate detailed dry-run manifest and report (non-destructive)."
    )
    p_dry.set_defaults(func=run_dry_run)

    # 2. List archives
    p_list = sub.add_parser("list", help="List all quarantine archives in quarantine/.")
    p_list.set_defaults(func=run_list_quarantine)

    # 3. Inspect archive
    p_insp = sub.add_parser("inspect", help="Inspect manifest of a specific quarantine archive.")
    p_insp.add_argument(
        "archive", help="Name of quarantine archive directory (e.g. archive_20261014_120000)"
    )
    p_insp.set_defaults(func=run_inspect_quarantine)

    # 4. Restore archive
    p_rest = sub.add_parser(
        "restore", help="Restore files from quarantine back to active workspace."
    )
    p_rest.add_argument("archive", help="Name of quarantine archive directory")
    p_rest.add_argument(
        "--items", nargs="*", default=None, help="Optional specific artifact IDs to restore"
    )
    p_rest.add_argument(
        "--overwrite", action="store_true", help="Overwrite conflicting active destination files"
    )
    p_rest.set_defaults(func=run_restore_quarantine)

    # 5. Purge archive
    p_purge = sub.add_parser(
        "purge", help="Permanently delete a quarantined archive (strictly gated)."
    )
    p_purge.add_argument(
        "archive", help="Name of quarantine archive directory to permanently purge"
    )
    p_purge.add_argument(
        "--yes", "-y", action="store_true", help="Explicit permanent deletion confirmation"
    )
    p_purge.set_defaults(func=run_purge_quarantine)

    # Backward-compatible flags on top-level `chess cleanup`
    parser.add_argument(
        "--yes",
        "-y",
        action="store_true",
        help="Actually execute quarantine move (default is dry-run report only).",
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default=None,
        help="Path to pre-approved JSON manifest to execute quarantine from.",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="runs/phase4_rl",
        help="Run dir to clean (default: runs/phase4_rl).",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="models/checkpoints_rl",
        help="Checkpoint dir to prune (default: models/checkpoints_rl).",
    )
    parser.add_argument(
        "--keep-checkpoints",
        type=int,
        default=3,
        help="Keep newest N periodic checkpoints (default: 3).",
    )
    parser.add_argument(
        "--keep-tfevents",
        type=int,
        default=1,
        help="Keep newest N TensorBoard event files per log dir (default: 1).",
    )
    parser.add_argument(
        "--include-buffer",
        action="store_true",
        default=False,
        help="Include preallocated replay buffer memmaps (*.bin) in cleanup scope.",
    )
    parser.add_argument(
        "--include-caches",
        action="store_true",
        default=False,
        help="Include regenerable development caches (__pycache__, .pytest_cache).",
    )
    parser.set_defaults(func=run_cleanup_default)


def run_dry_run(args: argparse.Namespace) -> int:
    """Generate dry-run cleanup manifest and print summary."""
    time.time()
    mgr = QuarantineManager(REPO_ROOT)

    # 1. Verify baseline checkpoint integrity
    baseline_ok, baseline_hash = mgr.verify_baseline_checkpoint()
    if not baseline_ok:
        console.print(
            f"[bold red]CRITICAL: Protected baseline model check failed: {baseline_hash}[/bold red]"
        )
        return 1

    manifest = mgr.generate_dry_run_manifest()
    md_path, json_path = mgr.write_dry_run_reports(manifest)

    # Render summary table in Rich
    table = Table(
        title="[bold bright_cyan]Cleanup & Quarantine Dry-Run Manifest[/bold bright_cyan]",
        box=box.ROUNDED,
        border_style=COLOR_BORDER,
        header_style="bold cyan",
        padding=(0, 2),
    )
    table.add_column("Category", style="bright_white")
    table.add_column("Candidates", justify="right", style="yellow")
    table.add_column("Aggregate Size", justify="right", style="magenta")
    table.add_column("Risk Level", style="cyan")
    table.add_column("Proposed Action", style="dim")

    for cat, items in sorted(manifest.items_by_category.items()):
        cat_sz = sum(it["size_bytes"] for it in items)
        risk = "Low" if cat in ("cache", "tmp_artifact", "tfevents") else "Medium (RL Artifacts)"
        table.add_row(
            cat,
            f"{len(items):,}",
            mgr._format_bytes(cat_sz),
            risk,
            "Quarantine archive",
        )

    console.print(table)

    summary_panel = Table.grid(padding=(0, 2))
    summary_panel.add_column(style="bold cyan")
    summary_panel.add_column(style="bright_white")
    summary_panel.add_row(
        "Total Cleanup Candidates:",
        f"{manifest.total_candidates_count:,} files ({manifest.total_candidates_formatted})",
    )
    summary_panel.add_row(
        "Supervised Warm-Start Model:",
        f"VERIFIED PROTECTED (SHA256: {manifest.baseline_model_hash[:16]}...)",
    )
    summary_panel.add_row("Detailed Markdown Report:", str(md_path))
    summary_panel.add_row("Machine-Readable JSON:", str(json_path))
    summary_panel.add_row("Active Workspace Freed:", manifest.total_candidates_formatted)
    summary_panel.add_row(
        "Physical Disk Space Reclaimed:", "0 B (Same-filesystem atomic move to quarantine)"
    )

    console.print(Table(title="Workspace Space Analysis", box=box.ROUNDED))
    console.print(
        "[bold yellow]NOTE:[/] This is a non-destructive dry-run. No real files have been moved or deleted.\n"
        "To execute quarantine for approved entries, run:\n"
        "  [bold cyan]chess cleanup --yes[/bold cyan]\n"
    )
    return 0


def run_cleanup_default(args: argparse.Namespace) -> int:
    """Default handler for `chess cleanup`."""
    mgr = QuarantineManager(REPO_ROOT)

    # If --yes is not supplied, always run non-destructive dry-run
    if not getattr(args, "yes", False):
        return run_dry_run(args)

    # Check interactive confirmation
    if is_interactive():
        console.print(
            "[bold yellow]WARNING: You are about to move obsolete candidates into quarantine.[/bold yellow]"
        )
        confirm = console.input("Proceed with quarantine? [y/N] > ").strip().lower()
        if confirm not in ("y", "yes"):
            console.print("[bold cyan]Quarantine cancelled by user. No files modified.[/bold cyan]")
            return 0

    manifest = mgr.generate_dry_run_manifest()
    try:
        res = mgr.execute_quarantine(manifest, approved_by_user=True)
        console.print(
            f"[bold green]Quarantine completed successfully![/bold green]\n"
            f"  Archive: {res['archive_name']}\n"
            f"  Files Moved: {res['moved_files']:,}\n"
            f"  Active Space Freed: {mgr._format_bytes(res['active_bytes_freed'])}\n"
            f"  Quarantine Stored: {mgr._format_bytes(res['quarantine_bytes_stored'])}\n"
            f"  Physical Disk Space Reclaimed: {mgr._format_bytes(res['disk_space_reclaimed_bytes'])}\n"
        )
        return 0
    except Exception as e:
        console.print(f"[bold red]Quarantine execution failed: {e}[/bold red]")
        return 1


def run_list_quarantine(args: argparse.Namespace) -> int:
    """List all existing quarantine archives."""
    mgr = QuarantineManager(REPO_ROOT)
    archives = mgr.list_archives()
    if not archives:
        console.print("[dim]No quarantine archives found in quarantine/.[/dim]")
        return 0

    table = Table(title="Quarantine Archives", box=box.ROUNDED, border_style=COLOR_BORDER)
    table.add_column("Archive Name", style="bold cyan")
    table.add_column("Created / Executed", style="dim")
    table.add_column("Files", justify="right", style="yellow")
    table.add_column("Total Size", justify="right", style="magenta")
    table.add_column("Status", style="green")

    for a in archives:
        table.add_row(
            a["name"],
            str(a["created_at"]),
            f"{a['file_count']:,}",
            a["total_formatted"],
            a["status"],
        )
    console.print(table)
    return 0


def run_inspect_quarantine(args: argparse.Namespace) -> int:
    """Inspect a quarantine archive."""
    mgr = QuarantineManager(REPO_ROOT)
    archive_dir = mgr.quarantine_dir / args.archive
    manifest_file = archive_dir / "manifest.json"
    if not manifest_file.is_file():
        console.print(
            f"[bold red]Archive '{args.archive}' not found or missing manifest.json.[/bold red]"
        )
        return 1

    import json

    with open(manifest_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    console.print(f"[bold cyan]Archive:[/] {args.archive}")
    console.print(f"[bold cyan]Executed:[/] {data.get('executed_at', 'unknown')}")
    console.print(f"[bold cyan]Items Total:[/] {len(data.get('items', [])):,}")

    table = Table(
        title=f"Archived Items ({args.archive})", box=box.ROUNDED, border_style=COLOR_BORDER
    )
    table.add_column("ID", style="yellow")
    table.add_column("Original Path", style="bright_white")
    table.add_column("Category", style="cyan")
    table.add_column("Size", style="dim")

    for it in data.get("items", [])[:50]:
        table.add_row(it["id"], it["original_rel_path"], it["category"], it["size_formatted"])

    console.print(table)
    if len(data.get("items", [])) > 50:
        console.print(f"[dim]... and {len(data.get('items', [])) - 50} more items.[/dim]")
    return 0


def run_restore_quarantine(args: argparse.Namespace) -> int:
    """Restore files from quarantine."""
    mgr = QuarantineManager(REPO_ROOT)
    try:
        res = mgr.restore_archive(args.archive, item_ids=args.items, overwrite=args.overwrite)
        console.print(
            f"[bold green]Restoration complete![/bold green]\n"
            f"  Restored Files: {res['restored_files']}\n"
            f"  Skipped (Collisions/Missing): {res['skipped_files']}\n"
            f"  Total Restored Bytes: {res['formatted_bytes']}\n"
        )
        return 0
    except Exception as e:
        console.print(f"[bold red]Restoration failed: {e}[/bold red]")
        return 1


def run_purge_quarantine(args: argparse.Namespace) -> int:
    """Permanently delete an archive from quarantine."""
    if not args.yes:
        console.print("[bold red]Purge requires explicit authorization via --yes.[/bold red]")
        return 1

    mgr = QuarantineManager(REPO_ROOT)
    try:
        res = mgr.purge_archive(args.archive, authorized=True)
        console.print(
            f"[bold green]Archive '{args.archive}' permanently purged.[/bold green]\n"
            f"  Physical Disk Space Reclaimed: {res['formatted_reclaimed']}\n"
        )
        return 0
    except Exception as e:
        console.print(f"[bold red]Purge failed: {e}[/bold red]")
        return 1
