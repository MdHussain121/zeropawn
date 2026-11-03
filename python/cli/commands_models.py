"""Model discovery, inspection, validation, export, and comparison subcommands for ZeroPawn."""

from __future__ import annotations

import argparse
import hashlib
import logging
from pathlib import Path

import torch
from rich import box
from rich.table import Table

from python.cli.common import REPO_ROOT, get_default_checkpoint_path, load_engine_model
from python.cli.models import (
    compare_models,
    discover_checkpoints,
    get_default_model_path,
    inspect_model,
)
from python.cli.ui import COLOR_BORDER, console
from python.model.checkpoint_registry import DefaultCheckpointError
from python.constants import NUM_ACTIONS, NUM_PLANES, BOARD_SIZE

logger = logging.getLogger(__name__)


def register_parser(subparsers: argparse._SubParsersAction) -> None:
    models_parser = subparsers.add_parser(
        "models",
        aliases=["model"],
        help="Model management: list, inspect, validate, export, compare, and check compatibility.",
        description="Inspect, validate, and compare neural network checkpoints and architectures.",
    )
    sub = models_parser.add_subparsers(dest="models_subcommand", help="Model actions")

    # 1. List
    p_list = sub.add_parser("list", help="List all discovered checkpoints with metadata.")
    p_list.set_defaults(func=run_models_list)

    # 2. Inspect
    p_insp = sub.add_parser("inspect", help="Inspect architecture and parameters of a checkpoint.")
    p_insp.add_argument(
        "checkpoint", nargs="?", default=None, help="Path to checkpoint (default: primary model)"
    )
    p_insp.add_argument(
        "--checkpoint",
        "-c",
        dest="checkpoint_flag",
        type=str,
        default=None,
        help="Path to checkpoint",
    )
    p_insp.set_defaults(func=run_models_inspect)

    # 3. Compare
    p_comp = sub.add_parser("compare", help="Compare two model checkpoints side-by-side.")
    p_comp.add_argument("model_a", help="Path to first model checkpoint")
    p_comp.add_argument("model_b", help="Path to second model checkpoint")
    p_comp.set_defaults(func=run_models_compare)

    # 4. Validate
    p_val = sub.add_parser(
        "validate",
        help="Validate neural network weights, input/output tensors, and numerical stability.",
    )
    p_val.add_argument("checkpoint", nargs="?", default=None, help="Path to checkpoint to validate")
    p_val.add_argument(
        "--device", type=str, default="cpu", choices=["cpu", "cuda"], help="Validation device"
    )
    p_val.set_defaults(func=run_models_validate)

    # 5. Export
    p_exp = sub.add_parser(
        "export",
        help="Export model checkpoint with optional optimizer stripping and SHA256 checksum.",
    )
    p_exp.add_argument("checkpoint", help="Source checkpoint path")
    p_exp.add_argument("--output", "-o", required=True, help="Destination export path (.pt)")
    p_exp.add_argument(
        "--weights-only", action="store_true", help="Strip optimizer, scheduler, and RNG state"
    )
    p_exp.set_defaults(func=run_models_export)

    # 6. Check Compatibility
    p_chk = sub.add_parser(
        "check-compat", help="Check checkpoint architecture compatibility against current engine."
    )
    p_chk.add_argument(
        "checkpoint",
        nargs="?",
        default=None,
        help="Path to checkpoint file (default: primary model)",
    )
    p_chk.add_argument(
        "--expected-blocks", type=int, default=12, help="Expected residual block count"
    )
    p_chk.add_argument("--expected-filters", type=int, default=128, help="Expected filter count")
    p_chk.set_defaults(func=run_models_check_compat)

    # 7. Download & Verify Checkpoint
    p_dl = sub.add_parser(
        "download",
        help="Download or verify pre-trained ZeroPawn neural network checkpoints.",
    )
    p_dl.add_argument(
        "--model",
        "-m",
        type=str,
        choices=["finetuned", "scaled", "both", "baseline"],
        default="finetuned",
        help=(
            "Which model to download/verify: 'baseline' (designated, local-only), "
            "'finetuned' or 'scaled' (superseded pre-remediation lineage), or "
            "'both' (default: finetuned)"
        ),
    )
    p_dl.add_argument(
        "--url",
        type=str,
        default=None,
        help="Custom URL to download weights from (default: official release asset)",
    )
    p_dl.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Destination path (overrides default location for single-model download)",
    )
    p_dl.add_argument(
        "--verify-only",
        action="store_true",
        help="Only verify SHA-256 of existing local checkpoint(s) without downloading",
    )
    p_dl.add_argument(
        "--allow-superseded",
        action="store_true",
        help=(
            "Confirm downloading artifacts the checkpoint registry does NOT designate "
            "(superseded pre-remediation lineage; the default-checkpoint resolver will "
            "still refuse them). Required for finetuned/scaled/both."
        ),
    )
    p_dl.set_defaults(func=run_models_download)


def run_models_list(args: argparse.Namespace) -> int:
    ckpts = discover_checkpoints()
    if not ckpts:
        console.print(
            "[bold yellow]No checkpoints discovered in models/ directories.[/bold yellow]"
        )
        return 0

    table = Table(
        title="[bold bright_cyan]Discovered Model Checkpoints[/bold bright_cyan]",
        box=box.ROUNDED,
        border_style=COLOR_BORDER,
        header_style="bold cyan",
        padding=(0, 1),
    )
    table.add_column("#", justify="right", style="bold yellow")
    table.add_column("Checkpoint Name", style="bright_white")
    table.add_column("Directory", style="dim")
    table.add_column("Step", justify="right", style="cyan")
    table.add_column("Architecture", style="green")
    table.add_column("Val Loss", justify="right", style="magenta")
    table.add_column("Size", justify="right", style="dim")
    table.add_column("Modified", style="dim")

    for idx, c in enumerate(ckpts, 1):
        step_str = f"{c['step']:,}" if c["step"] is not None else "N/A"
        arch_str = f"{c['blocks']}b x {c['filters']}f" if c["blocks"] and c["filters"] else "N/A"
        loss_str = f"{c['val_loss']:.4f}" if c["val_loss"] is not None else "N/A"
        table.add_row(
            str(idx),
            c["name"],
            c["dir"],
            step_str,
            arch_str,
            loss_str,
            f"{c['size_mb']:.1f} MB",
            c["mtime"],
        )

    console.print(table)
    return 0


def run_models_inspect(args: argparse.Namespace) -> int:
    try:
        ckpt_path = (
            getattr(args, "checkpoint_flag", None) or args.checkpoint or get_default_model_path()
        )
    except DefaultCheckpointError as e:
        console.print(f"[bold red]{e}[/bold red]")
        return 1
    return inspect_model(ckpt_path)


def run_models_compare(args: argparse.Namespace) -> int:
    return compare_models(args.model_a, args.model_b)


def run_models_validate(args: argparse.Namespace) -> int:
    try:
        ckpt_path = args.checkpoint or get_default_model_path()
    except DefaultCheckpointError as e:
        console.print(f"[bold red]{e}[/bold red]")
        return 1
    p = Path(ckpt_path) if Path(ckpt_path).is_absolute() else (REPO_ROOT / ckpt_path)
    if not p.is_file():
        console.print(f"[bold red]Checkpoint file not found: {ckpt_path}[/bold red]")
        return 1

    console.print(f"[bold cyan]Validating neural network checkpoint:[/] [white]{p.name}[/white]")

    try:
        model, info = load_engine_model(p, device=args.device)
        model.eval()

        # Check weights for NaN or Inf
        has_nan = False
        has_inf = False
        param_count = 0
        for name, param in model.named_parameters():
            param_count += param.numel()
            if torch.isnan(param).any():
                has_nan = True
            if torch.isinf(param).any():
                has_inf = True

        if has_nan or has_inf:
            console.print(
                "[bold red]CRITICAL: NaN or Inf weights detected in model parameters![/bold red]"
            )
            return 1

        # Test forward pass with synthetic (1, 19, 8, 8) input
        dummy_input = torch.zeros(
            (1, NUM_PLANES, BOARD_SIZE, BOARD_SIZE), dtype=torch.float32, device=args.device
        )
        with torch.no_grad():
            policy_logits, value_logits = model(dummy_input)

        # Validate shapes
        if policy_logits.shape != (1, NUM_ACTIONS):
            console.print(
                f"[bold red]Invalid policy logits shape: {policy_logits.shape}, expected (1, {NUM_ACTIONS})[/bold red]"
            )
            return 1
        if value_logits.shape != (1, 3):
            console.print(
                f"[bold red]Invalid value logits shape: {value_logits.shape}, expected (1, 3)[/bold red]"
            )
            return 1

        # Validate softmax
        p_probs = torch.softmax(policy_logits, dim=-1)
        v_probs = torch.softmax(value_logits, dim=-1)

        p_sum = float(torch.sum(p_probs).item())
        v_sum = float(torch.sum(v_probs).item())

        if not (0.99 <= p_sum <= 1.01) or not (0.99 <= v_sum <= 1.01):
            console.print(
                f"[bold red]Softmax probability sums invalid: policy={p_sum}, value={v_sum}[/bold red]"
            )
            return 1

        table = Table(
            title="[bold green]Model Validation Results: PASS[/bold green]", box=box.ROUNDED
        )
        table.add_column("Property", style="bold cyan")
        table.add_column("Status / Measurement", style="bold green")

        table.add_row("Parameters Checked", f"{param_count:,}")
        table.add_row("Numerical Stability", "0 NaN, 0 Inf values")
        table.add_row("Forward Pass", f"OK on {args.device}")
        table.add_row("Policy Output Shape", str(tuple(policy_logits.shape)))
        table.add_row("Value Output Shape", str(tuple(value_logits.shape)))
        table.add_row("Softmax Sum Check", f"Policy: {p_sum:.4f}, Value: {v_sum:.4f}")

        console.print(table)
        return 0

    except Exception as e:
        console.print(f"[bold red]Model validation failed: {e}[/bold red]")
        return 1


def run_models_export(args: argparse.Namespace) -> int:
    src_p = (
        Path(args.checkpoint)
        if Path(args.checkpoint).is_absolute()
        else (REPO_ROOT / args.checkpoint)
    )
    out_p = Path(args.output) if Path(args.output).is_absolute() else (REPO_ROOT / args.output)

    if not src_p.is_file():
        console.print(f"[bold red]Source checkpoint not found: {src_p}[/bold red]")
        return 1

    out_p.parent.mkdir(parents=True, exist_ok=True)
    # NOTE: the `--weights-only` flag below controls the *exported payload*
    # (it strips optimizer/scheduler state); it is unrelated to torch's own
    # `weights_only` unpickling guard, which is enforced explicitly here.
    raw = torch.load(src_p, map_location="cpu", weights_only=True)
    if not isinstance(raw, dict):
        raw = {"model": raw}

    if args.weights_only:
        # Strip optimizer, scheduler, etc. The canonical loss key stays
        # `best_val_loss` (readers look for that name); `val_loss` is kept as an
        # explicit alias for older consumers (audit CKP-13).
        loss = raw.get("best_val_loss", raw.get("val_loss", None))
        export_payload = {
            "checkpoint_type": "export",
            "weights_only_container": True,
            "inference_only": True,
            "model": raw.get("model", raw),
            "num_blocks": raw.get("num_blocks", 12),
            "num_filters": raw.get("num_filters", 128),
            "step": raw.get("step", 0),
            "best_val_loss": loss,
            "val_loss": loss,
            "source_checkpoint": str(src_p),
            "exported_at": __import__("time").time(),
        }
    else:
        export_payload = raw

    from python.train.train_supervised import _atomic_torch_save

    _atomic_torch_save(export_payload, out_p)

    # Compute SHA256
    with open(out_p, "rb") as f:
        sha = hashlib.sha256(f.read()).hexdigest()

    console.print(f"[bold green]Model exported to {out_p}[/bold green]")
    console.print(f"[dim]SHA-256: {sha}[/dim]")
    return 0


def run_models_check_compat(args: argparse.Namespace) -> int:
    try:
        ckpt_target = args.checkpoint or get_default_checkpoint_path()
    except DefaultCheckpointError as e:
        console.print(f"[bold red]{e}[/bold red]")
        return 1
    ckpt_p = Path(ckpt_target) if Path(ckpt_target).is_absolute() else (REPO_ROOT / ckpt_target)
    if not ckpt_p.is_file():
        console.print(f"[bold red]Checkpoint file not found: {ckpt_p}[/bold red]")
        return 1

    try:
        raw = torch.load(ckpt_p, map_location="cpu", weights_only=True)
        if not isinstance(raw, dict):
            raw = {"model": raw}
        header_blocks = raw.get("num_blocks")
        header_filters = raw.get("num_filters")

        state = raw.get("model", raw)
        state_keys = list(state.keys()) if isinstance(state, dict) else []

        stem = state.get("stem_conv.weight") if isinstance(state, dict) else None
        stem_filters = stem.shape[0] if stem is not None and hasattr(stem, "shape") else None

        # Blocks/filters are verified from the header when present and inferred
        # from the weights otherwise, so a header-less but structurally matching
        # checkpoint is recognized (audit CKP-13: the old check compared the
        # header against the expectation only, so header-less checkpoints always
        # reported "differences" and still exited 0).
        inferred_blocks = None
        if isinstance(state, dict):
            block_ids = set()
            for key in state:
                if key.startswith("blocks."):
                    parts = str(key).split(".")
                    if len(parts) > 1 and parts[1].isdigit():
                        block_ids.add(int(parts[1]))
            if block_ids:
                inferred_blocks = len(block_ids)
        blocks = header_blocks if header_blocks is not None else inferred_blocks
        filters = header_filters if header_filters is not None else stem_filters

        table = Table(title=f"Compatibility Check: {ckpt_p.name}", box=box.ROUNDED)
        table.add_column("Property", style="bold cyan")
        table.add_column("Header", style="yellow")
        table.add_column("Weights Inference", style="green")
        table.add_column("Engine Expectation", style="dim")

        table.add_row(
            "Residual Blocks", str(header_blocks), str(inferred_blocks), str(args.expected_blocks)
        )
        table.add_row(
            "Filter Channels", str(header_filters), str(stem_filters), str(args.expected_filters)
        )
        table.add_row(
            "State Dict Keys", f"{len(state_keys)} layers", f"{len(state_keys)} layers", "Verified"
        )

        console.print(table)
        blocks_ok = blocks is not None and blocks == args.expected_blocks
        filters_ok = filters is not None and filters == args.expected_filters
        if blocks_ok and filters_ok:
            console.print(
                "[bold green]Checkpoint is 100% compatible with current engine architecture.[/bold green]"
            )
            return 0
        # Audit CKP-13: a mismatch used to exit 0 with only a warning, so scripts
        # could not gate on compatibility. Unknown architecture information is
        # also a non-zero outcome: it cannot be verified as compatible.
        unknown = [
            name
            for name, value in (("num_blocks", blocks), ("num_filters", filters))
            if value is None
        ]
        if unknown:
            console.print(
                "[bold red]Cannot verify compatibility: checkpoint carries no "
                f"{', '.join(unknown)} in its header and none could be inferred from "
                "the weights.[/bold red]"
            )
            return 1
        console.print(
            "[bold red]Incompatible: architecture differences detected "
            f"(checkpoint {blocks}x{filters} vs expected "
            f"{args.expected_blocks}x{args.expected_filters}).[/bold red]"
        )
        return 1

    except Exception as e:
        console.print(f"[bold red]Compatibility check failed: {e}[/bold red]")
        return 1


RELEASE_BASE_URL = "https://github.com/MdHussain121/zeropawn/releases/download/v0.1.0"

#: Release assets published with v0.1.0. The DESIGNATED baseline (ckpt-10,
#: models/checkpoints_baseline_20261031/best_model.pt) is deliberately NOT
#: here — it has no release asset and no download URL may be invented for it.
_RELEASE_ASSETS = {
    "finetuned_best_model.pt",
    "best_model.pt",
}


def _registry_entries_by_sha() -> dict[str, dict]:
    """Map registry SHA-256 (lowercase) -> registry entry (audit S6-02).

    Read live from docs/checkpoint_registry.json (repo checkout first, staged
    package copy fallback). Returns {} when no registry is loadable — catalog
    entries then report registry_status "unregistered" instead of crashing.
    """
    from python.model.checkpoint_registry import load_registry

    try:
        registry = load_registry(REPO_ROOT)
    except (FileNotFoundError, ValueError) as e:
        logger.warning("Checkpoint registry unavailable for model catalog: %s", e)
        return {}
    by_sha: dict[str, dict] = {}
    for entry in registry.get("checkpoints", []):
        sha = entry.get("sha256")
        if sha:
            by_sha[str(sha).lower()] = entry
    return by_sha


def _designated_entry(entries: dict[str, dict]) -> dict | None:
    """The single designated, verified registry entry (ckpt-10 today)."""
    for entry in entries.values():
        if entry.get("status") == "verified" and entry.get("designated_warm_start"):
            return entry
    return None


def build_model_catalog() -> dict[str, dict]:
    """Downloadable-checkpoint catalog annotated with registry lineage (S6-02).

    The historical catalog hardcoded availability notes that contradicted the
    checkpoint registry: its two entries map to registry records that are
    ``status: missing`` and NOT designated, while the single designated
    baseline (ckpt-10) is a local artifact with no release asset. Every entry
    now carries ``registry_id`` / ``registry_status`` / ``designated`` /
    ``lineage_note`` derived from the registry at call time, and the
    designated baseline is listed explicitly as LOCAL-ONLY.
    """
    entries = _registry_entries_by_sha()
    designated = _designated_entry(entries)
    designated_id = str(designated.get("id")) if designated else None

    catalog: dict[str, dict] = {
        "finetuned": {
            "key": "finetuned",
            "name": "Fine-Tuned Model (v0.1.0 release asset)",
            "dest": REPO_ROOT / "models" / "checkpoints_finetune" / "best_model.pt",
            "sha256": "7E326883D61BB59B9D706BCB556858F2879E713729F59ABCF046E302C8252685",
            "val_loss": 1.7975,
            "filename": "finetuned_best_model.pt",
        },
        "scaled": {
            "key": "scaled",
            "name": "Scaled Baseline Model (v0.1.0 release asset)",
            "dest": REPO_ROOT / "models" / "checkpoints_scaled" / "best_model.pt",
            "sha256": "79A72EF746F4BFDA2555FE7D0EC04970BE7CCD4A39E282E16A05594ED15BC36B",
            "val_loss": 3.1355,
            "filename": "best_model.pt",
        },
    }

    for spec in catalog.values():
        spec["available"] = spec["filename"] in _RELEASE_ASSETS
        entry = entries.get(spec["sha256"].lower())
        spec["registry_id"] = str(entry.get("id")) if entry else None
        spec["registry_status"] = str(entry.get("status")) if entry else "unregistered"
        spec["designated"] = bool(entry and entry.get("designated_warm_start"))
        provenance = str((entry or {}).get("provenance", "")).strip()
        spec["lineage_note"] = (
            "Superseded pre-remediation lineage (value-head corruption audit SLQ-01 era): "
            f"{provenance} "
            f"Designated successor: {designated_id or 'none designated'}. "
            "Downloading it does NOT make it the default — the fail-closed resolver "
            "still refuses non-designated artifacts."
        )

    if designated is not None:
        rel_file = str(designated.get("file") or designated.get("path") or "")
        catalog["baseline"] = {
            "key": "baseline",
            "name": f"Designated Baseline ({designated_id})",
            "dest": REPO_ROOT / rel_file,
            "sha256": str(designated.get("sha256") or "").upper(),
            "available": False,
            "local_artifact": True,
            "registry_id": designated_id,
            "registry_status": str(designated.get("status")),
            "designated": True,
            "lineage_note": (
                "LOCAL artifact only — ships in the operator's model deployment "
                f"({rel_file}). Not published as a release asset; no download URL "
                "exists. Restore it from the deployment, or designate a verified "
                "checkpoint in docs/checkpoint_registry.json."
            ),
        }

    return catalog


def run_models_download(args: argparse.Namespace) -> int:
    """Download or verify pre-trained ZeroPawn neural network checkpoints.

    Atomic write (tmp + os.replace) with SHA-256 validation; catalog entries
    are annotated with their checkpoint-registry lineage (S6-02). Downloading
    a superseded-lineage artifact requires ``--allow-superseded`` so a fresh
    clone never silently restores non-designated checkpoints. The designated
    baseline is a LOCAL artifact and is never downloadable.
    """
    import os
    import urllib.error
    import urllib.request

    def sha256_of(path: Path) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            while chunk := f.read(1024 * 1024):
                h.update(chunk)
        return h.hexdigest().upper()

    MODEL_CATALOG = build_model_catalog()
    # Accepted checksums = every catalog entry's recorded SHA-256 (the two
    # orphan "Step 42k warm-start" / "Tactical fine-tune" hashes of the old
    # VALID_SHAS mapped to no catalog entry and no registry record — removed).
    VALID_SHAS = {spec["sha256"].upper() for spec in MODEL_CATALOG.values() if spec.get("sha256")}

    model_choice = getattr(args, "model", "finetuned")
    verify_only = getattr(args, "verify_only", False)
    custom_output = getattr(args, "output", None)
    custom_url = getattr(args, "url", None)
    allow_superseded = getattr(args, "allow_superseded", False)

    # Determine which model targets to process
    if custom_output:
        # Explicit output file targets a single model
        spec = MODEL_CATALOG.get(model_choice if model_choice != "both" else "finetuned")
        targets = [(spec, Path(custom_output).resolve())]
    elif model_choice == "both":
        targets = [
            (MODEL_CATALOG["finetuned"], MODEL_CATALOG["finetuned"]["dest"]),
            (MODEL_CATALOG["scaled"], MODEL_CATALOG["scaled"]["dest"]),
        ]
    else:
        spec = MODEL_CATALOG.get(model_choice, MODEL_CATALOG["finetuned"])
        targets = [(spec, spec["dest"])]

    console.print(
        "\n[bold cyan]ZeroPawn Model Manager[/bold cyan] — [italic]registry "
        "lineage shown per entry; the designated baseline is a local artifact "
        "and is not downloadable[/italic]"
    )

    all_ok = True

    for spec, dest_path in targets:
        console.print(f"\n[bold bright_white]{spec['name']}[/bold bright_white]")
        console.print(f"  Target Path: [dim]{dest_path}[/dim]")
        if spec.get("local_artifact"):
            console.print(
                "  Availability: [bold red]LOCAL ONLY — not available for "
                "download[/bold red] (no release asset exists)"
            )
        else:
            console.print("  Availability: [green]v0.1.0 release asset[/green]")
        console.print(f"  Expected SHA-256: [cyan]{spec['sha256']}[/cyan]")
        console.print(
            f"  Registry: {spec.get('registry_id') or 'no registry entry'} "
            f"[{spec.get('registry_status', 'unregistered')}, "
            f"{'designated' if spec.get('designated') else 'NOT designated'}]"
        )
        console.print(f"  Lineage: [yellow]{spec['lineage_note']}[/yellow]")
        if spec.get("val_loss") is not None:
            console.print(
                f"  Recorded val loss: {spec['val_loss']} (training metric; "
                "no head-to-head strength comparison exists)"
            )

        if verify_only:
            if not dest_path.is_file():
                console.print(
                    f"  [bold red]✗ Checkpoint file not found for verification:[/bold red] {dest_path}"
                )
                all_ok = False
                continue
            console.print("  Verifying SHA-256 ...")
            digest = sha256_of(dest_path)
            if digest in VALID_SHAS:
                console.print(f"  [bold green]✓ Checksum Verified:[/bold green] {digest}")
            else:
                console.print(f"  [bold red]✗ Checksum Mismatch:[/bold red] {digest}")
                console.print(f"    Expected: {spec['sha256']}")
                all_ok = False
            continue

        # Check if already present and valid (verification of an existing
        # artifact is not acquisition — the gate below only guards downloads)
        if dest_path.is_file():
            digest = sha256_of(dest_path)
            if digest in VALID_SHAS:
                console.print(
                    f"  [bold green]✓ Checkpoint already exists and verified:[/bold green] {dest_path}"
                )
                console.print(f"    SHA-256: {digest}")
                continue

        if spec.get("local_artifact"):
            console.print(
                "  [bold red]✗ Not available for download:[/bold red] the designated "
                "baseline is a local artifact; it ships in the operator's model "
                f"deployment ({spec['dest']}) and has no release asset. Restore it "
                "from the deployment or pass an explicit --checkpoint."
            )
            all_ok = False
            continue

        superseded = not spec.get("designated") or spec.get("registry_status") != "verified"
        if superseded and not allow_superseded:
            console.print(
                f"  [bold red]✗ Refusing to download superseded-lineage artifact[/bold red] "
                f"(registry {spec.get('registry_id') or 'unregistered'}: "
                f"status={spec.get('registry_status', 'unregistered')}, "
                f"designated={spec.get('designated')})."
            )
            console.print(
                "    The default-checkpoint resolver will not use it; a fresh clone "
                "restoring it silently would run non-designated weights. Re-run with "
                "[cyan]--allow-superseded[/cyan] to confirm, or restore the designated "
                "baseline from the operator model deployment."
            )
            all_ok = False
            continue

        # Download needed
        default_release_url = f"{RELEASE_BASE_URL}/{spec['filename']}"
        download_url = (
            (custom_url if len(targets) == 1 else None)
            or os.environ.get("ZEROPAWN_MODEL_URL")
            or default_release_url
        )

        dest_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = dest_path.with_suffix(".tmp")
        console.print(f"  Downloading from: [cyan]{download_url}[/cyan]")

        try:
            req = urllib.request.Request(
                download_url,
                headers={"User-Agent": "ZeroPawn-Model-Downloader/0.1.0"},
            )
            with (
                urllib.request.urlopen(req, timeout=30) as resp,
                open(tmp_path, "wb") as out_file,
            ):
                total_size = int(resp.headers.get("Content-Length", 0))
                downloaded = 0
                h = hashlib.sha256()
                while True:
                    chunk = resp.read(1024 * 64)
                    if not chunk:
                        break
                    out_file.write(chunk)
                    h.update(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        pct = (downloaded / total_size) * 100
                        console.print(
                            f"    Progress: {pct:.1f}% ({downloaded / (1024 * 1024):.1f} MB)",
                            end="\r",
                        )

            digest = h.hexdigest().upper()
            if digest not in VALID_SHAS:
                tmp_path.unlink(missing_ok=True)
                console.print(
                    f"\n  [bold red]Download failed: SHA-256 mismatch![/bold red] Received: {digest}"
                )
                all_ok = False
                continue

            os.replace(tmp_path, dest_path)
            console.print(
                f"\n  [bold green]✓ Download and verification complete:[/bold green] {dest_path}"
            )
            console.print(f"    SHA-256: {digest}")
            console.print(
                "    [yellow]Note:[/yellow] this artifact remains non-designated in "
                "docs/checkpoint_registry.json — the default-checkpoint resolver "
                "still requires the designated baseline."
            )

        except urllib.error.URLError as e:
            tmp_path.unlink(missing_ok=True)
            console.print(f"\n  [bold red]Network error during download:[/bold red] {e}")
            all_ok = False
        except Exception as e:
            tmp_path.unlink(missing_ok=True)
            console.print(f"\n  [bold red]Failed to download checkpoint:[/bold red] {e}")
            all_ok = False

    return 0 if all_ok else 1
