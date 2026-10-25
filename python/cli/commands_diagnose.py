"""Diagnostics subcommands for ZeroPawn: resignation, lag, cache, and guardrails."""

from __future__ import annotations

import argparse

from python.cli.runner import execute_script_with_monitoring


def register_parser(subparsers: argparse._SubParsersAction) -> None:
    diag_parser = subparsers.add_parser(
        "diagnose",
        help="Diagnostic tools: resignation analysis, version lag, NN cache curve, and value signs.",
        description="Search, training, and value calibration diagnostics.",
    )
    sub = diag_parser.add_subparsers(dest="diag_subcommand", help="Diagnostic actions")

    # 1. Resign
    p_res = sub.add_parser(
        "resign",
        aliases=["resign-rates", "resignation"],
        help="Analyze resignation false-positive rates and value calibration.",
    )
    p_res.set_defaults(func=run_diagnose_resign)

    # 2. Lag
    p_lag = sub.add_parser("lag", help="Measure model version lag between trainer and workers.")
    p_lag.set_defaults(func=run_diagnose_lag)

    # 3. Cache
    p_cch = sub.add_parser("cache", help="Measure NN transposition cache entries vs ply curve.")
    p_cch.set_defaults(func=run_diagnose_cache)

    # 4. Guardrails
    p_grd = sub.add_parser(
        "guardrails", help="Verify correctness guardrails & numerical safety checks."
    )
    p_grd.set_defaults(func=run_diagnose_guardrails)

    # 5. Value Signs
    p_val = sub.add_parser(
        "value-signs",
        aliases=["color-inversion"],
        help="Test value head sign and perspective across canonical positions.",
    )
    p_val.set_defaults(func=run_diagnose_value_signs)

    # 6. Env
    p_env = sub.add_parser(
        "env", help="Inspect Python, CUDA, C++ toolchain, dependencies, and system environment."
    )
    p_env.set_defaults(func=run_diagnose_env)

    # 7. Health
    p_hlth = sub.add_parser(
        "health",
        help="Run comprehensive health check across dependencies, C++ bindings, models, and disk space.",
    )
    p_hlth.set_defaults(func=run_diagnose_health)

    # 8. Parity
    p_par = sub.add_parser(
        "parity", help="Check CLI-to-documentation and terminal execution parity."
    )
    p_par.add_argument(
        "--check-terminal",
        action="store_true",
        help="Run full terminal parity regression test suite.",
    )
    p_par.add_argument(
        "--op", type=str, default=None, help="Specific operation to test parity for."
    )
    p_par.set_defaults(func=run_diagnose_parity)

    # 9. Bundle
    p_bnd = sub.add_parser(
        "bundle", help="Collect diagnostic bundle (logs, env, health, config) into an archive."
    )
    p_bnd.add_argument(
        "--output", "-o", type=str, default=None, help="Target path for diagnostic bundle (.zip)."
    )
    p_bnd.set_defaults(func=run_diagnose_bundle)


def run_diagnose_resign(args: argparse.Namespace) -> int:
    return execute_script_with_monitoring(
        script_rel_path="scripts/diagnose_resign.py",
        args=[],
        op_name="diagnose resign",
        description="Resignation diagnosis and value head calibration",
    )


def run_diagnose_lag(args: argparse.Namespace) -> int:
    return execute_script_with_monitoring(
        script_rel_path="scripts/measure_model_lag.py",
        args=[],
        op_name="diagnose lag",
        description="Model-version lag measurement",
    )


def run_diagnose_cache(args: argparse.Namespace) -> int:
    return execute_script_with_monitoring(
        script_rel_path="scripts/measure_cache_vs_ply.py",
        args=[],
        op_name="diagnose cache",
        description="NN cache entries vs ply curve",
    )


def run_diagnose_guardrails(args: argparse.Namespace) -> int:
    return execute_script_with_monitoring(
        script_rel_path="scripts/verify_guardrails.py",
        args=[],
        op_name="diagnose guardrails",
        description="Correctness guardrails and numerical verification",
    )


def run_diagnose_value_signs(args: argparse.Namespace) -> int:
    return execute_script_with_monitoring(
        script_rel_path="scripts/test_value_signs.py",
        args=[],
        op_name="diagnose value-signs",
        description="Value head sign perspective test",
    )


def run_diagnose_env(args: argparse.Namespace) -> int:
    import platform
    import shutil
    import subprocess
    import sys
    from pathlib import Path
    import psutil
    import torch
    from python.common.progress import banner, emit_report, results_table, run_context

    ctx = run_context("diagnose env")
    emit_report(banner("diagnose env", ctx["start"] + " (local)"))

    rows = []
    rows.append(("os", f"{platform.system()} {platform.release()} ({platform.machine()})"))
    rows.append(("python_executable", sys.executable))
    rows.append(("python_version", sys.version.split()[0]))
    rows.append(("torch_version", torch.__version__))
    cuda_avail = torch.cuda.is_available()
    rows.append(("cuda_available", str(cuda_avail)))
    if cuda_avail:
        rows.append(("cuda_version", str(torch.version.cuda)))
        rows.append(("cudnn_version", str(torch.backends.cudnn.version())))
        rows.append(("device_count", str(torch.cuda.device_count())))
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            vram_gb = props.total_memory / (1024**3)
            rows.append(
                (f"gpu_{i}", f"{props.name} ({vram_gb:.2f} GB VRAM, sm_{props.major}{props.minor})")
            )

    cpu_phys = psutil.cpu_count(logical=False)
    cpu_log = psutil.cpu_count(logical=True)
    rows.append(("cpu_cores", f"{cpu_phys} physical / {cpu_log} logical"))
    vmem = psutil.virtual_memory()
    rows.append(
        (
            "system_ram",
            f"{vmem.used / (1024**3):.2f} GB used / {vmem.total / (1024**3):.2f} GB total ({vmem.percent}%)",
        )
    )

    repo_root = Path.cwd()
    disk = psutil.disk_usage(str(repo_root))
    rows.append(
        (
            "disk_free",
            f"{disk.free / (1024**3):.2f} GB free / {disk.total / (1024**3):.2f} GB total",
        )
    )

    # Toolchains
    for tool in ["g++", "gcc", "clang++", "cmake", "ninja", "git"]:
        pth = shutil.which(tool)
        if pth:
            try:
                ver = subprocess.check_output(
                    [pth, "--version"], text=True, stderr=subprocess.STDOUT
                ).splitlines()[0]
                rows.append((f"tool_{tool}", f"{ver} ({pth})"))
            except Exception:
                rows.append((f"tool_{tool}", pth))
        else:
            rows.append((f"tool_{tool}", "Not found in PATH"))

    # C++ Extension
    try:
        from python.cli.common import ensure_cpp_extension

        ext_ok = ensure_cpp_extension()
        rows.append(("cpp_extension", "Available and functional" if ext_ok else "Not available"))
    except Exception as e:
        rows.append(("cpp_extension", f"Error: {e}"))

    emit_report(results_table(rows))
    return 0


def run_diagnose_health(args: argparse.Namespace) -> int:
    import hashlib
    import sys
    from pathlib import Path
    import psutil
    import torch
    from python.common.progress import banner, emit_report, results_table, run_context
    from python.cli.common import ensure_cpp_extension, get_default_stockfish_path

    ctx = run_context("diagnose health")
    emit_report(banner("diagnose health", ctx["start"] + " (local)"))

    checks = []
    overall_ok = True

    # 1. Python version >= 3.10
    py_ok = sys.version_info >= (3, 10)
    checks.append(
        (
            "python_version",
            f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            "PASS" if py_ok else "FAIL",
        )
    )
    if not py_ok:
        overall_ok = False

    # 2. PyTorch & CUDA
    cuda_ok = torch.cuda.is_available()
    checks.append(
        ("torch_cuda", f"CUDA available: {cuda_ok}", "PASS" if cuda_ok else "WARN (CPU mode)")
    )

    # 3. C++ extension
    ext_ok = ensure_cpp_extension()
    checks.append(
        ("cpp_extension", "chess_cpp native module", "PASS" if ext_ok else "WARN (fallback)")
    )

    # 4. Protected baseline checkpoint
    PROTECTED_PATH = Path("models/checkpoints_scaled/best_model.pt")
    VALID_SHAS = {
        "54D462823837F77C60621FB72E1A1474ECA3DC9DBB2A2969D87CED18ADA2B7DF",
        "5AC334C8DC2E36A2577473514641929EDABFDB73D9B09FF19B455C36DDC0A038",
        "79A72EF746F4BFDA2555FE7D0EC04970BE7CCD4A39E282E16A05594ED15BC36B",
    }
    if PROTECTED_PATH.exists():
        h = hashlib.sha256()
        with open(PROTECTED_PATH, "rb") as f:
            while chunk := f.read(1024 * 1024):
                h.update(chunk)
        actual_sha = h.hexdigest().upper()
        if actual_sha in VALID_SHAS:
            checks.append(("protected_baseline", f"{PROTECTED_PATH} SHA256 verified", "PASS"))
        else:
            checks.append(
                ("protected_baseline", f"Checksum mismatch! {actual_sha[:16]}...", "FAIL")
            )
            overall_ok = False
    else:
        checks.append(("protected_baseline", f"{PROTECTED_PATH} missing", "FAIL"))
        overall_ok = False

    # 5. Stockfish
    sf_path = get_default_stockfish_path()
    if sf_path and sf_path.exists():
        checks.append(("stockfish_engine", str(sf_path), "PASS"))
    else:
        checks.append(("stockfish_engine", "Stockfish not found", "WARN"))

    # 6. Disk space >= 2.0 GB
    disk = psutil.disk_usage(str(Path.cwd()))
    free_gb = disk.free / (1024**3)
    if free_gb >= 5.0:
        checks.append(("disk_space", f"{free_gb:.2f} GB free", "PASS"))
    elif free_gb >= 1.0:
        checks.append(("disk_space", f"{free_gb:.2f} GB free", "WARN (low disk space)"))
    else:
        checks.append(("disk_space", f"{free_gb:.2f} GB free", "FAIL (< 1 GB)"))
        overall_ok = False

    # 7. System RAM
    vmem = psutil.virtual_memory()
    avail_gb = vmem.available / (1024**3)
    if avail_gb >= 2.0:
        checks.append(("system_memory", f"{avail_gb:.2f} GB available", "PASS"))
    else:
        checks.append(("system_memory", f"{avail_gb:.2f} GB available", "WARN (low memory)"))

    # 8. Data and runs directories
    for dir_name in ["data", "models", "runs"]:
        p = Path(dir_name)
        if p.exists() and p.is_dir():
            checks.append((f"dir_{dir_name}", "Exists and accessible", "PASS"))
        else:
            checks.append((f"dir_{dir_name}", "Does not exist", "WARN"))

    rows = [(name, f"[{status}] {desc}") for name, desc, status in checks]
    rows.append(("OVERALL_HEALTH", "PASSED" if overall_ok else "ATTENTION REQUIRED"))
    emit_report(results_table(rows))
    return 0 if overall_ok else 1


def run_diagnose_parity(args: argparse.Namespace) -> int:
    import argparse
    from python.common.progress import banner, emit_report, results_table, run_context

    ctx = run_context("diagnose parity")
    emit_report(banner("diagnose parity", ctx["start"] + " (local)"))

    if getattr(args, "check_terminal", False):
        import subprocess
        import sys

        cmd = [sys.executable, "scripts/check_terminal_parity.py"]
        if getattr(args, "op", None):
            cmd.extend(["--op", args.op])
        res = subprocess.run(cmd)
        return res.returncode

    # Internal Catalog vs CLI Parity Check
    from python.cli.operations import OPERATIONS_CATALOG
    from python.cli.main import build_parser

    parser = build_parser()
    subparsers_actions = [
        action for action in parser._actions if isinstance(action, argparse._SubParsersAction)
    ]
    discovered_cli_commands = set()
    if subparsers_actions:
        for sa in subparsers_actions:
            discovered_cli_commands.update(sa.choices.keys())

    catalog_ops = set(OPERATIONS_CATALOG.keys())
    catalog_verbs = {
        op.direct_command.split()[1]
        for op in OPERATIONS_CATALOG.values()
        if len(op.direct_command.split()) > 1
    }
    uncovered_verbs = discovered_cli_commands - catalog_verbs - {"interactive"}
    missing_verbs = catalog_verbs - discovered_cli_commands

    parity_ok = (len(missing_verbs) == 0) and (len(catalog_ops) >= 42)
    rows = [
        ("total_catalog_operations", str(len(catalog_ops))),
        ("total_cli_top_level_subcommands", str(len(discovered_cli_commands))),
        (
            "cli_commands_cataloged",
            "YES" if not uncovered_verbs else f"Uncataloged: {uncovered_verbs}",
        ),
        (
            "catalog_ops_available",
            "YES" if not missing_verbs else f"Missing in CLI: {missing_verbs}",
        ),
        ("parity_status", "PASS" if parity_ok else "FAIL"),
    ]
    emit_report(results_table(rows))
    return 0 if parity_ok else 1


def run_diagnose_bundle(args: argparse.Namespace) -> int:
    import datetime
    import json
    import platform
    import sys
    import zipfile
    from pathlib import Path
    import psutil
    import torch
    from python.common.progress import banner, emit_report, results_table, run_context

    ctx = run_context("diagnose bundle")
    emit_report(banner("diagnose bundle", ctx["start"] + " (local)"))

    out_path = getattr(args, "output", None)
    if not out_path:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        bundle_dir = Path("artifacts/diagnostics")
        bundle_dir.mkdir(parents=True, exist_ok=True)
        out_file = bundle_dir / f"diag_bundle_{ts}.zip"
    else:
        out_file = Path(out_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)

    # Gather data
    sysinfo = {
        "timestamp": datetime.datetime.now().isoformat(),
        "platform": platform.platform(),
        "python": sys.version,
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "cpu_count": psutil.cpu_count(logical=True),
        "ram_gb": psutil.virtual_memory().total / (1024**3),
        "cwd": str(Path.cwd()),
    }

    with zipfile.ZipFile(out_file, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        # 1. System info
        zf.writestr("system_info.json", json.dumps(sysinfo, indent=2))

        # 2. Presets
        presets_path = Path("configs/presets.json")
        if presets_path.exists():
            zf.write(presets_path, arcname="configs/presets.json")

        # 3. Recent logs
        for log_file in Path("runs").glob("*.log"):
            if log_file.is_file():
                zf.write(log_file, arcname=f"runs/{log_file.name}")
        for log_file in Path("logs").glob("*.log"):
            if log_file.is_file():
                zf.write(log_file, arcname=f"logs/{log_file.name}")

    rows = [
        ("bundle_archive", str(out_file)),
        ("archive_size_bytes", str(out_file.stat().st_size)),
        ("status", "SUCCESS"),
    ]
    emit_report(results_table(rows))
    return 0
