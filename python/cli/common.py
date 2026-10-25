"""Common utilities, environment setup, formatting, and path helpers for ZeroPawn CLI."""

from __future__ import annotations

import logging
import os
import re
import shutil
import sys
import tempfile
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional, Union, Tuple, Any, Sequence

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

logger = logging.getLogger(__name__)

# Third-party loggers that install their own console handlers on import
# (notably: `import torch` adds a DEBUG StreamHandler on stderr with
# propagate=False). Reclaimed by quiet_third_party_loggers() so the single
# root setup owns every console line.
THIRD_PARTY_LOGGERS = ("torch", "matplotlib", "urllib3", "PIL")


def quiet_third_party_loggers() -> None:
    """Route third-party logs through the single root setup.

    Removes foreign handlers installed by libraries at import time and
    re-enables propagation, so file gets everything and the console sees
    WARNING+ in our format only. Safe to call repeatedly and before or
    after the libraries are imported (callers re-run it after imports).
    """
    root_handlers = list(logging.getLogger().handlers)
    for name in THIRD_PARTY_LOGGERS:
        try:
            third = logging.getLogger(name)
        except (AttributeError, ValueError):
            continue
        for handler in list(third.handlers):
            if handler not in root_handlers:
                try:
                    third.removeHandler(handler)
                except (AttributeError, ValueError):
                    pass
        third.propagate = True


# Ensure stdout uses UTF-8 to prevent charmap encoding errors on Windows
try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")
except (OSError, ValueError, AttributeError) as e:
    logger.debug("stdout reconfigure skipped: %s", e)


# Locate repo root based on canonical repository markers
def get_repo_root() -> Path:
    """Locate the ZeroPawn repository root using marker files and layout heuristics.

    Discovery strategy:
    1. Search upward from this file's location for standard markers:
       'CMakeLists.txt' and 'pyproject.toml'.
    2. Check current working directory for root markers.
    3. Check canonical ZeroPawn directory layouts ('zeropawn', 'zeropawn-worktree').
    4. Fall back to legacy 'chess-rl' subdirectory if present (backward compatibility).
    """
    # 1. Walk upward from this file
    for p in [Path(__file__).resolve()] + list(Path(__file__).resolve().parents):
        if (
            (p / "CMakeLists.txt").is_file()
            and (p / "pyproject.toml").is_file()
            and (p / "python").is_dir()
        ):
            return p

    # 2. Check current working directory
    cwd = Path.cwd()
    if (
        (cwd / "CMakeLists.txt").is_file()
        and (cwd / "pyproject.toml").is_file()
        and (cwd / "python").is_dir()
    ):
        return cwd

    # 3. Canonical ZeroPawn directory layouts relative to cwd
    for canonical_name in ["zeropawn", "zeropawn-worktree"]:
        candidate = cwd / canonical_name
        if (candidate / "CMakeLists.txt").is_file() and (candidate / "pyproject.toml").is_file():
            return candidate

    # 4. Documented legacy fallback for backward compatibility
    legacy_candidate = cwd / "chess-rl"
    if (legacy_candidate / "CMakeLists.txt").is_file() and (
        legacy_candidate / "pyproject.toml"
    ).is_file():
        return legacy_candidate

    return Path(__file__).resolve().parent.parent.parent


REPO_ROOT = get_repo_root()


def ensure_cpp_extension() -> bool:
    """Verify that the native C++ extension is compiled and importable."""
    try:
        import zeropawn_cpp  # noqa: F401

        return True
    except ImportError:
        for cand in [
            REPO_ROOT / "python" / "zeropawn_cpp.pyd",
            REPO_ROOT / "build" / "cpp" / "zeropawn_cpp.pyd",
            REPO_ROOT / "build" / "cmake" / "cpp" / "zeropawn_tests.exe",
        ]:
            if cand.is_file():
                return True
        return False


# Setup paths and Windows DLLs
def setup_environment() -> None:
    root_str = str(REPO_ROOT)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)

    build_candidates = [
        REPO_ROOT / "build" / "cpp",
        REPO_ROOT / "build",
        REPO_ROOT / "build" / "Release",
        REPO_ROOT / "build" / "Debug",
        REPO_ROOT / "build" / "branch" / "cpp",
        REPO_ROOT / "build" / "cmake" / "cpp",
    ]
    for b_dir in build_candidates:
        if b_dir.is_dir() and str(b_dir) not in sys.path:
            sys.path.insert(0, str(b_dir))

    # Add WinLibs GCC bin dir for Windows
    winlibs_candidates = [
        os.path.join(
            os.environ.get("LOCALAPPDATA", ""),
            "Microsoft",
            "WinGet",
            "Packages",
            "BrechtSanders.WinLibs.POSIX.UCRT_Microsoft.Winget.Source_8wekyb3d8bbwe",
            "mingw64",
            "bin",
        ),
        r"C:\mingw64\bin",
        r"C:\msys64\mingw64\bin",
    ]
    for candidate in winlibs_candidates:
        if os.path.isdir(candidate):
            try:
                os.add_dll_directory(candidate)
            except (AttributeError, OSError):
                pass
            if candidate not in os.environ.get("PATH", ""):
                os.environ["PATH"] = f"{candidate};{os.environ.get('PATH', '')}"
            break

    setup_logging()


def setup_logging(
    log_dir: Union[str, Path] = "logs",
    log_filename: str = "zeropawn.log",
    max_bytes: int = 10 * 1024 * 1024,
    backup_count: int = 5,
    level: int = logging.INFO,
    console_level: int = logging.WARNING,
) -> logging.Logger:
    """Configure structured logging with RotatingFileHandler and stderr ConsoleHandler.

    - Console handler (WARNING+) on stderr
    - Rotating file handler for everything (level, default INFO)
    - No duplicate lines if called multiple times
    """
    log_path = REPO_ROOT / Path(log_dir)
    try:
        log_path.mkdir(parents=True, exist_ok=True)
        target_file = (log_path / log_filename).resolve()

        root_logger = logging.getLogger()
        min_level = min(level, console_level)
        if root_logger.level == 0 or root_logger.level > min_level:
            root_logger.setLevel(min_level)

        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] [%(name)s:%(lineno)d] %(message)s"
        )

        # 1. RotatingFileHandler for all logs
        has_file_handler = False
        for h in root_logger.handlers:
            if isinstance(h, RotatingFileHandler):
                try:
                    if Path(getattr(h, "baseFilename", "")).resolve() == target_file:
                        has_file_handler = True
                        h.setLevel(level)
                        break
                except Exception:
                    pass

        if not has_file_handler:
            file_handler = RotatingFileHandler(
                target_file,
                maxBytes=max_bytes,
                backupCount=backup_count,
                encoding="utf-8",
            )
            file_handler.setFormatter(formatter)
            file_handler.setLevel(level)
            root_logger.addHandler(file_handler)

        # 2. Console handler (WARNING+) on stderr
        has_console_handler = False
        for h in root_logger.handlers:
            if isinstance(h, logging.StreamHandler) and not isinstance(h, RotatingFileHandler):
                if getattr(h, "stream", None) is sys.stderr:
                    has_console_handler = True
                    h.setLevel(console_level)
                    break

        if not has_console_handler:
            console_handler = logging.StreamHandler(sys.stderr)
            console_handler.setFormatter(formatter)
            console_handler.setLevel(console_level)
            root_logger.addHandler(console_handler)

        # Reclaim third-party loggers that installed their own handlers.
        quiet_third_party_loggers()

        return root_logger
    except OSError as e:
        logger.debug("setup_logging failed to create log file: %s", e)
        return logging.getLogger()


setup_environment()

console = Console()


def print_banner(title: str, subtitle: Optional[str] = None) -> None:
    """Print a styled section banner."""
    content = f"[bold cyan]{title}[/bold cyan]"
    if subtitle:
        content += f"\n[dim]{subtitle}[/dim]"
    console.print(
        Panel(
            content,
            box=box.DOUBLE,
            border_style="bright_blue",
            padding=(0, 2),
        )
    )


def print_success(message: str) -> None:
    console.print(f"[bold green]✔[/bold green] {message}")


def print_error(message: str) -> None:
    console.print(f"[bold red]✘ Error:[/bold red] {message}")


def print_warning(message: str) -> None:
    console.print(f"[bold yellow]⚠ Warning:[/bold yellow] {message}")


def print_info(message: str) -> None:
    console.print(f"[bold blue]ℹ[/bold blue] {message}")


DEFAULT_ALLOWED_ROOTS: Tuple[Path, ...] = (
    REPO_ROOT.resolve(),
    REPO_ROOT.parent.resolve(),
    Path(tempfile.gettempdir()).resolve(),
)


def _is_subpath_of(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except (ValueError, RuntimeError):
        return False


def resolve_path(
    relative_or_absolute: Union[str, Path],
    allowed_roots: Optional[Sequence[Path]] = None,
) -> Path:
    """Resolve path relative to repo root if not absolute, checking allowlist."""
    s = str(relative_or_absolute)
    if "\x00" in s:
        raise ValueError("Invalid path: NUL byte not allowed.")
    p = Path(s)
    if not p.is_absolute():
        p = REPO_ROOT / p
    resolved = p.resolve()
    roots = (
        DEFAULT_ALLOWED_ROOTS
        if allowed_roots is None
        else tuple(Path(r).resolve() for r in allowed_roots)
    )
    if not any(_is_subpath_of(resolved, r) for r in roots):
        raise ValueError(f"Path outside allowed directories: {s!r}")
    return resolved


def safe_resolve(
    user_input: Union[str, Path],
    base: Path = REPO_ROOT,
    allowed_roots: Optional[Sequence[Path]] = None,
) -> Path:
    """Resolve user-supplied path safely inside allowed roots.

    Rejects NUL bytes, parent traversal (..), and absolute paths
    escaping allowed roots (defaults to REPO_ROOT and temp dir).
    """
    s = str(user_input)
    if "\x00" in s:
        raise ValueError("Invalid path: NUL byte not allowed.")
    p = Path(s)
    if not p.is_absolute():
        p = base / p
    resolved = p.resolve()
    roots = list(
        DEFAULT_ALLOWED_ROOTS
        if allowed_roots is None
        else [Path(r).resolve() for r in allowed_roots]
    )
    if base.resolve() not in roots:
        roots.append(base.resolve())
    if not any(_is_subpath_of(resolved, r) for r in roots):
        raise ValueError(f"Path escapes allowed directories: {s!r}")
    return resolved


def sanitize_theme(theme: Optional[str]) -> Optional[str]:
    """Strip unsafe chars from puzzle theme filter."""
    if theme is None:
        return None
    return re.sub(r"[^A-Za-z0-9_-]", "", theme)


def resolve_device(requested: Optional[str] = None) -> str:
    """Central device probe: honour valid request, else cuda-if-available."""
    if requested:
        r = requested.strip().lower()
        if r in ("cuda", "gpu"):
            try:
                import torch

                if torch.cuda.is_available() and torch.cuda.device_count() > 0:
                    return "cuda"
            except (ImportError, RuntimeError, OSError) as e:
                logger.debug("cuda probe failed: %s", e)
            return "cpu"
        if r == "cpu":
            return "cpu"
        raise ValueError(f"Invalid --device {requested!r}: use 'cuda' or 'cpu'.")
    try:
        import torch

        return "cuda" if (torch.cuda.is_available() and torch.cuda.device_count() > 0) else "cpu"
    except (ImportError, RuntimeError, OSError) as e:
        logger.debug("device probe failed: %s", e)
        return "cpu"


def validate_fen_or_raise(fen: str) -> str:
    """Basic FEN sanity check before constructing ChessGame."""
    if not isinstance(fen, str) or not fen.strip():
        raise ValueError("Empty FEN string.")
    if "\x00" in fen:
        raise ValueError("Invalid FEN: NUL byte.")
    parts = fen.strip().split()
    if len(parts) < 2:
        raise ValueError(f"Invalid FEN (expected >=2 fields): {fen!r}")
    rows = parts[0].split("/")
    if len(rows) != 8:
        raise ValueError(f"Invalid FEN board part: {fen!r}")
    try:
        import chess  # type: ignore

        chess.Board(fen)  # raises ValueError on illegal FEN
    except ImportError:
        pass
    return fen.strip()


def resolve_stockfish(path: Optional[Union[str, Path]] = None) -> Optional[Path]:
    """Allowlisted Stockfish resolver: must live under REPO_ROOT/tools, end .exe, exist."""
    tools_base = (REPO_ROOT / "tools").resolve()
    if path is not None:
        s = str(path)
        if "\x00" in s:
            raise ValueError("Invalid stockfish path: NUL byte.")
        p = Path(s)
        if not p.is_absolute():
            p = REPO_ROOT / p
        rp = p.resolve()
        try:
            rp.relative_to(tools_base)
        except ValueError:
            raise ValueError(f"Stockfish path must be inside tools/: {s!r}") from None
        if sys.platform == "win32" and rp.suffix.lower() != ".exe":
            raise ValueError(f"Stockfish path must end with .exe on Windows: {s!r}")
        if not rp.is_file():
            raise FileNotFoundError(f"Stockfish binary not found: {rp}")
        return rp
    # Default allowlist candidates
    candidates = [
        REPO_ROOT / "tools" / "stockfish.exe",
        REPO_ROOT / "tools" / "stockfish",
        REPO_ROOT / "tools" / "stockfish" / "stockfish-windows-x86-64-avx2.exe",
        REPO_ROOT / "tools" / "stockfish" / "stockfish" / "stockfish-windows-x86-64-avx2.exe",
    ]
    for c in candidates:
        if c.is_file():
            return c
    which_sf = shutil.which("stockfish")
    if which_sf:
        logger.warning("Stockfish shutil.which fallback used: %s", which_sf)
        print_warning(f"Using system Stockfish fallback: {which_sf}")
        return Path(which_sf)
    return None


def ask_int(
    prompt: str, default: int, min_val: Optional[int] = None, max_val: Optional[int] = None
) -> int:
    """REPL integer prompt with range check; falls back to default on bad input."""
    try:
        raw = console.input(prompt).strip() or str(default)
    except (EOFError, KeyboardInterrupt):
        raise
    try:
        val = int(raw)
    except (ValueError, TypeError):
        print_warning(f"Invalid integer {raw!r}; using default {default}.")
        return default
    if min_val is not None and val < min_val:
        print_warning(f"Value {val} below min {min_val}; clamped.")
        return min_val
    if max_val is not None and val > max_val:
        print_warning(f"Value {val} above max {max_val}; clamped.")
        return max_val
    return val


def ask_float(
    prompt: str, default: float, min_val: Optional[float] = None, max_val: Optional[float] = None
) -> float:
    """REPL float prompt with range check; falls back to default on bad input."""
    try:
        raw = console.input(prompt).strip() or str(default)
    except (EOFError, KeyboardInterrupt):
        raise
    try:
        val = float(raw)
    except (ValueError, TypeError):
        print_warning(f"Invalid number {raw!r}; using default {default}.")
        return default
    if min_val is not None and val < min_val:
        print_warning(f"Value {val} below min {min_val}; clamped.")
        return min_val
    if max_val is not None and val > max_val:
        print_warning(f"Value {val} above max {max_val}; clamped.")
        return max_val
    return val


def wait_for_enter(prompt: str = "\nPress [Enter] to continue...") -> None:
    """Prompt the user to press Enter, but immediately return if stdin is not a TTY."""
    if sys.stdin and hasattr(sys.stdin, "isatty") and sys.stdin.isatty():
        try:
            console.input(prompt)
        except (EOFError, KeyboardInterrupt):
            pass


def get_default_stockfish_path() -> Optional[Path]:
    """Find Stockfish executable in tools/ or system PATH."""
    candidates = [
        REPO_ROOT / "tools" / "stockfish.exe",
        REPO_ROOT / "tools" / "stockfish" / "stockfish-windows-x86-64-avx2.exe",
        REPO_ROOT / "tools" / "stockfish" / "stockfish" / "stockfish-windows-x86-64-avx2.exe",
        REPO_ROOT / "tools" / "stockfish",
    ]
    for c in candidates:
        if c.is_file():
            return c

    # Check system path (fallback, warn since outside allowlist)
    which_sf = shutil.which("stockfish")
    if which_sf:
        logger.warning("Stockfish shutil.which fallback used: %s", which_sf)
        print_warning(f"Using system Stockfish fallback: {which_sf}")
        return Path(which_sf)

    return None


def get_available_checkpoints() -> list[dict[str, Any]]:
    """Scan and return all available checkpoints sorted by step or modification time."""
    import torch

    checkpoints: list[dict[str, Any]] = []

    dirs_to_check = [
        REPO_ROOT / "models" / "checkpoints_rl",
        REPO_ROOT / "models" / "checkpoints_scaled",
        REPO_ROOT / "models" / "checkpoints",
    ]

    seen = set()
    for d in dirs_to_check:
        if not d.is_dir():
            continue
        for f in d.glob("*.pt"):
            if f.name in seen or not f.is_file():
                continue
            seen.add(f.name)
            stat = f.stat()
            size_mb = stat.st_size / (1024 * 1024)
            mtime = stat.st_mtime

            # Read lightweight metadata if possible
            step = "unknown"
            blocks = 12
            filters = 128
            try:
                ckpt_data = torch.load(f, map_location="cpu", weights_only=True)
                if isinstance(ckpt_data, dict):
                    step = ckpt_data.get("step", "unknown")
                    blocks = ckpt_data.get("num_blocks", 12)
                    filters = ckpt_data.get("num_filters", 128)
            except (OSError, RuntimeError, ValueError, EOFError) as e:
                logger.debug("checkpoint metadata read failed for %s: %s", f, e)

            checkpoints.append(
                {
                    "path": f,
                    "name": f.name,
                    "dir": d.name,
                    "size_mb": round(size_mb, 1),
                    "mtime": mtime,
                    "step": step,
                    "num_blocks": blocks,
                    "num_filters": filters,
                }
            )

    # Sort so that latest RL checkpoint is first
    def sort_key(item: dict[str, Any]) -> tuple[int, int, int, float]:
        name = item["name"]
        step = item["step"]
        step_val = (
            int(step) if isinstance(step, int) or (isinstance(step, str) and step.isdigit()) else -1
        )

        special_rank = 0
        if name == "phase4_final_checkpoint.pt":
            special_rank = 3
        elif name == "latest_checkpoint.pt":
            special_rank = 2
        elif name == "live_weights.pt":
            special_rank = 1

        is_rl = 1 if "rl" in item["dir"].lower() else 0
        return (special_rank, is_rl, step_val, item["mtime"])

    checkpoints.sort(key=sort_key, reverse=True)
    return checkpoints


def get_default_checkpoint_path() -> Optional[Path]:
    """Find best/latest checkpoint to use as default.

    Prefers fine-tuned model (strongest) over scaled baseline.
    """
    candidates = [
        REPO_ROOT / "models" / "checkpoints_finetune" / "best_model.pt",
        REPO_ROOT / "models" / "checkpoints_rl" / "phase4_final_checkpoint.pt",
        REPO_ROOT / "models" / "checkpoints_rl" / "latest_checkpoint.pt",
        REPO_ROOT / "models" / "checkpoints_scaled" / "best_model.pt",
    ]
    for c in candidates:
        if c.is_file():
            return c

    all_ckpts = get_available_checkpoints()
    if all_ckpts:
        return all_ckpts[0]["path"]

    return None


def load_engine_model(
    checkpoint_path: Union[str, Path],
    device: Optional[str] = None,
) -> Tuple[Any, dict[str, Any]]:
    """Load ChessNet from checkpoint with architecture metadata auto-detection."""
    import torch
    from python.model.chess_net import ChessNet

    if device is None:
        device = resolve_device(None)
    else:
        device = resolve_device(device)

    path = safe_resolve(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"Model checkpoint not found at: {path}")

    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, ValueError, EOFError) as e:
        logger.debug("torch.load failed for %s: %s", path, e)
        raise

    num_blocks = 12
    num_filters = 128
    step = "unknown"

    if isinstance(ckpt, dict):
        step = ckpt.get("step", "unknown")
        num_blocks = ckpt.get("num_blocks", 12)
        num_filters = ckpt.get("num_filters", 128)
        state_dict = ckpt.get("model", ckpt.get("state_dict", ckpt))
    else:
        state_dict = ckpt

    # Determine num_blocks from state_dict keys if necessary
    block_indices = set()
    for k in state_dict.keys():
        if k.startswith("blocks."):
            parts = k.split(".")
            if len(parts) > 1 and parts[1].isdigit():
                block_indices.add(int(parts[1]))
    if block_indices:
        num_blocks = max(block_indices) + 1

    model = ChessNet(num_blocks=num_blocks, num_filters=num_filters)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    info = {
        "step": step,
        "num_blocks": num_blocks,
        "num_filters": num_filters,
        "device": device,
        "path": path,
    }
    return model, info


def render_ascii_board(fen: str, highlight_move: Optional[str] = None) -> Table:
    """Render a beautiful, colorful chessboard using Rich Table."""
    parts = fen.split()
    board_part = parts[0]
    side = "White" if (len(parts) > 1 and parts[1] == "w") else "Black"
    turn_color = "bright_white" if side == "White" else "bright_yellow"

    # Parse rows
    ranks = board_part.split("/")
    table = Table(
        title=f"Chess Board  •  Side to move: [{turn_color}]{side}[/{turn_color}]",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold cyan",
        title_style="bold magenta",
        padding=(0, 1),
    )

    table.add_column("", justify="center", style="bold cyan")
    for file_char in ["a", "b", "c", "d", "e", "f", "g", "h"]:
        table.add_column(file_char, justify="center")

    # Piece color styling mapping
    # Capital: White pieces (bold white / bold cyan)
    # Lowercase: Black pieces (bold yellow / bold red)
    piece_styles = {
        "P": "[bold white]P[/bold white]",
        "N": "[bold white]N[/bold white]",
        "B": "[bold white]B[/bold white]",
        "R": "[bold white]R[/bold white]",
        "Q": "[bold white]Q[/bold white]",
        "K": "[bold white]K[/bold white]",
        "p": "[bold yellow]p[/bold yellow]",
        "n": "[bold yellow]n[/bold yellow]",
        "b": "[bold yellow]b[/bold yellow]",
        "r": "[bold yellow]r[/bold yellow]",
        "q": "[bold yellow]q[/bold yellow]",
        "k": "[bold yellow]k[/bold yellow]",
    }

    for rank_idx, rank_str in enumerate(ranks):
        rank_num = str(8 - rank_idx)
        row_cells = [rank_num]
        file_idx = 0
        for ch in rank_str:
            if ch.isdigit():
                # Empty squares
                for _ in range(int(ch)):
                    sq_color = "dim white" if (rank_idx + file_idx) % 2 == 0 else "dim grey42"
                    row_cells.append(f"[{sq_color}]·[/{sq_color}]")
                    file_idx += 1
            else:
                styled_piece = piece_styles.get(ch, ch)
                row_cells.append(styled_piece)
                file_idx += 1
        table.add_row(*row_cells)

    return table
