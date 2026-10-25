"""Shared terminal reporter for training and self-play telemetry.

Single home for ALL structured console output (TERMINAL_SPEC):

- ``format_progress_line`` / ``ProgressReporter``: 10 s ``key=value`` lines.
- ``RateLimiter``: time-based, thread-safe emission gate for hot loops.
- ``banner`` / ``header_block`` / ``parameters_table``: one-shot framing.
- ``format_kv_line`` / ``status_line`` / ``event_line`` / ``heartbeat_line``.
- ``results_table`` / ``artifacts_list`` / ``file_record`` / ``summary_block``.
- ``error_block`` / ``warn_line``: stderr diagnostics (tracebacks stay in logs).

Rules (tested): plain ASCII-safe output (every line encodes to cp437), no
ANSI color codes emitted, report/progress on stdout, warnings/errors on
stderr, local-time ``[HH:MM:SS]`` stamps, no cursor rewriting.
"""

from __future__ import annotations

from datetime import datetime
import hashlib
import os
import sys
import threading
import time
from typing import Callable, IO, Optional, Sequence, Tuple


def format_progress_line(
    step: int,
    policy_loss: float,
    wdl_loss: float,
    buffer_size: int,
    games_count: int,
    would_resign_rate: float,
    draw_rate: float,
    cap_rate: float,
    avg_plies: float,
    sims_per_sec: float,
    gpu_util: float,
    model_version: int,
    timestamp: Optional[float] = None,
) -> str:
    """Format a single time-based training progress line.

    Format:
    [HH:MM:SS] step=... policy_loss=... wdl_loss=... buffer=... games=... resign=off would_resign_rate=... draw_rate=... cap_rate=... avg_plies=... sims_per_sec=... gpu=... model_version=...

    NOTE: resign_mode, resign_t, and false_resign_ub are explicitly removed.
    """
    ts = datetime.now() if timestamp is None else datetime.fromtimestamp(timestamp)
    time_str = ts.strftime("%H:%M:%S")

    return ascii_safe(
        f"[{time_str}] "
        f"step={step} "
        f"policy_loss={policy_loss:.4f} "
        f"wdl_loss={wdl_loss:.4f} "
        f"buffer={buffer_size} "
        f"games={games_count} "
        f"resign=off "
        f"would_resign_rate={would_resign_rate:.3f} "
        f"draw_rate={draw_rate:.3f} "
        f"cap_rate={cap_rate:.3f} "
        f"avg_plies={avg_plies:.1f} "
        f"sims_per_sec={sims_per_sec:.1f} "
        f"gpu={gpu_util:.1f}% "
        f"model_version={model_version}"
    )


def _flush_print(msg: str) -> None:
    print(ascii_safe(msg), flush=True)


def _flush_print_stderr(msg: str) -> None:
    print(ascii_safe(msg), flush=True, file=sys.stderr)


# ---------------------------------------------------------------------------
# Terminal capability helpers
# ---------------------------------------------------------------------------


def is_tty(stream: Optional[IO[str]] = None) -> bool:
    """Return True if *stream* (default stdout) is an interactive terminal."""
    if stream is None:
        stream = sys.stdout
    try:
        return bool(stream.isatty())
    except (AttributeError, ValueError):
        return False


def color_enabled(stream: Optional[IO[str]] = None) -> bool:
    """Return True if color output is allowed for *stream*.

    Color is allowed only when the stream is a TTY and NO_COLOR is unset.
    (The reporter itself never emits ANSI codes today; this gate exists so a
    future TTY-only color path cannot leak into pipes or respect NO_COLOR.)
    """
    if os.environ.get("NO_COLOR"):
        return False
    return is_tty(stream)


def ascii_safe(text: str) -> str:
    """Return *text* with every non-cp437 character replaced.

    Keeps output readable on Windows cmd code pages 437 and 65001.
    Common box-drawing/emoji input degrades to ASCII equivalents or '?'.
    """
    replacements = {
        "\u2550": "=",
        "\u2551": "|",
        "\u2554": "+",
        "\u2557": "+",
        "\u255a": "+",
        "\u255d": "+",
        "\u2560": "+",
        "\u2563": "+",
        "\u2564": "+",
        "\u2567": "+",
        "\u256c": "+",
        "\u2500": "-",
        "\u2502": "|",
        "\u250c": "+",
        "\u2510": "+",
        "\u2514": "+",
        "\u2518": "+",
        "\u251c": "+",
        "\u2524": "+",
        "\u252c": "+",
        "\u2534": "+",
        "\u253c": "+",
        "\u2192": "->",
        "\u2714": "OK",
        "\u2718": "FAIL",
        "\u2139": "i",
        "\u26a0": "!",
    }
    out = "".join(replacements.get(ch, ch) for ch in text)
    return out.encode("cp437", errors="replace").decode("cp437")


def format_time(timestamp: Optional[float] = None) -> str:
    """Local-time ``HH:MM:SS`` stamp for progress lines."""
    ts = datetime.now() if timestamp is None else datetime.fromtimestamp(timestamp)
    return ts.strftime("%H:%M:%S")


# ---------------------------------------------------------------------------
# Rate limiting (time-based, thread-safe — safe to call from hot loops)
# ---------------------------------------------------------------------------


class RateLimiter:
    """Thread-safe time-based emission gate.

    ``should_emit`` returns True at most once per ``interval_sec`` (or when
    ``force`` is True). Accepts an injected clock for deterministic tests.
    """

    def __init__(
        self,
        interval_sec: float,
        time_fn: Callable[[], float] = time.time,
    ) -> None:
        self.interval_sec = interval_sec
        self._time_fn = time_fn
        self._lock = threading.Lock()
        self.last_emit_time: float = 0.0

    def should_emit(self, force: bool = False, current_time: Optional[float] = None) -> bool:
        now = self._time_fn() if current_time is None else current_time
        with self._lock:
            if force or (now - self.last_emit_time >= self.interval_sec):
                self.last_emit_time = now
                return True
            return False

    def notify(self, current_time: Optional[float] = None) -> None:
        """Record externally-emitted output (resets silence windows)."""
        now = self._time_fn() if current_time is None else current_time
        with self._lock:
            self.last_emit_time = now


class Heartbeat:
    """30 s liveness gate: emits only when nothing else was printed.

    Call ``notify()`` after any other console output; ``maybe_emit`` returns
    a liveness line (or None) for the caller to print on stdout.
    """

    def __init__(
        self,
        interval_sec: float = 30.0,
        time_fn: Callable[[], float] = time.time,
        start_time: Optional[float] = None,
    ) -> None:
        self._limiter = RateLimiter(interval_sec, time_fn=time_fn)
        # Baseline at construction: the first liveness line goes out only
        # after a full silence window, never immediately.
        self._limiter.notify(current_time=start_time if start_time is not None else time_fn())

    def notify(self, current_time: Optional[float] = None) -> None:
        self._limiter.notify(current_time=current_time)

    def maybe_emit(
        self,
        state: str,
        current_time: Optional[float] = None,
        timestamp: Optional[float] = None,
    ) -> Optional[str]:
        if self._limiter.should_emit(current_time=current_time):
            ts = self._limiter._time_fn() if current_time is None else current_time
            return heartbeat_line(state, timestamp=ts if timestamp is None else timestamp)
        return None


# ---------------------------------------------------------------------------
# Line / block formatters (pure functions returning str — no I/O)
# ---------------------------------------------------------------------------


def format_kv_line(
    items: Sequence[Tuple[str, object]],
    timestamp: Optional[float] = None,
) -> str:
    """Format ``[HH:MM:SS] key=value ...`` with fixed caller-side key order."""
    return ascii_safe(f"[{format_time(timestamp)}] " + " ".join(f"{k}={v}" for k, v in items))


def status_line(items: Sequence[Tuple[str, object]], timestamp: Optional[float] = None) -> str:
    """Long-run live-status line (same greppable shape as progress lines)."""
    return format_kv_line(items, timestamp=timestamp)


def event_line(tag: str, message: str, timestamp: Optional[float] = None) -> str:
    """Visually distinct event line for pushes, checkpoints, evals, triggers."""
    return ascii_safe(f"[{format_time(timestamp)}] >>> {tag}: {message}")


def heartbeat_line(state: str, timestamp: Optional[float] = None) -> str:
    """Liveness line printed after 30 s of console silence."""
    return ascii_safe(f"[{format_time(timestamp)}] alive {state}")


def warn_line(message: str, timestamp: Optional[float] = None) -> str:
    """Warning line (stderr). Diagnostics, never results."""
    return ascii_safe(f"[{format_time(timestamp)}] WARNING: {message}")


def banner(title: str, subtitle: Optional[str] = None, width: int = 70) -> str:
    """ASCII operation banner (stdout). Fixed width, no content resizing."""
    bar = "=" * width
    lines = [bar, f"OPERATION: {title}", bar]
    if subtitle:
        lines.insert(2, f"{subtitle}")
    return "\n".join(ascii_safe(line) for line in lines)


def header_block(
    start_time: str,
    command: str,
    workdir: str,
    commit: str,
    config: str,
    checkpoint: str,
    device: str,
    seed: str,
) -> str:
    """Run-identity header (follows :func:`banner`). All values caller-measured."""
    lines = [
        f"START: {start_time} (local)",
        f"COMMAND: {command}",
        f"WORKDIR: {workdir}",
        f"GIT: {commit}  CONFIG: {config}  CHECKPOINT: {checkpoint}  DEVICE: {device}",
        f"SEED: {seed}",
        "-" * 70,
    ]
    return "\n".join(ascii_safe(line) for line in lines)


def parameters_table(params: Sequence[Tuple[str, object, str]]) -> str:
    """``name = value [source]`` table; source is ``user`` or ``default``."""
    if not params:
        return "PARAMETERS: none"
    width = max(len(str(name)) for name, _, _ in params)
    lines = ["PARAMETERS (name = value [source]):"]
    for name, value, source in params:
        lines.append(f"  {str(name):<{width}} = {value} [{source}]")
    return "\n".join(ascii_safe(line) for line in lines)


def results_table(rows: Sequence[Tuple[str, object]]) -> str:
    """``name: value`` results table with caller-side units in the values."""
    if not rows:
        return "RESULTS: none"
    width = max(len(str(name)) for name, _ in rows)
    lines = ["RESULTS:"]
    for name, value in rows:
        lines.append(f"  {str(name):<{width}}: {value}")
    return "\n".join(ascii_safe(line) for line in lines)


def file_record(path: str) -> Tuple[str, int, str]:
    """Return ``(path, size_bytes, sha256_hex)`` without raising.

    Missing/unreadable files report ``size=-1`` and ``sha256='missing'``.
    Chunked read keeps large checkpoints/datasets off the hot loop (call once
    at the end of a run, never per step).
    """
    try:
        digest = hashlib.sha256()
        total = 0
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                digest.update(chunk)
                total += len(chunk)
        return (str(path), total, digest.hexdigest())
    except OSError:
        return (str(path), -1, "missing")


def artifacts_list(records: Sequence[Tuple[str, int, Optional[str]]]) -> str:
    """Artifact lines: ``path  size_bytes  sha256`` (hash ``-`` if unknown)."""
    if not records:
        return "ARTIFACTS: none"
    lines = ["ARTIFACTS (path, size_bytes, sha256):"]
    for path, size, sha in records:
        lines.append(f"  {path}  {size}  {sha if sha else '-'}")
    return "\n".join(ascii_safe(line) for line in lines)


def summary_block(status: str, duration_sec: float, warnings: int, exit_code: int) -> str:
    """Footer: SUCCESS / FAILED / INTERRUPTED + duration + warnings + exit."""
    lines = [
        "-" * 70,
        f"STATUS: {status}  DURATION: {duration_sec:.1f}s  WARNINGS: {warnings}  EXIT: {exit_code}",
    ]
    return "\n".join(ascii_safe(line) for line in lines)


def error_block(
    what: str,
    likely: str,
    try_: str,
    log_path: str,
    verbose: bool = False,
    traceback_text: Optional[str] = None,
) -> str:
    """Fixed-shape error block (stderr). Traceback only with verbose=True."""
    lines = [
        f"ERROR: {what}",
        f"LIKELY: {likely}",
        f"TRY: {try_}",
        f"LOG: {log_path}",
    ]
    if verbose and traceback_text:
        lines.append("TRACEBACK:")
        lines.append(traceback_text)
    return "\n".join(ascii_safe(line) for line in lines)


# ---------------------------------------------------------------------------
# Run context (call once per run — never in a hot loop)
# ---------------------------------------------------------------------------


def git_commit(short: bool = True) -> str:
    """Short HEAD hash, or ``"unknown"`` when git is unavailable."""
    import subprocess

    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short" if short else "HEAD", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        text = out.stdout.strip()
        return text if text else "unknown"
    except (OSError, ValueError):
        return "unknown"


def run_context(operation: str) -> dict:
    """Identity snapshot for a run: start time, command, cwd, commit."""
    return {
        "operation": operation,
        "start": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "command": " ".join(sys.argv),
        "workdir": os.getcwd(),
        "commit": git_commit(),
    }


def short_hash(path: str) -> str:
    """First 8 hex chars of a file's sha256, ``"missing"`` if unreadable."""
    _, _, sha = file_record(path)
    return sha[:8] if sha not in ("missing", "") else "missing"


# ---------------------------------------------------------------------------
# Emit helpers (thin I/O wrappers — formatting stays in the pure functions)
# ---------------------------------------------------------------------------


def emit_report(text: str, output_fn: Optional[Callable[[str], None]] = None) -> str:
    """Emit report text to stdout or active output context; returns the sanitized text."""
    clean = ascii_safe(text)
    if output_fn is not None:
        output_fn(clean)
    else:
        try:
            from python.cli.output_context import OutputContext

            ctx = OutputContext.get_current()
            ctx.render_report(clean)
        except Exception:
            _flush_print(clean)
    return clean


def emit_error(text: str, output_fn: Optional[Callable[[str], None]] = None) -> str:
    """Emit diagnostics to stderr or active output context; returns the sanitized text."""
    clean = ascii_safe(text)
    if output_fn is not None:
        output_fn(clean)
    else:
        try:
            from python.cli.output_context import OutputContext

            ctx = OutputContext.get_current()
            ctx.render_error(clean)
        except Exception:
            _flush_print_stderr(clean)
    return clean


def print_error_block(
    what: str,
    likely: str,
    try_: str,
    log_path: str,
    verbose: bool = False,
    traceback_text: Optional[str] = None,
) -> str:
    """Format + emit the standard error block on stderr."""
    return emit_error(error_block(what, likely, try_, log_path, verbose, traceback_text))


class ProgressReporter:
    """Handles rate-limited time-based progress reporting (every 10s default).

    Thread-safe: worker threads may call :meth:`maybe_report` concurrently;
    at most one line is emitted per interval.
    """

    def __init__(
        self,
        interval_sec: float = 10.0,
        output_fn: Callable[[str], None] = _flush_print,
    ) -> None:
        self.interval_sec = interval_sec
        self.output_fn = output_fn
        self.last_report_time: float = 0.0
        self.last_sims_count: int = 0
        self.last_sims_time: float = 0.0
        self._limiter = RateLimiter(interval_sec)
        self._lock = threading.Lock()

    def compute_sims_per_sec(self, total_sims: int, current_time: Optional[float] = None) -> float:
        """Measure simulations per second from simulation counters over elapsed time."""
        now = time.time() if current_time is None else current_time
        with self._lock:
            if self.last_sims_time <= 0.0:
                self.last_sims_time = now
                self.last_sims_count = total_sims
                return 0.0

            dt = now - self.last_sims_time
            d_sims = total_sims - self.last_sims_count
            if dt > 0.0 and d_sims >= 0:
                sims_per_sec = d_sims / dt
            else:
                sims_per_sec = 0.0

            self.last_sims_time = now
            self.last_sims_count = total_sims
            return sims_per_sec

    def maybe_report(
        self,
        step: int,
        policy_loss: float,
        wdl_loss: float,
        buffer_size: int,
        games_count: int,
        would_resign_rate: float,
        draw_rate: float,
        cap_rate: float,
        avg_plies: float,
        total_sims: int,
        gpu_util: float,
        model_version: int,
        force: bool = False,
        current_time: Optional[float] = None,
    ) -> Optional[str]:
        """Emit a progress line if interval_sec has elapsed since last report, or if force is True."""
        now = time.time() if current_time is None else current_time
        if not self._limiter.should_emit(force=force, current_time=now):
            return None

        sims_per_sec = self.compute_sims_per_sec(total_sims, current_time=now)
        line = format_progress_line(
            step=step,
            policy_loss=policy_loss,
            wdl_loss=wdl_loss,
            buffer_size=buffer_size,
            games_count=games_count,
            would_resign_rate=would_resign_rate,
            draw_rate=draw_rate,
            cap_rate=cap_rate,
            avg_plies=avg_plies,
            sims_per_sec=sims_per_sec,
            gpu_util=gpu_util,
            model_version=model_version,
            timestamp=now,
        )
        self.last_report_time = now
        self._limiter.notify(current_time=now)
        self.output_fn(line)
        return line
