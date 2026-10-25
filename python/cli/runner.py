"""Subprocess execution runner with concurrent live output streaming, Ctrl+C handling, and cleanup for ZeroPawn."""

from __future__ import annotations

import collections
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Sequence, Tuple

from rich import box
from rich.panel import Panel
from rich.text import Text

from python.cli.common import REPO_ROOT
from python.cli.output_context import OutputContext, OutputMode
from python.cli.ui import console, err_console


def _stream_reader(
    stream: Any,
    is_stderr: bool,
    recent_buffer: Deque[Tuple[bool, str]],
    output_ctx: OutputContext,
    stop_event: threading.Event,
    log_fp: Optional[Any] = None,
    log_lock: Optional[threading.Lock] = None,
) -> None:
    """Read a subprocess pipe line-by-line concurrently and route through the output context."""
    try:
        for raw_line in iter(stream.readline, ""):
            if not raw_line and stop_event.is_set():
                break
            line = raw_line.rstrip("\r\n")
            if not line and not raw_line:
                break

            recent_buffer.append((is_stderr, line))
            if log_fp and log_lock:
                with log_lock:
                    try:
                        prefix = "[STDERR] " if is_stderr else "[STDOUT] "
                        log_fp.write(prefix + line + "\n")
                        log_fp.flush()
                    except Exception:
                        pass

            if output_ctx.mode == OutputMode.RICH:
                if is_stderr:
                    # Diagnostics/warning/error styling
                    if (
                        "error" in line.lower()
                        or "exception" in line.lower()
                        or "traceback" in line.lower()
                    ):
                        err_console.print(f"[bold red]! {line}[/bold red]")
                    elif "warning" in line.lower():
                        err_console.print(f"[bold yellow]! {line}[/bold yellow]")
                    else:
                        err_console.print(f"[dim]{line}[/dim]")
                else:
                    # Stdout styling
                    if line.startswith("[") and "] step=" in line:
                        # Progress line styling
                        console.print(f"[bold cyan]{line}[/bold cyan]")
                    elif line.startswith("[") and ">>>" in line:
                        console.print(f"[bold magenta]{line}[/bold magenta]")
                    elif line.startswith("=" * 10) or line.startswith("-" * 10):
                        console.print(f"[dim]{line}[/dim]")
                    elif line.startswith("STATUS:") or line.startswith("RESULTS:"):
                        console.print(f"[bold green]{line}[/bold green]")
                    elif line.startswith("ERROR:"):
                        err_console.print(f"[bold red]{line}[/bold red]")
                    else:
                        console.print(line)
            elif output_ctx.mode == OutputMode.PROTOCOL:
                # Protocol mode: keep stdout clean, stderr for diagnostics
                target = sys.stderr if is_stderr else sys.stdout
                target.write(line + "\n")
                target.flush()
            else:
                # Plain text mode
                target = sys.stderr if is_stderr else sys.stdout
                target.write(line + "\n")
                target.flush()
    except Exception:
        pass
    finally:
        try:
            stream.close()
        except Exception:
            pass
        if log_fp:
            try:
                log_fp.close()
            except Exception:
                pass


def terminate_process_tree(proc: subprocess.Popen, timeout: float = 3.0) -> None:
    """Terminate a process and its child processes safely on Windows and Unix."""
    if proc.poll() is not None:
        return

    try:
        proc.terminate()
        try:
            proc.wait(timeout=timeout)
            return
        except subprocess.TimeoutExpired:
            pass
    except Exception:
        pass

    # Escalated termination
    if sys.platform == "win32":
        try:
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=3.0,
            )
        except Exception:
            pass

    try:
        proc.kill()
        proc.wait(timeout=1.0)
    except Exception:
        pass


class SubprocessOutputBridge:
    """Manages concurrent stream consumption, live formatting, and process control for child processes."""

    def __init__(
        self,
        cmd: Sequence[str],
        op_name: str = "operation",
        output_context: Optional[OutputContext] = None,
        log_path: Optional[str | Path] = None,
        buffer_lines: int = 50,
        cwd: Optional[Path | str] = None,
        env: Optional[Dict[str, str]] = None,
    ) -> None:
        self.cmd = list(cmd)
        self.op_name = op_name
        self.output_ctx = output_context or OutputContext.get_current()
        self.log_path = Path(log_path) if log_path else (REPO_ROOT / "logs" / "zeropawn.log")
        self.buffer_lines = buffer_lines
        self.cwd = Path(cwd) if cwd else REPO_ROOT
        self.env = env
        self.recent_lines: Deque[Tuple[bool, str]] = collections.deque(maxlen=buffer_lines)
        self.stop_event = threading.Event()
        self._proc: Optional[subprocess.Popen] = None
        self._exit_code: Optional[int] = None

    def get_history(self) -> List[str]:
        """Return history of recent output lines."""
        return [line for _, line in self.recent_lines]

    def cancel(self) -> None:
        """Terminate the subprocess tree."""
        self.stop_event.set()
        if self._proc:
            terminate_process_tree(self._proc, timeout=3.0)

    def run(self, timeout: Optional[float] = None) -> int:
        """Execute the command and stream output concurrently until termination."""
        # Setup environment
        run_env = os.environ.copy() if self.env is None else self.env.copy()
        run_env["PYTHONUNBUFFERED"] = "1"
        run_env["PYTHONIOENCODING"] = "utf-8"
        run_env["ZEROPAWN_OUTPUT_MODE"] = self.output_ctx.mode.value

        # Propagate repo root and C++ extension build dirs to PYTHONPATH
        build_cpp_candidates = [
            str(REPO_ROOT),
            str(REPO_ROOT / "build" / "cpp"),
            str(REPO_ROOT / "build" / "cmake" / "cpp"),
            str(REPO_ROOT / "build" / "branch" / "cpp"),
            str(REPO_ROOT / "build"),
        ]
        existing_pythonpath = run_env.get("PYTHONPATH", "")
        run_env["PYTHONPATH"] = os.pathsep.join(
            build_cpp_candidates + ([existing_pythonpath] if existing_pythonpath else [])
        )

        # On Windows, ensure MinGW / compiler runtime DLLs can be located
        if sys.platform == "win32":
            local_app = os.environ.get("LOCALAPPDATA", "")
            compiler_dirs = [
                (
                    Path(local_app)
                    / "Microsoft"
                    / "WinGet"
                    / "Packages"
                    / "BrechtSanders.WinLibs.POSIX.UCRT_Microsoft.Winget.Source_8wekyb3d8bbwe"
                    / "mingw64"
                    / "bin"
                )
                if local_app
                else None,
                Path("C:/mingw64/bin"),
                Path("C:/msys64/mingw64/bin"),
            ]
            for cdir in compiler_dirs:
                if cdir and cdir.is_dir():
                    run_env["PATH"] = f"{cdir};{run_env.get('PATH', '')}"
                    break

        log_fp = None
        log_lock = threading.Lock()
        if self.log_path:
            try:
                self.log_path.parent.mkdir(parents=True, exist_ok=True)
                log_fp = open(self.log_path, "a", encoding="utf-8")
            except Exception:
                pass

        try:
            self._proc = subprocess.Popen(
                self.cmd,
                cwd=str(self.cwd),
                env=run_env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )

            stdout_thread = threading.Thread(
                target=_stream_reader,
                args=(
                    self._proc.stdout,
                    False,
                    self.recent_lines,
                    self.output_ctx,
                    self.stop_event,
                    log_fp,
                    log_lock,
                ),
                daemon=True,
            )
            stderr_thread = threading.Thread(
                target=_stream_reader,
                args=(
                    self._proc.stderr,
                    True,
                    self.recent_lines,
                    self.output_ctx,
                    self.stop_event,
                    log_fp,
                    log_lock,
                ),
                daemon=True,
            )
            stdout_thread.start()
            stderr_thread.start()

            self._exit_code = self._proc.wait(timeout=timeout)
            self.stop_event.set()
            stdout_thread.join(timeout=2.0)
            stderr_thread.join(timeout=2.0)
            return self._exit_code

        except subprocess.TimeoutExpired:
            self.cancel()
            self._exit_code = 124
            raise
        except Exception:
            self.cancel()
            self._exit_code = 1
            raise
        finally:
            if log_fp:
                try:
                    log_fp.close()
                except Exception:
                    pass


def execute_script_with_monitoring(
    script_rel_path: str,
    args: Sequence[str],
    op_name: str,
    description: str,
    parameters: Optional[Dict[str, Any]] = None,
    model_path: Optional[str] = None,
    cwd: Optional[Path] = None,
    timeout: Optional[float] = None,
) -> int:
    """Execute a python script in a managed subprocess with concurrent output bridge."""
    working_dir = cwd or REPO_ROOT
    script_full = REPO_ROOT / script_rel_path
    if not script_full.is_file():
        console.print(f"[bold red]Target script not found: {script_full}[/bold red]")
        return 1

    cmd: List[str] = [sys.executable, "-u", str(script_full)] + list(args)

    output_ctx = OutputContext.get_current()
    output_ctx.render_header(
        op_name=op_name,
        description=description,
        parameters=parameters,
        model_path=model_path,
    )

    t0 = time.time()
    status = "SUCCESS"
    exit_code = 0
    bridge = SubprocessOutputBridge(
        cmd=cmd,
        op_name=op_name,
        output_context=output_ctx,
        buffer_lines=50,
        cwd=working_dir,
    )

    try:
        exit_code = bridge.run(timeout=timeout)
        if exit_code != 0:
            status = "FAILED"
    except KeyboardInterrupt:
        status = "CANCELLED"
        exit_code = 130
        if output_ctx.mode == OutputMode.RICH:
            console.print(
                "\n[bold yellow]Operation cancelled by user (Ctrl+C). Terminating subprocess safely...[/bold yellow]"
            )
        else:
            sys.stderr.write(
                "\nOperation cancelled by user (Ctrl+C). Terminating subprocess safely...\n"
            )
        bridge.cancel()
    except subprocess.TimeoutExpired:
        status = "FAILED"
        exit_code = 124
        if output_ctx.mode == OutputMode.RICH:
            console.print(f"\n[bold red]Operation timed out after {timeout}s.[/bold red]")
        else:
            sys.stderr.write(f"\nOperation timed out after {timeout}s.\n")
    except Exception as e:
        status = "FAILED"
        exit_code = 1
        if output_ctx.mode == OutputMode.RICH:
            console.print(f"\n[bold red]Execution error: {e}[/bold red]")
        else:
            sys.stderr.write(f"\nExecution error: {e}\n")
    finally:
        duration = time.time() - t0
        output_ctx.render_summary(
            op_name=op_name,
            status=status,
            duration_sec=duration,
            exit_code=exit_code,
        )
        if exit_code != 0 and status != "CANCELLED" and output_ctx.mode == OutputMode.RICH:
            err_lines = [
                l
                for is_err, l in bridge.recent_lines
                if is_err or "error" in l.lower() or "exception" in l.lower()
            ]
            if err_lines:
                snippet = "\n".join(err_lines[-10:])
                console.print(
                    Panel(
                        Text(snippet, style="red"),
                        title=f"[bold red]Failure Diagnostics (Exit Code {exit_code})[/bold red]",
                        subtitle=f"Full log: {bridge.log_path}",
                        border_style="red",
                        box=box.ROUNDED,
                    )
                )

    return exit_code


def execute_subcommand(
    cmd: Sequence[str],
    op_name: str,
    description: str,
    parameters: Optional[Dict[str, Any]] = None,
    cwd: Optional[Path] = None,
    timeout: Optional[float] = None,
) -> int:
    """Execute an arbitrary subprocess command with live output streaming and Rich summary."""
    output_ctx = OutputContext.get_current()
    output_ctx.render_header(op_name=op_name, description=description, parameters=parameters)
    t0 = time.time()
    bridge = SubprocessOutputBridge(cmd=cmd, op_name=op_name, output_context=output_ctx, cwd=cwd)
    status = "SUCCESS"
    exit_code = 0
    try:
        exit_code = bridge.run(timeout=timeout)
        if exit_code != 0:
            status = "FAILED"
    except KeyboardInterrupt:
        status = "CANCELLED"
        exit_code = 130
        bridge.cancel()
    except subprocess.TimeoutExpired:
        status = "FAILED"
        exit_code = 124
    except Exception:
        status = "FAILED"
        exit_code = 1
    finally:
        output_ctx.render_summary(
            op_name=op_name, status=status, duration_sec=time.time() - t0, exit_code=exit_code
        )
    return exit_code
