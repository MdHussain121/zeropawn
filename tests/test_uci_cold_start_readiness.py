"""Cold-start readiness contract for the UCI engine (pre-push audit 2026-11-05, Low).

Defect under remediation: ``isready`` emitted ``readyok`` unconditionally — even
when the designated model had failed to load — so a GUI was told the engine was
ready before the required initialization completed, and the first short
``go movetime`` could degrade silently (uniform-policy search, or a 0-simulation
legal fallback move with no announcement).

Corrected contract (python/uci/uci_engine.py):
- ``readyok`` is emitted only after the initialization required by the
  configured mode has completed (designated model loaded, or model-less
  configuration — for which readiness is book-only readiness).
- A failed initialization surfaces ``info string engine initialization
  failed: ...`` on stdout, withholds ``readyok``, keeps the read loop
  responsive, and retries the load on the next ``isready`` (recovery).
- ``go`` on a not-initialized engine fails closed: ``info string engine not
  initialized: ...`` + ``bestmove 0000`` (the UCI null move — never an illegal
  move). A search never silently downgrades to the uniform-policy evaluator.
- Any search reply move is legal; a 0-simulation or failed search announces
  its legal fallback via ``info string`` before ``bestmove``.

All waits are bounded with real timeouts; no machine-specific paths (the tiny
checkpoint lives in pytest tmp dirs, which are inside the CLI's allowed roots).
"""

from __future__ import annotations

import io
import os
import queue
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Callable, List, Optional

import pytest

import python.uci.uci_engine as uci_engine_module
from python.cli.common import REPO_ROOT
from python.uci.uci_engine import UCIEngine

# ANSI escape sequences must never appear on the UCI stdout stream.
_ANSI_PATTERN = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")

# Tokens every legal stdout line of a healthy UCI session may start with.
_ALLOWED_LINE_PREFIXES = (
    "id ",
    "option ",
    "uciok",
    "readyok",
    "info ",
    "bestmove",
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tiny_checkpoint(tmp_path_factory) -> Path:
    """A tiny but REAL ChessNet checkpoint (fast CPU load, real forward pass)."""
    import torch

    from python.model.chess_net import ChessNet

    path = tmp_path_factory.mktemp("uci_readiness") / "tiny_uci_model.pt"
    if not path.exists():
        model = ChessNet(num_blocks=2, num_filters=32)
        torch.save(
            {"model": model.state_dict(), "num_blocks": 2, "num_filters": 32, "step": 1},
            path,
        )
    return path


@pytest.fixture()
def engine_factory(tmp_path, tiny_checkpoint):
    """Build designated-checkpoint paths for UCIEngine under the test tmp dir.

    ``available=True`` copies the tiny checkpoint into the test's tmp dir;
    ``available=False`` leaves the designated path missing (unavailable-model
    scenario).
    """

    def _make(name: str = "engine.pt", available: bool = True) -> Path:
        target = tmp_path / name
        if available:
            shutil.copyfile(tiny_checkpoint, target)
        return target

    return _make


def _capture(engine: UCIEngine, commands: List[str]) -> str:
    """Run UCI lines through the engine with stdout captured; waits for any
    started search thread to finish before releasing the capture."""
    out = io.StringIO()
    old_stdout = sys.stdout
    try:
        sys.stdout = out
        for cmd in commands:
            assert engine.handle_line(cmd) is True, f"handle_line({cmd!r}) failed"
        thread = engine._search_thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=60)
        assert thread is None or not thread.is_alive(), "search thread did not finish"
    finally:
        sys.stdout = old_stdout
    return out.getvalue()


def _legal_startpos_moves() -> List[str]:
    import chess as pychess

    return [m.uci() for m in pychess.Board().legal_moves]


def _has_line(text: str, marker: str) -> bool:
    """Whole-line marker check (a checkpoint path echoed inside an info string
    can legitimately CONTAIN the substring 'readyok' — only a standalone
    'readyok' line is the protocol handshake)."""
    return any(l.strip() == marker for l in text.splitlines())


class _UciProcess:
    """Line-oriented UCI subprocess with queue-based reads (real timeouts on
    Windows, where select() on pipes is unavailable)."""

    _EOF = object()

    def __init__(self, args: List[str]) -> None:
        self.proc = subprocess.Popen(
            [sys.executable, str(REPO_ROOT / "cli.py"), "uci", *args],
            cwd=str(REPO_ROOT),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            env=os.environ.copy(),
        )
        self._lines: queue.Queue = queue.Queue()
        self._reader = threading.Thread(target=self._pump, daemon=True)
        self._reader.start()
        self.collected: List[str] = []

    def _pump(self) -> None:
        try:
            for line in self.proc.stdout:
                self._lines.put(line.rstrip("\r\n"))
        finally:
            self._lines.put(self._EOF)

    def send(self, cmd: str) -> None:
        assert self.proc.stdin is not None
        self.proc.stdin.write(cmd + "\n")
        self.proc.stdin.flush()

    def send_many(self, cmds: List[str]) -> None:
        for cmd in cmds:
            self.send(cmd)

    def wait_for(
        self,
        predicate: Callable[[List[str]], bool],
        timeout: float,
        all_lines: Optional[List[str]] = None,
    ) -> List[str]:
        """Read lines until ``predicate(lines)`` is true or a wall-clock
        ``timeout`` (seconds) expires. ``all_lines`` (if given) accumulates
        every line ever seen, in order."""
        collected: List[str] = list(self.collected)
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                line = self._lines.get(timeout=remaining)
            except queue.Empty:
                break
            if line is self._EOF:
                break
            assert isinstance(line, str)
            collected.append(line)
            self.collected.append(line)
            if all_lines is not None:
                all_lines.append(line)
            if predicate(collected):
                return collected
        return collected

    def close(self, timeout: float = 30.0) -> int:
        try:
            self.send("quit")
        except (BrokenPipeError, OSError):
            pass
        try:
            return self.proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait(timeout=10)
            raise

    def kill(self) -> None:
        if self.proc.poll() is None:
            self.proc.kill()
            self.proc.wait(timeout=10)


def _assert_protocol_clean(lines: List[str]) -> None:
    joined = "\n".join(lines)
    assert _ANSI_PATTERN.search(joined) is None, f"ANSI escape on UCI stdout: {joined!r}"
    assert "Traceback" not in joined, f"Python traceback on UCI stdout: {joined!r}"
    for line in lines:
        if not line.strip():
            continue
        assert line.startswith(_ALLOWED_LINE_PREFIXES), f"stray stdout line: {line!r}"


def _single_legal_bestmove(text: str, allow_null_with_fallback: bool = False) -> str:
    """Exactly one bestmove line; the move is legal (or an announced 0000)."""
    bestmoves = [l for l in text.splitlines() if l.startswith("bestmove")]
    assert len(bestmoves) == 1, f"expected exactly one bestmove, got: {bestmoves}"
    move = bestmoves[0].split()[1]
    if move == "0000":
        assert allow_null_with_fallback and any(
            "0 completed simulations" in l for l in text.splitlines()
        ), "bestmove 0000 without an announced 0-sim fallback violates the contract"
    else:
        assert move in _legal_startpos_moves(), f"bestmove {move!r} is not a legal move"
    return move


# ---------------------------------------------------------------------------
# 1. Cold-process end-to-end sequence (subprocess, real tiny checkpoint)
# ---------------------------------------------------------------------------


def test_cold_process_sequence_readyok_then_legal_bestmove(tiny_checkpoint):
    """Genuinely cold process: uci -> isready -> readyok -> position -> short
    ``go movetime`` -> valid bestmove -> quit, with protocol-clean output.

    The CLI constructs UCIEngine (including the required model load) BEFORE the
    stdio loop starts, so readyok cannot precede initialization by
    construction; this test pins that end to end on a real cold process.
    """
    proc = _UciProcess(["--checkpoint", str(tiny_checkpoint), "--device", "cpu", "--sims", "24"])
    all_lines: List[str] = []
    try:
        proc.send_many(["uci", "isready"])
        lines = proc.wait_for(
            lambda ls: "readyok" in ls and "uciok" in ls, timeout=120.0, all_lines=all_lines
        )
        assert "uciok" in lines, f"no uciok on cold process: {lines[-10:]}"
        assert "readyok" in lines, f"no readyok on cold process: {lines[-10:]}"

        proc.send_many(["ucinewgame", "position startpos", "go movetime 100"])
        lines = proc.wait_for(
            lambda ls: any(l.startswith("bestmove") for l in ls),
            timeout=60.0,
            all_lines=all_lines,
        )
        assert any(l.startswith("bestmove") for l in lines), f"no bestmove: {lines[-10:]}"
        _single_legal_bestmove("\n".join(lines), allow_null_with_fallback=True)

        returncode = proc.close(timeout=30.0)
        assert returncode == 0, f"engine did not exit cleanly: rc={returncode}"
        _assert_protocol_clean(all_lines)
    finally:
        proc.kill()


# ---------------------------------------------------------------------------
# 2. readyok only after the required initialization completes (ordering pin)
# ---------------------------------------------------------------------------


def test_readyok_only_after_init_load_attempt(monkeypatch, engine_factory, tiny_checkpoint):
    """The (re)load attempt happens INSIDE the isready call, before the readyok
    decision — the lazy load can no longer trail the readiness announcement.

    Events are recorded synchronously on the calling thread, so the event list
    is the true chronological order.
    """
    events: List[str] = []
    real_loader = uci_engine_module.load_engine_model

    def spying_loader(path, **kwargs):
        events.append("load-attempt")
        try:
            return real_loader(path, **kwargs)
        finally:
            events.append("load-done")

    monkeypatch.setattr(uci_engine_module, "load_engine_model", spying_loader)

    # Designated model MISSING at construction: init stays pending.
    missing = engine_factory("pending_missing.pt", available=False)
    engine = UCIEngine(checkpoint_path=str(missing), device="cpu", use_book=False)
    assert engine.model is None
    assert not engine._init_done

    events.append("isready-1")
    out = _capture(engine, ["isready"])
    events.append("after-isready-1")
    assert events == [
        "isready-1",
        "load-attempt",
        "load-done",
        "after-isready-1",
    ], f"isready must attempt the required load before deciding readiness: {events}"
    assert not _has_line(out, "readyok"), "readyok must not precede a completed required init"
    assert "info string engine initialization failed" in out

    # Repair the artifact: the next isready retries the load and only THEN
    # announces readiness.
    shutil.copyfile(tiny_checkpoint, missing)
    events.append("isready-2")
    out = _capture(engine, ["isready"])
    events.append("after-isready-2")
    assert events[-4:] == ["isready-2", "load-attempt", "load-done", "after-isready-2"]
    assert _has_line(out, "readyok"), "recovered init must emit readyok"
    assert engine.model is not None and engine._init_done

    # Readiness is memoized: a further isready must NOT re-attempt the load.
    events.clear()
    events.append("isready-3")
    out = _capture(engine, ["isready"])
    assert _has_line(out, "readyok")
    assert "load-attempt" not in events, "initialized engine must not reload on isready"


# ---------------------------------------------------------------------------
# 3. Unavailable model: no readyok, error surfaced, loop stays responsive
# ---------------------------------------------------------------------------


def test_unavailable_model_withholds_readyok_and_stays_responsive(engine_factory):
    missing = engine_factory("unavailable.pt", available=False)
    engine = UCIEngine(checkpoint_path=str(missing), device="cpu", use_book=False)

    out = _capture(engine, ["isready"])
    assert not _has_line(out, "readyok"), "failed init must not announce readiness"
    assert "info string engine initialization failed" in out
    assert engine.model is None

    # No deadlock: the read loop processed the failed handshake and is still
    # fully responsive (protocol commands and further isready retries).
    out = _capture(engine, ["uci", "isready"])
    assert _has_line(out, "uciok")
    assert not _has_line(out, "readyok")
    assert "info string engine initialization failed" in out


def test_corrupt_checkpoint_is_an_init_failure_not_ready(tmp_path):
    """A designated checkpoint that exists but cannot be deserialized is an
    initialization failure: no readyok, the reason is surfaced, and the loop
    stays responsive. Repairing the artifact recovers via the isready retry."""
    corrupt = tmp_path / "corrupt.pt"
    corrupt.write_bytes(b"this is not a torch checkpoint archive")
    engine = UCIEngine(checkpoint_path=str(corrupt), device="cpu", use_book=False)
    assert engine.model is None

    out = _capture(engine, ["isready"])
    assert not _has_line(out, "readyok"), "corrupt checkpoint must not announce readiness"
    assert "info string engine initialization failed" in out

    # Loop still responsive after the failed load.
    out = _capture(engine, ["uci"])
    assert _has_line(out, "uciok")

    # Repair with a valid checkpoint: recovery via the isready retry path.
    import torch as _torch

    from python.model.chess_net import ChessNet as _ChessNet

    model = _ChessNet(num_blocks=2, num_filters=32)
    _torch.save(
        {"model": model.state_dict(), "num_blocks": 2, "num_filters": 32, "step": 1},
        corrupt,
    )
    out = _capture(engine, ["isready"])
    assert _has_line(out, "readyok"), "repaired checkpoint must recover to ready"
    assert engine.model is not None


# ---------------------------------------------------------------------------
# 4. go on a not-initialized engine fails closed, then recovers
# ---------------------------------------------------------------------------


def test_go_fails_closed_on_unavailable_model_then_recovers(engine_factory, tiny_checkpoint):
    missing = engine_factory("failclosed_missing.pt", available=False)
    engine = UCIEngine(checkpoint_path=str(missing), device="cpu", use_book=False)

    out = _capture(engine, ["ucinewgame", "position startpos", "go movetime 50"])
    assert "info string engine not initialized" in out, out
    assert "bestmove 0000" in out, "failed init must fail closed with the UCI null move"
    assert engine._search_thread is None, "no search may start on a failed init"
    assert engine.last_engine_move is None

    # Repair: isready completes the required init, go then searches normally
    # and returns a legal move within its (short) movetime budget.
    shutil.copyfile(tiny_checkpoint, missing)
    out = _capture(engine, ["isready"])
    assert _has_line(out, "readyok")
    assert engine.model is not None

    out = _capture(engine, ["position startpos", "go movetime 200"])
    _single_legal_bestmove(out, allow_null_with_fallback=True)


# ---------------------------------------------------------------------------
# 5. Zero-simulation search: legal fallback move, honestly announced
# ---------------------------------------------------------------------------


class _ZeroSimResult:
    """Mimics the native SearchResult after an abort with 0 completed sims."""

    best_action = -1
    action_visits: List[float] = []
    root_value = 0.0
    retained_visits = 0
    new_simulations = 0


class _ZeroSimMCTS:
    """Forwarding proxy returning the zero-simulation result (the pybind MCTS
    object is read-only, so wrap it — same pattern as tests/test_uci.py)."""

    def __init__(self, inner):
        self._inner = inner

    def search(self, *args, **kwargs):
        return _ZeroSimResult()

    def __getattr__(self, name):
        return getattr(self._inner, name)


def test_zero_simulation_search_announces_legal_fallback(engine_factory):
    ckpt = engine_factory("zerosim.pt", available=True)
    engine = UCIEngine(checkpoint_path=str(ckpt), device="cpu", use_book=False)
    engine.mcts = _ZeroSimMCTS(engine.mcts)

    out = _capture(engine, ["ucinewgame", "position startpos", "go movetime 100"])
    assert "0 completed simulations" in out, out
    assert "legal fallback move" in out, out
    _single_legal_bestmove(out, allow_null_with_fallback=False)


# ---------------------------------------------------------------------------
# 6. Failed search still answers legally and announces the degradation
# ---------------------------------------------------------------------------


def test_search_failure_announces_legal_fallback(engine_factory):
    ckpt = engine_factory("failing.pt", available=True)
    engine = UCIEngine(checkpoint_path=str(ckpt), device="cpu", use_book=False)
    engine.handle_line("ucinewgame")
    engine.handle_line("position startpos")

    def failing_eval(game):
        raise RuntimeError("Injected evaluator failure (cold-start readiness test)")

    engine._eval_callback = failing_eval

    out = _capture(engine, ["go movetime 100"])
    assert "info string search failed" in out, out
    assert "legal fallback move" in out, out
    _single_legal_bestmove(out, allow_null_with_fallback=False)

    # The engine remains usable after the failure.
    out = _capture(engine, ["isready"])
    assert _has_line(out, "readyok")


# ---------------------------------------------------------------------------
# 7. Model-less / book-only operation stays ready
# ---------------------------------------------------------------------------


def _write_one_position_book(tmp_path, uci_move: str = "e2e4") -> str:
    """Synthesize a tiny one-position book: startpos -> ``uci_move``."""
    import json

    import chess as pychess

    board = pychess.Board()
    key = board.epd(en_passant="pawns")
    book = {
        "positions": {
            key: {
                "canonical_key": key,
                "fen": board.fen(),
                "side_to_move": True,
                "ply": 0,
                "moves": {
                    uci_move: {
                        "uci": uci_move,
                        "frequency": 10,
                        "qualification": "frequency",
                    }
                },
            }
        },
        "summary": {},
    }
    book_path = tmp_path / "tiny_book.json"
    book_path.write_text(json.dumps(book), encoding="utf-8")
    return str(book_path)


def test_model_less_book_only_readyok_and_book_shortcircuit(tmp_path):
    """Model-less (explicit "" signal) + book-only configuration: readyok is
    correct immediately — readiness for that configuration is book-only
    readiness — and the book short-circuit still answers a short movetime."""
    book_path = _write_one_position_book(tmp_path)
    engine = UCIEngine(checkpoint_path="", book_path=book_path, use_book=True, device="cpu")
    assert engine.model is None

    out = _capture(engine, ["isready"])
    assert _has_line(out, "readyok"), "model-less configuration must be ready by construction"
    assert "info string" not in out, "no failure may be reported for a healthy model-less engine"

    out = _capture(engine, ["ucinewgame", "position startpos", "go movetime 100"])
    assert "info string book move e2e4" in out, out
    assert "bestmove e2e4" in out, out
    assert engine._last_search_ran is False


# ---------------------------------------------------------------------------
# 8. Search cancellation still works after a cold start
# ---------------------------------------------------------------------------


def test_stop_cancellation_after_cold_start(engine_factory):
    ckpt = engine_factory("cancel.pt", available=True)
    engine = UCIEngine(checkpoint_path=str(ckpt), device="cpu", use_book=False)

    out = _capture(engine, ["isready"])
    assert _has_line(out, "readyok")

    out = io.StringIO()
    old_stdout = sys.stdout
    try:
        sys.stdout = out
        assert engine.handle_line("ucinewgame") is True
        assert engine.handle_line("position startpos") is True
        assert engine.handle_line("go movetime 60000") is True
        thread = engine._search_thread
        assert thread is not None and thread.is_alive()
        assert engine.handle_line("stop") is True
        thread.join(timeout=30.0)
        assert not thread.is_alive(), "stop must end the search promptly"
    finally:
        sys.stdout = old_stdout

    text = out.getvalue()
    _single_legal_bestmove(text, allow_null_with_fallback=True)

    # Protocol continues to work after cancellation.
    out2 = _capture(engine, ["isready"])
    assert _has_line(out2, "readyok")
