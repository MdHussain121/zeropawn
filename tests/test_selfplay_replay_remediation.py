"""Regression tests for the 2026-10-31 self-play & replay-buffer remediation.

Covers, per finding:

- SR-01  shutdown/abort semantics: an interrupted game is never finalized as a
         draw, never sunk, and never counted as a draw; every shutdown path
         (book phase, mid-search, between recording and result assignment,
         after a genuine end, worker exception, repeated/concurrent stops)
         is exercised deterministically with stub aggregators/books.
- SR-12  per-game RNG isolation for unseeded runs.
- SR-14  search-mode support assertion vs. actual serial enforcement.
- SR-16  mover-relative outcome propagation pinned as a pure function.
- SR-17  the legacy IPC/multiprocessing paths are retained carriers whose
         import surface and serving loop stay outside production dispatch.

All tests are deterministic and use synthetic fixtures; no production
experience is generated and the historical replay buffer is never touched.
"""

from __future__ import annotations

import ast
import random
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pytest

import python  # noqa: F401  (repo-root package on sys.path)

from python.selfplay.worker import (
    GameSummary,
    InferenceError,
    SelfPlayGame,
    SelfPlayWorker,
    TERMINATION_ABORTED,
    WorkerBatchAggregator,
    outcomes_from_result,
)

GENUINE_TERMINATIONS = {
    "resignation",
    "tablebase_adjudication",
    "checkmate",
    "max_plies_cap",
    "stalemate",
    "50_move_rule",
    "repetition",
    "insufficient_material",
    "draw_other",
    "connect4_win_1",
    "connect4_win_2",
    "connect4_draw",
}


# ---------------------------------------------------------------------------
# Deterministic stubs
# ---------------------------------------------------------------------------


class StubAggregator:
    """Minimal ``WorkerBatchAggregator`` stand-in for engine-level tests.

    Serves a valid uniform policy (the C++ search masks + renormalizes over
    legal moves, so uniform is always a legal distribution) and a neutral
    value. ``on_eval`` runs inside the evaluation callback, which lets a test
    flip ``running`` / raise exactly between the record capture and the next
    loop-condition check.
    """

    def __init__(self, running: bool = True) -> None:
        self.running = running
        self.max_batch_size = 8
        self.last_model_version = 1
        self.evaluate_calls = 0
        self.on_eval = None

    def evaluate_game_state(self, game: Any) -> Tuple[np.ndarray, float]:
        self.evaluate_calls += 1
        if self.on_eval is not None:
            self.on_eval(self.evaluate_calls)
        return np.full(4672, 1.0 / 4672.0, dtype=np.float32), 0.0


class StubOpeningBook:
    """Book that returns the engine's first legal move, with a lookup hook."""

    def __init__(self, stop_after: Optional[int] = None) -> None:
        self.enabled = True
        self.max_depth_plies = 50
        self.positions: Dict[str, Any] = {"stub": True}
        self.on_lookup = None
        self.stop_after = stop_after

    def lookup(self, game: Any, current_ply: int = 0):
        if self.on_lookup is not None:
            self.on_lookup(current_ply)
        if self.stop_after is not None and current_ply >= self.stop_after:
            return None
        legal = game.legal_actions()
        if not legal:
            return None
        return game.action_to_string(legal[0]), {}


def _make_game(
    aggregator: StubAggregator,
    *,
    max_plies: int = 40,
    seed: Optional[int] = 7,
    book: Optional[StubOpeningBook] = None,
    on_sims=None,
) -> SelfPlayGame:
    game = SelfPlayGame(
        game_id=101,
        aggregator=aggregator,  # type: ignore[arg-type]
        game_type="chess",
        full_search_sims=4,
        fast_search_sims=2,
        full_search_prob=0.5,
        temp_cutoff_range=(2, 3),
        max_plies=max_plies,
        use_book=False,
        allow_resignation=False,
        min_resign_ply=10_000,  # resignation can never fire in these tests
        seed=seed,
        on_sims=on_sims,
    )
    if book is not None:
        game.opening_book = book
        game.use_book = True
    return game


# ---------------------------------------------------------------------------
# SR-01 — engine-level abort semantics
# ---------------------------------------------------------------------------


def test_shutdown_during_book_moves_aborts_without_result():
    """The audit's trigger: stop lands while the game is in its book phase.

    No evaluation is awaited during book plies, so the loop exits on
    ``aggregator.running == False`` — before the fix this fell through to
    ``draw_other`` and the partial positions were sunk as draw-labeled data.
    """
    agg = StubAggregator(running=True)
    book = StubOpeningBook(stop_after=3)

    def _flip(ply: int) -> None:
        if ply == 1:
            agg.running = False  # shutdown arrives between book plies

    book.on_lookup = _flip
    game = _make_game(agg, book=book, max_plies=40)

    positions, summary = game.play_game()

    assert summary.aborted is True
    assert summary.termination_reason == TERMINATION_ABORTED
    assert summary.result == "*"
    assert summary.completed is False
    assert positions == []  # no partial experience leaves the engine
    assert game.ply >= 1  # the book move really was played
    assert agg.evaluate_calls == 0  # never reached a search: pure book phase


def test_shutdown_between_recording_and_result_assignment_discards_positions():
    """Stop lands after a position was recorded but before the game ended."""
    agg = StubAggregator(running=True)

    def _flip(call: int) -> None:
        if call == 1:
            agg.running = False

    agg.on_eval = _flip
    game = _make_game(agg, max_plies=40)

    positions, summary = game.play_game()

    assert len(game.raw_positions) == 1  # the position WAS recorded in-loop
    assert positions == []  # ...and was then discarded by the abort guard
    assert summary.aborted is True
    assert summary.termination_reason == TERMINATION_ABORTED


def test_genuine_completion_is_preserved_when_stop_races_the_last_ply():
    """A game that reaches a genuine end (here the ply cap) keeps its result.

    Running goes false on the very iteration that reaches ``max_plies`` (the
    ``on_sims`` hook fires once per ply, right before that ply's search): the
    abort guard must NOT reclassify a finished game.
    """
    agg = StubAggregator(running=True)
    sims_calls = {"n": 0}

    def _flip(_num_sims: int) -> None:
        sims_calls["n"] += 1
        if sims_calls["n"] >= 2:  # the iteration whose move reaches max_plies == 2
            agg.running = False

    game = _make_game(agg, max_plies=2, on_sims=_flip)

    positions, summary = game.play_game()

    assert summary.aborted is False
    assert summary.completed is True
    assert summary.termination_reason == "max_plies_cap"
    assert summary.result == "1/2-1/2"
    assert len(positions) == 2
    assert all(p.outcome == 1 for p in positions)
    assert {p.game_id for p in positions} == {101}


def test_book_plies_produce_no_records_and_are_excluded_from_searched_plies():
    """SR-09 gap: book moves are not model experience.

    A game that plays 3 book plies and 2 searched plies must produce exactly 2
    records, all of them searched positions, with ply accounting consistent.
    """
    agg = StubAggregator(running=True)
    book = StubOpeningBook(stop_after=3)  # book supplies plies 0..2
    game = _make_game(agg, book=book, max_plies=5)

    positions, summary = game.play_game()

    assert summary.aborted is False
    assert game.book_plies == 3
    assert game.ply == 5  # 3 book + 2 searched plies
    assert len(positions) == 2  # exactly the searched plies
    assert len(game.raw_positions) == 2
    assert agg.evaluate_calls >= 2  # searches ran for the searched plies
    assert {p.game_id for p in positions} == {game.game_id}


def test_worker_exception_still_fails_closed():
    """An InferenceError raised mid-search propagates (game discarded)."""
    agg = StubAggregator(running=True)

    def _raise(call: int) -> None:
        raise InferenceError("aggregator stopped before evaluation completed")

    agg.on_eval = _raise
    game = _make_game(agg, max_plies=40)

    with pytest.raises(InferenceError):
        game.play_game()


def test_sink_admission_guard_rejects_aborted_summaries():
    """The static admission guard refuses aborted summaries (defense in depth)."""
    aborted = GameSummary(
        game_id=1,
        length_plies=5,
        result="*",
        resigned=False,
        resigned_side_white=None,
        would_have_resigned=False,
        is_false_resign=False,
        duration_sec=0.1,
        total_simulations=10,
        model_version=1,
        aborted=True,
        termination_reason=TERMINATION_ABORTED,
    )
    completed = GameSummary(
        game_id=2,
        length_plies=5,
        result="1/2-1/2",
        resigned=False,
        resigned_side_white=None,
        would_have_resigned=False,
        is_false_resign=False,
        duration_sec=0.1,
        total_simulations=10,
        model_version=1,
        termination_reason="draw_other",
    )
    assert SelfPlayWorker.accepts_for_sink(aborted) is False
    assert SelfPlayWorker.accepts_for_sink(completed) is True


# ---------------------------------------------------------------------------
# SR-01 — worker-level integration (fake inference client)
# ---------------------------------------------------------------------------


class BlockingFakeClient:
    """Inference client stand-in: returns valid logits, optionally blocking."""

    def __init__(self) -> None:
        self.version = 1
        self.release = threading.Event()
        self.release.set()
        self.calls = 0
        self._lock = threading.Lock()

    def block(self) -> None:
        self.release.clear()

    def unblock(self) -> None:
        self.release.set()

    def evaluate_leaves(self, states: np.ndarray):
        with self._lock:
            self.calls += 1
        self.release.wait(timeout=5.0)
        n = int(states.shape[0])
        # Uniform policy → the C++ mask+renormalize yields uniform over legal
        # moves; neutral WDL → value 0.0.
        return (
            np.zeros((n, 4672), dtype=np.float32),
            np.zeros((n, 3), dtype=np.float32),
            self.version,
        )


def _make_worker(
    client: BlockingFakeClient,
    sink_calls: List[Tuple[List[Any], GameSummary]],
    *,
    games: int = 3,
    max_plies: int = 8,
) -> SelfPlayWorker:
    def sink(positions, summary):
        sink_calls.append((positions, summary))

    return SelfPlayWorker(
        worker_id=0,
        client=client,
        num_concurrent_games=games,
        game_type="chess",
        full_search_sims=2,
        fast_search_sims=1,
        full_search_prob=0.5,
        temp_cutoff_range=(1, 2),
        max_plies=max_plies,
        allow_resignation=False,
        min_resign_ply=10_000,
        search_mode="serial",
        seed=1234,
        use_book=False,
        position_sink=sink,
    )


def _wait_until(predicate, timeout: float = 15.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return False


def test_worker_shutdown_sinks_no_partial_games():
    """Stop during active search: every sunk game is genuine; partials discarded."""
    client = BlockingFakeClient()
    client.block()  # games will park inside their first evaluation
    sunk: List[Tuple[List[Any], GameSummary]] = []
    worker = _make_worker(client, sunk, games=3, max_plies=8)
    try:
        worker.start()
        assert _wait_until(lambda: client.calls >= 1), "no evaluation ever started"
        worker.stop()
        client.unblock()  # let the parked calls finish; threads then observe stop
        assert _wait_until(
            lambda: (
                all(not t.is_alive() for t in worker.threads)
                or worker.aborted_games + worker.invalid_games + worker.total_games_played
                >= worker.game_counter - (1_000_000 + 1) + 1
            ),
            timeout=15.0,
        )
        # Give the final threads a moment to settle after stop()'s 1s join.
        time.sleep(0.3)
    finally:
        worker.stop()
        client.unblock()

    assert worker.total_games_played == len(sunk)
    for _positions, summary in sunk:
        assert summary.aborted is False
        assert summary.termination_reason != TERMINATION_ABORTED
        assert summary.termination_reason in GENUINE_TERMINATIONS
    # The stop interrupted in-flight games: they were discarded, not sunk.
    assert worker.aborted_games + worker.invalid_games >= 1
    # Sunk positions reconcile exactly with the reported counter.
    assert worker.total_positions_generated == sum(len(p) for p, _ in sunk)
    # No draw-labeled partial game can be attributed to an unfinished game:
    # every sunk game's recorded plies are consistent with a terminal result.
    for positions, summary in sunk:
        assert len(positions) == summary.length_plies or summary.length_plies == 0


def test_worker_repeated_and_concurrent_shutdown_is_safe():
    """Repeated and concurrent stop() calls are idempotent and safe."""
    client = BlockingFakeClient()
    sunk: List[Tuple[List[Any], GameSummary]] = []
    workers = [_make_worker(client, sunk, games=2, max_plies=6) for _ in range(3)]
    try:
        for w in workers:
            w.start()
        assert _wait_until(lambda: client.calls >= 1)

        errors: List[BaseException] = []

        def _stop(w: SelfPlayWorker) -> None:
            try:
                for _ in range(3):  # repeated requests from one caller
                    w.stop()
            except BaseException as exc:  # pragma: no cover - failure path
                errors.append(exc)

        threads = [threading.Thread(target=_stop, args=(w,)) for w in workers]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10.0)
        assert errors == []
        for w in workers:
            assert all(not t.is_alive() for t in w.threads) or w.threads == []
    finally:
        client.unblock()
        for w in workers:
            w.stop()

    for _positions, summary in sunk:
        assert summary.aborted is False


# ---------------------------------------------------------------------------
# SR-12 — per-game RNG isolation
# ---------------------------------------------------------------------------


def test_unseeded_games_do_not_touch_global_rng_and_are_independent():
    before = random.getstate()
    np_state_before = np.random.get_state()
    agg = StubAggregator()
    games = [_make_game(agg, seed=None, max_plies=4) for _ in range(5)]
    after = random.getstate()
    np_state_after = np.random.get_state()

    # The global streams are never consumed: unseeded games use their own.
    assert before == after
    assert np_state_before[1].tolist() == np_state_after[1].tolist()
    for g in games:
        assert isinstance(g._rng, random.Random)
        assert g._rng is not random  # not the module alias
        assert isinstance(g._np_rng, np.random.Generator)


def test_seeded_game_rng_contract_is_unchanged():
    """Seeded runs stay exactly reproducible (master_seed + game_id contract)."""
    a1 = _make_game(StubAggregator(), seed=4242, max_plies=4)
    a2 = _make_game(StubAggregator(), seed=4242, max_plies=4)
    b = _make_game(StubAggregator(), seed=4243, max_plies=4)
    assert a1.temp_cutoff == a2.temp_cutoff
    assert a1._rng.random() == a2._rng.random()
    assert a1.mcts_config.seed == 4242
    # A different seed is a different stream (no accidental global reuse).
    assert a1.mcts_config.seed != b.mcts_config.seed


# ---------------------------------------------------------------------------
# SR-14 — search-mode support vs. enforcement
# ---------------------------------------------------------------------------


def test_search_mode_support_and_enforcement():
    from python.eval.match_runner import (
        assert_search_mode_supported,
        enforce_serial_search,
        require_serial_search,
    )

    assert_search_mode_supported("serial", "test")
    assert_search_mode_supported("batched", "test")  # extension exports search_batched
    require_serial_search("serial", "test")  # legacy alias: support check only
    enforce_serial_search("serial", "test")
    with pytest.raises(RuntimeError):
        enforce_serial_search("batched", "test")


# ---------------------------------------------------------------------------
# SR-16 — mover-relative outcome propagation
# ---------------------------------------------------------------------------


def _raw(is_white: bool) -> Dict[str, Any]:
    return {
        "planes": np.zeros((19, 8, 8), dtype=np.float32),
        "visits": np.full(4672, 1.0 / 4672.0, dtype=np.float32),
        "is_white": is_white,
        "is_full_search": True,
        "model_version": 3,
    }


def test_outcomes_from_result_mover_relative_mapping():
    raws = [_raw(True), _raw(False), _raw(True)]
    saved = outcomes_from_result(raws, white_won=True, game_id=9)
    assert [p.outcome for p in saved] == [0, 2, 0]  # win for White-to-move, loss for Black

    saved = outcomes_from_result(raws, white_won=False, game_id=9)
    assert [p.outcome for p in saved] == [2, 0, 2]

    saved = outcomes_from_result(raws, white_won=None, game_id=9)
    assert [p.outcome for p in saved] == [1, 1, 1]  # draw for everyone
    assert all(p.game_id == 9 and p.model_version == 3 for p in saved)


# ---------------------------------------------------------------------------
# SR-17 — legacy IPC paths are retained carriers, isolated from production dispatch
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parent.parent


def _ast_import_targets(path: Path) -> set:
    """Return the set of module targets actually imported by a file (AST-level)."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="ignore"))
    except (OSError, SyntaxError):
        return set()
    targets: set = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.module:
                targets.add(node.module)
            # Relative imports: resolve against the file's package so
            # ``from .ipc import X`` inside python/selfplay is caught.
            if node.level:
                pkg = path.parent.relative_to(_REPO_ROOT).as_posix().replace("/", ".")
                if node.module:
                    targets.add(f"{pkg}.{node.module}")
                else:
                    targets.add(pkg)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                targets.add(alias.name)
    return targets


def _is_ipc_import(target: str) -> bool:
    if target == "selfplay.ipc" or target.endswith(".selfplay.ipc"):
        return True
    if target == "ipc" or target.endswith(".ipc"):
        # Relative-import resolution can only produce a bare/…ipc target that
        # means selfplay.ipc when it resolves into python/selfplay.
        return target == "ipc"
    return False


# Production dispatch surfaces the legacy SHM self-play path must never touch.
_DISPATCH_FILES = (
    "scripts/run_phase4_rl.py",  # orchestrator run loop
    "cli.py",  # repo-root CLI shim
    "python/uci/uci_engine.py",
    "python/selfplay/replay_buffer.py",
    "python/selfplay/metrics.py",
)
_DISPATCH_DIRS = (
    "python/cli",  # CLI command dispatch
    "zeropawn",  # python -m zeropawn shim package
)
# Symbols that exist only to serve the retained legacy multiprocessing path.
_LEGACY_SYMBOLS = (
    "run_inference_server_process",
    "run_worker_process",
    "start_serving",
    "ShmInferenceClient",
    "SharedMemoryPool",
)
# The only production modules allowed to import python/selfplay/ipc.py: the
# three files that carry the retained legacy code paths themselves.
_EXPECTED_IPC_IMPORTERS = {
    "python/selfplay/worker.py",
    "python/selfplay/client.py",
    "python/selfplay/inference_server.py",
}


def test_legacy_ipc_paths_are_retained_carriers_outside_production_dispatch():
    """Pin the current SR-17 contract (audit S6-04 rewrite).

    The original form of this test asserted that ``python/selfplay/ipc.py``
    had NO production importers; that premise no longer holds — the retained
    legacy carriers (worker.py's ``run_worker_process``, client.py's
    ``ShmInferenceClient``, inference_server.py's ``run_inference_server_process``)
    legitimately import the pool. The CURRENT true contract is:

    1. The IPC module's production import surface is EXACTLY those three
       retained-carrier modules — no fourth module may adopt the legacy pool.
    2. The legacy SHM serving loop and its worker-side counterpart are
       referenced by no production dispatch surface (CLI, orchestrator run
       loop, worker runtime) and are called from nowhere outside their
       defining module.
    3. Both retained entry points stay explicitly documented as LEGACY.
    4. ``eval/batched.py`` (the other SR-17 dead path) gains no importers.
    """
    # -- 1) exact production import surface of python/selfplay/ipc.py --------
    importers: Dict[str, str] = {}
    for base in ("python", "scripts"):
        for py in sorted((_REPO_ROOT / base).rglob("*.py")):
            rel = py.relative_to(_REPO_ROOT).as_posix()
            if rel in _EXPECTED_IPC_IMPORTERS:
                continue
            if any(_is_ipc_import(t) for t in _ast_import_targets(py)):
                importers[rel] = "real import"
    for root_file in (
        "cli.py",
        *(p.as_posix() for p in sorted((_REPO_ROOT / "zeropawn").glob("*.py"))),
    ):
        py = _REPO_ROOT / root_file
        if py.exists() and any(_is_ipc_import(t) for t in _ast_import_targets(py)):
            importers[root_file] = "real import"
    assert not importers, (
        "python/selfplay/ipc.py gained production importers beyond the three "
        f"retained carriers {sorted(_EXPECTED_IPC_IMPORTERS)}: {sorted(importers)}"
    )
    # The three carriers must still exist and still import the pool (the
    # retained path must not be silently gutted behind this test's back).
    for rel in _EXPECTED_IPC_IMPORTERS:
        tree_targets = _ast_import_targets(_REPO_ROOT / rel)
        assert any(_is_ipc_import(t) for t in tree_targets), (
            f"{rel} stopped importing python.selfplay.ipc; the retained legacy "
            "carrier contract (SR-17) changed — update this test deliberately"
        )

    # -- 2) legacy serving loop stays outside production dispatch ------------
    for base_dir in _DISPATCH_DIRS:
        for py in sorted((_REPO_ROOT / base_dir).rglob("*.py")):
            text = py.read_text(encoding="utf-8", errors="ignore")
            for needle in _LEGACY_SYMBOLS:
                assert needle not in text, (
                    f"{py.relative_to(_REPO_ROOT).as_posix()} references legacy "
                    f"SHM-path symbol {needle}"
                )
    for rel in _DISPATCH_FILES:
        py = _REPO_ROOT / rel
        assert py.exists(), f"dispatch surface {rel} disappeared"
        text = py.read_text(encoding="utf-8", errors="ignore")
        for needle in _LEGACY_SYMBOLS:
            assert needle not in text, f"{rel} references legacy SHM-path symbol {needle}"

    # No caller of the serving loop anywhere in production trees; within its
    # defining module it must never be *called* (only defined + documented).
    offenders = []
    for base in ("python", "scripts"):
        for py in sorted((_REPO_ROOT / base).rglob("*.py")):
            rel = py.relative_to(_REPO_ROOT).as_posix()
            if rel == "python/selfplay/inference_server.py":
                continue
            text = py.read_text(encoding="utf-8", errors="ignore")
            for needle in ("run_inference_server_process", "start_serving"):
                if needle in text:
                    offenders.append(f"{rel} references {needle}")
    for root_file in (
        "cli.py",
        *(p.as_posix() for p in sorted((_REPO_ROOT / "zeropawn").glob("*.py"))),
    ):
        py = _REPO_ROOT / root_file
        if not py.exists():
            continue
        text = py.read_text(encoding="utf-8", errors="ignore")
        for needle in ("run_inference_server_process", "start_serving"):
            if needle in text:
                offenders.append(f"{root_file} references {needle}")
    assert offenders == [], (
        f"the legacy SHM serving loop gained references outside its retained carrier: {offenders}"
    )
    server_tree = ast.parse(
        (_REPO_ROOT / "python/selfplay/inference_server.py").read_text(encoding="utf-8")
    )
    self_calls = [
        node.lineno
        for node in ast.walk(server_tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "run_inference_server_process"
    ]
    assert self_calls == [], (
        f"inference_server.py calls run_inference_server_process at lines {self_calls}; "
        "the legacy serving loop must stay unreachable, not self-dispatched"
    )

    # Worker runtime: worker.py may reference the legacy client/pool symbols
    # only in its import block and inside the retained run_worker_process
    # region — never in the production SelfPlayWorker runtime above it.
    worker_text = (_REPO_ROOT / "python/selfplay/worker.py").read_text(encoding="utf-8")
    head, sep, _tail = worker_text.partition("def run_worker_process")
    assert sep, "worker.run_worker_process (retained legacy entry) went missing"
    for lineno, line in enumerate(head.splitlines(), start=1):
        if any(sym in line for sym in _LEGACY_SYMBOLS):
            stripped = line.strip()
            assert stripped.startswith(("from python.selfplay", "import ")), (
                f"worker.py production runtime references legacy IPC symbol "
                f"outside an import (line {lineno}): {stripped}"
            )

    # -- 3) retained entry points stay documented as LEGACY ------------------
    server_text = (_REPO_ROOT / "python/selfplay/inference_server.py").read_text(encoding="utf-8")
    marker = server_text.split("def run_inference_server_process")[1][:1500]
    assert "LEGACY" in marker, "run_inference_server_process must stay explicitly marked as legacy"
    worker_marker = worker_text.split("def run_worker_process")[1][:1500]
    assert "LEGACY" in worker_marker, "run_worker_process must stay explicitly marked as legacy"

    # -- 4) eval/batched.py keeps its zero-importer (dead) status ------------
    batched_importers = []
    for base in ("python", "scripts"):
        for py in sorted((_REPO_ROOT / base).rglob("*.py")):
            if py.name == "batched.py" and py.parent.name == "eval":
                continue
            targets = _ast_import_targets(py)
            if any(t == "eval.batched" or t.endswith(".eval.batched") for t in targets):
                batched_importers.append(py.relative_to(_REPO_ROOT).as_posix())
    assert batched_importers == [], f"eval/batched.py gained importers: {batched_importers}"
