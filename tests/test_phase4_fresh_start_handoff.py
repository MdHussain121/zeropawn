"""Phase 4 handoff regression tests (audit SLQ-07 remediation, 2026-10-31).

Covers the two fresh-start blockers that made every RL campaign start fail:
  1. ``run_phase4_rl.py:690`` read ``self._warm_start_info`` — an attribute
     never assigned — so a fresh start crashed with AttributeError before
     training. The validated record is ``self._warm_start_record``.
  2. The duplicate ``--allow-unverified-warm-start`` argparse definition made
     ``--help`` (and therefore every start) die with ArgumentError.

No RL campaign is started here: the orchestrator is constructed with stubbed
server/buffer/trainer and a tmp checkpoint dir, reaching exactly the trainer
setup boundary where the crash used to happen.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))


def _load_orchestrator_module():
    scripts_dir = _repo_root / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    return importlib.import_module("run_phase4_rl")


def test_slq07_help_does_not_die_on_duplicate_argument(capsys):
    """The launcher's argparse must accept --help (no duplicate optionals)."""
    import scripts.run_phase4_rl as p4

    parser = p4.build_parser() if hasattr(p4, "build_parser") else None
    if parser is None:
        # Fall back to invoking the module's parser construction the way
        # main() does.
        import io
        from contextlib import redirect_stdout

        argv_backup = sys.argv
        sys.argv = ["run_phase4_rl.py", "--help"]
        try:
            with redirect_stdout(io.StringIO()) as buf:
                with pytest.raises(SystemExit) as excinfo:
                    p4.main()
            assert excinfo.value.code == 0
            assert "allow-unverified-warm-start" in buf.getvalue()
        finally:
            sys.argv = argv_backup
    else:
        import io
        from contextlib import redirect_stdout

        with redirect_stdout(io.StringIO()) as buf:
            with pytest.raises(SystemExit) as excinfo:
                parser.parse_args(["--help"])
        assert excinfo.value.code == 0
        assert buf.getvalue().count("--allow-unverified-warm-start") >= 1


def test_slq07_fresh_start_reaches_trainer_setup(tmp_path, monkeypatch):
    """A fresh (non-resuming) start must pass the trainer-setup branch that
    used to crash on the never-assigned ``_warm_start_info`` attribute."""
    p4 = _load_orchestrator_module()

    # Stub the heavy construction dependencies: no GPU server, no buffer IO,
    # no puzzle loading. Only the orchestration logic under test runs.
    class _StubBuffer:
        def __init__(self, *a, **k):
            self.capacity = k.get("capacity", 0)
            self.meta = {}

    from python.model.chess_net import ChessNet

    class _StubServer:
        def __init__(self, *a, **k):
            # A real ChessNet on CPU: the orchestrator copies the server's
            # state_dict into the trainer's model during construction.
            self.model = ChessNet(
                in_channels=19, num_filters=128, num_blocks=12, num_actions=4672
            )
            self.model_version = 0

    class _StubClient:
        def __init__(self, *a, **k):
            pass

    class _StubTrainer:
        def __init__(self, *a, **k):
            self.run_metadata = {}
            self.step_num = 0
            self.inherited_step_num = 0
            self.direct_server = None

        def load_checkpoint(self, *a, **k):  # pragma: no cover - resume path
            raise AssertionError("fresh start must not resume")

    class _StubMetrics:
        def __init__(self, *a, **k):
            pass

    class _StubWorker:
        def __init__(self, *a, **k):
            self.games = []

    monkeypatch.setattr(p4, "ReplayBuffer", _StubBuffer)
    monkeypatch.setattr(p4, "InferenceServer", _StubServer)
    monkeypatch.setattr(p4, "DirectInferenceClient", _StubClient)
    monkeypatch.setattr(p4, "RLTrainer", _StubTrainer)
    monkeypatch.setattr(p4, "SelfPlayWorker", _StubWorker)
    if hasattr(p4, "MetricsTracker"):
        monkeypatch.setattr(p4, "MetricsTracker", _StubMetrics)
    monkeypatch.setattr(p4, "load_puzzles", lambda path: [])

    orch = p4.Phase4Orchestrator(
        base_checkpoint=None,  # random init: pure fresh start
        checkpoint_dir=str(tmp_path / "ckpts"),
        log_dir=tmp_path,  # buffer lands inside the tmp tree
        puzzle_csv_path=str(tmp_path / "puzzles.csv"),
        openings_path=str(tmp_path / "openings.epd"),
        stockfish_path=str(tmp_path / "stockfish.exe"),
        is_verify_mode=True,
        require_certified_eval=False,
        skip_baseline_eval=True,
        human_data_dir=str(tmp_path / "human"),
        device="cpu",
        train_device="cpu",
        infer_device="cpu",
    )
    # The crash line: this attribute must exist and be used.
    assert getattr(orch, "_warm_start_record", "MISSING") is None
    assert orch.trainer.run_metadata.get("warm_start") == "random_init"
    assert orch.trainer.inherited_step_num == 0


def test_slq07_warm_start_refusal_is_fail_closed(tmp_path, monkeypatch):
    """A missing warm-start artifact must stop construction with a clear
    refusal (no silent random-init fallback)."""
    p4 = _load_orchestrator_module()

    class _StubBuffer:
        def __init__(self, *a, **k):
            self.meta = {}

    monkeypatch.setattr(p4, "ReplayBuffer", _StubBuffer)
    monkeypatch.setattr(p4, "load_puzzles", lambda path: [])

    missing = tmp_path / "no_such_checkpoint.pt"
    with pytest.raises(RuntimeError) as excinfo:
        p4.Phase4Orchestrator(
            base_checkpoint=str(missing),
            checkpoint_dir=str(tmp_path / "ckpts"),
            log_dir=tmp_path,
            puzzle_csv_path=str(tmp_path / "puzzles.csv"),
            openings_path=str(tmp_path / "openings.epd"),
            stockfish_path=str(tmp_path / "stockfish.exe"),
            is_verify_mode=True,
            require_certified_eval=False,
            skip_baseline_eval=True,
            human_data_dir=str(tmp_path / "human"),
            device="cpu",
            train_device="cpu",
            infer_device="cpu",
        )
    assert "Refusing to start Phase 4" in str(excinfo.value) or "not found" in str(
        excinfo.value
    ).lower() or "missing" in str(excinfo.value).lower()
