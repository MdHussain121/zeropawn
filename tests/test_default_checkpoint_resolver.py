"""M-2 regression tests: one authoritative, fail-closed default-checkpoint resolver.

The 2026-10-31 model-architecture/inference audit (finding M-2) found two
competing resolvers with different candidate lists and a silent fallback to
"newest checkpoint on disk" — which, after the certified artifacts went
missing, made every command quietly evaluate a registry-DISQUALIFIED weak
checkpoint.  The remediation makes ``docs/checkpoint_registry.json`` the
single source of truth: the default is the designated, verified artifact,
SHA-256-verified against its registry record; anything else fails closed with
an actionable diagnostic (never a substitute artifact).

All resolver tests are hermetic (fixture repo trees, real tiny checkpoints in
``tmp_path``).  A final integration test resolves the real repository's
designated baseline when the artifact is present on disk.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from python.cli.common import (  # noqa: E402
    get_default_checkpoint_path,
    load_engine_model,
)
from python.cli.models import get_default_model_path  # noqa: E402
from python.model.checkpoint_registry import (  # noqa: E402
    DefaultCheckpointError,
    compute_sha256,
    resolve_default_checkpoint,
)

SETTLED_ARTIFACT = "models/checkpoints_baseline_20261031/best_model.pt"
START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def _write_checkpoint(path: Path, step: int = 7) -> str:
    from python.model.chess_net import ChessNet

    path.parent.mkdir(parents=True, exist_ok=True)
    model = ChessNet(num_blocks=1, num_filters=8)
    torch.save(
        {"model": model.state_dict(), "num_blocks": 1, "num_filters": 8, "step": step},
        path,
    )
    return compute_sha256(path)


def _make_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    (repo / "docs").mkdir(parents=True, exist_ok=True)
    return repo


def _entry(repo: Path, rel: str, status="verified", designated=False, create=True, **extra):
    """Registry entry for a file under ``repo``; creates the file when requested."""
    p = repo / rel
    if create and not p.is_file():
        _write_checkpoint(p, step=extra.get("step") or 7)
    e = {
        "id": extra.pop("id", f"ckpt-test-{rel.replace('/', '-')}"),
        "path": rel,
        "sha256": extra.pop("sha256", compute_sha256(p) if p.is_file() else "0" * 64),
        "size_bytes": extra.pop("size_bytes", p.stat().st_size if p.is_file() else 123),
        "status": status,
        "designated_warm_start": designated,
        "role": extra.pop("role", "test"),
        "step": extra.pop("step", 7),
    }
    e.update(extra)
    return e


def _write_registry(repo: Path, entries) -> None:
    (repo / "docs" / "checkpoint_registry.json").write_text(
        json.dumps({"version": 1, "checkpoints": list(entries)}, indent=1),
        encoding="utf-8",
    )


def _weak_disqualified_entry(repo: Path):
    rel = "models/checkpoints_scaled/latest_checkpoint.pt"
    _write_checkpoint(repo / rel)
    return {
        "id": "ckpt-01-weak",
        "path": rel,
        "sha256": compute_sha256(repo / rel),
        "size_bytes": (repo / rel).stat().st_size,
        "status": "disqualified",
        "designated_warm_start": False,
        "role": "supervised_scaled_continuation",
        "disqualification": {
            "reason": "superseded_value_head",
            "detail": "value head trained on corrupted targets",
        },
    }


# ---------------------------------------------------------------------------
# Core resolution policy
# ---------------------------------------------------------------------------


def test_single_designated_verified_entry_resolves_deterministically(tmp_path):
    repo = _make_repo(tmp_path)
    entry = _entry(
        repo, SETTLED_ARTIFACT, designated=True, id="ckpt-10-test_baseline", step=4802
    )
    _write_registry(repo, [entry])

    r1 = resolve_default_checkpoint(repo_root=repo)
    r2 = resolve_default_checkpoint(repo_root=repo)
    assert r1.path == repo / SETTLED_ARTIFACT
    assert r1.sha256 == entry["sha256"]
    assert r1.entry_id == "ckpt-10-test_baseline"
    assert r1.step == 4802
    assert (r1.path, r1.sha256, r1.entry_id) == (r2.path, r2.sha256, r2.entry_id)


def test_all_preferred_defaults_missing_fails_closed_not_silent(tmp_path):
    """The historical real-world state: every originally preferred artifact is
    gone, a disqualified weak checkpoint exists on disk.  Resolution must
    raise — never quietly select the weak file."""
    repo = _make_repo(tmp_path)
    entries = [
        _entry(repo, "models/checkpoints_finetune/best_model.pt", status="missing",
               create=False, id="ckpt-09"),
        _entry(repo, "models/checkpoints_scaled/best_model.pt", status="missing",
               create=False, id="ckpt-08"),
        _entry(repo, "models/checkpoints_rl/phase4_final_checkpoint.pt", status="missing",
               create=False, id="ckpt-04"),
        _weak_disqualified_entry(repo),
    ]
    _write_registry(repo, entries)

    with pytest.raises(DefaultCheckpointError) as exc:
        resolve_default_checkpoint(repo_root=repo)
    msg = str(exc.value)
    assert "No designated, verified default checkpoint" in msg
    assert "superseded_value_head" in msg  # ineligibility reason surfaced
    assert "Recovery:" in msg
    assert "Refusing to fall back" in msg


def test_designated_file_missing_on_disk_fails_closed(tmp_path):
    repo = _make_repo(tmp_path)
    entries = [_entry(repo, SETTLED_ARTIFACT, designated=True, create=False, id="ckpt-10-gone")]
    _write_registry(repo, entries)

    with pytest.raises(DefaultCheckpointError) as exc:
        resolve_default_checkpoint(repo_root=repo)
    msg = str(exc.value)
    assert "MISSING on disk" in msg
    assert str(repo / SETTLED_ARTIFACT) in msg
    assert "ckpt-10-gone" in msg
    assert "Recovery:" in msg


def test_designated_hash_mismatch_fails_closed(tmp_path):
    repo = _make_repo(tmp_path)
    entries = [_entry(repo, SETTLED_ARTIFACT, designated=True, id="ckpt-10-tampered",
                      sha256="f" * 64)]
    _write_registry(repo, entries)

    with pytest.raises(DefaultCheckpointError) as exc:
        resolve_default_checkpoint(repo_root=repo)
    msg = str(exc.value)
    assert "SHA-256" in msg
    assert "f" * 16 in msg  # registry hash shown
    assert "Recovery:" in msg


def test_designated_size_mismatch_fails_closed(tmp_path):
    repo = _make_repo(tmp_path)
    entries = [_entry(repo, SETTLED_ARTIFACT, designated=True, id="ckpt-10-wrong-size",
                      size_bytes=1)]
    _write_registry(repo, entries)

    with pytest.raises(DefaultCheckpointError) as exc:
        resolve_default_checkpoint(repo_root=repo)
    assert "size" in str(exc.value).lower()


def test_quarantined_and_undesignated_never_become_default(tmp_path):
    repo = _make_repo(tmp_path)
    entries = [
        _entry(repo, "models/checkpoints_rl/quarantined/x.pt", status="quarantined",
               create=False, id="ckpt-05", sha256="b" * 64),
        _entry(repo, "models/checkpoints_rl/ok.pt", designated=False, id="ckpt-03"),
    ]
    _write_registry(repo, entries)

    with pytest.raises(DefaultCheckpointError):
        resolve_default_checkpoint(repo_root=repo, verify_hash=False)


def test_multiple_designated_picks_highest_id_deterministically(tmp_path, caplog):
    repo = _make_repo(tmp_path)
    entries = [
        _entry(repo, "models/a/best_model.pt", designated=True, id="ckpt-02"),
        _entry(repo, "models/b/best_model.pt", designated=True, id="ckpt-11"),
    ]
    _write_registry(repo, entries)

    import logging

    with caplog.at_level(logging.WARNING):
        r = resolve_default_checkpoint(repo_root=repo)
    assert r.entry_id == "ckpt-11"  # highest id wins (deterministic, documented)
    assert any("Multiple designated checkpoints" in rec.message for rec in caplog.records)
    assert resolve_default_checkpoint(repo_root=repo).entry_id == "ckpt-11"


def test_missing_registry_fails_closed(tmp_path):
    repo = tmp_path / "empty_repo"
    repo.mkdir()
    with pytest.raises(DefaultCheckpointError) as exc:
        resolve_default_checkpoint(repo_root=repo)
    assert "registry" in str(exc.value).lower()
    assert "docs" in str(exc.value)


def test_designated_entry_without_sha256_is_ineligible(tmp_path):
    repo = _make_repo(tmp_path)
    entries = [_entry(repo, SETTLED_ARTIFACT, designated=True, create=False,
                      sha256=None, id="ckpt-10-nosha")]
    _write_registry(repo, entries)
    with pytest.raises(DefaultCheckpointError) as exc:
        resolve_default_checkpoint(repo_root=repo)
    assert "no SHA-256" in str(exc.value)


# ---------------------------------------------------------------------------
# One authoritative resolver: the two CLI entry points must agree
# ---------------------------------------------------------------------------


def test_legacy_resolvers_agree_on_the_designated_default(tmp_path, monkeypatch):
    repo = _make_repo(tmp_path)
    entries = [_entry(repo, SETTLED_ARTIFACT, designated=True, id="ckpt-10-agree")]
    _write_registry(repo, entries)

    import python.cli.common as common_mod
    import python.cli.models as models_mod

    monkeypatch.setattr(common_mod, "REPO_ROOT", repo)
    monkeypatch.setattr(models_mod, "REPO_ROOT", repo)

    p_common = get_default_checkpoint_path()
    p_models = get_default_model_path()
    assert p_common == p_models == repo / SETTLED_ARTIFACT


def test_resolvers_fail_closed_together(tmp_path, monkeypatch):
    repo = _make_repo(tmp_path)
    _write_registry(repo, [])
    import python.cli.common as common_mod
    import python.cli.models as models_mod

    monkeypatch.setattr(common_mod, "REPO_ROOT", repo)
    monkeypatch.setattr(models_mod, "REPO_ROOT", repo)

    with pytest.raises(DefaultCheckpointError):
        get_default_checkpoint_path()
    with pytest.raises(DefaultCheckpointError):
        get_default_model_path()


def test_no_cli_module_keeps_a_local_default_candidate_list():
    """Anti-regression guard: no CLI/UCI module may reintroduce a private
    default-selection policy (literal best_model fallbacks / candidate lists).
    Everything must route through the canonical resolvers."""
    cli_dir = ROOT / "python" / "cli"
    offenders = []
    for py in sorted(cli_dir.glob("*.py")):
        if py.name in {"common.py", "models.py"}:
            continue  # the canonical resolver module itself
        src = py.read_text(encoding="utf-8")
        if 'or "models/checkpoints_scaled/best_model.pt"' in src:
            offenders.append(f"{py.name}: literal scaled/best_model fallback")
        if "checkpoints_finetune/best_model.pt" in src and "candidates" in src:
            offenders.append(f"{py.name}: private candidate list")
    for uci_py in sorted((ROOT / "python" / "uci").glob("*.py")):
        src = uci_py.read_text(encoding="utf-8")
        if 'or "models/checkpoints_scaled/best_model.pt"' in src:
            offenders.append(f"uci/{uci_py.name}: literal scaled/best_model fallback")
    assert not offenders, offenders


# ---------------------------------------------------------------------------
# CLI commands fail closed with the actionable diagnostic
# ---------------------------------------------------------------------------


@pytest.fixture()
def fail_closed_repo(tmp_path, monkeypatch):
    """Repo whose only registered checkpoint is disqualified: every default
    load path must refuse and explain, not run the weak artifact."""
    repo = _make_repo(tmp_path)
    _write_registry(repo, [_weak_disqualified_entry(repo)])
    import python.cli.common as common_mod

    monkeypatch.setattr(common_mod, "REPO_ROOT", repo)
    return repo


def test_match_live_fails_closed_without_designated_default(fail_closed_repo, capsys):
    from types import SimpleNamespace

    import python.cli.commands_match as cm

    rc = cm.run_live_game(SimpleNamespace(checkpoint=None, device="cpu", fen=START_FEN))
    assert rc == 1
    out = capsys.readouterr().out
    assert "No designated, verified default checkpoint" in out
    assert "superseded_value_head" in out


def test_match_human_fails_closed_without_designated_default(fail_closed_repo, capsys):
    from types import SimpleNamespace

    import python.cli.commands_match as cm

    rc = cm.run_human_game(SimpleNamespace(checkpoint=None, device="cpu"))
    assert rc == 1
    assert "No designated, verified default checkpoint" in capsys.readouterr().out


def test_puzzles_solve_fails_closed_without_designated_default(fail_closed_repo, capsys):
    from types import SimpleNamespace

    import python.cli.commands_puzzles as cp

    rc = cp.run_solve_puzzles(
        SimpleNamespace(checkpoint=None, device="cpu", puzzles_file="x", start_index=0)
    )
    assert rc == 1
    assert "No designated, verified default checkpoint" in capsys.readouterr().out


def test_models_validate_fails_closed_without_designated_default(fail_closed_repo):
    from types import SimpleNamespace

    import python.cli.commands_models as cmd

    rc = cmd.run_models_validate(SimpleNamespace(checkpoint=None, device="cpu"))
    assert rc == 1


def test_models_check_compat_fails_closed_without_designated_default(fail_closed_repo):
    from types import SimpleNamespace

    import python.cli.commands_models as cmd

    rc = cmd.run_models_check_compat(SimpleNamespace(checkpoint=None))
    assert rc == 1


# ---------------------------------------------------------------------------
# Explicit checkpoint paths: honored, validated, provenance surfaced
# ---------------------------------------------------------------------------


def test_explicit_unregistered_checkpoint_loads_with_warning(tmp_path):
    repo = _make_repo(tmp_path)
    _write_registry(repo, [])
    p = tmp_path / "my_random_run.pt"
    _write_checkpoint(p, step=3)

    import python.cli.common as common_mod

    old_root = common_mod.REPO_ROOT
    common_mod.REPO_ROOT = repo
    try:
        _, info = load_engine_model(p, device="cpu")
    finally:
        common_mod.REPO_ROOT = old_root
    assert info["path"] == Path(p)
    assert info["registry_entry_id"] is None
    assert info["provenance_warning"] and "not in the checkpoint registry" in info[
        "provenance_warning"
    ]


def test_explicit_disqualified_checkpoint_warns_loudly(tmp_path):
    repo = _make_repo(tmp_path)
    weak = _weak_disqualified_entry(repo)
    _write_registry(repo, [weak])

    import python.cli.common as common_mod

    old_root = common_mod.REPO_ROOT
    common_mod.REPO_ROOT = repo
    try:
        _, info = load_engine_model(repo / weak["path"], device="cpu")
    finally:
        common_mod.REPO_ROOT = old_root
    # Explicit disqualified path is honored (operator's call) but never silent.
    assert info["registry_status"] == "disqualified"
    assert "DISQUALIFIED" in info["provenance_warning"]
    assert "superseded_value_head" in info["provenance_warning"]


def test_explicit_registered_verified_checkpoint_has_no_warning(tmp_path):
    repo = _make_repo(tmp_path)
    entries = [_entry(repo, SETTLED_ARTIFACT, designated=True, id="ckpt-10-verified")]
    _write_registry(repo, entries)

    import python.cli.common as common_mod

    old_root = common_mod.REPO_ROOT
    common_mod.REPO_ROOT = repo
    try:
        _, info = load_engine_model(repo / SETTLED_ARTIFACT, device="cpu")
    finally:
        common_mod.REPO_ROOT = old_root
    assert info["registry_entry_id"] == "ckpt-10-verified"
    assert info["registry_status"] == "verified"
    assert info["provenance_warning"] is None


def test_explicit_missing_path_raises_filenotfound(tmp_path):
    missing = tmp_path / "nope.pt"
    with pytest.raises(FileNotFoundError):
        load_engine_model(missing, device="cpu")


def test_explicit_corrupted_checkpoint_raises(tmp_path):
    p = tmp_path / "corrupt.pt"
    p.write_bytes(b"this is not a torch checkpoint")
    with pytest.raises(Exception) as exc:
        load_engine_model(p, device="cpu")
    assert not isinstance(exc.value, FileNotFoundError)


def test_explicit_incompatible_state_dict_raises_strict_load_error(tmp_path):
    p = tmp_path / "wrong_arch.pt"
    torch.save({"model": {"not_a_real_layer.weight": torch.zeros(3)}}, p)
    with pytest.raises(RuntimeError):
        load_engine_model(p, device="cpu")


# ---------------------------------------------------------------------------
# Registry architecture signature (invalid architecture is rejected)
# ---------------------------------------------------------------------------


def test_warm_start_rejects_wrong_architecture_signature(tmp_path):
    from python.model.checkpoint_registry import (
        WarmStartValidationError,
        validate_warm_start,
    )

    p = tmp_path / "small_net.pt"
    _write_checkpoint(p)  # 1x8 net: not the 12x128 project architecture
    with pytest.raises(WarmStartValidationError) as exc:
        validate_warm_start(p, allow_unverified=True)
    assert "architecture" in str(exc.value).lower() or "parameter count" in str(exc.value)


# ---------------------------------------------------------------------------
# Real repository integration (runs where the artifact exists on disk)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not (ROOT / SETTLED_ARTIFACT).is_file(),
    reason="designated baseline not on disk in this tree (gitignored models/)",
)
def test_real_registry_resolves_the_designated_baseline():
    r = resolve_default_checkpoint(repo_root=ROOT)
    assert r.path == ROOT / SETTLED_ARTIFACT
    assert r.entry_id.startswith("ckpt-10")
    # Hash was verified against the registry by the resolver itself.
    assert r.sha256 == compute_sha256(r.path)
