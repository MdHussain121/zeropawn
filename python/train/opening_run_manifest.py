"""Run-manifest provenance for ``book train`` (opening-policy fine-tuning).

Audit SA3-03 (2026-11-05, Low): ``book train --resume`` hardcoded
``models/checkpoints_scaled/best_model.pt`` — a registry-``missing``,
superseded baseline — and resumed whatever bytes sat at that path without
checking provenance or compatibility. A later run could silently continue
from an artifact produced by an unrelated campaign (different dataset,
different configuration) or from a checkpoint the project's own registry
quarantined/disqualified.

This module gives opening-policy runs the same fail-closed resume contract
the rest of the project already enforces (checkpoint registry warm-start
validation, RL buffer meta.json reconciliation, supervised run manifests):

1. Every run writes ``run_manifest.json`` into its output checkpoint
   directory *when the run starts* (status ``running``), recording the run's
   inputs and configuration: resolved dataset directories plus the
   tamper-evident validation-receipt digests, the resume checkpoint's
   identity (path + SHA-256 + registry standing), the architecture, and the
   scalar hyperparameters. When the run finishes, the manifest is finalized
   with the SHA-256 of every generated checkpoint (status ``completed``).
2. Resuming from an artifact that a previous opening-policy run produced
   validates the new invocation against that run's manifest: a different
   dataset, changed dataset bytes, a different architecture, or drifted
   hyperparameters refuse with a clear error (each overridable by an
   explicit, logged operator flag). An interrupted run's manifest (status
   ``running``) validates identically — that is the interrupted-run case.
3. A trainer-output artifact with *no* manifest (a legacy partial output or
   a stripped directory) is refused unless explicitly allowed.
4. The resume checkpoint's registry standing is checked before any load:
   quarantined/disqualified/missing/hash-mismatched registry entries refuse
   unless the operator overrides. Unregistered explicit paths remain
   reachable (the operator's choice) but log a loud provenance warning —
   identical to the ``explicit_checkpoint_warnings`` policy used at every
   other explicit load site.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Union

from python.model.checkpoint_registry import (
    compute_sha256,
    resolve_default_checkpoint,
    verify_checkpoint_identity,
)

logger = logging.getLogger(__name__)

MANIFEST_NAME = "run_manifest.json"
MANIFEST_VERSION = "1.0"
MANIFEST_KIND = "opening_policy_run"

#: Artifact names this trainer generates. A resume checkpoint carrying one of
#: these names is a previous opening-policy run's output, not a foreign
#: baseline, so manifest provenance applies to it.
TRAINER_OUTPUT_ARTIFACT_NAMES = frozenset({"best_opening_model.pt", "latest_opening_model.pt"})
PERIODIC_CHECKPOINT_PREFIX = "checkpoint_epoch_"

#: Config fields whose change on resume is a deliberate operator decision and
#: must be explicitly acknowledged (logged) rather than silently applied.
DRIFT_TOLERATED_CONFIG_FIELDS = (
    "epochs",
    "batch_size",
    "lr",
    "max_steps",
    "seed",
    "mixing_ratio_opening",
    "general_data_dir",
    "extra_data_dirs",
)


class OpeningRunProvenanceError(RuntimeError):
    """Raised when resuming would silently continue an incompatible run.

    The message names the manifest, the mismatched field, and the override
    flag that keeps the operation reachable. Callers must surface it
    (CLI: print + exit 1) rather than substituting a different artifact.
    """


@dataclass
class ResumeProvenance:
    """Identity of the checkpoint a run is resuming from."""

    path: Path
    sha256: str
    registry_entry_id: Optional[str] = None
    registry_status: Optional[str] = None
    detail: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "path": str(self.path),
            "sha256": self.sha256,
            "registry_entry_id": self.registry_entry_id,
            "registry_status": self.registry_status,
            "detail": self.detail,
        }


def is_trainer_output_artifact(path: Union[str, Path]) -> bool:
    """True when ``path`` names an artifact this trainer generates."""
    name = Path(path).name
    if name in TRAINER_OUTPUT_ARTIFACT_NAMES:
        return True
    return name.startswith(PERIODIC_CHECKPOINT_PREFIX) and name.endswith(".pt")


def receipt_files_digest(receipt: Optional[Mapping[str, Any]]) -> Optional[str]:
    """Tamper-evident digest of a validation receipt's bound file identities.

    Mirrors ``train_supervised._build_run_metadata``: the digest covers the
    receipt's ``files`` section only, so re-validating identical bytes
    (which refreshes ``validated_at``) does not break an existing binding.
    """
    if not isinstance(receipt, Mapping) or "files" not in receipt:
        return None
    files_str = json.dumps(receipt["files"], sort_keys=True)
    return f"sha256:{hashlib.sha256(files_str.encode()).hexdigest()[:16]}"


def build_run_manifest(
    *,
    config: Dict[str, Any],
    dataset_identity: Dict[str, Any],
    resume_provenance: Optional[ResumeProvenance],
    output_dir: Union[str, Path],
    git_commit: Optional[str] = None,
) -> Dict[str, Any]:
    """Assemble the run manifest written when an opening-policy run starts."""
    manifest: Dict[str, Any] = {
        "manifest_version": MANIFEST_VERSION,
        "kind": MANIFEST_KIND,
        "status": "running",
        "entry": "python.train.train_opening_policy",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": git_commit,
        "config": config,
        "dataset_identity": dataset_identity,
        "resume_checkpoint": resume_provenance.to_dict() if resume_provenance else None,
        "output": {"dir": str(Path(output_dir))},
    }
    return manifest


def write_run_manifest(output_dir: Union[str, Path], manifest: Dict[str, Any]) -> Path:
    """Atomically persist the manifest next to the run's checkpoints.

    Same tmp + fsync + replace primitive as the supervised trainer's
    ``run_manifest.json`` writer. A previous manifest is rotated aside (never
    silently lost), matching ``train_supervised._RunTelemetry``.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / MANIFEST_NAME
    if manifest_path.exists():
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        try:
            manifest_path.replace(out_dir / f"run_manifest_{stamp}.json")
        except OSError:
            logger.debug("opening run manifest rotation failed", exc_info=True)
    return _atomic_write_manifest(manifest_path, manifest)


def read_run_manifest(output_dir: Union[str, Path]) -> Optional[Dict[str, Any]]:
    """Load ``run_manifest.json`` from a run directory; ``None`` when absent."""
    path = Path(output_dir) / MANIFEST_NAME
    if not path.is_file():
        return None
    try:
        with open(path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
    except (OSError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def finalize_run_manifest(
    output_dir: Union[str, Path],
    *,
    extra: Optional[Dict[str, Any]] = None,
) -> Path:
    """Record the run's generated output artifacts and mark it completed."""
    out_dir = Path(output_dir)
    manifest = read_run_manifest(out_dir)
    if manifest is None:
        # A caller-finalized run without a start manifest (only possible for
        # direct API users that opted out) gets a minimal completed record
        # rather than a silently missing one.
        manifest = build_run_manifest(
            config={}, dataset_identity={}, resume_provenance=None, output_dir=out_dir
        )
    outputs: Dict[str, Any] = {}
    if out_dir.is_dir():
        for p in sorted(out_dir.iterdir()):
            if not p.is_file():
                continue
            if p.name not in TRAINER_OUTPUT_ARTIFACT_NAMES and not (
                p.name.startswith(PERIODIC_CHECKPOINT_PREFIX) and p.name.endswith(".pt")
            ):
                continue
            try:
                outputs[p.name] = {
                    "sha256": compute_sha256(p),
                    "size_bytes": p.stat().st_size,
                }
            except OSError:
                logger.debug("could not hash output artifact %s", p, exc_info=True)
    manifest["status"] = "completed"
    manifest["finished_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    manifest["outputs"] = outputs
    if extra:
        manifest.update(extra)
    out_dir.mkdir(parents=True, exist_ok=True)
    return _atomic_write_manifest(out_dir / MANIFEST_NAME, manifest)


def _atomic_write_manifest(manifest_path: Path, manifest: Dict[str, Any]) -> Path:
    tmp = manifest_path.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True)
        fh.write("\n")
        fh.flush()
        try:
            os.fsync(fh.fileno())
        except OSError:
            logger.debug("fsync failed for %s", tmp)
    os.replace(tmp, manifest_path)
    return manifest_path


# ---------------------------------------------------------------------------
# Resume-side validation
# ---------------------------------------------------------------------------


def resolve_resume_checkpoint(
    resume_checkpoint: Union[str, Path, None],
    *,
    repo_root: Union[str, Path],
) -> ResumeProvenance:
    """Resolve the resume checkpoint to an exact, hashed artifact identity.

    ``None`` means "use the default": the registry-designated, verified
    baseline resolved fail-closed via :func:`resolve_default_checkpoint` —
    never the superseded hardcoded path SA3-03 flagged. A relative path is
    interpreted against the repository root (artifact paths are
    repository-relative by contract), falling back to the given path when no
    such file exists there.

    Raises:
        DefaultCheckpointError: when no designated, verified default
            checkpoint is available and none was given explicitly.
        FileNotFoundError: when an explicitly given path does not exist.
    """
    if resume_checkpoint is None:
        resolution = resolve_default_checkpoint(repo_root=repo_root)
        return ResumeProvenance(
            path=resolution.path,
            sha256=resolution.sha256,
            registry_entry_id=resolution.entry_id,
            registry_status="verified",
            detail=f"registry-designated default (entry {resolution.entry_id})",
        )

    p = Path(resume_checkpoint)
    if not p.is_absolute():
        candidate = Path(repo_root) / p
        if candidate.is_file():
            p = candidate
    if not p.is_file():
        raise FileNotFoundError(
            f"Resume checkpoint not found: {p}. Refusing to fall back to any other "
            "artifact (no silent substitution)."
        )
    try:
        resolved = p.resolve()
    except OSError:
        resolved = p
    verdict = verify_checkpoint_identity(resolved, hash_file=True)
    sha = verdict.get("sha256") or compute_sha256(resolved)
    return ResumeProvenance(
        path=resolved,
        sha256=str(sha),
        registry_entry_id=verdict.get("entry_id"),
        registry_status=verdict.get("status"),
        detail=verdict.get("detail"),
    )


def validate_resume_provenance(
    provenance: ResumeProvenance,
    *,
    allow_superseded: bool = False,
) -> Optional[str]:
    """Check the registry standing of the resume checkpoint (fail-closed).

    Registered artifacts with standing ``quarantined``, ``disqualified``,
    ``missing``, ``hash_mismatch``, or ``copied_from_other_path`` refuse —
    resuming training from a superseded artifact is exactly the silent
    provenance mix SA3-03 describes — unless ``allow_superseded`` is set by
    the operator (logged loudly). Unregistered explicit paths return a
    warning string instead of raising: an explicit path is the operator's
    choice, consistent with ``explicit_checkpoint_warnings`` at every other
    explicit load site.

    Returns:
        A provenance warning string when the resume is allowed but unverified;
        ``None`` when the artifact is a verified registry entry.
    """
    status = provenance.registry_status
    if status == "verified":
        return None
    if status == "unregistered" or status is None:
        message = (
            f"Resume checkpoint {provenance.path} (sha256 {provenance.sha256[:16]}...) is not "
            "covered by the checkpoint registry; provenance unverified (explicit-path use)."
        )
        logger.warning("%s", message)
        return message
    if status == "unreadable":
        raise OpeningRunProvenanceError(
            f"Resume checkpoint {provenance.path} cannot be verified: {provenance.detail}. "
            "Refusing to resume from a missing/unreadable artifact."
        )
    if allow_superseded:
        logger.warning(
            "[RESUME OVERRIDE] Superseded resume checkpoint %s accepted by operator "
            "(registry status %r: %s); record this decision in the run log.",
            provenance.path,
            status,
            provenance.detail,
        )
        return (
            f"Resume checkpoint {provenance.path} has registry status {status!r} "
            f"({provenance.detail}); accepted via explicit superseded-resume override."
        )
    raise OpeningRunProvenanceError(
        f"Refusing to resume from {provenance.path}: registry status {status!r} "
        f"({provenance.detail}). This artifact is superseded/untrustworthy for training. "
        "Pass --allow-superseded-resume to override with a logged warning, or pick a "
        "verified checkpoint (registry: docs/checkpoint_registry.json)."
    )


def validate_manifest_for_resume(
    previous_manifest: Mapping[str, Any],
    *,
    manifest_dir: Union[str, Path],
    dataset_identity: Mapping[str, Any],
    config: Mapping[str, Any],
    resume_provenance: ResumeProvenance,
    allow_config_drift: bool = False,
    allow_superseded: bool = False,
) -> Optional[str]:
    """Validate a previous run's manifest against the current invocation.

    Binding fields — dataset identity (resolved directories + validation
    receipt digests) and the model architecture — must match exactly; scalar
    hyperparameters must match unless ``allow_config_drift`` is set (logged).
    When the previous manifest recorded output hashes (a completed run), the
    artifact being resumed is re-hashed against that record so a replaced or
    corrupted output cannot silently stand in for the original.
    """
    kind = previous_manifest.get("kind")
    version = previous_manifest.get("manifest_version")
    if kind != MANIFEST_KIND or version != MANIFEST_VERSION:
        raise OpeningRunProvenanceError(
            f"Refusing to resume from {manifest_dir}: its {MANIFEST_NAME} declares "
            f"kind={kind!r} version={version!r}, not {MANIFEST_KIND!r}/{MANIFEST_VERSION}. "
            "The previous run is not a compatible opening-policy run."
        )

    previous_identity = previous_manifest.get("dataset_identity") or {}
    if dict(previous_identity) != dict(dataset_identity):
        diff = _describe_mapping_diff(previous_identity, dataset_identity)
        raise OpeningRunProvenanceError(
            f"Refusing to resume from {manifest_dir}: the previous run's dataset identity "
            f"differs from this invocation ({diff}). Continuing would mix datasets across "
            "runs. Rebuild with the original dataset, or start a fresh output directory."
        )

    previous_config = previous_manifest.get("config") or {}
    for field in ("num_blocks", "num_filters"):
        if previous_config.get(field) != config.get(field):
            raise OpeningRunProvenanceError(
                f"Refusing to resume from {manifest_dir}: architecture changed "
                f"({field}: previous run {previous_config.get(field)!r}, this invocation "
                f"{config.get(field)!r}). The checkpoint is not compatible."
            )
    drifted = [
        field
        for field in DRIFT_TOLERATED_CONFIG_FIELDS
        if previous_config.get(field) != config.get(field)
    ]
    warning: Optional[str] = None
    if drifted:
        if not allow_config_drift:
            details = ", ".join(
                f"{f}: previous run {previous_config.get(f)!r} -> this invocation {config.get(f)!r}"
                for f in drifted
            )
            raise OpeningRunProvenanceError(
                f"Refusing to resume from {manifest_dir}: training configuration changed "
                f"({details}). Pass --allow-resume-config-drift to accept with a logged "
                "warning, or keep the original configuration."
            )
        warning = (
            f"[RESUME CONFIG DRIFT] {manifest_dir}: fields {drifted} differ from the "
            "previous run's manifest; accepted by explicit operator override."
        )
        logger.warning("%s", warning)

    outputs = previous_manifest.get("outputs") or {}
    if isinstance(outputs, dict) and outputs:
        name = resume_provenance.path.name
        recorded = outputs.get(name)
        if isinstance(recorded, dict) and recorded.get("sha256"):
            if recorded["sha256"] != resume_provenance.sha256:
                if allow_superseded:
                    warning = (
                        f"[RESUME OVERRIDE] {resume_provenance.path} does not match the "
                        "SHA-256 recorded in the previous run's manifest (replaced or "
                        "corrupted output); accepted via explicit superseded-resume override."
                    )
                    logger.warning("%s", warning)
                else:
                    raise OpeningRunProvenanceError(
                        f"Refusing to resume from {resume_provenance.path}: its contents do "
                        f"not match the SHA-256 recorded in {manifest_dir}/{MANIFEST_NAME} "
                        "(the artifact was replaced or corrupted after that run finished). "
                        "Pass --allow-superseded-resume to override with a logged warning."
                    )
    return warning


def require_manifest_for_resume_artifact(
    resume_path: Union[str, Path],
    *,
    allow_missing: bool = False,
) -> Optional[str]:
    """Refuse trainer-output artifacts that carry no run manifest.

    A checkpoint named like this trainer's output (``best_opening_model.pt``,
    ``latest_opening_model.pt``, ``checkpoint_epoch_*.pt``) whose directory
    has no ``run_manifest.json`` is an orphaned/legacy partial output: its
    provenance cannot be established. Refuse unless the operator explicitly
    accepts the legacy behavior.

    Returns a warning string when the missing manifest is explicitly allowed.
    """
    path = Path(resume_path)
    if not is_trainer_output_artifact(path):
        return None
    manifest = read_run_manifest(path.parent)
    if manifest is not None:
        return None
    if allow_missing:
        warning = (
            f"[RESUME OVERRIDE] {path} looks like an opening-policy output artifact but "
            f"its directory carries no {MANIFEST_NAME}; provenance cannot be validated. "
            "Accepted via explicit missing-manifest override."
        )
        logger.warning("%s", warning)
        return warning
    raise OpeningRunProvenanceError(
        f"Refusing to resume from {path}: it is an opening-policy output artifact, but its "
        f"directory has no {MANIFEST_NAME} (interrupted legacy run or stripped directory), so "
        "its provenance cannot be validated. Re-run book train to regenerate it, point "
        "--resume at the original baseline, or pass --allow-missing-run-manifest to override."
    )


def _describe_mapping_diff(previous: Mapping[str, Any], current: Mapping[str, Any]) -> str:
    keys = sorted(set(previous) | set(current))
    changed: List[str] = []
    for key in keys:
        before, after = previous.get(key), current.get(key)
        if before != after:
            changed.append(f"{key}: {before!r} -> {after!r}")
    return "; ".join(changed) if changed else "no field differences detected"
