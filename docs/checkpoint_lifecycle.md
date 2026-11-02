# ZeroPawn Checkpoint & Artifact Lifecycle Contract

Established by the **2026-10-31 checkpoint & artifact lifecycle remediation**, which
closed the findings of the 2026-10-30/31 audit (`audit/checkpoint-artifact-lifecycle-20261030`,
commit `6ccdc29`) — CKP-01…CKP-15, SUS-01/02, INF-01…03.

This document is the operator-facing contract. It states where artifacts may live,
what a checkpoint must contain, what "resume" and "warm start" mean, how promotion
works, and which failures are loud by design.

---

## 1. Artifact roots are hermetic (CKP-04)

* **No artifact path is ever resolved against the process working directory.**
  Relative paths resolve against the repository root discovered from the code's own
  location (`python.cli.common.REPO_ROOT`); absolute paths are used as given.
* **Default directories do not exist for trainers.** `RLTrainer` writes into a
  session temporary directory when no `checkpoint_dir` is supplied (logged loudly);
  production callers must name their output tree.
* **Protected production names need an explicit ticket.** Writing any of
  `live_weights.pt`, `latest_checkpoint.pt`, `best_model.pt`, `best_eval.pt`,
  `best_eval.json`, `phase4_final_checkpoint.pt`, `checkpoint_registry.json`,
  `manifest.json`, or `run_manifest.json` **inside `models/`** requires a
  `python.common.artifacts.ProductionWriteTicket`. Tickets are minted only by
  naming an explicit destination (`issue_for_explicit_path`) or by a production
  launcher (`zeropawn train --mode supervised`, `zeropawn finetune`,
  `scripts/run_phase4_rl.py`). Tests, diagnostics, and ad-hoc scripts hold none and
  are refused with a diagnostic. There is deliberately **no environment variable and
  no bare boolean** that authorizes production writes.
* **Architecture cannot silently change under a name.** Replacing an existing
  checkpoint whose recorded `num_blocks`/`num_filters` differ from the payload's is
  refused. This is the 2026-10-30 signature (a 2×32 test model replacing the 12×128
  `live_weights.pt`), and the guard applies even with a ticket. Deliberate
  migrations pass `allow_arch_change=True`.
* **Two writers on one path fail loudly** (SUS-02). Every checkpoint write holds a
  cross-process OS lock on the artifact (`python.common.artifacts.artifact_lock`,
  `<artifact>.lock`); contention raises `ArtifactLockError` naming the holder's pid,
  owner tag, and age. Locks die with the process — there is no stale-lock heuristic,
  and lock files carry no state.

## 2. What a checkpoint contains (CKP-01, CKP-10, CKP-12)

Current formats: **RL = `checkpoint_format: 2`**, **supervised = `checkpoint_format: 1`**.
The schema contract lives in `python/model/checkpoint_schema.py`
(`CURRENT_FORMATS`, `REQUIRED_FIELDS`, `validate_checkpoint_schema`).

Required in every current-format training checkpoint:

| Field | Meaning |
|---|---|
| `model`, `optimizer`, `scheduler` | Full training state (weights-only containers must not carry these) |
| `checkpoint_type`, `checkpoint_format` | Declared family and schema version |
| `num_blocks`, `num_filters` | Architecture header, cross-checked against the payload |
| `run_metadata` | The **run-status ledger** (below) |
| `provenance` | Immutable identity of the producing run |
| `buffer_state` (RL) | `{write_head, size, total_added, capacity, generation}` (see §4) |
| `epoch` (supervised) | Epoch counter |
| `rng_state`, `numpy_rng_state`, `python_rng_state` | CPU RNG continuity |
| `cuda_rng_state` | Per-device CUDA RNG tensors, `null` on CPU runs (never fabricated) |

A note on the stored learning rate: the optimizer param-group LR inside a saved
checkpoint is the LR the **last executed update** used (`LR(step-1)` for a run at
`step`). On resume the schedule is recomputed from the checkpointed step counter,
so the next update uses `LR(step)` — exactly what an uninterrupted run would have
used. The schedule *position* (step counter) is the restored quantity; the stored
group LR is informational.

Legacy artifacts (no version marker, or `checkpoint_format: 1` for RL) are accepted
through an **explicit legacy path** that logs a warning listing what is missing;
they are never silently treated as current. A *newer* version than this build
supports is rejected rather than reinterpreted field-by-field.

### Run-status ledger

`run_metadata` is the authoritative lifecycle record and is written **inside the
artifact** (the CKP-01 defect was that the orchestrator stamped an in-memory dict
that never reached any checkpoint).

States: `created → running → completed | failed | interrupted | cancelled`.
Terminal states are final; a continuation is a **new run** whose
`parent_checkpoint` names the predecessor (with its sha256), and whose
`predecessor_run` preserves the predecessor's full ledger — including a
`failed`/`interrupted` state, so a failure can never be read as success.

Other ledger fields: `run_id`, `run_type`, `created_at`, `started_at`,
`last_update_at`, `completed_at`, `failure_reason`, `step`, `epoch`, `git_commit`,
`seed`, `config`, `dataset`, `last_checkpoint` (path/size/mtime/sha256).

### Provenance

`provenance` records `run_id`, `run_type`, `created_at`, `git_commit`, the model
architecture and parameter count, the training configuration, dataset receipt
identity, optimizer/scheduler configuration, and the parent checkpoint's identity.
A file cannot contain its own hash; the hash of a saved artifact is recorded where
it can be verified: the run's telemetry, the eval history, and `best_eval.json`.

## 3. Resume vs warm start (CKP-02)

| Operation | CLI | Restores |
|---|---|---|
| **Full resume** | `--resume PATH` (supervised: full state; RL: full state) | model, optimizer, scheduler geometry/position, step/epoch, RNG (CPU + CUDA when compatible), run ledger lineage |
| **Weights-only warm start** | `--checkpoint PATH` / `-c` / `--weights` | model weights only: fresh optimizer, schedule from step 0, new run ledger; the source's step is kept as `inherited_step_num`/provenance |
| **Explicit override** | `--resume-weights-only` with `--resume PATH` | weights only, deliberately |

Contradictions are rejected: passing both `--checkpoint` and `--resume` is an error
(two different operations), and `--resume-weights-only` without a path is an error.
The mode taken is printed (`init_mode`) and logged.

A weights-only container (export, opening-policy checkpoint, curriculum container)
is **never** accepted as a full resume, and a weights-only file inside
`models/checkpoints_rl/` is **never** selected as the auto-resume point
(`is_resumable_candidate`): a candidate must carry SGD state or declare
`checkpoint_type: rl`.

## 4. The checkpoint ↔ replay-buffer contract (CKP-05)

A replay buffer is identified by a **generation** (`buffer_generation` in
`meta.json`, minted at creation). Every RL checkpoint stores the buffer's
`(write_head, size, total_added, capacity, generation)` snapshot.

On resume the checkpoint snapshot and the live buffer are reconciled **as a unit**:

| Situation | Result |
|---|---|
| Snapshots identical | restore (no-op), decision `identical_snapshot` |
| Live buffer same/unknown generation and equal-or-newer | **adopt the live triple as a unit** (`adopted_live_newer`); the checkpoint's snapshot is kept in `buffer_resume_decision` as provenance |
| Live buffer older/emptier than the checkpoint expects | **refused** (`BufferRestoreError`) — restoring would discard records |
| Both generations known and different | **refused** — splicing generations mixes unrelated self-play data |
| `--reset-buffer` passed | live state kept as-is, checkpoint snapshot retained as provenance only; **no records are ever invented** |
| Checkpoint has no snapshot (legacy) | live state untouched, decision recorded |

The head is never rewound over newer metadata and `size` is never left behind: the
historical `head=15,947 / size=96,067` contradiction cannot be reconstructed.

## 5. Writing, promotion, and durability (CKP-03, CKP-07, CKP-15, SUS-01)

* **Atomic saves.** Every checkpoint writer funnels through
  `_atomic_torch_save` (temp file → fsync file → `os.replace` → fsync directory)
  while holding the artifact lock. `latest_checkpoint.pt` is a byte copy of the
  just-written step checkpoint, produced under the same lock.
* **Promotion is transactional.** `_promote_best_eval` hashes the candidate, copies
  it to `best_eval.pt.tmp`, fsyncs and **verifies the copy's hash**, atomically
  replaces `best_eval.pt`, fsyncs the directory, and only then writes
  `best_eval.json` — which carries the same `generation` id and
  `checkpoint_sha256`. A tear at any step leaves the previous valid artifact and its
  metadata untouched; a caught failure removes its temp file.
* **Evaluation history is real evidence.** `eval_history.csv` rows are written after
  the promotion decision and include `promotion_decision`, `promotion_reason`, and
  `promotion_generation` alongside the measurements, checkpoint name/sha256,
  configuration hash, and commit. Rows and headers are flushed **and fsynced**.
  Telemetry CSV rows are fsynced too (SUS-01).
* **A failed final save is never silent** (CKP-07): the failure is logged, recorded
  as a `checkpoint_saved_failed` telemetry event, the run is finalized as `failed`,
  and `train_supervised` raises instead of returning success. The previous valid
  checkpoint is left untouched.

## 6. Governance: registry, discovery, and eligibility (CKP-08, CKP-11, CKP-14)

`docs/checkpoint_registry.json` is the **fitness authority**; `models/manifest.json`
remains the lineage authority for release weights.

* Registry writes go through `write_registry`/`upsert_registry_entry`: schema-checked
  JSON, atomic tmp+fsync+replace, under the registry lock.
* **Matching is path-first, hashes second.** A byte-identical copy at another path
  reports `copied_from_other_path` and does **not** inherit the original's
  designation or status.
* Load paths verify identity against the registry: `hash_mismatch`, `quarantined`,
  and `disqualified` are refused before any state is adopted; unregistered files
  load by explicit path with a logged note (or are refused when the caller requires
  registration).
* The registry is discovered from the artifact's own tree
  (`find_registry_root`), so checkpoints in a worktree are governed by that
  worktree's registry.
* **Default model selection is fail-closed** (`resolve_default_checkpoint`,
  the single resolver every consumer routes through): the default is the
  registry entry that is `verified` AND `designated_warm_start`, re-hashed
  against its registry record before use; anything else — no designated
  entry, missing file, hash/size mismatch, missing registry — raises
  `DefaultCheckpointError` naming the expected artifact and recovery actions.
  There is no "newest file on disk" fallback (see `docs/model_io_contract.md` §4).
* Weights-only containers are registered with an explicit status (see
  `ckpt-11-unattributed_warm_start_weights`, `disqualified`) or refused.

## 7. What remains intentionally strict

* Explicitly typed foreign artifacts (`checkpoint_type` not `rl`) are refused by the
  RL trainer on **every** path, including `warm_start=True`. The production RL warm
  start therefore runs through `validate_warm_start` (registry + architecture +
  designated status) and the inference server, transferring only weights — or
  through a weights-only container. Loading a `checkpoint_type: supervised` file
  directly through `RLTrainer.load_checkpoint` is refused with a diagnostic naming
  the sanctioned flow.
* Quarantined artifacts are never auto-selected, never resumed, and never promoted,
  regardless of later file placement.
* Missing artifacts stay declared missing. No placeholder is created and no hash is
  invented.
* A wrong hash is a hard failure, never a warning-then-continue.

## 8. Where things live

| Path | Meaning |
|---|---|
| `python/common/artifacts.py` | path policy, tickets, write guards, locks, atomic JSON |
| `docs/checkpoint_lifecycle.md` | this contract (operator-facing) |
| `python/model/checkpoint_schema.py` | schema versions, run-state machine, provenance |
| `python/model/checkpoint_registry.py` | registry read/write/upsert, identity verification, warm-start gate |
| `models/checkpoints_rl/` | RL training checkpoints (`latest_checkpoint.pt`, `phase4_ckpt_v*`, `live_weights.pt` runtime sync, `best_eval.*`) |
| `models/checkpoints_scaled/`, `models/checkpoints_finetune/`, `models/checkpoints_baseline_20261031/` | supervised training trees |
| `models/checkpoints_curriculum/` | curriculum weights containers (never in the RL tree) |
| `runs/<run>/buffer/` | replay buffer memmaps + `meta.json` (with `buffer_generation`) |
