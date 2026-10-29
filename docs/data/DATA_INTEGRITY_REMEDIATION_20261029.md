# ZeroPawn — Data Integrity Remediation, Leakage Prevention & Dataset Certification

**Date:** 2026-10-29
**Branch:** `fix/data-integrity-20261029` (worktree `chess-rl-data-integrity-20261029`, merged to `main`)
**Scope:** Remediation of findings F-01..F-10 from `docs/audits/data-pipeline-integrity-leakage-audit.md`.
**Production data touched:** **none.** Every dataset under `data/` was left byte-identical; all verification used read-only scans or isolated temp fixtures.

## 1. Findings status

| ID | Root cause established | Remediation | Status |
|----|------------------------|-------------|--------|
| F-01 | See §2 | Role contract + gates; artifact documented as unusable for WDL supervision | **Resolved (semantics); artifact intentionally left in place** |
| F-02/F-06 | Builder dedup keyed on the C++ engine hash, which distinguishes states the 19-plane encoder serializes identically; val candidates were filtered only against train hashes | `build_tactics.py` rewritten: puzzle-family connected components by exact serialized state before split assignment; same-state/different-target conflicts counted and recorded; exhaustive serialized-output verification gates the build | **Fixed for future builds** |
| F-03 | Fallback `date_white_black` identity merged distinct games | Identity now `Date_White_Black + sha256(mainline UCI)`; exact duplicates deterministic; result-conflicting duplicates counted | **Fixed for future builds** |
| F-04 | Validator checked byte sizes only; leakage CLI sampled and exited 0 on incomplete checks | New layered validator (A structural / B semantic / C provenance / D split-leakage) with tamper-evident receipts; leakage CLI fail-closed (exit 0 only for a definitive exhaustive pass) | **Fixed** |
| F-05 | Metadata named a 1,000-row CSV for a 757k-row shard | Lineage proven by reproduction: a deterministic rebuild from `data/tactics_train_300k.csv` (cap 6) reproduces all four small arrays byte-identically and sampled planes exactly. See §3 | **Resolved** |
| F-07 | No completion/Range/atomicity proof in downloaders | `download_file`: `.part` + HTTP 206/Content-Range validation + final byte-count enforcement + atomic publish + `.sha256`/`.provenance.json` sidecars; elite ZIP validation before reuse; fetchers write temp→verify→atomic-replace with provenance | **Fixed** |
| F-08 | Per-row RNG split, ordinal IDs, cross-split duplication fallback | Hash-based PuzzleId-family split, real source-ID hashes in `game_ids.bin`, empty-split failure, build record | **Fixed** |
| F-09 | Plane-only first-wins dedup; hard-coded independence claims | Merger classifies exact duplicates / label conflicts / target conflicts, records examples, computes benchmark overlap from actual IDs and files (or reports NOT_VERIFIED) | **Fixed** |
| F-10 | Certificate lacked output hash; evaluator logged only | Certificate now records `output_sha256`, row count, ID-list hash, identity rule, scope and explicit limitations; `verify_eval_certification` re-checks current bytes; Phase 4 refuses uncertified eval by default | **Fixed** |
| F-11 | (informational) | No change required | Closed |

## 2. F-01: canonical game shard label semantics — evidence dossier

**Question:** is the all-`255` (2,490,528/2,490,528) `dataset_training/train` shard intentionally policy-only, or inconsistent with the intended contract?

**Conclusion: the artifact is inconsistent with the intended WDL-supervised contract, and its value labels are unrecoverable. Production data was not modified.**

Evidence:

1. **Intended contract.** Committed builder code (unchanged functionally from the 2026-10-17 commit through the 10-20 build day) writes mover-perspective WDL (0/1/2) from the game `Result` header for PGN data; `OUTCOME_UNAVAILABLE=255` is reserved for tactical/masked rows. The vault note *"Reference - Chess RL Unified Training Dataset and Master Corpus Consolidation"* records the same contract (masked targets for tactics only).
2. **Build-time forensics.** All train files share one mtime (2026-10-20 11:24:49.812), 0.3 s before the val files — written by one build session (tactics built 2 min later, same session). The 255s were written **at build time**, not by a later loader overwrite. The val shard has proper W/D/L (9,335/17,771/9,209) with `candidate_val_positions`/`dedup_removed_positions` keys matching the committed `build_binary_dataset` val path.
3. **The build ran uncommitted code.** No committed version of `build_dataset.py` writes 255 for PGN games. No stash, no driver script, no build log from that session survives (`logs/` starts 2026-10-26; `build_dataset.log` has no 2026-10-20 entries).
4. **Labels cannot be reconstructed from source.** `filtered.pgn` collapses to 18,234 distinct fallback identities for 20,980 games; **602 identity groups carry conflicting `Result` headers**, so (game_id, ply) → outcome is ambiguous. Any label reconstruction would be fabrication.
5. **Latent destroyer removed.** `GpuResidentDataLoader` (dataset.py) used to open `outcomes.bin` in `r+` mode and overwrite every row with 255 whenever `tactics_dir` was set — an in-place WDL-label destroyer. This is fixed (synthetic in-memory arrays; source files are never rewritten); it did **not** cause the current artifact (mtimes prove it), but it could have destroyed any real dataset loaded that way.

**Consequence:** the current train shard may be consumed only as **policy-only** data with an explicit declaration (`--data-role policy_tactics` / `expected_role="policy_tactics"`), never as WDL-supervised game data. A future production rebuild through the fixed builder is the only way to obtain true WDL labels (mission §11 procedure; intentionally not executed).

## 3. F-05: tactics artifact lineage — resolved by reproduction

- Metadata source: `data\puzzles\tactics_train_300k.csv` (currently 1,000 rows — a subset of the 300k file; the path recorded the invocation argument, not the source content actually used).
- Verification executed 2026-10-29: `build_tactics_dataset("data/tactics_train_300k.csv", temp, max_solver_plies=6)` reproduced the production artifact **exactly**: train 757,151 / val-candidates 15,675 / dedup-removed 104 / final val 15,571 (identical to metadata); `moves.bin`, `game_ids.bin`, `plies.bin`, `outcomes.bin` **byte-identical (SHA-256)** in both splits; plane rows spot-checked at stride 997 (train) and 101 (val) — 0 mismatches.
- Lineage: **source = `data/tactics_train_300k.csv` (300,000 rows), builder = `build_tactics.py` (pre-remediation logic), config = `--max-solver-plies 6`, built 2026-10-20 11:26:51.** Current source SHA-256: `119fcffcba4fb4254a7159e9abce3150f9cde225d80dfb691559266313ed550b`. The rebuild proves ID/column equivalence with the historical source; the exact historical file bytes are not independently attested (no hash was recorded at the time), which is precisely why new builds write provenance sidecars.

## 4. New exhaustive measurements (first-time results)

The new Layer-D tooling ran the first **exhaustive** plane scans (read-only):

| Artifact | Group IDs | Exact encoded states train↔val | Within-split duplicate rows | Verdict |
|---|---|---|---|---|
| `dataset_tactics` | 0 shared (293,932 / 6,050) | **22** (22 same-target, 0 conflicting) | train 2,752 | FAIL (matches audit exactly) |
| `dataset_training` (games) | 0 shared (17,851 / 357) | **28** (19 same-target, **9 conflicting-target**) | train 568,595 | FAIL |

The 28 game overlaps are **new evidence** (the audit could only bound-check 2,000 rows): the game builder's val-dedup used the C++ engine hash, which distinguishes rule state that the encoder serializes identically. The 9 conflicting-target cases are legitimate played-move differences from different games, but they violate the strict position-disjointness contract. The builder's val dedup now keys on the exact serialized-state MD5 (same fix as tactics), so future builds cannot produce these.

## 5. Validator behavior — before vs after

| Aspect | Before | After |
|---|---|---|
| `data validate` | byte sizes only; missing `game_ids`/`plies` unnoticed; manifest fields defaulted to `PASS` | Layer A (exact sizes incl. trailing bytes, required arrays, schema), B (outcome domains, move range, plane finiteness/[0,1] with explicit coverage, value/policy target validity, role-vs-labels invariant), C (SHA-256 identity, build-record agreement), D (group-ID + encoded-state disjointness) |
| Result semantics | "PASS - 100% INTEGRITY" | `pass` only when every executed check passed; `incomplete` (exit 3) when checks were bounded/skipped — no definitive claim |
| `data check-leakage` | first-2,000-rows plane sample; exceptions printed as "skipped" and exit 0; "properly disjoint" always printed | Exhaustive by default; exit 0 = definitive disjoint, 1 = leakage, 2 = could not complete, 3 = bounded (needs `--allow-bounded`); never claims disjoint on incomplete evidence |
| Receipts | none | `validation_receipt.json` binds results to per-file SHA-256/size/mtime; consumers re-verify (deep=hash or stat-level); incomplete receipts are rejected |
| Consumption | loaders/training never validated | `require_validated_dataset` gate in supervised trainer, replay buffer (human data), Phase 4; explicit role declaration required; WDL-required role + all-masked labels is an unwaivable failure; `--allow-uncertified-data` bypasses receipts only, loudly |
| `data split` | silently created empty train/val dirs | fails with exit 2 (never implemented a split) |

## 6. Regression tests

New suite `tests/test_data_integrity.py` (23 tests) covers the twelve required scenarios: mover-perspective WDL on known PGN builds (win/loss), WDL-required all-masked rejection (validator + gate + trainer), explicit policy-only acceptance (with and without receipts), content-digest identity for same-date/player games, deterministic duplicate handling, tactics position-family integrity with adversarial shared-state/conflict fixtures, semantic failures (move range, outcome domain, NaN planes, bad value targets), receipt tamper/staleness/incompleteness, split-integrity fail-closed behavior, download truncation/invalid-206/ignored-Range/corrupt-same-size sidecar verification, eval-certificate tamper detection, and trainer pre-consumption refusal.

**Results:** `test_data_integrity.py` 23 passed; combined data suites (`test_dataset`, `test_split_integrity`, `test_download`, `test_puzzle_dataset`, `test_tactics_builder_smoke`, `test_replay_buffer`, `test_data_integrity`) **49 passed**; `test_dataset_training`, `test_tactical_sampling`, `test_puzzles` **17 passed, 5 skipped** (pre-existing skips). Two legacy test expectations were updated for intended behavior changes (deterministic split assignment in `test_puzzle_dataset`; fixture role declarations in `test_replay_buffer`).

Production verification (read-only): `data validate` on `dataset_tactics` and `dataset_training` FAILs exactly as tabled in §4; `data check-leakage` on tactics exits 1 with the 22 overlaps classified. `GpuResidentDataLoader` regression test proves source arrays are never rewritten.

## 7. Artifacts built

No production artifact was built or replaced. All fixture builds were in temp directories. The one controlled rebuild (F-05 verification) wrote to a temp dir and was deleted after comparison.

## 8. Consumers protected by gates

- `python/train/train_supervised.py` — role + receipt gate before any loader/optimizer work (`--data-role`, `--allow-uncertified-data`, `--verify-identity`).
- `python/selfplay/replay_buffer.py` — human-data ingestion gate.
- `scripts/run_phase4_rl.py` — human-data gate via buffer; certified-eval gate (`require_certified_eval`, `--allow-uncertified-eval`, `--no-require-certified-eval`).
- `python/data/dataset.py` — `GpuResidentDataLoader` no longer writes to source datasets.

## 9. Commits & Git state

Logical commits on `fix/data-integrity-20261029`, merged into `main` (no push, no branch/worktree deletion). See `git log` for the final hashes; integration was verified with post-merge focused test runs.

## 10. Remaining limitations & next action

1. **The current production `dataset_training/train` still has no usable WDL labels** (F-01 artifact). Until a controlled rebuild (mission §11) is approved and executed, consume it only with an explicit `policy_tactics` declaration, and do not report its value metrics as game-outcome learning.
2. The legacy v1/v2 builders (`build_tactics_dataset.py`, `build_elite_master_dataset.py`, `build_master_additional.py`) still emit hard-coded manifest claims; they are not on any active build path — remediate or retire them before their next use.
3. Within-train duplicate positions (games: 568,595 rows; tactics: 2,752) are reported but deliberately not removed for games (legitimate repeated positions across games); reconsider weighting policy separately.
4. Adaptive evaluation history remains unreconstructable; the exposure-ledger work (audit P3) is future work requiring runtime logging, not backfill.

**Highest-priority next action:** approve and execute the controlled production rebuild of `dataset_training` from `data/filtered.pgn` using the fixed builder (staging directory, full validation + certification, atomic activation, rollback retained), then re-run Phase 4 with certified receipts.
