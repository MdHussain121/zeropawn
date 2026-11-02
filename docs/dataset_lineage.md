# ZeroPawn Supervised Dataset Lineage — `data/dataset_training`

**Status:** authoritative record, established 2026-11-03 during the post-merge
production-readiness remediation (branch
`fix/post-merge-production-readiness-20261103`), closing audit finding **S2-01**
(undocumented third dataset rebuild; checkpoint↔receipt binding drift).
**This document records lineage only. No dataset sample, label, or split file
was read-modified; all row counts below were verified independently by
byte-size arithmetic against the on-disk artifacts.**

Row arithmetic authority: `planes.bin` is `rows × 19×8×8 × float32` =
`rows × 1216 × 4` bytes; `outcomes.bin` is 1 byte/row; `game_ids.bin` is 8
bytes/row. All three agree exactly for every generation below.

## Generation 1 — 2026-10-20 build (DI-01 failing; QUARANTINED)

| Property | Value |
|---|---|
| Composition | train 2,490,528 / val 36,315 (two splits) |
| Source | `data/elite_2500.pgn` era build (uncommitted sentinel-writing builder variant) |
| Defect | 100% of value labels masked (DI-01 Critical — no WDL supervision possible) |
| Current state | `data/QUARANTINE_dataset_training_superseded_20261031/` (no QUARANTINE.json in that dir; quarantined during the 2026-10-31 SL remediation) |
| Consumed by | nothing remaining |

## Generation 2 — 2026-10-31 SL-remediation rebuild (TRAINED ckpt-10; QUARANTINED)

| Property | Value |
|---|---|
| Composition | train 2,458,622 / val 36,767 / test 19,374 |
| Source | `data/filtered.pgn` |
| Build window | 2026-10-31 12:32–18:06 (commit-era SL remediation) |
| Receipt | `sha256:156d528959dd463c`, overall=pass, definitive; `masks.bin`+`stm.bin` present; dedup active |
| Consumer | **designated supervised baseline ckpt-10** (`models/checkpoints_baseline_20261031/best_model.pt`, run `20261031-125838-b8868fd6`, steps 0→4802) |
| Later disposition | quarantined as `data/QUARANTINE_dataset_training_unfiltered_bot_source_20261031/` (DI-08: BOT games present in the source PGN). The original receipt and `remediation_validation_20261031.json` are preserved inside that directory |
| Consumed by today | nothing live; **ckpt-10 remains a trained-on-generation-2 artifact** |

## Generation 3 — 2026-11-01 rebuild (CURRENT LIVE dataset; BOT-purged)

| Property | Value |
|---|---|
| Composition | **train 384,587 / val 6,120 / test 3,266** (verified 2026-11-03 by byte arithmetic + the definitive receipt) |
| Source | `data/filtered_elite_human_2500.pgn` (sha256 `87f77b11…`) — from `data/elite_2500.pgn` (20,980 games) with **16,269 BOT games purged**, 4,702 human games retained |
| Build | 2026-11-01T12:32:08Z, build commit `61193202`, schema 1.1, `build_record.json` (builder/env/label_semantics/outputs recorded), dedup removed 967 val + 563 test |
| Receipt | `validation_receipt.json` overall=**pass**, definitive=**true**, 39/39 checks pass (restored 2026-11-03 via a full-plane-scan `data validate --write-receipt` after the 2026-11-02 accidental bounded-scan rewrite documented in `docs/checkpoint_registry.json` history and the 2026-11-03 remediation; dataset bytes unchanged since 2026-11-01 18:02) |
| Trained checkpoints | **none** — no production checkpoint has been trained on generation 3 |
| Provenance record | this document + `models/checkpoints_baseline_20261031/run_manifest.addendum_20261103.json` (addendum, not a rewrite of the historical manifest) |

> The 2026-11-01 rebuild was performed correctly (best-certified generation so
> far: genuine WDL labels, BOT-free source, dedup, definitive receipt) but was
> not recorded in the vault or repo docs at the time (audit S2-01). This
> document is that missing record. Note the split-shrinkage caveat: val/test
> draw from only 79/46 distinct games, so split-level metrics carry higher
> game-level correlation than generation 2.

## Checkpoint ↔ dataset compatibility contract

1. **No silent relabeling of history.** A checkpoint's embedded/manifest
   provenance always refers to the generation it actually trained on. ckpt-10
   is and remains a generation-2 artifact; nothing may represent it as
   trained on generation 3.
2. **Load-time distinction.** The DI-04 checkpoint↔dataset binding compares
   receipt identities at load: resuming/warm-starting from ckpt-10 against
   generation 3 fires the explicit dataset-receipt mismatch warning (verified
   2026-11-03) — loud, non-fatal, by design. `allow_uncertified_data` never
   bypasses the role-vs-labels invariant.
3. **Quarantined generations are inert.** `QUARANTINE_dataset_*` directories
   are not discoverable as datasets by any consumption gate (they live outside
   the consumed paths and carry quarantine markers). Their receipts describe
   only the quarantined bytes.
4. **ckpt-10 usability.** ckpt-10 remains valid as a weights/init source and
   as a supervised `--resume` source (legacy pre-schema ledger accepted per
   the 2026-11-03 checkpoint remediation). Any *reported training result*
   produced from generation 3 must state that the warm start came from a
   different data generation (distribution shift: 6.4× smaller, human-only).
