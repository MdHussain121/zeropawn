# Licensing Decision Record

Status: **OPEN DECISION — requires owner authorization.**
Recorded: 2026-11-03 (post-merge production remediation, audit finding S6-13).
Neutral record of facts and options; this document is not legal advice.

## 1. Current posture (as shipped)

| Surface | License | Source of record |
| --- | --- | --- |
| ZeroPawn repository code (Python + C++) | MIT | `LICENSE`, `pyproject.toml` (`license = MIT` + MIT classifier) |
| Runtime dependency `python-chess` (imports as `chess`) | GNU GPL-3.0 | `pyproject.toml` dependencies; THIRD_PARTY_NOTICES §2 |
| Vendored `chess-library` (C++, header-only) | MIT | THIRD_PARTY_NOTICES §1; `cpp/vendor/chess-library/` |
| `pybind11` | BSD 3-Clause | THIRD_PARTY_NOTICES §3 |
| Stockfish | GNU GPL-3.0 | **Process-separated** (UCI subprocess), never redistributed by this repository; `tools/` is gitignored (THIRD_PARTY_NOTICES §4) |
| `tools/`, `runs/`, `models/`, `data/`, `scratch/` | not distributed | gitignored operator/deployment surfaces |

The combined-distribution implication is already acknowledged in-repo:
`README.md` (Licensing & Third-Party Notices section) and
`THIRD_PARTY_NOTICES.md` §2 both state that bundling or distributing
ZeroPawn together with GPLv3 `python-chess` (e.g. as a standalone binary)
subjects the combined work as a whole to GNU GPLv3 terms. Source-only
distribution of this repository remains under MIT for the original code.

## 2. The previously cited (but missing) review

The project's architecture review (vault, "Repository Architecture and
Cross-Platform Portability" §7) recommended that **the entire repository
should be formally licensed under GNU General Public License v3.0** upon
public GitHub release, and cited `docs/LICENSING_DECISION.md` as its record.
That file did not exist in the tracked tree (verified 2026-11-02, audit
S6-13) — this document is that missing record. The review's GPL-3.0
conclusion was **not** applied: the shipped `LICENSE` remained MIT.

## 3. The open decision

Two candidate postures are on the table. Choosing between them is an owner
decision; neither has been authorized to date, so the repository stays in
the documented §1 posture until then.

### Option A — Keep MIT (original code) + GPLv3 notice (status quo)

- The repository stays MIT; the README/THIRD_PARTY_NOTICES hedge remains the
  compliance mechanism for combined distributions.
- Lowest friction for permissive downstream use of the original code.
- Consequence: any distribution that bundles python-chess (wheels that
  install it, binaries, app bundles) is effectively GPLv3 as a whole anyway,
  so the practical licensing of a *usable* distribution is already GPL-3.0
  regardless of the repo's own license field.
- Consequence: the vault review's GPL-3.0 recommendation would remain
  unimplemented and this record should state that the owner chose to
  overrule it (with reasons).

### Option B — Adopt GPL-3.0 for the whole repository

- Relicense `LICENSE`, `pyproject.toml` metadata/classifiers, and the
  README license section to GPL-3.0, matching the review's conclusion and
  the effective license of any combined distribution.
- Consequence: permissive-licensed downstream consumers of the original code
  (MIT/BSD projects) can no longer take it under their own license terms;
  GPL-3.0 obligations (source availability on distribution) apply to the
  original code itself, not only to combined works.
- Consequence: mechanical changes required (LICENSE text, pyproject license
  expression + classifier, README badge/section, CITATION.cff license note,
  THIRD_PARTY_NOTICES preamble); all authors of the original code must agree
  (this project has a single author lineage — "ZeroPawn Maintainers" — which
  makes relicensing administratively simple, but the decision is still the
  owner's).

### Not in either option

- Changing the python-chess dependency or the Stockfish process separation —
  both stay as-is in every outcome. Stockfish is not distributed by this
  repository and is never linked; it remains an optional, operator-installed
  subprocess.

## 4. Decision log

- 2026-11-03 — S6-13 remediation: created this record; documented the
  current posture and the open decision. No license change made. Follow-up
  owner action: pick Option A (and record the overruling rationale) or
  Option B (and execute the mechanical relicense list above).
