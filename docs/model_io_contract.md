# ZeroPawn Model Input/Output Contract (tracked reference)

Canonical, repository-tracked statement of the neural-model interface. It
exists because the same contract facts were previously spread across vault
notes, audit reports, and code comments — including stale references (the
removed `python/model/policy_inference.py`, the "8x8x19" plane-order
phrasing). This file is authoritative for the current implementation; the
legacy internal notes is the project's general documentation (read-only for agent
sessions) and several vault notes pre-date the 2026-10-29/31 remediations.

Established by the 2026-10-31 model-architecture/inference remediation
(branch `fix/model-inference-remediation-20261031`; findings M-1, M-2, L-1,
L-2, L-3, I-1..I-5; audit baseline `main` @ `a2f348b`).

## 1. Model

- Sole chess architecture: `ChessNet` in `python/model/chess_net.py`
  (conv stem + 12 residual blocks x 128 filters). `Connect4Net` is a separate
  game. The C++ extension performs **no neural arithmetic** — it provides
  state encoding, move<->index mapping, and MCTS.
- Trainable parameters: **8,624,931**. state_dict numel: **8,631,454**
  (= params + 6,523 BatchNorm buffer elements, including
  `num_batches_tracked`). Both figures are correct in their own units; the
  difference is BatchNorm bookkeeping, not an architecture mismatch. The
  registry's architecture fallback signature (`8_631_454`) is the
  state_dict numel.
- `forward(x, legal_mask=None) -> (policy_logits (B,4672), wdl_logits (B,3))`
  returns **raw logits**; no softmax inside the model.

## 2. Input contract (19 x 8 x 8, side-to-move perspective)

Producer: `zeropawn_cpp.encode_state(game)` -> float32 `(19, 8, 8)`,
plane-major. C++: `cpp/src/encoder.cpp`. Flattened boundary form (legacy
batched path): row-major `(n, 1216)`, `plane*64 + r*8 + c`.

Orientation: own pieces are always planes 0-5; `r = rank` for White,
`r = 7 - rank` for Black; files never flipped. Consequently a
color-mirrored position ("White up a queen, White to move" vs "Black up a
queen, Black to move") encodes to **identical** planes.

| Planes | Content |
|---|---|
| 0-5 | own pieces P,N,B,R,Q,K (side to move) |
| 6-11 | opponent pieces |
| 12-15 | castling: own-KS, own-QS, opp-KS, opp-QS (side-to-move relative) |
| 16 | en-passant target square — **written only when an en-passant capture is actually legal** |
| 17 | halfmove clock / 100.0 (broadcast over the plane) |
| 18 | one-fold repetition flag (`is_repetition(1)`) — **requires move history** |

Two contract points that external tooling gets wrong:

1. **Plane 16 is not raw-FEN en passant.** The chess library normalizes
   obsolete ep fields on move application, so after `1.e4` the FEN still
   says `e3` but plane 16 is empty (no capturer exists); after
   `1.e4 c5 2.e5 d5` plane 16 is set (d6 capturable). Pinned by
   `tests/test_encoder.py::test_ep_plane_written_only_for_capturable_en_passant`.
2. **Plane 18 needs history.** Any consumer holding a live game object MUST
   pass that object to the encoder. Rebuilding the position with
   `ChessGame(game.get_fen())` erases history and zeroes plane 18
   (2026-10-31 audit M-1; the fixed 2026-10-26 H-2 class). All play paths
   (UCI, match runner, puzzles library, and the CLI `match live` /
   `match human` / `puzzles solve` callbacks via
   `python/cli/eval_callbacks.py`) encode the passed game directly; pinned by
   `tests/test_repetition_plane_parity.py`, `tests/test_repetition_play_path.py`,
   and `tests/test_cli_eval_path_history.py`.

## 3. Output contract

- WDL class order: **0 = win for the side to move, 1 = draw, 2 = loss for
  the side to move**. Training labels share this convention.
- Every consumer applies **exactly one softmax** (over dim -1) to the WDL
  logits, and exactly one to the policy logits before search masking.
- Scalar search value = `P(win) - P(loss)` in [-1, 1], side-to-move relative.
  Canonical conversion helpers: `ChessNet.wdl_logits_to_value(logits)`
  (torch, used by `predict_value`, the CLI eval factory and this contract's
  consumers), and the identical numpy form `w_probs[0] - w_probs[2]` used by
  UCI/match/puzzle callbacks and the self-play aggregator.
- The model never sees terminal positions: C++ short-circuits
  `terminal_value()` (mate/draw) before any evaluator call. Consequently the
  value head is not calibrated on checkmate positions (diagnostic note).
- Policy head: 4,672 outputs = canonical AlphaZero `8 x 8 x 73` layout
  (from-square x plane; planes 0-55 queen rays dir-major x distance,
  56-63 knight, 64-72 underpromotions n/b/r x capture file). Queen
  promotion is the plain forward-one ray move. `move_to_index` /
  `index_to_move` are the bijection used by every path; spot check: e2e4
  from startpos = 877.
- Search consumes `(policy full-softmax [4672], scalar)`; C++ masks to legal
  actions and renormalizes, which is exactly softmax-over-legal-logits
  (common denominator cancels) — matching the masked cross-entropy used in
  training. Uniform fallback fires only if legal mass <= 1e-8.
- Legacy raw batched path (`search_batched`): evaluator must return exactly
  `policy (n, 4672) float32` + `values (n,) float32`, finite; anything else
  raises (shape, dtype, NaN/Inf are all rejected — never truncated or
  zero-filled).

## 4. Checkpoint resolution (single authoritative policy)

`python/model/checkpoint_registry.py::resolve_default_checkpoint` is the one
resolver every CLI/inference consumer uses (`common.get_default_checkpoint_path`
and `models.get_default_model_path` both delegate to it). Policy:

1. The default is the registry (`docs/checkpoint_registry.json`) entry with
   `status == "verified"` AND `designated_warm_start == true`.
2. Its SHA-256 is re-computed and compared to the registry record before use
   (size pre-checked when recorded).
3. Multiple designated entries resolve deterministically (highest registry
   id wins; a warning is logged).
4. Anything else — no designated entry, missing file, hash/size mismatch,
   missing registry — raises `DefaultCheckpointError` with the expected
   artifact, the ineligibility reasons of every candidate, and recovery
   actions. There is no "newest file on disk" fallback; a checkpoint that
   merely exists or loads is not eligible (2026-10-31 audit M-2).
5. Explicit operator paths (`--checkpoint`) are honored after validation
   (strict load, architecture derivation) with provenance warnings for
   unregistered / disqualified / hash-drifted files.
6. Empty-string checkpoint semantics (`""`): "no checkpoint requested" —
   `zeropawn search` runs the uniform-policy evaluator and says so; UCI
   starts model-less (uniform evaluator) only with a warning log.

As of this writing the designated default is
`models/checkpoints_baseline_20261031/best_model.pt`
(sha256 `01cead16a445f5a2...`, registry `ckpt-10-supervised_baseline_20261031`,
step 4802). The historical `checkpoints_scaled/latest_checkpoint.pt` and
`checkpoints_finetune/latest_checkpoint.pt` are **disqualified**
(`superseded_value_head`, SLQ-01) and can never be a default; they remain on
disk for archaeology only.

## 5. Vault references known to be stale (vault is read-only)

- `01 Work/Projects/Project - Chess RL Engine.md` and
  `Reference - Chess RL Test Verification Baseline and Hardware Benchmarks.md`:
  DEF-30 references `python/model/policy_inference.py`, which no longer
  exists — the prior normalization logic lives in C++
  (`MCTS::set_priors`) and `tests/test_policy_inference.py`.
- `Reference - Chess RL Test Verification Baseline and Hardware Benchmarks.md`
  and `Reference - ZeroPawn Retired Experiment Branch Archive`: "8x8x19"
  plane-order phrasing — the actual tensor is `(B, 19, 8, 8)` (planes
  first), written here as "19x8x8" or "(19, 8, 8)".

## 6. Native game/MCTS API contract (folded from the removed `zeropawn_cpp.pyi` stub)

Audit S6-11 (2026-11-02): `python/zeropawn_cpp.pyi` was documentation-only —
no type checker was configured, it was not shipped, and it sat beside no
importable module. Its contract notes are preserved here; the stub is
removed. The API signatures remain discoverable in the pybind11 bindings
(`cpp/src/bindings.cpp`).

- **En-passant FEN policy:** `ChessGame.get_fen()` records the EP square
  after every double pawn push (classic FEN). This differs from
  python-chess's default `en_passant="legal"`; compare FENs across the
  boundary with `en_passant="fen"`. (Distinct from plane 16 above, which is
  written only when an EP capture is actually legal.)
- **Zobrist hashing:** the raw EP square is hashed even when no legal EP
  capture exists (phantom EP). Internally consistent; do not use native
  hashes as cross-library (polyglot-style) position identity.
- **Repetition / draws:** `is_repetition(count)` counts *prior* occurrences
  of the current position, so the threefold rule is `is_repetition(2)`.
  `is_game_over()` auto-draws at threefold repetition, the 50-move clock
  reaching 100, and insufficient material — a deliberate RL-efficiency
  policy, stricter than python-chess's fivefold/75-move auto-draws.
- **perft:** depth must satisfy `0 <= depth <= 32` (`perft(0) == 1`);
  negative or excessive depths raise `ValueError`.
- **Connect4:** `cell(r, c)` raises `ValueError` outside
  `[0,6) x [0,7)`; `apply_action(col)` raises `ValueError` on out-of-range
  columns, full columns, and finished games; only a single column digit
  `"0".."6"` is decodable by `string_to_action`.
- **MCTSConfig:** validated when `MCTS` is constructed — `c_puct >= 0`,
  `dirichlet_alpha > 0`, `dirichlet_weight >= 0`, `fpu_reduction >= 0`,
  `virtual_loss >= 0`, `batch_size >= 1`, `max_nodes >= 1`; violations raise
  `ValueError`. `seed` is a uint64 (values >= 2**64 overflow).
- **apply_action (ChessGame):** the action index must decode to a *legal*
  move; illegal or undecodable indices raise `ValueError` and leave the
  game state untouched.
- **Evaluator callbacks:** receive a non-owning `Game` wrapper that is valid
  only for the duration of the callback — never store it. Callbacks must not
  re-enter the owning `MCTS` (search/clear/advance_root). Evaluator output is
  validated exactly: a wrong-shaped policy or value is rejected with
  `RuntimeError`, never truncated.
- **Threading:** `cache_size()` / `live_node_count()` must be called from the
  thread that owns the MCTS while no search is in flight on it.
