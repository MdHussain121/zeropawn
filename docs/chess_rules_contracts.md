# Chess Rules Contracts — FEN Parsing, En-Passant Identity, Terminal Conventions

Authoritative statement of ZeroPawn's chess-rules boundary behavior. The rules
authority itself is the vendored **Disservin/chess-library v0.9.4**
(`cpp/vendor/chess-library/chess.hpp`); `ChessGame` (`cpp/src/chess_game.cpp`)
is a thin fail-closed wrapper. This document records the parser's accepted-input
contract, the en-passant position-identity policy, and the intentional engine
terminal conventions. Regression tests: `tests/test_chess_rules_remediation.py`
and `cpp/tests/test_main.cpp::test_chess_rules_remediation`.

## 1. FEN parser contract (`ChessGame::set_fen`)

Every input is classified before the vendored parser runs. A structural gate
(`validate_fen_structure`) rejects malformed board layouts and out-of-range
counters; the vendored parser then handles the remaining fields. Rejection is
fail-closed: an invalid FEN never mutates existing game state (board, hash,
counters, move history are untouched), matching the DEF-02 contract.

| Class | Behavior | Examples |
|---|---|---|
| **Rejected (structural)** | `set_fen` returns false / constructor raises `ValueError`; prior state preserved | rank ≠ 8 squares (`qqqqqqqqK`, `4K`, `44K`); ranks ≠ 8 (`7` ranks, `9` ranks, `//`, trailing `/`); piece chars outside `PNBRQKpnbrqk`; compression digits `0`/`9`; more than 6 fields; consecutive spaces (empty field); not exactly one king per side; any trailing non-whitespace byte inside a counter field (e.g. a CRLF `\r` surviving in `"... 0 1\r"` — strip line endings before calling) |
| **Rejected (counters)** | same fail-closed behavior | halfmove not an integer in `[0, 255]` (`-5`, `256`, `1x`, `+5`, `5.5`, empty field via consecutive spaces); fullmove not an integer in `[1, 32768]` (`-3`, `0`, `32769`) |
| **Rejected (semantic, pre-existing)** | unchanged from C-1 / wrapper guards | side token other than `w`/`b`; EP square syntax invalid (`e9`, `z3`); side *not* to move in check (includes adjacent kings) |
| **Normalized (accepted)** | deterministic, safe-direction; see §2 | garbage castling characters; impossible castling rights; impossible / no-capturer EP squares; missing trailing fields (side `w`, castling `-`, EP `-`, halfmove `0`, fullmove `1` — a 4-field FEN is valid and its counters default); pawns on back ranks |

A genuinely missing trailing field (e.g. 4-field FENs without counters) is
defaulted, per the normalization policy; only an *empty field between
separators* (consecutive spaces) is rejected, because the vendored splitter
would silently turn it into a default.

King-count validation happens in the pre-parse gate, not after the vendored
parse: the vendored `setFenCommon` calls `kingSq(c)` for **both** colors
unconditionally while initializing its castling-path table (before any king
validation and before the `cr_.has()` skip), and `lsb()` of an empty king
bitboard is undefined — any FEN without exactly one king per side can crash
there with an out-of-bounds table access (flaky segfault, reproduced at ~70%
per attempt on the pre-remediation build with
`r3K2r/…/R3K2R w KQkq - 0 1`, Kiwipete with the black king replaced by a
white king; remediation finding REM-CR-08). Gating king counts in
`validate_fen_structure` keeps such input out of the vendored parser. The
vendored header itself is unmodified; this wrapper-side gate is the minimal
fix (an upstream `require_kings`-before-castling-path reorder is the
natural vendor-side remedy to report).

Counter bounds are storage-derived: the halfmove clock is stored in `uint8`
(bound 255) and the ply counter in `uint16` (a fullmove `N` maps to
`plies = 2N - 2 (+1 if black to move)`, bound `N ≤ 32768`). Values outside
these ranges previously wrapped silently (halfmove `-5` became `251`, instantly
reporting the 50-move draw state); they are now rejected.

Normalization is deterministic and idempotent: the serialized form of a
normalized position re-parses to the identical state
(`ChessGame(fen).get_fen()` is a fixed point except for non-canonical but valid
rank compressions such as `44`, which serialize canonically as `8` while
preserving the board exactly).

## 2. En-passant identity policy (canonical internal convention)

ZeroPawn uses **one** en-passant convention everywhere, internal and
serialized:

> The en-passant square exists (in `ep_sq_`, the Zobrist hash, the FEN output,
> and the encoder's EP plane) **if and only if** an en-passant capture is
> fully legal for the side to move — capturer present *and* the capture does
> not expose the capturing side's king (`isEpSquareValid`).

- **Play path:** all move application (`make_move_uci`, `apply_action`,
  `compute_perft`) uses the vendor's `makeMove<EXACT=true>`, which performs
  the same `isEpSquareValid` test at play time. (Before the CR-01 remediation
  the default `EXACT=false` template recorded the square on a pseudo-legal
  attack test, so the same physical position carried different hashes
  depending on whether it was reached by play or by FEN load.)
- **FEN-load path:** the vendor `setFen` already validated EP with
  `isEpSquareValid`; unchanged.
- **Consequences:** `get_fen()` is idempotent; play-reached and FEN-loaded
  copies of the same position share one Zobrist hash, one serialized FEN, one
  legal-move list, and one repetition identity. Perft counts are unaffected
  (EP legality was always computed independently of the recorded square).
- **Preserved intentional nuance (diagonal/family cases):** a rook pin along
  the double-pushed pawn's *file* does not suppress the EP capture — the
  capturing pawn lands on the ep square (e.g. `d3`), still blocking the ray.
  The horizontal two-pawn removal case and diagonal pins of the capturer do
  suppress it, on both paths identically.

This supersedes the earlier documented differences F-07/F-08 of the
cpp-python-parity audit ("classic FEN EP serialization", "phantom-EP Zobrist
hashing"): those described the pre-remediation play path and no longer exist.
Serialized EP output now matches python-chess's default (legal-EP) convention.

## 3. Terminal conventions (intentional engine policy)

ZeroPawn is **not FIDE-exact** by design; the deviations below are deliberate
RL-efficiency choices applied consistently by the native `is_game_over()` and
every consumer (self-play worker, match runner, search terminal handling):

| Rule | ZeroPawn | FIDE |
|---|---|---|
| Checkmate / stalemate | exact; **mate takes precedence** over every draw rule (evaluated first) | same |
| 50-move rule | **automatic draw at halfmove clock ≥ 100** | claimable at 100, automatic at 150 |
| 75-move rule | not implemented — subsumed by the automatic 100-halfmove draw (strictly earlier) | automatic at 150 |
| Threefold repetition | **automatic draw at the 3rd occurrence** (`is_repetition(2)` counts prior occurrences) | claimable at 3, automatic at 5 |
| Fivefold repetition | not implemented — subsumed by the automatic threefold draw (strictly earlier) | automatic at 5 |
| Insufficient material | material-only classifier, see §4 | dead-position rule |
| Draw claims | not modeled; automatic draws only | claim-based mechanisms |

Because both deviations fire *strictly earlier* than their FIDE automatic
counterparts, a game that ZeroPawn ends by convention would under FIDE either
be claimable (draw) or continue toward the same automatic draw; a checkmate
delivered on the 100th halfmove still wins (mate priority), matching FIDE.

## 4. Insufficient-material classifier

`ChessGame::is_insufficient_material` mirrors python-chess's
`has_insufficient_material` for both colors (the project oracle), which reduces
to the vendored ≤4-piece verdicts and extends the same-color-bishop rule to
any piece count:

- Any pawn, rook, or queen on a side ⇒ that side has sufficient material.
- King + exactly one knight ⇒ insufficient only if the opponent holds nothing
  but kings and queens.
- Bishops on a side ⇒ insufficient only if **all** bishops on the board (both
  sides) share one square color and there are no pawns or knights anywhere.
- King alone ⇒ always insufficient.

Known documented convention (shared with the oracle and the pre-remediation
vendor behavior): single-minor-piece endings are classified insufficient under
the traditional convention even though contrived helpmates exist for some
(e.g. KNN vs K is correctly *not* insufficient; KN vs K is classified
insufficient). No false positive beyond that documented family is possible:
every configuration the classifier calls dead provably cannot produce
checkmate (all-bishops-one-color with no pawns/knights cannot cover the
mated king's opposite-colored escape squares).

## 5. UCI opening-book depth accounting

The engine tracks the true game ply: `prev_base_ply` is captured from the bare
position's own metadata at each `position` command (startpos ⇒ 0; a FEN at
fullmove `N` ⇒ `(N-1)*2 [+1 if black to move]`), and book lookups pass
`current_ply = prev_base_ply + len(replayed moves)`. FEN-loaded games
therefore respect `BookMaxPly` from their actual depth instead of restarting
at 0. Selected book moves remain legality-gated; the ply derived from the FEN
fullmove field does not reconstruct any pre-FEN history.
