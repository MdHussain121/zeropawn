# Local Patches to the Vendored chess-library

This directory vendors Disservin's chess-library. Despite the "AUTO GENERATED
DO NOT CHANGE MANUALLY" banner in `chess.hpp`, the vendored copy carries
**exactly one local functional patch** relative to upstream. This file is the
authoritative record of it (audit finding S1-01, 2026-11-02; the earlier
statement "the vendored header is unmodified" in the chess-rules remediation
report was wrong and is superseded here).

## Vendored base

- Upstream repository: https://github.com/Disservin/chess-library
- Version: **0.9.4** (the project uses no git tags; identified by the
  "bump to 0.9.4" commit of 2026-03-29)
- Vendored at upstream commit **`61e0a694`** (the last 0.9.4-era commit,
  before the 0.9.5 bump of 2026-10-22). The vendored file equals upstream
  `61e0a694` except for the single hunk below. (Differences seen against the
  earlier `8ea1fd17` snapshot — the `Movelist` copy/move constructors and the
  `backtrackTo` docstring — are upstream 0.9.4-development evolution present
  in `61e0a694`, not local patches.)

## Patch 1 — `Square::is_valid()` must reject `NO_SQ` (negative squares)

Location: `chess.hpp` lines 417-419, inside `class Square`.

```cpp
// vendored (chess.hpp:417-419)                        // upstream 0.9.4 (61e0a694)
[[nodiscard]] constexpr bool is_valid() const noexcept {
    return static_cast<std::int8_t>(sq) >= 0 &&        return static_cast<std::int8_t>(sq) < 64;
           static_cast<std::int8_t>(sq) < 64;
}
```

The vendored version adds the `>= 0` lower bound; upstream only checks
`< 64`.

### Why it is load-bearing

`Board::enpassantSq()` returns `Square::NO_SQ` (`-1`) when no en-passant
target is set. The ZeroPawn encoder guards the EP plane with that predicate
(`cpp/src/encoder.cpp:53`, `if (ep.is_valid())`). Under upstream semantics,
`-1 < 64` passes, and the guard accepts `NO_SQ`; the subsequent
`ep.index()` of `-1` writes to flat index `16*64 + 0*8 - 1 = 1023` — the
last cell of input **plane 15** (opponent queen-side castling) — silently
corrupting every encoded position without an EP square. With the patch,
`NO_SQ` is rejected and plane 15 stays intact.

### Consequences for maintenance

- The file's own banner ("THIS FILE IS AUTO GENERATED DO NOT CHANGE
  MANUALLY") is **not literally true** for this hunk: do not blindly
  overwrite `chess.hpp` with a fresh upstream copy. When bumping the vendor
  version, re-apply or re-verify this hunk (and ideally check whether
  upstream has since fixed `is_valid()`/the EP accessor semantics so the
  patch can be dropped).
- Do not "fix" the encoder by removing the `is_valid()` guard instead — the
  guard is correct only because this patch makes the predicate total.
- Upstreaming the guard to Disservin/chess-library would be welcome; until
  then this file documents the deviation.
