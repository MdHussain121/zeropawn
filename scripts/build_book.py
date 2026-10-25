"""Build a high-quality self-play opening book from elite games.

Quality requirements (what makes a book "good"):
  1. POPULAR moves, not just legal ones: build a move tree from high-rated
     games counting every move at every node; book lines follow only the
     most-played continuations (masters vote with their moves).
  2. STRONG source: elite 2500+ games (data/elite_2500.pgn).
  3. USEFUL depth: lines are 6-12 plies, deep enough to dodge prep,
     short enough that the engine's own search takes over in relevant
     positions.
  4. NAMED openings: lines tagged with ECO code + name from Lichess's
     chess-openings dataset (fetched once, cached).
  5. COVERAGE + balance: cap lines per first move so e4 doesn't drown out
     Nf3/c4/g3; transpositions deduped by EPD.
  6. VERIFIED: every line replayed through python-chess before writing.

Usage:
    python scripts/build_book.py --pgn data/elite_2500.pgn \\
        --book-plies 12 --min-count 5 --max-lines 400 \\
        --out data/openings/book_selfplay
"""

from __future__ import annotations

import argparse
import json
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

import chess
import chess.pgn

ECO_TSV_URLS = [
    f"https://raw.githubusercontent.com/lichess-org/chess-openings/master/{c}.tsv"
    for c in ("a", "b", "c", "d", "e")
]
ECO_CACHE = Path("data/openings/eco_openings.json")


def _epd(board: chess.Board) -> str:
    """Transposition-stable key: piece placement + turn + castling + ep."""
    return " ".join(board.fen().split()[:4])


def load_eco_table() -> list[dict]:
    """ECO entries [{eco, name, uci}] from lichess-org/chess-openings (cached)."""
    eco_cache = ECO_CACHE
    if not eco_cache.is_file() and not eco_cache.is_absolute():
        try:
            from python.cli.common import REPO_ROOT

            if (REPO_ROOT / eco_cache).is_file():
                eco_cache = REPO_ROOT / eco_cache
        except Exception:
            pass
    if eco_cache.is_file():
        try:
            return json.loads(eco_cache.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            pass
    eco: list[dict] = []
    for url in ECO_TSV_URLS:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "ZeroPawn/1.0"})
            text = urllib.request.urlopen(req, timeout=60).read().decode("utf-8")
            for line in text.splitlines()[1:]:
                parts = line.split("\t")
                # TSV: eco, name, pgn (SAN movetext) - convert SAN -> UCI
                if len(parts) < 3 or not parts[2]:
                    continue
                try:
                    board = chess.Board()
                    ucis: list[str] = []
                    for token in parts[2].split():
                        if token.endswith(".") or token.isdigit():
                            continue
                        mv = board.parse_san(token)
                        ucis.append(mv.uci())
                        board.push(mv)
                except Exception:
                    continue
                eco.append({"eco": parts[0], "name": parts[1], "uci": ucis})
        except Exception as e:
            print(f"  ECO fetch failed for {url}: {e}")
    if not eco:
        print("  ECO table empty - not caching (will retry next run)")
        return eco
    print(f"  ECO table: {len(eco)} named openings")
    ECO_CACHE.parent.mkdir(parents=True, exist_ok=True)
    ECO_CACHE.write_text(json.dumps(eco), encoding="utf-8")
    return eco


def tag_eco(uci_line: list[str], eco_table: list[dict]) -> tuple[str, str]:
    """Longest-prefix ECO match for a book line ('' if match < 4 plies)."""
    best = ("", "", 0)
    for e in eco_table:
        n = 0
        for a, b in zip(uci_line, e["uci"]):
            if a != b:
                break
            n += 1
        if n > best[2]:
            best = (e["eco"], e["name"], n)
    if best[2] >= 4:
        return best[0], best[1]
    return "", ""


def build_book(
    pgn_path: Path,
    book_plies: int = 12,
    min_count: int = 5,
    max_lines: int = 400,
    max_per_first: int = 80,
) -> list[dict]:
    # 1) move tree: epd -> {uci: count} over the first book_plies of every game
    tree: dict = defaultdict(lambda: defaultdict(int))
    games = 0
    with open(pgn_path, "r", encoding="utf-8", errors="replace") as f:
        while True:
            game = chess.pgn.read_game(f)
            if game is None:
                break
            games += 1
            board = game.board()
            for ply, mv in enumerate(game.mainline_moves()):
                if ply >= book_plies:
                    break
                tree[_epd(board)][mv.uci()] += 1
                board.push(mv)
    print(f"Scanned {games:,} elite games -> {len(tree):,} book nodes")

    # 2) extract popular lines: DFS from root following edges with
    #    count >= min_count, most-played first, up to 2 branches per node
    #    (keeps strong alternatives, drops rare 3rd+ options).
    lines: list[dict] = []
    seen_epds: set = set()

    def walk(board: chess.Board, ucis: list[str], depth: int) -> None:
        popular = [(u, c) for u, c in tree.get(_epd(board), {}).items() if c >= min_count]
        if not popular or depth >= book_plies:
            if len(ucis) >= 6 and _epd(board) not in seen_epds:
                seen_epds.add(_epd(board))
                # popularity = games that played this line's FINAL move
                b2 = chess.Board()
                cnt = 0
                for u in ucis:
                    cnt = tree.get(_epd(b2), {}).get(u, 0)
                    b2.push_uci(u)
                lines.append({"fen": board.fen(), "uci": list(ucis), "pop": cnt})
            return
        popular.sort(key=lambda x: -x[1])
        # at the ROOT branch into every popular first move (coverage!);
        # deeper nodes take the 2 strongest continuations (mainline + alt)
        branch = popular if depth == 0 else popular[:2]
        for u, _c in branch:
            board.push_uci(u)
            walk(board, ucis + [u], depth + 1)
            board.pop()

    walk(chess.Board(), [], 0)
    print(f"Extracted {len(lines)} raw popular lines")

    # 3) balance across first moves + global cap
    buckets: dict = defaultdict(list)
    for ln in lines:
        buckets[ln["uci"][0]].append(ln)
    kept: list[dict] = []
    for _first, bucket in buckets.items():
        bucket.sort(key=lambda ln: -ln["pop"])
        kept.extend(bucket[:max_per_first])
    kept.sort(key=lambda ln: -ln["pop"])
    kept = kept[:max_lines]

    # 4) ECO tags + SAN + legality verification
    eco_table = load_eco_table()
    verified: list[dict] = []
    for ln in kept:
        try:
            b = chess.Board()
            sans: list[str] = []
            for u in ln["uci"]:
                mv = chess.Move.from_uci(u)
                if mv not in b.legal_moves:
                    raise ValueError(f"illegal {u}")
                sans.append(b.san(mv))
                b.push_uci(u)
            eco, name = tag_eco(ln["uci"], eco_table)
            verified.append(
                {
                    "fen": b.fen(),
                    "uci": ln["uci"],
                    "san": sans,
                    "eco": eco,
                    "name": name,
                    "count": ln["pop"],
                    "plies": len(ln["uci"]),
                    "first": ln["uci"][0],
                }
            )
        except Exception:
            continue  # never write an unsound line
    spread = dict(sorted(Counter(v["first"] for v in verified).items()))
    print(f"Verified {len(verified)} lines (first-move spread: {spread})")
    named = sum(1 for v in verified if v["eco"])
    print(f"ECO-tagged: {named}/{len(verified)}")
    return verified


def main() -> None:
    parser = argparse.ArgumentParser(description="Build high-quality opening book.")
    parser.add_argument("--pgn", "--input", dest="pgn", type=str, default="data/elite_2500.pgn")
    parser.add_argument("--book-plies", "--max-ply", dest="book_plies", type=int, default=12)
    parser.add_argument("--min-count", type=int, default=5)
    parser.add_argument("--max-lines", type=int, default=400)
    parser.add_argument("--max-per-first", type=int, default=80)
    parser.add_argument(
        "--out", "--output", dest="out", type=str, default="data/openings/curated_book.json"
    )
    args = parser.parse_args()

    pgn_path = Path(args.pgn)
    if not pgn_path.is_file() and not pgn_path.is_absolute():
        try:
            from python.cli.common import REPO_ROOT

            if (REPO_ROOT / pgn_path).is_file():
                pgn_path = REPO_ROOT / pgn_path
        except Exception:
            pass

    book = build_book(pgn_path, args.book_plies, args.min_count, args.max_lines, args.max_per_first)
    out = Path(args.out)
    if out.suffix.lower() == ".json":
        json_path = out
        epd_path = out.with_suffix(".epd")
    elif out.suffix.lower() == ".epd":
        epd_path = out
        json_path = out.with_suffix(".json")
    elif out.suffix.lower() == ".bin":
        base = out.with_suffix("")
        json_path = Path(f"{base}.json")
        epd_path = Path(f"{base}.epd")
    else:
        json_path = Path(f"{out}.json")
        epd_path = Path(f"{out}.epd")

    json_path.parent.mkdir(parents=True, exist_ok=True)
    epd_path.parent.mkdir(parents=True, exist_ok=True)

    with open(epd_path, "w", encoding="utf-8") as f:
        for i, line in enumerate(book):
            tag = f"{line['eco']} {line['name']}" if line["eco"] else f"line-{i:03d}"
            f.write(f'{line["fen"]} ; id "b{i:03d} x{line["count"]} {tag}";\n')
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"plies": args.book_plies, "source": str(pgn_path), "lines": book}, f, indent=2)
    print(f"Wrote {len(book)} verified lines to {epd_path} / {json_path}")

    try:
        from python.cli.common import REPO_ROOT

        locations = [
            REPO_ROOT / "data/openings/curated_book.json",
            Path("data/openings/curated_book.json"),
        ]
        content = json_path.read_text(encoding="utf-8")
        epd_content = epd_path.read_text(encoding="utf-8")
        for loc in locations:
            try:
                if not loc.resolve().samefile(json_path.resolve()):
                    loc.parent.mkdir(parents=True, exist_ok=True)
                    loc.write_text(content, encoding="utf-8")
                    loc.with_suffix(".epd").write_text(epd_content, encoding="utf-8")
            except Exception:
                pass
    except Exception:
        pass


if __name__ == "__main__":
    main()
