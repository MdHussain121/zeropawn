"""Generate fixed, balanced 50-opening benchmark suite in EPD format.

Each opening is verified with python-chess to be valid and legal.
The suite is saved once to data/openings/eval_openings_50.epd.
"""

from __future__ import annotations

from pathlib import Path
import chess

OPENINGS = [
    ("e4 e5 Nf3 Nc6 Bb5 a6", "Ruy Lopez Morphy"),
    ("e4 e5 Nf3 Nc6 Bb5 Nf6", "Ruy Lopez Berlin"),
    ("e4 e5 Nf3 Nc6 Bc4 Bc5", "Italian Giuoco Piano"),
    ("e4 e5 Nf3 Nc6 Bc4 Nf6", "Italian Two Knights"),
    ("e4 e5 Nf3 Nc6 d4 exd4", "Scotch Game"),
    ("e4 e5 f4 exf4", "Kings Gambit"),
    ("e4 c5 Nf3 d6 d4 cxd4", "Sicilian Open"),
    ("e4 c5 Nf3 Nc6 d4 cxd4", "Sicilian Old"),
    ("e4 c5 Nf3 e6 d4 cxd4", "Sicilian French"),
    ("e4 c5 Nc3 Nc6 g3 g6", "Closed Sicilian"),
    ("e4 c5 c3 d5 exd5 Qxd5", "Sicilian Alapin"),
    ("e4 e6 d4 d5 Nc3 Bb4", "French Winawer"),
    ("e4 e6 d4 d5 Nd2 Nf6", "French Tarrasch"),
    ("e4 e6 d4 d5 e5 c5", "French Advance"),
    ("e4 c6 d4 d5 Nc3 dxe4", "Caro-Kann Classical"),
    ("e4 c6 d4 d5 e5 Bf5", "Caro-Kann Advance"),
    ("e4 c6 d4 d5 exd5 cxd5", "Caro-Kann Exchange"),
    ("e4 d6 d4 Nf6 Nc3 g6", "Pirc Defense"),
    ("e4 d5 exd5 Qxd5 Nc3 Qa5", "Scandinavian Center"),
    ("e4 Nf6 e5 Nd5 d4 d6", "Alekhine Defense"),
    ("d4 d5 c4 e6 Nc3 Nf6", "QGD Orthodox"),
    ("d4 d5 c4 e6 Nc3 c5", "Tarrasch Defense"),
    ("d4 d5 c4 c6 Nf3 Nf6", "Slav Defense"),
    ("d4 d5 c4 c6 Nc3 e6", "Semi-Slav Defense"),
    ("d4 d5 c4 dxc4 Nf3 Nf6", "QGA 3.Nf3"),
    ("d4 Nf6 c4 g6 Nc3 Bg7", "Kings Indian Classical"),
    ("d4 Nf6 c4 g6 f3 Bg7", "Kings Indian Saemisch"),
    ("d4 Nf6 c4 e6 Nc3 Bb4", "Nimzo-Indian"),
    ("d4 Nf6 c4 e6 Nf3 b6", "Queens Indian"),
    ("d4 Nf6 c4 e6 g3 d5", "Catalan Open"),
    ("d4 Nf6 c4 c5 d5 b5", "Benko Gambit"),
    ("d4 Nf6 c4 c5 d5 e6", "Modern Benoni"),
    ("d4 Nf6 Nf3 e6 Bf4 d5", "London System"),
    ("d4 d5 Bf4 Nf6 e3 c5", "London vs d5"),
    ("d4 f5 g3 Nf6 Bg2 e6", "Dutch Leningrad"),
    ("c4 e5 Nc3 Nf6 g3 Bb4", "English Four Knights"),
    ("c4 c5 Nf3 Nf6 d4 cxd4", "Symmetrical English"),
    ("c4 e6 Nf3 d5 g3 Nf6", "English Agincourt"),
    ("Nf3 d5 g3 Nf6 Bg2 c6", "Reti Opening"),
    ("Nf3 Nf6 c4 g6 b3 Bg7", "King Indian Attack"),
    ("e4 e5 Nf3 f5 exf5 e4", "Latvian Gambit"),
    ("e4 e5 Nf3 d6 d4 exd4", "Philidor Defense"),
    ("e4 e5 Nc3 Nf6 f4 d5", "Vienna Gambit"),
    ("e4 c5 b3 d6 Bb2 Nf6", "Sicilian Nimzowitsch"),
    ("d4 d5 c4 e5 dxe5 d4", "Albin Countergambit"),
    ("d4 d5 Nc3 Nf6 Bg5 Nbd7", "Richter-Veresov"),
    ("d4 Nf6 Bg5 d5 Bxf6 exf6", "Trompowsky Attack"),
    ("b3 e5 Bb2 Nc6 e3 d5", "Nimzo-Larsen Attack"),
    ("g3 d5 Bg2 Nf6 Nf3 c6", "Benko Opening"),
    ("e4 g6 d4 Bg7 Nf3 d6", "Modern Defense"),
]


def generate_openings_epd(output_path: Path = Path("data/openings/eval_openings_50.epd")) -> Path:
    assert len(OPENINGS) == 50, f"Expected 50 openings, got {len(OPENINGS)}"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for san_moves, name in OPENINGS:
            board = chess.Board()
            for m in san_moves.split():
                board.push_san(m)
            fen = board.fen()
            f.write(f'{fen} ; id "{name}";\n')

    print(f"Successfully generated {len(OPENINGS)} openings to {output_path}")
    return output_path


if __name__ == "__main__":
    generate_openings_epd()
