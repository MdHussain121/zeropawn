"""Regression tests for the chess-rules & position-state remediation (CR-01..CR-07).

Closes the blind spot that hid CR-01 from the existing suite (audit §24.3):
equivalent positions reached through play and through FEN loading must share
one internal identity policy, malformed FEN must be rejected instead of
silently transformed, counters cannot wrap into false draw states, the
insufficient-material classifier must agree with the python-chess oracle,
and UCI book-depth accounting must use the true game ply.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import chess  # python-chess: independent comparison oracle

from python.uci.uci_engine import UCIEngine
import zeropawn_cpp

STARTPOS = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

# ---------------------------------------------------------------------------
# CR-01 — en-passant position identity
# ---------------------------------------------------------------------------

# Audit minimal repro: after 1.d4 the black e4 pawn's EP capture is
# pseudo-legal but illegal (exd3 e.p. vacates d4+e4, opening the Qh4-a4 rank
# against the a4 king).
CR01_WHITE = "8/8/8/8/k3p2Q/8/3P4/3K4 w - - 0 1"
CR01_WHITE_AFTER = "8/8/8/8/k2Pp2Q/8/8/3K4 b - - 0 1"
# Vertical mirror with colors swapped (black plays the double push).
CR01_BLACK = "3k4/3p4/8/8/K2pP2q/8/8/8 b - - 0 1"


class TestCREnPassantIdentity:
    def test_play_and_fen_paths_agree_white(self):
        play = zeropawn_cpp.ChessGame(CR01_WHITE)
        assert play.make_move_uci("d2d4")
        load = zeropawn_cpp.ChessGame(play.get_fen())
        assert play.get_hash() == load.get_hash()
        assert play.get_fen() == load.get_fen() == CR01_WHITE_AFTER
        assert play.legal_moves_uci() == load.legal_moves_uci()

    def test_play_and_fen_paths_agree_black(self):
        play = zeropawn_cpp.ChessGame(CR01_BLACK)
        assert play.make_move_uci("d7d5")
        load = zeropawn_cpp.ChessGame(play.get_fen())
        assert play.get_hash() == load.get_hash()
        assert play.get_fen() == load.get_fen()

    def test_illegal_ep_square_not_serialized(self):
        play = zeropawn_cpp.ChessGame(CR01_WHITE)
        play.make_move_uci("d2d4")
        assert " d3 " not in play.get_fen()
        assert " - " in play.get_fen()

    def test_legal_ep_still_recorded_and_consistent(self):
        # Black pawn on d4 makes exd3 e.p. fully legal after e2e4.
        play = zeropawn_cpp.ChessGame("4k3/8/8/8/3p4/8/4P3/4K3 w - - 0 1")
        assert play.make_move_uci("e2e4")
        assert " e3 " in play.get_fen()
        load = zeropawn_cpp.ChessGame(play.get_fen())
        assert play.get_hash() == load.get_hash()
        # The EP capture itself is generated and legal on both paths.
        assert "d4e3" in play.legal_moves_uci()
        assert "d4e3" in load.legal_moves_uci()

    def test_ep_without_capturer_cleared_on_both_paths(self):
        # 1.e4 from the start position: no black pawn can capture e3.
        play = zeropawn_cpp.ChessGame()
        assert play.make_move_uci("e2e4")
        assert " e3 " not in play.get_fen()
        load = zeropawn_cpp.ChessGame(play.get_fen())
        assert play.get_hash() == load.get_hash()

    def test_diagonally_pinned_capturer_cleared_on_both_paths(self):
        # Black d4 pawn is pinned by the h8 bishop against the a1 king along
        # the a1-h8 diagonal; exd3 e.p. leaves that diagonal entirely (the
        # capturer would land on e3), so the capture is illegal. Note a
        # plain rook pin along the d-file does NOT suppress the capture:
        # the pawn lands on d3 and still blocks the ray.
        start = "7B/8/8/8/3p4/8/4P3/k6K w - - 0 1"
        play = zeropawn_cpp.ChessGame(start)
        assert play.make_move_uci("e2e4")
        after = play.get_fen()
        assert " e3 " not in after
        load = zeropawn_cpp.ChessGame(after)
        assert play.get_hash() == load.get_hash()
        # Cross-check the EP-field convention against the oracle.
        oracle_after = chess.Board(after)
        assert oracle_after.ep_square is None
        assert "d4e3" not in [m.uci() for m in oracle_after.legal_moves]
        assert "d4e3" not in play.legal_moves_uci()

    def test_file_pinned_capturer_is_not_suppressed(self):
        # The rook pin along the d-file does not make exd3 e.p. illegal
        # (the capturing pawn still lands on the d-file at d3): the EP
        # square must be recorded and the capture generated, on both paths.
        play = zeropawn_cpp.ChessGame("3r2k1/8/8/8/3p4/8/4P3/3K4 w - - 0 1")
        assert play.make_move_uci("e2e4")
        assert " e3 " in play.get_fen()
        assert "d4e3" in play.legal_moves_uci()
        load = zeropawn_cpp.ChessGame(play.get_fen())
        assert play.get_hash() == load.get_hash()
        assert "d4e3" in load.legal_moves_uci()

    def test_fen_serialization_is_idempotent(self):
        fens = [
            STARTPOS,
            CR01_WHITE_AFTER,
            "3k4/8/8/K2pP2q/8/8/8/8 w - - 0 1",
            "4k3/8/8/8/3pP3/8/8/4K3 b - e3 0 1",
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
            "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
        ]
        for fen in fens:
            g = zeropawn_cpp.ChessGame(fen)
            once = g.get_fen()
            assert once == fen, f"first serialization changed {fen!r} -> {once!r}"
            twice = zeropawn_cpp.ChessGame(once).get_fen()
            assert twice == once, f"serialization not idempotent for {once!r}"

    def test_legal_ep_state_remains_distinguishable(self):
        with_ep = zeropawn_cpp.ChessGame("4k3/8/8/8/3pP3/8/8/4K3 b - e3 0 1")
        without_ep = zeropawn_cpp.ChessGame("4k3/8/8/8/3pP3/8/8/4K3 b - - 0 1")
        assert with_ep.get_hash() != without_ep.get_hash()
        assert with_ep.legal_moves_uci() != without_ep.legal_moves_uci()

    def test_repetition_semantics_survive_fen_boundary(self):
        # The repro position reached by play and by FEN load must report the
        # same repetition state, since they now share one identity.
        play = zeropawn_cpp.ChessGame(CR01_WHITE)
        play.make_move_uci("d2d4")
        load = zeropawn_cpp.ChessGame(play.get_fen())
        assert play.is_repetition(2) == load.is_repetition(2)

    def test_deterministic_seeded_walk_identity(self):
        # Randomized seeded walks: after every move, loading the played FEN
        # must reproduce the exact hash (the audit's walk-level CR-01 check).
        starts = [
            STARTPOS,
            "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
            "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
        ]
        seeds = [12345, 67890, 24680]
        for start_fen, seed in zip(starts, seeds):
            game = zeropawn_cpp.ChessGame(start_fen)
            rng_state = seed
            for _ in range(120):
                moves = game.legal_moves_uci()
                if not moves or game.is_game_over():
                    break
                rng_state = (rng_state * 6364136223846793005 + 1442695040888963407) % (1 << 64)
                game.make_move_uci(moves[rng_state % len(moves)])
                reloaded = zeropawn_cpp.ChessGame(game.get_fen())
                assert reloaded.get_hash() == game.get_hash(), (
                    f"hash diverged across FEN boundary after {game.get_fen()}"
                )


# ---------------------------------------------------------------------------
# CR-02 — FEN rank overflow / malformed board layouts
# ---------------------------------------------------------------------------

# NOTE on pinning: the entries below that pin NEW rejection behavior (i.e.
# were ACCEPTED-and-transformed or wrapped on the pre-fix build, proven by a
# pre-fix probe) are the rank-overflow / short-rank / 9-rank / 0-9-digit /
# trailing-separator / trailing-garbage-field / king-count entries. The
# empty-rank ("//"), invalid-piece-char, consecutive-separator and
# bad-side-token entries were already rejected pre-fix by other mechanisms
# and are retained as defense-in-depth against regressions of those paths.
BAD_PLACEMENT_FENS = [
    "4k3/8/8/8/8/8/8/qqqqqqqqK w - - 0 1",  # the audit's rank-overflow repro
    "4k3/8/8/8/8/8/8/9q w - - 0 1",  # compression digit 9
    "4k3/8/8/8/8/8/8/0q w - - 0 1",  # compression digit 0
    "4k3/8/8/8/8/8/8/4K2 w - - 0 1",  # final rank too short
    "4k3/8/8/8/8/8/4K/8 w - - 0 1",  # mid rank too short
    "4k3/8/8/8/8/8/8/44K w - - 0 1",  # expanded width 9
    "4k3/8/8/8/8/8/8/4K3/8 w - - 0 1",  # 9 ranks
    "4k3/8/8/8/8/8/8 w - - 0 1",  # 7 ranks
    "4k3/8/8/8/8/8/8//4K3 w - - 0 1",  # empty rank
    "4k3/8/8/8/8/8/8/4K3/ w - - 0 1",  # trailing separator
    "4k3/8/8/8/8/8/8/4K3X w - - 0 1",  # invalid piece char
    "4k3/8/8/8/8/8/8/4K3 w - - 0 1 extra",  # trailing garbage field
    "4k3  8/8/8/8/8/8/8/4K3 w - - 0 1",  # consecutive separators
    "4k3/8/8/8/8/8/8/4K3 QQQQ - 0 1",  # invalid side token
    # REM-CR-08: king-less / multi-king placements. The vendored setFen
    # dereferences a missing king (undefined lsb) while initializing its
    # castling-path table when the king-less side has castling rights --
    # flaky segfault on the pre-remediation build. Must be rejected
    # pre-parse. First entry is the fuzz killer.
    "r3K2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
    "4k3/8/8/8/8/8/8/8 w KQkq - 0 1",
    "8/8/8/8/8/8/8/4K3 w - - 0 1",
    "4k3/8/8/8/8/8/8/K3K3 w - - 0 1",
    "8/8/8/8/8/8/8/8 w - - 0 1",
]


class TestCRFenRankOverflow:
    @pytest.mark.parametrize("fen", BAD_PLACEMENT_FENS)
    def test_malformed_fen_rejected_by_constructor(self, fen):
        with pytest.raises(ValueError):
            zeropawn_cpp.ChessGame(fen)

    @pytest.mark.parametrize("fen", BAD_PLACEMENT_FENS)
    def test_malformed_fen_preserves_state(self, fen):
        game = zeropawn_cpp.ChessGame(STARTPOS)
        game.make_move_uci("e2e4")
        pre_fen, pre_hash = game.get_fen(), game.get_hash()
        assert not game.set_fen(fen)
        assert game.get_fen() == pre_fen
        assert game.get_hash() == pre_hash
        assert game.history_len() == 1

    @pytest.mark.parametrize(
        "fen",
        [
            STARTPOS,
            "4k3/8/8/8/8/8/44/4K3 w - - 0 1",  # valid 44 compression
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
            "2K2r2/4P3/8/8/8/8/8/3k4 w - - 0 1",  # promotion-adjacent position
            "8/8/8/8/k2Pp2Q/8/8/3K4 b - - 0 1",
        ],
    )
    def test_valid_fen_round_trips(self, fen):
        game = zeropawn_cpp.ChessGame(fen)
        # Canonical inputs serialize byte-exact; non-canonical but valid
        # compressions (44) serialize canonically. Either way the reloaded
        # board must be identical.
        once = game.get_fen()
        twice = zeropawn_cpp.ChessGame(once).get_fen()
        assert once == twice
        assert zeropawn_cpp.ChessGame(once).get_hash() == game.get_hash()
        if fen.count("/") == 7 and "44" not in fen:
            assert once == fen


# ---------------------------------------------------------------------------
# CR-03 — FEN counter validation
# ---------------------------------------------------------------------------

BAD_COUNTER_FENS = [
    "4k3/8/8/8/8/8/8/4K3 w - - -5 1",
    "4k3/8/8/8/8/8/8/4K3 b - - 0 -3",
    "4k3/8/8/8/8/8/8/4K3 w - - 256 1",
    "4k3/8/8/8/8/8/8/4K3 w - - 0 32769",
    "4k3/8/8/8/8/8/8/4K3 w - - 1x 1",
    "4k3/8/8/8/8/8/8/4K3 w - - x1 1",
    "4k3/8/8/8/8/8/8/4K3 w - - +5 1",
    "4k3/8/8/8/8/8/8/4K3 w - -  1",  # empty halfmove field
    "4k3/8/8/8/8/8/8/4K3 w - - 0 0",  # fullmove below 1
    "4k3/8/8/8/8/8/8/4K3 w - - 5.5 1",
]


class TestCRFenCounters:
    @pytest.mark.parametrize("fen", BAD_COUNTER_FENS)
    def test_invalid_counter_rejected(self, fen):
        with pytest.raises(ValueError):
            zeropawn_cpp.ChessGame(fen)

    def test_invalid_counter_preserves_state(self):
        game = zeropawn_cpp.ChessGame(STARTPOS)
        pre_fen, pre_hash = game.get_fen(), game.get_hash()
        assert not game.set_fen("4k3/8/8/8/8/8/8/4K3 w - - -5 1")
        assert game.get_fen() == pre_fen and game.get_hash() == pre_hash
        # No false 50-move-draw state can be smuggled in.
        assert not game.is_50_move_rule()

    @pytest.mark.parametrize(
        "halfmove,fullmove",
        [(0, 1), (100, 1), (255, 1), (0, 32768), (255, 32768)],
    )
    def test_counter_boundaries_accepted(self, halfmove, fullmove):
        fen = f"4k3/8/8/8/8/8/8/4K3 w - - {halfmove} {fullmove}"
        game = zeropawn_cpp.ChessGame(fen)
        assert game.half_move_clock() == halfmove
        assert game.full_move_number() == fullmove
        assert game.get_fen() == fen


# ---------------------------------------------------------------------------
# CR-04 — insufficient material (python-chess-equivalent classifier)
# ---------------------------------------------------------------------------

MATERIAL_CONFIG_FENS = [
    # name, fen, expected
    ("K vs K", "8/8/8/8/8/8/8/K6k w - - 0 1", True),
    ("KN vs K", "8/8/8/8/1N6/8/8/K6k w - - 0 1", True),
    ("KB vs K", "8/8/8/8/1B6/8/8/K6k w - - 0 1", True),
    ("KNN vs K (helpmate exists)", "8/8/8/8/1NN5/8/8/K6k w - - 0 1", False),
    ("KN vs KN", "8/8/8/8/1Nn5/8/8/K6k w - - 0 1", False),
    ("KB vs KB same color (both light)", "8/8/8/8/2b5/8/8/K4B1k w - - 0 1", True),
    ("KB vs KB opposite", "8/8/8/8/1Bb5/8/8/K6k w - - 0 1", False),
    ("KBB-same vs K (both light)", "7k/8/8/8/8/8/8/K4B1B w - - 0 1", True),
    ("KBB-opposite vs K (f1/g1)", "7k/8/8/8/8/8/8/K4BB1 w - - 0 1", False),
    ("CR-04: KBB-light vs KB-light", "8/8/8/4k3/2b5/8/8/4KB1B w - - 0 1", True),
    ("KBB-mixed vs KBB-mixed", "8/8/8/4k3/2b5/8/8/2b1KB1B w - - 0 1", False),
    ("KBB vs KN", "8/8/8/4k3/2n5/8/8/4KB1B w - - 0 1", False),
    ("KN vs KQ", "8/8/8/4k3/2q5/8/8/4K1N1 w - - 0 1", False),
    ("KN vs KR", "8/8/8/4k3/2r5/8/8/4K1N1 w - - 0 1", False),
    ("KP vs K", "8/8/8/4k3/8/8/4P3/4K3 w - - 0 1", False),
    ("KR vs K", "8/8/8/4k3/8/8/8/4KR2 w - - 0 1", False),
    ("KQ vs K", "8/8/8/4k3/8/8/8/4KQ2 w - - 0 1", False),
    ("KBN vs K", "8/8/8/4k3/8/8/8/4KBN1 w - - 0 1", False),
]


class TestCRInsufficientMaterial:
    @pytest.mark.parametrize("name,fen,expected", MATERIAL_CONFIG_FENS)
    def test_named_configurations(self, name, fen, expected):
        game = zeropawn_cpp.ChessGame(fen)
        assert game.is_insufficient_material() == expected, name

    @pytest.mark.parametrize("name,fen,expected", MATERIAL_CONFIG_FENS)
    def test_named_configurations_match_oracle(self, name, fen, expected):
        oracle = chess.Board(fen).is_insufficient_material()
        assert oracle == expected, f"oracle disagrees on {name}"
        assert zeropawn_cpp.ChessGame(fen).is_insufficient_material() == oracle, name

    def test_previously_verified_small_configs_unchanged(self):
        # The audit's 15 verified configurations (audit report D §6): the
        # remediation must not flip any of these verdicts.
        prior = {
            "8/8/8/4k3/8/8/8/4K3 w - - 0 1": True,
            "8/8/8/4k3/8/8/8/4K1N1 w - - 0 1": True,
            "8/8/8/4k3/8/8/8/4KB2 w - - 0 1": True,
            "8/8/8/4k3/8/8/8/4KN2 w - - 0 1": True,
            "8/8/8/4k3/2b5/8/8/4K1B1 w - - 0 1": False,
            "8/8/8/4k3/2b5/8/8/4KB2 w - - 0 1": True,
            "8/8/8/4k3/8/8/8/4KBB1 w - - 0 1": False,
            "8/8/8/4k3/2b5/8/8/4KBB1 w - - 0 1": False,
            "8/8/8/4k3/2n5/8/8/4KBB1 w - - 0 1": False,
            "8/8/8/4k3/2b5/8/8/4KBB1 b - - 0 1": False,
            "8/8/8/4k3/2n5/8/8/4K1N1 w - - 0 1": False,
            "8/8/8/4k3/8/8/4P3/4K3 w - - 0 1": False,
            "8/8/8/4k3/8/8/8/4KR2 w - - 0 1": False,
            "8/8/8/4k3/8/8/8/4KBN1 w - - 0 1": False,
            "8/8/1b6/4k3/8/8/8/4KB2 w - - 0 1": False,
        }
        for fen, expected in prior.items():
            assert zeropawn_cpp.ChessGame(fen).is_insufficient_material() == expected, fen
            oracle = chess.Board(fen).is_insufficient_material()
            assert oracle == expected, f"oracle disagrees on prior config {fen}"

    def test_dead_position_ends_game(self):
        game = zeropawn_cpp.ChessGame("8/8/8/4k3/2b5/8/8/4KB1B w - - 0 1")
        assert game.is_insufficient_material()
        assert game.is_game_over()

    def test_playable_position_not_terminated(self):
        # A false positive here would end playable games: KBB vs KN can
        # still produce mate (the knight side may selfmate-contribute).
        game = zeropawn_cpp.ChessGame("8/8/8/4k3/2n5/8/8/4KB1B w - - 0 1")
        assert not game.is_insufficient_material()
        assert not game.is_game_over()


# ---------------------------------------------------------------------------
# CR-05 — FEN normalization policy (intentional leniency, locked down)
# ---------------------------------------------------------------------------


class TestCRFenNormalization:
    def test_garbage_castling_characters_ignored(self):
        game = zeropawn_cpp.ChessGame("4k3/8/8/8/8/8/8/4K2R w KQkqXZ - 0 1")
        assert game.get_fen().split()[2] == "K"

    def test_impossible_castling_rights_dropped(self):
        # White rook is not on h1: the K right cannot be honored.
        game = zeropawn_cpp.ChessGame("4k3/8/8/8/8/8/8/3K4 w KQkq - 0 1")
        assert game.get_fen().split()[2] == "-"

    def test_reordered_castling_letters_parsed(self):
        # Reordered letters are all honored; a garbage letter is ignored.
        game = zeropawn_cpp.ChessGame("r3k2r/8/8/8/8/8/8/R3K2R w qkQKr - 0 1")
        assert sorted(game.get_fen().split()[2]) == ["K", "Q", "k", "q"]

    def test_wrong_rank_ep_cleared(self):
        # EP square e6 with black to move is impossible (rank 6 requires
        # white to move); it is silently cleared, not rejected.
        game = zeropawn_cpp.ChessGame("4k3/8/8/8/8/8/8/4K3 w - e6 0 1")
        assert game.get_fen().split()[3] == "-"

    def test_ep_without_capturer_cleared(self):
        game = zeropawn_cpp.ChessGame("4k3/8/8/8/4P3/8/8/4K3 b - e3 0 1")
        assert game.get_fen().split()[3] == "-"

    def test_missing_fields_defaulted(self):
        game = zeropawn_cpp.ChessGame("4k3/8/8/8/8/8/8/4K2R w K")
        parts = game.get_fen().split()
        assert parts[3:] == ["-", "0", "1"]

    def test_back_rank_pawns_accepted(self):
        # Pawns on their own back ranks generate no moves; accepted as a
        # safe normalization, never a crash.
        game = zeropawn_cpp.ChessGame("P3k3/8/8/8/8/8/8/4K2p w - - 0 1")
        # Loads without error; the back-rank pawns generate no moves (only
        # the white king can move).
        assert all(m[:2] == "e1" for m in game.legal_moves_uci())

    def test_syntax_invalid_ep_still_rejected(self):
        with pytest.raises(ValueError):
            zeropawn_cpp.ChessGame("4k3/8/8/8/8/8/8/4K2R w KQkq e9 0 1")

    def test_normalization_deterministic(self):
        raw = "4k3/8/8/8/8/8/8/3K4 w KQkqXZ e6 7 3"
        first = zeropawn_cpp.ChessGame(raw).get_fen()
        second = zeropawn_cpp.ChessGame(first).get_fen()
        assert first == second  # normalized output is stable


# ---------------------------------------------------------------------------
# CR-06 — engine terminal conventions (intentional, locked down)
# ---------------------------------------------------------------------------


class TestCRTerminalConventions:
    def test_mate_takes_priority_at_100_halfmoves(self):
        # hfm reaches 100 on the mating move: checkmate wins, the 50-move
        # auto-draw does not fire first.
        game = zeropawn_cpp.ChessGame("k7/7R/1K6/8/8/8/8/8 w - - 99 100")
        assert game.make_move_uci("h7h8")
        assert game.half_move_clock() == 100
        assert game.is_checkmate()
        assert game.is_game_over()
        assert game.terminal_value() == pytest.approx(-1.0)

    def test_auto_draw_at_100_halfmoves_without_mate(self):
        game = zeropawn_cpp.ChessGame("k7/7R/1K6/8/8/8/8/8 w - - 100 100")
        assert game.is_50_move_rule()
        assert game.is_game_over()
        assert not game.is_checkmate()
        assert game.terminal_value() == pytest.approx(0.0)

    def test_threefold_fires_at_third_occurrence(self):
        game = zeropawn_cpp.ChessGame()
        for uci in ["g1f3", "g8f6", "f3g1", "f6g8"]:
            assert game.make_move_uci(uci)
        assert not game.is_repetition(2)  # 2nd occurrence of startpos
        assert not game.is_game_over()
        for uci in ["g1f3", "g8f6", "f3g1", "f6g8"]:
            assert game.make_move_uci(uci)
        # After the 8th halfmove startpos occurs the 3rd time -> auto-draw.
        assert game.is_repetition(2)
        assert game.is_game_over()

    def test_fivefold_subsumed_by_threefold(self):
        # The engine never reaches a fivefold repetition: the game already
        # ended automatically at the third occurrence (documented policy).
        game = zeropawn_cpp.ChessGame()
        for uci in ["g1f3", "g8f6", "f3g1", "f6g8", "g1f3", "g8f6", "f3g1", "f6g8"]:
            game.make_move_uci(uci)
        # The third occurrence (8 halfmoves) already auto-ended the game;
        # a fivefold count is unreachable.
        assert game.is_game_over()  # threefold already fired

    def test_75_move_rule_subsumed_by_100(self):
        game = zeropawn_cpp.ChessGame("k7/7R/1K6/8/8/8/8/8 w - - 150 200")
        assert game.is_50_move_rule()
        assert game.is_game_over()

    def test_stalemate_still_detected(self):
        game = zeropawn_cpp.ChessGame("k7/8/1Q6/8/8/8/8/K7 b - - 0 1")
        assert game.is_stalemate()
        assert game.is_game_over()
        assert game.terminal_value() == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# CR-07 — UCI opening-book ply accounting
# ---------------------------------------------------------------------------


class BookProbe:
    """Opening-book stand-in recording the ply it was asked about."""

    def __init__(self):
        self.enabled = True
        self.recorded = []

    def lookup(self, game, current_ply=None):
        self.recorded.append(current_ply)
        return None


class TestCRUciBookPly:
    def _engine(self):
        return UCIEngine(use_book=True)

    def test_startpos_ply_accounting(self):
        engine = self._engine()
        probe = BookProbe()
        engine.opening_book = probe
        engine.handle_line("position startpos moves e2e4 e7e5")
        engine.search_and_respond(15)
        assert probe.recorded == [2]

    def test_startpos_ply_accounting_no_moves(self):
        engine = self._engine()
        probe = BookProbe()
        engine.opening_book = probe
        engine.handle_line("position startpos")
        engine.search_and_respond(15)
        assert probe.recorded == [0]

    def test_fen_loaded_white_ply(self):
        engine = self._engine()
        probe = BookProbe()
        engine.opening_book = probe
        engine.handle_line("position fen rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 30")
        engine.search_and_respond(15)
        assert probe.recorded == [58]

    def test_fen_loaded_black_ply(self):
        engine = self._engine()
        probe = BookProbe()
        engine.opening_book = probe
        engine.handle_line("position fen rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1")
        engine.search_and_respond(15)
        assert probe.recorded == [1]

    def test_fen_then_replayed_moves_increment(self):
        engine = self._engine()
        probe = BookProbe()
        engine.opening_book = probe
        engine.handle_line(
            "position fen rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 10 moves e2e4 e7e5"
        )
        engine.search_and_respond(15)
        assert probe.recorded == [18 + 2]

    def test_ply_matches_python_chess_convention(self):
        engine = self._engine()
        probe = BookProbe()
        engine.opening_book = probe
        fens = [
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1",
            "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
        ]
        for fen in fens:
            engine.handle_line(f"position fen {fen}")
            engine.search_and_respond(15)
            assert probe.recorded[-1] == chess.Board(fen).ply(), fen

    def test_book_moves_stay_legality_gated(self):
        # Even with a book returning a nonsense move, the engine must not
        # play it (legality gate preserved).
        class IllegalBook(BookProbe):
            def lookup(self, game, current_ply=None):
                self.recorded.append(current_ply)
                return ("e1e8", {})  # kings cannot move like this

        engine = self._engine()
        engine.opening_book = IllegalBook()
        engine.handle_line("position startpos")
        engine.search_and_respond(15)
        assert engine.last_engine_move != "e1e8"

    def test_ucinewgame_resets_book_ply(self):
        engine = self._engine()
        probe = BookProbe()
        engine.opening_book = probe
        engine.handle_line("position fen rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 30")
        engine.handle_line("ucinewgame")
        engine.handle_line("position startpos")
        engine.search_and_respond(15)
        assert probe.recorded == [0]


class TestCREnPassantActionPath:
    def test_apply_action_path_shares_identity_policy(self):
        # The search hot path (apply_action, action indices) must record EP
        # under the same EXACT policy as make_move_uci.
        play = zeropawn_cpp.ChessGame(CR01_WHITE)
        action = play.string_to_action("d2d4")
        assert action >= 0
        play.apply_action(action)
        load = zeropawn_cpp.ChessGame(play.get_fen())
        assert play.get_hash() == load.get_hash()
        assert play.get_fen() == load.get_fen() == CR01_WHITE_AFTER
