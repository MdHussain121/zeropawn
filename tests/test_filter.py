import io
import pytest
from python.data.filter import (
    FilterStats,
    parse_time_control,
    is_rated,
    is_rating_above_threshold,
    is_normal_termination,
    check_time_control,
    filter_games,
)

SAMPLE_PGN = """
[Event "Rated Blitz game"]
[Site "https://lichess.org/valid01"]
[WhiteElo "2300"]
[BlackElo "2250"]
[Termination "Normal"]
[TimeControl "180+0"]

1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O Be7 6. Re1 b5 7. Bb3 d6 8. c3 O-O 9. h3 Nb8 10. d4 Nbd7 1-0

[Event "Casual Blitz game"]
[Site "https://lichess.org/unrated"]
[WhiteElo "2300"]
[BlackElo "2300"]
[Termination "Normal"]
[TimeControl "180+0"]

1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O Be7 6. Re1 b5 7. Bb3 d6 8. c3 O-O 9. h3 Nb8 10. d4 Nbd7 1-0

[Event "Rated Blitz game"]
[Site "https://lichess.org/lowelo"]
[WhiteElo "2199"]
[BlackElo "2300"]
[Termination "Normal"]
[TimeControl "180+0"]

1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O Be7 6. Re1 b5 7. Bb3 d6 8. c3 O-O 9. h3 Nb8 10. d4 Nbd7 1-0

[Event "Rated Blitz game"]
[Site "https://lichess.org/timeforfeit"]
[WhiteElo "2400"]
[BlackElo "2400"]
[Termination "Time forfeit"]
[TimeControl "180+0"]

1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O Be7 6. Re1 b5 7. Bb3 d6 8. c3 O-O 9. h3 Nb8 10. d4 Nbd7 1-0

[Event "Rated Bullet game"]
[Site "https://lichess.org/bullet"]
[WhiteElo "2400"]
[BlackElo "2400"]
[Termination "Normal"]
[TimeControl "60+0"]

1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O Be7 6. Re1 b5 7. Bb3 d6 8. c3 O-O 9. h3 Nb8 10. d4 Nbd7 1-0

[Event "Rated Blitz game"]
[Site "https://lichess.org/shortgame"]
[WhiteElo "2400"]
[BlackElo "2400"]
[Termination "Normal"]
[TimeControl "180+0"]

1. e4 e5 2. Qh5 Nc6 3. Bc4 Nf6 4. Qxf7# 1-0
""".strip()


def test_parse_time_control():
    assert parse_time_control("180+0") == 180
    assert parse_time_control("180+2") == 260
    assert parse_time_control("60+0") == 60
    assert parse_time_control("300+3") == 420
    assert parse_time_control("-") is None
    assert parse_time_control("corresp") is None


def test_header_predicates():
    assert is_rated({"Event": "Rated Blitz game"}) is True
    assert is_rated({"Event": "Casual Blitz game"}) is False

    assert is_rating_above_threshold({"WhiteElo": "2200", "BlackElo": "2200"}, 2200) is True
    assert is_rating_above_threshold({"WhiteElo": "2199", "BlackElo": "2400"}, 2200) is False

    assert is_normal_termination({"Termination": "Normal"}) is True
    assert is_normal_termination({"Termination": "Time forfeit"}) is False
    assert is_normal_termination({"Termination": "Abandoned"}) is False

    assert check_time_control({"TimeControl": "180+0"}, min_seconds=180) is True
    assert check_time_control({"TimeControl": "60+0"}, min_seconds=180) is False
    assert check_time_control({"TimeControl": "60+3"}, min_seconds=180) is True  # 60 + 120 = 180


def test_filter_pipeline_sample():
    stream = io.StringIO(SAMPLE_PGN)
    stats = FilterStats()
    kept = list(filter_games(stream, stats, min_rating=2200, min_plies=20, min_tc_seconds=180))

    assert len(kept) == 1
    assert stats.total_seen == 6
    assert stats.passed_rated == 5  # dropped unrated
    assert stats.passed_rating_2200 == 4  # dropped low Elo
    assert stats.passed_normal_term == 3  # dropped Time forfeit
    assert stats.passed_time_control == 2  # dropped Bullet
    assert stats.passed_min_plies == 1  # dropped 7-ply mate
    assert stats.total_kept == 1
    assert kept[0].headers["Site"] == "https://lichess.org/valid01"
