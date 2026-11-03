"""S5-05 remediation tests: the SPRT promotion gate must NOT score 150-ply
capped games as draws (cap != draw — the MatchRunner F-01 contract, ported).

Covers:
- Shared classification contract: ``scripts/run_sprt.py`` and
  ``python/eval/match_runner.py`` classify game records through ONE
  implementation, cross-checked behaviorally over the full result/color
  cross product (byte-identical by import, proven equal by behavior).
- Stubbed ``SprtMatchRunner.play_single_game`` endings (NO real searches:
  ``zeropawn_cpp.MCTS`` is replaced by a scripted fake): checkmate,
  stalemate, 50-move-countdown rule draw, ply cap under both cap policies,
  value-cutoff adjudications, illegal-move forfeit.
- SPRT tally/summary accounting on synthetic records: capped games excluded
  from the primary W/D/L and LLR; per-color capped counts; uncounted
  (incomplete/invalid) records reported but never scored; capped-as-draw /
  capped-as-loss / capped-by-material sensitivity trio.
- Resume: legacy (pre-S5-05) state records default to uncapped; new-schema
  records reclassify consistently; mid-run resume with a stubbed game player.

The S5-06 closed-form LLR pins live in ``tests/test_sprt.py``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import zeropawn_cpp
import python.eval.match_runner as match_runner_module
from python.eval.match_runner import (
    classify_game_record,
    classify_game_records,
    engine_outcome,
)
from python.eval.sprt import compute_sprt_llr
import scripts.run_sprt as run_sprt_module
from scripts.run_sprt import (
    SprtMatchRunner,
    capped_sensitivity,
    classify_sprt_record,
    sprt_tally,
)

START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
# White is checkmated (Fool's mate pattern).
WHITE_MATED_FEN = "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 0 1"
# Black is checkmated.
BLACK_MATED_FEN = "r1bqkbnr/pppp1Qpp/2n5/4p3/2B1P3/8/PPPP1PPP/RNB1K1NR b KQkq - 0 1"
# Black to move is stalemated.
STALEMATE_FEN = "7k/5Q2/6K1/8/8/8/8/8 b - - 0 1"
# Halfmove clock at 90 (>= 60 triggers the harness countdown; < 100 is not a
# rule termination yet).
FIFTY_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 90 60"
# After 1.e4: black to move, so "e2e4" is an illegal reply.
AFTER_E4_FEN = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"

CAND_CFG = {"c_puct": 2.0, "fpu_reduction": 0.2}
BASE_CFG = {"c_puct": 2.0, "fpu_reduction": 0.2}


# ---------------------------------------------------------------------------
# Shared classification contract (one implementation, both harnesses)
# ---------------------------------------------------------------------------


def test_shared_classifier_is_a_single_implementation():
    """run_sprt imports MatchRunner's classifier — no local copy exists."""
    assert run_sprt_module.classify_game_record is match_runner_module.classify_game_record
    assert run_sprt_module.classify_game_records is match_runner_module.classify_game_records
    assert run_sprt_module.is_capped_game_record is match_runner_module.is_capped_game_record


_FINE_TO_ENGINE_OUTCOME = {
    "decisive_win": "win",
    "forfeit_win": "win",
    "decisive_loss": "loss",
    "forfeit_loss": "loss",
    "genuine_draw": "draw",
    "capped": "capped",
}


@pytest.mark.parametrize("cand_white", [True, False])
@pytest.mark.parametrize("result", ["1-0", "0-1", "1/2-1/2", "*", "capped", "inconclusive"])
def test_classifier_matches_engine_outcome_for_every_result(result, cand_white):
    """Cross-check: the shared record classifier and MatchRunner's
    ``engine_outcome`` agree on win/draw/loss/capped for EVERY board result
    string and candidate color."""
    fine = classify_game_record({"board_result": result, "cand_white": cand_white})
    assert fine in _FINE_TO_ENGINE_OUTCOME, f"unexpected fine class {fine!r}"
    assert _FINE_TO_ENGINE_OUTCOME[fine] == engine_outcome(result, cand_white)


def test_classifier_accepts_both_record_vocabularies():
    """MatchRunner-style (result/engine_is_white) and SPRT-style
    (board_result/cand_white) records classify identically."""
    a = classify_game_record({"result": "1-0", "engine_is_white": False})
    b = classify_game_record({"board_result": "1-0", "cand_white": False})
    assert a == b == "decisive_loss"
    a = classify_game_record({"result": "*", "engine_is_white": True})
    b = classify_game_record({"board_result": "*", "cand_white": True})
    assert a == b == "capped"


def test_classify_game_record_outcome_table():
    assert classify_game_record({"board_result": "1-0", "cand_white": True}) == "decisive_win"
    assert classify_game_record({"board_result": "1-0", "cand_white": False}) == "decisive_loss"
    assert classify_game_record({"board_result": "0-1", "cand_white": True}) == "decisive_loss"
    assert classify_game_record({"board_result": "0-1", "cand_white": False}) == "decisive_win"
    assert classify_game_record({"board_result": "1/2-1/2", "cand_white": True}) == "genuine_draw"
    # Ply cap: "*" and the cap_policy="draw" board string with capped
    # provenance BOTH classify as capped — never as a genuine draw (S5-05).
    assert classify_game_record({"board_result": "*", "cand_white": True}) == "capped"
    assert (
        classify_game_record(
            {"board_result": "1/2-1/2", "cand_white": True, "outcome_class": "capped"}
        )
        == "capped"
    )
    assert (
        classify_game_record(
            {
                "board_result": "1/2-1/2",
                "cand_white": True,
                "termination_reason": "max_plies_cap",
            }
        )
        == "capped"
    )
    # Forfeits (illegal move / timeout) are wins/losses, tracked separately.
    assert (
        classify_game_record(
            {"board_result": "0-1", "cand_white": True, "outcome_class": "forfeit"}
        )
        == "forfeit_loss"
    )
    assert (
        classify_game_record(
            {"board_result": "0-1", "cand_white": False, "outcome_class": "forfeit"}
        )
        == "forfeit_win"
    )
    # Incomplete and invalid records are reported, never scored.
    assert classify_game_record({"cand_white": True}) == "uncounted"
    assert classify_game_record({"board_result": None, "cand_white": True}) == "uncounted"
    assert classify_game_record({"board_result": "2-1", "cand_white": True}) == "uncounted"
    assert classify_game_record({"board_result": "1-0"}) == "uncounted"  # no color


def test_classify_game_records_parity_with_match_runner_semantics():
    """The extracted classifier reproduces MatchRunner.run_match's historical
    inline tally exactly (incl. engine-as-Black forfeit wins)."""
    games = [
        {"result": "1-0", "engine_is_white": True, "outcome_class": "decisive"},
        {"result": "0-1", "engine_is_white": False, "outcome_class": "forfeit"},
        {
            "result": "1/2-1/2",
            "engine_is_white": True,
            "outcome_class": "genuine_draw",
            "termination_reason": "rule_draw",
        },
        {"result": "1/2-1/2", "engine_is_white": False, "termination_reason": "max_plies_cap"},
        {"result": "*", "engine_is_white": True},
        {"result": "0-1", "engine_is_white": True, "outcome_class": "forfeit"},
    ]
    assert classify_game_records(games) == {
        "decisive_wins": 1,
        "decisive_losses": 0,
        "genuine_draws": 1,
        "capped_games": 2,
        "forfeit_wins": 1,
        "forfeit_losses": 1,
        "uncounted_games": 0,
    }


def test_same_records_same_classification_in_both_harnesses():
    """The SAME synthetic records produce the same W/D/L/capped split whether
    tallied through MatchRunner's classifier or run_sprt's sprt_tally."""
    records = _synthetic_records()
    counts = classify_game_records(records)
    tally = sprt_tally(records)
    assert tally["wins"] == counts["decisive_wins"] + counts["forfeit_wins"]
    assert tally["draws"] == counts["genuine_draws"]
    assert tally["losses"] == counts["decisive_losses"] + counts["forfeit_losses"]
    assert tally["capped_games"] == counts["capped_games"]
    assert tally["uncounted_games"] == counts["uncounted_games"]


# ---------------------------------------------------------------------------
# Stubbed play_single_game endings (scripted MCTS — no real searches)
# ---------------------------------------------------------------------------


class _FakeSearchResult:
    def __init__(self, best_action: int, root_value: float) -> None:
        self.best_action = best_action
        self.root_value = root_value


def _fake_mcts_factory(root_value: float, action_fn=None):
    class _FakeMCTS:
        """Scripted stand-in for zeropawn_cpp.MCTS (never searches)."""

        def __init__(self, cfg) -> None:
            self.cfg = cfg

        def search(self, game, sims, eval_fn):
            assert callable(eval_fn)
            if action_fn is not None:
                best = action_fn(game)
            else:
                best = game.legal_actions()[0]
            return _FakeSearchResult(best, root_value)

    return _FakeMCTS


def _bare_runner(cap_policy: str = "exclude") -> SprtMatchRunner:
    runner = SprtMatchRunner.__new__(SprtMatchRunner)
    runner.device = "cpu"
    runner.sims = 1
    runner.workers = 1
    runner.cap_policy = cap_policy
    runner.candidate_checkpoint = "missing_candidate.pt"
    runner.base_checkpoint = "missing_base.pt"
    runner.candidate_model = None
    runner.base_model = None
    runner.openings = [START_FEN]
    return runner


def _play(
    monkeypatch,
    fen: str,
    cand_white: bool = True,
    root_value: float = 0.0,
    max_plies: int = 150,
    action_fn=None,
    cap_policy: str = "exclude",
):
    monkeypatch.setattr(zeropawn_cpp, "MCTS", _fake_mcts_factory(root_value, action_fn))
    runner = _bare_runner(cap_policy=cap_policy)
    info: dict = {}
    result, moves = runner.play_single_game(
        fen, cand_white, dict(CAND_CFG), dict(BASE_CFG), 12345, max_plies, info
    )
    return result, info, moves


def _record(result: str, info: dict, cand_white: bool, moves=None) -> dict:
    """Build the exact per-game record schema run_sprt persists, then classify
    it through the shared contract (mirrors the batch-record construction)."""
    record = {
        "board_result": result,
        "cand_white": cand_white,
        "adjudicated": bool(info.get("adjudicated", False)),
        "adjudication_reason": info.get("adjudication_reason"),
        "termination_reason": info.get("termination_reason"),
        "outcome_class": info.get("outcome_class"),
        "plies": int(info.get("plies", len(moves or []))),
        "final_material_balance": info.get("final_material_balance", 0),
    }
    record["cand_outcome"] = classify_sprt_record(record)
    return record


def test_play_checkmate_is_a_decisive_win(monkeypatch):
    result, info, _ = _play(monkeypatch, BLACK_MATED_FEN, cand_white=True)
    assert result == "1-0"
    assert info["outcome_class"] == "decisive"
    assert info["termination_reason"] == "checkmate"
    assert info["adjudicated"] is False
    rec = _record(result, info, cand_white=True)
    assert rec["cand_outcome"] == "win"


def test_play_being_checkmated_is_a_decisive_loss(monkeypatch):
    # Candidate IS the mated White side: board "0-1" -> candidate loss.
    result, info, _ = _play(monkeypatch, WHITE_MATED_FEN, cand_white=True)
    assert result == "0-1"
    assert info["outcome_class"] == "decisive"
    assert info["termination_reason"] == "checkmate"
    rec = _record(result, info, cand_white=True)
    assert rec["cand_outcome"] == "loss"


def test_play_stalemate_is_a_genuine_draw_not_a_cap(monkeypatch):
    """A natural game-over draw (stalemate) must classify as a genuine draw —
    the pre-S5-05 harness mislabeled this ending 'max_plies_cap'."""
    result, info, _ = _play(monkeypatch, STALEMATE_FEN, cand_white=True)
    assert result == "1/2-1/2"
    assert info["outcome_class"] == "genuine_draw"
    assert info["termination_reason"] == "rule_draw"
    assert info["adjudicated"] is False
    rec = _record(result, info, cand_white=True)
    assert rec["cand_outcome"] == "draw"


def test_play_fifty_move_countdown_is_a_genuine_draw(monkeypatch):
    result, info, _ = _play(monkeypatch, FIFTY_FEN, cand_white=True, root_value=0.0)
    assert result == "1/2-1/2"
    assert info["outcome_class"] == "genuine_draw"
    assert info["termination_reason"] == "fifty_move_countdown"
    assert info["adjudicated"] is True
    assert info["adjudication_reason"] == "fifty_move_countdown"
    rec = _record(result, info, cand_white=True)
    assert rec["cand_outcome"] == "draw"


def test_play_ply_cap_is_capped_and_excluded(monkeypatch):
    """150-ply-cap semantics (exercised at max_plies=6; the cap branch is
    'plies >= max_plies' either way, and 6 plies from the start position
    cannot reach any other terminal): result '*', outcome_class 'capped',
    EXCLUDED from the score (cand_outcome 'capped', not 'draw')."""
    result, info, _ = _play(monkeypatch, START_FEN, cand_white=True, max_plies=6)
    assert result == "*"
    assert info["outcome_class"] == "capped"
    assert info["termination_reason"] == "max_plies_cap"
    assert info["adjudicated"] is True
    assert info["adjudication_reason"] == "max_plies_cap"
    rec = _record(result, info, cand_white=True)
    assert rec["cand_outcome"] == "capped"


def test_play_ply_cap_with_draw_policy_still_classifies_capped(monkeypatch):
    """cap_policy='draw' keeps the legacy '1/2-1/2' board string, but the
    recorded outcome_class='capped' makes the classification (and any
    resumed tally) policy-independent."""
    result, info, _ = _play(monkeypatch, START_FEN, cand_white=True, max_plies=6, cap_policy="draw")
    assert result == "1/2-1/2"
    assert info["outcome_class"] == "capped"
    assert info["termination_reason"] == "max_plies_cap"
    rec = _record(result, info, cand_white=True)
    assert rec["cand_outcome"] == "capped"


def test_play_value_cutoff_loss_adjudication(monkeypatch):
    """Root value -1.0 on every ply: the 4th consecutive hopeless check fires
    with Black to move, so Black loses ('1-0') regardless of candidate color —
    deterministic under the symmetric two-arm harness."""
    result, info, _ = _play(monkeypatch, START_FEN, cand_white=False, root_value=-1.0)
    assert result == "1-0"
    assert info["outcome_class"] == "decisive"
    assert info["termination_reason"] == "value_cutoff_loss"
    assert info["adjudicated"] is True
    rec = _record(result, info, cand_white=False)
    assert rec["cand_outcome"] == "loss"


def test_play_value_cutoff_win_adjudication(monkeypatch):
    result, info, _ = _play(monkeypatch, START_FEN, cand_white=True, root_value=1.0)
    assert result == "0-1"  # winner is Black (to move at the 4th check)
    assert info["outcome_class"] == "decisive"
    assert info["termination_reason"] == "value_cutoff_win"
    assert info["adjudicated"] is True
    rec = _record(result, info, cand_white=True)
    assert rec["cand_outcome"] == "loss"


def test_play_illegal_move_is_a_forfeit_not_a_draw(monkeypatch):
    """The scripted MCTS returns the index of an illegal 'e2e4' reply: the
    mover (Black = candidate) forfeits — counted as a loss, never a draw."""
    illegal = {}

    def action_fn(game):
        illegal["idx"] = zeropawn_cpp.move_to_index(game, "e2e4")
        return illegal["idx"]

    result, info, _ = _play(
        monkeypatch, AFTER_E4_FEN, cand_white=False, root_value=0.0, action_fn=action_fn
    )
    assert result == "1-0"
    assert info["outcome_class"] == "forfeit"
    assert info["termination_reason"] == "forfeit_illegal_move"
    assert info["adjudicated"] is False
    rec = _record(result, info, cand_white=False)
    assert rec["cand_outcome"] == "loss"


def test_play_records_plies_and_material_balance(monkeypatch):
    result, info, moves = _play(monkeypatch, START_FEN, cand_white=True, max_plies=6)
    assert info["plies"] == len(moves) == 6
    assert "final_fen" in info
    assert info["final_material_balance"] == 0  # symmetric start position


def test_sprt_match_runner_rejects_unknown_cap_policy():
    with pytest.raises(ValueError):
        SprtMatchRunner(candidate_checkpoint="a.pt", base_checkpoint="b.pt", cap_policy="banana")


# ---------------------------------------------------------------------------
# Tally / sensitivity / summary accounting on synthetic records
# ---------------------------------------------------------------------------


def _synthetic_records() -> list[dict]:
    return [
        # candidate wins as White (checkmate)
        {
            "board_result": "1-0",
            "cand_white": True,
            "outcome_class": "decisive",
            "termination_reason": "checkmate",
            "final_material_balance": 9,
        },
        # candidate wins as Black (checkmate)
        {
            "board_result": "0-1",
            "cand_white": False,
            "outcome_class": "decisive",
            "termination_reason": "checkmate",
            "final_material_balance": 9,
        },
        # genuine rule draw (stalemate)
        {
            "board_result": "1/2-1/2",
            "cand_white": True,
            "outcome_class": "genuine_draw",
            "termination_reason": "rule_draw",
            "final_material_balance": 0,
        },
        # candidate loss
        {
            "board_result": "0-1",
            "cand_white": True,
            "outcome_class": "decisive",
            "termination_reason": "checkmate",
            "final_material_balance": -9,
        },
        # capped, candidate White, +3 material (by-material -> win)
        {
            "board_result": "*",
            "cand_white": True,
            "outcome_class": "capped",
            "termination_reason": "max_plies_cap",
            "final_material_balance": 3,
        },
        # capped, candidate Black, equal material (by-material -> draw)
        {
            "board_result": "*",
            "cand_white": False,
            "outcome_class": "capped",
            "termination_reason": "max_plies_cap",
            "final_material_balance": 0,
        },
        # forfeit win (opponent played an illegal move)
        {
            "board_result": "1-0",
            "cand_white": True,
            "outcome_class": "forfeit",
            "termination_reason": "forfeit_illegal_move",
        },
        # incomplete record (crashed mid-game, no result)
        {"board_result": None, "cand_white": False},
        # invalid result string
        {"board_result": "2-1", "cand_white": True},
    ]


def test_sprt_tally_excludes_capped_and_counts_per_color():
    counts = sprt_tally(_synthetic_records())
    assert counts["wins"] == 3  # 2 decisive + 1 forfeit
    assert counts["draws"] == 1
    assert counts["losses"] == 1
    assert counts["capped_games"] == 2
    assert counts["capped_candidate_white"] == 1
    assert counts["capped_candidate_black"] == 1
    assert counts["uncounted_games"] == 2
    # The SPRT math input covers ONLY the uncapped, valid games.
    r = compute_sprt_llr(counts["wins"], counts["draws"], counts["losses"])
    assert r["n"] == 5


def test_capped_sensitivity_trio_matches_direct_computation():
    records = _synthetic_records()
    counts = sprt_tally(records)
    sens = capped_sensitivity(records, elo0=0.0, elo1=30.0)
    primary = compute_sprt_llr(counts["wins"], counts["draws"], counts["losses"])
    as_draw = compute_sprt_llr(
        counts["wins"], counts["draws"] + counts["capped_games"], counts["losses"]
    )
    as_loss = compute_sprt_llr(
        counts["wins"], counts["draws"], counts["losses"] + counts["capped_games"]
    )
    # By material: +3-pawn capped game -> win, equal-material capped -> draw.
    by_material = compute_sprt_llr(counts["wins"] + 1, counts["draws"] + 1, counts["losses"])

    assert sens["primary"]["llr"] == primary["llr"]
    assert sens["primary"]["decision"] == primary["decision"]
    assert sens["capped_as_draw"]["llr"] == as_draw["llr"]
    assert sens["capped_as_loss"]["llr"] == as_loss["llr"]
    assert sens["capped_by_material"]["llr"] == by_material["llr"]
    assert sens["capped_games"] == 2
    # The policies genuinely differ: the legacy bias is measurable.
    assert as_draw["llr"] != as_loss["llr"]


def test_capped_sensitivity_never_changes_primary_decision():
    """The primary decision is the exclude-based one even when an alternative
    capped-scoring policy would decide differently: 60 decisive wins + 140
    capped games ACCEPT under exclusion (and capped-as-draw) but REJECT under
    capped-as-loss — and the primary decision is unaffected."""
    records = [
        {
            "board_result": "1-0" if w else "0-1",
            "cand_white": bool(w),
            "outcome_class": "decisive",
            "termination_reason": "checkmate",
        }
        for w in [True, False] * 30
    ]
    records += [
        {
            "board_result": "*",
            "cand_white": bool(i % 2 == 0),
            "outcome_class": "capped",
            "termination_reason": "max_plies_cap",
            "final_material_balance": 0,
        }
        for i in range(140)
    ]
    counts = sprt_tally(records)
    assert counts["wins"] == 60 and counts["capped_games"] == 140
    assert counts["wins"] + counts["draws"] + counts["losses"] == 60

    sens = capped_sensitivity(records, elo0=0.0, elo1=30.0)
    assert sens["primary"]["decision"] == "ACCEPT_H1"
    assert sens["capped_as_draw"]["decision"] == "ACCEPT_H1"
    assert sens["capped_as_loss"]["decision"] == "REJECT_H1"
    # The legacy (capped-as-loss) verdict would have been the opposite of the
    # primary verdict; the primary exclude-based decision stands.
    assert sens["primary"]["decision"] != sens["capped_as_loss"]["decision"]


# ---------------------------------------------------------------------------
# run_sprt() flow on synthetic / stubbed games (no real searches)
# ---------------------------------------------------------------------------


def _write_state(path: Path, name: str, games: list) -> Path:
    path.write_text(json.dumps({"name": name, "games": games}), encoding="utf-8")
    return path


def test_run_sprt_summary_excludes_capped_and_reports_accounting(tmp_path):
    runner = _bare_runner()
    games = _synthetic_records()
    _write_state(tmp_path / "cap_sem_state.json", "cap_sem", games)
    result = runner.run_sprt(
        "cap_sem", dict(CAND_CFG), dict(BASE_CFG), tmp_path, max_games=len(games)
    )
    counts = sprt_tally(games)
    assert result["games_count"] == len(games)
    assert result["cap_policy"] == "exclude"
    assert result["sprt"]["n"] == counts["wins"] + counts["draws"] + counts["losses"] == 5
    assert result["sprt"]["wins"] == 3
    assert result["sprt"]["draws"] == 1
    assert result["sprt"]["losses"] == 1
    assert result["capped_games"] == 2
    assert result["capped_candidate_white"] == 1
    assert result["capped_candidate_black"] == 1
    assert result["uncounted_games"] == 2
    # Sensitivity is reported alongside and agrees with the primary decision.
    assert result["capped_sensitivity"]["primary"]["llr"] == result["sprt"]["llr"]
    assert result["capped_sensitivity"]["capped_games"] == 2


def test_legacy_state_records_default_to_uncapped(tmp_path):
    """Pre-S5-05 state files resume unchanged: records without
    outcome_class/termination_reason classify from the board result alone, so
    the historical capped-draw residue keeps its recorded draw status
    (documented default-uncapped compatibility)."""
    runner = _bare_runner()
    legacy_games = [
        {
            "board_result": "1-0",
            "cand_white": True,
            "cand_outcome": "win",
            "moves": [],
            "seed": 1,
        },
        {
            # a legacy capped game (draw board string, cap adjudication)
            "board_result": "1/2-1/2",
            "cand_white": False,
            "cand_outcome": "draw",
            "adjudicated": True,
            "adjudication_reason": "max_plies_cap",
            "moves": [],
            "seed": 2,
        },
        {
            "board_result": "0-1",
            "cand_white": True,
            "cand_outcome": "loss",
            "moves": [],
            "seed": 3,
        },
        {
            "board_result": "1/2-1/2",
            "cand_white": True,
            "cand_outcome": "draw",
            "moves": [],
            "seed": 4,
        },
    ]
    _write_state(tmp_path / "legacy_state.json", "legacy", legacy_games)
    result = runner.run_sprt("legacy", dict(CAND_CFG), dict(BASE_CFG), tmp_path, max_games=4)
    assert result["sprt"]["n"] == 4
    assert result["sprt"]["wins"] == 1
    assert result["sprt"]["draws"] == 2  # legacy capped record stays a draw
    assert result["sprt"]["losses"] == 1
    assert result["capped_games"] == 0


def test_new_schema_state_reclassifies_capped_on_resume(tmp_path):
    runner = _bare_runner()
    games = [
        {
            "board_result": "1-0",
            "cand_white": True,
            "outcome_class": "decisive",
            "termination_reason": "checkmate",
        },
        {
            "board_result": "0-1",
            "cand_white": False,
            "outcome_class": "decisive",
            "termination_reason": "checkmate",
        },
        {
            "board_result": "*",
            "cand_white": True,
            "outcome_class": "capped",
            "termination_reason": "max_plies_cap",
        },
        {
            "board_result": "*",
            "cand_white": False,
            "outcome_class": "capped",
            "termination_reason": "max_plies_cap",
        },
    ]
    _write_state(tmp_path / "recl_state.json", "recl", games)
    result = runner.run_sprt("recl", dict(CAND_CFG), dict(BASE_CFG), tmp_path, max_games=4)
    assert result["sprt"]["n"] == 2
    assert result["sprt"]["wins"] == 2
    assert result["sprt"]["draws"] == 0
    assert result["capped_games"] == 2
    assert result["capped_candidate_white"] == 1
    assert result["capped_candidate_black"] == 1


def test_run_sprt_resume_plays_remaining_games_with_stubbed_player(tmp_path, monkeypatch):
    runner = _bare_runner()
    capped_pair = [
        {
            "board_result": "*",
            "cand_white": True,
            "cand_outcome": "capped",
            "outcome_class": "capped",
            "termination_reason": "max_plies_cap",
            "final_material_balance": 0,
            "plies": 150,
        },
        {
            "board_result": "*",
            "cand_white": False,
            "cand_outcome": "capped",
            "outcome_class": "capped",
            "termination_reason": "max_plies_cap",
            "final_material_balance": 0,
            "plies": 150,
        },
    ]
    _write_state(tmp_path / "resume_state.json", "resume", capped_pair)

    played: list[tuple[bool, int]] = []

    def fake_play(opening_fen, cand_white, cand_cfg, base_cfg, game_seed, max_plies=150, info=None):
        played.append((cand_white, max_plies))
        if info is not None:
            info.update(
                {
                    "outcome_class": "decisive",
                    "termination_reason": "checkmate",
                    "adjudicated": False,
                    "adjudication_reason": None,
                    "plies": 34,
                    "final_material_balance": 5,
                    "final_fen": "n/a",
                }
            )
        return ("1-0" if cand_white else "0-1", ["e2e4", "e7e5", "d2d4", "d7d5"])

    monkeypatch.setattr(runner, "play_single_game", fake_play)
    result = runner.run_sprt("resume", dict(CAND_CFG), dict(BASE_CFG), tmp_path, max_games=4)

    # Exactly the two remaining games were played, with the production cap.
    assert len(played) == 2
    assert all(m == 150 for _, m in played)
    assert {c for c, _ in played} == {True, False}
    # Stubbed checkmate wins count; the capped pair stays excluded.
    assert result["sprt"]["n"] == 2
    assert result["sprt"]["wins"] == 2
    assert result["capped_games"] == 2
    assert result["games_count"] == 4
    # The rewritten state file carries the new-schema fields on every record.
    state = json.loads((tmp_path / "resume_state.json").read_text(encoding="utf-8"))
    assert state["capped_games"] == 2
    assert state["cap_policy"] == "exclude"
    assert all("outcome_class" in g and "plies" in g for g in state["games"])
    assert all(g["cand_outcome"] in ("win", "capped") for g in state["games"])
    # PGN recorded only the newly played games, with their board results.
    pgn = (tmp_path / "resume.pgn").read_text(encoding="utf-8")
    assert pgn.count("[Result ") == 2
    assert '[Result "1-0"]' in pgn
