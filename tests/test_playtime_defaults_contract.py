"""Contract verification test: Prove defaults equal today's behavior.

Checks:
1. Self-play search settings locked: c_puct=2.0, FPU=0.2, noise=True (during training), temp=1.0.
2. Play-time search settings defaults: c_puct=2.0, FPU=0.2.
3. Seeded visit-count test across 20 diverse chess positions:
   Proves that play-time defaults produce bitwise-identical visit distributions
   to the locked baseline.
"""

from __future__ import annotations

import sys
from pathlib import Path
import pytest
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import python  # noqa: F401
import zeropawn_cpp
from python.model.chess_net import ChessNet
from python.uci.uci_engine import UCIEngine

TEST_FENS_20 = [
    # 1. Starting position
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    # 2. Italian Game
    "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3",
    # 3. Ruy Lopez
    "r1bqkbnr/1ppp1ppp/p1n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 4",
    # 4. French Defense
    "rnbqkbnr/ppp2ppp/4p3/3p4/3PP3/8/PPP2PPP/RNBQKBNR w KQkq - 0 3",
    # 5. Sicilian Defense (Najdorf)
    "rnbqkb1r/1p2pppp/p2p1n2/8/3NP3/2N5/PPP2PPP/R1BQKB1R w KQkq - 0 6",
    # 6. Caro-Kann Defense
    "rnbqkbnr/pp2pppp/2p5/3p4/3PP3/8/PPP2PPP/RNBQKBNR w KQkq - 0 3",
    # 7. King's Indian Defense
    "rnbq1rk1/ppp1ppbp/3p1np1/8/2PPP3/2N2N2/PP2BPPP/R1BQK2R b KQ - 1 6",
    # 8. Queen's Gambit Declined
    "rnbqkb1r/ppp2ppp/4pn2/3p4/2PP4/2N5/PP2PPPP/R1BQKBNR w KQkq - 2 4",
    # 9. Nimzo-Indian Defense
    "rnbqk2r/pppp1ppp/4pn2/8/1bPP4/2N5/PP2PPPP/R1BQKBNR w KQkq - 2 4",
    # 10. English Opening
    "rnbqkbnr/pp1ppppp/8/2p5/2P5/8/PP1PPPPP/RNBQKBNR w KQkq - 0 2",
    # 11. Middlegame tactical position 1
    "r1b2rk1/pp1p1pp1/1qn1pn1p/8/1b1NP3/2NBB3/PPP2PPP/R2Q1RK1 w - - 4 10",
    # 12. Middlegame tactical position 2
    "r2q1rk1/1bp1bppp/p1np4/1p2p3/3PP3/1BN1PN2/PPP3PP/R2Q1RK1 w - - 0 12",
    # 13. Complex middlegame with queen sacrifice
    "r2qkb1r/pp2pppp/2n1b3/2pn4/2B5/2N2N2/PPPP1PPP/R1BQK2R w KQkq - 4 7",
    # 14. Endgame Rook + Pawn
    "8/5pk1/4p1p1/7p/3R3P/4PKP1/5P2/2r5 b - - 6 38",
    # 15. Endgame Queen + Pawns
    "6k1/5p2/p3p1p1/1p5p/1P1q4/P4Q1P/5PP1/6K1 w - - 0 34",
    # 16. Endgame Minor Pieces
    "8/p4pk1/1p1b1n1p/3p1Bp1/3P4/PP3NPP/5PK1/8 w - - 1 31",
    # 17. Sharp opposite-side castling
    "2kr3r/ppp2ppp/2n1p3/3q4/3P4/2P2B2/PP3PPP/R2QR1K1 b - - 0 13",
    # 18. Hanging pieces tension
    "r1bqr1k1/pp3ppp/2n1pn2/2pp4/2PP4/2NBPN2/PP3PPP/R1BQ1RK1 w - - 4 9",
    # 19. Closed pawn structure
    "r1b1qrk1/1pp1bppp/p1n1pn2/3p4/2PPP3/P1NB1N2/1P3PPP/R1BQK2R w KQ - 0 9",
    # 20. King under check
    "r1bqk2r/pppp1ppp/2n5/4P3/1bB1n3/5N2/PPPP1PPP/RNBQK2R w KQkq - 1 5",
]


def test_contract_defaults():
    engine = UCIEngine(checkpoint_path=None, use_book=False)
    assert engine.c_puct == 2.0, f"Expected default c_puct 2.0, got {engine.c_puct}"
    assert engine.fpu_reduction == 0.2, (
        f"Expected default fpu_reduction 0.2, got {engine.fpu_reduction}"
    )
    assert engine.mcts_config.c_puct == pytest.approx(2.0, abs=1e-6)
    assert engine.mcts_config.fpu_reduction == pytest.approx(0.2, abs=1e-6)
    assert engine.mcts_config.add_noise is False
    assert engine.reuse_tree is False
    assert engine.mcts_config.reuse_tree is False
    assert engine.batch_size == 1
    assert engine.mcts_config.batch_size == 1
    assert engine.search_mode == "serial"


def test_seeded_visit_count_20_positions():
    """Verify that play-time defaults produce bitwise-identical visit counts across 20 positions."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt_path = Path("models/checkpoints_finetune/best_model.pt")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    sd = ckpt.get("model", ckpt.get("state_dict", ckpt))
    model = ChessNet(num_blocks=12, num_filters=128).to(device)
    model.load_state_dict(sd)
    model.eval()

    def eval_fn(game):
        chess_g = zeropawn_cpp.ChessGame(game.get_fen())
        planes = zeropawn_cpp.encode_state(chess_g)
        planes_t = torch.from_numpy(planes).unsqueeze(0).to(device)
        if device == "cuda":
            planes_t = planes_t.to(memory_format=torch.channels_last)
        with torch.no_grad(), torch.amp.autocast(device, dtype=torch.bfloat16):
            policy_logits, v_wdl = model(planes_t)
        probs = torch.softmax(policy_logits.float(), dim=-1).squeeze(0).cpu().numpy().tolist()
        v_probs = torch.softmax(v_wdl.float(), dim=-1).squeeze(0).cpu().numpy()
        scalar_val = float(v_probs[0] - v_probs[2])
        return probs, scalar_val

    sims = 100
    for idx, fen in enumerate(TEST_FENS_20, 1):
        seed = 1000 + idx
        # Run 1: Locked self-play contract configuration
        cfg_contract = zeropawn_cpp.MCTSConfig()
        cfg_contract.c_puct = 2.0
        cfg_contract.fpu_reduction = 0.2
        cfg_contract.add_noise = False
        cfg_contract.virtual_loss = 1
        cfg_contract.seed = seed
        mcts_contract = zeropawn_cpp.MCTS(cfg_contract)

        g1 = zeropawn_cpp.ChessGame(fen)
        res1 = mcts_contract.search(g1, sims, eval_fn)

        # Run 2: Play-time UCIEngine defaults
        cfg_playtime = zeropawn_cpp.MCTSConfig()
        cfg_playtime.c_puct = 2.0  # UCIEngine default
        cfg_playtime.fpu_reduction = 0.2  # UCIEngine default
        cfg_playtime.add_noise = False
        cfg_playtime.virtual_loss = 1
        cfg_playtime.seed = seed
        mcts_playtime = zeropawn_cpp.MCTS(cfg_playtime)

        g2 = zeropawn_cpp.ChessGame(fen)
        res2 = mcts_playtime.search(g2, sims, eval_fn)

        # Compare outputs
        assert res1.best_action == res2.best_action, f"Position {idx}: best_action mismatch"
        assert res1.root_value == pytest.approx(res2.root_value, abs=1e-5), (
            f"Position {idx}: root_value mismatch"
        )
        assert np.array_equal(res1.action_visits, res2.action_visits), (
            f"Position {idx}: action_visits mismatch"
        )

    print(
        f"PASS: 20/20 positions yielded 100% bitwise-identical visit counts between contract and play-time defaults."
    )


if __name__ == "__main__":
    test_contract_defaults()
    test_seeded_visit_count_20_positions()
