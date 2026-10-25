import pytest
import numpy as np
import zeropawn_cpp


def uniform_evaluator(game):
    n = game.num_actions()
    policy = [1.0 / n] * n
    return policy, 0.0


def test_mcts_mate_in_one():
    mate_cases = [
        ("r1bqkb1r/pppp1ppp/2n5/4p3/2B1n3/5Q2/PPPP1PPP/RNB1K1NR w KQkq - 0 1", "f3f7"),
        ("rnbqkbnr/ppppp2p/8/5pp1/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 1", "d1h5"),
        ("6k1/5ppp/8/8/8/8/8/4R1K1 w - - 0 1", "e1e8"),
        ("8/8/8/8/8/5K2/4Q3/7k w - - 0 1", ["e2g2", "e2f1"]),
        ("7k/5K2/8/8/8/8/8/R7 w - - 0 1", "a1h1"),
    ]

    print(f"\n  [Op 1/2] Initializing C++ MCTS search engine (c_puct=2.0, seed=42)...")
    cfg = zeropawn_cpp.MCTSConfig()
    cfg.c_puct = 2.0
    cfg.add_noise = False
    cfg.seed = 42
    mcts = zeropawn_cpp.MCTS(cfg)

    print(
        f"  [Op 2/2] Solving {len(mate_cases)} mate-in-1 tactical positions with 200 simulations each..."
    )
    for i, (fen, expected) in enumerate(mate_cases, 1):
        game = zeropawn_cpp.ChessGame(fen)
        res = mcts.search(game, 200, uniform_evaluator)
        found = game.action_to_string(res.best_action)
        if isinstance(expected, list):
            assert found in expected
        else:
            assert found == expected
        print(f"    [OK] Case #{i}: Found mate '{found}' (expected {expected})")


def test_mcts_visit_distribution_and_determinism():
    print(
        "\n  [Op 1/2] Searching starting position (150 sims) and validating visit distribution sums to 1.0..."
    )
    game = zeropawn_cpp.ChessGame()
    cfg = zeropawn_cpp.MCTSConfig()
    cfg.seed = 999
    mcts = zeropawn_cpp.MCTS(cfg)

    res1 = mcts.search(game, 150, uniform_evaluator)
    visits = res1.action_visits
    assert isinstance(visits, np.ndarray)
    assert abs(visits.sum() - 1.0) < 1e-4
    print("    [OK] Action visits sum to 1.0 (valid probability distribution).")

    # Test determinism
    print("  [Op 2/2] Verifying search determinism across runs with fixed seed=999...")
    mcts.set_seed(999)
    res2 = mcts.search(game, 150, uniform_evaluator)
    assert res1.best_action == res2.best_action
    assert np.allclose(res1.action_visits, res2.action_visits)
    print("    [OK] Exact reproducibility verified: best_action and visit probabilities match.")


def batched_uniform_evaluator(batch_planes: np.ndarray):
    n = batch_planes.shape[0]
    policies = np.full((n, 4672), 1.0 / 4672, dtype=np.float32)
    values = np.zeros(n, dtype=np.float32)
    return policies, values


def test_mcts_search_batched():
    game = zeropawn_cpp.ChessGame()
    cfg = zeropawn_cpp.MCTSConfig()
    cfg.seed = 42
    cfg.add_noise = False
    mcts = zeropawn_cpp.MCTS(cfg)

    res1 = mcts.search_batched(game, 100, 1, batched_uniform_evaluator)
    assert abs(res1.action_visits.sum() - 1.0) < 1e-4
    assert res1.best_action >= 0

    mcts.clear()
    res16 = mcts.search_batched(game, 100, 16, batched_uniform_evaluator)
    assert abs(res16.action_visits.sum() - 1.0) < 1e-4
    assert res16.best_action >= 0
    assert -1.0 <= res16.root_value <= 1.0


def test_mcts_batched_mate_in_one():
    fen = "r1bqkb1r/pppp1ppp/2n5/4p3/2B1n3/5Q2/PPPP1PPP/RNB1K1NR w KQkq - 0 1"
    game = zeropawn_cpp.ChessGame(fen)
    cfg = zeropawn_cpp.MCTSConfig()
    cfg.c_puct = 2.0
    cfg.add_noise = False
    cfg.seed = 42
    mcts = zeropawn_cpp.MCTS(cfg)

    res = mcts.search_batched(game, 200, 16, batched_uniform_evaluator)
    found = game.action_to_string(res.best_action)
    assert found == "f3f7"
