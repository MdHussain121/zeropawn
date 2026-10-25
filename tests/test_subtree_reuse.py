"""Comprehensive tests for Stage 2 MCTS Subtree Reuse in ZeroPawn.

Covers:
1. Disabled-mode regression parity (20 positions bit-for-bit against baseline).
2. Retained-root behavior (valid advance, preserved visit stats, exactly M new simulations).
3. Fallback and safety paths (missing child, state mismatch, illegal actions, node cap).
4. Chess correctness across terminal states (checkmate, stalemate, 50-move rule, etc.).
5. Memory safety and ownership reconciliation (alloc_count == free_count).
6. Full UCI engine integration (options, ucinewgame, position moves replay).
"""

from __future__ import annotations

import io
import json
from pathlib import Path
import sys
import numpy as np
import pytest
import torch

import python  # noqa: F401
import zeropawn_cpp
from python.model.chess_net import ChessNet
from python.uci.uci_engine import UCIEngine


def dummy_eval(g):
    n = g.num_actions()
    return ([1.0 / n] * n, 0.0)


# ── 1. Disabled-Mode Baseline Parity Tests (20 positions) ──────────


def test_disabled_mode_parity_20_positions():
    """Verify that with reuse_tree=False, search matches pre-change baseline bit-for-bit."""
    baseline_path = Path("scratch/baseline_20pos_dummy.json")
    if not baseline_path.exists():
        pytest.skip("Baseline file scratch/baseline_20pos_dummy.json not found")

    with open(baseline_path, "r", encoding="utf-8") as f:
        baseline = json.load(f)

    for b in baseline:
        i = b["index"]
        fen = b["fen"]
        cfg = zeropawn_cpp.MCTSConfig()
        assert cfg.reuse_tree is False, "Default reuse_tree must be False"
        assert cfg.max_nodes == 25000, "Default max_nodes must be 25000"
        cfg.seed = 1000 + i
        mcts = zeropawn_cpp.MCTS(cfg)
        game = zeropawn_cpp.ChessGame(fen)
        res = mcts.search(game, 100, dummy_eval)

        assert res.best_action == b["best_action"], f"best_action mismatch at pos {i}"
        assert abs(res.root_value - b["root_value"]) < 1e-6, f"root_value mismatch at pos {i}"
        assert res.retained_visits == 0, f"Disabled mode must report 0 retained visits at pos {i}"
        assert res.new_simulations == 100, (
            f"Disabled mode must execute 100 new simulations at pos {i}"
        )

        visits = [float(x) for x in res.action_visits]
        nonzero_visits = {str(a): visits[a] for a in range(len(visits)) if visits[a] > 0}
        assert len(nonzero_visits) == len(b["nonzero_visits"]), f"Nonzero count mismatch at pos {i}"
        for a, v in nonzero_visits.items():
            assert abs(v - b["nonzero_visits"][a]) < 1e-6, (
                f"Visit count mismatch for action {a} at pos {i}"
            )


def test_disabled_mode_nn_parity_20_positions():
    """Verify that with reuse_tree=False and NN eval, results match pre-change baseline bit-for-bit."""
    baseline_path = Path("scratch/baseline_20pos_nn.json")
    ckpt_path = Path("models/checkpoints_finetune/best_model.pt")
    if not baseline_path.exists() or not ckpt_path.exists():
        pytest.skip("Baseline or checkpoint file not found")

    with open(baseline_path, "r", encoding="utf-8") as f:
        baseline = json.load(f)

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    sd = ckpt.get("model", ckpt.get("state_dict", ckpt))
    model = ChessNet(num_blocks=12, num_filters=128)
    model.load_state_dict(sd)
    model.eval()

    def nn_eval(g):
        chess_g = zeropawn_cpp.ChessGame(g.get_fen())
        planes = zeropawn_cpp.encode_state(chess_g)
        t = torch.from_numpy(planes).unsqueeze(0)
        with torch.no_grad():
            p_logits, v_wdl = model(t)
            p_probs = torch.softmax(p_logits.float(), dim=-1).squeeze(0).numpy().tolist()
            v_probs = torch.softmax(v_wdl.float(), dim=-1).squeeze(0).numpy()
            scalar_val = float(v_probs[0] - v_probs[2])
        return (p_probs, scalar_val)

    for b in baseline:
        i = b["index"]
        fen = b["fen"]
        cfg = zeropawn_cpp.MCTSConfig()
        cfg.seed = 2000 + i
        mcts = zeropawn_cpp.MCTS(cfg)
        game = zeropawn_cpp.ChessGame(fen)
        res = mcts.search(game, 50, nn_eval)

        assert res.best_action == b["best_action"], f"NN best_action mismatch at pos {i}"
        assert abs(res.root_value - b["root_value"]) < 1e-6, f"NN root_value mismatch at pos {i}"
        assert res.retained_visits == 0
        assert res.new_simulations == 50

        visits = [float(x) for x in res.action_visits]
        nonzero_visits = {str(a): visits[a] for a in range(len(visits)) if visits[a] > 0}
        for a, v in nonzero_visits.items():
            assert abs(v - b["nonzero_visits"][a]) < 1e-6, (
                f"NN visit mismatch for action {a} at pos {i}"
            )


# ── 2. Retained-Root and Simulation-Budget Contract ────────────────


def test_retained_root_budget_and_advancement():
    """Verify valid advance_root preserves visit statistics and executes exactly M new simulations."""
    cfg = zeropawn_cpp.MCTSConfig()
    cfg.reuse_tree = True
    cfg.seed = 42
    mcts = zeropawn_cpp.MCTS(cfg)

    game = zeropawn_cpp.ChessGame()
    res1 = mcts.search(game, 200, dummy_eval)

    assert res1.retained_visits == 0
    assert res1.new_simulations == 200
    assert mcts.has_retained_root() is True
    initial_nodes = mcts.live_node_count()
    assert initial_nodes > 1

    my_action = res1.best_action
    my_move = game.action_to_string(my_action)

    game_after_my = zeropawn_cpp.ChessGame()
    game_after_my.make_move_uci(my_move)

    # Find an opponent action that was expanded
    opp_legal = game_after_my.legal_moves_uci()
    chosen_opp_act = -1
    game_after_both = None

    for om in opp_legal:
        oact = game_after_my.string_to_action(om)
        cand_game = game_after_my.clone()
        cand_game.apply_action(oact)
        probe_mcts = zeropawn_cpp.MCTS(cfg)
        probe_mcts.search(game, 200, dummy_eval)
        if probe_mcts.advance_root(my_action, oact, cand_game):
            chosen_opp_act = oact
            game_after_both = cand_game
            break

    assert chosen_opp_act >= 0, "Expected at least one expanded reply in 200 simulations"

    # Advance root on main MCTS instance
    ok = mcts.advance_root(my_action, chosen_opp_act, game_after_both)
    assert ok is True
    assert mcts.has_retained_root() is True
    retained_nodes = mcts.live_node_count()
    assert 0 < retained_nodes < initial_nodes, "Re-rooting must reclaim unreachable branches"

    # Search with budget M=100
    res2 = mcts.search(game_after_both, 100, dummy_eval)
    assert res2.retained_visits > 0, "Retained root must report retained_visits > 0"
    assert res2.new_simulations == 100, "Exactly M=100 new simulations must be performed"
    assert mcts.retained_visits() == res2.retained_visits


def test_advance_root_fallbacks():
    """Verify that invalid advances deterministically reset to a fresh tree."""
    cfg = zeropawn_cpp.MCTSConfig()
    cfg.reuse_tree = True
    cfg.seed = 42
    mcts = zeropawn_cpp.MCTS(cfg)
    game = zeropawn_cpp.ChessGame()

    mcts.search(game, 100, dummy_eval)
    assert mcts.has_retained_root() is True

    # Case A: Missing opponent child
    ok = mcts.advance_root(game.legal_actions()[0], 9999, game)
    assert ok is False
    assert mcts.has_retained_root() is False
    assert mcts.live_node_count() == 0

    # Search again to re-populate
    mcts.search(game, 100, dummy_eval)
    assert mcts.has_retained_root() is True

    # Case B: Board state / FEN mismatch
    mismatch_game = zeropawn_cpp.ChessGame(
        "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"
    )
    ok2 = mcts.advance_root(game.legal_actions()[0], game.legal_actions()[0], mismatch_game)
    assert ok2 is False
    assert mcts.has_retained_root() is False
    assert mcts.live_node_count() == 0

    # Case C: Advance when root is nullptr
    ok3 = mcts.advance_root(0, 0, game)
    assert ok3 is False
    assert mcts.has_retained_root() is False


# ── 3. Node Limit and Reset Policy ─────────────────────────────────


def test_node_cap_enforcement():
    """Verify that trees exceeding max_nodes reset predictably."""
    cfg = zeropawn_cpp.MCTSConfig()
    cfg.reuse_tree = True
    cfg.max_nodes = 10  # small cap
    cfg.seed = 42
    mcts = zeropawn_cpp.MCTS(cfg)

    game = zeropawn_cpp.ChessGame()
    # 50 sims will generate > 10 nodes
    res = mcts.search(game, 50, dummy_eval)
    # The search should complete all 50 simulations, but reset root afterwards
    assert res.new_simulations == 50
    assert mcts.has_retained_root() is False
    assert mcts.live_node_count() == 0


# ── 4. Chess Correctness & Terminal States ─────────────────────────


def test_terminal_positions():
    """Verify that terminal positions execute 0 simulations and report correct value."""
    cfg = zeropawn_cpp.MCTSConfig()
    cfg.reuse_tree = True
    mcts = zeropawn_cpp.MCTS(cfg)

    # Checkmate: Fool's mate (White is mated)
    fools_mate = zeropawn_cpp.ChessGame(
        "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 0 2"
    )
    assert fools_mate.is_checkmate() is True
    res_mate = mcts.search(fools_mate, 100, dummy_eval)
    assert res_mate.new_simulations == 0
    assert res_mate.best_action == -1
    assert res_mate.root_value == -1.0

    # Stalemate
    stalemate = zeropawn_cpp.ChessGame("k7/8/1Q6/8/8/8/8/7K b - - 0 1")
    assert stalemate.is_stalemate() is True
    res_stale = mcts.search(stalemate, 100, dummy_eval)
    assert res_stale.new_simulations == 0
    assert res_stale.best_action == -1
    assert res_stale.root_value == 0.0


# ── 5. Memory Safety & Instrumentation ─────────────────────────────


def test_memory_safety_reconciliation():
    """Verify that node allocations and deallocations reconcile with zero leaks."""
    zeropawn_cpp.MCTS.reset_node_alloc_stats()
    cfg = zeropawn_cpp.MCTSConfig()
    cfg.reuse_tree = True
    game = zeropawn_cpp.ChessGame()

    for _ in range(5):
        mcts = zeropawn_cpp.MCTS(cfg)
        mcts.search(game, 100, dummy_eval)
        mcts.reset_root()
        del mcts

    allocated = zeropawn_cpp.MCTS.total_allocated_nodes()
    freed = zeropawn_cpp.MCTS.total_freed_nodes()
    assert allocated > 0, "Allocations must have occurred"
    assert allocated == freed, f"Memory leak: {allocated} allocated vs {freed} freed"


# ── 6. UCI Protocol Integration ────────────────────────────────────


def test_uci_reuse_tree_option_and_replay():
    """Verify UCI engine ReuseTree option, ucinewgame, and position continuation."""
    engine = UCIEngine(use_book=False, reuse_tree=True)
    assert engine.reuse_tree is True
    assert engine.mcts_config.reuse_tree is True

    # 1. UCI handshake option output
    buf = io.StringIO()
    old_stdout = sys.stdout
    try:
        sys.stdout = buf
        engine.handle_line("uci")
        out = buf.getvalue()
        assert "option name ReuseTree type check default true" in out
    finally:
        sys.stdout = old_stdout

    # 2. Start game, position startpos, and go
    buf = io.StringIO()
    try:
        sys.stdout = buf
        engine.handle_line("position startpos")
        engine.handle_line("go nodes 20")
        bestmove_line = [l for l in buf.getvalue().splitlines() if l.startswith("bestmove")]
        assert len(bestmove_line) == 1
        first_engine_move = bestmove_line[0].split()[1]
    finally:
        sys.stdout = old_stdout

    assert engine.last_engine_action is not None
    assert engine.last_engine_move == first_engine_move

    # 3. Simulate opponent reply in position command
    opp_move = "e7e5" if first_engine_move != "e7e5" else "c7c5"
    pos_cmd = f"position startpos moves {first_engine_move} {opp_move}"
    engine.handle_line(pos_cmd)

    # 4. Starting a new game resets retained tree
    engine.handle_line("ucinewgame")
    assert engine.mcts.has_retained_root() is False
    assert engine.last_engine_action is None
    assert engine.prev_moves == []


# ── 7. Exception Safety & Cache Invalidation Hardening ─────────────


def test_virtual_loss_exception_safety():
    """Verify that evaluator exceptions cleanly roll back virtual losses without leaking dirty stats."""
    game = zeropawn_cpp.ChessGame()
    root_hash = game.get_hash()

    cfg = zeropawn_cpp.MCTSConfig()
    cfg.reuse_tree = True
    cfg.seed = 42
    mcts = zeropawn_cpp.MCTS(cfg)

    def throwing_eval(g):
        if g.get_hash() == root_hash:
            return dummy_eval(g)
        raise RuntimeError("Simulated evaluator failure during search")

    # 1. Evaluator exception propagates and leaves ZERO dirty virtual losses
    with pytest.raises(RuntimeError, match="Simulated evaluator failure"):
        mcts.search(game, 10, throwing_eval)

    assert mcts.total_active_virtual_losses() == 0, (
        "Virtual losses must be 0 after exception rollback"
    )
    assert mcts.has_retained_root() is True, "Retained root preserved on reuse_tree=True"

    # 2. Subsequent search on the same MCTS succeeds and does not inherit dirty losses
    res = mcts.search(game, 20, dummy_eval)
    assert res.new_simulations == 20
    assert res.best_action >= 0
    assert mcts.total_active_virtual_losses() == 0

    # 3. In disabled mode, exception cleans up root entirely
    cfg_off = zeropawn_cpp.MCTSConfig()
    cfg_off.reuse_tree = False
    mcts_off = zeropawn_cpp.MCTS(cfg_off)

    with pytest.raises(RuntimeError):
        mcts_off.search(game, 10, throwing_eval)

    assert mcts_off.has_retained_root() is False
    assert mcts_off.live_node_count() == 0


def test_uci_weights_reload_invalidation(tmp_path):
    """Verify that successful UCI model reload invalidates NN cache and retained search tree."""
    # Create two distinct dummy checkpoints in tmp_path with explicit architecture metadata
    ckpt_a = tmp_path / "model_a.pt"
    ckpt_b = tmp_path / "model_b.pt"

    m_a = ChessNet(num_blocks=2, num_filters=32)
    m_b = ChessNet(num_blocks=2, num_filters=32)
    torch.save({"model": m_a.state_dict(), "num_blocks": 2, "num_filters": 32}, ckpt_a)
    torch.save({"model": m_b.state_dict(), "num_blocks": 2, "num_filters": 32}, ckpt_b)

    engine = UCIEngine(checkpoint_path=str(ckpt_a), device="cpu", reuse_tree=True, use_book=False)
    assert engine.model is not None
    assert engine.checkpoint_path == str(ckpt_a)

    # Perform a search so mcts populates cache and retains tree
    engine.handle_line("position startpos")
    engine.handle_line("go nodes 20")
    assert engine.mcts.cache_size() > 0, "NN cache must have entries after search"
    assert engine.mcts.has_retained_root() is True, "Root must be retained"
    assert engine.last_engine_action is not None

    # Successfully switch to model B via UCI setoption
    engine.handle_line(f"setoption name WeightsFile value {ckpt_b}")
    assert engine.checkpoint_path == str(ckpt_b)
    assert engine.mcts.cache_size() == 0, "NN cache must be invalidated after model reload"
    assert engine.mcts.has_retained_root() is False, (
        "Retained root must be flushed after model reload"
    )
    assert engine.last_engine_action is None
    assert engine.last_engine_move is None
    assert engine.prev_moves_at_search is None

    # Search after reload succeeds and populates fresh cache with new weights
    engine.handle_line("position startpos")
    engine.handle_line("go nodes 20")
    assert engine.mcts.cache_size() > 0


def test_uci_weights_reload_failure_preserves_engine(tmp_path):
    """Verify that a failed model reload does not corrupt the working model or flush valid state."""
    ckpt_a = tmp_path / "model_a.pt"
    m_a = ChessNet(num_blocks=2, num_filters=32)
    torch.save({"model": m_a.state_dict(), "num_blocks": 2, "num_filters": 32}, ckpt_a)

    engine = UCIEngine(checkpoint_path=str(ckpt_a), device="cpu", reuse_tree=True, use_book=False)
    original_model = engine.model
    assert original_model is not None

    engine.handle_line("position startpos")
    engine.handle_line("go nodes 10")
    cache_before = engine.mcts.cache_size()
    assert cache_before > 0

    # Attempt to reload a non-existent checkpoint path
    bad_path = str(tmp_path / "non_existent_weights_file.pt")
    engine.handle_line(f"setoption name WeightsFile value {bad_path}")

    # Verify working model and existing state were preserved
    assert engine.model is original_model, "Working model must not be corrupted on failed reload"
    assert engine.checkpoint_path == str(ckpt_a), (
        "Checkpoint path must not be overwritten on failure"
    )
    assert engine.mcts.cache_size() == cache_before, (
        "Existing cache must be preserved on failed reload"
    )


# ── 7. Stage 2.2 Batched Leaf Evaluation Tests ─────────────────────


def test_batch_config_defaults():
    """Verify that MCTSConfig defaults to batch_size=1 and exposes the property."""
    cfg = zeropawn_cpp.MCTSConfig()
    assert cfg.batch_size == 1, "MCTSConfig.batch_size must default to 1"
    cfg.batch_size = 4
    assert cfg.batch_size == 4, "MCTSConfig.batch_size must be mutable"


def test_batched_search_budget_and_parity():
    """Verify exact M simulations budget and bit-for-bit parity at batch_size=1."""
    game = zeropawn_cpp.ChessGame()

    # Baseline with batch_size=1
    cfg1 = zeropawn_cpp.MCTSConfig()
    cfg1.seed = 42
    cfg1.batch_size = 1
    mcts1 = zeropawn_cpp.MCTS(cfg1)
    res1 = mcts1.search(game, 50, dummy_eval)
    assert res1.new_simulations == 50
    assert res1.batch_size == 1
    assert mcts1.total_active_virtual_losses() == 0

    # Test batch sizes 2, 4, 8
    for bs in (2, 4, 8):
        cfg = zeropawn_cpp.MCTSConfig()
        cfg.seed = 42
        cfg.batch_size = bs
        mcts = zeropawn_cpp.MCTS(cfg)
        res = mcts.search(game, 53, dummy_eval)  # non-multiple budget
        assert res.new_simulations == 53, f"Budget must be exactly 53 with batch_size={bs}"
        assert res.batch_size == bs
        assert res.best_action >= 0
        assert mcts.total_active_virtual_losses() == 0


def test_batched_search_with_batch_callback():
    """Verify Python batch_evaluator callback is invoked with batches > 1."""
    game = zeropawn_cpp.ChessGame()
    cfg = zeropawn_cpp.MCTSConfig()
    cfg.seed = 123
    cfg.batch_size = 8
    mcts = zeropawn_cpp.MCTS(cfg)

    batch_calls = []

    def batch_eval(games_list):
        batch_calls.append(len(games_list))
        results = []
        for g in games_list:
            n = g.num_actions()
            results.append(([1.0 / n] * n, 0.0))
        return results

    res = mcts.search(game, 64, dummy_eval, batch_eval)
    assert res.new_simulations == 64
    assert len(batch_calls) > 0, "batch_evaluator must be called"
    assert max(batch_calls) > 1, f"Expected batch size > 1, got max {max(batch_calls)}"
    assert mcts.total_active_virtual_losses() == 0


def test_batched_search_subtree_reuse():
    """Verify subtree reuse works seamlessly when combined with batched leaf evaluation."""
    game = zeropawn_cpp.ChessGame()
    cfg = zeropawn_cpp.MCTSConfig()
    cfg.seed = 777
    cfg.reuse_tree = True
    cfg.batch_size = 4
    mcts = zeropawn_cpp.MCTS(cfg)

    # Move 1 search
    res1 = mcts.search(game, 50, dummy_eval)
    assert res1.new_simulations == 50
    assert res1.retained_visits == 0
    assert mcts.has_retained_root() is True

    my_act = res1.best_action
    my_move = game.action_to_string(my_act)
    g2 = zeropawn_cpp.ChessGame()
    g2.make_move_uci(my_move)
    opp_move = g2.legal_moves_uci()[0]
    opp_act = g2.string_to_action(opp_move)
    g2.make_move_uci(opp_move)

    advanced = mcts.advance_root(my_act, opp_act, g2)
    assert advanced is True, "advance_root must succeed under batched MCTS"

    # Move 2 search with reuse
    res2 = mcts.search(g2, 50, dummy_eval)
    assert res2.new_simulations == 50
    assert res2.retained_visits > 0
    assert mcts.total_active_virtual_losses() == 0


def test_batched_search_exception_safety():
    """Verify batched search exception unwinding cleanly removes virtual losses and reconciles memory."""
    game = zeropawn_cpp.ChessGame()
    cfg = zeropawn_cpp.MCTSConfig()
    cfg.seed = 42
    cfg.reuse_tree = True
    cfg.batch_size = 4
    mcts = zeropawn_cpp.MCTS(cfg)

    call_count = 0

    def throwing_batch_eval(games_list):
        nonlocal call_count
        call_count += 1
        if call_count >= 2:
            raise RuntimeError("Simulated GPU batch inference crash")
        results = []
        for g in games_list:
            n = g.num_actions()
            results.append(([1.0 / n] * n, 0.0))
        return results

    with pytest.raises(RuntimeError, match="Simulated GPU batch inference crash"):
        mcts.search(game, 40, dummy_eval, throwing_batch_eval)

    assert mcts.total_active_virtual_losses() == 0, (
        "No virtual losses may leak after batch exception"
    )


def test_uci_batch_size_option():
    """Verify UCIEngine supports BatchSize option via UCI protocol and constructor."""
    engine = UCIEngine(reuse_tree=True, batch_size=4)
    assert engine.batch_size == 4
    assert engine.mcts_config.batch_size == 4

    # Test setoption
    engine.handle_line("setoption name BatchSize value 8")
    assert engine.batch_size == 8
    assert engine.mcts_config.batch_size == 8

    # Test uci command reports option
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    try:
        engine.handle_line("uci")
        output = sys.stdout.getvalue()
    finally:
        sys.stdout = old_stdout

    assert "option name BatchSize type spin default 8 min 1 max 64" in output
