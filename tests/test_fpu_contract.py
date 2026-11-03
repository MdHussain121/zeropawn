"""Root-FPU contract tests (audit M-1).

The search-quality audit flagged a divergence between the vault's DEF-01
record ("root unvisited children inherit -root_eval") and the shipped code,
where every search path initializes the root FPU base to 0.0 (unvisited root
children score ``0 - fpu_reduction``).

Decision recorded in the remediation doc: the shipped 0-based root FPU is the
INTENDED behavior. It matches Lc0's historical ``fpu-at-root=disabled``
convention, it is what every strength measurement to date was collected with
(the locked play-time contract), and the vault's literal ``-root_eval`` formula
is perspective-inconsistent with the codebase's own non-root convention, where
unvisited children of a node inherit that node's own value (from its mover's
perspective) minus the reduction — the KataGo-style parent-value FPU.

These tests pin both halves of the contract so any future change is a
deliberate, reviewed decision:

1. Root: unvisited root children score ``0 - fpu_reduction``, INDEPENDENT of
   the root evaluation (i.e. NOT ``-root_eval - fpu_reduction``).
2. Non-root: unvisited children score ``parent_q - fpu_reduction`` where
   ``parent_q`` is the node's value from its own mover's perspective (arriving
   via backup), i.e. the FPU base follows the position value, not 0.
"""

import numpy as np

import zeropawn_cpp


def make_mcts(c_puct=2.0, fpu_reduction=0.2, seed=42, reuse_tree=False):
    cfg = zeropawn_cpp.MCTSConfig()
    cfg.c_puct = c_puct
    cfg.fpu_reduction = fpu_reduction
    cfg.add_noise = False
    cfg.virtual_loss = 1
    cfg.seed = seed
    cfg.reuse_tree = reuse_tree
    return zeropawn_cpp.MCTS(cfg)


def test_root_fpu_base_is_zero_not_negative_root_eval():
    """With every position valued +0.9 for its mover, a visited root child
    backs up Q = -0.9 (the opponent thinks THEY are winning). Under the
    shipped root FPU base (0), unvisited root children score -0.2 and the
    search keeps expanding NEW root children. Under the vault's literal
    -root_eval base (-0.9 - 0.2 = -1.1), every visit would return to the
    already-expanded child. We pin the spreading behavior."""
    N_SIMS = 10

    def const_eval(game):
        n = game.num_actions()
        return (np.full(n, 1.0 / n, dtype=np.float32), 0.9)

    game = zeropawn_cpp.ChessGame()
    mcts = make_mcts(seed=101)
    res = mcts.search(game, N_SIMS, const_eval)

    visits = np.asarray(res.action_visits, dtype=np.float64)
    nonzero = int((visits > 0).sum())
    # Base 0: each simulation expands a fresh root child (uniform priors,
    # FPU -0.2 dominates the backed-up -0.9). All N sims land on distinct
    # children, so the (normalized) visit distribution has N nonzero entries.
    assert nonzero == N_SIMS, (
        f"expected {N_SIMS} distinct explored root children, got {nonzero}; "
        "root FPU base appears to depend on the root evaluation"
    )
    # With uniform visits the best action is the first-expanded child (tie ->
    # lowest index) and root_value reflects the child value -0.9.
    assert res.root_value < 0.0


def test_root_fpu_with_default_settings_mate_still_found():
    """Guardrail: the pinned root FPU does not break the shipped mate-in-1
    behavior at the locked play-time defaults (c_puct=2.0, fpu=0.2)."""
    cfg = zeropawn_cpp.MCTSConfig()
    cfg.add_noise = False
    cfg.seed = 42
    mcts = zeropawn_cpp.MCTS(cfg)

    def uniform_eval(game):
        n = game.num_actions()
        return (np.full(n, 1.0 / n, dtype=np.float32), 0.0)

    game = zeropawn_cpp.ChessGame("rnbqkbnr/ppppp2p/8/5pp1/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 1")
    res = mcts.search(game, 200, uniform_eval)
    assert game.action_to_string(res.best_action) == "d1h5"


def test_non_root_fpu_follows_parent_value():
    """Deep in the tree, unvisited children inherit the node's own value minus
    the reduction (KataGo-style parent-value FPU), not a flat 0 base.

    Mock: White to move values +0.9 (White winning), Black to move -0.9
    (Black losing) — a consistent world where White is always better. The
    root policy is concentrated on the first legal move, so every simulation
    descends into the same child C (Black to move, V = -0.9):

      - At C, backed-up children Q = -0.9 (White's replies are good for White)
        and the FPU base parent_q = -Q(edge into C) = -0.9, so unvisited
        children score -1.1 < -0.9: C keeps REVISITING its first child. The
        search follows a single line and each simulation extends the tree by
        exactly one node.

      - If non-root FPU were 0-based instead, unvisited children at C would
        score -0.2 > -0.9 and every visit to C would expand a fresh branch —
        roughly twice as many nodes for the same budget.
    """
    N_SIMS = 20
    game = zeropawn_cpp.ChessGame()
    root_first_action = game.legal_actions()[0]

    def sided_eval(g):
        n = g.num_actions()
        policy = np.full(n_of := n, 1.0 / n, dtype=np.float32)
        if g.history_len() == 0:  # root only: concentrate on the first legal move
            policy[root_first_action] = 1.0
        value = 0.9 if g.side_to_move() else -0.9
        return (policy, value)

    mcts = make_mcts(seed=202, reuse_tree=True)  # retain tree so node count is observable
    mcts.search(game, N_SIMS, sided_eval)

    # Path-shaped tree: root + one node per simulation.
    live_nodes = mcts.live_node_count()
    assert live_nodes == N_SIMS + 1, (
        f"expected a single-line search ({N_SIMS + 1} live nodes), got {live_nodes}; "
        "non-root FPU does not follow the parent value"
    )
