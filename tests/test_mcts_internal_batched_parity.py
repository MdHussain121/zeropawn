"""SA4-01 remediation: reference-parity regression tests for the internal-batched
(B > 1) MCTS path.

AUDIT FINDING (pre-push audit 2026-11-05, section 4, finding SA4-01, Medium):
the existing B=1 parity test routes to the serial branch of ``MCTS::search()``
(the C++ dispatches on ``config_.batch_size <= 1``), so the internal-batched
leaf-evaluation path was never exercised against the reference implementation.
These tests pin the internal-batched branch against
``tests/reference/ref_mcts.py::reference_internal_batched_search``.

HOW THE BATCHED PATH IS SELECTED
    ``MCTS::search()`` dispatches on ``MCTSConfig.batch_size``: values <= 1
    take the serial branch; values > 1 take the internal-batched branch
    (virtual-loss-aware batched selection, collision detection with batch
    truncation, deferred batched leaf evaluation through the BatchEvaluator
    callback, non-needs_eval child attachment from terminal values / the NN
    cache, once-per-batch abort polling). The Python binding
    ``MCTS.search(game, sims, eval_fn, batch_evaluator=None, abort_fn=None)``
    reaches it whenever ``MCTSConfig.batch_size > 1``; with
    ``batch_evaluator=None`` the C++ side wraps the per-state evaluator as the
    batch evaluator (one leaf at a time — identical scheduling, no batching
    benefit; documented remediation F-17). UCI ``SearchMode batched``
    (``BatchSize > 1``) routes through this path via
    ``UCIEngine.mcts.search(..., eval_batch_callback, ...)``.

PARITY CONTRACT — why it is mathematically valid
    The internal-batched path is a scheduling re-ordering of the same
    sequential PUCT algorithm, not a different algorithm: candidate
    construction inside a batch is still sequential, virtual loss is applied
    to each edge as its candidate is built (so sibling selection emulates
    sequential play), and evaluations are merely deferred to the batch
    boundary. Therefore, given a state-pure evaluator and a fresh NN cache,
    the entire trajectory is deterministic and comparable stage by stage.

(a) Exact parity (production internal-batched vs
    ``reference_internal_batched_search``), asserted by these tests:
      * integer root visit counts — asserted with atol=1e-6 on the normalized
        distribution, which is equivalent to exact integer equality because
        one visit differs by >= 1/num_sims >> 1e-6;
      * best_action (both implementations break ties toward the lowest legal
        index via strict ``score > best`` comparisons);
      * root_value, within float32 (production) vs float64 (reference)
        accumulation tolerance;
      * new_simulations equals the requested count;
      * degenerate cases are EXACTLY serial: with B=1 (routed to the serial
        branch by design) and for any B when num_sims <= B (single-candidate
        batches have no in-flight virtual loss at selection time and no
        possible collision), so the batched machinery must reproduce the
        serial result bit-for-bit up to float accumulation.

(b) Scheduling-tolerant invariants (hold for any batch schedule):
      * raw root visit mass equals completed simulations;
      * virtual losses fully unwind (``total_active_virtual_losses() == 0``),
        including on batch-evaluator failures;
      * aborted searches return valid partial aggregates (raw visit mass ==
        new_simulations, legal bestmove). The batched path polls the abort
        predicate once per batch — SA4-02 documents that granularity
        asymmetry against the serial path's 32-sim + wall-clock poll; it is
        intentional and not forced to match serial here;
      * with tree reuse: raw visit mass == retained visits + new simulations.

(c) Intentionally different / documented, NOT forced to be equal:
      * Batched B>1 trajectories differ from the SERIAL path by design
        (virtual-loss emulation changes selection order); the parity
        requirement is batched-vs-reference-batched, except for the
        degenerate single-candidate cases in (a).
      * Production memoizes evaluations in the NN cache and skips
        re-evaluating known states; the reference re-evaluates every
        candidate. With a state-pure evaluator both yield identical
        policy/value, so this is an evaluator-call-count difference, not a
        result difference. Parity therefore REQUIRES the evaluator to be a
        deterministic function of the position (no sampling/dropout). All
        tests use scripted deterministic evaluators — no checkpoint
        dependency for the core parity assertions.
      * A non-needs_eval leaf whose cache entry was evicted mid-batch would
        receive uniform priors in production (known finding SA4-04); the
        tests keep cache pressure far below the 100000-entry default cap so
        that window is never hit.
      * The reference re-calls ``eval_fn`` on terminal/stalemate candidates
        (the returned value is discarded by ``_build_node``); production
        never calls the evaluator on terminal leaves. That production-side
        contract — terminal leaves are resolved by game rules, not the
        network — is asserted directly.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
import pytest

# Reference implementation lives in the test tree (read-only test asset).
_REF_DIR = Path(__file__).resolve().parent / "reference"
if str(_REF_DIR) not in sys.path:
    sys.path.insert(0, str(_REF_DIR))

import ref_mcts  # noqa: E402

zeropawn_cpp = pytest.importorskip("zeropawn_cpp")


# ---------------------------------------------------------------------------
# Deterministic scripted evaluators (state-pure; no checkpoint dependency)
# ---------------------------------------------------------------------------


def _scripted_eval(
    salt: int,
    num_actions: int,
    policy_mode: str = "weights",
    value_mode: str = "hash",
) -> Callable:
    """Build a state-pure evaluator: policy from fixed per-action weights,
    value from the position hash. Deterministic across calls and processes."""
    rng = np.random.default_rng(1000 + salt)
    weights = 1.0 + rng.random(num_actions) * 3.0

    def eval_fn(game):
        actions = list(game.legal_actions())
        if not actions or game.is_terminal():
            # Total function: terminal/no-legal positions still return a
            # well-formed pair. Production never uses it there (rule-resolved
            # nodes); the reference calls it there but discards the result.
            val = game.terminal_value() if game.is_terminal() else 0.0
            return [0.0] * num_actions, float(val)
        p = [0.0] * num_actions
        if policy_mode == "nan":
            p = [float("nan")] * num_actions
        elif policy_mode == "illegal_spiky":
            # Huge mass on ILLEGAL actions, distinct small weights on legal
            # ones: after correct legal masking the priors are the normalized
            # legal weights; broken masking/indexing diverges immediately.
            legal_set = set(actions)
            for a in actions:
                p[a] = float(weights[a] % 1.0) + 0.01
            for i in range(num_actions):
                if i not in legal_set:
                    p[i] = 1000.0
        else:
            sub = weights[actions]
            s = float(np.sum(sub))
            if s > 1e-8:
                for a, w in zip(actions, sub / s):
                    p[a] = float(w)
            else:
                for a in actions:
                    p[a] = 1.0 / len(actions)
        if value_mode == "nan":
            return p, float("nan")
        h = game.get_hash()
        return p, math.sin(float(h & 0xFFFF) * 0.001) * 0.8

    return eval_fn


def _batch_wrapper(
    eval_fn: Callable,
    log: Optional[list] = None,
    max_batch: Optional[int] = None,
) -> Callable:
    """Wrap a state-pure per-state evaluator as a batch evaluator with
    contract assertions: never empty, never larger than the configured B,
    output length equals input length. Optionally logs (is_terminal) flags."""

    def batch_fn(states):
        n = len(states)
        assert 0 < n, "batch evaluator must never receive an empty batch"
        if max_batch is not None:
            assert n <= max_batch, f"batch size {n} exceeds configured B={max_batch}"
        if log is not None:
            log.extend(bool(s.is_terminal()) for s in states)
        return [eval_fn(s) for s in states]

    return batch_fn


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------


def _prod_cfg(B: int, seed: int = 42, reuse: bool = False) -> "zeropawn_cpp.MCTSConfig":
    cfg = zeropawn_cpp.MCTSConfig()
    cfg.c_puct = 2.0
    cfg.fpu_reduction = 0.2
    cfg.virtual_loss = 1
    cfg.add_noise = False
    cfg.seed = seed
    cfg.reuse_tree = reuse
    cfg.batch_size = B
    return cfg


def _ref_cfg(c_puct: float = 2.0, fpu: float = 0.2, vl: int = 1) -> dict:
    return {
        "c_puct": c_puct,
        "fpu": fpu,
        "fpu_reduction": fpu,
        "virtual_loss": vl,
        "add_noise": False,
        "seed": 42,
    }


def _assert_parity(
    prod,
    ref: Tuple[int, float, List[float]],
    sims: int,
    B: int,
    legal_actions: Sequence[int],
    num_actions: int,
    exact_visits: bool = True,
) -> None:
    """Assert the exact-parity contract (a) of the module docstring."""
    assert prod.new_simulations == sims, (
        f"new_simulations {prod.new_simulations} != requested {sims}"
    )
    assert prod.batch_size == B
    prod_visits = np.asarray(prod.action_visits, dtype=np.float64)
    ref_visits = np.asarray(ref[2], dtype=np.float64)

    # Legal-move consistency: visit support only on legal actions.
    illegal = np.ones(num_actions, dtype=bool)
    illegal[list(legal_actions)] = False
    assert np.all(prod_visits[illegal] == 0.0), "visits leaked to illegal actions"

    if exact_visits:
        # atol=1e-6 on a normalized integer distribution == exact integer
        # equality (1 visit >= 1/num_sims >> 1e-6).
        assert np.allclose(prod_visits, ref_visits, atol=1e-6), (
            f"visit counts diverge: max diff {np.max(np.abs(prod_visits - ref_visits))}"
        )
    assert prod.best_action == ref[0], f"best_action {prod.best_action} != reference {ref[0]}"
    assert abs(prod.root_value - ref[1]) < 1e-6, (
        f"root_value {prod.root_value} != reference {ref[1]}"
    )


# ---------------------------------------------------------------------------
# (a) Exact parity: degenerate cases
# ---------------------------------------------------------------------------


def test_sa401_reference_b1_degenerates_to_reference_serial():
    """Internal validity of the reference: its batched algorithm with B=1 must
    equal its serial algorithm (single-candidate batches have no in-flight VL
    and no collisions), validating the reference used for the B>1 assertions."""
    game = zeropawn_cpp.ChessGame()
    eval_fn = _scripted_eval(1, game.num_actions())
    ref_b1 = ref_mcts.reference_internal_batched_search(game.clone(), 30, eval_fn, _ref_cfg(), 1)
    ref_serial = ref_mcts.reference_serial_search(game.clone(), 30, eval_fn, _ref_cfg())
    assert ref_b1 == ref_serial


def test_sa401_production_b1_routes_to_serial_branch():
    """Pin the dispatch contract: MCTSConfig.batch_size == 1 routes to the
    serial branch even when a batch evaluator is supplied (the batch callback
    is never invoked). The audit's SA4-01 observation — the old B=1 'batched'
    test actually tested the serial branch — is intended behavior, and this
    test makes the routing explicit rather than accidental."""
    game = zeropawn_cpp.ChessGame()
    eval_fn = _scripted_eval(2, game.num_actions())
    batch_calls = []

    def batch_fn(states):
        batch_calls.append(len(states))
        return [eval_fn(s) for s in states]

    serial = zeropawn_cpp.MCTS(_prod_cfg(1, seed=123)).search(game.clone(), 30, eval_fn)
    batched_cfg = zeropawn_cpp.MCTS(_prod_cfg(1, seed=123))
    via_batch_flag = batched_cfg.search(game.clone(), 30, eval_fn, batch_fn)

    assert batch_calls == [], "B=1 must route to the serial branch (batch callback unused)"
    assert serial.best_action == via_batch_flag.best_action
    assert np.allclose(serial.action_visits, via_batch_flag.action_visits, atol=1e-7)
    assert abs(serial.root_value - via_batch_flag.root_value) < 1e-7

    # And B=1 equals the independent reference serial search exactly.
    ref = ref_mcts.reference_serial_search(game.clone(), 30, eval_fn, _ref_cfg())
    _assert_parity(serial, ref, 30, 1, game.legal_actions(), game.num_actions())


def test_sa401_single_candidate_batch_is_exactly_serial():
    """For any B, num_sims <= B produces single-candidate batches: no virtual
    loss in flight at selection, no possible collision — the batched branch
    must reproduce the serial result exactly."""
    game = zeropawn_cpp.ChessGame()
    eval_fn = _scripted_eval(3, game.num_actions())
    batch_fn = _batch_wrapper(eval_fn, max_batch=4)

    prod_b4 = zeropawn_cpp.MCTS(_prod_cfg(4)).search(game.clone(), 1, eval_fn, batch_fn)
    prod_b1 = zeropawn_cpp.MCTS(_prod_cfg(1)).search(game.clone(), 1, eval_fn)
    assert prod_b4.best_action == prod_b1.best_action
    assert np.allclose(prod_b4.action_visits, prod_b1.action_visits, atol=1e-7)
    assert abs(prod_b4.root_value - prod_b1.root_value) < 1e-7

    # Partial batch (3 sims < B=8) against the reference.
    ref = ref_mcts.reference_internal_batched_search(game.clone(), 3, eval_fn, _ref_cfg(), 8)
    prod = zeropawn_cpp.MCTS(_prod_cfg(8)).search(game.clone(), 3, eval_fn, batch_fn)
    _assert_parity(prod, ref, 3, 8, game.legal_actions(), game.num_actions())


# ---------------------------------------------------------------------------
# (a) Exact parity: reference-vs-production across batch sizes and positions
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("B", [2, 3, 5, 8])
def test_sa401_startpos_batch_size_sweep_parity(B):
    """Chess startpos: production internal-batched matches the independent
    reference exactly (integer visit counts, best action, root value) for
    several batch sizes, exercising in-flight VL selection and collision
    truncation at each granularity."""
    game = zeropawn_cpp.ChessGame()
    eval_fn = _scripted_eval(10, game.num_actions())
    batch_fn = _batch_wrapper(eval_fn, max_batch=B)

    prod = zeropawn_cpp.MCTS(_prod_cfg(B)).search(game.clone(), 40, eval_fn, batch_fn)
    ref = ref_mcts.reference_internal_batched_search(game.clone(), 40, eval_fn, _ref_cfg(), B)
    _assert_parity(prod, ref, 40, B, game.legal_actions(), game.num_actions())


@pytest.mark.parametrize(
    "fen,salt",
    [
        ("r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 5 4", 20),
        ("4k3/8/8/8/8/8/4P3/4K3 w - - 0 1", 21),
        ("k7/8/8/8/8/8/5q2/K7 w - - 0 1", 22),
    ],
)
def test_sa401_midgame_endgame_position_parity(fen, salt):
    """Midgame (Italian), bare-King-and-pawn endgame, and a forced
    single-legal-move position (root collision every batch => one candidate
    per batch) all match the reference exactly."""
    game = zeropawn_cpp.ChessGame()
    game.set_fen(fen)
    assert not game.is_terminal()
    assert len(game.legal_actions()) > 0
    eval_fn = _scripted_eval(salt, game.num_actions())
    batch_fn = _batch_wrapper(eval_fn, max_batch=4)

    prod = zeropawn_cpp.MCTS(_prod_cfg(4)).search(game.clone(), 36, eval_fn, batch_fn)
    ref = ref_mcts.reference_internal_batched_search(game.clone(), 36, eval_fn, _ref_cfg(), 4)
    _assert_parity(prod, ref, 36, 4, game.legal_actions(), game.num_actions())


def test_sa401_connect4_parity():
    """Second game family (Connect4, 7-action space, different terminal
    semantics) matches the reference under B=4."""
    game = zeropawn_cpp.Connect4()
    eval_fn = _scripted_eval(30, game.num_actions())
    batch_fn = _batch_wrapper(eval_fn, max_batch=4)

    prod = zeropawn_cpp.MCTS(_prod_cfg(4)).search(game.clone(), 40, eval_fn, batch_fn)
    ref = ref_mcts.reference_internal_batched_search(game.clone(), 40, eval_fn, _ref_cfg(), 4)
    _assert_parity(prod, ref, 40, 4, game.legal_actions(), game.num_actions())


def test_sa401_legal_indexing_spiky_policy_parity():
    """Legal-mask / legal-index consistency of the batched path: the evaluator
    returns huge mass on ILLEGAL actions and distinct small weights on legal
    ones. Correct masking normalizes over legal weights only; any edge-rank
    vs action-id indexing mistake or masking leak changes the trajectory and
    breaks exact visit parity."""
    game = zeropawn_cpp.ChessGame()
    eval_fn = _scripted_eval(40, game.num_actions(), policy_mode="illegal_spiky")
    batch_fn = _batch_wrapper(eval_fn, max_batch=4)

    prod = zeropawn_cpp.MCTS(_prod_cfg(4)).search(game.clone(), 40, eval_fn, batch_fn)
    ref = ref_mcts.reference_internal_batched_search(game.clone(), 40, eval_fn, _ref_cfg(), 4)
    _assert_parity(prod, ref, 40, 4, game.legal_actions(), game.num_actions())
    # The distribution's support must be inside the legal move set.
    prod_visits = np.asarray(prod.action_visits, dtype=np.float64)
    assert prod.best_action in game.legal_actions()


def test_sa401_no_batch_callback_fallback_parity():
    """The batch_evaluator=None route (C++ wraps the per-state evaluator as
    the batch evaluator, remediation F-17) follows the identical scheduling
    and must match the reference exactly."""
    game = zeropawn_cpp.ChessGame()
    eval_fn = _scripted_eval(50, game.num_actions())

    prod = zeropawn_cpp.MCTS(_prod_cfg(4)).search(game.clone(), 40, eval_fn)
    ref = ref_mcts.reference_internal_batched_search(game.clone(), 40, eval_fn, _ref_cfg(), 4)
    _assert_parity(prod, ref, 40, 4, game.legal_actions(), game.num_actions())


# ---------------------------------------------------------------------------
# (a) Exact parity: terminal handling
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "fen,expected_value",
    [
        ("rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3", -1.0),
        ("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1", 0.0),
    ],
)
def test_sa401_terminal_root_checkmate_and_stalemate(fen, expected_value):
    """Terminal roots (checkmate: value -1; stalemate: value 0) return the
    rule value with zero simulations and no best move — identical to the
    reference early return and without ever invoking the PRODUCTION evaluator
    (the reference calls eval_fn on the terminal root and discards the result;
    that asymmetry is documented in the module docstring, section (c))."""
    game = zeropawn_cpp.ChessGame()
    game.set_fen(fen)
    assert game.is_terminal()
    prod_eval_calls = []
    ref_eval_calls = []

    def make_logged_eval(log):
        def eval_fn(g):
            log.append(True)
            return [0.0] * g.num_actions(), 0.0

        return eval_fn

    prod = zeropawn_cpp.MCTS(_prod_cfg(4)).search(
        game.clone(), 20, make_logged_eval(prod_eval_calls)
    )
    ref = ref_mcts.reference_internal_batched_search(
        game.clone(), 20, make_logged_eval(ref_eval_calls), _ref_cfg(), 4
    )
    assert prod_eval_calls == [], "production must not evaluate a terminal root"
    assert prod.best_action == -1 == ref[0]
    assert prod.new_simulations == 0
    assert abs(prod.root_value - expected_value) < 1e-9
    assert abs(ref[1] - expected_value) < 1e-9
    assert not any(prod.action_visits)


def test_sa401_in_tree_checkmate_resolved_by_rules():
    """Position one move before fool's mate (black plays d8h4#): the terminal
    grandchild must be resolved by game rules. Production must (i) never pass
    a terminal position to the scalar evaluator or batch evaluator, and
    (ii) match the reference exactly, with the mating move taking over the
    root (root value positive)."""
    game = zeropawn_cpp.ChessGame()
    game.set_fen("rnbqkbnr/pppp1ppp/8/4p3/6P1/5P2/PPPPP2P/RNBQKBNR b KQkq g3 0 2")
    assert not game.is_terminal()
    mate_action = game.string_to_action("d8h4")
    assert mate_action in game.legal_actions()

    scalar_terminal_calls = []
    batch_terminal_calls = []
    ref_scalar_terminal_calls = []
    # Scripted prior that favors the mating move so the search reaches the
    # forced mate within the small simulation budget (deterministic).
    num_actions = game.num_actions()

    def make_eval(scalar_log):
        def eval_fn(g):
            if g.is_terminal():
                scalar_log.append(g.get_hash())
                return [0.0] * num_actions, 0.0
            p = [0.0] * num_actions
            for a in g.legal_actions():
                p[a] = 6.0 if a == mate_action else 1.0
            h = g.get_hash()
            return p, math.sin(float(h & 0xFFFF) * 0.001) * 0.8

        return eval_fn

    prod_eval = make_eval(scalar_terminal_calls)
    batch_fn = _batch_wrapper(prod_eval, log=batch_terminal_calls, max_batch=4)

    prod = zeropawn_cpp.MCTS(_prod_cfg(4)).search(game.clone(), 24, prod_eval, batch_fn)
    ref = ref_mcts.reference_internal_batched_search(
        game.clone(), 24, make_eval(ref_scalar_terminal_calls), _ref_cfg(), 4
    )
    _assert_parity(prod, ref, 24, 4, game.legal_actions(), game.num_actions())
    assert scalar_terminal_calls == [], "production scalar evaluator saw a terminal position"
    assert not any(batch_terminal_calls), "production batch evaluator saw a terminal position"
    assert prod.best_action == mate_action, "mating move must dominate the root"
    assert prod.root_value > 0.0, "search must see the forced mate value"


def test_sa401_zero_simulations_matches_reference_and_skips_batches():
    """num_sims = 0: no batch is built, visits are all zero, and both
    implementations agree on the degenerate aggregate (first legal action,
    root value 0.0)."""
    game = zeropawn_cpp.ChessGame()
    eval_fn = _scripted_eval(70, game.num_actions())
    batch_calls = []

    prod = zeropawn_cpp.MCTS(_prod_cfg(4)).search(
        game.clone(), 0, eval_fn, _batch_wrapper(eval_fn, log=batch_calls)
    )
    ref = ref_mcts.reference_internal_batched_search(game.clone(), 0, eval_fn, _ref_cfg(), 4)
    assert batch_calls == [], "no batch may be built for zero simulations"
    _assert_parity(prod, ref, 0, 4, game.legal_actions(), game.num_actions(), exact_visits=False)
    assert not any(prod.action_visits)
    assert prod.root_value == 0.0


# ---------------------------------------------------------------------------
# (a) Exact parity: NaN/Inf evaluator outputs where the contract defines them
# ---------------------------------------------------------------------------


def test_sa401_nan_policy_falls_back_to_uniform_parity():
    """NaN policy entries make the normalization sum fail the >1e-8 guard in
    BOTH implementations, which must identically fall back to uniform priors
    over legal actions — exact parity must survive degenerate NN output."""
    game = zeropawn_cpp.ChessGame()
    eval_fn = _scripted_eval(80, game.num_actions(), policy_mode="nan")
    batch_fn = _batch_wrapper(eval_fn, max_batch=4)

    prod = zeropawn_cpp.MCTS(_prod_cfg(4)).search(game.clone(), 24, eval_fn, batch_fn)
    ref = ref_mcts.reference_internal_batched_search(game.clone(), 24, eval_fn, _ref_cfg(), 4)
    _assert_parity(prod, ref, 24, 4, game.legal_actions(), game.num_actions())


def test_sa401_nan_value_propagates_identically():
    """A NaN value has no defined correct result, but the contract requires
    identical propagation: both implementations produce NaN root values and
    identical (deterministic) visit structures — NaN never wins a selection
    comparison in either implementation."""
    game = zeropawn_cpp.ChessGame()
    eval_fn = _scripted_eval(81, game.num_actions(), value_mode="nan")
    batch_fn = _batch_wrapper(eval_fn, max_batch=2)

    prod = zeropawn_cpp.MCTS(_prod_cfg(2)).search(game.clone(), 12, eval_fn, batch_fn)
    ref = ref_mcts.reference_internal_batched_search(game.clone(), 12, eval_fn, _ref_cfg(), 2)
    assert math.isnan(prod.root_value)
    assert math.isnan(ref[1])
    assert prod.best_action == ref[0]
    assert np.allclose(prod.action_visits, ref[2], atol=1e-6)


# ---------------------------------------------------------------------------
# (b) Scheduling-tolerant invariants
# ---------------------------------------------------------------------------


def test_sa401_batch_evaluator_wrong_length_raises_and_unwinds_vl():
    """A batch evaluator returning the wrong number of results must raise,
    and every in-flight virtual loss must be unwound (batch VL guard)."""
    game = zeropawn_cpp.ChessGame()
    eval_fn = _scripted_eval(90, game.num_actions())

    def bad_batch_fn(states):
        out = [eval_fn(s) for s in states]
        return out[:-1]  # wrong length

    mcts = zeropawn_cpp.MCTS(_prod_cfg(4))
    with pytest.raises(RuntimeError):
        mcts.search(game.clone(), 20, eval_fn, bad_batch_fn)
    assert mcts.total_active_virtual_losses() == 0


def test_sa401_batch_callback_size_and_empty_contract():
    """The batch evaluator receives between 1 and B states per call and is
    never invoked with an empty batch; terminal leaves never appear in a
    batch. (Empty-batch handling: num_sims=0 case covered by
    test_sa401_zero_simulations_matches_reference_and_skips_batches.)"""
    game = zeropawn_cpp.ChessGame()
    eval_fn = _scripted_eval(100, game.num_actions())
    sizes = []
    terminal_flags = []

    def batch_fn(states):
        sizes.append(len(states))
        assert len(states) > 0
        terminal_flags.extend(bool(s.is_terminal()) for s in states)
        return [eval_fn(s) for s in states]

    prod = zeropawn_cpp.MCTS(_prod_cfg(4)).search(game.clone(), 40, eval_fn, batch_fn)
    assert not any(terminal_flags)
    assert all(1 <= n <= 4 for n in sizes)
    assert prod.new_simulations == 40


def test_sa401_abort_returns_valid_partial_aggregate():
    """Batched abort granularity is once per batch (SA4-02 asymmetry is
    documented, not forced to serial). With B=2 and an abort predicate that
    fires on the third poll, exactly two batches (4 simulations) complete and
    the partial result stays self-consistent."""
    game = zeropawn_cpp.ChessGame()
    eval_fn = _scripted_eval(110, game.num_actions())
    batch_fn = _batch_wrapper(eval_fn, max_batch=2)
    polls = {"n": 0}

    def abort_fn():
        polls["n"] += 1
        return polls["n"] > 2

    mcts = zeropawn_cpp.MCTS(_prod_cfg(2))
    prod = mcts.search(game.clone(), 100, eval_fn, batch_fn, abort_fn)
    # Deterministic: poll 1 and 2 let two 2-candidate batches through.
    assert prod.new_simulations == 4
    raw_visits = np.round(np.asarray(prod.action_visits, dtype=np.float64) * 4)
    assert raw_visits.sum() == 4, "raw visit mass must equal completed simulations"
    assert prod.best_action in game.legal_actions()
    assert mcts.total_active_virtual_losses() == 0


def test_sa401_tree_reuse_batched_visit_mass_identity():
    """With reuse_tree=True and B>1 (the internal-batched path supports reuse,
    unlike legacy search_batched): raw visit mass after the second search
    equals retained visits + new simulations, and virtual losses unwind."""
    game = zeropawn_cpp.ChessGame()
    eval_fn = _scripted_eval(120, game.num_actions())
    batch_fn = _batch_wrapper(eval_fn, max_batch=4)

    mcts = zeropawn_cpp.MCTS(_prod_cfg(4, reuse=True))
    res1 = mcts.search(game.clone(), 30, eval_fn, batch_fn)
    best = res1.best_action
    assert best >= 0

    game_next = game.clone()
    game_next.apply_action(best)
    assert mcts.advance_root(best, game_next) is True
    retained = mcts.retained_visits()
    assert retained > 0

    res2 = mcts.search(game_next, 20, eval_fn, batch_fn)
    assert res2.new_simulations == 20
    raw = np.asarray(res2.action_visits, dtype=np.float64) * (retained + 20)
    assert int(round(float(raw.sum()))) == retained + 20
    assert mcts.total_active_virtual_losses() == 0


# ---------------------------------------------------------------------------
# (a) Exact parity: batch-boundary truncation
# ---------------------------------------------------------------------------


def test_sa401_many_small_batches_stay_exact():
    """A sims count that is not a multiple of B (last batch truncated) must
    still match the reference exactly — the truncation path
    (min(B, num_sims - sims_done)) is exercised at the boundary."""
    game = zeropawn_cpp.ChessGame()
    eval_fn = _scripted_eval(130, game.num_actions())
    batch_fn = _batch_wrapper(eval_fn, max_batch=3)

    prod = zeropawn_cpp.MCTS(_prod_cfg(3)).search(game.clone(), 47, eval_fn, batch_fn)
    ref = ref_mcts.reference_internal_batched_search(game.clone(), 47, eval_fn, _ref_cfg(), 3)
    _assert_parity(prod, ref, 47, 3, game.legal_actions(), game.num_actions())
