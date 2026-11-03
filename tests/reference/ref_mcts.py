"""Independent reference implementation of ZeroPawn's documented MCTS/PUCT
algorithm (audit A-05 diagnostic; read-only with respect to the repository).

Written from the DOCUMENTED algorithm contract (PUCT selection with FPU,
one-leaf expansion per simulation, sign-flipping backup, visit-count root
policy, virtual-loss-aware batched selection) — not by transliterating
mcts.cpp. Exact equality of visit statistics between this reference and the
production native search is evidence that the production implementation
follows the intended algorithm; any divergence pinpoints the first stage
where behavior differs.

zeropawn_cpp Game objects are used only as move generators / position
containers. No production search code participates in the reference
computation.

Reference mirrors:
  reference_serial_search            — search() with batch_size <= 1
  reference_internal_batched_search  — search() with batch_size > 1
  reference_legacy_batched_search    — search_batched() (raw-planes API)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

# Optional per-batch diagnostic trace (populated only when TRACE is not None).
TRACE: Optional[List[dict]] = None


@dataclass
class RefNode:
    legal_actions: List[int]
    priors: List[float] = field(default_factory=list)
    visits: List[int] = field(default_factory=list)
    values: List[float] = field(default_factory=list)  # per-edge total value
    vl: List[int] = field(default_factory=list)  # per-edge active virtual loss
    children: List[Optional["RefNode"]] = field(default_factory=list)
    is_terminal: bool = False
    terminal_val: float = 0.0
    eval_val: float = 0.0
    scored: bool = False  # legacy path: has set_priors been applied


def _make_node(game, eval_fn, is_root: bool, cfg) -> RefNode:
    policy, value = eval_fn(game)
    return _build_node(game, policy, value, is_root, cfg)


def _build_node(game, policy, value, is_root: bool, cfg) -> RefNode:
    node = RefNode(legal_actions=list(game.legal_actions()))
    k = len(node.legal_actions)
    node.priors = []
    node.visits = []
    node.values = []
    node.vl = []
    node.children = []
    if game.is_terminal():
        node.is_terminal = True
        node.terminal_val = game.terminal_value()
        node.scored = True
        return node
    if k == 0:
        node.is_terminal = True
        node.terminal_val = 0.0
        node.scored = True
        return node
    pri = [float(policy[a]) for a in node.legal_actions]
    s = sum(pri)
    if s > 1e-8:
        pri = [p / s for p in pri]
    else:
        pri = [1.0 / k] * k
    if is_root and cfg["add_noise"] and k > 0:
        noise = cfg["noise_fn"](k)
        eps = cfg["dirichlet_weight"]
        pri = [(1.0 - eps) * p + eps * n for p, n in zip(pri, noise)]
    node.priors = pri
    node.visits = [0] * k
    node.values = [0.0] * k
    node.vl = [0] * k
    node.children = [None] * k
    node.eval_val = float(value)
    node.scored = True
    return node


def _make_unscored_node(game) -> RefNode:
    """Legacy search_batched: nodes are created before evaluation with
    uniform priors over legal actions (create_unscored_node)."""
    node = RefNode(legal_actions=list(game.legal_actions()))
    k = len(node.legal_actions)
    if game.is_terminal():
        node.is_terminal = True
        node.terminal_val = game.terminal_value()
        node.scored = True
        return node
    if k == 0:
        node.is_terminal = True
        node.terminal_val = 0.0
        node.scored = True
        return node
    node.priors = [1.0 / k] * k
    node.visits = [0] * k
    node.values = [0.0] * k
    node.vl = [0] * k
    node.children = [None] * k
    return node


def _set_priors(node: RefNode, policy, cfg, is_root: bool) -> None:
    k = len(node.legal_actions)
    if k == 0:
        return
    pri = [float(policy[a]) for a in node.legal_actions]
    s = sum(pri)
    if s > 1e-8:
        pri = [p / s for p in pri]
    else:
        pri = [1.0 / k] * k
    if is_root and cfg["add_noise"] and k > 0:
        noise = cfg["noise_fn"](k)
        eps = cfg["dirichlet_weight"]
        pri = [(1.0 - eps) * p + eps * n for p, n in zip(pri, noise)]
    node.priors = pri
    node.scored = True


def _select_serial(node: RefNode, parent_q: float, cfg) -> int:
    """Serial select_action: no virtual loss is ever in flight at selection
    time in the single-threaded serial path (applied and removed within one
    simulation; ancestors only, never siblings)."""
    k = len(node.legal_actions)
    if k == 0:
        return -1
    total = sum(node.visits)
    sqrt_total = math.sqrt(float(total))
    best_idx, best_score = 0, -1e9
    for i in range(k):
        n = node.visits[i]
        q = node.values[i] / float(n) if n > 0 else parent_q - cfg["fpu"]
        u = cfg["c_puct"] * node.priors[i] * sqrt_total / (1.0 + float(n))
        if q + u > best_score:
            best_score, best_idx = q + u, i
    return best_idx


def _select_vl(node: RefNode, parent_q: float, cfg) -> int:
    """Batched select_action: virtual losses in flight affect both the
    inflated visit count and the penalized value term."""
    k = len(node.legal_actions)
    if k == 0:
        return -1
    total = sum(node.visits)  # sqrt term excludes in-flight VL (as documented)
    sqrt_total = math.sqrt(float(total))
    best_idx, best_score = 0, -1e9
    for i in range(k):
        n = node.visits[i] + node.vl[i]
        if n > 0:
            q = (node.values[i] - float(node.vl[i])) / float(n)
        else:
            q = parent_q - cfg["fpu"]
        u = cfg["c_puct"] * node.priors[i] * sqrt_total / (1.0 + float(n))
        if q + u > best_score:
            best_score, best_idx = q + u, i
    return best_idx


def _backup(path: List[Tuple[RefNode, int]], leaf_val: float, vl_unit: int = 0) -> None:
    v = leaf_val
    for node, idx in reversed(path):
        if vl_unit:
            node.vl[idx] -= vl_unit
        v = -v
        node.values[idx] += v
        node.visits[idx] += 1


def _aggregate(root: RefNode, num_actions: int) -> Tuple[int, float, List[float]]:
    total, best_act, max_vis, sum_q = 0, -1, -1, 0.0
    dist = [0.0] * num_actions
    for i, act in enumerate(root.legal_actions):
        n = root.visits[i]
        total += n
        dist[act] = float(n)
        if n > max_vis:
            max_vis, best_act = n, act
        if n > 0:
            sum_q += root.values[i]
    root_value = (sum_q / total) if total > 0 else 0.0
    if total > 0:
        dist = [d / total for d in dist]
    return best_act, root_value, dist


# ---------------------------------------------------------------------------
# Serial path
# ---------------------------------------------------------------------------


def reference_serial_search(root_game, num_sims: int, eval_fn: Callable, cfg: dict):
    game = root_game.clone()
    root = _make_node(game, eval_fn, True, cfg)
    if not root.legal_actions or root.is_terminal:
        return (-1, root.terminal_val, [0.0] * root_game.num_actions())
    for _ in range(num_sims):
        sim = root_game.clone()
        curr = root
        path: List[Tuple[RefNode, int]] = []
        parent_q = 0.0
        while not curr.is_terminal and curr.legal_actions:
            idx = _select_serial(curr, parent_q, cfg)
            if idx < 0:
                break
            path.append((curr, idx))
            if curr.visits[idx] > 0:
                parent_q = -(curr.values[idx] / float(curr.visits[idx]))
            sim.apply_action(curr.legal_actions[idx])
            if curr.children[idx] is None:
                curr.children[idx] = _make_node(sim, eval_fn, False, cfg)
                curr = curr.children[idx]
                break
            curr = curr.children[idx]
        leaf_val = curr.terminal_val if curr.is_terminal else curr.eval_val
        _backup(path, leaf_val)
    return _aggregate(root, root_game.num_actions())


# ---------------------------------------------------------------------------
# Internal-batched path (search() with config.batch_size = B > 1)
# ---------------------------------------------------------------------------


def reference_internal_batched_search(
    root_game, num_sims: int, eval_fn: Callable, cfg: dict, B: int
):
    game = root_game.clone()
    root = _make_node(game, eval_fn, True, cfg)
    if not root.legal_actions or root.is_terminal:
        return (-1, root.terminal_val, [0.0] * root_game.num_actions())
    B = max(1, int(B))
    vl_unit = cfg["virtual_loss"]
    sims_done = 0
    while sims_done < num_sims:
        target = min(B, num_sims - sims_done)
        candidates = []
        pending_keys = set()
        for _ in range(target):
            sim = root_game.clone()
            curr = root
            path: List[Tuple[RefNode, int]] = []
            parent_q = 0.0
            collided = False
            parent_node, action_idx = None, -1
            needs_eval = False
            leaf_val = 0.0
            while not curr.is_terminal and curr.legal_actions:
                idx = _select_vl(curr, parent_q, cfg)
                if idx < 0:
                    break
                path.append((curr, idx))
                curr.vl[idx] += vl_unit
                if curr.visits[idx] > 0:
                    parent_q = -(curr.values[idx] / float(curr.visits[idx]))
                sim.apply_action(curr.legal_actions[idx])
                if curr.children[idx] is None:
                    if (id(curr), idx) in pending_keys:
                        collided = True
                        for n_, i_ in path:
                            n_.vl[i_] -= vl_unit
                        path = []
                        break
                    pending_keys.add((id(curr), idx))
                    parent_node, action_idx = curr, idx
                    if sim.is_terminal():
                        needs_eval, leaf_val = False, sim.terminal_value()
                    elif not sim.legal_actions():
                        needs_eval, leaf_val = False, 0.0
                    else:
                        needs_eval = True
                    break
                curr = curr.children[idx]
            if collided:
                break
            if parent_node is None:
                leaf_val = curr.terminal_val
                needs_eval = False
            candidates.append(
                {
                    "path": path,
                    "parent": parent_node,
                    "idx": action_idx,
                    "needs_eval": needs_eval,
                    "leaf_val": leaf_val,
                    "sim": sim,
                }
            )
        if not candidates:
            break
        if TRACE is not None:
            TRACE.append(
                {
                    "batch": len(TRACE),
                    "cands": [
                        {
                            "path": [(id(n_), i_) for n_, i_ in cand["path"]],
                            "leaf_hash": cand["sim"].get_hash(),
                            "leaf_val": cand["leaf_val"],
                            "needs_eval": cand["needs_eval"],
                        }
                        for cand in candidates
                    ],
                    "root_visits": list(root.visits),
                    "root_values": list(root.values),
                    "root_vl": list(root.vl),
                }
            )
        # Evaluation step: each needs_eval leaf is evaluated exactly once
        # (state-pure evaluator => batch wrapper produces identical results).
        for cand in candidates:
            if cand["needs_eval"]:
                policy, val = eval_fn(cand["sim"])
                cand["leaf_val"] = val
                cand["parent"].children[cand["idx"]] = _build_node(
                    cand["sim"], policy, val, False, cfg
                )
        for cand in candidates:
            if not cand["needs_eval"] and cand["parent"] is not None:
                policy, val = eval_fn(cand["sim"])
                cand["parent"].children[cand["idx"]] = _build_node(
                    cand["sim"], policy, val, False, cfg
                )
        for cand in candidates:
            _backup(cand["path"], cand["leaf_val"], vl_unit)
        sims_done += len(candidates)
    return _aggregate(root, root_game.num_actions())


# ---------------------------------------------------------------------------
# Legacy-batched path (search_batched; fresh root per call, raw-plane eval)
# ---------------------------------------------------------------------------


def reference_legacy_batched_search(root_game, num_sims: int, eval_fn: Callable, cfg: dict, B: int):
    """Mirror of MCTS::search_batched: unscored-node expansion with
    pending_index / hash dedup, immediate terminal backups, cached-eval
    scoring, one batch evaluation per frontier batch. Uses game.get_hash()
    as the position hash (Connect4 has no draw history; the chess cache key
    additionally mixes draw history — tests using this reference on chess
    positions must avoid hash-collision-sensitive scenarios)."""
    B = max(1, int(B))
    vl_unit = cfg["virtual_loss"]
    stats = {"dedup_attaches": 0}
    game = root_game.clone()
    root = _make_unscored_node(game)
    num_legal = len(root.legal_actions)
    if num_legal == 0 or root.is_terminal:
        return (-1, root.terminal_val, [0.0] * root_game.num_actions())

    cache: Dict[int, Tuple[List[float], float]] = {}

    # Root evaluation (single-element batch, cached afterwards)
    root_h = root_game.get_hash()
    policy_root, root_val = eval_fn(root_game)
    cache[root_h] = (list(policy_root), root_val)
    _set_priors(root, policy_root, cfg, True)

    sims_done = 0
    while sims_done < num_sims:
        batch: List[dict] = []  # {node, sim, hash, paths}
        pending_index: Dict[int, int] = {}
        hash_index: Dict[int, int] = {}

        while sims_done < num_sims and len(batch) < B:
            sim = root_game.clone()
            curr = root
            path: List[Tuple[RefNode, int]] = []
            parent_q = 0.0
            consumed = False

            while not curr.is_terminal and curr.legal_actions:
                if id(curr) in pending_index:
                    for n_, i_ in path:
                        n_.vl[i_] -= vl_unit
                    consumed = True
                    break
                idx = _select_vl(curr, parent_q, cfg)
                if idx < 0:
                    for n_, i_ in path:
                        n_.vl[i_] -= vl_unit
                    consumed = True
                    break
                path.append((curr, idx))
                curr.vl[idx] += vl_unit
                if curr.visits[idx] > 0:
                    parent_q = -(curr.values[idx] / float(curr.visits[idx]))
                sim.apply_action(curr.legal_actions[idx])
                if curr.children[idx] is None:
                    curr.children[idx] = _make_unscored_node(sim)
                    curr = curr.children[idx]
                    break
                curr = curr.children[idx]
            if consumed:
                break

            if curr.is_terminal:
                _backup(path, curr.terminal_val, vl_unit)
                sims_done += 1
                continue

            h = sim.get_hash()
            if h in cache:
                cached_policy, cached_val = cache[h]
                _set_priors(curr, cached_policy, cfg, False)
                _backup(path, cached_val, vl_unit)
                sims_done += 1
                continue

            if h in hash_index:
                batch[hash_index[h]]["paths"].append(path)
                # INTENDED semantics (differs from production, deliberately):
                # a node whose state duplicates a pending leaf in this batch
                # must still receive NN priors once that leaf is evaluated.
                # Production leaves such nodes with uniform priors forever
                # (see audit finding on legacy-batched dedup); the reference
                # scores them so divergence localizes exactly to that defect.
                stats["dedup_attaches"] += 1
                _dedup_pending = batch[hash_index[h]]
                if not _dedup_pending.get("dedup_nodes"):
                    _dedup_pending["dedup_nodes"] = []
                _dedup_pending["dedup_nodes"].append(curr)
                sims_done += 1
                continue

            batch.append({"node": curr, "sim": sim, "hash": h, "paths": [path]})
            pending_index[id(curr)] = len(batch) - 1
            hash_index[h] = len(batch) - 1
            sims_done += 1

        if not batch:
            break

        for leaf in batch:
            policy, val = eval_fn(leaf["sim"])
            _set_priors(leaf["node"], policy, cfg, False)
            cache[leaf["hash"]] = (list(policy), val)
            # Intended dedup scoring: duplicate-state nodes created in this
            # batch receive the same priors (identical position).
            for dup in leaf.get("dedup_nodes", ()):
                if not dup.scored:
                    _set_priors(dup, policy, cfg, False)
            for path in leaf["paths"]:
                _backup(path, val, vl_unit)
    return _aggregate(root, root_game.num_actions()), stats
