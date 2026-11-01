# ZeroPawn MCTS & Search Correctness Contracts (Tracked Reference)

Canonical, repository-tracked statement of the ZeroPawn MCTS and search integration contracts. It establishes the authoritative technical specifications resulting from the MCTS & Search Correctness Audit and Remediation.

---

## 1. Search Architecture & Execution Paths

ZeroPawn's search is implemented in native C++ (`cpp/src/mcts.cpp`, exposed as `zeropawn_cpp` via pybind11). There is **no Python MCTS implementation**; all Python components serve solely as orchestration around the native C++ engine.

### Three Search Execution Paths

1. **Serial Path (`zeropawn_cpp.MCTS.search` with `batch_size <= 1`)**:
   - Single simulation per step.
   - Immediate leaf evaluation via Python callback `Evaluator(game) -> (policy, value)`.
   - Used by UCI default, self-play serial mode, match runner, puzzles, and CLI search.

2. **Internal-Batched Path (`zeropawn_cpp.MCTS.search` with `batch_size > 1`)**:
   - Gathers up to $B$ candidate rollouts using virtual loss to encourage diversity.
   - Evaluates leaves in a single batched call via `BatchEvaluator(list[Game]) -> list[(policy, value)]`.
   - Collision detection (`pending_expansions`) prevents duplicate expansion within the same batch.
   - Supports subtree reuse (`reuse_tree=True`).
   - Used by UCI when `BatchSize > 1` or `SearchMode=batched`, and match runner with `--batch-size auto`.

3. **Legacy-Batched Path (`zeropawn_cpp.MCTS.search_batched`)**:
   - Raw flattened feature planes interface: `(n, state_size) -> (policy (n, 4672), values (n,))`.
   - Optimistic unscored node creation before batch evaluation.
   - Builds a fresh local tree per invocation; explicitly rejects `reuse_tree=True` (M-2).
   - In-batch transposition deduplication records identical-hash nodes in `duplicate_nodes` and attaches evaluated policy priors upon batch completion (S-4).
   - Used by self-play worker in `search_mode="batched"`.

### Dead Code Classification
- `python/eval/batched.py` is a legacy standalone reference module detailing the raw-planes protocol. It has zero production importers (pinned by regression tests) and is preserved strictly for protocol reference.

---

## 2. Selection PUCT & FPU Formula

Every search path executes the identical mathematical selection contract:

$$\text{score}(s, a_i) = Q(s, a_i) + c_{\text{puct}} \cdot P(s, a_i) \cdot \frac{\sqrt{\sum_b N(s, b)}}{1 + N(s, a_i) + V(s, a_i)}$$

Where:
- $N(s, b)$ is the edge visit count.
- $V(s, a_i)$ is the active virtual loss count.
- The exploration $\sqrt{\sum N}$ term excludes in-flight virtual losses.
- $c_{\text{puct}} = 2.0$ (locked baseline parameter; no dynamic scaling).

### Q-Value & FPU Formulation

$$Q(s, a_i) = \begin{cases} \dfrac{W(s, a_i) - V(s, a_i)}{N(s, a_i) + V(s, a_i)} & \text{if } N(s, a_i) + V(s, a_i) > 0 \\[2ex] Q_{\text{base}} - \text{fpu\_reduction} & \text{if } N(s, a_i) + V(s, a_i) = 0 \end{cases}$$

- **FPU Reduction**: `fpu_reduction = 0.2` across all modes.
- **Root FPU Base ($Q_{\text{base}}$ at root)**: Constant `0.0`. Unvisited root children score $0.0 - 0.2 = -0.2$. This ensures broad root exploration regardless of root evaluation score.
- **Non-Root FPU Base ($Q_{\text{base}}$ at depth $\ge 1$)**: KataGo-style parent-value FPU: $Q_{\text{base}} = -(W_{\text{edge}} / N_{\text{edge}})$ of the entered edge, representing the child position value from the perspective of the player to move.
- **Tie-Breaking**: Strict greater-than comparison (`>`) over legal action order; ties resolve deterministically to the lowest legal action index.

---

## 3. Value Semantics & Perspective Flow

1. **Neural Network Output Ordering**:
   - Model value head produces 3 logits: `[Win, Draw, Loss]` (indices `0 = Win`, `1 = Draw`, `2 = Loss`) from the **side-to-move** perspective.
2. **Scalar Conversion**:
   - $v = P(\text{win}) - P(\text{loss}) \in [-1, 1]$.
3. **Backup Negation**:
   - Exactly one sign inversion $v \leftarrow -v$ per ply walked from leaf to root.
   - $W(s, a_i)$ stores the accumulated value for the mover of action $a_i$.
   - Root value is the visit-weighted mean of visited root edges, from root side-to-move perspective.
4. **Terminal States**:
   - Checkmate = $-1.0$ (mover loses).
   - Stalemate, threefold repetition, 50-move rule, insufficient material = $0.0$.
   - Terminal leaves are identified by chess move generation and rules; they are **never evaluated by the neural network**.

---

## 4. Exploration, Noise & Temperature

- **Dirichlet Noise**:
  - Applied strictly at the root node during initial creation.
  - Distribution: $P'(s, a_i) = (1 - \varepsilon) P(s, a_i) + \varepsilon \cdot \eta_i$, where $\eta \sim \text{Dirichlet}(\alpha = 0.3)$ and $\varepsilon = 0.25$.
  - Gated by `is_root && config.add_noise`; active only in self-play data generation. Off in UCI, match, and evaluation.
- **Move Selection & Temperature**:
  - UCI, match, puzzles: Argmax visit count (temperature-free).
  - Self-play: Proportional sampling ($\tau = 1$) for early plies up to a random cutoff $U(15, 30)$, followed by argmax. No fractional temperature annealing exists.

---

## 5. Tree Reuse & Retained Visits

- **Subtree Promotion (`advance_root`)**:
  - Supports 1-action and 2-action advancement.
  - Strict validation: state cache key (mixing board hash and draw history), terminal status, legal action vector, and `max_nodes` subtree cap.
  - Virtual losses on the promoted root are reset to zero.
- **Retained Visits Reporting (S-3)**:
  - `MCTS.retained_visits()` reflects the accurate visit mass of the promoted root immediately upon `advance_root`, before the next `search()` is invoked.
  - Resetting or failing advancement resets `retained_visits` to 0.

---

## 6. UCI Time Control, Deadlines & Exception Recovery

- **Deadline Enforcement (S-1)**:
  - `go movetime` and `go wtime/btime` translate time allocations into wall-clock deadlines (`deadline_ms`).
  - Monotonic deadlines are passed through `_make_abort_fn` and polled inside the C++ search loop every ~32 simulations (serial) or per batch.
  - Search aborts promptly upon deadline expiry and returns the best move found so far.
- **Responsive Stop (S-1)**:
  - Budgeted searches execute asynchronously via `_search_worker`, keeping stdin open and responsive to `stop` and `quit`.
- **Honest EWMA Throughput (S-1)**:
  - Throughput tracking in `SearchTimeManager` records **actual completed simulations** (`new_simulations`), never planned budgets or soft caps.
- **Evaluator Exception Safety (S-5)**:
  - Evaluator exceptions during UCI search are logged safely.
  - A valid legal fallback move is emitted (`bestmove`).
  - Virtual losses and native resources are cleaned, preserving engine stability for subsequent commands.

---

## 7. Stale Documentation & Vault Discrepancy Ledger

Legacy internal notes are superseded by this contract. The following known stale statements in the vault and legacy documents are superseded by this contract:

| Document | Stale Clause | Authoritative Production Contract |
|---|---|---|
| `<vault>/.../Chess RL Deep Logic Audit...md` | DEF-01 states root FPU base is `-root_eval` | Root FPU base is constant `0.0`. Non-root FPU base follows parent value. |
| `<vault>/.../Manual - ZeroPawn...md` §6.2 | Claims WDL order `[Loss, Draw, Win]` | Output order is `[Win, Draw, Loss]` (0=Win, 1=Draw, 2=Loss). |
| `<vault>/.../Manual - ZeroPawn...md` §6.2 | Claims `virtual_loss = 3` default | Default is `virtual_loss = 1`. |
| `<vault>/.../Manual - ZeroPawn...md` §6.2 | Claims temperature anneal 1.0 -> 0.2 at ply 30 | Binary switch: $\tau=1$ sampling for plies $< U(15, 30)$, then argmax. |
| `python/uci/time_management.py` (prior to S-6) | Stale comment claiming MCTS has "no stop callback" | Corrected: C++ MCTS supports `abort_fn` deadline and stop polling. |
