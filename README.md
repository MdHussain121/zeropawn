# ZeroPawn: AlphaZero-Style Chess Reinforcement Learning Platform

[![CI](https://github.com/MdHussain121/zeropawn/actions/workflows/ci.yml/badge.svg)](https://github.com/MdHussain121/zeropawn/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![C++17](https://img.shields.io/badge/C%2B%2B-17-blue.svg)](https://en.cppreference.com/w/cpp/17)
[![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20Linux%20%7C%20macOS-lightgrey.svg)]()
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**ZeroPawn** is an end-to-end AlphaZero-style chess reinforcement learning platform and chess engine designed for high throughput, reproducibility, and robust evaluation on consumer hardware.

ZeroPawn pairs an optimized C++17 bitboard chess engine and Monte Carlo Tree Search (MCTS) core with a PyTorch dual-head Residual Neural Network (`ChessNet`). It incorporates an asynchronous batched inference server, memory-mapped replay buffers with dynamic human data tapering, Syzygy endgame tablebase adjudication, Stockfish calibration baselines, and a comprehensive 80-operation Rich terminal CLI.

---

## ⚡ Architecture at a Glance

```text
                               ┌────────────────────────────────────────────────┐
                               │             ZeroPawn Control Center            │
                               │     .\zeropawn.cmd / python cli.py (Rich CLI)  │
                               └───────┬──────────────────────────────┬─────────┘
                                       │                              │
                    ┌──────────────────▼─────────────┐ ┌──────────────▼────────────────┐
                    │     Self-Play Workers (C++)    │ │  Supervised Warm-Start Pipeline│
                    │   - Bitboard move generation   │ │  - 2500+ Elo Human Master PGNs│
                    │   - Batched MCTS tree search   │ │  - Curated Tactical Puzzles   │
                    │   - Syzygy tablebase checks    │ │  - Binary memmap float32 data │
                    └──────────────────┬─────────────┘ └──────────────┬────────────────┘
                                       │                              │
                                       │ Batched Tensor Requests      │ Training Samples
                                       ▼                              ▼
                    ┌────────────────────────────────┐ ┌───────────────────────────────┐
                    │     Async InferenceServer      │ │         RLTrainer             │
                    │   - Dynamic batching (128-512) │ │   - Ring replay buffer        │
                    │   - Lock-free forward pass     │ │   - Human data linear taper   │
                    │   - Dual CUDA streams (-1 / 0) │ │   - Policy & WDL loss heads   │
                    └──────────────────┬─────────────┘ └──────────────┬────────────────┘
                                       │                              │
                                       └──────────────┬───────────────┘
                                                      │ Pushes Weights (Hot-Reload)
                                                      ▼
                                       ┌──────────────────────────────┐
                                       │      Dual-Head ResNet        │
                                       │   12 Blocks, 128 Filters     │
                                       │   Policy (4,672) / WDL (3)   │
                                       └──────────────────────────────┘
```

---

## 🛠️ Prerequisites & Installation

### System Requirements
- **Python:** 3.11 or 3.12 (64-bit)
- **C++ Compiler:** C++17 compliant compiler:
  - **Windows:** MinGW-W64 GCC 11+ (recommended: WinLibs UCRT via `winget install BrechtSanders.WinLibs.POSIX.UCRT`) or MSVC 2019+
  - **Linux:** GCC 10+ or Clang 12+ (`sudo apt install build-essential ninja-build cmake`)
  - **macOS:** Xcode Command Line Tools (`xcode-select --install`)
- **CMake & Ninja:** CMake 3.20+ and Ninja build system
- **Stockfish (Optional):** Recommended for evaluation benchmarks and calibration (placed in `tools/` or on system PATH)

### 1. Clone & Set Up Python Environment

```bash
git clone https://github.com/MdHussain121/zeropawn.git
cd zeropawn

# Create and activate virtual environment
python -m venv .venv

# On Windows:
.venv\Scripts\activate
# On Linux / macOS:
source .venv/bin/activate

# Install ZeroPawn in editable mode with development dependencies
pip install -e ".[dev]"
```

### 2. Build the Native C++ Engine Extension

ZeroPawn utilizes `pybind11` to link high-speed C++ search routines directly into Python:

```bash
# Configure and build Release binaries with CMake & Ninja:
cmake -B build/cmake -S . -DCMAKE_BUILD_TYPE=Release
cmake --build build/cmake --config Release -j 4

# Run native C++ unit test runner:
ctest --test-dir build/cmake --output-on-failure
```

---

## 🚀 30-Second Quickstart (Verify Installation)

Once installed, verify the entire hardware and software stack with built-in telemetry:

```powershell
# 1. System health, GPU detection, and compiler discovery:
.\zeropawn.cmd status

# 2. Automated end-to-end health check:
.\zeropawn.cmd diagnose health

# 3. Python <-> C++ binding compatibility test:
python cli.py build compat

# 4. Native C++ engine tests (11 test suites):
.\zeropawn.cmd build test

# 5. Supervised learning 10-step smoke test:
.\zeropawn.cmd supervised --verify
```

On Windows, launch the interactive Control Center directly:
```powershell
.\zeropawn.cmd
```

---

## 🧠 Model Checkpoints & Pre-Trained Weights

ZeroPawn's dual-head ResNet (`ChessNet`, 12 residual blocks, 128 filters) uses PyTorch state dictionaries (`.pt`).

### Available Pre-Trained Models

We provide two pre-trained model checkpoints:

| Model | Path | Val Loss | Size | Status & Relative Strength |
| :--- | :--- | :---: | :---: | :--- |
| **Fine-Tuned Model** *(Recommended)* | `models/checkpoints_finetune/best_model.pt` | **1.7975** | 96.8 MB | **Strongest model.** Significantly outperforms the scaled baseline in tactical puzzle accuracy, blunder avoidance, and tournament win rate. Recommended for general play and UCI engine matches. |
| **Scaled Baseline Model** | `models/checkpoints_scaled/best_model.pt` | **3.1355** | 98.9 MB | Certified supervised warm-start baseline (19,460 steps). Maintained for reference benchmarking, research ablation, and training reproduction. |

> [!TIP]
> **Model Strength:** The **Fine-Tuned Model** is substantially stronger than the **Scaled Model**. If you want the highest playing strength and sharpest tactical calculation, use the fine-tuned model.

### Checkpoint Discovery Hierarchy
When launching the engine without an explicit `--checkpoint` flag, ZeroPawn automatically selects the strongest available checkpoint in this order:
1. `models/checkpoints_finetune/best_model.pt` *(Fine-tuned model — strongest)*
2. `models/checkpoints_scaled/best_model.pt` *(Scaled supervised baseline)*
3. `models/checkpoints_rl/best_model.pt` *(Reinforcement learning self-play checkpoint)*
4. Any `.pt` path explicitly passed via `--checkpoint <path>`

### Downloading Official Checkpoints

Use the built-in CLI to download or verify release checkpoints:

```bash
# 1. Download the Fine-Tuned Model (Strongest, recommended for play):
zeropawn models download --model finetuned

# 2. Download the Scaled Baseline Model:
zeropawn models download --model scaled

# 3. Download both models simultaneously:
zeropawn models download --model both

# 4. Verify SHA-256 integrity of local checkpoints:
zeropawn models download --verify-only
```

#### Certified Checksums
- **Fine-Tuned Model:** `SHA-256: 7E326883D61BB59B9D706BCB556858F2879E713729F59ABCF046E302C8252685`
- **Scaled Baseline Model:** `SHA-256: 79A72EF746F4BFDA2555FE7D0EC04970BE7CCD4A39E282E16A05594ED15BC36B`

### Training a Fresh Baseline Checkpoint
You can also train a baseline model directly on your hardware:
```bash
# Verify the training pipeline (10 micro-steps, ~8 seconds):
zeropawn supervised --verify

# Run full supervised training with warm-start dataset:
zeropawn train supervised --profile 8gb-gpu --lr 0.0002
```

---

## 🎮 Playing & Evaluation

### Universal Chess Interface (UCI) Mode
ZeroPawn fully complies with the UCI protocol. Connect it to any chess GUI (Arena, Cutechess, BanksiaGUI, Nibbler):
```bash
zeropawn uci
```
*Note:* In UCI mode, all ANSI escape codes, telemetry, and banners are automatically suppressed to guarantee protocol purity.

### Live Terminal Match
Watch ZeroPawn play against itself or Stockfish with real-time ASCII board updates and MCTS visit distributions:
```bash
# Watch engine search live:
zeropawn match live

# Play interactively as human vs ZeroPawn in your terminal:
zeropawn match human --color white
```

### Calibrated Elo Evaluation
Benchmark ZeroPawn against calibrated Stockfish rating levels:
```bash
zeropawn eval baseline --elo 1500 --openings 50 --sims 800
```

---

## 📚 Core CLI Workflows & Interactive Discovery

ZeroPawn provides an exhaustive Rich terminal CLI spanning 114 executable commands and 360+ arguments across 10 functional categories.

### Interactive Discovery & Argument Builders
Explore and configure all supported commands and combinations directly from your terminal:
```bash
# Launch interactive Command Browser across all categories:
zeropawn interactive --browser

# Inspect the full argument inventory for any command:
zeropawn interactive --options "train"

# Explore valid argument combinations and constraint rules:
zeropawn interactive --combinations "eval baseline"

# Interactive wizard with custom argument configuration and dry-run preview:
zeropawn interactive --builder "match live"
```

### Key Operational Commands

| Command | Action | Description |
|---|---|---|
| `zeropawn status` | System Status | Displays GPU, CUDA, CPU, RAM, compiler, and checkpoint inventory. |
| `zeropawn diagnose health` | Diagnostics | Verifies Python, PyTorch, C++ extensions, paths, and disk space. |
| `zeropawn build compat` | C++ Verification | Asserts Python/C++ binding contracts, zero-copy buffers, and FEN parity. |
| `zeropawn models list` | Model Inventory | Lists all discovered checkpoints, architectures, steps, and loss metrics. |
| `zeropawn models inspect` | Model Inspection | Prints layer parameter counts, weight norms, and numerical stability. |
| `zeropawn train supervised` | Supervised Training | Warm-starts policy and value networks on master games and puzzles. |
| `zeropawn train rl` | Self-Play RL | Launches asynchronous MCTS self-play workers and live training server. |
| `zeropawn tablebase download` | Syzygy Tablebases | Downloads and verifies 145 Syzygy 3-4-5 piece WDL (.rtbw) files. |
| `zeropawn puzzles eval` | Tactical Evaluation | Evaluates engine tactical puzzle-solving rate across rating tiers. |

---

## 💻 Hardware Requirements & Safety Envelope

- **Inference, Puzzles & Unit Tests:** Any modern 64-bit multi-core CPU, 4 GB RAM. (GPU not required).
- **Full Self-Play RL & Supervised Training:** NVIDIA GPU with $\ge 6$ GB VRAM (tested on NVIDIA GeForce RTX 4060 Laptop GPU, 8 GB VRAM, CUDA 12.6, PyTorch 2.13).
- **Memory Safety Guardrails:** ZeroPawn incorporates an active daemon watchdog monitoring host RAM:
  - Process RSS ceiling: `< 7.5 GB`
  - System free RAM floor: `> 2.0 GB`
  - Automatic emergency checkpoint save and graceful exit on threshold violation.

---

## 🏷️ Brand Identity & Legacy Compatibility

ZeroPawn was formerly known as `chess-rl`. Full backwards compatibility is preserved:
- **Canonical Launcher:** `zeropawn.cmd` (or `zeropawn <command>` when installed via `pip`).
- **Compatibility Forwarder:** `chess.cmd` (forwards all arguments to `zeropawn.cmd` with a deprecation notice).
- **Python Module:** Compiled native module is `zeropawn_cpp`. A transparent forwarding wrapper `chessrl_cpp.py` is maintained.
- **Environment Variables:** `ZEROPAWN_*` variables are primary; `CHESSRL_*` variables are accepted as fallback.

---

## 🤝 Contributing & Community

Contributions are welcome! Please review:
- [**Contributing Guidelines**](CONTRIBUTING.md) — Toolchain setup, coding style, and PR standards.
- [**Code of Conduct**](CODE_OF_CONDUCT.md) — Contributor Covenant v2.1 community pledge.
- [**Security Policy**](SECURITY.md) — Responsible vulnerability reporting guidelines.
- [**Citation Metadata**](CITATION.cff) — BibTeX and CFF citation formats.

---

## 📜 Licensing & Third-Party Notices

ZeroPawn's original source code is licensed under the [**MIT License**](LICENSE).

ZeroPawn incorporates and interfaces with open-source third-party software. Please review [**THIRD_PARTY_NOTICES.md**](THIRD_PARTY_NOTICES.md) for full licensing details and attributions:
- **chess-library** (MIT License, Copyright © Disservin, Niklas Fiekas) — Vendored in `cpp/vendor/chess-library/`.
- **python-chess** (GNU GPLv3, Copyright © Niklas Fiekas) — Runtime dependency. Notice: combined binary packaging or distribution is subject to GNU GPLv3 terms.
- **pybind11** (BSD 3-Clause, Copyright © Wenzel Jakob) — Compile-time header dependency.
- **Stockfish** (GNU GPLv3, Copyright © Stockfish Developers) — External engine invoked via standard UCI subprocess pipes.
