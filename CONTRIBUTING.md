# Contributing to ZeroPawn

Thank you for your interest in contributing to **ZeroPawn**! This document provides guidelines for setting up your development environment, testing code changes, and submitting pull requests.

---

## 1. Development Setup

### Prerequisites
- **Python:** 3.11 or 3.12 (64-bit)
- **C++ Compiler:** C++17 compliant compiler:
  - **Windows:** MinGW-W64 GCC 11+ (recommended: WinLibs UCRT via `winget install BrechtSanders.WinLibs.POSIX.UCRT`) or MSVC 2019+
  - **Linux:** GCC 10+ or Clang 12+ (`sudo apt install build-essential ninja-build cmake`)
  - **macOS:** Xcode Command Line Tools (`xcode-select --install`)
- **CMake & Ninja:** CMake 3.20+ and Ninja build system
- **Stockfish:** Recommended for evaluation benchmarks (placed in `tools/` or available on PATH)

### Quick Start: Environment Setup

```bash
# Clone the repository
git clone https://github.com/MdHussain121/zeropawn.git
cd zeropawn

# Create and activate virtual environment
python -m venv .venv

# On Linux/macOS:
source .venv/bin/activate
# On Windows (PowerShell):
.venv\Scripts\Activate.ps1
# On Windows (cmd):
.venv\Scripts\activate.bat

# Install package in editable mode with development dependencies
pip install -e ".[dev]"
```

### Building the C++ Engine Extension

You can build the pybind11 module (`zeropawn_cpp`) and unit tests using CMake presets or standard commands:

```bash
# Using CMake Presets:
cmake --preset default
cmake --build --preset default

# Or manual configuration:
cmake -B build/cmake -S . -DCMAKE_BUILD_TYPE=Release
cmake --build build/cmake --config Release -j
```

---

## 2. Testing & Verification

Before submitting changes, run both the C++ and Python test suites:

### Running C++ Tests
```bash
ctest --test-dir build/cmake --output-on-failure
# Or directly run the binary:
./build/cmake/cpp/zeropawn_tests
```

### Running Python Tests
```bash
# Fast test suites
pytest -k "not slow and not soak"

# Comprehensive test suite
pytest tests/
```

### Code Formatting and Linting
We use [Ruff](https://github.com/astral-sh/ruff) for fast Python linting and formatting:
```bash
ruff check .
ruff format --check .
```

---

## 3. Pull Request Guidelines

1. **Keep Related Changes Synchronized:** If you update a CLI argument or model interface, make sure to update corresponding documentation, tests, preset definitions, and help strings.
2. **Never Commit Large Artifacts or Secrets:** Checkpoints, datasets, memory maps, and temporary logs should never be committed into version control. Ensure `.gitignore` covers any new generated files.
3. **Cross-Platform Compatibility:** Do not hard-code OS-specific paths or shell assumptions. Use `pathlib.Path` and test with paths containing spaces.
4. **Fill out the PR Template:** State what was changed, which tests were run, and any potential breaking changes.

---

## 4. Reporting Issues

- Use the GitHub Issue Tracker to report bugs or request features.
- Provide full system details (OS, Python version, GPU / CPU model, commit hash).
- Include minimal reproduction steps or code snippets.
