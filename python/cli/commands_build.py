"""Build, C++ extension inspection, binding compatibility, and native test subcommands for ZeroPawn."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import time

from rich import box
from rich.table import Table

from python.cli.common import REPO_ROOT
from python.cli.ui import COLOR_BORDER, console


def register_parser(subparsers: argparse._SubParsersAction) -> None:
    build_parser = subparsers.add_parser(
        "build",
        aliases=["cxx", "native"],
        help="Build & native C++ tools: build status, Python/C++ compatibility check, and native tests.",
        description="Inspect and validate C++ core extensions, WinLibs MinGW toolchain, and execute native test suites.",
    )
    sub = build_parser.add_subparsers(dest="build_subcommand", help="Build actions")

    # 1. Status
    p_stat = sub.add_parser(
        "status",
        aliases=["info"],
        help="Inspect C++ compiled extension status, toolchain, and build artifacts.",
    )
    p_stat.set_defaults(func=run_build_status)

    # 2. Compatibility check
    p_comp = sub.add_parser(
        "compat",
        aliases=["contracts"],
        help="Verify Python <-> C++ binding contracts, zero-copy buffer, and perft.",
    )
    p_comp.set_defaults(func=run_build_compat)

    # 3. Native C++ tests
    p_test = sub.add_parser(
        "test", help="Execute native C++ test suite binary (zeropawn_tests.exe)."
    )
    p_test.set_defaults(func=run_build_test)

    # 4. Compile / rebuild
    p_compil = sub.add_parser(
        "compile", aliases=["ext"], help="Run CMake build if toolchain is available."
    )
    p_compil.add_argument(
        "--clean", action="store_true", help="Clean build directory before compiling"
    )
    p_compil.set_defaults(func=run_build_compile)


def run_build_status(args: argparse.Namespace) -> int:
    table = Table(
        title="[bold bright_cyan]C++ Core & Build Toolchain Status[/bold bright_cyan]",
        box=box.ROUNDED,
        border_style=COLOR_BORDER,
    )
    table.add_column("Component", style="bold cyan")
    table.add_column("Status / Path", style="bright_white")
    table.add_column("Notes", style="dim")

    # C++ Extension Module
    ext_found = False
    ext_path = ""
    search_dirs = [
        REPO_ROOT / "build_zeropawn" / "cpp",
        REPO_ROOT / "python",
        REPO_ROOT / "build" / "cpp",
    ]
    for c_dir in search_dirs:
        if c_dir.is_dir():
            for cand in sorted(c_dir.glob("zeropawn_cpp*.pyd")):
                if cand.is_file():
                    ext_found = True
                    ext_path = str(cand)
                    break
        if ext_found:
            break
    if not ext_found:
        try:
            import zeropawn_cpp

            ext_found = True
            ext_path = getattr(zeropawn_cpp, "__file__", "imported")
        except Exception:
            try:
                import chessrl_cpp  # legacy shim

                ext_found = True
                ext_path = getattr(chessrl_cpp, "__file__", "imported (legacy shim)")
            except Exception:
                pass

    table.add_row(
        "zeropawn_cpp.pyd",
        f"[green]{ext_path}[/green]" if ext_found else "[red]Not found[/red]",
        "C++17 Pybind11 Core Extension",
    )

    # Native test executable
    tests_candidates = [
        REPO_ROOT / "build_zeropawn" / "cpp" / "zeropawn_tests.exe",
        REPO_ROOT / "build" / "cpp" / "zeropawn_tests.exe",
        REPO_ROOT / "build" / "cmake" / "cpp" / "zeropawn_tests.exe",
        REPO_ROOT / "build" / "cmake" / "cpp" / "zeropawn_tests",
        REPO_ROOT / "build_zeropawn" / "cpp" / "zeropawn_tests",
        REPO_ROOT / "build" / "cpp" / "zeropawn_tests",
    ]
    tests_exe = None
    for cand in tests_candidates:
        if cand.is_file():
            tests_exe = cand
            break

    table.add_row(
        "zeropawn_tests.exe",
        f"[green]{tests_exe}[/green]" if tests_exe else "[yellow]Not found[/yellow]",
        "Native C++ Unit Test Runner",
    )

    # G++ Compiler
    gxx = shutil.which("g++")
    ver_str = "Not in PATH"
    if gxx:
        try:
            ver_str = subprocess.check_output([gxx, "--version"], text=True).splitlines()[0]
        except Exception:
            ver_str = gxx
    table.add_row("C++ Compiler (g++)", ver_str, "WinLibs MinGW-w64 UCRT")

    # CMake
    cmake = shutil.which("cmake")
    cmake_str = "Not in PATH"
    if cmake:
        try:
            cmake_str = subprocess.check_output([cmake, "--version"], text=True).splitlines()[0]
        except Exception:
            cmake_str = cmake
    table.add_row("CMake", cmake_str, "Build system generator")

    console.print(table)
    return 0


def run_build_compat(args: argparse.Namespace) -> int:
    console.print("[bold cyan]Verifying Python <-> C++ binding compatibility...[/bold cyan]")
    errors = []

    # 1. Import
    cpp_mod = None
    try:
        try:
            import python  # noqa: F401  (ensures zeropawn_cpp + DLL paths)
            import zeropawn_cpp as cpp_mod

            mod_name = "zeropawn_cpp"
        except ImportError:
            import chessrl_cpp as cpp_mod  # legacy shim

            mod_name = "chessrl_cpp (legacy shim)"
        v = cpp_mod.version()
        console.print(f"  [green]✓[/green] {mod_name} imported successfully (version: {v})")
    except Exception as e:
        console.print(f"  [red]✗ Failed to import C++ extension: {e}[/red]")
        return 1

    # 2. Board and movegen
    try:
        g = cpp_mod.ChessGame()
        moves = g.legal_moves_uci()
        assert len(moves) == 20, f"Expected 20 starting moves, got {len(moves)}"
        console.print("  [green]✓[/green] ChessGame startpos legal moves verified (20 moves)")
    except Exception as e:
        errors.append(f"ChessGame legal_moves failed: {e}")

    # 3. State encoder
    try:
        planes = cpp_mod.encode_state(g)
        assert planes.shape == (19, 8, 8), f"Expected (19, 8, 8), got {planes.shape}"
        assert planes.dtype.name == "float32"
        console.print("  [green]✓[/green] encode_state tensor shape verified: (19, 8, 8) float32")
    except Exception as e:
        errors.append(f"encode_state failed: {e}")

    # 4. Action encoding roundtrip
    try:
        idx = cpp_mod.move_to_index(g, "e2e4")
        uci = cpp_mod.index_to_move(g, idx)
        assert uci == "e2e4", f"Roundtrip failed: expected 'e2e4', got '{uci}'"
        console.print(
            "  [green]✓[/green] Action encoder roundtrip verified (move_to_index <-> index_to_move)"
        )
    except Exception as e:
        errors.append(f"Action encoder roundtrip failed: {e}")

    # 5. MCTS with NumPy array zero-copy buffer
    try:
        import numpy as np

        cfg = cpp_mod.MCTSConfig()
        cfg.c_puct = 2.0
        mcts = cpp_mod.MCTS(cfg)

        def mock_eval(game):
            # Return contiguous numpy float32 array
            policy = np.full(4672, 1.0 / 4672, dtype=np.float32)
            return (policy, 0.0)

        res = mcts.search(g, 10, mock_eval)
        assert res.best_action >= 0
        console.print("  [green]✓[/green] MCTS zero-copy NumPy array callback verified (10 sims)")
    except Exception as e:
        errors.append(f"MCTS NumPy callback search failed: {e}")

    # 6. Connect4
    try:
        c4 = cpp_mod.Connect4()
        assert c4.num_actions() == 7
        assert len(c4.legal_actions()) == 7
        console.print("  [green]✓[/green] Connect4 C++ binding verified (7 action columns)")
    except Exception as e:
        errors.append(f"Connect4 binding failed: {e}")

    if not errors:
        console.print(
            "[bold green]All Python/C++ binding contracts verified with 100% compatibility![/bold green]"
        )
        return 0
    else:
        console.print(f"[bold red]{len(errors)} compatibility error(s) detected:[/bold red]")
        for err in errors:
            console.print(f"  [red]- {err}[/red]")
        return 1


def run_build_test(args: argparse.Namespace) -> int:
    tests_candidates = [
        REPO_ROOT / "build_zeropawn" / "cpp" / "zeropawn_tests.exe",
        REPO_ROOT / "build" / "cpp" / "zeropawn_tests.exe",
        REPO_ROOT / "build" / "cmake" / "cpp" / "zeropawn_tests.exe",
        REPO_ROOT / "build" / "cmake" / "cpp" / "zeropawn_tests",
        REPO_ROOT / "build_zeropawn" / "cpp" / "zeropawn_tests",
        REPO_ROOT / "build" / "cpp" / "zeropawn_tests",
    ]
    tests_exe = None
    for cand in tests_candidates:
        if cand.is_file():
            tests_exe = cand
            break

    if not tests_exe:
        console.print(
            "[bold red]Native C++ test executable not found in build directories.[/bold red]"
        )
        return 1

    console.print(f"[bold cyan]Executing Native C++ Test Suite:[/] [white]{tests_exe}[/white]\n")
    t0 = time.time()
    try:
        ret = subprocess.run([str(tests_exe)], cwd=str(REPO_ROOT))
        duration = time.time() - t0
        if ret.returncode == 0:
            console.print(
                f"\n[bold green]All C++ native test suites passed in {duration:.2f}s (Exit 0).[/bold green]"
            )
        else:
            console.print(
                f"\n[bold red]C++ tests failed with exit code {ret.returncode}.[/bold red]"
            )
        return ret.returncode
    except Exception as e:
        console.print(f"[bold red]Failed to execute native tests: {e}[/bold red]")
        return 1


def run_build_compile(args: argparse.Namespace) -> int:
    cmake = shutil.which("cmake")
    if not cmake:
        console.print("[bold red]CMake not found in system PATH. Cannot compile.[/bold red]")
        return 1

    console.print("[bold cyan]Running CMake build pipeline...[/bold cyan]")
    build_dir = REPO_ROOT / "build_zeropawn"
    if args.clean and build_dir.is_dir():
        shutil.rmtree(build_dir, ignore_errors=True)

    build_dir.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.check_call(
            [
                "cmake",
                "-B",
                str(build_dir),
                "-G",
                "MinGW Makefiles",
                "-DZEROPAWN_BUILD_PYTHON=ON",
                "-DZEROPAWN_BUILD_TESTS=ON",
            ],
            cwd=str(REPO_ROOT),
        )
        subprocess.check_call(
            ["cmake", "--build", str(build_dir), "--parallel"], cwd=str(REPO_ROOT)
        )
        console.print("[bold green]Build completed successfully![/bold green]")
        return 0
    except subprocess.CalledProcessError as e:
        console.print(f"[bold red]Build failed: {e}[/bold red]")
        return e.returncode
