"""Build the ZeroPawn wheel: stage native extension + data files, then PEP 517 build.

Audit S6-01: the historical wheel shipped only ``python*`` with no native
extension, no checkpoint registry, and no presets, so nothing engine-related
worked in an installed context. This script is the documented build step that
makes the wheel functional:

1. **Stage the native extension** — COPY (never move, never modify) the built
   pyd from ``build/cpp/`` into ``python/_native/``, together with the MinGW
   runtime DLL if present. ``python/_native/`` is gitignored; the binary is a
   build artifact, not source.
2. **Stage data files** — COPY ``docs/checkpoint_registry.json`` and
   ``configs/presets.json`` into ``python/_data/`` (also declared as
   ``[tool.setuptools.package-data]`` so direct ``python -m build`` without
   this script still ships the tracked copies).
3. **Build the wheel** via ``python -m build --wheel --no-isolation``, with
   setuptools' intermediate build base redirected OUT of ``build/`` (that
   name is a junction to the production build tree in remediation worktrees,
   and a setuptools build would otherwise create ``build/lib`` inside it).

Usage::

    python scripts/build_wheel.py [--output-dir dist] [--skip-native] [--check-only]

``--check-only`` stages nothing and only verifies the expected sources exist.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
NATIVE_DIR = REPO_ROOT / "python" / "_native"
DATA_DIR = REPO_ROOT / "python" / "_data"

# Sources are read from the production build tree (read-only from this lane's
# perspective). The canonical deployed extension is zeropawn_cpp.cp312-*.pyd;
# legacy chessrl_cpp*.pyd / backup files are ignored.
NATIVE_SOURCE_DIR = REPO_ROOT / "build" / "cpp"
NATIVE_SOURCE_GLOB = "zeropawn_cpp.cp*.pyd"
NATIVE_RUNTIME_DLLS = ("libwinpthread-1.dll",)

DATA_SOURCES = (
    (REPO_ROOT / "docs" / "checkpoint_registry.json", "checkpoint_registry.json"),
    (REPO_ROOT / "configs" / "presets.json", "presets.json"),
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def stage_native() -> list[Path]:
    """Copy the built native extension (+ runtime DLLs) into python/_native/."""
    if not NATIVE_SOURCE_DIR.is_dir():
        raise SystemExit(
            f"Native build directory not found: {NATIVE_SOURCE_DIR}. "
            "Build the C++ extension first (e.g. zeropawn.cmd build python), "
            "or pass --skip-native for a library-only wheel."
        )
    pyds = sorted(NATIVE_SOURCE_DIR.glob(NATIVE_SOURCE_GLOB))
    if not pyds:
        raise SystemExit(
            f"No {NATIVE_SOURCE_GLOB} found in {NATIVE_SOURCE_DIR}; build the "
            "C++ extension first, or pass --skip-native."
        )
    NATIVE_DIR.mkdir(parents=True, exist_ok=True)
    staged: list[Path] = []
    for src in pyds[:1]:  # exactly one canonical pyd
        dst = NATIVE_DIR / src.name
        shutil.copy2(src, dst)  # copy, never move
        print(f"[stage-native] {src} -> {dst}")
        print(f"[stage-native]   sha256 {sha256(dst)}  size {dst.stat().st_size} bytes")
        staged.append(dst)
    for dll_name in NATIVE_RUNTIME_DLLS:
        dll_src = NATIVE_SOURCE_DIR / dll_name
        if dll_src.is_file():
            dll_dst = NATIVE_DIR / dll_name
            shutil.copy2(dll_src, dll_dst)
            print(f"[stage-native] {dll_src} -> {dll_dst}")
            staged.append(dll_dst)
    return staged


def stage_data() -> list[Path]:
    """Copy the tracked registry + presets into python/_data/."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    staged: list[Path] = []
    for src, name in DATA_SOURCES:
        if not src.is_file():
            raise SystemExit(f"Data source not found: {src}")
        dst = DATA_DIR / name
        shutil.copy2(src, dst)
        print(f"[stage-data] {src} -> {dst}")
        staged.append(dst)
    return staged


def check_sources(native: bool) -> int:
    """Verify the staging sources exist; report what a build would stage."""
    ok = True
    for src, _ in DATA_SOURCES:
        exists = src.is_file()
        print(f"[check] {'OK ' if exists else 'MISSING'} {src}")
        ok &= exists
    if native:
        pyds = (
            sorted(NATIVE_SOURCE_DIR.glob(NATIVE_SOURCE_GLOB)) if NATIVE_SOURCE_DIR.is_dir() else []
        )
        print(f"[check] {'OK ' if pyds else 'MISSING'} {NATIVE_SOURCE_DIR / NATIVE_SOURCE_GLOB}")
        ok &= bool(pyds)
    return 0 if ok else 1


def inject_root_extension(wheel_path: Path) -> None:
    """Also place the native extension at site-packages root via ``<name>.data``.

    The staged copy under ``python/_native/`` is importable once ``import
    python`` has run (see the bootstrap in ``python/__init__.py``), but the
    installed-wheel contract is that plain ``import zeropawn_cpp`` works with
    no prior import. Files under ``<name>.data/platlib/`` are installed into
    the environment's platlib root (site-packages), where a pyd named
    ``zeropawn_cpp.cp312-win_amd64.pyd`` IS a top-level importable extension.
    The runtime DLL sits beside it, found via the altered search path used for
    extension imports. RECORD is regenerated so pip's hash verification passes.
    """
    import base64
    import csv
    import hashlib
    import io
    import zipfile

    stem = wheel_path.name[: -len(".whl")]  # zeropawn-0.1.0-cp312-cp312-win_amd64
    parts = stem.split("-")
    dist = f"{parts[0]}-{parts[1]}"  # zeropawn-0.1.0
    data_prefix = f"{dist}.data/platlib/"
    record_name = f"{dist}.dist-info/RECORD"

    payloads: list[tuple[str, bytes]] = []
    for staged in sorted(NATIVE_DIR.glob("zeropawn_cpp.cp*.pyd"))[:1] + [
        NATIVE_DIR / dll for dll in NATIVE_RUNTIME_DLLS if (NATIVE_DIR / dll).is_file()
    ]:
        payloads.append((data_prefix + staged.name, staged.read_bytes()))
    if not payloads:
        print("[inject] no staged native extension found; skipping root injection")
        return

    def digest_row(data: bytes) -> str:
        b64 = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
        return f"sha256={b64}"

    with zipfile.ZipFile(wheel_path, "r") as zin:
        entries = {info.filename: zin.read(info.filename) for info in zin.infolist()}
        record_text = entries[record_name].decode("utf-8")
        rows = list(csv.reader(io.StringIO(record_text)))
        record_self_row = [r for r in rows if r and r[0] == record_name]
        keep = [
            r for r in rows if r and r[0] != record_name and r[0] not in {p for p, _ in payloads}
        ]

    for path, data in payloads:
        keep.append([path, digest_row(data), str(len(data))])
    buf = io.StringIO()
    csv.writer(buf, lineterminator="\n").writerows(keep + record_self_row)
    entries[record_name] = buf.getvalue().encode("utf-8")
    for path, data in payloads:
        entries[path] = data

    tmp = wheel_path.with_suffix(".whl.tmp")
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zout:
        for name, data in entries.items():
            zout.writestr(name, data)
    tmp.replace(wheel_path)
    for path, data in payloads:
        print(f"[inject] {path} ({len(data)} bytes; sha256 {hashlib.sha256(data).hexdigest()})")


def build_wheel(output_dir: Path) -> Path:
    """Run ``python -m build --wheel --no-isolation`` with a redirected build base."""
    output_dir.mkdir(parents=True, exist_ok=True)
    # Redirect setuptools' intermediate build dir: `build` is a junction to the
    # production build tree in remediation worktrees, and a default build would
    # create build/lib inside it. DIST_EXTRA_CONFIG is honored by setuptools'
    # config file parsing (verified: intermediate dirs land in build_base).
    with tempfile.TemporaryDirectory(prefix="zeropawn-wheel-build-") as tmp:
        build_base = Path(tmp) / "setuptools-build"
        cfg = Path(tmp) / "dist-extra.cfg"
        cfg.write_text(f"[build]\nbuild_base = {build_base.as_posix()}\n", encoding="utf-8")
        env = dict(os.environ)
        env["DIST_EXTRA_CONFIG"] = str(cfg)
        cmd = [
            sys.executable,
            "-m",
            "build",
            "--wheel",
            "--no-isolation",
            "--outdir",
            str(output_dir),
        ]
        print(f"[build] {' '.join(cmd)}  (build_base={build_base})")
        subprocess.run(cmd, cwd=REPO_ROOT, env=env, check=True)
    wheels = sorted(output_dir.glob("zeropawn-*.whl"))
    if not wheels:
        raise SystemExit("Wheel build reported success but no zeropawn-*.whl was produced.")
    return wheels[-1]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "dist")
    parser.add_argument(
        "--skip-native",
        action="store_true",
        help="Do not stage the native extension (library-only wheel; engine will not load).",
    )
    parser.add_argument(
        "--check-only", action="store_true", help="Verify staging sources exist and exit."
    )
    args = parser.parse_args(argv)

    if args.check_only:
        return check_sources(native=not args.skip_native)

    if not args.skip_native:
        stage_native()
    stage_data()
    wheel = build_wheel(args.output_dir)
    if not args.skip_native:
        inject_root_extension(wheel)
    print(f"[done] {wheel}")
    print(f"[done]   sha256 {sha256(wheel)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
