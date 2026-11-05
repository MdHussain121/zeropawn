"""THIRD_PARTY_NOTICES.md coverage regression (audit SA1-04).

Guards the third-party notice document against dependency drift:

1. every runtime dependency declared in ``pyproject.toml`` has a notice entry
   (naming normalized: case-insensitive, ``_``/``-`` equivalent, with an
   explicit alias map for package names whose notice heading differs);
2. the key vendored/redistributed native components (Disservin chess-library,
   pybind11, the redistributed MinGW-w64 ``libwinpthread-1.dll``) remain
   documented;
3. the document's numbered-section structure holds (sequential, unique);
4. the python-chess license wording stays GPL-3.0-or-later (pip metadata:
   ``GPL-3.0+``).

Purely file/metadata based — no network access.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
NOTICES_PATH = REPO_ROOT / "THIRD_PARTY_NOTICES.md"
PYPROJECT_PATH = REPO_ROOT / "pyproject.toml"

# Runtime dependency name (normalized) -> notice tokens that satisfy the
# requirement, lowercase; any one token match suffices. ``torch`` is noticed
# under the "PyTorch" heading, hence the explicit alias.
DEP_NOTICE_TOKENS: dict[str, tuple[str, ...]] = {
    "torch": ("pytorch",),
    "numpy": ("numpy",),
    "python-chess": ("python-chess",),
    "rich": ("rich",),
    "zstandard": ("zstandard",),
    "psutil": ("psutil",),
    "tensorboard": ("tensorboard",),
}

# Vendored / redistributed native components that must never lose their entry.
REQUIRED_NATIVE_COMPONENTS = (
    "chess-library",
    "pybind11",
    "libwinpthread-1.dll",
)


def _normalize(name: str) -> str:
    return name.lower().replace("_", "-")


def _requirement_name(requirement: str) -> str:
    """Extract the distribution name from a PEP 508 requirement string."""
    match = re.match(r"[A-Za-z0-9][A-Za-z0-9._-]*", requirement)
    assert match is not None, f"unparseable requirement: {requirement!r}"
    return _normalize(match.group(0))


def _runtime_dependency_names() -> list[str]:
    with PYPROJECT_PATH.open("rb") as fh:
        data = tomllib.load(fh)
    deps = data["project"]["dependencies"]
    names = [_requirement_name(req) for req in deps]
    assert names, "pyproject.toml declares no runtime dependencies"
    return names


def _notices_text() -> str:
    assert NOTICES_PATH.is_file(), "THIRD_PARTY_NOTICES.md missing from repo root"
    return NOTICES_PATH.read_text(encoding="utf-8")


@pytest.mark.parametrize("dep", sorted(DEP_NOTICE_TOKENS))
def test_runtime_dependency_has_notice_entry(dep: str) -> None:
    assert dep in _runtime_dependency_names(), f"stale token-map entry: {dep!r}"
    text = _notices_text().lower()
    tokens = DEP_NOTICE_TOKENS[dep]
    assert any(token in text for token in tokens), (
        f"runtime dependency {dep!r} has no THIRD_PARTY_NOTICES.md entry"
    )


def test_pyproject_runtime_dependencies_all_mapped() -> None:
    unmapped = [name for name in _runtime_dependency_names() if name not in DEP_NOTICE_TOKENS]
    assert not unmapped, (
        f"runtime dependencies added to pyproject.toml without a notice "
        f"mapping in test_third_party_notices.py: {unmapped}"
    )


@pytest.mark.parametrize("component", REQUIRED_NATIVE_COMPONENTS)
def test_vendored_native_components_documented(component: str) -> None:
    text = _notices_text().lower()
    assert component.lower() in text, f"missing notice for native component {component!r}"


def test_notices_numbered_sections_sequential_and_unique() -> None:
    text = _notices_text()
    assert text.startswith("# Third-Party Software Notices and Licenses")
    headings = re.findall(r"^## (\d+)\. .+$", text, flags=re.MULTILINE)
    assert headings, "no numbered '## N.' sections found in THIRD_PARTY_NOTICES.md"
    numbers = [int(n) for n in headings]
    assert numbers == sorted(numbers), "section numbers not ascending"
    assert len(numbers) == len(set(numbers)), "duplicate section numbers"
    assert numbers == list(range(1, len(numbers) + 1)), "section numbers not sequential"


def test_python_chess_license_is_gpl3_or_later() -> None:
    text = _notices_text().lower()
    assert "gpl-3.0-or-later" in text or "gplv3+" in text, (
        "python-chess notice should state GPL-3.0-or-later (pip metadata: GPL-3.0+)"
    )
