# (C) Copyright IBM 2026
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Structural checks on the package source: every file parses, and files
inside the package use relative imports.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
PACKAGE_ROOT = REPO_ROOT / "src" / "pelagos"


def _python_files() -> list[Path]:
    files = sorted(PACKAGE_ROOT.rglob("*.py"))
    files.append(REPO_ROOT / "pelagos_benchmark.py")
    return [f for f in files if f.exists() and "__pycache__" not in f.parts]


#: `pelagos_benchmark.py` is a root script: a consumer of the installed
#: package, not part of it, so it imports `pelagos.*` absolutely. Only files
#: under `src/pelagos/` are package-internal and must use relative imports.
def _package_internal_files() -> list[Path]:
    return [f for f in _python_files() if PACKAGE_ROOT in f.parents]


def test_there_are_python_files_to_check():
    assert len(_python_files()) >= 9


@pytest.mark.parametrize("path", _python_files(), ids=lambda p: p.name)
def test_file_parses(path: Path):
    ast.parse(path.read_text())


@pytest.mark.parametrize("path", _package_internal_files(), ids=lambda p: p.name)
def test_no_absolute_intra_package_import(path: Path):
    """Package-internal files (under ``src/pelagos/``) must use relative
    imports. Root scripts and test files are consumers of the installed
    package and are exempt.
    """
    for lineno, line in enumerate(path.read_text().splitlines(), start=1):
        text = line.strip()
        if not text.startswith("from "):
            continue
        target = text[len("from ") :].split()[0]
        assert not target.startswith("pelagos."), (
            f"{path.name}:{lineno}: absolute intra-package import "
            f"'{target}' — relative imports must stay relative"
        )
