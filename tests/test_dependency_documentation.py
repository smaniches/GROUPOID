"""Documented NumPy/SciPy upper bounds match the package metadata.

README.md, REPRODUCE.md and LIMITATIONS.md stated ``scipy<1.16`` after
``pyproject.toml`` had moved to ``scipy>=1.10,<1.18`` (CORRECTION_NOTICE.md,
F06). In these documents a specifier-style bound (``scipy<X`` or
``scipy>=A,<X``) states the current requirement, so it must equal the one the
installer enforces; superseded requirements are described in prose.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import tomllib

ROOT = Path(__file__).resolve().parents[1]
DOCUMENTS = ("README.md", "REPRODUCE.md", "LIMITATIONS.md")


def _metadata_upper_bound(package: str) -> str:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    requirements = [d for d in project["dependencies"] if re.match(rf"{package}\b", d)]
    assert len(requirements) == 1, requirements
    bounds = re.findall(r"<\s*([0-9][0-9.]*)", requirements[0])
    assert len(bounds) == 1, requirements[0]
    return str(bounds[0])


@pytest.mark.parametrize("package", ["numpy", "scipy"])
def test_documented_upper_bounds_match_pyproject(package: str) -> None:
    expected = _metadata_upper_bound(package)
    pattern = re.compile(rf"\b{package}\s*(?:>=\s*[0-9][0-9.]*\s*,\s*)?<\s*([0-9][0-9.]*)")
    found: dict[str, list[str]] = {}
    for name in DOCUMENTS:
        for match in pattern.finditer((ROOT / name).read_text()):
            found.setdefault(match.group(1), []).append(name)
    # README.md must state the bound, so this check cannot pass vacuously.
    assert "README.md" in found.get(expected, []), found
    assert set(found) == {expected}, found
