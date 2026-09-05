"""Every third-party import in the tree must be a declared dependency.

This exists because five runtime dependencies were undeclared for months, and
nothing caught it: `openai-whisper`, `ffmpeg-python`, `librosa`, `loguru` and
`rotary-embedding-torch`. The last two are imported by the frozen backbone
package, which was itself undeclared. The failure mode is quiet, since an
environment that happens to have them works fine, so it needs a test rather
than a review habit. See docs/restoration/ISSUE_LEDGER.md I-019 and I-020.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

try:
    import tomllib
except ModuleNotFoundError:  # Python < 3.11; project supports 3.10 (pyproject.toml)
    import tomli as tomllib

PROJECT_ROOT = Path(__file__).resolve().parent.parent

SKIP_DIRS = {".git", ".restoration", "build", "dist", "notebooks", "__pycache__", ".venv", "venv"}

# Distribution name -> the module names it provides, where they differ.
DISTRIBUTION_MODULES: dict[str, set[str]] = {
    "pyyaml": {"yaml"},
    "openai-whisper": {"whisper"},
    "ffmpeg-python": {"ffmpeg"},
    "rotary-embedding-torch": {"rotary_embedding_torch"},
    "sr-corrnet-ss": {"sr_corrnet"},
    "huggingface_hub": {"huggingface_hub"},
    "pytest-cov": {"pytest_cov"},
    "pre-commit": {"pre_commit"},
    "scikit-learn": {"sklearn"},
}

# Imported only under a guard, with a documented fallback when absent.
OPTIONAL_WITH_FALLBACK: set[str] = {
    "silero_vad",  # models/condition.py falls back to an energy VAD
    "onnxruntime",  # eval/dnsmos.py degrades when the scorer is unavailable
    "pesq",  # eval/pesq_metric.py is skipped when absent
    "sklearn",  # calibration falls back to a closed-form fit
    # Stdlib on Python 3.11+; on 3.10 (the project's floor) sys.stdlib_module_names
    # does not include it, and the declared fallback is "tomli", not "tomllib",
    # so the plain stdlib/declared checks below never see this name as covered.
    "tomllib",
}


def _first_party_names() -> set[str]:
    """Top-level names that resolve inside this repository."""
    names = {"tests", "conftest"}
    for entry in PROJECT_ROOT.iterdir():
        if entry.name in SKIP_DIRS:
            continue
        if entry.is_dir() and (entry / "__init__.py").exists():
            names.add(entry.name)
        elif entry.suffix == ".py":
            names.add(entry.stem)
    src = PROJECT_ROOT / "src"
    if src.is_dir():
        names.update(p.name for p in src.iterdir() if p.is_dir())
    return names


def _declared_modules() -> set[str]:
    """Every module name the project declares it depends on."""
    data = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    project = data["project"]
    requirements = list(project["dependencies"])
    for extra in project.get("optional-dependencies", {}).values():
        requirements.extend(extra)

    modules: set[str] = set()
    for raw in requirements:
        # Strip extras, direct references and version specifiers.
        name = raw.split("@")[0].split("[")[0]
        for sep in (">=", "<=", "==", "!=", "~=", ">", "<", ";"):
            name = name.split(sep)[0]
        name = name.strip().lower()
        if not name:
            continue
        modules.update(DISTRIBUTION_MODULES.get(name, {name.replace("-", "_")}))
    return modules


def _imported_top_level_modules() -> dict[str, set[str]]:
    """Map each imported top-level module to the files that import it."""
    found: dict[str, set[str]] = {}
    for path in PROJECT_ROOT.rglob("*.py"):
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:  # pragma: no cover - a parse failure is its own bug
            continue
        rel = str(path.relative_to(PROJECT_ROOT)).replace("\\", "/")
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    found.setdefault(alias.name.split(".")[0], set()).add(rel)
            elif isinstance(node, ast.ImportFrom):
                if node.level == 0 and node.module:
                    found.setdefault(node.module.split(".")[0], set()).add(rel)
    return found


def test_every_third_party_import_is_declared():
    first_party = _first_party_names()
    declared = _declared_modules()
    stdlib = sys.stdlib_module_names

    undeclared: dict[str, set[str]] = {}
    for module, files in _imported_top_level_modules().items():
        if module in stdlib or module in first_party or module in declared:
            continue
        if module in OPTIONAL_WITH_FALLBACK:
            continue
        undeclared[module] = files

    assert not undeclared, "undeclared third-party imports: " + "; ".join(
        f"{mod} (imported by {', '.join(sorted(files)[:3])})"
        for mod, files in sorted(undeclared.items())
    )


def test_the_backbone_loader_is_declared_with_a_pinned_commit():
    """An unpinned backbone would let a later revision move the patched attribute paths."""
    text = (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert "SR_CorrNet_SS" in text, "the frozen backbone loader must be a declared dependency"
    assert "@7340365b" in text, "the backbone must be pinned to a commit, not a branch"


@pytest.mark.parametrize("module", sorted(OPTIONAL_WITH_FALLBACK))
def test_optional_imports_are_guarded(module):
    """Anything on the fallback list must never be imported at module scope."""
    offenders = []
    for path in PROJECT_ROOT.rglob("*.py"):
        if any(part in SKIP_DIRS for part in path.parts) or path.parts[-2] == "tests":
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:  # pragma: no cover
            continue
        for node in tree.body:  # module scope only
            names = []
            if isinstance(node, ast.Import):
                names = [a.name.split(".")[0] for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names = [node.module.split(".")[0]]
            if module in names:
                offenders.append(str(path.relative_to(PROJECT_ROOT)))
    assert (
        not offenders
    ), f"{module} is declared optional but imported at module scope in {offenders}"
