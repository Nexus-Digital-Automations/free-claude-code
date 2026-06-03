"""Acceptance tests for context_optimizer.repo_index.dep_graph.

Scope: dep_graph is a security-relevant integrity boundary — wrong import
edges would mislead the agent about which files are actually depended on,
which is exactly the failure mode the tool exists to prevent. So this
suite is in scope for the testing policy.

These tests use a real on-disk git repo (set up per-test via tmp_path)
because dep_graph's HEAD-sha cache reads through to git itself; mocking
git rev-parse would freeze a snapshot of one git version's behaviour.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from context_optimizer.repo_index import dep_graph


@pytest.fixture(autouse=True)
def _reset_dep_graph_cache() -> None:
    """Each test starts with a clean cache so HEAD-invalidation tests don't
    inherit stale graphs from earlier tests."""
    dep_graph._reset_cache_for_tests()
    yield
    dep_graph._reset_cache_for_tests()


@pytest.fixture
def python_repo(tmp_path: Path) -> Path:
    """A git repo with a small Python package exercising every import shape.

    Layout:
        pkg/__init__.py         (empty)
        pkg/main.py             (multiple import shapes)
        pkg/sibling.py          (target of `from . import sibling`)
        pkg/sub/__init__.py     (target of `from pkg import sub`)
        pkg/sub/helper.py       (target of `from pkg.sub import helper`)
    """
    repo = tmp_path / "py_fixture"
    repo.mkdir()
    _git(repo, "init", "--initial-branch=main", "-q")
    _git(repo, "config", "user.email", "t@t.test")
    _git(repo, "config", "user.name", "T")
    (repo / "pkg").mkdir()
    (repo / "pkg" / "sub").mkdir()
    (repo / "pkg" / "__init__.py").write_text("")
    (repo / "pkg" / "sub" / "__init__.py").write_text("")
    (repo / "pkg" / "sibling.py").write_text("X = 1\n")
    (repo / "pkg" / "sub" / "helper.py").write_text("Y = 2\n")
    (repo / "pkg" / "main.py").write_text(
        "from pkg.sub import helper\nfrom . import sibling\nimport os\n"
    )
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "init")
    return repo


def _git(cwd: Path, *args: str) -> None:
    subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    )


# ── imports_of ─────────────────────────────────────────────────────────────


def test_imports_of_resolves_absolute_submodule_to_module_file(
    python_repo: Path,
) -> None:
    edges = dep_graph.imports_of(str(python_repo), "pkg/main.py")
    by_raw = {e.raw: e for e in edges}
    assert "pkg.sub.helper" in by_raw
    assert by_raw["pkg.sub.helper"].resolved == "pkg/sub/helper.py"


def test_imports_of_resolves_relative_import_to_sibling_module(
    python_repo: Path,
) -> None:
    edges = dep_graph.imports_of(str(python_repo), "pkg/main.py")
    by_raw = {e.raw: e for e in edges}
    assert ".sibling" in by_raw
    assert by_raw[".sibling"].resolved == "pkg/sibling.py"


def test_imports_of_leaves_third_party_imports_unresolved(python_repo: Path) -> None:
    edges = dep_graph.imports_of(str(python_repo), "pkg/main.py")
    by_raw = {e.raw: e for e in edges}
    assert "os" in by_raw
    assert by_raw["os"].resolved is None


def test_imports_of_returns_empty_list_for_untracked_file(python_repo: Path) -> None:
    assert dep_graph.imports_of(str(python_repo), "nonexistent.py") == []


# ── imported_by ────────────────────────────────────────────────────────────


def test_imported_by_includes_main_for_helper(python_repo: Path) -> None:
    importers = dep_graph.imported_by(str(python_repo), "pkg/sub/helper.py")
    assert importers == {"pkg/main.py"}


def test_imported_by_includes_main_for_sibling(python_repo: Path) -> None:
    importers = dep_graph.imported_by(str(python_repo), "pkg/sibling.py")
    assert importers == {"pkg/main.py"}


def test_imported_by_returns_empty_set_for_leaf_file(python_repo: Path) -> None:
    assert dep_graph.imported_by(str(python_repo), "pkg/main.py") == set()


# ── unresolved_count ───────────────────────────────────────────────────────


def test_unresolved_count_matches_third_party_imports(python_repo: Path) -> None:
    # main.py imports os — one unresolved edge.
    assert dep_graph.unresolved_count(str(python_repo), "pkg/main.py") == 1


# ── language_of ────────────────────────────────────────────────────────────


def test_language_of_returns_python_for_py_file(python_repo: Path) -> None:
    assert dep_graph.language_of(str(python_repo), "pkg/main.py") == "python"


def test_language_of_returns_empty_string_for_unknown_path(python_repo: Path) -> None:
    assert dep_graph.language_of(str(python_repo), "missing.py") == ""


# ── caching + HEAD invalidation ────────────────────────────────────────────


def test_warm_cache_call_does_not_invoke_parse_repo(
    python_repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Cold call must run the parse pass; warm hit must skip it entirely.

    A counter on parse_repo is deterministic, unlike timing — it doesn't
    flake when the tree-sitter parser is already warm from a previous test.
    """
    from context_optimizer.repo_index import tagger

    real_parse_repo = tagger.parse_repo
    counter = {"calls": 0}

    def counting_parse_repo(*args: object, **kwargs: object) -> dict:
        counter["calls"] += 1
        return real_parse_repo(*args, **kwargs)

    monkeypatch.setattr(
        "context_optimizer.repo_index.dep_graph.tagger.parse_repo",
        counting_parse_repo,
    )

    dep_graph.imports_of(str(python_repo), "pkg/main.py")
    assert counter["calls"] == 1, "cold call should invoke parse_repo exactly once"
    dep_graph.imports_of(str(python_repo), "pkg/main.py")
    assert counter["calls"] == 1, "warm call must not re-parse"


def test_head_change_invalidates_and_rebuilds_graph(python_repo: Path) -> None:
    edges_before = dep_graph.imports_of(str(python_repo), "pkg/main.py")
    raws_before = {e.raw for e in edges_before}
    assert "json" not in raws_before

    (python_repo / "pkg" / "main.py").write_text(
        "from pkg.sub import helper\nfrom . import sibling\nimport os\nimport json\n"
    )
    _git(python_repo, "add", "pkg/main.py")
    _git(python_repo, "commit", "-q", "-m", "add json import")

    edges_after = dep_graph.imports_of(str(python_repo), "pkg/main.py")
    raws_after = {e.raw for e in edges_after}
    assert "json" in raws_after, "post-commit edges should reflect the new import"


def test_build_for_repo_raises_when_not_a_git_repo(tmp_path: Path) -> None:
    plain = tmp_path / "plain"
    plain.mkdir()
    (plain / "x.py").write_text("import os\n")
    with pytest.raises(RuntimeError, match="not a git repo"):
        dep_graph.build_for_repo(str(plain))
