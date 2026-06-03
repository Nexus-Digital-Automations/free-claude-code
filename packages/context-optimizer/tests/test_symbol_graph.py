"""Acceptance tests for context_optimizer.repo_index.symbol_graph.

Scope: symbol_graph is a security-relevant integrity boundary — wrong
def/ref attribution would mislead the agent about which call sites it's
about to edit, which is exactly the failure mode this tool exists to
prevent. So this suite is in scope for the testing policy.

Uses real on-disk git repos (per-test via tmp_path) because symbol_graph's
HEAD-sha cache reads through to git itself; mocking would freeze a
snapshot of one git version's behaviour.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from context_optimizer.repo_index import symbol_graph


@pytest.fixture(autouse=True)
def _reset_symbol_graph_cache() -> None:
    """Each test starts with a clean cache so HEAD-invalidation tests don't
    inherit stale graphs from earlier tests."""
    symbol_graph._reset_cache_for_tests()
    yield
    symbol_graph._reset_cache_for_tests()


def _git(cwd: Path, *args: str) -> None:
    subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    )


@pytest.fixture
def python_repo(tmp_path: Path) -> Path:
    """A git repo with two Python modules so we can test cross-file refs.

    Layout:
        pkg/__init__.py
        pkg/main.py     — defines `process`, calls `helper`
        pkg/util.py     — defines `helper`, calls `process`
    """
    repo = tmp_path / "py_fixture"
    repo.mkdir()
    _git(repo, "init", "--initial-branch=main", "-q")
    _git(repo, "config", "user.email", "t@t.test")
    _git(repo, "config", "user.name", "T")
    (repo / "pkg").mkdir()
    (repo / "pkg" / "__init__.py").write_text("")
    (repo / "pkg" / "main.py").write_text(
        "from pkg.util import helper\n\ndef process(x):\n    return helper(x)\n"
    )
    (repo / "pkg" / "util.py").write_text(
        "from pkg.main import process\n\ndef helper(x):\n    return process(x)\n"
    )
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "init")
    return repo


# ── definition_of ──────────────────────────────────────────────────────────


def test_definition_of_returns_def_site_for_top_level_function(
    python_repo: Path,
) -> None:
    defs = symbol_graph.definition_of(str(python_repo), "process")
    assert any(d.file == "pkg/main.py" and d.kind == "def" for d in defs)


def test_definition_of_returns_empty_for_unknown_name(python_repo: Path) -> None:
    assert symbol_graph.definition_of(str(python_repo), "nonexistent_symbol") == []


def test_definition_of_uses_one_based_line_numbers(python_repo: Path) -> None:
    """Tagger emits 0-based lines; symbol_graph normalises to 1-based to match
    git_blame and editors. process is on line 3 of pkg/main.py."""
    defs = symbol_graph.definition_of(str(python_repo), "process")
    main_def = next(d for d in defs if d.file == "pkg/main.py")
    assert main_def.line == 3


# ── references_to ──────────────────────────────────────────────────────────


def test_references_to_picks_up_cross_file_call_site(python_repo: Path) -> None:
    """`process` is called from pkg/util.py — references_to must find it."""
    refs = symbol_graph.references_to(str(python_repo), "process")
    files_referencing = {r.file for r in refs}
    assert "pkg/util.py" in files_referencing


def test_references_to_returns_empty_for_unknown_name(python_repo: Path) -> None:
    assert symbol_graph.references_to(str(python_repo), "nonexistent_symbol") == []


# ── caching + HEAD invalidation ────────────────────────────────────────────


def test_warm_cache_call_does_not_invoke_parse_repo(
    python_repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Cold call must run the parse pass; warm hit must skip it entirely.

    Counter-based assertion is deterministic — timing-based flakes when
    tree-sitter parser is already warm from earlier tests.
    """
    from context_optimizer.repo_index import tagger

    real_parse_repo = tagger.parse_repo
    counter = {"calls": 0}

    def counting_parse_repo(*args: object, **kwargs: object) -> dict:
        counter["calls"] += 1
        return real_parse_repo(*args, **kwargs)

    monkeypatch.setattr(
        "context_optimizer.repo_index.symbol_graph.tagger.parse_repo",
        counting_parse_repo,
    )

    symbol_graph.definition_of(str(python_repo), "process")
    assert counter["calls"] == 1, "cold call should invoke parse_repo exactly once"
    symbol_graph.definition_of(str(python_repo), "process")
    assert counter["calls"] == 1, "warm call must not re-parse"


def test_head_change_invalidates_and_rebuilds_graph(python_repo: Path) -> None:
    """A new commit that adds a definition should appear after the HEAD changes."""
    before = symbol_graph.definition_of(str(python_repo), "newly_added")
    assert before == []

    (python_repo / "pkg" / "main.py").write_text(
        "from pkg.util import helper\n"
        "\n"
        "def process(x):\n"
        "    return helper(x)\n"
        "\n"
        "def newly_added():\n"
        "    return 42\n"
    )
    _git(python_repo, "add", "pkg/main.py")
    _git(python_repo, "commit", "-q", "-m", "add newly_added")

    after = symbol_graph.definition_of(str(python_repo), "newly_added")
    assert any(d.file == "pkg/main.py" for d in after), (
        "post-commit graph should reflect the new definition"
    )


def test_build_for_repo_raises_when_not_a_git_repo(tmp_path: Path) -> None:
    plain = tmp_path / "plain"
    plain.mkdir()
    (plain / "x.py").write_text("def x(): pass\n")
    with pytest.raises(RuntimeError, match="not a git repo"):
        symbol_graph.build_for_repo(str(plain))
