"""Acceptance tests for dep_graph language coverage beyond Python.

Two distinct tiers under test:
  - TypeScript: relative imports resolve to in-repo paths; bare specifiers stay unresolved.
  - Go:         raw_only — raw import strings are captured but not resolved.

Both prove the same thing the Python tests already prove (file-level edges
are correct), but for the two extractor paths that lacked coverage.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from context_optimizer.repo_index import dep_graph


@pytest.fixture(autouse=True)
def _reset_dep_graph_cache() -> None:
    dep_graph._reset_cache_for_tests()
    yield
    dep_graph._reset_cache_for_tests()


def _git(cwd: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True)


def _make_repo(tmp_path: Path, name: str) -> Path:
    repo = tmp_path / name
    repo.mkdir()
    _git(repo, "init", "--initial-branch=main", "-q")
    _git(repo, "config", "user.email", "t@t.test")
    _git(repo, "config", "user.name", "T")
    return repo


# ── TypeScript ─────────────────────────────────────────────────────────────


@pytest.fixture
def ts_repo(tmp_path: Path) -> Path:
    repo = _make_repo(tmp_path, "ts_fixture")
    (repo / "src").mkdir()
    (repo / "src" / "main.ts").write_text(
        'import { helper } from "./helper";\n'
        'import shared from "./shared";\n'
        'import React from "react";\n'
    )
    (repo / "src" / "helper.ts").write_text("export const helper = 1;\n")
    (repo / "src" / "shared.ts").write_text("export default 'shared';\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "init")
    return repo


def test_ts_relative_named_import_resolves_to_module_file(ts_repo: Path) -> None:
    edges = dep_graph.imports_of(str(ts_repo), "src/main.ts")
    by_raw = {e.raw: e for e in edges}
    assert "./helper" in by_raw
    assert by_raw["./helper"].resolved == "src/helper.ts"


def test_ts_relative_default_import_resolves_to_module_file(ts_repo: Path) -> None:
    edges = dep_graph.imports_of(str(ts_repo), "src/main.ts")
    by_raw = {e.raw: e for e in edges}
    assert "./shared" in by_raw
    assert by_raw["./shared"].resolved == "src/shared.ts"


def test_ts_bare_specifier_stays_unresolved(ts_repo: Path) -> None:
    edges = dep_graph.imports_of(str(ts_repo), "src/main.ts")
    by_raw = {e.raw: e for e in edges}
    assert "react" in by_raw
    assert by_raw["react"].resolved is None


def test_ts_imported_by_links_main_to_helper(ts_repo: Path) -> None:
    importers = dep_graph.imported_by(str(ts_repo), "src/helper.ts")
    assert importers == {"src/main.ts"}


# ── Go (raw-only) ──────────────────────────────────────────────────────────


@pytest.fixture
def go_repo(tmp_path: Path) -> Path:
    repo = _make_repo(tmp_path, "go_fixture")
    (repo / "main.go").write_text(
        "package main\n"
        "\n"
        "import (\n"
        '    "fmt"\n'
        '    "github.com/example/lib"\n'
        ")\n"
        "\n"
        "func main() { fmt.Println(lib.Hello) }\n"
    )
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "init")
    return repo


def test_go_imports_captured_with_raw_strings(go_repo: Path) -> None:
    edges = dep_graph.imports_of(str(go_repo), "main.go")
    raws = {e.raw for e in edges}
    assert "fmt" in raws
    assert "github.com/example/lib" in raws


def test_go_imports_remain_unresolved_under_raw_only_tier(go_repo: Path) -> None:
    edges = dep_graph.imports_of(str(go_repo), "main.go")
    for edge in edges:
        assert edge.resolved is None, (
            f"Go imports should be raw-only, got resolved={edge.resolved!r}"
        )
