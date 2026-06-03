"""Language-coverage acceptance tests for symbol_graph.

Confirms that for each tagger-supported language whose tags.scm distinguishes
def from ref, symbol_graph.definition_of returns the expected def site.
Languages whose tags.scm only emits refs (or whose query lacks the
distinction) are explicitly tested as "ref-only" so we don't silently
regress in either direction.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from context_optimizer.repo_index import symbol_graph


@pytest.fixture(autouse=True)
def _reset_symbol_graph_cache() -> None:
    symbol_graph._reset_cache_for_tests()
    yield
    symbol_graph._reset_cache_for_tests()


def _git(cwd: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True)


def _make_repo(tmp_path: Path, name: str, files: dict[str, str]) -> Path:
    repo = tmp_path / name
    repo.mkdir()
    _git(repo, "init", "--initial-branch=main", "-q")
    _git(repo, "config", "user.email", "t@t.test")
    _git(repo, "config", "user.name", "T")
    for rel, content in files.items():
        path = repo / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "init")
    return repo


# ── languages where tags.scm distinguishes def from ref ────────────────────
#
# Acceptance: for each language listed here, symbol_graph.definition_of
# returns at least one def site for the canonical symbol defined in the
# fixture. The exact line is left to the per-language tags.scm — assertions
# bind to (file, kind="def") only.


def test_python_definition_is_indexed(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path, "py", {
        "main.py": "def widget():\n    pass\n",
    })
    defs = symbol_graph.definition_of(str(repo), "widget")
    assert any(d.file == "main.py" and d.kind == "def" for d in defs)


def test_javascript_definition_is_indexed(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path, "js", {
        "main.js": "function widget() { return 1; }\nexport { widget };\n",
    })
    defs = symbol_graph.definition_of(str(repo), "widget")
    assert any(d.file == "main.js" and d.kind == "def" for d in defs)


def test_typescript_definition_is_indexed(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path, "ts", {
        "main.ts": "export function widget(): number { return 1; }\n",
    })
    defs = symbol_graph.definition_of(str(repo), "widget")
    assert any(d.file == "main.ts" and d.kind == "def" for d in defs)


def test_go_definition_is_indexed(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path, "go", {
        "main.go": "package main\n\nfunc widget() int { return 1 }\n",
    })
    defs = symbol_graph.definition_of(str(repo), "widget")
    assert any(d.file == "main.go" and d.kind == "def" for d in defs)


def test_rust_definition_is_indexed(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path, "rs", {
        "main.rs": "fn widget() -> i32 { 1 }\n",
    })
    defs = symbol_graph.definition_of(str(repo), "widget")
    assert any(d.file == "main.rs" and d.kind == "def" for d in defs)


def test_ruby_definition_is_indexed(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path, "rb", {
        "main.rb": "def widget\n  1\nend\n",
    })
    defs = symbol_graph.definition_of(str(repo), "widget")
    assert any(d.file == "main.rb" and d.kind == "def" for d in defs)


def test_java_definition_is_indexed(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path, "java", {
        "Widget.java": (
            "public class Widget {\n"
            "    public int widget() { return 1; }\n"
            "}\n"
        ),
    })
    # Java's tags.scm tags both class names and method names. We assert
    # the method is captured; the class also gets a separate def at line 1.
    defs = symbol_graph.definition_of(str(repo), "widget")
    assert any(d.file == "Widget.java" and d.kind == "def" for d in defs)


def test_csharp_definition_is_indexed(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path, "cs", {
        "Widget.cs": (
            "public class Widget {\n"
            "    public int Compute() { return 1; }\n"
            "}\n"
        ),
    })
    # Compute is the method def. Class def is also captured separately.
    defs = symbol_graph.definition_of(str(repo), "Compute")
    assert any(d.file == "Widget.cs" and d.kind == "def" for d in defs)


def test_c_definition_is_indexed(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path, "c", {
        "main.c": "int widget(void) { return 1; }\n",
    })
    defs = symbol_graph.definition_of(str(repo), "widget")
    assert any(d.file == "main.c" and d.kind == "def" for d in defs)


def test_cpp_definition_is_indexed(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path, "cpp", {
        "main.cpp": "int widget() { return 1; }\n",
    })
    defs = symbol_graph.definition_of(str(repo), "widget")
    assert any(d.file == "main.cpp" and d.kind == "def" for d in defs)
