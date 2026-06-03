"""memory-recall — MCP server exposing cross-session search over plan and
spec markdown files.

V1 backend: SQLite FTS5 over the raw markdown body + frontmatter title and
status. Embeddings/semantic search are intentionally out of scope; FTS5 is
~300x simpler, has zero ML deps, and works well for the actual corpus
(dozens to a few hundred small markdown files).

Public surface: `cli()` in server.py.
"""
