"""lint-digester — MCP server that runs a linter and groups its findings
by rule, returning one Ollama-digested example per rule plus the file
list. Avoids the "one line per violation × N violations = K-token dump"
pattern that linters produce when a file has many issues.

Public surface: `cli()` in server.py.
"""
