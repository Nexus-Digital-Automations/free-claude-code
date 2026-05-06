"""test-digester — MCP server exposing a single tool that runs tests and
returns a failures-only summary, with each failure body digested through
context-optimizer's shared digest_core.

Public surface: `cli()` in server.py (the `test-digester` script entry point).
"""
