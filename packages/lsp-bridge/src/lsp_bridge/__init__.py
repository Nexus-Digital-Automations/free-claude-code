"""lsp-bridge — MCP server exposing point-query type/hover and project-wide
symbol-definition lookup for Python.

V1 uses Jedi (pure-Python static analysis) instead of a subprocess language
server. Jedi gives us infer/help/get_signatures and project-wide search with
no LSP protocol surface to manage. TypeScript and other languages are
intentionally out of scope; adding them later means swapping in a real LSP
client and is the natural extension point.

Public surface: `cli()` in server.py.
"""
