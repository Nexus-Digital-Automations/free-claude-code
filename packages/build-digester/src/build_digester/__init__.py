"""build-digester — MCP server that runs a build/typecheck command in the
current project, parses compiler errors, and returns failures-only with
each error body digested through context-optimizer's digest_core.

Same architectural shape as test-digester: framework detection from cwd
markers, structured-output flags where the tool supports them, body
digestion above a byte threshold so cheap errors stay verbatim.

Public surface: `cli()` in server.py.
"""
