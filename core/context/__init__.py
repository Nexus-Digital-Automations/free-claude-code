"""Context-optimization integration for the proxy.

Bridges the standalone `context_optimizer` package to the proxy's Anthropic
Pydantic models. Lives under `core/` because compaction is shared request-path
logic, not provider-specific (mirrors `core/anthropic/`).
"""

from .optimizer_adapter import ContextOptimizer

__all__ = ["ContextOptimizer"]
