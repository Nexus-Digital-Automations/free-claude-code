"""context-optimizer — standalone LLM conversation compaction library.

pip install:
  pip install git+https://github.com/Nexus-Digital-Automations/free-claude-code.git#subdirectory=packages/context-optimizer

Quickstart:
    from context_optimizer import ContextOptimizer, ContextOptimizerSettings

    settings = ContextOptimizerSettings(
        compact_threshold_tokens=65_000,  # cold-start emergency seal threshold
        ollama_model="qwen2.5:7b",
    )

    new_messages, new_system, token_count = await ContextOptimizer.optimize(
        messages=messages,       # list[dict] in Anthropic/OpenAI format
        system="You are...",
        settings=settings,
    )

All conversation-level compaction is handled by the local Ollama-driven
block tower (Layer 0). The legacy `llm_provider` parameter is preserved
on the optimize() signature for backward compatibility but is unused.
"""

from .ollama_supervisor import OllamaSupervisor
from .optimizer import ContextOptimizer
from .settings import ContextOptimizerSettings

__all__ = ["ContextOptimizer", "ContextOptimizerSettings", "OllamaSupervisor"]
__version__ = "0.2.0"
