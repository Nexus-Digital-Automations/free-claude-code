"""context-optimizer — standalone LLM conversation compaction library.

pip install:
  pip install git+https://github.com/Nexus-Digital-Automations/free-claude-code.git#subdirectory=packages/context-optimizer

Quickstart:
    from context_optimizer import ContextOptimizer, ContextOptimizerSettings

    settings = ContextOptimizerSettings(
        compact_threshold_tokens=65_000,
        ollama_model="qwen2.5:7b",
    )

    async def my_llm(prompt: str) -> str:
        resp = await openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.choices[0].message.content

    new_messages, new_system, token_count = await ContextOptimizer.optimize(
        messages=messages,       # list[dict] in Anthropic/OpenAI format
        system="You are...",
        settings=settings,
        llm_provider=my_llm,    # used for Tier 2b sync compaction
    )
"""

from .ollama_supervisor import OllamaSupervisor
from .optimizer import ContextOptimizer
from .settings import ContextOptimizerSettings

__all__ = ["ContextOptimizer", "ContextOptimizerSettings", "OllamaSupervisor"]
__version__ = "0.1.0"
