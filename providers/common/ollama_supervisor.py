"""Re-exports OllamaSupervisor from the context-optimizer package.

All supervisor logic now lives in packages/context-optimizer. This shim
keeps the existing import path (providers.common.ollama_supervisor) working
for api/app.py and any other proxy callers.

Counterpart: packages/context-optimizer/src/context_optimizer/ollama_supervisor.py
"""

from context_optimizer.ollama_supervisor import (  # noqa: F401
    OllamaSupervisor,
    _api_root,
)
