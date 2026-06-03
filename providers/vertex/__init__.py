"""Vertex AI Model Garden provider package.

Targets self-deployed Gemma endpoints exposing OpenAI-compatible
``/chat/completions``. Auth is ADC-preferred with a static-token
fallback (see :mod:`providers.vertex.auth`).
"""

from .client import VertexProvider

VERTEX_BASE_URL_TEMPLATE = (
    "https://{region}-aiplatform.googleapis.com/v1/projects/{project}"
    "/locations/{region}/endpoints/{endpoint_id}"
)

__all__ = ["VERTEX_BASE_URL_TEMPLATE", "VertexProvider"]
