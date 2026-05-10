"""Request body builder for Vertex AI Model Garden Gemma endpoints.

Owner: providers/vertex package. Vertex's OpenAI-compat surface accepts
the standard OpenAI ``/chat/completions`` schema. We translate the
proxy's Anthropic-format request via the shared
:func:`build_base_request_body` and apply a Gemma-specific tweak: Gemma
deployments require a fully-qualified ``model`` string of the form
``google/<gemma-id>`` when hitting the openapi shim, so we prefix
unprefixed IDs.
"""

from __future__ import annotations

from typing import Any

from providers.common.message_converter import build_base_request_body


def build_request_body(request: Any, *, thinking_enabled: bool) -> dict:
    """Translate an Anthropic-shaped proxy request into Vertex OpenAI-compat JSON.

    @stable — called per request from
    :class:`providers.vertex.client.VertexProvider._build_request_body`.
    """
    body = build_base_request_body(request, include_thinking=thinking_enabled)
    body["model"] = _normalize_model_id(body.get("model", ""))
    return body


def _normalize_model_id(raw: str) -> str:
    """Prefix ``google/`` for the openapi shim if not already namespaced."""
    if not raw or "/" in raw:
        return raw
    if raw.startswith("gemma"):
        return f"google/{raw}"
    return raw
