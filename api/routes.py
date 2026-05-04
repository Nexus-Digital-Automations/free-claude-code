"""FastAPI route handlers."""

import time
import traceback
import uuid

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from fastapi.responses import StreamingResponse
from loguru import logger

from config.settings import Settings
from providers.common import get_user_facing_error_message
from providers.common.context_optimizer import ContextOptimizer
from providers.exceptions import InvalidRequestError, ProviderError

from .dependencies import get_provider_for_type, get_settings, require_api_key
from .metrics import (
    COMPACTION_INVOCATION_TOTAL,
    COMPACTION_TOKENS_SAVED,
    REQUEST_DURATION_SECONDS,
    REQUEST_TOTAL,
    render_exposition,
)
from .models.anthropic import MessagesRequest, TokenCountRequest
from .models.responses import ModelResponse, ModelsListResponse, TokenCountResponse
from .optimization_handlers import try_optimizations
from .request_utils import get_token_count

router = APIRouter()


SUPPORTED_CLAUDE_MODELS = [
    ModelResponse(
        id="claude-opus-4-7",
        display_name="Claude Opus 4.7",
        created_at="2025-09-01T00:00:00Z",
    ),
    ModelResponse(
        id="claude-sonnet-4-6",
        display_name="Claude Sonnet 4.6",
        created_at="2025-07-01T00:00:00Z",
    ),
    ModelResponse(
        id="claude-haiku-4-5-20251001",
        display_name="Claude Haiku 4.5",
        created_at="2025-10-01T00:00:00Z",
    ),
    ModelResponse(
        id="claude-opus-4-20250514",
        display_name="Claude Opus 4",
        created_at="2025-05-14T00:00:00Z",
    ),
    ModelResponse(
        id="claude-sonnet-4-20250514",
        display_name="Claude Sonnet 4",
        created_at="2025-05-14T00:00:00Z",
    ),
    ModelResponse(
        id="claude-haiku-4-20250514",
        display_name="Claude Haiku 4",
        created_at="2025-05-14T00:00:00Z",
    ),
    ModelResponse(
        id="claude-3-opus-20240229",
        display_name="Claude 3 Opus",
        created_at="2024-02-29T00:00:00Z",
    ),
    ModelResponse(
        id="claude-3-5-sonnet-20241022",
        display_name="Claude 3.5 Sonnet",
        created_at="2024-10-22T00:00:00Z",
    ),
    ModelResponse(
        id="claude-3-haiku-20240307",
        display_name="Claude 3 Haiku",
        created_at="2024-03-07T00:00:00Z",
    ),
    ModelResponse(
        id="claude-3-5-haiku-20241022",
        display_name="Claude 3.5 Haiku",
        created_at="2024-10-22T00:00:00Z",
    ),
]


def _probe_response(allow: str) -> Response:
    """Return an empty success response for compatibility probes."""
    return Response(status_code=204, headers={"Allow": allow})


async def _instrumented_stream(
    upstream, request_id: str, started: float, provider: str, model: str,
):
    """Wrap a provider stream so REQUEST: completed fires on stream end.

    Logs once on success (when the upstream generator exhausts) and once on
    exception. Also bumps the proxy_request_total/duration metrics with
    outcome=success|failed. Counterpart: providers/openai_compat.py emits
    PROVIDER: done inside the same try/finally for the same request_id.
    """
    try:
        async for chunk in upstream:
            yield chunk
    except Exception as exc:
        elapsed = time.monotonic() - started
        REQUEST_TOTAL.labels(provider=provider, model=model, outcome="failed").inc()
        REQUEST_DURATION_SECONDS.labels(provider=provider, model=model).observe(elapsed)
        logger.error(
            "REQUEST: stream_failed request_id={} duration_ms={:.0f} "
            "error_type={} error={}",
            request_id, elapsed * 1000, type(exc).__name__, exc,
        )
        raise
    else:
        elapsed = time.monotonic() - started
        REQUEST_TOTAL.labels(provider=provider, model=model, outcome="success").inc()
        REQUEST_DURATION_SECONDS.labels(provider=provider, model=model).observe(elapsed)
        logger.info(
            "REQUEST: completed request_id={} duration_ms={:.0f}",
            request_id, elapsed * 1000,
        )


# =============================================================================
# Routes
# =============================================================================
@router.post("/v1/messages")
async def create_message(
    request_data: MessagesRequest,
    raw_request: Request,
    settings: Settings = Depends(get_settings),
    _auth=Depends(require_api_key),
):
    """Create a message (always streaming).

    Every request gets a request_id stamped at the top so error handlers,
    optimizations, provider call sites, and the client (via X-Request-ID
    response header) all share one correlation key.
    """
    request_id = f"req_{uuid.uuid4().hex[:12]}"
    started = time.monotonic()
    with logger.contextualize(request_id=request_id):
        try:
            if not request_data.messages:
                raise InvalidRequestError("messages cannot be empty")

            optimized = try_optimizations(request_data, settings)
            if optimized is not None:
                return optimized
            logger.debug("No optimization matched, routing to provider")

            # Resolve provider from the model-aware mapping
            resolved = request_data.resolved_provider_model or settings.model
            provider_type = Settings.parse_provider_type(resolved)
            provider = get_provider_for_type(provider_type)
            logger.info(
                "REQUEST: provider_selected request_id={} provider={} model={} resolved={}",
                request_id, provider_type, request_data.model, resolved,
            )

            raw_tokens = get_token_count(
                request_data.messages, request_data.system, request_data.tools,
                tokenizer_name=settings.context_tokenizer_model,
            )
            logger.info(
                "REQUEST: start request_id={} model={} provider={} messages={} input_tokens={}",
                request_id,
                request_data.model,
                provider_type,
                len(request_data.messages),
                raw_tokens,
            )
            # FULL_PAYLOAD is the highest-signal line for reverse-engineering
            # Claude Code: it captures the entire system prompt, tool list,
            # and message history. INFO under LOG_FULL_PAYLOAD=1 (default)
            # so it shows in default-level scrapes; DEBUG otherwise.
            payload_level = "INFO" if settings.log_full_payload else "DEBUG"
            logger.log(
                payload_level,
                "FULL_PAYLOAD request_id={} payload={}",
                request_id, request_data.model_dump(),
            )

            # Tier 1 strips stale thinking blocks; Tier 2 summarizes older turns
            # via the same provider when the request crosses the token threshold.
            # Counterpart: providers/common/context_optimizer.py — returns the
            # final token count so we don't redo a full tiktoken pass here.
            if settings.context_optimize:
                try:
                    request_data, input_tokens = await ContextOptimizer.optimize(
                        request_data, settings, provider
                    )
                    COMPACTION_INVOCATION_TOTAL.labels(
                        tier="orchestrator", outcome="success",
                    ).inc()
                except Exception:
                    COMPACTION_INVOCATION_TOTAL.labels(
                        tier="orchestrator", outcome="error",
                    ).inc()
                    raise
            else:
                input_tokens = raw_tokens

            saved = raw_tokens - input_tokens
            if saved > 0:
                COMPACTION_TOKENS_SAVED.observe(saved)
            if input_tokens != raw_tokens:
                logger.info(
                    "REQUEST: optimized request_id={} input_tokens_after={} saved={}",
                    request_id, input_tokens, saved,
                )

            # Pre-flight to seed message_start.usage.input_tokens with the
            # provider's actual prompt_tokens. Without this Claude Code's TUI
            # shows cl100k_base estimates that diverge from the upstream
            # tokenizer (DeepSeek ~1.65-2.35x cl100k for typical payloads).
            # Counterpart: providers/openai_compat.py:preflight_token_count
            # LlamaCpp/LMStudio providers extend BaseProvider directly and
            # don't implement this; isinstance check also rejects MagicMock
            # returns from test fixtures so format/arithmetic stays type-safe.
            if settings.preflight_token_count and hasattr(provider, "preflight_token_count"):
                preflight = await provider.preflight_token_count(request_data)
                if isinstance(preflight, int) and preflight != input_tokens:
                    logger.info(
                        "REQUEST: preflight request_id={} estimate={} actual={} diff={:+d}",
                        request_id, input_tokens, preflight, preflight - input_tokens,
                    )
                    input_tokens = preflight

            return StreamingResponse(
                _instrumented_stream(
                    provider.stream_response(
                        request_data,
                        input_tokens=input_tokens,
                        request_id=request_id,
                    ),
                    request_id=request_id,
                    started=started,
                    provider=provider_type,
                    model=request_data.model,
                ),
                media_type="text/event-stream",
                headers={
                    "X-Accel-Buffering": "no",
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Request-ID": request_id,
                },
            )

        except ProviderError:
            REQUEST_TOTAL.labels(
                provider="unknown", model=request_data.model, outcome="failed",
            ).inc()
            raise
        except Exception as e:
            elapsed = time.monotonic() - started
            REQUEST_TOTAL.labels(
                provider="unknown", model=request_data.model, outcome="failed",
            ).inc()
            REQUEST_DURATION_SECONDS.labels(
                provider="unknown", model=request_data.model,
            ).observe(elapsed)
            logger.error(
                "REQUEST: failed request_id={} duration_ms={:.0f} error_type={} error={}\n{}",
                request_id,
                elapsed * 1000,
                type(e).__name__,
                e,
                traceback.format_exc(),
            )
            raise HTTPException(
                status_code=getattr(e, "status_code", 500),
                detail=get_user_facing_error_message(e),
            ) from e


@router.api_route("/v1/messages", methods=["HEAD", "OPTIONS"])
async def probe_messages(_auth=Depends(require_api_key)):
    """Respond to Claude compatibility probes for the messages endpoint."""
    return _probe_response("POST, HEAD, OPTIONS")


@router.post("/v1/messages/count_tokens")
async def count_tokens(
    request_data: TokenCountRequest,
    settings: Settings = Depends(get_settings),
    _auth=Depends(require_api_key),
):
    """Count tokens for a request."""
    request_id = f"req_{uuid.uuid4().hex[:12]}"
    with logger.contextualize(request_id=request_id):
        try:
            tokens = get_token_count(
                request_data.messages, request_data.system, request_data.tools,
                tokenizer_name=settings.context_tokenizer_model,
            )
            logger.info(
                "COUNT_TOKENS: request_id={} model={} messages={} input_tokens={}",
                request_id,
                getattr(request_data, "model", "unknown"),
                len(request_data.messages),
                tokens,
            )
            return TokenCountResponse(input_tokens=tokens)
        except Exception as e:
            logger.error(
                "COUNT_TOKENS_ERROR: request_id={} error={}\n{}",
                request_id,
                get_user_facing_error_message(e),
                traceback.format_exc(),
            )
            raise HTTPException(
                status_code=500, detail=get_user_facing_error_message(e)
            ) from e


@router.api_route("/v1/messages/count_tokens", methods=["HEAD", "OPTIONS"])
async def probe_count_tokens(_auth=Depends(require_api_key)):
    """Respond to Claude compatibility probes for the token count endpoint."""
    return _probe_response("POST, HEAD, OPTIONS")


@router.get("/")
async def root(
    settings: Settings = Depends(get_settings), _auth=Depends(require_api_key)
):
    """Root endpoint."""
    return {
        "status": "ok",
        "provider": settings.provider_type,
        "model": settings.model,
    }


@router.api_route("/", methods=["HEAD", "OPTIONS"])
async def probe_root(_auth=Depends(require_api_key)):
    """Respond to compatibility probes for the root endpoint."""
    return _probe_response("GET, HEAD, OPTIONS")


@router.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}


@router.api_route("/health", methods=["HEAD", "OPTIONS"])
async def probe_health():
    """Respond to compatibility probes for the health endpoint."""
    return _probe_response("GET, HEAD, OPTIONS")


@router.get("/v1/models", response_model=ModelsListResponse)
async def list_models(_auth=Depends(require_api_key)):
    """List the Claude model ids this proxy advertises for compatibility."""
    return ModelsListResponse(
        data=SUPPORTED_CLAUDE_MODELS,
        first_id=SUPPORTED_CLAUDE_MODELS[0].id if SUPPORTED_CLAUDE_MODELS else None,
        has_more=False,
        last_id=SUPPORTED_CLAUDE_MODELS[-1].id if SUPPORTED_CLAUDE_MODELS else None,
    )


@router.get("/metrics")
async def metrics(settings: Settings = Depends(get_settings)):
    """Prometheus exposition. 404 unless METRICS_ENABLED=1.

    No auth required so a sidecar Prometheus can scrape without leaking the
    Anthropic auth token. The endpoint exposes only metric counters,
    histograms, and the Ollama supervisor state — no request payloads.
    """
    if not settings.metrics_enabled:
        raise HTTPException(status_code=404, detail="metrics disabled")
    body, content_type = render_exposition()
    return Response(content=body, media_type=content_type)


@router.post("/stop")
async def stop_cli(request: Request, _auth=Depends(require_api_key)):
    """Stop all CLI sessions and pending tasks."""
    handler = getattr(request.app.state, "message_handler", None)
    if not handler:
        # Fallback if messaging not initialized
        cli_manager = getattr(request.app.state, "cli_manager", None)
        if cli_manager:
            await cli_manager.stop_all()
            logger.info("STOP_CLI: source=cli_manager cancelled_count=N/A")
            return {"status": "stopped", "source": "cli_manager"}
        raise HTTPException(status_code=503, detail="Messaging system not initialized")

    count = await handler.stop_all_tasks()
    logger.info("STOP_CLI: source=handler cancelled_count={}", count)
    return {"status": "stopped", "cancelled_count": count}
