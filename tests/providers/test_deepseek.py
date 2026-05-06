"""Tests for DeepSeek provider."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from providers.base import ProviderConfig
from providers.deepseek import DEEPSEEK_BASE_URL, DeepSeekProvider


class MockMessage:
    def __init__(self, role, content):
        self.role = role
        self.content = content


class MockBlock:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


class MockRequest:
    def __init__(self, **kwargs):
        self.model = "deepseek-chat"
        self.messages = [MockMessage("user", "Hello")]
        self.max_tokens = 100
        self.temperature = 0.5
        self.top_p = 0.9
        self.system = "System prompt"
        self.stop_sequences = None
        self.tools = []
        self.extra_body = {}
        self.thinking = MagicMock()
        self.thinking.enabled = True
        for key, value in kwargs.items():
            setattr(self, key, value)


@pytest.fixture
def deepseek_config():
    return ProviderConfig(
        api_key="test_deepseek_key",
        base_url=DEEPSEEK_BASE_URL,
        rate_limit=10,
        rate_window=60,
        enable_thinking=True,
    )


@pytest.fixture(autouse=True)
def mock_rate_limiter():
    """Mock the global rate limiter to prevent waiting."""
    with patch("providers.openai_compat.GlobalRateLimiter") as mock:
        instance = mock.get_instance.return_value
        instance.wait_if_blocked = AsyncMock(return_value=False)

        async def _passthrough(fn, *args, **kwargs):
            return await fn(*args, **kwargs)

        instance.execute_with_retry = AsyncMock(side_effect=_passthrough)
        yield instance


@pytest.fixture
def deepseek_provider(deepseek_config):
    return DeepSeekProvider(deepseek_config)


def test_init(deepseek_config):
    """Test provider initialization."""
    with patch("providers.openai_compat.AsyncOpenAI") as mock_openai:
        provider = DeepSeekProvider(deepseek_config)
        assert provider._api_key == "test_deepseek_key"
        assert provider._base_url == DEEPSEEK_BASE_URL
        mock_openai.assert_called_once()


def test_build_request_body_enables_thinking_for_chat_model(deepseek_provider):
    """Thinking-enabled requests add DeepSeek's thinking payload for chat model."""
    req = MockRequest(model="deepseek-chat")
    body = deepseek_provider._build_request_body(req)

    assert body["model"] == "deepseek-chat"
    assert body["extra_body"]["thinking"] == {"type": "enabled"}
    assert body["messages"][0]["role"] == "system"


def test_build_request_body_global_disable_blocks_request_thinking():
    """Global disable suppresses provider-side thinking even if the request enables it."""
    provider = DeepSeekProvider(
        ProviderConfig(
            api_key="test_deepseek_key",
            base_url=DEEPSEEK_BASE_URL,
            rate_limit=10,
            rate_window=60,
            enable_thinking=False,
        )
    )
    req = MockRequest(model="deepseek-chat")
    body = provider._build_request_body(req)

    assert "extra_body" not in body or "thinking" not in body["extra_body"]


def test_build_request_body_request_disable_blocks_global_thinking(deepseek_provider):
    """Request-level disable suppresses provider-side thinking when global is enabled."""
    req = MockRequest(model="deepseek-chat")
    req.thinking.enabled = False
    body = deepseek_provider._build_request_body(req)

    assert "extra_body" not in body or "thinking" not in body["extra_body"]


def test_build_request_body_reasoner_skips_thinking_extra(deepseek_provider):
    """deepseek-reasoner does not need an extra thinking payload."""
    req = MockRequest(model="deepseek-reasoner")
    body = deepseek_provider._build_request_body(req)

    assert body["model"] == "deepseek-reasoner"
    assert "extra_body" not in body or "thinking" not in body["extra_body"]


def test_build_request_body_preserves_caller_thinking_override(deepseek_provider):
    """Caller-provided thinking payload should not be overwritten."""
    req = MockRequest(
        model="deepseek-chat",
        extra_body={"thinking": {"type": "manual"}},
    )
    body = deepseek_provider._build_request_body(req)

    assert body["extra_body"]["thinking"] == {"type": "manual"}


def test_build_request_body_preserves_reasoning_content(deepseek_provider):
    """Thinking blocks are mirrored into reasoning_content for continuation."""
    req = MockRequest(
        system=None,
        messages=[
            MockMessage(
                "assistant",
                [
                    MockBlock(type="thinking", thinking="First think"),
                    MockBlock(type="text", text="Then answer"),
                ],
            )
        ],
    )

    body = deepseek_provider._build_request_body(req)

    assert body["messages"][0]["reasoning_content"] == "First think"


# ---- Parallel-tool-call nudge ---------------------------------------------


def _mock_tool(name: str = "read_file") -> MockBlock:
    return MockBlock(name=name, description="x", input_schema={"type": "object"})


def test_build_request_body_appends_parallel_tool_call_nudge_when_enabled(deepseek_config):
    """With nudge ON + tools + system, the system content ends with the nudge."""
    from providers.deepseek.request import _PARALLEL_TOOL_CALL_NUDGE

    with patch("providers.openai_compat.AsyncOpenAI"):
        provider = DeepSeekProvider(deepseek_config, parallel_tool_call_nudge=True)
    req = MockRequest(model="deepseek-chat", tools=[_mock_tool()])

    body = provider._build_request_body(req)

    assert body["messages"][0]["role"] == "system"
    content = body["messages"][0]["content"]
    assert content.endswith(_PARALLEL_TOOL_CALL_NUDGE)
    assert "System prompt" in content  # original system content preserved


def test_build_request_body_skips_nudge_when_no_tools_present(deepseek_config):
    """Tool-less requests must not receive the nudge — keeps the gate symmetric with parallel_tool_calls."""
    from providers.deepseek.request import _PARALLEL_TOOL_CALL_NUDGE

    with patch("providers.openai_compat.AsyncOpenAI"):
        provider = DeepSeekProvider(deepseek_config, parallel_tool_call_nudge=True)
    req = MockRequest(model="deepseek-chat", tools=[])

    body = provider._build_request_body(req)

    assert _PARALLEL_TOOL_CALL_NUDGE not in body["messages"][0]["content"]


def test_build_request_body_skips_nudge_when_setting_disabled(deepseek_config):
    """Kill switch: parallel_tool_call_nudge=False produces an unmutated system message."""
    from providers.deepseek.request import _PARALLEL_TOOL_CALL_NUDGE

    with patch("providers.openai_compat.AsyncOpenAI"):
        provider = DeepSeekProvider(deepseek_config, parallel_tool_call_nudge=False)
    req = MockRequest(model="deepseek-chat", tools=[_mock_tool()])

    body = provider._build_request_body(req)

    assert _PARALLEL_TOOL_CALL_NUDGE not in body["messages"][0]["content"]


def test_build_request_body_does_not_invent_a_system_message_for_the_nudge(deepseek_config):
    """If the request has no system prompt, no system message is fabricated just to carry the nudge."""
    from providers.deepseek.request import _PARALLEL_TOOL_CALL_NUDGE

    with patch("providers.openai_compat.AsyncOpenAI"):
        provider = DeepSeekProvider(deepseek_config, parallel_tool_call_nudge=True)
    req = MockRequest(model="deepseek-chat", tools=[_mock_tool()], system=None)

    body = provider._build_request_body(req)

    roles = [m.get("role") for m in body["messages"]]
    assert "system" not in roles
    joined = "".join(str(m.get("content")) for m in body["messages"])
    assert _PARALLEL_TOOL_CALL_NUDGE not in joined


def test_strengthened_nudge_carries_cost_framing_and_concrete_examples(deepseek_config):
    """The nudge text must include cost framing AND ≥2 mini-examples — drives the lift over the prior plain rule."""
    from providers.deepseek.request import _PARALLEL_TOOL_CALL_NUDGE

    assert "re-ships" in _PARALLEL_TOOL_CALL_NUDGE  # cost framing
    assert "Read, Read, Read" in _PARALLEL_TOOL_CALL_NUDGE  # concrete example #1
    assert "Grep, Grep" in _PARALLEL_TOOL_CALL_NUDGE  # concrete example #2


def _tool_result_block(tool_use_id: str, text: str = "ok") -> MockBlock:
    return MockBlock(type="tool_result", tool_use_id=tool_use_id, content=text)


def test_appends_parallel_reminder_user_message_when_two_or_more_tool_results_present(
    deepseek_config,
):
    """Happy path: ≥2 tool_results in the last user turn → trailing user reminder appended."""
    from providers.deepseek.request import _PARALLEL_TOOL_CALL_REMINDER

    with patch("providers.openai_compat.AsyncOpenAI"):
        provider = DeepSeekProvider(deepseek_config, parallel_tool_call_nudge=True)
    req = MockRequest(
        model="deepseek-chat",
        tools=[_mock_tool()],
        messages=[
            MockMessage("user", "Find me three files"),
            MockMessage("assistant", [MockBlock(type="text", text="Reading...")]),
            MockMessage(
                "user",
                [
                    _tool_result_block("t1"),
                    _tool_result_block("t2"),
                ],
            ),
        ],
    )

    body = provider._build_request_body(req)

    assert body["messages"][-1] == {
        "role": "user",
        "content": _PARALLEL_TOOL_CALL_REMINDER,
    }


def test_does_not_append_parallel_reminder_when_single_tool_result(deepseek_config):
    """A single tool_result is not a parallel-call signal — no reminder."""
    from providers.deepseek.request import _PARALLEL_TOOL_CALL_REMINDER

    with patch("providers.openai_compat.AsyncOpenAI"):
        provider = DeepSeekProvider(deepseek_config, parallel_tool_call_nudge=True)
    req = MockRequest(
        model="deepseek-chat",
        tools=[_mock_tool()],
        messages=[
            MockMessage("user", "Find one file"),
            MockMessage("assistant", [MockBlock(type="text", text="Reading...")]),
            MockMessage("user", [_tool_result_block("t1")]),
        ],
    )

    body = provider._build_request_body(req)

    joined = "".join(str(m.get("content")) for m in body["messages"])
    assert _PARALLEL_TOOL_CALL_REMINDER not in joined


def test_does_not_append_parallel_reminder_when_no_tool_results(deepseek_config):
    """Plain text user turn → no reminder (no parallel-call opportunity to repeat)."""
    from providers.deepseek.request import _PARALLEL_TOOL_CALL_REMINDER

    with patch("providers.openai_compat.AsyncOpenAI"):
        provider = DeepSeekProvider(deepseek_config, parallel_tool_call_nudge=True)
    req = MockRequest(model="deepseek-chat", tools=[_mock_tool()])  # default last msg is plain user text

    body = provider._build_request_body(req)

    joined = "".join(str(m.get("content")) for m in body["messages"])
    assert _PARALLEL_TOOL_CALL_REMINDER not in joined


def test_does_not_append_parallel_reminder_when_nudge_flag_disabled(deepseek_config):
    """Kill-switch covers both A and B together — nudge=False suppresses the reminder too."""
    from providers.deepseek.request import _PARALLEL_TOOL_CALL_REMINDER

    with patch("providers.openai_compat.AsyncOpenAI"):
        provider = DeepSeekProvider(deepseek_config, parallel_tool_call_nudge=False)
    req = MockRequest(
        model="deepseek-chat",
        tools=[_mock_tool()],
        messages=[
            MockMessage("user", "Find me three files"),
            MockMessage("assistant", [MockBlock(type="text", text="Reading...")]),
            MockMessage(
                "user",
                [_tool_result_block("t1"), _tool_result_block("t2")],
            ),
        ],
    )

    body = provider._build_request_body(req)

    joined = "".join(str(m.get("content")) for m in body["messages"])
    assert _PARALLEL_TOOL_CALL_REMINDER not in joined


# ---- Parallel-miss observability (D) --------------------------------------


def _sse_with_n_tool_calls(n: int):
    """Build a minimal SSEBuilder-like double whose tool_states dict has n entries."""
    sse = MagicMock()
    sse.blocks = MagicMock()
    sse.blocks.tool_states = {i: object() for i in range(n)}
    return sse


def test_emits_parallel_miss_true_log_when_one_tool_call_after_multiple_tool_results(
    deepseek_provider, caplog
):
    """1 tool_call emitted in response to ≥2 tool_results = parallel-call opportunity declined."""
    import logging

    req = MockRequest(
        messages=[
            MockMessage("user", "Find files"),
            MockMessage("assistant", [MockBlock(type="text", text="Reading...")]),
            MockMessage(
                "user",
                [_tool_result_block("t1"), _tool_result_block("t2")],
            ),
        ],
    )
    with caplog.at_level(logging.INFO):
        deepseek_provider._on_stream_finish(
            req,
            _sse_with_n_tool_calls(1),
            request_id="req_abc",
            error_occurred=False,
        )

    matches = [r for r in caplog.records if "parallel_miss" in r.getMessage()]
    assert len(matches) == 1
    msg = matches[0].getMessage()
    assert "parallel_miss=True" in msg
    assert "tool_calls_emitted=1" in msg
    assert "tool_results_in=2" in msg
    assert "request_id=req_abc" in msg


def test_emits_parallel_miss_false_log_when_response_has_multiple_tool_calls(
    deepseek_provider, caplog
):
    """≥2 tool_calls emitted = the model already batched; parallel_miss=False."""
    import logging

    req = MockRequest(
        messages=[
            MockMessage("user", "Find files"),
            MockMessage("assistant", [MockBlock(type="text", text="Reading...")]),
            MockMessage(
                "user",
                [_tool_result_block("t1"), _tool_result_block("t2")],
            ),
        ],
    )
    with caplog.at_level(logging.INFO):
        deepseek_provider._on_stream_finish(
            req,
            _sse_with_n_tool_calls(3),
            request_id="req_xyz",
            error_occurred=False,
        )

    matches = [r for r in caplog.records if "parallel_miss" in r.getMessage()]
    assert len(matches) == 1
    assert "parallel_miss=False" in matches[0].getMessage()
    assert "tool_calls_emitted=3" in matches[0].getMessage()


@pytest.mark.asyncio
async def test_stream_response_reasoning_content(deepseek_provider):
    """reasoning_content deltas are emitted as thinking blocks."""
    req = MockRequest()

    mock_chunk = MagicMock()
    mock_chunk.choices = [
        MagicMock(
            delta=MagicMock(content=None, reasoning_content="Thinking..."),
            finish_reason="stop",
        )
    ]
    mock_chunk.usage = MagicMock(completion_tokens=2)

    async def mock_stream():
        yield mock_chunk

    with patch.object(
        deepseek_provider._client.chat.completions, "create", new_callable=AsyncMock
    ) as mock_create:
        mock_create.return_value = mock_stream()

        events = [event async for event in deepseek_provider.stream_response(req)]

        assert any(
            '"thinking_delta"' in event and "Thinking..." in event for event in events
        )
