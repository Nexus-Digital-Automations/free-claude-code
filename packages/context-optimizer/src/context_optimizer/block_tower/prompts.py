"""Owns: Ollama prompt templates for sealing and selecting blocks.

Two prompts live here:

    build_seal_prompt    — produce a frozen block body + one-line header
                           from an uncompacted message tail.
    build_select_prompt  — given existing block headers + the current user
                           message, choose which block indices to include.

Both prompts produce delimited output sections so a single Ollama call
yields all needed parts (no second round-trip). Parsers in sealer.py and
selector.py reverse the encoding.

Does NOT own: the Ollama call itself (sealer/selector), token counting,
or response application.
Called by: sealer.py, selector.py.
"""

from __future__ import annotations

from .._core import render_content


_SEAL_TEMPLATE = """\
You are compacting a contiguous segment of an AI coding assistant
conversation into a single immutable summary block.

Output EXACTLY two sections, in this order, separated by the literal
delimiters shown. Do not add commentary outside the sections.

<<<HEADER>>>
A single-line summary, ≤120 chars, factual, no pleasantries. This will
be shown to a small relevance-scoring model — make it concretely
descriptive (mention files/functions/decisions). Example:
"Refactored auth middleware in api/routes.py to use JWT; added tests"
<<<END_HEADER>>>

<<<BODY>>>
A dense factual summary of the segment, target ~{target_tokens} tokens.
Preserve: decisions made, code written, errors encountered, conclusions
reached, file paths and identifiers. Discard: pleasantries, redundant
context, intermediate reasoning that led nowhere.
<<<END_BODY>>>

Conversation segment to compact:

{rendered}
"""


_SELECT_TEMPLATE = """\
You are selecting which historical conversation summary blocks are
relevant to the current user message.

Each block has an integer index and a one-line header summarising its
content. Return ONLY the integer indices of blocks that are likely
relevant to answering the current message. Be inclusive when in doubt
— omitting a relevant block costs more than including an irrelevant one.

Output EXACTLY this format on one line, with no explanation:

INCLUDE: <comma-separated indices, or NONE>

Available blocks:
{block_list}

Current user message:
{current_message}
"""


def build_seal_prompt(messages_tail: list[dict], target_tokens: int, preview_chars: int) -> str:
    """Render the sealing prompt for `messages_tail`.

    `preview_chars` caps the rendered length of each message so a runaway
    paste can't blow the prompt budget. Counterpart: settings.render_preview_chars.
    """
    rendered_lines = []
    for i, msg in enumerate(messages_tail):
        role = msg.get("role", "?")
        text = render_content(msg.get("content", ""))
        if len(text) > preview_chars:
            text = text[: preview_chars] + f"... [{len(text) - preview_chars} chars trimmed]"
        rendered_lines.append(f"[{i}] {role}: {text}")
    return _SEAL_TEMPLATE.format(
        target_tokens=target_tokens,
        rendered="\n\n".join(rendered_lines),
    )


def build_select_prompt(block_headers: list[tuple[int, str]], current_message: str) -> str:
    """Render the selection prompt from `(index, header)` pairs.

    The selector caller must guarantee non-empty `block_headers` — an
    empty list means selection should be skipped entirely (no Ollama call).
    """
    block_list = "\n".join(f"  {idx}: {header}" for idx, header in block_headers)
    truncated_message = current_message[:1500]
    return _SELECT_TEMPLATE.format(
        block_list=block_list,
        current_message=truncated_message,
    )


def parse_seal_response(content: str) -> tuple[str, str] | None:
    """Extract `(header, body)` from a seal response. Returns None on parse failure.

    Failure modes that yield None (and are logged by the caller):
      • Either delimiter pair is missing.
      • Header is empty after strip.
      • Body is empty after strip.
    """
    header = _extract_section(content, "<<<HEADER>>>", "<<<END_HEADER>>>")
    body = _extract_section(content, "<<<BODY>>>", "<<<END_BODY>>>")
    if not header or not body:
        return None
    return header, body


def parse_select_response(content: str, max_index: int) -> list[int] | None:
    """Extract the included block indices. Returns None on parse failure.

    Indices outside `[1, max_index]` are silently dropped. "NONE" yields [].
    """
    for line in content.splitlines():
        stripped = line.strip()
        if not stripped.upper().startswith("INCLUDE:"):
            continue
        payload = stripped.split(":", 1)[1].strip()
        if payload.upper() == "NONE":
            return []
        included: list[int] = []
        for token in payload.split(","):
            token = token.strip()
            if not token:
                continue
            try:
                idx = int(token)
            except ValueError:
                return None
            if 1 <= idx <= max_index:
                included.append(idx)
        return sorted(set(included))
    return None


def _extract_section(content: str, open_tag: str, close_tag: str) -> str:
    # rfind on the open tag: small models occasionally echo the prompt
    # template before emitting their real answer (e.g. "Here is the
    # output: <<<HEADER>>> ... <<<HEADER>>> Refactored auth ...
    # <<<END_HEADER>>>"). Anchoring on the LAST open-tag means the echoed
    # leading copy gets ignored. Mis-extraction still degrades to "" → the
    # caller treats the parse as failed and writes a placeholder block.
    open_idx = content.rfind(open_tag)
    if open_idx < 0:
        return ""
    close_idx = content.find(close_tag, open_idx + len(open_tag))
    if close_idx < 0:
        return ""
    return content[open_idx + len(open_tag) : close_idx].strip()
