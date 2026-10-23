"""One Claude call per chunk, returning the model's claimed action items as RawItems.

Nothing here trusts the model: the quotes and line numbers it returns are checked in code by
`verify.py`. This module only has to get the tool input out of the response without crashing.
"""

import os
import sys
from typing import Any, List, Optional

from .schema import ACTION_ITEMS_TOOL, SYSTEM_PROMPT, TOOL_NAME
from .transcript import Chunk, render_chunk
from .verify import RawItem

DEFAULT_MODEL = "claude-opus-5-5"  # overridable via TRANSCRIPT_ACTIONS_MODEL env var
MODEL_ENV_VAR = "TRANSCRIPT_ACTIONS_MODEL"
MAX_TOKENS = 16000


def resolve_model(model: Optional[str] = None) -> str:
    """Explicit argument, then the TRANSCRIPT_ACTIONS_MODEL env var, then DEFAULT_MODEL."""
    if model:
        return model
    from_env = os.environ.get(MODEL_ENV_VAR, "").strip()
    if from_env:
        return from_env
    return DEFAULT_MODEL


def _as_str(value: Any) -> str:
    if isinstance(value, str):
        return value
    if value is None:
        return ""
    return str(value)


def _as_int(value: Any) -> Optional[int]:
    """An int, or None for anything else. bool is an int subclass in Python, so it is excluded."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    return None


def items_from_tool_input(tool_input: dict) -> List[RawItem]:
    """Convert the tool_use input dict into RawItems. Missing owner/due -> "". Non-int line
    numbers -> that item gets line_start=line_end=0 (so verify rejects it, never crashes)."""
    if not isinstance(tool_input, dict):
        return []
    entries = tool_input.get("items")
    if not isinstance(entries, list):
        return []

    items = []  # type: List[RawItem]
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        line_start = _as_int(entry.get("line_start"))
        line_end = _as_int(entry.get("line_end"))
        if line_start is None or line_end is None:
            line_start = 0
            line_end = 0
        items.append(
            RawItem(
                task=_as_str(entry.get("task")),
                owner=_as_str(entry.get("owner")),
                due=_as_str(entry.get("due")),
                quote=_as_str(entry.get("quote")),
                line_start=line_start,
                line_end=line_end,
            )
        )
    return items


def extract_chunk(chunk: Chunk, client: Any, model: Optional[str] = None) -> List[RawItem]:
    """One client.messages.create call for this chunk.

    tool_choice is "auto" because current models reject forced tool choice; the system prompt
    tells the model to always call the tool, with an empty list when nothing was committed to.
    No thinking parameter is passed. A refusal, or a response with no record_action_items
    tool_use block, yields an empty list. So that a refused chunk is not mistaken for a chunk
    with nothing in it, a refusal also prints one line on stderr naming the chunk, its line
    range and the refusal category from `stop_details`; a response cut off at max_tokens
    prints a note saying so and is then parsed as far as it goes.
    """
    response = client.messages.create(
        model=resolve_model(model),
        max_tokens=MAX_TOKENS,
        system=SYSTEM_PROMPT,
        tools=[ACTION_ITEMS_TOOL],
        tool_choice={"type": "auto"},
        messages=[{"role": "user", "content": render_chunk(chunk)}],
    )

    stop_reason = getattr(response, "stop_reason", None)
    if stop_reason == "refusal":
        category = getattr(getattr(response, "stop_details", None), "category", None)
        print(
            "chunk %d (lines %d-%d): model refused (category=%s); no items taken from it"
            % (chunk.index, chunk.start, chunk.end, category or "unknown"),
            file=sys.stderr,
        )
        return []
    if stop_reason == "max_tokens":
        print(
            "chunk %d (lines %d-%d): response was cut off at max_tokens=%d; raise MAX_TOKENS in "
            "extract.py or lower --chunk-size" % (chunk.index, chunk.start, chunk.end, MAX_TOKENS),
            file=sys.stderr,
        )

    items = []  # type: List[RawItem]
    for block in getattr(response, "content", None) or []:
        if getattr(block, "type", None) != "tool_use":
            continue
        if getattr(block, "name", None) != TOOL_NAME:
            continue
        tool_input = getattr(block, "input", None)
        # The SDK hands us a dict already. Anything else is not something we parse.
        if not isinstance(tool_input, dict):
            continue
        items.extend(items_from_tool_input(tool_input))
    return items


def make_client() -> Any:
    """Return anthropic.Anthropic(). Credentials are left to the SDK, which reads
    ANTHROPIC_API_KEY, ANTHROPIC_AUTH_TOKEN and `ant auth login` profiles.

    The constructor itself can raise anthropic.AnthropicError: the SDK resolves the credential
    chain at construction, so ANTHROPIC_PROFILE or ANTHROPIC_CONFIG_DIR pointing at a profile
    that does not exist, or a malformed profile file, fails here rather than at the first
    request. The CLI therefore calls this inside the same try block as the pipeline. It maps
    AuthenticationError and APIConnectionError to the credentials and connection hints, every
    other APIStatusError (unknown model, rate limit, overload) to a one-line message with the
    HTTP status, and the AnthropicError base class (credential-chain failures, including an
    expired or revoked profile at request time) back to the credentials hint. All exit 2."""
    import anthropic

    return anthropic.Anthropic()
