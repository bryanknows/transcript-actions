"""Tests for schema.py and extract.py. Fake client objects only; nothing here touches the network."""

from types import SimpleNamespace
from typing import Any, Dict, List

import pytest

from transcript_actions import extract
from transcript_actions.extract import DEFAULT_MODEL, MODEL_ENV_VAR, extract_chunk, items_from_tool_input, resolve_model
from transcript_actions.schema import ACTION_ITEMS_TOOL, SYSTEM_PROMPT, TOOL_NAME
from transcript_actions.transcript import Chunk, Line, render_chunk
from transcript_actions.verify import RawItem


# --- fakes -------------------------------------------------------------------------------


def tool_block(items: List[Dict[str, Any]], name: str = TOOL_NAME, raw_input: Any = None) -> SimpleNamespace:
    tool_input = raw_input if raw_input is not None else {"items": items}
    return SimpleNamespace(type="tool_use", id="toolu_01", name=name, input=tool_input)


def text_block(text: str) -> SimpleNamespace:
    return SimpleNamespace(type="text", text=text)


def fake_response(blocks: List[Any], stop_reason: str = "tool_use") -> SimpleNamespace:
    return SimpleNamespace(stop_reason=stop_reason, content=list(blocks), stop_details=None)


class FakeMessages:
    def __init__(self, responses: List[Any]) -> None:
        self._responses = list(responses)
        self.calls = []  # type: List[Dict[str, Any]]

    def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if len(self._responses) > 1:
            return self._responses.pop(0)
        return self._responses[0]


class FakeClient:
    def __init__(self, *responses: Any) -> None:
        self.messages = FakeMessages(list(responses))


def make_line(number: int, speaker: str, text: str) -> Line:
    raw = "%s: %s" % (speaker, text) if speaker else text
    return Line(number=number, speaker=speaker, text=text, raw=raw)


@pytest.fixture
def chunk() -> Chunk:
    lines = [
        make_line(1, "Marisol", "Who owns the export timeout?"),
        make_line(2, "Desmond", "I'll open the ticket tonight and assign it to myself."),
        make_line(3, "Marisol", "Great."),
    ]
    return Chunk(index=0, start=1, end=3, lines=lines)


FULL_ITEM = {
    "task": "Open the export timeout ticket",
    "owner": "Desmond",
    "due": "tonight",
    "quote": "I'll open the ticket tonight and assign it to myself.",
    "line_start": 2,
    "line_end": 2,
}


# --- schema ------------------------------------------------------------------------------


def test_tool_definition_is_strict_at_both_levels():
    assert ACTION_ITEMS_TOOL["name"] == TOOL_NAME
    assert ACTION_ITEMS_TOOL["strict"] is True
    schema = ACTION_ITEMS_TOOL["input_schema"]
    assert schema["type"] == "object"
    assert schema["additionalProperties"] is False
    assert schema["required"] == ["items"]
    item_schema = schema["properties"]["items"]["items"]
    assert item_schema["type"] == "object"
    assert item_schema["additionalProperties"] is False
    assert sorted(item_schema["required"]) == ["line_end", "line_start", "quote", "task"]
    assert item_schema["properties"]["line_start"]["type"] == "integer"
    assert item_schema["properties"]["line_end"]["type"] == "integer"
    for key in ("task", "owner", "due", "quote"):
        assert item_schema["properties"][key]["type"] == "string"


def test_system_prompt_says_to_always_call_the_tool():
    assert TOOL_NAME in SYSTEM_PROMPT
    assert "empty items list" in SYSTEM_PROMPT
    assert "verbatim" in SYSTEM_PROMPT


# --- items_from_tool_input -----------------------------------------------------------------


def test_full_item_round_trips():
    items = items_from_tool_input({"items": [dict(FULL_ITEM)]})
    assert items == [
        RawItem(
            task=FULL_ITEM["task"],
            owner="Desmond",
            due="tonight",
            quote=FULL_ITEM["quote"],
            line_start=2,
            line_end=2,
        )
    ]


def test_missing_owner_and_due_become_empty_strings():
    entry = {"task": "t", "quote": "a b c d", "line_start": 1, "line_end": 1}
    (item,) = items_from_tool_input({"items": [entry]})
    assert item.owner == ""
    assert item.due == ""
    assert item.task == "t"
    assert item.quote == "a b c d"


@pytest.mark.parametrize(
    "line_start, line_end",
    [
        ("2", "2"),
        (2.0, 2),
        (2, None),
        (None, None),
        (True, 2),
        ("two", 3),
        ([2], 2),
    ],
)
def test_non_int_line_numbers_zero_out_both_lines(line_start, line_end):
    entry = dict(FULL_ITEM, line_start=line_start, line_end=line_end)
    (item,) = items_from_tool_input({"items": [entry]})
    assert item.line_start == 0
    assert item.line_end == 0
    assert item.quote == FULL_ITEM["quote"]


def test_missing_line_numbers_zero_out():
    entry = {"task": "t", "quote": "one two three"}
    (item,) = items_from_tool_input({"items": [entry]})
    assert (item.line_start, item.line_end) == (0, 0)


def test_extra_keys_are_ignored():
    entry = dict(FULL_ITEM, confidence=0.9, speaker_id="S2", notes=["x"])
    (item,) = items_from_tool_input({"items": [entry]})
    assert item.task == FULL_ITEM["task"]
    assert item.line_start == 2
    assert not hasattr(item, "confidence")


def test_non_string_task_and_quote_are_coerced_not_crashed():
    entry = {"task": None, "quote": 12345, "line_start": 1, "line_end": 1, "owner": 7}
    (item,) = items_from_tool_input({"items": [entry]})
    assert item.task == ""
    assert item.quote == "12345"
    assert item.owner == "7"


def test_non_dict_entries_are_skipped():
    items = items_from_tool_input({"items": [dict(FULL_ITEM), "junk", None, 3, dict(FULL_ITEM)]})
    assert len(items) == 2


@pytest.mark.parametrize("tool_input", [{}, {"items": None}, {"items": "x"}, {"items": {}}, [], None, "{}"])
def test_malformed_tool_input_yields_empty_list(tool_input):
    assert items_from_tool_input(tool_input) == []


def test_empty_items_list_yields_empty_list():
    assert items_from_tool_input({"items": []}) == []


# --- extract_chunk -------------------------------------------------------------------------


def test_extract_chunk_returns_items_from_tool_use_block(chunk):
    client = FakeClient(fake_response([text_block("Here you go."), tool_block([dict(FULL_ITEM)])]))
    items = extract_chunk(chunk, client)
    assert len(items) == 1
    assert items[0].quote == FULL_ITEM["quote"]
    assert items[0].line_start == 2


def test_extract_chunk_sends_the_contracted_request(chunk, monkeypatch):
    monkeypatch.delenv(MODEL_ENV_VAR, raising=False)
    client = FakeClient(fake_response([tool_block([])]))
    extract_chunk(chunk, client)
    (call,) = client.messages.calls
    assert call["model"] == DEFAULT_MODEL
    assert call["max_tokens"] == 16000
    assert call["system"] == SYSTEM_PROMPT
    assert call["tools"] == [ACTION_ITEMS_TOOL]
    assert call["tool_choice"] == {"type": "auto"}
    assert "thinking" not in call
    assert call["messages"] == [{"role": "user", "content": render_chunk(chunk)}]


def test_rendered_chunk_carries_line_numbers(chunk):
    text = render_chunk(chunk)
    assert "0002 | Desmond: I'll open the ticket tonight" in text


def test_text_only_response_yields_empty_list(chunk):
    client = FakeClient(fake_response([text_block("Nobody committed to anything.")], stop_reason="end_turn"))
    assert extract_chunk(chunk, client) == []


def test_refusal_yields_empty_list_even_with_tool_block(chunk):
    client = FakeClient(fake_response([tool_block([dict(FULL_ITEM)])], stop_reason="refusal"))
    assert extract_chunk(chunk, client) == []


def test_refusal_prints_the_chunk_and_category_on_stderr(chunk, capsys):
    response = fake_response([tool_block([dict(FULL_ITEM)])], stop_reason="refusal")
    response.stop_details = SimpleNamespace(type="refusal", category="general_harms", explanation="not this")
    assert extract_chunk(chunk, FakeClient(response)) == []
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.strip().splitlines() == [
        "chunk 0 (lines 1-3): model refused (category=general_harms); no items taken from it"
    ]


def test_refusal_without_stop_details_still_names_the_chunk(chunk, capsys):
    client = FakeClient(fake_response([], stop_reason="refusal"))  # stop_details is None
    assert extract_chunk(chunk, client) == []
    err = capsys.readouterr().err
    assert "chunk 0 (lines 1-3): model refused (category=unknown)" in err


def test_max_tokens_prints_a_note_and_keeps_whatever_was_parsed(chunk, capsys):
    client = FakeClient(fake_response([tool_block([dict(FULL_ITEM)])], stop_reason="max_tokens"))
    items = extract_chunk(chunk, client)
    assert [item.quote for item in items] == [FULL_ITEM["quote"]]
    err = capsys.readouterr().err
    assert err.startswith("chunk 0 (lines 1-3): response was cut off at max_tokens=16000")
    assert "MAX_TOKENS" in err


def test_normal_responses_print_nothing_on_stderr(chunk, capsys):
    extract_chunk(chunk, FakeClient(fake_response([tool_block([dict(FULL_ITEM)])])))
    extract_chunk(chunk, FakeClient(fake_response([text_block("nothing")], stop_reason="end_turn")))
    assert capsys.readouterr().err == ""


def test_empty_content_yields_empty_list(chunk):
    client = FakeClient(fake_response([], stop_reason="end_turn"))
    assert extract_chunk(chunk, client) == []


def test_tool_use_block_with_other_name_is_ignored(chunk):
    client = FakeClient(fake_response([tool_block([dict(FULL_ITEM)], name="some_other_tool")]))
    assert extract_chunk(chunk, client) == []


def test_multiple_tool_use_blocks_are_concatenated(chunk):
    second = dict(FULL_ITEM, quote="Great.", task="Say great", line_start=3, line_end=3)
    client = FakeClient(fake_response([tool_block([dict(FULL_ITEM)]), tool_block([second])]))
    items = extract_chunk(chunk, client)
    assert [item.line_start for item in items] == [2, 3]


def test_string_input_is_never_parsed(chunk):
    block = tool_block([], raw_input='{"items": [%s]}' % (FULL_ITEM,))
    client = FakeClient(fake_response([block]))
    assert extract_chunk(chunk, client) == []


def test_explicit_model_argument_wins(chunk, monkeypatch):
    monkeypatch.setenv(MODEL_ENV_VAR, "from-env")
    client = FakeClient(fake_response([tool_block([])]))
    extract_chunk(chunk, client, model="from-arg")
    assert client.messages.calls[0]["model"] == "from-arg"


def test_env_var_overrides_default_model(chunk, monkeypatch):
    monkeypatch.setenv(MODEL_ENV_VAR, "from-env")
    client = FakeClient(fake_response([tool_block([])]))
    extract_chunk(chunk, client)
    assert client.messages.calls[0]["model"] == "from-env"


def test_resolve_model_falls_back_to_default(monkeypatch):
    monkeypatch.delenv(MODEL_ENV_VAR, raising=False)
    assert resolve_model(None) == DEFAULT_MODEL
    assert resolve_model("") == DEFAULT_MODEL
    monkeypatch.setenv(MODEL_ENV_VAR, "   ")
    assert resolve_model(None) == DEFAULT_MODEL


def test_default_model_is_current_opus():
    assert DEFAULT_MODEL == "claude-opus-5-5"


# --- make_client -------------------------------------------------------------------------


def test_make_client_returns_sdk_client_without_checking_env(monkeypatch):
    import anthropic

    monkeypatch.setenv("ANTHROPIC_API_KEY", "not-a-real-key-tests-never-call-the-network")
    client = extract.make_client()
    assert isinstance(client, anthropic.Anthropic)
