"""Tests for pipeline.py: run end to end with a fake client, dedupe across overlapping chunks,
and write_outputs. Nothing here touches the network."""

import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List

import pytest

from transcript_actions.pipeline import Result, ResultItem, dedupe, run, summary_line, write_outputs
from transcript_actions.schema import TOOL_NAME
from transcript_actions.transcript import load_transcript

SAMPLES = Path(__file__).resolve().parent.parent / "samples"
TRANSCRIPT = SAMPLES / "transcript.txt"
ITEMS = SAMPLES / "items.json"


# --- fakes -------------------------------------------------------------------------------


def tool_block(items: List[Dict[str, Any]]) -> SimpleNamespace:
    return SimpleNamespace(type="tool_use", id="toolu_01", name=TOOL_NAME, input={"items": items})


def fake_response(blocks: List[Any], stop_reason: str = "tool_use") -> SimpleNamespace:
    return SimpleNamespace(stop_reason=stop_reason, content=list(blocks), stop_details=None)


class FakeMessages:
    """Returns the same response for every call and records the kwargs of each call."""

    def __init__(self, response: Any) -> None:
        self._response = response
        self.calls = []  # type: List[Dict[str, Any]]

    def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        return self._response


class FakeClient:
    def __init__(self, response: Any) -> None:
        self.messages = FakeMessages(response)


def client_returning(items: List[Dict[str, Any]], stop_reason: str = "tool_use") -> FakeClient:
    return FakeClient(fake_response([tool_block(items)], stop_reason=stop_reason))


@pytest.fixture(scope="module")
def lines():
    return load_transcript(str(TRANSCRIPT))


@pytest.fixture(scope="module")
def sample_items() -> List[Dict[str, Any]]:
    with open(str(ITEMS), "r", encoding="utf-8") as handle:
        return json.load(handle)["items"]


def make_result_item(quote: str, status: str = "verified", **overrides: Any) -> ResultItem:
    fields = dict(
        task="t",
        owner="",
        due="",
        quote=quote,
        line_start=1,
        line_end=1,
        status=status,
        reason="",
    )
    fields.update(overrides)
    return ResultItem(**fields)


# --- sample data sanity --------------------------------------------------------------------


def test_sample_transcript_shape(lines):
    assert len(lines) == 140
    blanks = [line.number for line in lines if line.text == ""]
    assert len(blanks) >= 3
    assert lines[13].speaker == "Desmond"
    assert lines[13].number == 14


def test_sample_items_are_the_tool_input_shape(sample_items):
    assert len(sample_items) == 8
    for entry in sample_items:
        assert set(entry) == {"task", "owner", "due", "quote", "line_start", "line_end"}
        assert isinstance(entry["line_start"], int)
        assert isinstance(entry["line_end"], int)


# --- run end to end --------------------------------------------------------------------------


def test_run_end_to_end_with_sample_items(lines, sample_items):
    client = client_returning(sample_items)
    result = run(lines, client)

    assert result.counts() == {"verified": 4, "review": 1, "rejected": 3}

    verified_quotes = {item.quote for item in result.verified}
    assert "I'll send the pricing page draft to the whole team by Thursday" in verified_quotes
    for item in result.verified:
        assert item.reason == ""
        assert item.found_start is None and item.found_end is None

    (review,) = result.review
    assert review.quote.startswith("I'll open the ticket for the export timeout")
    assert (review.line_start, review.line_end) == (124, 124)
    assert (review.found_start, review.found_end) == (14, 14)
    assert "14" in review.reason and "124" in review.reason

    reasons = {item.quote: item.reason for item in result.rejected}
    assert reasons["Will do."] == "quote too short to verify"
    assert reasons["I'll rewrite the export service in Rust before the demo."] == "quote not found in transcript"
    assert reasons["I'll email Pembrook the data residency summary tomorrow morning."] == "line range out of bounds"


def test_run_uses_default_chunking_and_dedupes_across_the_overlap(lines, sample_items):
    client = client_returning(sample_items)
    result = run(lines, client)
    # 140 lines, size 120, overlap 10 -> windows 1..120 and 111..140 -> two model calls.
    assert len(client.messages.calls) == 2
    # The fake returned all 8 items for both chunks; dedupe keeps one copy of each.
    assert result.counts() == {"verified": 4, "review": 1, "rejected": 3}
    assert all(item.chunk_index == 0 for item in result.verified)


def test_run_with_small_chunks_still_yields_one_copy_per_item(lines, sample_items):
    client = client_returning(sample_items)
    result = run(lines, client, chunk_size=30, overlap=5)
    assert len(client.messages.calls) >= 4
    assert result.counts() == {"verified": 4, "review": 1, "rejected": 3}


def test_run_passes_model_and_chunk_text_through(lines):
    client = client_returning([])
    run(lines, client, model="test-model", chunk_size=50, overlap=0)
    assert client.messages.calls
    for call in client.messages.calls:
        assert call["model"] == "test-model"
        assert call["tool_choice"] == {"type": "auto"}
    first = client.messages.calls[0]["messages"][0]["content"]
    assert first.startswith("0001 | Marisol: Morning, everyone.")


def test_quotes_split_across_a_speaker_label_verify(lines):
    split_items = [
        {
            "task": "Send the pricing page draft",
            "owner": "Marisol",
            "due": "Thursday",
            "quote": "I'll send the pricing page draft to the whole team by Thursday",
            "line_start": 33,
            "line_end": 34,
        },
        {
            "task": "Put staging on the new runtime",
            "owner": "Theo",
            "due": "before Wednesday's demo",
            "quote": "I'll have the staging cluster on the new runtime before Wednesday's demo",
            "line_start": 44,
            "line_end": 45,
        },
        {
            "task": "Write up and circulate the incident retro",
            "owner": "Beatrix",
            "due": "end of day Monday",
            "quote": "I will write up the incident retro and circulate it by end of day Monday",
            "line_start": 66,
            "line_end": 67,
        },
    ]
    result = run(lines, client_returning(split_items))
    assert result.counts() == {"verified": 3, "review": 0, "rejected": 0}


def test_run_with_refusal_yields_empty_result(lines, sample_items):
    result = run(lines, client_returning(sample_items, stop_reason="refusal"))
    assert result.counts() == {"verified": 0, "review": 0, "rejected": 0}


def test_run_with_no_items_yields_empty_result(lines):
    result = run(lines, client_returning([]))
    assert result.counts() == {"verified": 0, "review": 0, "rejected": 0}
    assert result.to_dicts() == {"verified": [], "review": [], "rejected": []}


def test_run_on_empty_transcript_makes_no_bad_calls():
    client = client_returning([])
    result = run([], client)
    assert result.counts() == {"verified": 0, "review": 0, "rejected": 0}


def test_run_records_chunk_index_of_first_sighting(lines):
    # An item from the tail of the transcript, which only the second chunk sees.
    tail_item = {
        "task": "Add the incident tickets to the retro doc",
        "owner": "Beatrix",
        "due": "",
        "quote": "I'll add the four customer tickets from the incident to the retro doc.",
        "line_start": 135,
        "line_end": 135,
    }
    client = client_returning([tail_item])
    result = run(lines, client)
    (item,) = result.verified
    # The fake returns the item for every chunk, so the first sighting is chunk 0.
    assert item.chunk_index == 0
    assert (item.line_start, item.line_end) == (135, 135)


# --- dedupe ----------------------------------------------------------------------------------


def test_dedupe_keeps_first_of_equal_normalized_quotes():
    first = make_result_item("I'll send the draft by Thursday.", chunk_index=0)
    second = make_result_item("i'll send the draft by thursday", chunk_index=1)
    third = make_result_item("  I'LL   send the draft, by Thursday!  ", chunk_index=1)
    kept = dedupe([first, second, third])
    assert kept == [first]


def test_dedupe_keeps_distinct_quotes_in_order():
    a = make_result_item("I'll send the draft by Thursday.")
    b = make_result_item("I'll open the ticket tonight.")
    c = make_result_item("I will write the migration script this week.")
    assert dedupe([b, a, c]) == [b, a, c]


def test_dedupe_on_empty_list():
    assert dedupe([]) == []


# --- Result helpers ---------------------------------------------------------------------------


def test_result_add_buckets_by_status_and_unknown_goes_to_rejected():
    result = Result()
    result.add(make_result_item("a b c", status="verified"))
    result.add(make_result_item("d e f", status="review", reason="found elsewhere"))
    result.add(make_result_item("g h i", status="rejected", reason="nope"))
    result.add(make_result_item("j k l", status="weird"))
    assert result.counts() == {"verified": 1, "review": 1, "rejected": 2}


def test_to_dicts_has_every_field():
    result = Result()
    result.add(make_result_item("a b c", status="verified", found_start=None, found_end=None, chunk_index=2))
    dicts = result.to_dicts()
    assert set(dicts) == {"verified", "review", "rejected"}
    (entry,) = dicts["verified"]
    assert set(entry) == {
        "task",
        "owner",
        "due",
        "quote",
        "line_start",
        "line_end",
        "status",
        "reason",
        "found_start",
        "found_end",
        "chunk_index",
    }
    assert entry["chunk_index"] == 2


def test_summary_line_format():
    assert summary_line({"verified": 3, "review": 1, "rejected": 2}) == "verified=3 review=1 rejected=2"
    assert summary_line({}) == "verified=0 review=0 rejected=0"


# --- write_outputs ---------------------------------------------------------------------------


def test_write_outputs_file_names_and_shape(tmp_path, lines, sample_items):
    result = run(lines, client_returning(sample_items))
    out_dir = tmp_path / "out"
    paths = write_outputs(result, str(out_dir))

    assert [os.path.basename(p) for p in paths] == ["verified.json", "review.json", "rejected.json"]
    assert all(os.path.isfile(p) for p in paths)

    with open(paths[0], "r", encoding="utf-8") as handle:
        text = handle.read()
    verified = json.loads(text)
    assert isinstance(verified, list)
    assert len(verified) == 4
    entry = verified[0]
    assert list(entry.keys()) == sorted(entry.keys())
    assert set(entry) >= {"task", "owner", "due", "quote", "line_start", "line_end", "status", "reason"}
    assert entry["status"] == "verified"
    assert "\n  " in text  # indent=2

    with open(paths[1], "r", encoding="utf-8") as handle:
        review = json.load(handle)
    assert len(review) == 1
    assert review[0]["found_start"] == 14

    with open(paths[2], "r", encoding="utf-8") as handle:
        rejected = json.load(handle)
    assert len(rejected) == 3
    assert all(item["status"] == "rejected" for item in rejected)


def test_write_outputs_creates_nested_directory_and_writes_empty_lists(tmp_path):
    out_dir = tmp_path / "deep" / "er" / "out"
    paths = write_outputs(Result(), str(out_dir))
    assert out_dir.is_dir()
    for path in paths:
        with open(path, "r", encoding="utf-8") as handle:
            assert json.load(handle) == []
