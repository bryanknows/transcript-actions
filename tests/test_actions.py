import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import actions  # noqa: E402
from actions import Line, chunk, dedupe, normalize, verify  # noqa: E402
from mock_client import MockClient  # noqa: E402

SAMPLE = Path(__file__).resolve().parents[1] / "samples" / "team_meeting.txt"
LINES = [
    Line(1, "[00:00:20] Kelsey: Dan, can you get the cleaners booked?"),
    Line(2, "[00:00:31] Dan: Yes, I'll book both by Monday and send you the quotes."),
]


def item(**over):
    base = {"task": "Book cleaners", "owner": "Dan", "due": "Monday",
            "source_lines": [2], "quote": "I'll book both by Monday", "confidence": 0.9}
    return {**base, **over}


def test_accepts_exact_quote():
    it, reason = verify(item(), LINES)
    assert reason is None and not it.needs_review


def test_accepts_quote_with_different_punctuation_and_case():
    it, _ = verify(item(quote="i'll book both, by monday!"), LINES)
    assert it is not None


def test_accepts_quote_spanning_two_cited_lines():
    it, _ = verify(item(source_lines=[1, 2], quote="cleaners booked Yes I'll book both"), LINES)
    assert it is not None


def test_rejects_invented_quote():
    it, reason = verify(item(quote="Dan will book the cleaners this week"), LINES)
    assert it is None and "does not appear" in reason


def test_rejects_quote_from_uncited_line():
    # Quote is real, but it's on line 2, not line 1 as cited.
    it, reason = verify(item(source_lines=[1]), LINES)
    assert it is None


def test_rejects_nonexistent_line():
    it, reason = verify(item(source_lines=[99]), LINES)
    assert it is None and "99" in reason


def test_rejects_tiny_quote():
    it, reason = verify(item(quote="Monday"), LINES)
    assert it is None and "too short" in reason


def test_flags_missing_owner_and_low_confidence():
    it, _ = verify(item(owner=None, confidence=0.4), LINES)
    assert it.needs_review and len(it.review_reasons) == 2


def test_flags_owner_not_in_transcript():
    it, _ = verify(item(owner="Marcus"), LINES)
    assert it.needs_review and "owner name not found" in it.review_reasons[0]


def test_dedupe_keeps_most_confident():
    a, _ = verify(item(confidence=0.6), LINES)
    b, _ = verify(item(confidence=0.95), LINES)
    out = dedupe([a, b])
    assert len(out) == 1 and out[0].confidence == 0.95


def test_chunking_overlaps_and_covers_everything():
    lines = [Line(i, f"line {i}") for i in range(1, 301)]
    parts = chunk(lines, size=120, overlap=10)
    covered = {l.number for p in parts for l in p}
    assert covered == set(range(1, 301))
    assert parts[0][-1].number - parts[1][0].number + 1 == 10  # 10-line overlap


def test_normalize_strips_timestamps_and_prefixes():
    assert normalize("L12: [00:01:27] Hello, World!") == "hello world"


def test_end_to_end_mock_run_retries_and_filters():
    client = MockClient()
    result = actions.run(SAMPLE, client)
    assert client.messages.calls == 2          # first answer was malformed -> retried
    assert len(result["action_items"]) == 5
    assert len(result["rejected"]) == 3
    assert all(r["reason"] for r in result["rejected"])
