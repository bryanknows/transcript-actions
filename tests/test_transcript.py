"""Tests for transcript loading, numbering and chunking.

All names and the company (Orbitwell, maker of the Flakefinder CI tool) are invented.
"""

import pytest

from transcript_actions.transcript import (
    Chunk,
    Line,
    chunk_lines,
    load_transcript,
    parse_transcript,
    render_chunk,
)

SAMPLE = (
    "Mara Quill: Morning everyone, quick sync on the Flakefinder beta.\n"
    "Devin Okafor-Lind: The retry classifier is green on the sample repos.\n"
    "\n"
    "Priya Natarajan: I will write the onboarding guide this week.\n"
    "a line with no label at all\n"
)


def test_parse_numbers_speakers_text_and_raw():
    lines = parse_transcript(SAMPLE)
    assert [line.number for line in lines] == [1, 2, 3, 4, 5]
    assert lines[0] == Line(
        number=1,
        speaker="Mara Quill",
        text="Morning everyone, quick sync on the Flakefinder beta.",
        raw="Mara Quill: Morning everyone, quick sync on the Flakefinder beta.",
    )
    assert lines[4].speaker == ""
    assert lines[4].text == "a line with no label at all"
    assert lines[4].raw == "a line with no label at all"


def test_blank_lines_keep_their_numbers():
    lines = parse_transcript(SAMPLE)
    blank = lines[2]
    assert blank.number == 3
    assert blank.speaker == ""
    assert blank.text == ""
    assert blank.raw == ""
    # the line after the blank keeps the file's numbering
    assert lines[3].number == 4
    assert lines[3].speaker == "Priya Natarajan"


def test_trailing_newline_and_crlf_do_not_add_lines():
    assert len(parse_transcript("one\ntwo\n")) == 2
    assert len(parse_transcript("one\ntwo")) == 2
    assert parse_transcript("") == []
    windows = parse_transcript("Mara Quill: hi\r\nDevin Okafor-Lind: hello\r\n")
    assert [line.raw for line in windows] == ["Mara Quill: hi", "Devin Okafor-Lind: hello"]
    assert windows[1].text == "hello"


def test_speaker_names_with_spaces_dots_and_hyphens():
    lines = parse_transcript(
        "Priya Natarajan: one\n"
        "Devin Okafor-Lind: two\n"
        "Dr. T. Brightwater: three\n"
        "Mara O'Quill: four\n"
        "Mara O’Quill: five\n"
        "‘Tomasz’ Brightwater: six\n"
    )
    assert [line.speaker for line in lines] == [
        "Priya Natarajan",
        "Devin Okafor-Lind",
        "Dr. T. Brightwater",
        "Mara O'Quill",
        "Mara O’Quill",
    ] + [""]
    # The label must still start with a capital letter; a leading quotation mark is not one.
    assert lines[5].text == "‘Tomasz’ Brightwater: six"
    assert [line.text for line in lines[:5]] == ["one", "two", "three", "four", "five"]


def test_label_stops_at_first_colon_and_strips_remainder():
    (line,) = parse_transcript("Mara Quill:    decision: ship on Monday   \n")
    assert line.speaker == "Mara Quill"
    assert line.text == "decision: ship on Monday"
    assert line.raw == "Mara Quill:    decision: ship on Monday   "


def test_shapes_the_speaker_regex_rejects():
    lines = parse_transcript(
        "note:no space after the colon\n"
        "http://example.test/path: lowercase start\n"
        "10:30 Mara Quill: digit start\n"
        "  Mara Quill: indented label\n"
        "Mara Quill:\n"
    )
    assert [line.speaker for line in lines] == ["", "", "", "", ""]
    # text is the stripped raw line when there is no label
    assert lines[3].text == "Mara Quill: indented label"
    assert lines[4].text == "Mara Quill:"


def test_capitalized_label_words_count_as_speakers_by_design():
    # The contract regex cannot tell a short name from a label word, so "Note: ..." parses
    # as speaker "Note". The verifier is unaffected: the label is stripped from the text the
    # same way a real name is.
    (line,) = parse_transcript("Note: the deadline moved to Friday\n")
    assert line.speaker == "Note"
    assert line.text == "the deadline moved to Friday"


def _numbered(count):
    return parse_transcript("\n".join("Mara Quill: line %d" % i for i in range(1, count + 1)) + "\n")


def test_chunk_lines_overlap_and_partial_tail():
    lines = _numbered(25)
    chunks = chunk_lines(lines, size=10, overlap=3)
    assert [(c.index, c.start, c.end) for c in chunks] == [
        (0, 1, 10),
        (1, 8, 17),
        (2, 15, 24),
        (3, 22, 25),
    ]
    # each chunk's lines agree with its declared range
    for chunk in chunks:
        assert [line.number for line in chunk.lines] == list(range(chunk.start, chunk.end + 1))
    # consecutive chunks share exactly `overlap` lines
    for earlier, later in zip(chunks, chunks[1:]):
        shared = set(l.number for l in earlier.lines) & set(l.number for l in later.lines)
        assert len(shared) == 3
    # the last chunk is partial and still carries the overlap before the tail
    assert len(chunks[-1].lines) == 4
    # every line is seen at least once
    seen = set()
    for chunk in chunks:
        seen.update(line.number for line in chunk.lines)
    assert seen == set(range(1, 26))


def test_chunk_lines_exact_fit_single_chunk_and_empty():
    exact = chunk_lines(_numbered(10), size=10, overlap=3)
    assert [(c.start, c.end) for c in exact] == [(1, 10)]
    short = chunk_lines(_numbered(5))
    assert len(short) == 1
    assert short[0] == Chunk(index=0, start=1, end=5, lines=_numbered(5))
    assert chunk_lines([]) == []
    no_overlap = chunk_lines(_numbered(7), size=3, overlap=0)
    assert [(c.start, c.end) for c in no_overlap] == [(1, 3), (4, 6), (7, 7)]


def test_chunk_lines_rejects_bad_sizes():
    lines = _numbered(5)
    with pytest.raises(ValueError):
        chunk_lines(lines, size=10, overlap=10)
    with pytest.raises(ValueError):
        chunk_lines(lines, size=5, overlap=8)
    with pytest.raises(ValueError):
        chunk_lines(lines, size=5, overlap=-1)


def test_render_chunk_format():
    lines = parse_transcript("Mara Quill: hi\n\nDevin Okafor-Lind: hello there\n")
    (chunk,) = chunk_lines(lines)
    assert render_chunk(chunk) == (
        "0001 | Mara Quill: hi\n"
        "0002 | \n"
        "0003 | Devin Okafor-Lind: hello there"
    )
    # numbers come from the lines themselves, not from the position inside the chunk
    second = chunk_lines(_numbered(12), size=10, overlap=3)[1]
    assert render_chunk(second).splitlines()[0] == "0008 | Mara Quill: line 8"
    assert render_chunk(second).splitlines()[-1] == "0012 | Mara Quill: line 12"


def test_load_transcript_reads_utf8_with_or_without_bom(tmp_path):
    plain = tmp_path / "plain.txt"
    plain.write_text(SAMPLE, encoding="utf-8")
    assert load_transcript(str(plain)) == parse_transcript(SAMPLE)

    with_bom = tmp_path / "bom.txt"
    with_bom.write_bytes(b"\xef\xbb\xbf" + SAMPLE.encode("utf-8"))
    loaded = load_transcript(str(with_bom))
    assert loaded[0].speaker == "Mara Quill"
    assert len(loaded) == 5
