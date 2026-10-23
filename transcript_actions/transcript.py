"""Load a transcript, number its lines, and cut it into overlapping chunks.

Line numbers are 1-based and match the file, blank lines included, so a line number the
model reports can be checked against the source with no offset arithmetic.
"""

import re
from dataclasses import dataclass
from typing import List, Tuple

# A speaker label is a capitalized name (letters, digits, spaces, dots, apostrophes and
# hyphens, at most 41 characters) followed by a colon and one whitespace character, at the
# very start of the line. Both the straight apostrophe and the typographic ones (U+2019 and
# U+2018, which Word, Google Docs and most transcription tools emit) are allowed, so
# "O'Brien:" and "O’Brien:" both parse; verify.normalize folds the typographic forms to
# the straight one, and the quote side reuses this regex, so the two stay consistent. The
# lazy quantifier stops at the first "colon plus whitespace", so "Mara: note: x" yields the
# speaker "Mara". A capitalized word such as "Note: ..." matches too; the pattern cannot
# tell a short name from a label word and the contract accepts that.
SPEAKER_RE = re.compile(r"^([A-Z][\w .'’‘-]{0,40}?):\s")


@dataclass(frozen=True)
class Line:
    number: int  # 1-based
    speaker: str  # "" when the line has no "Name:" prefix
    text: str  # the line without the speaker label, stripped
    raw: str  # the original line, stripped of trailing newline


def split_speaker(raw: str) -> Tuple[str, str]:
    """Return (speaker, text) for one raw line. speaker is "" when there is no label."""
    match = SPEAKER_RE.match(raw)
    if match is None:
        return "", raw.strip()
    return match.group(1).strip(), raw[match.end():].strip()


def parse_transcript(text: str) -> List[Line]:
    """Split text into numbered lines. Blank lines are kept (so numbers match the file)
    with speaker="" and text="". A speaker label is `^([A-Z][\\w .'’‘-]{0,40}?):\\s` at line
    start (straight or typographic apostrophes are both accepted inside a name).

    Only "\\n" separates lines (a preceding "\\r" is dropped), so the numbering agrees with
    what an editor shows. A trailing newline ends the last line; it does not add an empty one.
    """
    if not text:
        return []
    parts = text.split("\n")
    if parts[-1] == "":
        parts.pop()
    lines: List[Line] = []
    for number, part in enumerate(parts, start=1):
        raw = part.rstrip("\r")
        speaker, body = split_speaker(raw)
        lines.append(Line(number=number, speaker=speaker, text=body, raw=raw))
    return lines


def load_transcript(path: str) -> List[Line]:
    """Read a UTF-8 text file (a byte-order mark is tolerated) and parse it."""
    with open(path, "r", encoding="utf-8-sig") as handle:
        return parse_transcript(handle.read())


@dataclass(frozen=True)
class Chunk:
    index: int  # 0-based
    start: int  # first line number (inclusive)
    end: int  # last line number (inclusive)
    lines: List[Line]


def chunk_lines(lines: List[Line], size: int = 120, overlap: int = 10) -> List[Chunk]:
    """Consecutive windows of `size` lines, each sharing `overlap` lines with the previous one
    so an item that straddles a boundary is seen whole at least once. size > overlap >= 0.

    The last window may be shorter than `size`; it still begins `overlap` lines before the
    end of the previous window so the tail is seen with its context.
    """
    if overlap < 0 or size <= overlap:
        raise ValueError(
            "chunk_lines needs size > overlap >= 0, got size=%d overlap=%d" % (size, overlap)
        )
    total = len(lines)
    if total == 0:
        return []
    step = size - overlap
    chunks: List[Chunk] = []
    start = 0
    index = 0
    while True:
        window = lines[start:start + size]
        chunks.append(Chunk(index=index, start=window[0].number, end=window[-1].number, lines=window))
        if start + size >= total:
            break
        start += step
        index += 1
    return chunks


def render_chunk(chunk: Chunk) -> str:
    """The text sent to the model: one line per transcript line, formatted
    `{number:04d} | {raw}` so every line carries its number."""
    return "\n".join(f"{line.number:04d} | {line.raw}" for line in chunk.lines)
