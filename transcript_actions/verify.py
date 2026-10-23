"""The quote check: pure functions that decide whether a quote the model reported really
sits on the lines it named. No I/O and no model calls live here, so the check runs and
tests without an API key.

Matching is done on normalized words (see `normalize`) and is whole-word: "ship the beta"
does not match inside "reship the beta". The same rule is used for the claimed window and
for the whole-transcript search, so the two can never disagree about whether a quote exists.
"""

import re
from dataclasses import dataclass
from typing import List, Optional, Set, Tuple

from .transcript import Line, split_speaker

# Shorter quotes match by accident too easily to be worth anything. The count is taken on
# the spoken words of the quote: the only thing left out is a leading label that names a
# speaker the transcript carries (see `_spoken_words`), so the "too short" reason is never
# the result of spoken words being dropped.
MIN_QUOTE_WORDS = 3

# Everything that is not a word character, whitespace or an apostrophe.
_PUNCTUATION_RE = re.compile(r"[^\w\s']")


@dataclass(frozen=True)
class RawItem:
    task: str
    owner: str  # "" if unknown
    due: str  # "" if none stated
    quote: str  # verbatim supporting text the model claims is in the transcript
    line_start: int
    line_end: int


@dataclass(frozen=True)
class Verdict:
    status: str  # "verified" | "review" | "rejected"
    reason: str  # "" for verified; one plain sentence otherwise
    found_start: Optional[int] = None  # where the quote was actually found, if anywhere
    found_end: Optional[int] = None


def normalize(text: str) -> str:
    """Lowercase, collapse whitespace, strip punctuation except apostrophes, strip ends.

    Curly apostrophes become straight ones first, so a model that types ' still matches a
    transcript typed with the typographic form. Punctuation is treated as a word separator
    rather than deleted, so "end-to-end" and "end to end" normalize to the same words. An
    apostrophe inside a word is kept ("i'll", "theo's"); one at either end of a word is a
    quotation mark rather than part of the word and is dropped, so a commitment the
    transcript wraps in 'single quotes' still matches. Both the quote and the transcript go
    through this function, so contractions and possessives stay consistent on both sides.
    """
    lowered = text.lower().replace("’", "'").replace("‘", "'")
    separated = _PUNCTUATION_RE.sub(" ", lowered)
    words = [word.strip("'") for word in separated.split()]
    return " ".join(word for word in words if word)


def strip_speaker(raw: str) -> str:
    """Remove a leading `Name:` label if present (same regex as transcript.py)."""
    return split_speaker(raw)[1]


def window_text(lines: List[Line], start: int, end: int) -> str:
    """Join lines start..end (inclusive, 1-based) by their `text` (speaker labels already removed),
    separated by single spaces. The speaker-label stripping is what makes a quote that spans two
    lines verify: the label between the two halves is not part of the spoken words.

    Blank lines contribute nothing, so the separator stays a single space.
    """
    return " ".join(line.text for line in lines if start <= line.number <= end and line.text)


def _known_speakers(lines: List[Line]) -> Set[str]:
    """The speaker labels the transcript actually carries, normalized, so a label the model
    typed with a straight apostrophe still matches one the transcript spells with a
    typographic one (`normalize` folds the two, and case)."""
    return {normalize(line.speaker) for line in lines if line.speaker}


def _spoken_words(quote: str, known: Set[str]) -> List[str]:
    """Normalized words of the spoken part of a quote.

    A leading `Name:` label is dropped only when Name is one of `known`, the speakers the
    transcript carries: then the model copied the label along with the words, and labels are
    not part of the text that is searched. Any other capitalized word before a colon, such as
    "Decision: ship Monday" or "I'll handle Oakhollow: they asked", is spoken text and stays,
    which is also what the transcript side keeps in `Line.text`, so the two sides agree.
    """
    speaker, text = split_speaker(quote)
    if speaker and normalize(speaker) in known:
        return normalize(text).split()
    return normalize(quote).split()


def _contains(haystack: str, needle: str) -> bool:
    """Whole-word containment of one normalized string in another."""
    if not needle:
        return False
    return (" " + needle + " ") in (" " + haystack + " ")


def find_quote(lines: List[Line], quote: str) -> Optional[Tuple[int, int]]:
    """Search the whole transcript for the normalized quote. Returns (start, end) line numbers
    of the smallest window containing it, or None. Quotes with fewer than 3 normalized spoken
    words never match (too easy to match by accident).

    Ties on window size go to the earliest occurrence.
    """
    words = _spoken_words(quote, _known_speakers(lines))
    if len(words) < MIN_QUOTE_WORDS:
        return None
    tokens: List[str] = []
    owners: List[int] = []
    for line in lines:
        for token in normalize(line.text).split():
            tokens.append(token)
            owners.append(line.number)
    count = len(words)
    best: Optional[Tuple[int, int]] = None
    for i in range(len(tokens) - count + 1):
        if tokens[i] != words[0] or tokens[i:i + count] != words:
            continue
        start, end = owners[i], owners[i + count - 1]
        if best is None or (end - start) < (best[1] - best[0]):
            best = (start, end)
    return best


def verify_item(item: RawItem, lines: List[Line]) -> Verdict:
    """Rules, in order:
    1. empty quote, or fewer than 3 normalized spoken words    -> rejected ("quote too short to verify")
    2. line_start/line_end outside 1..len(lines) or start > end -> rejected ("line range out of bounds")
    3. normalized quote in window_text(lines, start, end)       -> verified
    4. found elsewhere via find_quote                           -> review ("quote found at lines X-Y, not A-B"),
                                                                   found_start/found_end set
    5. otherwise                                                -> rejected ("quote not found in transcript")

    The spoken words are the quote with a leading speaker label dropped, but only when that
    label names a speaker the transcript carries (see `_spoken_words`); anything else the
    model put before a colon counts as spoken text, for rule 1 and for the matching alike.
    """
    words = _spoken_words(item.quote, _known_speakers(lines))
    if len(words) < MIN_QUOTE_WORDS:
        return Verdict(status="rejected", reason="quote too short to verify")

    start, end = item.line_start, item.line_end
    if start < 1 or end > len(lines) or start > end:
        return Verdict(status="rejected", reason="line range out of bounds")

    needle = " ".join(words)
    if _contains(normalize(window_text(lines, start, end)), needle):
        return Verdict(status="verified", reason="")

    found = find_quote(lines, item.quote)
    if found is not None:
        found_start, found_end = found
        return Verdict(
            status="review",
            reason="quote found at lines %d-%d, not %d-%d" % (found_start, found_end, start, end),
            found_start=found_start,
            found_end=found_end,
        )

    return Verdict(status="rejected", reason="quote not found in transcript")
