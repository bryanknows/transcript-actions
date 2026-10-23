"""Tests for the pure quote verifier.

All names and the company (Orbitwell, maker of the Flakefinder CI tool) are invented.
"""

import pytest

from transcript_actions.transcript import parse_transcript
from transcript_actions.verify import (
    RawItem,
    Verdict,
    find_quote,
    normalize,
    strip_speaker,
    verify_item,
    window_text,
)

TRANSCRIPT = (
    "Mara Quill: Morning everyone, quick sync on the Flakefinder beta.\n"  # 1
    "Devin Okafor-Lind: The retry classifier is green on the sample repos. I'll finish the\n"  # 2
    "Devin Okafor-Lind: dashboard export by Friday so the pilot teams can pull their own data.\n"  # 3
    "\n"  # 4
    "Priya Natarajan: Nice. I will write the onboarding guide this week.\n"  # 5
    "Mara Quill: Devin, can you also ship the beta invite emails on Monday?\n"  # 6
    "Devin Okafor-Lind: Yes, I'll ship the beta invite emails on Monday.\n"  # 7
    "Tomasz Brightwater: Design review for the settings page is set for Wednesday.\n"  # 8
    "Mara Quill: Please finish the dashboard export by Friday, that is the key one.\n"  # 9
    "Priya Natarajan: Sounds good. Reship the beta build once the export lands.\n"  # 10
)

LINES = parse_transcript(TRANSCRIPT)


def item(quote, start, end, task="task", owner="", due=""):
    return RawItem(task=task, owner=owner, due=due, quote=quote, line_start=start, line_end=end)


def test_normalize_lowercases_collapses_and_strips_punctuation():
    assert normalize("  Hello,   World!  It's   DONE.  ") == "hello world it's done"
    assert normalize("I’ll ship “the beta”") == "i'll ship the beta"
    assert normalize("end-to-end (q3/q4)") == "end to end q3 q4"
    assert normalize("...") == ""
    assert normalize("") == ""


def test_normalize_drops_quotation_marks_but_keeps_apostrophes_inside_words():
    # A single quotation mark sits at a word's edge; an apostrophe sits inside the word.
    assert normalize("said 'I'll open the ticket' and") == "said i'll open the ticket and"
    assert normalize("‘I’ll ship it Monday’") == "i'll ship it monday"
    # A plural possessive loses its trailing apostrophe on both sides alike.
    assert normalize("the teams' data") == "the teams data"
    # Lone quotation marks vanish rather than leaving empty words behind.
    assert normalize("' ''  '") == ""
    assert normalize("'''word'''") == "word"


def test_strip_speaker_removes_only_a_leading_label():
    assert strip_speaker("Devin Okafor-Lind: ship the beta") == "ship the beta"
    assert strip_speaker("no label here") == "no label here"
    assert strip_speaker("note:no whitespace after colon") == "note:no whitespace after colon"
    assert strip_speaker("Mara Quill: decision: ship Monday") == "decision: ship Monday"


def test_window_text_joins_spoken_words_without_labels_or_blanks():
    assert window_text(LINES, 2, 5) == (
        "The retry classifier is green on the sample repos. I'll finish the "
        "dashboard export by Friday so the pilot teams can pull their own data. "
        "Nice. I will write the onboarding guide this week."
    )
    assert window_text(LINES, 4, 4) == ""
    assert window_text(LINES, 6, 5) == ""


def test_quote_spanning_two_lines_with_a_label_between_verifies():
    # "I'll finish the" ends line 2; "dashboard export by Friday" starts line 3 after a
    # repeated "Devin Okafor-Lind:" label. The label is not part of the spoken words.
    verdict = verify_item(item("I'll finish the dashboard export by Friday", 2, 3), LINES)
    assert verdict == Verdict(status="verified", reason="")
    assert find_quote(LINES, "I'll finish the dashboard export by Friday") == (2, 3)


def test_find_quote_returns_the_smallest_window():
    # The words appear across lines 2-3 and again inside line 9 alone.
    assert find_quote(LINES, "finish the dashboard export by Friday") == (9, 9)
    # Ties on size go to the earliest occurrence (lines 6 and 7 both contain it).
    assert find_quote(LINES, "ship the beta invite emails on Monday") == (6, 6)
    assert find_quote(LINES, "we will cancel the beta launch") is None


def test_find_quote_never_matches_short_quotes():
    assert find_quote(LINES, "Flakefinder beta") is None
    assert find_quote(LINES, "on Monday") is None
    assert find_quote(LINES, "") is None
    assert find_quote(LINES, "!!! ...") is None


def test_rule_1_short_quote_is_rejected():
    expected = Verdict(status="rejected", reason="quote too short to verify")
    assert verify_item(item("", 1, 1), LINES) == expected
    assert verify_item(item("on Monday", 7, 7), LINES) == expected
    assert verify_item(item("--- ... !!!", 7, 7), LINES) == expected
    # the short-quote rule comes before the bounds rule
    assert verify_item(item("Monday", 99, 0), LINES) == expected


@pytest.mark.parametrize("start,end", [(0, 1), (1, 11), (11, 12), (3, 2), (0, 0), (-2, -1)])
def test_rule_2_out_of_bounds_or_inverted_range_is_rejected(start, end):
    verdict = verify_item(item("I will write the onboarding guide this week", start, end), LINES)
    assert verdict == Verdict(status="rejected", reason="line range out of bounds")


def test_rule_2_runs_before_the_whole_transcript_search():
    # The quote exists on line 5, but a bad range is rejected before find_quote is consulted.
    verdict = verify_item(item("I will write the onboarding guide this week", 5, 4), LINES)
    assert verdict.status == "rejected"
    assert verdict.reason == "line range out of bounds"
    assert verdict.found_start is None


def test_rule_3_quote_inside_claimed_window_is_verified():
    assert verify_item(item("write the onboarding guide this week", 5, 5), LINES) == Verdict(
        status="verified", reason=""
    )
    # case and punctuation do not matter
    assert verify_item(item("WRITE THE ONBOARDING GUIDE, THIS WEEK!", 5, 5), LINES).status == "verified"
    # a wider window that contains the quote still verifies
    assert verify_item(item("write the onboarding guide this week", 1, 10), LINES).status == "verified"
    # a quote that carries the speaker label along with the words still verifies
    assert verify_item(item("Priya Natarajan: I will write the onboarding guide", 5, 5), LINES).status == "verified"


def test_rule_4_quote_found_elsewhere_goes_to_review():
    verdict = verify_item(item("I'll finish the dashboard export by Friday", 5, 6), LINES)
    assert verdict == Verdict(
        status="review",
        reason="quote found at lines 2-3, not 5-6",
        found_start=2,
        found_end=3,
    )


def test_rule_5_quote_absent_everywhere_is_rejected():
    verdict = verify_item(item("we will cancel the beta launch", 1, 1), LINES)
    assert verdict == Verdict(status="rejected", reason="quote not found in transcript")
    assert verdict.found_start is None
    assert verdict.found_end is None


def test_matching_is_whole_word():
    # "Reship the beta build" contains the characters "ship the beta build" but not the words.
    verdict = verify_item(item("ship the beta build once", 10, 10), LINES)
    assert verdict == Verdict(status="rejected", reason="quote not found in transcript")
    assert find_quote(LINES, "ship the beta build once") is None


# --- quotation marks around a verbatim quote ----------------------------------------------------

QUOTED_LINES = parse_transcript(
    "Mara Quill: Devin said 'I'll open the ticket tonight and assign it' and I believe him.\n"  # 1
    "Priya Natarajan: She said ‘I’ll ship it Monday for sure’ twice.\n"  # 2
)


def test_quote_wrapped_in_straight_single_quotes_verifies():
    verdict = verify_item(item("I'll open the ticket tonight and assign it", 1, 1), QUOTED_LINES)
    assert verdict == Verdict(status="verified", reason="")
    assert find_quote(QUOTED_LINES, "I'll open the ticket tonight and assign it") == (1, 1)


def test_quote_wrapped_in_curly_single_quotes_verifies():
    verdict = verify_item(item("I’ll ship it Monday for sure", 2, 2), QUOTED_LINES)
    assert verdict == Verdict(status="verified", reason="")
    # The straight-apostrophe spelling of the same words matches the curly transcript too.
    assert verify_item(item("I'll ship it Monday for sure", 2, 2), QUOTED_LINES).status == "verified"
    assert find_quote(QUOTED_LINES, "I'll ship it Monday for sure") == (2, 2)


def test_quote_that_includes_the_quotation_marks_still_verifies():
    assert verify_item(item("'I'll open the ticket tonight and assign it'", 1, 1), QUOTED_LINES).status == "verified"


# --- speaker labels with a typographic apostrophe -----------------------------------------------


def test_two_line_quote_across_a_label_with_a_typographic_apostrophe_verifies():
    lines = parse_transcript(
        "Mara O’Quill: I will finish the\n"
        "Mara O’Quill: dashboard export by Friday so teams can pull data.\n"
    )
    assert [line.speaker for line in lines] == ["Mara O’Quill", "Mara O’Quill"]
    verdict = verify_item(item("I will finish the dashboard export by Friday", 1, 2), lines)
    assert verdict == Verdict(status="verified", reason="")
    assert find_quote(lines, "I will finish the dashboard export by Friday") == (1, 2)
    # A quote carrying the curly-apostrophe label verifies the same way the straight form does.
    assert verify_item(item("Mara O’Quill: I will finish the", 1, 1), lines).status == "verified"
    assert verify_item(item("Mara O'Quill: I will finish the", 1, 1), lines).status == "verified"


# --- a leading label is dropped only when it names a real speaker --------------------------------

LABEL_LINES = parse_transcript(
    "Priya Natarajan: I'll handle Oakhollow: they asked about the audit log export.\n"  # 1
    "Mara Quill: Decision: ship Monday\n"  # 2
)


def test_capitalized_phrase_with_a_colon_inside_the_quote_is_spoken_text():
    # "Oakhollow:" is spoken; all five words are on line 1 and nothing is "too short".
    verdict = verify_item(item("I'll handle Oakhollow: they asked", 1, 1), LABEL_LINES)
    assert verdict == Verdict(status="verified", reason="")
    assert find_quote(LABEL_LINES, "I'll handle Oakhollow: they asked") == (1, 1)


def test_quote_that_opens_with_a_label_like_word_keeps_its_words():
    # "Decision:" is not a speaker, so the three words are matched as given.
    verdict = verify_item(item("Decision: ship Monday", 2, 2), LABEL_LINES)
    assert verdict == Verdict(status="verified", reason="")
    assert find_quote(LABEL_LINES, "Decision: ship Monday") == (2, 2)


def test_real_speaker_label_on_the_quote_is_still_dropped():
    assert verify_item(item("Mara Quill: Decision: ship Monday", 2, 2), LABEL_LINES).status == "verified"
    assert verify_item(item("Priya Natarajan: I'll handle Oakhollow: they asked", 1, 1), LABEL_LINES).status == "verified"
    assert find_quote(LABEL_LINES, "Mara Quill: Decision: ship Monday") == (2, 2)


def test_short_quote_rule_counts_spoken_words_only():
    # A real speaker label does not count toward the three words: two spoken words behind a
    # label are too short, and are never matched as a two-word (or one-word) needle.
    expected = Verdict(status="rejected", reason="quote too short to verify")
    assert verify_item(item("Priya Natarajan: I will", 5, 5), LINES) == expected
    assert verify_item(item("Priya Natarajan: I", 5, 5), LINES) == expected
    assert verify_item(item("Mara Quill: Great.", 1, 1), LINES) == expected
    assert find_quote(LINES, "Priya Natarajan: I will") is None
    # A capitalized word that is not a speaker counts as a spoken word and is matched.
    assert verify_item(item("Oakhollow: they asked", 1, 1), LABEL_LINES).status == "verified"
    assert find_quote(LABEL_LINES, "Oakhollow: they asked") == (1, 1)
