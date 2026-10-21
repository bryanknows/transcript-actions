"""
A stand-in for anthropic.Anthropic() so the pipeline can be demoed without an
API key. Its canned answers deliberately include the failure modes real models
produce on long transcripts:

  - a first response with a missing field (exercises the retry loop)
  - an invented action item with a quote that is not in the transcript
  - an item citing a line number that does not exist
  - a paraphrased "quote"
  - a duplicate item
  - a vague item with no owner and low confidence (goes to human review)

Running `python actions.py samples/team_meeting.txt --mock` shows the verifier
catching each one.
"""

from types import SimpleNamespace

_MALFORMED = {"items": [{
    "task": "Book cleaners and painter for Maple Street unit 2",
    "owner": "Dan", "due": "by Monday", "source_lines": [3, 4],
    "quote": "I'll book both by Monday and send you the quotes",
    # "confidence" missing on purpose
}]}

_GOOD = {"items": [
    # --- real, grounded items ---
    {"task": "Book cleaners and painter for Maple Street unit 2 and send quotes",
     "owner": "Dan", "due": "by Monday", "source_lines": [3, 4],
     "quote": "I'll book both by Monday and send you the quotes", "confidence": 0.95},
    {"task": "Shoot new listing photos for Maple Street unit 2",
     "owner": "Priya", "due": "Thursday afternoon", "source_lines": [6],
     "quote": "I can shoot the photos", "confidence": 0.85},
    {"task": "Pull October Suffolk County probate filings into the shared folder",
     "owner": "Priya", "due": "end of day tomorrow", "source_lines": [10, 11],
     "quote": "I'll have them in the folder by end of day tomorrow", "confidence": 0.95},
    {"task": "Flag uncategorized QuickBooks transactions over $500 and send to Kelsey",
     "owner": "Dan", "due": None, "source_lines": [15],
     "quote": "flag anything over five hundred dollars and send it to me", "confidence": 0.9},
    # --- vague: real quote, but no owner and low confidence -> human review ---
    {"task": "Revisit switching accountants",
     "owner": None, "due": "January", "source_lines": [18],
     "quote": "Let's revisit in January", "confidence": 0.5},
    # --- duplicate of the first item with lower confidence -> removed by dedupe ---
    {"task": "Book cleaners and painter",
     "owner": "Dan", "due": "Monday", "source_lines": [4],
     "quote": "I'll book both by Monday and send you the quotes", "confidence": 0.7},
    # --- hallucinations the verifier must reject ---
    {"task": "Call the accountant to schedule a review",
     "owner": "Kelsey", "due": "Friday", "source_lines": [17],
     "quote": "Kelsey will call the accountant on Friday", "confidence": 0.8},
    {"task": "Send the updated rent roll",
     "owner": "Dan", "due": None, "source_lines": [42],
     "quote": "send the updated rent roll", "confidence": 0.75},
    {"task": "Book the cleaners",
     "owner": "Dan", "due": None, "source_lines": [4],
     "quote": "Dan will book the cleaners this week", "confidence": 0.8},
]}


class _Messages:
    def __init__(self):
        self.calls = 0

    def create(self, **kwargs):
        self.calls += 1
        payload = _MALFORMED if self.calls == 1 else _GOOD
        block = SimpleNamespace(type="tool_use", id=f"toolu_mock_{self.calls}",
                                name="record_action_items", input=payload)
        return SimpleNamespace(content=[block], stop_reason="tool_use")


class MockClient:
    def __init__(self):
        self.messages = _Messages()
