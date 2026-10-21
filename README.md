# Transcript → Verified Action Items

A small Python tool that turns a meeting transcript (e.g. a Fireflies export) into action items, **each linked back to the exact lines it came from**. Anything the model can't back up with a real quote is rejected in code, not trusted.

## The problem

On long meetings, LLMs return action items that sound right but were never said: invented owners, made-up deadlines, paraphrased "quotes." You can't put that into ClickUp.

## How it works

```
transcript ─► number every line ─► split into overlapping chunks
          ─► Claude API (forced tool call → structured JSON)
          ─► verify each item in code ─► flag weak ones ─► dedupe ─► JSON + Markdown
```

1. **Line numbers.** Every line is labelled `L1, L2, …` so the model can cite precisely.
2. **Chunking.** Long transcripts are split into ~120-line chunks with a 10-line overlap, so nothing falls through a boundary.
3. **Structured output.** Claude must answer through a tool with a strict schema: `task, owner, due, source_lines, quote, confidence`. Temperature is 0, and the prompt says returning nothing beats guessing.
4. **Verification in code (the key part).**
   - Cited lines must exist in the chunk that was sent.
   - The quote must actually appear in those cited lines (case and punctuation tolerant, can span consecutive lines).
   - Quotes under 3 words are rejected as too weak to verify.
5. **Human review.** Real but weak items (low confidence, no owner, owner not in the transcript) are kept but flagged instead of being auto-assigned.
6. **Dedupe.** Overlapping chunks can produce the same item twice; the most confident copy wins.

Every rejection is kept in the output with its reason, so you can audit what was thrown away.

## Run it

```bash
pip install -r requirements.txt

# Demo without an API key (canned model output that includes deliberate hallucinations)
python actions.py samples/team_meeting.txt --mock

# Real run
export ANTHROPIC_API_KEY=sk-ant-...
python actions.py path/to/transcript.txt

# Tests
python -m pytest -q
```

Output lands in `output/<name>.md` and `output/<name>.json`. Model can be changed with `ACTIONS_MODEL`.

## Example output (mock run)

```
## Verified
- Book cleaners and painter for Maple Street unit 2 and send quotes | Owner: Dan | Due: by Monday
  > "I'll book both by Monday and send you the quotes" (L3, L4)
...
## Needs human review
- Revisit switching accountants | Owner: unassigned | Due: January
  Reason: low model confidence (0.50); no owner stated

## Rejected (could not be verified against the transcript)
- ~~Call the accountant to schedule a review~~ (quote does not appear in the cited lines)
- ~~Send the updated rent roll~~ (cited line(s) [42] are not in this part of the transcript)
- ~~Book the cleaners~~ (quote does not appear in the cited lines)
```

## What broke and how I fixed it

- **Multi-line quotes were wrongly rejected.** A commitment often spans two lines ("…can you get it booked?" / "Dan: Yes, I'll book both…"). The speaker label sitting between the lines broke exact matching, so real items were thrown out. Fix: strip timestamps and speaker labels before comparing, and added a test for quotes that span lines.
- **Malformed responses.** If the model returns an item missing a required field, the tool sends the specific error back as a `tool_result` and asks it to correct itself (up to 2 retries) instead of crashing or silently dropping data.
- **Duplicates from chunk overlap.** The overlap that prevents lost items also produced double items, so results are de-duplicated by normalized quote, keeping the highest-confidence version.

## Next steps

- Read Fireflies exports directly via their API
- Push verified items to ClickUp as tasks with a link back to the transcript timestamp
- Generate SOP drafts from the same verified segments
