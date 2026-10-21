"""
Turn a meeting transcript into action items that are tied to the exact lines
they came from, and drop anything the model can't back up with a real quote.

Pipeline:
  1. Number every transcript line (L1, L2, ...).
  2. Split long transcripts into overlapping chunks.
  3. Ask Claude for action items via a forced tool call (structured JSON),
     each with the line numbers and the exact quote it is based on.
  4. Verify in code: cited lines must exist in that chunk, and the quote
     must actually appear in those lines. Unverified items are rejected.
  5. Flag low-confidence or ownerless items for human review.
  6. De-duplicate across chunk overlaps and write JSON + Markdown.

Usage:
  python actions.py samples/team_meeting.txt            # real Claude API call
  python actions.py samples/team_meeting.txt --mock     # no API key needed
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path

MODEL = os.environ.get("ACTIONS_MODEL", "claude-sonnet-5-5")
CHUNK_LINES = 120       # roughly 10-15 minutes of a typical transcript
CHUNK_OVERLAP = 10      # so items spanning a boundary aren't lost
MAX_RETRIES = 2
REVIEW_CONFIDENCE = 0.7

SYSTEM_PROMPT = """You extract action items from meeting transcripts.

Rules:
- Only extract tasks someone explicitly committed to or was explicitly asked to do.
- Every item MUST cite the line number(s) it comes from and copy an exact quote
  (verbatim, at least a few words) from those lines. Never paraphrase the quote.
- Do not invent owners or due dates. If none is stated, use null.
- Ideas, questions, or things explicitly postponed are not action items unless
  someone committed to a concrete follow-up.
- If there are no action items, return an empty list. Returning nothing is
  better than guessing."""

TOOL = {
    "name": "record_action_items",
    "description": "Record the action items found in this transcript chunk.",
    "input_schema": {
        "type": "object",
        "properties": {
            "items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "task": {"type": "string", "description": "Short imperative description."},
                        "owner": {"type": ["string", "null"], "description": "Person responsible, as named in the transcript."},
                        "due": {"type": ["string", "null"], "description": "Deadline exactly as stated, e.g. 'by Monday'."},
                        "source_lines": {"type": "array", "items": {"type": "integer"}, "minItems": 1},
                        "quote": {"type": "string", "description": "Verbatim text copied from the cited lines."},
                        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    },
                    "required": ["task", "owner", "due", "source_lines", "quote", "confidence"],
                },
            }
        },
        "required": ["items"],
    },
}


# ---------- data ----------

@dataclass
class Line:
    number: int
    text: str


@dataclass
class ActionItem:
    task: str
    owner: str | None
    due: str | None
    source_lines: list[int]
    quote: str
    confidence: float
    needs_review: bool = False
    review_reasons: list[str] = field(default_factory=list)


@dataclass
class Rejected:
    item: dict
    reason: str


# ---------- transcript handling ----------

def load_lines(path: Path) -> list[Line]:
    raw = [l.rstrip() for l in path.read_text(encoding="utf-8").splitlines()]
    return [Line(i + 1, t) for i, t in enumerate(raw) if t.strip()]


def chunk(lines: list[Line], size: int = CHUNK_LINES, overlap: int = CHUNK_OVERLAP) -> list[list[Line]]:
    if len(lines) <= size:
        return [lines]
    step = size - overlap
    return [lines[i:i + size] for i in range(0, len(lines) - overlap, step)]


def render(lines: list[Line]) -> str:
    return "\n".join(f"L{l.number}: {l.text}" for l in lines)


def normalize(s: str) -> str:
    s = s.lower().replace("’", "'").replace("“", '"').replace("”", '"')
    s = re.sub(r"^\s*l?\d+:\s*", "", s)                 # stray "L12:" prefixes
    s = re.sub(r"\[\d{1,2}:\d{2}(:\d{2})?\]", " ", s)    # timestamps
    s = re.sub(r"[^\w\s']", " ", s)                      # punctuation
    return re.sub(r"\s+", " ", s).strip()


def strip_speaker(s: str) -> str:
    """'[00:00:31] Dan: Yes, I'll book...' -> 'Yes, I'll book...'"""
    return re.sub(r"^\s*(\[[\d:]+\]\s*)?[A-Z][\w .'-]{0,30}:\s+", "", s)


# ---------- verification (the part that keeps output honest) ----------

def verify(raw: dict, chunk_lines: list[Line]) -> tuple[ActionItem | None, str | None]:
    """Return (item, None) if the item is grounded in the transcript, else (None, reason)."""
    by_number = {l.number: l.text for l in chunk_lines}
    cited = raw.get("source_lines") or []
    quote = (raw.get("quote") or "").strip()

    if not cited:
        return None, "no source lines cited"
    missing = [n for n in cited if n not in by_number]
    if missing:
        return None, f"cited line(s) {missing} are not in this part of the transcript"
    if len(normalize(quote).split()) < 3:
        return None, "quote too short to verify"

    # Allow the quote to span consecutive cited lines. Speaker labels sit between
    # lines ("... booked? Dan: Yes, I'll ..."), so strip them before matching.
    cited_text = normalize(" ".join(strip_speaker(by_number[n]) for n in sorted(set(cited))))
    if normalize(strip_speaker(quote)) not in cited_text:
        return None, "quote does not appear in the cited lines (likely invented or paraphrased)"

    item = ActionItem(
        task=raw["task"].strip(),
        owner=(raw.get("owner") or None),
        due=(raw.get("due") or None),
        source_lines=sorted(set(cited)),
        quote=quote,
        confidence=float(raw.get("confidence", 0)),
    )

    if item.confidence < REVIEW_CONFIDENCE:
        item.review_reasons.append(f"low model confidence ({item.confidence:.2f})")
    if not item.owner:
        item.review_reasons.append("no owner stated")
    elif normalize(item.owner) not in cited_text and normalize(item.owner) not in normalize(
        " ".join(by_number.values())
    ):
        item.review_reasons.append("owner name not found in transcript")
    item.needs_review = bool(item.review_reasons)
    return item, None


def dedupe(items: list[ActionItem]) -> list[ActionItem]:
    """Chunk overlaps can produce the same item twice; keep the most confident copy."""
    best: dict[str, ActionItem] = {}
    for it in items:
        key = normalize(it.quote)
        if key not in best or it.confidence > best[key].confidence:
            best[key] = it
    return sorted(best.values(), key=lambda i: i.source_lines[0])


# ---------- model call ----------

def extract_chunk(client, chunk_lines: list[Line]) -> list[dict]:
    messages = [{
        "role": "user",
        "content": "Transcript (each line is prefixed with its line number):\n\n"
                   f"{render(chunk_lines)}\n\nRecord every action item using the tool.",
    }]
    last_error = None
    for _ in range(MAX_RETRIES + 1):
        resp = client.messages.create(
            model=MODEL,
            max_tokens=2000,
            temperature=0,
            system=SYSTEM_PROMPT,
            tools=[TOOL],
            tool_choice={"type": "tool", "name": TOOL["name"]},
            messages=messages,
        )
        block = next((b for b in resp.content if getattr(b, "type", "") == "tool_use"), None)
        problem = None
        if block is None:
            problem = "No tool call returned."
        elif not isinstance(block.input.get("items"), list):
            problem = "'items' must be a list."
        else:
            bad = [i for i, it in enumerate(block.input["items"])
                   if not all(k in it for k in TOOL["input_schema"]["properties"]["items"]["items"]["required"])]
            if bad:
                problem = f"Items {bad} are missing required fields."
        if problem is None:
            return block.input["items"]

        # Tell the model exactly what was wrong and let it try again.
        last_error = problem
        messages += [
            {"role": "assistant", "content": resp.content},
            {"role": "user", "content": [{
                "type": "tool_result", "tool_use_id": block.id, "is_error": True,
                "content": f"Invalid output: {problem} Fix it and call the tool again.",
            }] if block else f"Invalid output: {problem} Call the record_action_items tool."},
        ]
    raise RuntimeError(f"Model output still invalid after retries: {last_error}")


def run(path: Path, client) -> dict:
    lines = load_lines(path)
    kept: list[ActionItem] = []
    rejected: list[Rejected] = []
    for part in chunk(lines):
        for raw in extract_chunk(client, part):
            item, reason = verify(raw, part)
            if item:
                kept.append(item)
            else:
                rejected.append(Rejected(raw, reason))
    kept = dedupe(kept)
    return {
        "source": str(path),
        "lines": len(lines),
        "action_items": [asdict(i) for i in kept],
        "rejected": [asdict(r) for r in rejected],
    }


# ---------- output ----------

def to_markdown(result: dict) -> str:
    out = [f"# Action items: {Path(result['source']).name}", ""]
    ok = [i for i in result["action_items"] if not i["needs_review"]]
    review = [i for i in result["action_items"] if i["needs_review"]]

    def row(i):
        lines = ", ".join(f"L{n}" for n in i["source_lines"])
        return (f"- **{i['task']}** | Owner: {i['owner'] or 'unassigned'} | Due: {i['due'] or 'not stated'}\n"
                f"  > \"{i['quote']}\" ({lines})")

    out += ["## Verified", *(row(i) for i in ok)] if ok else ["## Verified", "_None_"]
    if review:
        out += ["", "## Needs human review"]
        out += [row(i) + f"\n  Reason: {'; '.join(i['review_reasons'])}" for i in review]
    if result["rejected"]:
        out += ["", "## Rejected (could not be verified against the transcript)"]
        out += [f"- ~~{r['item'].get('task', '?')}~~ ({r['reason']})" for r in result["rejected"]]
    return "\n".join(out) + "\n"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("transcript", type=Path)
    p.add_argument("--mock", action="store_true", help="use canned model output (no API key needed)")
    p.add_argument("--out", type=Path, default=Path("output"))
    args = p.parse_args(argv)

    if args.mock:
        from mock_client import MockClient
        client = MockClient()
    else:
        if not os.environ.get("ANTHROPIC_API_KEY"):
            sys.exit("Set ANTHROPIC_API_KEY, or run with --mock.")
        import anthropic
        client = anthropic.Anthropic()

    result = run(args.transcript, client)
    args.out.mkdir(exist_ok=True)
    stem = args.transcript.stem
    (args.out / f"{stem}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    md = to_markdown(result)
    (args.out / f"{stem}.md").write_text(md, encoding="utf-8")
    print(md)
    print(f"Saved {args.out / stem}.json and .md  |  "
          f"{len(result['action_items'])} kept, {len(result['rejected'])} rejected")


if __name__ == "__main__":
    main()
