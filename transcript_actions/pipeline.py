"""Orchestrate: chunk the transcript, extract items per chunk, verify each quote in code, and
bucket the results by verdict."""

import json
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from .extract import extract_chunk
from .transcript import Line, chunk_lines
from .verify import RawItem, Verdict, normalize, verify_item

STATUSES = ("verified", "review", "rejected")


@dataclass
class ResultItem:
    task: str
    owner: str
    due: str
    quote: str
    line_start: int
    line_end: int
    status: str
    reason: str
    found_start: Optional[int] = None
    found_end: Optional[int] = None
    chunk_index: int = 0


@dataclass
class Result:
    verified: List[ResultItem] = field(default_factory=list)
    review: List[ResultItem] = field(default_factory=list)
    rejected: List[ResultItem] = field(default_factory=list)

    def bucket(self, status: str) -> List[ResultItem]:
        """The list for a status. Unknown statuses land in rejected so nothing is silently lost."""
        if status == "verified":
            return self.verified
        if status == "review":
            return self.review
        return self.rejected

    def add(self, item: ResultItem) -> None:
        self.bucket(item.status).append(item)

    def counts(self) -> Dict[str, int]:
        return {
            "verified": len(self.verified),
            "review": len(self.review),
            "rejected": len(self.rejected),
        }

    def to_dicts(self) -> Dict[str, List[dict]]:
        return {
            "verified": [asdict(item) for item in self.verified],
            "review": [asdict(item) for item in self.review],
            "rejected": [asdict(item) for item in self.rejected],
        }


def result_item(raw: RawItem, verdict: Verdict, chunk_index: int = 0) -> ResultItem:
    """Join a RawItem with its Verdict."""
    return ResultItem(
        task=raw.task,
        owner=raw.owner,
        due=raw.due,
        quote=raw.quote,
        line_start=raw.line_start,
        line_end=raw.line_end,
        status=verdict.status,
        reason=verdict.reason,
        found_start=verdict.found_start,
        found_end=verdict.found_end,
        chunk_index=chunk_index,
    )


def verify_raw_items(raw_items: List[RawItem], lines: List[Line], chunk_index: int = 0) -> List[ResultItem]:
    """Run the code-level quote check on each item and attach the verdict."""
    return [result_item(raw, verify_item(raw, lines), chunk_index) for raw in raw_items]


def dedupe(items: List[ResultItem]) -> List[ResultItem]:
    """Overlapping chunks can return the same item twice. Two items are duplicates when their
    normalized quotes are equal; keep the first."""
    seen = set()
    kept = []  # type: List[ResultItem]
    for item in items:
        key = normalize(item.quote)
        if key in seen:
            continue
        seen.add(key)
        kept.append(item)
    return kept


def summary_line(counts: Dict[str, int]) -> str:
    """The one-line summary the CLI prints: `verified=3 review=1 rejected=2`."""
    return " ".join("%s=%d" % (status, counts.get(status, 0)) for status in STATUSES)


def run(
    lines: List[Line],
    client: Any,
    model: Optional[str] = None,
    chunk_size: int = 120,
    overlap: int = 10,
) -> Result:
    """chunk -> extract_chunk per chunk -> verify_item per item -> bucket by status -> dedupe each bucket."""
    result = Result()
    for chunk in chunk_lines(lines, size=chunk_size, overlap=overlap):
        raw_items = extract_chunk(chunk, client, model)
        for item in verify_raw_items(raw_items, lines, chunk.index):
            result.add(item)
    result.verified = dedupe(result.verified)
    result.review = dedupe(result.review)
    result.rejected = dedupe(result.rejected)
    return result


def write_outputs(result: Result, out_dir: str) -> List[str]:
    """Write verified.json, review.json, rejected.json (indent=2, sorted keys) and return the paths.

    Each file holds a JSON array of item objects, in the bucket's order."""
    os.makedirs(out_dir, exist_ok=True)
    dicts = result.to_dicts()
    paths = []  # type: List[str]
    for status in STATUSES:
        path = os.path.join(out_dir, status + ".json")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(dicts[status], handle, indent=2, sort_keys=True)
            handle.write("\n")
        paths.append(path)
    return paths
