# transcript-actions

Turn a meeting transcript into a list of action items where every item links back to the exact lines it came from. Claude reads the transcript in numbered chunks and proposes items, each with a verbatim quote and the line numbers that quote sits on. Plain Python then checks every quote against the transcript before anything is reported. An item whose quote really is on the lines the model named is verified; one whose quote exists somewhere else in the transcript goes to review; one whose quote cannot be found at all, or is too short to be worth checking, is rejected. The model never gets the last word on what was said.

## How it works

The pipeline has four stages, and only the second one talks to the API.

First, `transcript.py` loads the file and numbers every line, blank lines included, so the numbers match what an editor shows. A line that begins with a capitalized name and a colon (`Marisol: Morning, everyone.`) has that label split off as the speaker; the remainder is the spoken text. The lines are then cut into overlapping chunks, 120 lines with a 10 line overlap by default, and each chunk is rendered with a `NNNN | ` prefix on every line so the model always has a number to cite.

Second, `extract.py` sends each rendered chunk to Claude with a single tool, `record_action_items`, whose input schema (in `schema.py`) requires `task`, `quote`, `line_start` and `line_end` for every item and allows `owner` and `due`. The system prompt tells the model to record only commitments a specific person actually made, to copy the quote word for word, to take line numbers from the prefixes, never to invent anything, and to call the tool with an empty list when nobody committed to anything. The tool input arrives as a dict and is turned into `RawItem` records without parsing any text; a missing owner or due date becomes an empty string, and a non-integer line number zeroes both line fields so the verifier rejects that item instead of crashing.

Third, `verify.py` checks every item against the transcript. It is pure code with no I/O and no model access. The rules run in order, and the first one that applies decides the verdict:

| Rule | Condition | Verdict |
|---|---|---|
| 1 | Quote is empty or, once a leading label naming a transcript speaker is dropped, has fewer than 3 normalized words | rejected, "quote too short to verify" |
| 2 | `line_start` or `line_end` is outside 1..N, or start > end | rejected, "line range out of bounds" |
| 3 | Normalized quote appears, whole-word, inside the claimed lines | verified |
| 4 | Quote appears somewhere else in the transcript | review, "quote found at lines X-Y, not A-B" |
| 5 | Quote appears nowhere | rejected, "quote not found in transcript" |

Normalization lowercases the text, maps curly apostrophes to straight ones, treats every punctuation mark other than an apostrophe as a word separator, drops an apostrophe at either end of a word (there it is a quotation mark, not part of the word, so a commitment the transcript wraps in 'single quotes' still matches), and collapses whitespace. A leading `Name:` label on the quote is dropped before matching only when Name is a speaker the transcript carries; any other capitalized word before a colon, such as `Decision:`, is spoken text and is matched like any other word. Matching is on whole words, so "ship the beta" does not match inside "reship the beta". The same matching rule is used for the claimed window (rule 3) and the whole-transcript search (rule 4), so the two can never disagree about whether a quote exists. For a review verdict the verifier also records `found_start` and `found_end`, the smallest window where the quote actually appears.

Fourth, `pipeline.py` attaches each verdict to its item, buckets the results into `verified`, `review` and `rejected`, removes the duplicates that overlapping chunks produce, and writes one JSON array per bucket. `cli.py` wraps all of this in two subcommands.

### Modules and data shapes

The package is small enough to read in one sitting. Each module has one job. `Line`, `Chunk`, `RawItem` and `Verdict` are frozen dataclasses; `ResultItem` and `Result` are ordinary dataclasses because `run()` replaces the bucket lists after dedupe.

| Module | Role | Public shapes |
|---|---|---|
| `transcript.py` | load, number, chunk, render | `Line(number, speaker, text, raw)`, `Chunk(index, start, end, lines)`, `parse_transcript`, `load_transcript`, `chunk_lines(lines, size=120, overlap=10)`, `render_chunk` |
| `verify.py` | the quote check, no I/O | `RawItem(task, owner, due, quote, line_start, line_end)`, `Verdict(status, reason, found_start, found_end)`, `normalize`, `strip_speaker`, `window_text`, `find_quote`, `verify_item` |
| `schema.py` | the tool definition and system prompt | `TOOL_NAME`, `ACTION_ITEMS_TOOL`, `SYSTEM_PROMPT` |
| `extract.py` | one API call per chunk | `DEFAULT_MODEL`, `items_from_tool_input`, `extract_chunk(chunk, client, model=None)`, `make_client` |
| `pipeline.py` | orchestrate and write | `ResultItem` (the `RawItem` fields plus `status`, `reason`, `found_start`, `found_end`, `chunk_index`), `Result(verified, review, rejected)`, `dedupe`, `run`, `write_outputs` |
| `cli.py` | argparse entry point | `main(argv=None) -> int` |

The API call in `extract_chunk` is one `client.messages.create` with the model name, `max_tokens=16000`, the system prompt, `tools=[ACTION_ITEMS_TOOL]`, `tool_choice={"type": "auto"}` and the rendered chunk as the single user message. No thinking parameter is passed. If the response stops with a refusal, or carries no `tool_use` block named `record_action_items`, the chunk contributes no items. A refusal also prints one line on stderr naming the chunk, its line range and the refusal category, and a response cut off at `max_tokens` prints a note saying so, so neither is mistaken for a chunk in which nobody committed to anything. The client is injected, which is how the tests run the whole pipeline against a fake.

The default model is `claude-opus-5-5`. The `TRANSCRIPT_ACTIONS_MODEL` environment variable overrides it, and `--model` on the command line overrides both.

## Why the check is in code

The model proposes, the code disposes. A language model asked for "the exact words" will, some fraction of the time, hand back a tidy paraphrase, a quote assembled from two different speakers, or a line number that is off by a few. Each of those looks perfectly plausible on its own. The only way to know whether a quote is real is to look for it in the transcript, and that is a string search, not a judgment call, so it belongs in deterministic code that runs the same way every time and can be tested without an API key.

Paraphrases are rejected on purpose. A quote that captures the gist but changes the words cannot be checked, and an action item that cannot be checked is exactly the kind that ends up assigned to someone who never agreed to it. The sample transcript includes a rewritten version of a real commitment ("I'll rewrite the export service in Rust before the demo") and the verifier rejects it, because nobody said that. It also includes a real quote cited at the wrong line, the recap at line 124 instead of the original at line 14, and that goes to review with both locations named, since the words are genuine and a person can decide in seconds which citation to keep.

## Setup

Python 3.9 or newer is required, and the only runtime dependency is the `anthropic` package.

```sh
python3 -m venv .venv
. .venv/bin/activate
pip install --upgrade pip
pip install -e ".[dev]"
```

The `pip install --upgrade pip` line matters on a stock macOS Python 3.9, whose bundled pip (21.2) cannot do an editable install of a project that has only a `pyproject.toml`; pip 21.3 or newer can. The `[dev]` extra adds `pytest`. The editable install puts a `transcript-actions` console script on the path of the virtual environment.

The `run` subcommand needs Anthropic credentials. Either export an API key:

```sh
export ANTHROPIC_API_KEY=...
```

or sign in once with the Anthropic CLI, which stores a profile the SDK picks up on its own:

```sh
ant auth login
```

The code does not inspect the environment for credentials itself. If none can be found, the key is rejected, or an `ant auth login` profile is missing, malformed or expired, `run` prints one line on stderr telling you to set `ANTHROPIC_API_KEY` or run `ant auth login`, and exits with status 2. Any other API failure, such as an unknown model name, a rate limit or an outage, prints one line carrying the HTTP status and the API's error type, and also exits 2. `verify-only` needs no credentials at all.

## Usage

There are two subcommands. `run` does the whole pipeline, model call included. `verify-only` skips the model and runs just the code-level check on an items file you already have, so the verifier can be demonstrated, and tested, without touching the API.

To extract action items from the sample transcript with Claude:

```sh
transcript-actions run samples/transcript.txt --out out
```

`run` prints a one-line summary followed by the three output paths, and exits 0:

```
verified=7 review=1 rejected=0
out/verified.json
out/review.json
out/rejected.json
```

The counts above are illustrative. What the model returns varies from call to call, which is the reason the verifier exists. `--model NAME`, `--chunk-size N` and `--overlap N` are accepted; the chunk size must be larger than the overlap, and the overlap must be at least zero, or the command exits 1 before making any call. If the output directory cannot be created or written to, for example because `--out` names an existing file, either subcommand prints `Could not write outputs: ...` on stderr and exits 1.

To run only the verifier against the hand-written items file that ships with the samples:

```sh
transcript-actions verify-only samples/transcript.txt samples/items.json --out samples/output
```

`verify-only` prints one line per item, then the same summary line, then the output paths when `--out` was given:

```
verified  lines 20-20     Callum     Ship the keyboard shortcut fix behind a feature flag
verified  lines 87-87     Ingrid     Put the empty-state mockups in the shared folder
verified  lines 70-70     Theo       Write the migration script for the retention table
verified  lines 33-34     Marisol    Send the pricing page draft to the whole team
review    lines 124-124   Desmond    Open the ticket for the export timeout and self-assign it  (quote found at lines 14-14, not 124-124)
rejected  lines 44-44     Theo       Rewrite the export service before the demo  (quote not found in transcript)
rejected  lines 39-39     Ingrid     Add the comparison table note to the draft comments  (quote too short to verify)
rejected  lines 970-971   Beatrix    Email Pembrook the data residency summary  (line range out of bounds)
verified=4 review=1 rejected=3
samples/output/verified.json
samples/output/review.json
samples/output/rejected.json
```

Without `--out`, `verify-only` prints and writes nothing to disk. The items file is either the exact tool-input shape, `{"items": [...]}`, or a bare list of item objects. Any other shape, such as a misspelled `items` key, exits 1 with a message on stderr rather than reporting zero items as if the run were clean.

Each output file is a JSON array with one object per item, keys sorted, carrying the six fields the model supplied (`task`, `owner`, `due`, `quote`, `line_start`, `line_end`) plus `status`, `reason`, `found_start`, `found_end` and `chunk_index`. `found_start` and `found_end` are null except on review items, where they point at the lines the quote was actually found on.

## What's in samples/

`samples/transcript.txt` is a 140 line product sync at a fictional software startup, Fennelworks, which makes a log search tool called Tidewrack. Every speaker (Marisol, Desmond, Theo, Callum, Ingrid, Beatrix) and every customer named in it (Oakhollow Logistics, Pembrook) is invented. The transcript was written to exercise the verifier: three commitments are split across two lines with a different speaker's label in the middle, there are suggestions and withdrawn thoughts that are not commitments, and the closing recap restates several commitments in the chair's words.

`samples/items.json` is a hand-written file in the exact shape the `record_action_items` tool returns. It holds eight items chosen to land in every bucket: four that verify (one of them the two-line quote at lines 33 and 34), one real quote cited at the recap line instead of the original, and three that are rejected for the three different reasons the verifier can give.

`samples/output/` was produced by running `verify-only` on that items file with the command shown in Usage, and it is committed so the output format can be read without running anything. To be clear about what that means: the code-level verifier ran for real to produce these files, but `items.json` stands in for what the model returns. No API call was involved in generating the sample output. Running `run` on the same transcript will give different items, and possibly different counts, because the model decides what to propose.

## Design choices

Line-numbered chunks with overlap. The transcript is sent in windows rather than whole so a long meeting fits comfortably in one request and the model has a bounded amount of text to search for exact words. The windows overlap by ten lines so a commitment that straddles a boundary is seen whole in at least one of them. Each line carries its number in the rendered text because the verifier needs to check a claimed location, and a number the model copies from the line it is reading is far more reliable than one it computes.

Speaker labels stripped before matching. In a real transcript a single sentence is often broken across two lines by a second label, as when someone finishes another person's sentence, or when the transcription tool wraps at a speaker change. The spoken words are the same either way. Stripping the labels from the transcript text and from the quote before matching means those sentences verify, and a quote the model copied together with its label still matches. On the quote side the label is dropped only when it names a speaker the transcript carries, so spoken text that happens to open with a capitalized word and a colon keeps all of its words.

Three buckets instead of two. A quote that is genuinely in the transcript but cited at the wrong lines is a different situation from one that is not there at all. Rejecting it would throw away a real commitment over a citation slip; accepting it would silently keep a wrong location. The review bucket keeps the item, names both locations, and leaves a quick decision to a person.

Dedupe by normalized quote. Overlapping chunks mean the model can see the same commitment twice and report it twice, with trivially different capitalization or punctuation. The quote is the one field that is anchored in the transcript rather than written by the model, so comparing normalized quotes catches these duplicates without comparing the task wording, which the model may phrase differently each time. The first sighting is kept.

Strict tool schema with `tool_choice` set to auto. The tool is declared strict, with `additionalProperties` set to false on both the outer object and the item object, so the API guarantees the input validates against the schema: required keys present, line numbers as integers, nothing extra. That removes a whole class of parsing defenses from the extraction code. The tool choice is left on auto because current models reject a forced tool choice; the system prompt instructs the model to always call the tool, with an empty list when there is nothing to record, and the code treats a response without the tool call as an empty result rather than an error.

## Limitations

Chunk boundaries can still cut a commitment in two if the sentence is longer than the overlap, and a chunk with a split commitment may then produce an item whose quote fails to verify. Raising `--overlap` helps at the cost of more tokens.

The verifier can only check what the model proposed. It catches invented and misplaced quotes; it cannot notice a commitment the model skipped. Recall is the model's job and will vary by model and by transcript.

Every chunk is a separate request, so a long transcript costs tokens roughly in proportion to its length, plus the overlap. There is no caching across runs.

Normalization is tuned for English transcripts: it folds case with Python's default lowercasing, treats an ASCII or curly apostrophe inside a word as part of it and one at a word's edge as a quotation mark, and splits on everything else. Transcripts in other languages or scripts may normalize poorly, and no stemming or synonym handling is attempted, by design.

There is no speaker diarization. The tool relies on the transcript already carrying `Name:` labels at the start of lines. Any capitalized word followed by a colon and a space is treated as a label, so a line beginning with `Note: ...` parses with speaker "Note". Names may contain spaces, dots, hyphens and either kind of apostrophe, so `O'Brien:` and `O’Brien:` both parse. The verifier is unaffected, since it strips labels the same way regardless, but the `owner` field is only as good as the labels in the file.

## Tests

Run the suite from the project root:

```sh
python -m pytest -q
```

The tests never call the API. `extract.py` and `pipeline.py` take the client as an argument, and the tests pass a fake whose `messages.create` returns canned responses in the SDK's shape, including refusals, text-only replies, tool blocks with the wrong name and string inputs that must not be parsed. The CLI tests replace `make_client` the same way and raise the SDK's own `AuthenticationError`, `APIConnectionError`, `NotFoundError`, `RateLimitError` and the `AnthropicError` base class from the fake, or from `make_client` itself, to check the exit codes and the one-line messages. The one test that constructs a real `anthropic.Anthropic` sets a placeholder key and never sends a request.

Coverage follows the modules: `test_transcript.py` covers numbering, blank lines, the speaker regex and its documented quirks, chunk overlap and the partial last window, and the rendered line format. `test_verify.py` walks each of the five rules in order, the two-line quote with a label in between, the smallest-window search, and whole-word matching. `test_extract.py` checks the tool schema is strict at both levels and that the request sent to the client matches the parameters above. `test_pipeline.py` runs the sample transcript end to end with the sample items, including small chunk sizes that force heavy overlap, and checks dedupe and the output files. `test_cli.py` runs both subcommands against the sample files.

## License

MIT. See `LICENSE`.
