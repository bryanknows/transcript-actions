"""The tool definition Claude must satisfy, and the system prompt that goes with it.

The tool is declared strict so the API guarantees that `tool_use.input` validates against
`input_schema` exactly: no missing required keys, no extra keys, correct types.
"""

TOOL_NAME = "record_action_items"

_ITEM_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["task", "quote", "line_start", "line_end"],
    "properties": {
        "task": {
            "type": "string",
            "description": "The commitment in one short imperative sentence.",
        },
        "owner": {
            "type": "string",
            "description": "The speaker who made the commitment, as named in the transcript label. "
            "Empty string if unknown.",
        },
        "due": {
            "type": "string",
            "description": "The deadline as spoken, for example 'Thursday' or 'end of day Monday'. "
            "Empty string if none was stated.",
        },
        "quote": {
            "type": "string",
            "description": "verbatim words from the transcript, no paraphrase",
        },
        "line_start": {
            "type": "integer",
            "description": "Line number (the NNNN | prefix) of the first line the quote comes from.",
        },
        "line_end": {
            "type": "integer",
            "description": "Line number (the NNNN | prefix) of the last line the quote comes from. "
            "Equal to line_start when the quote sits on one line.",
        },
    },
}

ACTION_ITEMS_TOOL = {
    "name": TOOL_NAME,
    "description": (
        "Record the action items found in this transcript chunk. Call it exactly once per "
        "chunk. Pass an empty items list when nobody committed to anything."
    ),
    "strict": True,
    "input_schema": {
        "type": "object",
        "additionalProperties": False,
        "required": ["items"],
        "properties": {
            "items": {
                "type": "array",
                "items": _ITEM_SCHEMA,
            }
        },
    },
}

SYSTEM_PROMPT = """You extract action items from a meeting transcript.

Every transcript line is prefixed with its line number in the form `NNNN | `. The numbers are 1-based and refer to the original file. Blank lines keep their numbers.

Rules:
1. Record only commitments that a specific person actually made: a promise to do something, usually in the first person ("I'll send", "I will have", "I can get that to you by"). Suggestions, wishes, questions, hypotheticals and "someone should" statements are not commitments. A sentence that is later explicitly withdrawn or called "just a thought" is not a commitment either.
2. The quote must be verbatim words from the transcript, copied exactly as written: no paraphrase, no ellipses, no added or dropped words. A quote may span consecutive lines when one spoken sentence is split across them. When it does, copy only the spoken words and skip the speaker label that begins the second line, because the label is not part of what was said.
3. line_start and line_end are the `NNNN |` prefixes of the first and last transcript lines the quote comes from. For a quote on a single line they are equal.
4. owner is the speaker who made the commitment, written as the name appears in the transcript label. due is the deadline as spoken ("Thursday", "end of day Monday", "this week"). Use an empty string for either when it is not stated.
5. Never invent an item, a quote or a line number. If you are not sure the words are in the transcript, leave the item out.
6. Always call the record_action_items tool, exactly once, even when nothing was committed to. In that case pass an empty items list. Do not answer in prose.
"""
