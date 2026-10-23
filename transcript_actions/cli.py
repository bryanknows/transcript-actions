"""Command line entry point.

    transcript-actions run <transcript.txt> [--out out/] [--model NAME] [--chunk-size N] [--overlap N]
    transcript-actions verify-only <transcript.txt> <items.json> [--out out/]

`run` calls Claude once per chunk and verifies every returned quote in code. `verify-only` skips
the model and runs just the code-level check on an items file in the tool-input shape, so the
verifier can be demonstrated without an API key.
"""

import argparse
import json
import sys
from typing import List, Optional

from .extract import DEFAULT_MODEL, MODEL_ENV_VAR, items_from_tool_input, make_client
from .pipeline import Result, run as run_pipeline, summary_line, verify_raw_items, write_outputs
from .transcript import load_transcript

CREDENTIALS_HINT = (
    "Could not authenticate with the Anthropic API: set ANTHROPIC_API_KEY or run `ant auth login`."
)
CONNECTION_HINT = (
    "Could not reach the Anthropic API: check your network, then set ANTHROPIC_API_KEY "
    "or run `ant auth login`."
)
ITEMS_SHAPE_HINT = 'Items file must be {"items": [...]} or a bare list of item objects.'


def _one_line(exc: BaseException) -> str:
    """An exception's message with any line breaks collapsed, so every failure the CLI
    reports is exactly one line on stderr."""
    return " ".join(str(exc).split())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="transcript-actions",
        description="Turn a meeting transcript into action items whose quotes are verified in code.",
    )
    subparsers = parser.add_subparsers(dest="command")

    run_parser = subparsers.add_parser("run", help="extract with Claude, then verify every quote")
    run_parser.add_argument("transcript", help="path to the transcript text file")
    run_parser.add_argument("--out", default="out", help="output directory (default: out/)")
    run_parser.add_argument(
        "--model",
        default=None,
        help="model name (default: %s, or $%s when set)" % (DEFAULT_MODEL, MODEL_ENV_VAR),
    )
    run_parser.add_argument("--chunk-size", type=int, default=120, help="lines per chunk (default: 120)")
    run_parser.add_argument("--overlap", type=int, default=10, help="lines shared between chunks (default: 10)")

    verify_parser = subparsers.add_parser(
        "verify-only", help="run only the code-level quote check on an existing items file"
    )
    verify_parser.add_argument("transcript", help="path to the transcript text file")
    verify_parser.add_argument("items", help="path to a JSON file shaped like the tool input: {\"items\": [...]}")
    verify_parser.add_argument("--out", default=None, help="also write verified/review/rejected JSON here")

    return parser


def _print_paths(paths: List[str]) -> None:
    for path in paths:
        print(path)


def _cmd_run(args: argparse.Namespace) -> int:
    import anthropic

    if args.chunk_size <= args.overlap or args.overlap < 0:
        print("--chunk-size must be greater than --overlap, and --overlap must be at least 0.", file=sys.stderr)
        return 1

    try:
        lines = load_transcript(args.transcript)
    except OSError as exc:
        print("Could not read transcript: %s" % exc, file=sys.stderr)
        return 1

    # make_client() sits inside the try because the SDK resolves the credential chain in the
    # constructor: a mis-pointed ANTHROPIC_PROFILE or ANTHROPIC_CONFIG_DIR raises there.
    try:
        client = make_client()
        result = run_pipeline(
            lines,
            client,
            model=args.model,
            chunk_size=args.chunk_size,
            overlap=args.overlap,
        )
    except anthropic.AuthenticationError:
        # Must stay before APIStatusError, which it subclasses.
        print(CREDENTIALS_HINT, file=sys.stderr)
        return 2
    except anthropic.APIConnectionError:
        print(CONNECTION_HINT, file=sys.stderr)
        return 2
    except anthropic.APIStatusError as exc:
        # Every other HTTP failure: unknown model (404), bad request (400), rate limit (429),
        # overloaded (529) and the rest.
        print(
            "Anthropic API error %d (%s): %s" % (exc.status_code, exc.type or "unknown", _one_line(exc)),
            file=sys.stderr,
        )
        return 2
    except anthropic.AnthropicError as exc:
        # The SDK's base class, raised without an HTTP status when the credential chain itself
        # fails: a missing or malformed `ant auth login` profile at construction, or an expired
        # or revoked profile token at request time. The three branches above all subclass it,
        # so this one has to come last.
        print("%s (%s)" % (CREDENTIALS_HINT, _one_line(exc)), file=sys.stderr)
        return 2
    except TypeError as exc:
        # The SDK raises a plain TypeError at call time when no credential source exists at all.
        if "authentication method" in str(exc):
            print(CREDENTIALS_HINT, file=sys.stderr)
            return 2
        raise

    try:
        paths = write_outputs(result, args.out)
    except OSError as exc:
        print("Could not write outputs: %s" % exc, file=sys.stderr)
        return 1
    print(summary_line(result.counts()))
    _print_paths(paths)
    return 0


def _format_item_line(item) -> str:
    where = "%d-%d" % (item.line_start, item.line_end)
    head = "%-9s lines %-9s %-10s %s" % (item.status, where, item.owner or "-", item.task or item.quote)
    if item.reason:
        return head + "  (" + item.reason + ")"
    return head


def _cmd_verify_only(args: argparse.Namespace) -> int:
    try:
        lines = load_transcript(args.transcript)
    except OSError as exc:
        print("Could not read transcript: %s" % exc, file=sys.stderr)
        return 1

    try:
        with open(args.items, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, ValueError) as exc:
        print("Could not read items file: %s" % exc, file=sys.stderr)
        return 1

    if isinstance(data, list):
        data = {"items": data}
    # A misspelled key, a string or a number would otherwise read as "nothing to verify" and
    # exit 0 with all-zero counts, which is indistinguishable from a clean run.
    if not isinstance(data, dict) or not isinstance(data.get("items"), list):
        print(ITEMS_SHAPE_HINT, file=sys.stderr)
        return 1
    raw_items = items_from_tool_input(data)

    result = Result()
    for item in verify_raw_items(raw_items, lines):
        result.add(item)

    for status in ("verified", "review", "rejected"):
        for item in result.bucket(status):
            print(_format_item_line(item))
    print(summary_line(result.counts()))

    if args.out:
        try:
            paths = write_outputs(result, args.out)
        except OSError as exc:
            print("Could not write outputs: %s" % exc, file=sys.stderr)
            return 1
        _print_paths(paths)
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "run":
        return _cmd_run(args)
    if args.command == "verify-only":
        return _cmd_verify_only(args)
    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
