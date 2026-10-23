"""Tests for cli.main() covering both subcommands. The model is replaced by a fake client via
monkeypatch; nothing here touches the network."""

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List

import anthropic
import httpx
import pytest

from transcript_actions import cli
from transcript_actions.extract import DEFAULT_MODEL, MODEL_ENV_VAR
from transcript_actions.schema import TOOL_NAME

SAMPLES = Path(__file__).resolve().parent.parent / "samples"
TRANSCRIPT = str(SAMPLES / "transcript.txt")
ITEMS = str(SAMPLES / "items.json")

SUMMARY_FOR_SAMPLES = "verified=4 review=1 rejected=3"


# --- fakes -------------------------------------------------------------------------------


def tool_block(items: List[Dict[str, Any]]) -> SimpleNamespace:
    return SimpleNamespace(type="tool_use", id="toolu_01", name=TOOL_NAME, input={"items": items})


def fake_response(blocks: List[Any], stop_reason: str = "tool_use") -> SimpleNamespace:
    return SimpleNamespace(stop_reason=stop_reason, content=list(blocks), stop_details=None)


class FakeMessages:
    def __init__(self, response: Any = None, exc: Any = None) -> None:
        self._response = response
        self._exc = exc
        self.calls = []  # type: List[Dict[str, Any]]

    def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if self._exc is not None:
            raise self._exc
        return self._response


class FakeClient:
    def __init__(self, response: Any = None, exc: Any = None) -> None:
        self.messages = FakeMessages(response, exc)


def sample_items() -> List[Dict[str, Any]]:
    with open(ITEMS, "r", encoding="utf-8") as handle:
        return json.load(handle)["items"]


def install_fake_client(monkeypatch, client: FakeClient) -> FakeClient:
    monkeypatch.setattr(cli, "make_client", lambda: client)
    return client


def fake_request() -> httpx.Request:
    return httpx.Request("POST", "https://example.invalid/v1/messages")


def auth_error() -> anthropic.AuthenticationError:
    response = httpx.Response(401, request=fake_request())
    return anthropic.AuthenticationError("invalid x-api-key", response=response, body=None)


def connection_error() -> anthropic.APIConnectionError:
    return anthropic.APIConnectionError(request=fake_request())


def not_found_error() -> anthropic.NotFoundError:
    response = httpx.Response(404, request=fake_request())
    return anthropic.NotFoundError("model not found", response=response, body=None)


def rate_limit_error() -> anthropic.RateLimitError:
    body = {"type": "error", "error": {"type": "rate_limit_error", "message": "rate limited"}}
    response = httpx.Response(429, request=fake_request())
    return anthropic.RateLimitError("rate limited", response=response, body=body)


def credential_chain_error(message: str) -> anthropic.AnthropicError:
    """The SDK's base class, which the credential chain raises without an HTTP status."""
    return anthropic.AnthropicError(message)


# --- verify-only --------------------------------------------------------------------------


def test_verify_only_on_samples_prints_summary(capsys):
    code = cli.main(["verify-only", TRANSCRIPT, ITEMS])
    out = capsys.readouterr().out
    assert code == 0
    assert SUMMARY_FOR_SAMPLES in out.splitlines()
    assert "quote too short to verify" in out
    assert "line range out of bounds" in out
    assert "quote not found in transcript" in out
    assert "quote found at lines 14-14, not 124-124" in out


def test_verify_only_writes_outputs_when_out_is_given(tmp_path, capsys):
    out_dir = tmp_path / "out"
    code = cli.main(["verify-only", TRANSCRIPT, ITEMS, "--out", str(out_dir)])
    out = capsys.readouterr().out
    assert code == 0
    assert SUMMARY_FOR_SAMPLES in out
    for name in ("verified.json", "review.json", "rejected.json"):
        path = out_dir / name
        assert path.is_file()
        assert str(path) in out
    with open(str(out_dir / "verified.json"), "r", encoding="utf-8") as handle:
        assert len(json.load(handle)) == 4
    with open(str(out_dir / "rejected.json"), "r", encoding="utf-8") as handle:
        assert len(json.load(handle)) == 3


def test_verify_only_does_not_write_without_out(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    assert cli.main(["verify-only", TRANSCRIPT, ITEMS]) == 0
    assert not (tmp_path / "out").exists()
    assert not (tmp_path / "verified.json").exists()


def test_verify_only_accepts_a_bare_list(tmp_path, capsys):
    items_path = tmp_path / "bare.json"
    items_path.write_text(json.dumps(sample_items()[:2]), encoding="utf-8")
    code = cli.main(["verify-only", TRANSCRIPT, str(items_path)])
    out = capsys.readouterr().out
    assert code == 0
    assert "verified=2 review=0 rejected=0" in out


def test_verify_only_with_missing_items_file(tmp_path, capsys):
    code = cli.main(["verify-only", TRANSCRIPT, str(tmp_path / "nope.json")])
    err = capsys.readouterr().err
    assert code == 1
    assert "items file" in err


def test_verify_only_with_invalid_json(tmp_path, capsys):
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    code = cli.main(["verify-only", TRANSCRIPT, str(bad)])
    assert code == 1
    assert "items file" in capsys.readouterr().err


def test_verify_only_with_missing_transcript(tmp_path, capsys):
    code = cli.main(["verify-only", str(tmp_path / "missing.txt"), ITEMS])
    assert code == 1
    assert "transcript" in capsys.readouterr().err


def test_verify_only_rejects_an_object_without_an_items_list(tmp_path, capsys):
    typo = tmp_path / "typo.json"
    typo.write_text(json.dumps({"item": sample_items()[:2]}), encoding="utf-8")
    code = cli.main(["verify-only", TRANSCRIPT, str(typo)])
    captured = capsys.readouterr()
    assert code == 1
    assert '{"items": [...]}' in captured.err
    assert "bare list" in captured.err
    assert "verified=" not in captured.out


@pytest.mark.parametrize("payload", ['"hello"', "42", "null", '{"items": "x"}', '{"items": {"a": 1}}'])
def test_verify_only_rejects_items_files_of_the_wrong_shape(tmp_path, capsys, payload):
    bad = tmp_path / "bad.json"
    bad.write_text(payload, encoding="utf-8")
    code = cli.main(["verify-only", TRANSCRIPT, str(bad)])
    captured = capsys.readouterr()
    assert code == 1
    assert '{"items": [...]}' in captured.err
    assert captured.out == ""


def test_verify_only_out_pointing_at_a_file_exits_1(tmp_path, capsys):
    blocker = tmp_path / "afile"
    blocker.write_text("not a directory", encoding="utf-8")
    code = cli.main(["verify-only", TRANSCRIPT, ITEMS, "--out", str(blocker)])
    captured = capsys.readouterr()
    assert code == 1
    assert captured.err.startswith("Could not write outputs: ")
    assert len(captured.err.strip().splitlines()) == 1
    assert blocker.read_text(encoding="utf-8") == "not a directory"


# --- run ----------------------------------------------------------------------------------------


def test_run_with_fake_client_writes_outputs_and_prints_summary(tmp_path, monkeypatch, capsys):
    client = install_fake_client(monkeypatch, FakeClient(fake_response([tool_block(sample_items())])))
    out_dir = tmp_path / "out"

    code = cli.main(["run", TRANSCRIPT, "--out", str(out_dir)])
    out = capsys.readouterr().out

    assert code == 0
    lines = out.splitlines()
    assert lines[0] == SUMMARY_FOR_SAMPLES
    assert lines[1:] == [
        str(out_dir / "verified.json"),
        str(out_dir / "review.json"),
        str(out_dir / "rejected.json"),
    ]
    assert len(client.messages.calls) == 2
    with open(str(out_dir / "verified.json"), "r", encoding="utf-8") as handle:
        verified = json.load(handle)
    assert len(verified) == 4
    assert all(item["status"] == "verified" for item in verified)
    with open(str(out_dir / "review.json"), "r", encoding="utf-8") as handle:
        (review,) = json.load(handle)
    assert review["found_start"] == 14


def test_run_defaults_out_dir_to_out(tmp_path, monkeypatch, capsys):
    install_fake_client(monkeypatch, FakeClient(fake_response([tool_block([])])))
    monkeypatch.chdir(tmp_path)
    code = cli.main(["run", TRANSCRIPT])
    out = capsys.readouterr().out
    assert code == 0
    assert "verified=0 review=0 rejected=0" in out
    assert (tmp_path / "out" / "verified.json").is_file()


def test_run_passes_model_and_chunk_options_through(tmp_path, monkeypatch):
    client = install_fake_client(monkeypatch, FakeClient(fake_response([tool_block([])])))
    code = cli.main(
        ["run", TRANSCRIPT, "--out", str(tmp_path), "--model", "my-model", "--chunk-size", "40", "--overlap", "4"]
    )
    assert code == 0
    assert client.messages.calls
    assert all(call["model"] == "my-model" for call in client.messages.calls)
    # 140 lines in windows of 40 stepping by 36 -> 1, 37, 73, 109 -> four calls.
    assert len(client.messages.calls) == 4


def test_run_rejects_overlap_not_smaller_than_chunk_size(tmp_path, monkeypatch, capsys):
    client = install_fake_client(monkeypatch, FakeClient(fake_response([tool_block([])])))
    code = cli.main(["run", TRANSCRIPT, "--out", str(tmp_path), "--chunk-size", "10", "--overlap", "10"])
    assert code == 1
    assert "--chunk-size" in capsys.readouterr().err
    assert client.messages.calls == []


def test_run_with_missing_transcript(tmp_path, monkeypatch, capsys):
    client = install_fake_client(monkeypatch, FakeClient(fake_response([tool_block([])])))
    code = cli.main(["run", str(tmp_path / "missing.txt"), "--out", str(tmp_path)])
    assert code == 1
    assert "transcript" in capsys.readouterr().err
    assert client.messages.calls == []


def test_run_authentication_error_prints_fix_and_exits_2(tmp_path, monkeypatch, capsys):
    install_fake_client(monkeypatch, FakeClient(exc=auth_error()))
    code = cli.main(["run", TRANSCRIPT, "--out", str(tmp_path)])
    captured = capsys.readouterr()
    assert code == 2
    assert "ANTHROPIC_API_KEY" in captured.err
    assert "ant auth login" in captured.err
    assert len(captured.err.strip().splitlines()) == 1
    assert not (tmp_path / "verified.json").exists()


def test_run_connection_error_prints_fix_and_exits_2(tmp_path, monkeypatch, capsys):
    install_fake_client(monkeypatch, FakeClient(exc=connection_error()))
    code = cli.main(["run", TRANSCRIPT, "--out", str(tmp_path)])
    captured = capsys.readouterr()
    assert code == 2
    assert "ANTHROPIC_API_KEY" in captured.err
    assert len(captured.err.strip().splitlines()) == 1


def test_run_sdk_no_credentials_type_error_exits_2(tmp_path, monkeypatch, capsys):
    exc = TypeError("Could not resolve authentication method. Expected one of api_key, auth_token, or credentials to be set.")
    install_fake_client(monkeypatch, FakeClient(exc=exc))
    code = cli.main(["run", TRANSCRIPT, "--out", str(tmp_path)])
    captured = capsys.readouterr()
    assert code == 2
    assert "ANTHROPIC_API_KEY" in captured.err


def test_run_unrelated_type_error_is_not_swallowed(tmp_path, monkeypatch):
    install_fake_client(monkeypatch, FakeClient(exc=TypeError("something else entirely")))
    with pytest.raises(TypeError):
        cli.main(["run", TRANSCRIPT, "--out", str(tmp_path)])


def test_run_api_status_error_prints_one_line_and_exits_2(tmp_path, monkeypatch, capsys):
    # An unknown or retired model id comes back as a 404 NotFoundError.
    install_fake_client(monkeypatch, FakeClient(exc=not_found_error()))
    code = cli.main(["run", TRANSCRIPT, "--out", str(tmp_path / "out")])
    captured = capsys.readouterr()
    assert code == 2
    assert captured.err.strip().splitlines() == ["Anthropic API error 404 (unknown): model not found"]
    assert captured.out == ""
    assert not (tmp_path / "out").exists()


def test_run_rate_limit_error_reports_status_and_api_error_type(tmp_path, monkeypatch, capsys):
    install_fake_client(monkeypatch, FakeClient(exc=rate_limit_error()))
    code = cli.main(["run", TRANSCRIPT, "--out", str(tmp_path)])
    captured = capsys.readouterr()
    assert code == 2
    assert captured.err.strip().splitlines() == ["Anthropic API error 429 (rate_limit_error): rate limited"]


def test_run_authentication_error_still_gets_the_credentials_hint_not_the_status_line(tmp_path, monkeypatch, capsys):
    # AuthenticationError subclasses APIStatusError; its branch must stay first.
    install_fake_client(monkeypatch, FakeClient(exc=auth_error()))
    assert cli.main(["run", TRANSCRIPT, "--out", str(tmp_path)]) == 2
    err = capsys.readouterr().err
    assert "Anthropic API error" not in err
    assert "ant auth login" in err


def test_run_credential_chain_error_from_the_client_constructor_exits_2(tmp_path, monkeypatch, capsys):
    # anthropic.Anthropic() resolves the credential chain in __init__: a mis-pointed
    # ANTHROPIC_PROFILE or ANTHROPIC_CONFIG_DIR raises AnthropicError before any request.
    def broken_make_client():
        raise credential_chain_error("Config file not found at /nowhere/configs/work.json (profile 'work').")

    monkeypatch.setattr(cli, "make_client", broken_make_client)
    code = cli.main(["run", TRANSCRIPT, "--out", str(tmp_path / "out")])
    captured = capsys.readouterr()
    assert code == 2
    assert "ANTHROPIC_API_KEY" in captured.err
    assert "ant auth login" in captured.err
    assert "Config file not found" in captured.err
    assert "Traceback" not in captured.err
    assert len(captured.err.strip().splitlines()) == 1
    assert not (tmp_path / "out").exists()


def test_run_credential_chain_error_at_request_time_exits_2(tmp_path, monkeypatch, capsys):
    # An expired or revoked `ant auth login` profile fails inside messages.create with the
    # same base class, not an APIStatusError, and the SDK re-raises it unwrapped.
    exc = credential_chain_error("Credentials file not found at /nowhere/credentials.json (profile 'default').")
    install_fake_client(monkeypatch, FakeClient(exc=exc))
    code = cli.main(["run", TRANSCRIPT, "--out", str(tmp_path)])
    captured = capsys.readouterr()
    assert code == 2
    assert "ant auth login" in captured.err
    assert "profile 'default'" in captured.err
    assert len(captured.err.strip().splitlines()) == 1


def test_run_multi_line_sdk_message_is_collapsed_to_one_stderr_line(tmp_path, monkeypatch, capsys):
    exc = credential_chain_error("first line\nsecond line\n\n   third line")
    install_fake_client(monkeypatch, FakeClient(exc=exc))
    assert cli.main(["run", TRANSCRIPT, "--out", str(tmp_path)]) == 2
    err = capsys.readouterr().err
    assert len(err.strip().splitlines()) == 1
    assert "first line second line third line" in err


def test_run_refusal_prints_summary_and_names_the_refused_chunk(tmp_path, monkeypatch, capsys):
    response = fake_response([tool_block(sample_items())], stop_reason="refusal")
    response.stop_details = SimpleNamespace(type="refusal", category="general_harms", explanation="no")
    install_fake_client(monkeypatch, FakeClient(response))
    code = cli.main(["run", TRANSCRIPT, "--out", str(tmp_path)])
    captured = capsys.readouterr()
    assert code == 0
    assert "verified=0 review=0 rejected=0" in captured.out
    # 140 lines, size 120, overlap 10 -> two chunks, both refused.
    assert "chunk 0 (lines 1-120): model refused (category=general_harms)" in captured.err
    assert "chunk 1 (lines 111-140): model refused (category=general_harms)" in captured.err


def test_run_out_pointing_at_a_file_exits_1(tmp_path, monkeypatch, capsys):
    install_fake_client(monkeypatch, FakeClient(fake_response([tool_block([])])))
    blocker = tmp_path / "afile"
    blocker.write_text("not a directory", encoding="utf-8")
    code = cli.main(["run", TRANSCRIPT, "--out", str(blocker)])
    captured = capsys.readouterr()
    assert code == 1
    assert captured.err.startswith("Could not write outputs: ")
    assert "verified=" not in captured.out


def test_run_help_names_the_default_model_and_the_env_var(capsys):
    with pytest.raises(SystemExit) as excinfo:
        cli.main(["run", "--help"])
    assert excinfo.value.code == 0
    out = capsys.readouterr().out
    assert DEFAULT_MODEL in out
    assert MODEL_ENV_VAR in out


# --- top level ----------------------------------------------------------------------------------


def test_no_subcommand_prints_help_and_exits_1(capsys):
    code = cli.main([])
    out = capsys.readouterr().out
    assert code == 1
    assert "verify-only" in out


def test_unknown_subcommand_is_an_argparse_error():
    with pytest.raises(SystemExit) as excinfo:
        cli.main(["frobnicate"])
    assert excinfo.value.code == 2


def test_help_flag_exits_zero():
    with pytest.raises(SystemExit) as excinfo:
        cli.main(["--help"])
    assert excinfo.value.code == 0
