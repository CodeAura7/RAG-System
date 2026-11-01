"""Keys come from: environment variable, then the local .env file, then the OS credential store."""
import pytest

from hrag import secrets


@pytest.fixture()
def env_file(tmp_path, monkeypatch):
    path = tmp_path / ".env"
    monkeypatch.setattr(secrets, "_dotenv_path", lambda: path)
    monkeypatch.setattr(secrets, "_keyring", lambda: None)          # never touch a real credential store in tests
    for name in ("H1_API_TOKEN", "OPENROUTER_API_KEY", "H1_API_USERNAME", "ONLY_IN_ENV"):
        monkeypatch.delenv(name, raising=False)
    return path


def test_reads_values_from_dotenv(env_file):
    env_file.write_text("OPENROUTER_API_KEY=sk-or-abc\nH1_API_USERNAME=alice\n", encoding="utf-8")
    assert secrets.get_secret("OPENROUTER_API_KEY") == "sk-or-abc"
    assert secrets.secret_source("OPENROUTER_API_KEY") == ".env"
    assert secrets.require_secret("H1_API_USERNAME") == "alice"


def test_environment_variable_beats_dotenv(env_file, monkeypatch):
    env_file.write_text("H1_API_TOKEN=from-file\n", encoding="utf-8")
    monkeypatch.setenv("H1_API_TOKEN", "from-env")
    assert secrets.get_secret("H1_API_TOKEN") == "from-env"
    assert secrets.secret_source("H1_API_TOKEN") == "env"


def test_dotenv_syntax_windows_editors_produce(env_file):
    # BOM (Notepad), CRLF line endings, comments, export prefix, quotes, spaces around '=', inline comment
    env_file.write_bytes(
        b"\xef\xbb\xbf# my keys\r\n"
        b"export H1_API_USERNAME = \"alice\"\r\n"
        b"\r\n"
        b"H1_API_TOKEN='tok=en#1'\r\n"
        b"OPENROUTER_API_KEY=sk-or-abc # trailing note\r\n"
    )
    assert secrets.get_secret("H1_API_USERNAME") == "alice"
    assert secrets.get_secret("H1_API_TOKEN") == "tok=en#1"          # '=' and '#' inside quotes are kept
    assert secrets.get_secret("OPENROUTER_API_KEY") == "sk-or-abc"


def test_empty_values_and_missing_file_count_as_missing(env_file):
    assert secrets.secret_source("H1_API_TOKEN") == "missing"         # no file at all
    env_file.write_text("H1_API_TOKEN=\nOPENROUTER_API_KEY=   \n", encoding="utf-8")
    assert secrets.get_secret("H1_API_TOKEN") is None
    assert secrets.get_secret("OPENROUTER_API_KEY") is None
    with pytest.raises(secrets.MissingKeyError) as err:
        secrets.require_secret("H1_API_TOKEN")
    assert ".env" in str(err.value)                                    # the message tells the user about the .env option


def test_dotenv_is_read_fresh_and_never_exported_to_the_environment(env_file):
    import os
    env_file.write_text("ONLY_IN_ENV=one\n", encoding="utf-8")
    assert secrets.get_secret("ONLY_IN_ENV") == "one"
    assert "ONLY_IN_ENV" not in os.environ                             # reading must not leak keys to child processes
    env_file.write_text("ONLY_IN_ENV=two\n", encoding="utf-8")
    assert secrets.get_secret("ONLY_IN_ENV") == "two"
