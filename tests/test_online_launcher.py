"""Offline boundaries for the reusable browser integration."""
import importlib
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


def test_cli_help_imports_without_opening_a_browser():
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run([sys.executable, "scripts/online.py", "--help"], cwd=root, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "--url" in result.stdout


def test_site_url_is_explicit_and_does_not_accept_embedded_credentials(monkeypatch):
    online = importlib.import_module("trucoai.integration.online")
    monkeypatch.delenv("TRUCOAI_SITE_URL", raising=False)
    with pytest.raises(ValueError, match="TRUCOAI_SITE_URL"):
        online.site_url()
    monkeypatch.setenv("TRUCOAI_SITE_URL", "https://table.example/game")
    assert online.site_url() == "https://table.example/game"
    for value in ("file:///private", "https://user:secret@table.example", "https://table.example/?token=secret"):
        monkeypatch.setenv("TRUCOAI_SITE_URL", value)
        with pytest.raises(ValueError):
            online.site_url()


def test_cdp_selects_configured_origin_and_keeps_other_tabs(monkeypatch, tmp_path):
    online = importlib.import_module("trucoai.integration.online")
    monkeypatch.setenv("TRUCOAI_SITE_URL", "https://table.example/game")
    expected = SimpleNamespace(url="https://table.example/game")
    context = SimpleNamespace(pages=[SimpleNamespace(url="https://other.example/"), expected])
    pw = SimpleNamespace(chromium=SimpleNamespace(connect_over_cdp=lambda url: SimpleNamespace(contexts=[context])))
    ctx, page = online._launch(pw, tmp_path, cdp="http://127.0.0.1:9222")
    assert ctx is context and page is expected
    assert ctx._trucoai_external


def test_sensitive_session_fields_are_redacted():
    online = importlib.import_module("trucoai.integration.online")
    record = {"token": "private", "uid": "person", "usn": "nickname", "rid": "room", "nested": [{"cookie": "session", "sq": 4}]}
    cleaned = online.redact(record)
    assert all(cleaned[key] == "<redigido>" for key in ("token", "uid", "usn", "rid"))
    assert cleaned["nested"][0] == {"cookie": "<redigido>", "sq": 4}
