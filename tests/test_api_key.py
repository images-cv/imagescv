import pytest

import imagescv
from imagescv import client


class _Resp:
    def __init__(self, status, payload):
        self.status_code = status
        self._payload = payload
        self.headers = {"Content-Type": "application/json"}
        self.text = ""

    def json(self):
        return self._payload


def test_key_is_sent_in_x_api_key_header():
    assert client._headers()["X-API-Key"] == "icv_test"
    imagescv.login("icv_mine")
    assert client._headers()["X-API-Key"] == "icv_mine"
    assert "Authorization" not in client._headers()


def test_missing_key_gives_clear_error(monkeypatch):
    monkeypatch.delenv("IMAGESCV_API_KEY")
    with pytest.raises(RuntimeError, match="No API key"):
        client._headers()


def test_rejected_key_and_daily_limit_messages(monkeypatch):
    monkeypatch.setattr(client.requests, "post", lambda *a, **k: _Resp(401, {"error": "x"}))
    with pytest.raises(RuntimeError, match="API key rejected"):
        client._post("/get_link", {})
    monkeypatch.setattr(client.requests, "post", lambda *a, **k: _Resp(429, {"error": "Daily download limit reached (25 per user)"}))
    with pytest.raises(RuntimeError, match="Daily download limit"):
        client._post("/download_images", {})


def test_user_agent_starts_with_imagescv_and_carries_the_version():
    # The server's key gate matches on the "imagescv/" prefix.
    assert client._HEADERS["User-Agent"].startswith("imagescv/")
    assert imagescv.__version__ in client._HEADERS["User-Agent"]
