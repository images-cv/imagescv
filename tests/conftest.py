import pytest

from imagescv import client


@pytest.fixture(autouse=True)
def fake_api_key(monkeypatch):
    monkeypatch.setattr(client, "_api_key", None)
    monkeypatch.setenv("IMAGESCV_API_KEY", "icv_test")
