import imagescv
from imagescv import client


class _Resp:
    def __init__(self, payload):
        self._p = payload

    def json(self):
        return self._p


def test_search_merges_and_sorts(monkeypatch):
    monkeypatch.setattr(client, "_post", lambda path, body, **k: {"data": [
        {"data_key": "traffic_cone", "count": 600, "parent": None},
        {"data_key": "cone", "count": 40, "parent": "Shape"},
    ]})
    monkeypatch.setattr(client.requests, "get", lambda *a, **k: _Resp({"data": [
        {"main_category": "cone", "parent": "Shape"},
        {"main_category": "conehead", "parent": None},
    ]}))
    rows = imagescv.search("cone")
    assert [r["name"] for r in rows] == ["traffic_cone", "cone", "conehead"]
    assert rows[2]["images"] is None


def test_search_empty_query():
    assert imagescv.search("  ") == []
