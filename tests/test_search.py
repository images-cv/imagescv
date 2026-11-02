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


def test_search_includes_downloadable_community_datasets(monkeypatch):
    def fake_post(path, body, **k):
        if path == "/search_community_datasets":
            return {"data": [
                {"data_key": "my_cones", "title": "My cones", "image_count": 120, "main_label": "Cone", "hosted": True},
                {"data_key": "link_only", "image_count": 5, "hosted": False},
                {"data_key": "traffic_cone", "image_count": 3, "hosted": True},  # clash: catalog wins
            ]}
        return {"data": [{"data_key": "traffic_cone", "count": 600, "parent": None}]}
    monkeypatch.setattr(client, "_post", fake_post)
    monkeypatch.setattr(client.requests, "get", lambda *a, **k: _Resp({"data": []}))
    rows = {r["name"]: r for r in imagescv.search("cone")}
    assert set(rows) == {"traffic_cone", "my_cones"}
    assert rows["my_cones"]["community"] is True and rows["my_cones"]["images"] == 120
    assert rows["traffic_cone"]["community"] is False


def test_search_survives_community_failure(monkeypatch):
    def fake_post(path, body, **k):
        if path == "/search_community_datasets":
            raise RuntimeError("boom")
        return {"data": [{"data_key": "cone", "count": 40, "parent": None}]}
    monkeypatch.setattr(client, "_post", fake_post)
    monkeypatch.setattr(client.requests, "get", lambda *a, **k: _Resp({"data": []}))
    assert [r["name"] for r in imagescv.search("cone")] == ["cone"]
