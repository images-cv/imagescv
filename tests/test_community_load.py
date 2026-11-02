import io
import zipfile

import pytest

from imagescv import client


def _zip_bytes():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("pkg/meta.json", '{"classes": ["a"], "counts": {"train": 1, "val": 0, "test": 0}}')
        z.writestr("pkg/data/train/a/1.jpg", b"x")
    return buf.getvalue()


def _patch(monkeypatch, tmp_path, community_exists=True):
    calls = []

    def fake_post(path, body, **k):
        calls.append((path, body))
        if path == "/download_images":
            raise RuntimeError("/download_images failed (404): {'error': 'No dataset'}")
        if path == "/create_hosted_dataset_download":
            if not community_exists:
                raise RuntimeError("/create_hosted_dataset_download failed (400): {'error': 'Dataset not found'}")
            return {"data": {"download_id": "d1"}}
        if path == "/get_hosted_dataset_download_status":
            return {"data": {"status": "READY", "url": "https://x/zip"}}
        if path == "/search_community_datasets":
            return {"data": []}
        return {"data": []}

    def fake_stream(url, dest):
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(_zip_bytes())

    monkeypatch.setattr(client, "_post", fake_post)
    monkeypatch.setattr(client, "_stream", fake_stream)
    monkeypatch.setattr(client.requests, "get", lambda *a, **k: type("R", (), {"json": lambda s: {"data": []}})())
    return calls


def test_load_falls_back_to_community(monkeypatch, tmp_path):
    calls = _patch(monkeypatch, tmp_path)
    ds = client.load("my_cones", size="size_64", augmentations=["hflip"], download_path=tmp_path)
    assert ds is not None
    body = dict(calls)["/create_hosted_dataset_download"]
    assert body["data_key"] == "my_cones"
    assert body["options"]["output_size"] == "size_64"
    assert body["options"]["augmentations"] == {"hflip": True}


def test_load_unknown_everywhere_raises_value_error(monkeypatch, tmp_path):
    _patch(monkeypatch, tmp_path, community_exists=False)
    with pytest.raises(ValueError, match="No dataset named"):
        client.load("nope", download_path=tmp_path)


def test_community_rejects_size_16(monkeypatch, tmp_path):
    _patch(monkeypatch, tmp_path)
    with pytest.raises(ValueError, match="size_16"):
        client.load("my_cones", size="size_16", download_path=tmp_path)
