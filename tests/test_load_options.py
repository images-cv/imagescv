import pytest

import imagescv
from imagescv import client


def test_rejects_bad_options():
    with pytest.raises(ValueError):
        imagescv.load("cat", size="size_99")
    with pytest.raises(ValueError):
        imagescv.load("cat", color="sepia")
    with pytest.raises(ValueError):
        imagescv.load("cat", augmentations=["warp"])
    with pytest.raises(ValueError):
        imagescv.load("cat", augmentation_percent=150)
    with pytest.raises(ValueError):
        imagescv.load("cat", split=(0.9, 0.9, 0.9))


def test_request_body_and_download_path(monkeypatch, tmp_path):
    sent = {}

    def fake_post(path, body, **k):
        sent[path] = body
        return {"data": {"link_id": "x"}}

    monkeypatch.setattr(client, "_post", fake_post)
    monkeypatch.setattr(client, "_wait_for_url", lambda *a, **k: "http://z")

    def fake_stream(url, dest):
        import zipfile
        dest.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(dest, "w") as z:
            z.writestr("data/train/a/1.png", b"x")

    monkeypatch.setattr(client, "_stream", fake_stream)
    ds = imagescv.load("cat", size="size_64", color="gray", augmentations=["blur", "rotate"],
                       augmentation_percent=40, download_path=tmp_path)
    body = sent["/download_images"]
    assert body["output_size"] == "size_64" and body["color_type"] == "gray"
    assert body["augmentations"] == [{"name": "blur", "checked": True}, {"name": "rotate", "checked": True}]
    assert body["augmentation_percent"] == 40
    assert ds.package == tmp_path / "cat_size_64_gray_70-20_blur-rotate_p40"
    assert ds.classes == ["a"]


def test_default_download_path_is_cwd(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(client, "_post", lambda *a, **k: {"data": {"link_id": "x"}})
    monkeypatch.setattr(client, "_wait_for_url", lambda *a, **k: "http://z")

    def fake_stream(url, dest):
        import zipfile
        dest.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(dest, "w") as z:
            z.writestr("data/train/a/1.png", b"x")

    monkeypatch.setattr(client, "_stream", fake_stream)
    ds = imagescv.load("cat")
    assert ds.package.parent == tmp_path


def test_server_wrapper_folder_is_flattened(monkeypatch, tmp_path):
    monkeypatch.setattr(client, "_post", lambda *a, **k: {"data": {"link_id": "x"}})
    monkeypatch.setattr(client, "_wait_for_url", lambda *a, **k: "http://z")

    def fake_stream(url, dest):
        import zipfile
        dest.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(dest, "w") as z:
            z.writestr("images.cv_abc/data/train/a/1.png", b"x")
            z.writestr("images.cv_abc/meta.json", '{"amount_of_images": 1}')

    monkeypatch.setattr(client, "_stream", fake_stream)
    ds = imagescv.load("cat", download_path=tmp_path)
    folder = tmp_path / "cat_size_128_color_70-20"
    assert (folder / "data" / "train" / "a" / "1.png").exists()
    assert (folder / "meta.json").exists()
    assert not (folder / ".extract").exists()
    assert ds.meta["amount_of_images"] == 1
    assert ds.classes == ["a"]

    # asking again reuses the folder, and force=True re-downloads cleanly
    assert imagescv.load("cat", download_path=tmp_path).package == folder
    imagescv.load("cat", download_path=tmp_path, force=True)
    assert (folder / "data" / "train" / "a" / "1.png").exists()


def test_info_license_logic(monkeypatch):
    def fake_post(path, body, **k):
        if path == "/get_items_image_category":
            return {"data": {"sumCounter": 100, "mainCategory": ["a", "b"], "dataSource": ["kaggle_x", "other"]}}
        key = body["key"]
        return {"data": [{"name": key, "link": "http://l", "license_type": "cc0" if key == "kaggle-x" else "cc2"}]}

    monkeypatch.setattr(client, "_post", fake_post)
    card = imagescv.info("cat")
    assert card["images"] == 100 and card["classes"] == 2
    assert card["license"] == "Mixed"
    monkeypatch.setattr(client, "_post", lambda p, b, **k: (
        {"data": {"sumCounter": 1, "mainCategory": [], "dataSource": ["s"]}} if "items" in p
        else {"data": [{"name": "s", "license_type": ""}]}))
    assert imagescv.info("cat")["license"] == "Unknown"
