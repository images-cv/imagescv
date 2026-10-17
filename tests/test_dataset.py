from imagescv.dataset import Dataset


def test_nested_split_layout(tmp_path):
    for split in ("train", "val"):
        for cls in ("a", "b"):
            d = tmp_path / "data" / split / cls
            d.mkdir(parents=True)
            (d / "1.png").write_bytes(b"x")
    ds = Dataset("t", tmp_path)
    assert ds.classes == ["a", "b"]
    assert ds.splits == ["train", "val"]
    assert len(ds) == 4
