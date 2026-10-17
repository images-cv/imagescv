from pathlib import Path

_IMG = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def _find_split_root(package):
    """The ZIP may nest the splits (e.g. data/train); find the folder that holds train/."""
    package = Path(package)
    for p in [package, *sorted(package.rglob("*"))]:
        if p.is_dir() and (p / "train").is_dir():
            return p
    return package


class Dataset:
    def __init__(self, name, package):
        self.name = name
        self.package = Path(package)
        self.root = _find_split_root(self.package)

    @property
    def meta(self):
        """The package's meta.json (counts, split, size, color, augmentation), or {}."""
        path = self.package / "meta.json"
        if not path.exists():
            return {}
        import json
        return json.loads(path.read_text(encoding="utf8"))

    @property
    def splits(self):
        return sorted(d.name for d in self.root.iterdir() if d.is_dir())

    @property
    def classes(self):
        train = self.root / "train"
        base = train if train.is_dir() else self.root
        return sorted(d.name for d in base.iterdir() if d.is_dir())

    def paths(self, split="train"):
        base = self.root / split
        return sorted(p for p in base.rglob("*") if p.suffix.lower() in _IMG)

    def __len__(self):
        return sum(len(self.paths(s)) for s in self.splits)

    def to_torch(self, split="train", transform=None):
        try:
            from torchvision import datasets
        except ImportError as exc:
            raise ImportError("Install with: pip install imagescv[torch]") from exc
        return datasets.ImageFolder(str(self.root / split), transform=transform)

    def __repr__(self):
        return f"Dataset({self.name!r}, {len(self)} images, {len(self.classes)} classes, splits={self.splits})"
