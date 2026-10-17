# imagescv

Download [images.cv](https://images.cv) datasets in one line.

```
pip install imagescv
```

Free. Create an API key at <https://images.cv/dashboard/api-keys> (up to 25 dataset downloads per
day per account).

```python
import imagescv
imagescv.login("icv_...")                 # or set IMAGESCV_API_KEY

imagescv.search("firetruck")              # [{"name": "firetruck", "images": 562, ...}]
imagescv.info("firetruck")                # dataset card: images, classes, license, sources

ds = imagescv.load(
    "firetruck",
    size="size_128",                      # nochange, size_16, size_32, size_64, size_128, size_256
    color="gray",                         # color or gray
    augmentations=["rotate", "hflip"],    # see imagescv.AUGMENTATIONS
    augmentation_percent=30,              # share of images augmented
    split=(0.7, 0.2, 0.1),                # train / val / test
    download_path="data",                 # optional; default is the current folder
)
print(ds)                                 # images, classes, splits
train = ds.to_torch("train")              # needs: pip install imagescv[torch]
```

Datasets download into the current directory, in a folder like
`firetruck_size_128_color_70-20/` that contains `data/train|val|test/<class>/` and
`meta.json` (also available as `ds.meta`). Pass `download_path="D:/data"` to choose another
folder. Asking again for the same options reuses the folder instead of downloading again
(`force=True` re-downloads).
Dataset licenses vary; `info()["license"]` is "Unknown" when a source has none on file.
