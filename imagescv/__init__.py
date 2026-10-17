"""images.cv datasets in one line:

    import imagescv
    imagescv.login("icv_...")          # or set IMAGESCV_API_KEY
    ds = imagescv.load("firetruck")
    print(ds.classes, ds.root)
"""
from ._version import __version__
from .client import login, info, load, search, SIZES, COLORS, AUGMENTATIONS
from .dataset import Dataset

__all__ = ["__version__", "login", "info", "load", "search", "Dataset", "SIZES", "COLORS", "AUGMENTATIONS"]
