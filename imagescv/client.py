import os
import secrets
import shutil
import time
import zipfile
from pathlib import Path

import requests
from tqdm.auto import tqdm

from ._version import __version__
from .dataset import Dataset

API_BASE_URL = os.environ.get("IMAGESCV_API_URL", "https://api.images.cv")

# The API sits behind Cloudflare, which blocks the default python-requests
# User-Agent. Same headers the Colab starter notebook uses.
_HEADERS = {
    "Content-Type": "application/json",
    "Accept": "application/json",
    # Identifies the library honestly. The word "python" is avoided on purpose:
    # the site's Cloudflare rule "Block Data Scraping Bots" blocks it.
    "User-Agent": f"imagescv/{__version__} (+https://images.cv)",
}

_api_key = None


def login(api_key):
    """Set the API key for this session (otherwise IMAGESCV_API_KEY is used)."""
    global _api_key
    _api_key = api_key


KEYS_URL = "https://images.cv/dashboard/api-keys"


def _headers():
    key = _api_key or os.environ.get("IMAGESCV_API_KEY")
    if not key:
        raise RuntimeError(
            f"No API key. Create a free one at {KEYS_URL}, then call imagescv.login('icv_...') "
            "or set the IMAGESCV_API_KEY environment variable.")
    h = dict(_HEADERS)
    h["X-API-Key"] = key
    return h


def _post(path, body, timeout=45, retries=2):
    resp = None
    for attempt in range(retries + 1):
        try:
            resp = requests.post(API_BASE_URL + path, json=body, headers=_headers(), timeout=timeout)
            break
        except requests.exceptions.RequestException as exc:
            if attempt == retries:
                raise RuntimeError(f"Network error calling {path}: {exc}") from exc
            time.sleep(2 * (attempt + 1))
    if "application/json" not in resp.headers.get("Content-Type", ""):
        raise RuntimeError(
            f"{path} returned non-JSON ({resp.status_code}); the request was probably "
            f"blocked before reaching the API. First bytes: {resp.text[:120]!r}")
    data = resp.json()
    if resp.status_code == 401:
        raise RuntimeError(f"API key rejected (invalid or revoked). Check or create one at {KEYS_URL}")
    if resp.status_code == 429:
        raise RuntimeError("Rate limited: " + str(data.get("error", "too many requests")))
    if resp.status_code != 200 or data.get("isError"):
        raise RuntimeError(f"{path} failed ({resp.status_code}): {data}")
    return data


SIZES = ("nochange", "size_16", "size_32", "size_64", "size_128", "size_256")
COLORS = ("color", "gray")
AUGMENTATIONS = (
    "rotate", "hflip", "vflip", "crop", "change_aspect_ratio", "perspective",
    "edge_enhance", "max_filter", "min_filter", "sharpen", "smooth", "blur",
    "brightness", "contrast", "saturation", "color_jitter", "shuffle_pixels",
    "random_noise", "cutout", "jpeg_compression",
)

# Same spellings the website resolves (stored value -> name, url).
_LICENSES = {
    "cc": ("CC0: Public Domain", "https://creativecommons.org/publicdomain/zero/1.0/"),
    "cc1": ("Database: Open Database, Contents: Database Contents", "https://opendatacommons.org/licenses/odbl/1-0/"),
    "cc2": ("CC BY 4.0", "https://creativecommons.org/licenses/by/4.0/"),
    "cc3": ("(c) Original Authors", None),
    "nn": ("GPL 2", "http://www.gnu.org/licenses/old-licenses/gpl-2.0.en.html"),
    "nn1": ("GNU Lesser General Public License 3.0", "http://www.gnu.org/licenses/lgpl-3.0.html"),
    "nn2": ("CC BY-NC-ND 4.0", "https://creativecommons.org/licenses/by-nc-nd/4.0/"),
    "nn3": ("Academic Research (c) Original Authors", None),
}
_LICENSE_ALIASES = {"cc0": "cc", "nn0": "nn"}


def _resolve_license(value):
    raw = (value or "").strip().lower()
    return _LICENSES.get(_LICENSE_ALIASES.get(raw, raw))


def info(name):
    """The dataset card: image count, classes, sources and license.

    license is the shared license name when every source agrees, "Mixed" when
    sources have different known licenses, and "Unknown" when a source has none
    on file (always check the source before use).
    """
    d = _post("/get_items_image_category", {"category": name, "start_after": 0})["data"]
    sources = []
    for raw in d.get("dataSource") or []:
        clean = raw[:-1] if raw.endswith("_") else raw
        slug = clean.strip().lower().replace("_", "-").replace(" ", "-")
        entry = {"name": clean.replace("_", " ").title(), "link": None, "license": None, "license_url": None}
        try:
            rows = _post("/get_data_sources", {"key": slug}).get("data") or []
        except RuntimeError:
            rows = []
        if rows:
            lic = _resolve_license(rows[0].get("license_type"))
            entry.update(name=(rows[0].get("name") or clean).replace("_", " ").title(),
                         link=rows[0].get("link") or None)
            if lic:
                entry.update(license=lic[0], license_url=lic[1])
        sources.append(entry)

    names = {s["license"] for s in sources}
    if sources and None not in names:
        license_text = names.pop() if len(names) == 1 else "Mixed"
    else:
        license_text = "Unknown"
    classes = d.get("mainCategory") or []
    return {
        "name": name,
        "images": d.get("sumCounter") or 0,
        "classes": len(classes),
        "class_names": classes,
        "license": license_text,
        "sources": sources,
        "premium": bool(d.get("is_premium")),
    }


def search(query, min_images=-1, limit=50):
    """Find datasets whose name starts with, or whose category contains, the query.

    Returns a list of dicts: name (use it with load()), images, parent.
    Uses the same two lookups as the website search: the category and its
    sub-categories (with image counts), plus name-prefix suggestions.
    """
    q = (query or "").strip()
    if not q:
        return []
    found = {}
    for it in _post("/api/search_advanced", {"q": q, "minImages": min_images}).get("data") or []:
        found[it["data_key"]] = {"name": it["data_key"], "images": it.get("count", 0),
                                 "parent": it.get("parent")}
    try:
        resp = requests.get(API_BASE_URL + "/api/autocomplete", params={"q": q},
                            headers=_headers(), timeout=45)
        suggestions = resp.json().get("data") or []
    except (requests.exceptions.RequestException, ValueError):
        suggestions = []
    for s in suggestions:
        key = s.get("main_category") if isinstance(s, dict) else None
        if key and key not in found:
            found[key] = {"name": key, "images": None, "parent": s.get("parent")}
    for row in found.values():
        row["community"] = False
    # Community datasets (uploaded by users, moderated) live in a separate index.
    try:
        community = _post("/search_community_datasets", {"q": q, "limit": 12}).get("data") or []
    except RuntimeError:
        community = []  # a community lookup problem must never break catalog search
    for it in community:
        key = it.get("data_key")
        if not key or key in found or not it.get("hosted"):
            continue  # only datasets that can be downloaded; the catalog wins a name clash
        found[key] = {"name": key, "images": it.get("image_count"), "parent": it.get("main_label"),
                      "community": True, "title": it.get("title")}
    rows = sorted(found.values(), key=lambda r: (-(r["images"] or 0), r["name"]))
    return rows[:limit]


def _wait_for_url(link_id, timeout, poll=3.0):
    deadline = time.monotonic() + timeout
    with tqdm(total=timeout, unit="s", desc="Preparing package") as bar:
        while time.monotonic() < deadline:
            url = _post("/get_link", {"link_id": link_id})["data"].get("public_url") or ""
            if url:
                bar.n = bar.total
                bar.refresh()
                return url
            time.sleep(poll)
            bar.update(poll)
    raise RuntimeError(
        f"Package was not ready after {timeout}s (job id {link_id}). Large datasets can take "
        f"longer: try load(..., timeout=1800), or start with a smaller dataset.")


def _stream(url, dest):
    resp = requests.get(url, stream=True, timeout=60,
                        headers={"User-Agent": _HEADERS["User-Agent"]})
    if resp.status_code != 200 or "html" in resp.headers.get("Content-Type", "").lower():
        raise RuntimeError(f"Download failed ({resp.status_code})")
    total = int(resp.headers.get("Content-Length", 0)) or None
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "wb") as fh, tqdm(total=total, unit="B", unit_scale=True, desc="Downloading") as bar:
        for chunk in resp.iter_content(1024 * 1024):
            fh.write(chunk)
            bar.update(len(chunk))
    with open(dest, "rb") as fh:
        if fh.read(2) != b"PK":
            dest.unlink()
            raise RuntimeError("Downloaded file is not a ZIP")


def _safe_extract(zip_path, target):
    target = Path(target).resolve()
    with zipfile.ZipFile(zip_path) as z:
        for member in z.namelist():
            if not str((target / member).resolve()).startswith(str(target)):
                raise RuntimeError("Unsafe path in ZIP: " + member)
        z.extractall(target)


def _unknown_dataset_message(name):
    msg = f"No dataset named '{name}' in the images.cv catalog."
    try:
        close = [r["name"] for r in search(name.replace("_", " "), limit=5)]
    except Exception:
        close = []
    if close:
        msg += " Did you mean: " + ", ".join(close) + "?"
    return msg + " Use imagescv.search() to find the exact name."


def _community_url(name, size, color, augs, augmentation_percent, split, timeout, poll=3.0):
    """Start a configured download of a community dataset and wait for its ZIP url.
    Returns None when no approved, downloadable community dataset has this name."""
    if size == "size_16":
        raise ValueError("Community datasets support these sizes: nochange, size_32, size_64, "
                         "size_128, size_256 (size_16 is catalog-only).")
    options = {"output_size": size, "color_type": color, "split_set": list(split),
               "augmentations": {a: True for a in augs}, "augmentation_percent": int(augmentation_percent)}
    try:
        download_id = _post("/create_hosted_dataset_download",
                            {"data_key": name, "options": options})["data"]["download_id"]
    except RuntimeError as exc:
        if "Dataset not found" in str(exc):
            return None
        raise
    deadline = time.monotonic() + timeout
    with tqdm(total=timeout, unit="s", desc="Preparing package") as bar:
        while time.monotonic() < deadline:
            st = _post("/get_hosted_dataset_download_status", {"download_id": download_id})["data"]
            if st.get("status") == "READY" and st.get("url"):
                bar.n = bar.total
                bar.refresh()
                return st["url"]
            if st.get("status") == "FAILED":
                raise RuntimeError(f"Community dataset package failed: {st.get('error') or 'unknown error'}")
            time.sleep(poll)
            bar.update(poll)
    raise RuntimeError(f"Package was not ready after {timeout}s (job id {download_id}). "
                       "Try load(..., timeout=1800).")


def load(name, size="size_128", color="color", augmentations=None,
         augmentation_percent=30, split=(0.7, 0.2, 0.1), download_path=None,
         force=False, timeout=1800):
    """Download (or reuse the cached copy of) a dataset and return a Dataset.

    size: one of SIZES ("nochange", "size_16" ... "size_256").
    color: "color" or "gray".
    augmentations: names from AUGMENTATIONS, e.g. ["rotate", "hflip", "blur"].
        Each image gets one random augmentation from your list;
        augmentation_percent (0-100) is the share of images augmented.
    split: train/val/test fractions.
    download_path: folder to download into (default: the current directory).
        The dataset goes in <download_path>/<name>_<options>/, and asking for
        the same options again reuses it instead of downloading again.
    """
    if size not in SIZES:
        raise ValueError(f"size must be one of {SIZES}")
    if color not in COLORS:
        raise ValueError(f"color must be one of {COLORS}")
    augs = sorted(set(augmentations or []))
    bad = [a for a in augs if a not in AUGMENTATIONS]
    if bad:
        raise ValueError(f"unknown augmentations {bad}; valid: {AUGMENTATIONS}")
    if not 0 <= augmentation_percent <= 100:
        raise ValueError("augmentation_percent must be between 0 and 100")
    if len(split) != 3 or abs(sum(split) - 1.0) > 0.01:
        raise ValueError("split must be three fractions that sum to 1, e.g. (0.7, 0.2, 0.1)")

    slug = "".join(c if c.isalnum() or c in "-_." else "_" for c in name)
    variant = f"{size}_{color}_{int(round(split[0] * 100))}-{int(round(split[1] * 100))}"
    if augs:
        variant += "_" + "-".join(augs) + f"_p{int(augmentation_percent)}"
    root = Path(download_path) if download_path else Path.cwd()
    folder = root / f"{slug}_{variant}"
    package = folder  # data/ and meta.json sit directly in this folder
    marker = folder / ".complete"

    if marker.exists() and not force:
        return Dataset(name, package)

    body = {
        "output_size": size, "color_type": color, "images": [],
        "categories": [name], "augmentations": [{"name": a, "checked": True} for a in augs],
        "split_set": list(split),
        "link_id": secrets.token_hex(20), "request_type": "CALL_FROM_SEARCH",
        "user_email": [], "user_id": "", "augmentation_percent": int(augmentation_percent),
    }
    url = None
    try:
        link_id = _post("/download_images", body)["data"].get("link_id")
        if not link_id:
            raise RuntimeError("The server did not accept the download job")
        url = _wait_for_url(link_id, timeout)
    except RuntimeError as exc:
        if "failed (404)" not in str(exc):
            raise
        # Not in the catalog: it may be a community dataset (the name is its data_key).
        url = _community_url(name, size, color, augs, augmentation_percent, split, timeout)
        if url is None:
            raise ValueError(_unknown_dataset_message(name)) from exc
    zip_path = folder / "package.zip"
    _stream(url, zip_path)
    staging = folder / ".extract"
    if staging.exists():
        shutil.rmtree(staging)
    _safe_extract(zip_path, staging)
    # The server wraps everything in one images.cv_<id>/ folder; drop that
    # wrapper so you get <folder>/data/train/... and <folder>/meta.json.
    entries = list(staging.iterdir())
    source = entries[0] if len(entries) == 1 and entries[0].is_dir() else staging
    for item in source.iterdir():
        target = folder / item.name
        if target.is_dir():
            shutil.rmtree(target)
        elif target.exists():
            target.unlink()
        shutil.move(str(item), str(target))
    shutil.rmtree(staging)
    zip_path.unlink()
    marker.write_text("ok")
    return Dataset(name, package)
