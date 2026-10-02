"""Download the official LFWA images, 40 attributes and split indices.

Run from the repository root: python scripts/prepare_lfwa.py --download
"""
import argparse
import hashlib
import os
from pathlib import Path
import sys
import zipfile
from html.parser import HTMLParser

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Official LFWA+ folder linked by https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html
FILES = {
    "lfw.zip": ("0B7EVK8r0v71pZDFOOGxhbm1oakE", 191778528),
    "lfw_att_40.mat": ("0B7EVK8r0v71pZ1ZVTWhrZGNxdzA", 6171224),
    "indices_train_test.mat": ("0B7EVK8r0v71pdjRRNWNZWlo0WlU", 65227),
}


class DownloadForm(HTMLParser):
    def __init__(self):
        super().__init__()
        self.action = None
        self.fields = {}

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "form" and attrs.get("id") == "download-form":
            self.action = attrs.get("action")
        if tag == "input" and "name" in attrs:
            self.fields[attrs["name"]] = attrs.get("value", "")


def download_file(name, destination):
    import gdown
    import requests

    file_id, expected_bytes = FILES[name]
    destination = Path(destination)
    if destination.exists() and destination.stat().st_size == expected_bytes:
        print(f"Already downloaded: {destination}", flush=True)
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    try:
        gdown.download(url, str(partial), quiet=False, use_cookies=False)
    except Exception as error:
        print(f"gdown: {error}; trying the public Drive download form", flush=True)
    if not partial.exists() or partial.stat().st_size != expected_bytes:
        # Older gdown releases do not handle the current Drive confirmation form.
        with requests.Session() as session:
            response = session.get(url, stream=True, timeout=(30, 120))
            response.raise_for_status()
            if "text/html" in response.headers.get("Content-Type", ""):
                form = DownloadForm()
                form.feed(response.text)
                response.close()
                if not form.action or not form.action.startswith("https://drive.usercontent.google.com/"):
                    raise RuntimeError(f"Google Drive did not provide a download for {name}")
                response = session.get(form.action, params=form.fields, stream=True, timeout=(30, 120))
                response.raise_for_status()
            if "text/html" in response.headers.get("Content-Type", ""):
                raise RuntimeError(f"Google Drive returned HTML instead of {name}")
            received = 0
            with partial.open("wb") as output:
                for chunk in response.iter_content(1024 * 1024):
                    output.write(chunk)
                    received += len(chunk)
                    if received // (32 * 1024 * 1024) != (received - len(chunk)) // (32 * 1024 * 1024):
                        print(f"{name}: {received:,}/{expected_bytes:,} bytes", flush=True)
            response.close()
    if partial.stat().st_size != expected_bytes:
        raise RuntimeError(f"Incomplete or changed official file {name}: {partial.stat().st_size} != {expected_bytes}")
    os.replace(partial, destination)
    print(f"Downloaded: {destination} ({expected_bytes:,} bytes)", flush=True)


def extract_images(archive, image_root):
    image_root = Path(image_root).resolve()
    image_root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as source:
        for member in source.infolist():
            target = (image_root / member.filename).resolve()
            if image_root not in target.parents and target != image_root:
                raise ValueError(f"Invalid archive path: {member.filename}")
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            elif not target.is_file() or target.stat().st_size != member.file_size:
                source.extract(member, image_root)
        n_images = sum(m.filename.lower().endswith((".jpg", ".jpeg", ".png")) for m in source.infolist())
    print(f"Extracted image archive: {n_images:,} images, root={image_root}", flush=True)


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--download-only", action="store_true")
    parser.add_argument("--image-root", type=Path, default=ROOT / "Images/LFWA")
    parser.add_argument("--output", type=Path, default=ROOT / "csv/LFWA.csv")
    args = parser.parse_args()
    raw = args.image_root / "raw"
    for name in FILES:
        if args.download:
            download_file(name, raw / name)
        if not (raw / name).is_file():
            raise FileNotFoundError(f"Missing {raw / name}; run with --download")
    extract_images(raw / "lfw.zip", args.image_root)
    if args.download_only:
        return
    from lfwa import prepare_csv
    prepare_csv(raw, args.image_root, args.output)


if __name__ == "__main__":
    main()
