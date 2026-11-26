"""Step 0 (big dataset): prepare a real-photo fashion dataset for the code.

Default source: Kaggle "Fashion Product Images (Small)" (styles.csv + images/<id>.jpg).
Download it from Kaggle yourself and unzip it anywhere, then run:

    python scripts/00_prepare_fashion_products.py --source ~/Downloads/fashion-product-images-small

What it does:
  1. reads the label file (default: styles.csv, label column 'articleType')
  2. keeps the N most common classes (default 10), or the classes you list
  3. takes at most --max-per-class images per class (balanced, seed 42)
  4. pads each image to a square on white and resizes it (default 64 x 64)
  5. writes data/raw/fashion_products/<class>/<id>.jpg  (the 'image_folder' layout)
  6. saves a summary in data/raw/fashion_products/_summary.json
  If it stops halfway, just run it again: images already made are skipped.

Then set in configs/default.yaml:  datasets.big.name: image_folder

Other CSV-labelled datasets work too: change --csv, --id-col, --label-col, --image-dir.
"""

from __future__ import annotations

import _bootstrap  # noqa: F401
import argparse
import json
import random
from collections import Counter
from pathlib import Path

import pandas as pd
from PIL import Image, ImageOps

from fic.config import data_path, load_config


def find(source: Path, name: str, is_dir: bool) -> Path:
    """Find a file/folder by name anywhere under source (Kaggle zips nest folders)."""
    direct = source / name
    if direct.exists():
        return direct
    for p in source.rglob(name):
        if p.is_dir() == is_dir:
            return p
    raise FileNotFoundError(f"Could not find '{name}' under {source}")


def safe(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in name.strip())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True, help="folder where you unzipped the Kaggle dataset")
    ap.add_argument("--csv", default="styles.csv")
    ap.add_argument("--image-dir", default="images")
    ap.add_argument("--id-col", default="id")
    ap.add_argument("--label-col", default="articleType")
    ap.add_argument("--ext", default=".jpg")
    ap.add_argument("--classes", nargs="*", default=None, help="exact class names to keep (default: top N)")
    ap.add_argument("--top", type=int, default=10, help="number of most common classes to keep")
    ap.add_argument("--max-per-class", type=int, default=2000)
    ap.add_argument("--size", type=int, default=64, help="output size in pixels (must be divisible by 4)")
    ap.add_argument("--out-name", default=None, help="output folder name (default: config datasets.big.folder_name)")
    ap.add_argument("--overwrite", action="store_true", help="redo images that already exist")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    assert args.size % 4 == 0, "size must be divisible by 4 (autoencoder needs it)"

    cfg = load_config(args.config)
    source = Path(args.source).expanduser()
    csv_path = find(source, args.csv, is_dir=False)
    img_dir = find(source, args.image_dir, is_dir=True)
    out = data_path(cfg) / (args.out_name or cfg.datasets.big.folder_name)

    # styles.csv has a few malformed rows (extra commas in names): skip them
    df = pd.read_csv(csv_path, on_bad_lines="skip", usecols=[args.id_col, args.label_col])
    df = df.dropna()
    df[args.id_col] = df[args.id_col].astype(str)
    df["exists"] = df[args.id_col].map(lambda i: (img_dir / f"{i}{args.ext}").exists())
    missing = int((~df.exists).sum())
    df = df[df.exists]

    counts = Counter(df[args.label_col])
    classes = args.classes or [c for c, _ in counts.most_common(args.top)]
    unknown = [c for c in classes if c not in counts]
    if unknown:
        raise SystemExit(f"Unknown classes: {unknown}. Available: {sorted(counts)}")

    rng = random.Random(42)
    summary = {"source": str(source), "label_col": args.label_col, "size": args.size,
               "missing_images_skipped": missing, "classes": {}}
    for c in classes:
        ids = sorted(df.loc[df[args.label_col] == c, args.id_col])
        rng.shuffle(ids)
        ids = ids[: args.max_per_class]
        folder = out / safe(c)
        folder.mkdir(parents=True, exist_ok=True)
        written = 0
        for i in ids:
            if (folder / f"{i}.jpg").exists() and not args.overwrite:   # resume a stopped run
                written += 1
                continue
            try:
                with Image.open(img_dir / f"{i}{args.ext}") as im:
                    im = ImageOps.pad(im.convert("RGB"), (args.size, args.size), color=(255, 255, 255))
                    im.save(folder / f"{i}.jpg", quality=95)
                    written += 1
            except Exception as exc:  # broken image file
                print(f"skip {i}: {exc}")
        summary["classes"][c] = written
        print(f"{c:<20} {written:>6} images")

    summary["total"] = sum(summary["classes"].values())
    json.dump(summary, open(out / "_summary.json", "w"), indent=2)
    print(f"\nDone: {summary['total']} images in {out}")
    print("Next: set datasets.big.name to image_folder in configs/default.yaml, then run")
    print("      python scripts/01_check_data.py --dataset big")


if __name__ == "__main__":
    main()
