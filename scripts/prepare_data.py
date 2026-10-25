#!/usr/bin/env python
"""Create portable training/validation lists from separate image directories."""

import argparse
import json
from pathlib import Path

from PIL import Image


SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def collect(directory, min_size):
    directory = directory.resolve(strict=True)
    if not directory.is_dir():
        raise NotADirectoryError(directory)
    paths, excluded = set(), []
    for path in sorted(directory.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUFFIXES:
            continue
        with Image.open(path) as image:
            width, height = image.size
            image.verify()
        if min(width, height) < min_size:
            excluded.append(str(path.resolve()))
        else:
            paths.add(path.resolve())
    return sorted(paths), excluded


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-dir", required=True, type=Path)
    parser.add_argument("--val-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path, help="New data_root directory")
    parser.add_argument("--rounds", type=int, choices=(1, 2, 3, 4), default=3)
    parser.add_argument("--min-size", type=int, default=0, help="Minimum source height and width")
    args = parser.parse_args()
    if args.min_size < 0:
        raise ValueError("--min-size must be nonnegative")
    if args.output.exists():
        raise FileExistsError(args.output)
    train, train_excluded = collect(args.train_dir, args.min_size)
    val, val_excluded = collect(args.val_dir, args.min_size)
    if len(train) < args.rounds + 1 or len(val) < args.rounds + 1:
        raise ValueError(f"Each split needs at least {args.rounds + 1} eligible images")
    if set(train) & set(val):
        raise ValueError("Training and validation paths overlap")
    listing = args.output / "list"
    listing.mkdir(parents=True)
    for split, paths in (("train", train), ("val", val)):
        (listing / f"{split}.txt").write_text("".join(str(path) + "\n" for path in paths))
    report = {"train_count": len(train), "val_count": len(val), "min_size": args.min_size,
              "excluded_small_train": train_excluded, "excluded_small_val": val_excluded,
              "split_check": "resolved-path disjointness"}
    (args.output / "data_summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"data_root: {args.output.resolve()} (train={len(train)}, val={len(val)})")


if __name__ == "__main__":
    main()
