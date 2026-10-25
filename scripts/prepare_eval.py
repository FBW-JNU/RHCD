#!/usr/bin/env python
"""Freeze a full source list and deterministic cover/secret pair manifest."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from base.config import load_cfg_from_cfg_file
from dataset.paper import RHCDDataset
from utils.checkpoint import sha256
from utils.image_io import source_crop_sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/three_round.yaml")
    parser.add_argument("--list", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=131)
    parser.add_argument("--portable-root", type=Path,
                        help="Store image paths relative to this directory")
    parser.add_argument("--label", default="benchmark", help="Dataset/protocol name recorded in the manifest")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    cfg = load_cfg_from_cfg_file(args.config)
    cfg.data_root = str(args.data_root.resolve())
    cfg.fixed_val_secret_seed = args.seed
    data = RHCDDataset(args.list, cfg)
    if len(data.images) != len(data.pool):
        raise ValueError("Evaluation source list must contain no repeated resolved paths")
    portable_root = args.portable_root.resolve() if args.portable_root else None
    names = {path: str(Path(path).relative_to(portable_root)) if portable_root else path for path in data.pool}
    rows = [{"id": f"pair_{index:06d}", "cover": names[data.sample_paths(index)[0]],
             "secrets": [names[path] for path in data.sample_paths(index)[1:]]} for index in range(len(data))]
    hashes = {names[path]: sha256(path) for path in data.pool}
    crop_hashes = {names[path]: source_crop_sha256(path, cfg.val_crop_size) for path in data.pool}
    report = {"format": "rhcd_pairs_v1", "label": args.label, "rounds": int(cfg.rounds),
              "source_crop_size": int(cfg.val_crop_size), "seed": args.seed,
              "source_list": args.list.name if portable_root else str(args.list.resolve()),
              "source_list_sha256": sha256(args.list), "path_mode": "relative" if portable_root else "absolute",
              "source_count": len(data.pool), "pair_count": len(rows), "source_sha256": hashes,
              "protocol": "All listed covers; distinct secret sources within the same list; center crop or resize",
              "source_crop_rgb8_sha256": crop_hashes, "pairs": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"{args.output.resolve()}: {len(rows)} pairs, {len(data.pool)} source images")


if __name__ == "__main__":
    main()
