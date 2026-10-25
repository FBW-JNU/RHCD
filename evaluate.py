#!/usr/bin/env python
"""Evaluate every frozen pair on native and direct/stepwise retained paths."""

import argparse
import csv
import json
import shlex
import sys
from functools import lru_cache
from pathlib import Path

import cv2
import torch

from inference import RHCD
from metrics.image_quality import image_metrics
from utils.checkpoint import sha256
from utils.image_io import read_source, source_crop_sha256


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--rd-checkpoint", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--source-root", type=Path, help="Image directory for a relative-path manifest")
    parser.add_argument("--torch-threads", type=int, default=4)
    parser.add_argument("--rd-input", choices=("raw", "clamp"), default="clamp",
                        help="Bicubic input handling; learned RD uses its checkpoint protocol")
    args = parser.parse_args()
    if args.torch_threads < 1:
        raise ValueError("--torch-threads must be positive")
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(args.torch_threads)
    cv2.setNumThreads(0)
    predictor = RHCD(args.checkpoint, args.device, args.rd_checkpoint)
    manifest = json.loads(args.manifest.read_text())
    if manifest.get("format") != "rhcd_pairs_v1" or manifest["rounds"] != predictor.rounds:
        raise ValueError("Pair manifest and checkpoint hierarchy differ")
    if manifest["pair_count"] != len(manifest["pairs"]) or not manifest["pairs"]:
        raise ValueError("Invalid or empty pair manifest")
    ids = [row["id"] for row in manifest["pairs"]]
    if len(set(ids)) != len(ids):
        raise ValueError("Repeated pair IDs")
    relative = manifest.get("path_mode", "absolute") == "relative"
    if relative and args.source_root is None:
        raise ValueError("Relative manifests require --source-root")
    source_root = args.source_root.resolve() if relative else None

    def source_path(name):
        if source_root is None:
            return Path(name)
        path = (source_root / name).resolve()
        path.relative_to(source_root)
        return path

    crop_hashes = manifest.get("source_crop_rgb8_sha256")
    if crop_hashes is not None and set(crop_hashes) != set(manifest["source_sha256"]):
        raise ValueError("Source crop hash keys differ from source file keys")
    for path, expected in manifest["source_sha256"].items():
        actual_path = source_path(path)
        if crop_hashes is not None:
            if source_crop_sha256(actual_path, manifest["source_crop_size"]) != crop_hashes[path]:
                raise ValueError("Source crop pixels changed: " + path)
        elif sha256(actual_path) != expected:
            raise ValueError("Source image content changed: " + path)

    @lru_cache(maxsize=256)
    def source_tensor(name):
        return read_source(source_path(name), manifest["source_crop_size"])
    args.output.mkdir(parents=True)
    rows = []
    for index, pair in enumerate(manifest["pairs"]):
        paths = [pair["cover"], *pair["secrets"]]
        if len(paths) != predictor.rounds + 1 or len(set(paths)) != len(paths):
            raise ValueError("Each pair requires distinct cover and secret sources")
        if any(path not in manifest["source_sha256"] for path in paths):
            raise ValueError("Pair references an unhashed source")
        sources = [source_tensor(path) for path in paths]
        base, secrets, targets = predictor.prepare(sources)
        stegos = predictor.embed(base, secrets)
        row = {"id": pair["id"], "cover": pair["cover"]}

        def record(name, prediction, target):
            for key, value in image_metrics(prediction, target)[0].items():
                row[name + "_" + key] = value

        for level, stego in enumerate(stegos, 1):
            record(f"stego{level}", stego, targets[level - 1])
            record(f"native_s{level}", predictor.model.decode(stego), secrets[level - 1])
            for route in ("direct", "stepwise"):
                if level < 2:
                    continue
                recovered = predictor.extract(stego, level, list(range(1, level)),
                                              "bicubic", route, args.rd_input)
                for target_level, value in recovered.items():
                    suffix = "" if route == "direct" else "_step"
                    record(f"y{level}{suffix}_s{target_level}", value, secrets[target_level - 1])
        if predictor.rd is not None:
            if predictor.rd_metadata.get("native_y2_supported", False):
                value = predictor.extract(stegos[1], 2, [1], "learned")[1]
                record("learned_y2_s1", value, secrets[0])
            for route in ("direct", "stepwise"):
                recovered = predictor.extract(stegos[2], 3, [1, 2], "learned", route)
                for level, value in recovered.items():
                    suffix = "" if route == "direct" else "_step"
                    record(f"learned_y3{suffix}_s{level}", value, secrets[level - 1])
        rows.append(row)
        if index == 0 or (index + 1) % 20 == 0:
            print(f"{index + 1}/{len(manifest['pairs'])}", flush=True)
    keys = [key for key in rows[0] if key.endswith(("_psnr", "_ssim"))]
    means = {key: sum(row[key] for row in rows) / len(rows) for key in keys}
    with (args.output / "per_pair.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = {"status": "complete", "command": shlex.join([sys.executable, *sys.argv]),
              "checkpoint": str(args.checkpoint.resolve()), "checkpoint_sha256": predictor.checkpoint_sha256,
              "checkpoint_epoch": predictor.checkpoint["epoch"], "architecture": predictor.model.architecture_version,
              "manifest": str(args.manifest.resolve()), "manifest_sha256": sha256(args.manifest),
              "runtime_state": {"texture_bias": predictor.model.current_texture_bias,
                                "secret_lambda": predictor.model.secret_lambda},
              "pair_count": len(rows), "dataset_label": manifest.get("label"), "means": means,
              "channel": "FP32 tensors, no intermediate quantization or save/reopen",
              "metrics": "per-image RGB PSNR and valid 11x11 Gaussian SSIM; metric-only clamp; full mean",
              "precision": "FP32, TF32 disabled",
              "rd_input": args.rd_input,
              "stepwise_protocol": "clamp each RD input; outputs unclipped" if args.rd_input == "clamp" and
                  bool(predictor.cfg.get("rd_stepwise_clamp", False)) else
                  "clamp initial received carrier only; intermediate outputs unclipped" if args.rd_input == "clamp" else
                  "raw float inputs and outputs",
              "source_verification": "RGB8 source-crop SHA256" if crop_hashes is not None else "file SHA256",
              "config": dict(predictor.cfg)}
    if relative:
        report["source_root"] = str(source_root)
    report["bicubic_stepwise_protocol"] = report["stepwise_protocol"]
    if args.rd_checkpoint:
        report["rd_checkpoint_sha256"] = sha256(args.rd_checkpoint)
        report["learned_rd_protocol"] = predictor.rd_metadata["protocol"]
        report["learned_stepwise_protocol"] = "clamp each RD input; outputs unclipped" if predictor.rd.chain_input_clamp else \
            "clamp initial received carrier only; intermediate outputs unclipped"
        report["native_y2_to_y1_operator"] = "learned" if predictor.rd_metadata.get("native_y2_supported", False) else "bicubic"
    (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(means, indent=2))


if __name__ == "__main__":
    main()
