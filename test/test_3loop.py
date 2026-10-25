#!/usr/bin/env python
"""Progressive image embedding, resolution downgrade and content extraction."""

import argparse
import json
import shlex
import sys
from pathlib import Path

import cv2
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from inference import RHCD, RHCDReceiver, RHCDRelay
from metrics.image_quality import image_metrics
from utils.checkpoint import sha256
from utils.image_io import read_carrier, read_source, save_tensor


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="action", required=True)
    for action in ("embed", "extract", "downgrade"):
        sub = actions.add_parser(action)
        sub.add_argument("--output", type=Path, required=True, help="New output directory")
        sub.add_argument("--device", default="cpu")
        sub.add_argument("--torch-threads", type=int, default=4)
        if action == "embed":
            sub.add_argument("--checkpoint", type=Path, required=True)
            sub.add_argument("--cover", type=Path, required=True)
            sub.add_argument("--secrets", nargs="+", type=Path, required=True)
            sub.add_argument("--crop-size", type=int, help="Defaults to the checkpoint validation crop")
        else:
            if action == "extract":
                weights = sub.add_mutually_exclusive_group(required=True)
                weights.add_argument("--checkpoint", type=Path)
                weights.add_argument("--hce-checkpoint", type=Path)
            sub.add_argument("--stego", type=Path, required=True, help="RGB HWC float32 .npy, or a decoded image")
            sub.add_argument("--carrier-level", type=int, required=True)
            sub.add_argument("--levels", type=int, nargs="+")
            sub.add_argument("--rd-checkpoint", type=Path, required=action == "downgrade")
            sub.add_argument("--route", choices=("direct", "stepwise"), default="direct")
            if action == "extract":
                sub.add_argument("--rd-mode", choices=("bicubic", "learned"), default="learned")
                sub.add_argument("--rd-input", choices=("clamp", "raw"), default="clamp")
    return parser.parse_args()


def run_downgrade(args):
    relay = RHCDRelay(args.rd_checkpoint, args.device)
    carrier = read_carrier(args.stego)
    received = relay.downgrade(carrier, args.carrier_level, args.levels, args.route)
    args.output.mkdir(parents=True)
    report = {"command": shlex.join([sys.executable, *sys.argv]), "action": "downgrade",
              "rd_checkpoint": str(args.rd_checkpoint.resolve()), "rd_checkpoint_sha256": relay.checkpoint_sha256,
              "base_checkpoint_sha256": relay.base_checkpoint_sha256,
              "carrier": str(args.stego.resolve()), "carrier_sha256": sha256(args.stego),
              "carrier_level": args.carrier_level, "levels": list(received), "route": args.route,
              "rd_mode": "learned", "rd_input": "clamp", "rd_protocol": relay.rd_metadata["protocol"],
              "geometry": relay.sizes, "scale": relay.scale,
              "carrier_representation": "raw_float32" if args.stego.suffix.lower() == ".npy" else "decoded_image",
              "raw_format": "RGB HWC float32 NPY, no clipping or quantization",
              "preview_format": "round(255*clamp(raw,0,1)) PNG, display only",
              "files": {f"stego_{level}": save_tensor(args.output, f"stego_{level}", value)
                        for level, value in received.items()}}
    (args.output / "run.json").write_text(json.dumps(report, indent=2) + "\n")
    print(args.output.resolve())


@torch.inference_mode()
def main():
    args = parse_args()
    if args.torch_threads < 1:
        raise ValueError("--torch-threads must be positive")
    torch.set_num_threads(args.torch_threads)
    cv2.setNumThreads(0)
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.action == "downgrade":
        run_downgrade(args)
        return
    hce_only = args.action == "extract" and args.hce_checkpoint is not None
    checkpoint_path = args.hce_checkpoint if hce_only else args.checkpoint
    predictor_type = RHCDReceiver if hce_only else RHCD
    predictor = predictor_type(checkpoint_path, args.device, getattr(args, "rd_checkpoint", None))
    cfg = predictor.cfg
    report = {"command": shlex.join([sys.executable, *sys.argv]), "action": args.action,
              "checkpoint": str(checkpoint_path.resolve()), "checkpoint_sha256": predictor.checkpoint_sha256,
              "checkpoint_role": "receiver" if hce_only else "main",
              "base_checkpoint_sha256": predictor.base_checkpoint_sha256,
              "checkpoint_epoch": predictor.checkpoint["epoch"], "effective_config": dict(cfg),
              "files": {}, "metrics": {},
              "runtime_state": {"texture_bias": predictor.model.current_texture_bias,
                                "secret_lambda": predictor.model.secret_lambda},
              "raw_format": "RGB HWC float32 NPY, no clipping or quantization",
              "preview_format": "round(255*clamp(raw,0,1)) PNG, display only"}
    if args.action == "embed":
        paths = [args.cover, *args.secrets]
        crop = cfg.val_crop_size if args.crop_size is None else args.crop_size
        sources = [read_source(path, crop) for path in paths]
        base, secrets, targets = predictor.prepare(sources)
        stegos = predictor.embed(base, secrets)
        args.output.mkdir(parents=True)
        report["source_crop"] = crop
        report["inputs"] = [{"path": str(path.resolve()), "sha256": sha256(path)} for path in paths]
        report["files"]["cover_lr"] = save_tensor(args.output, "cover_lr", base)
        for level, (stego, secret, target) in enumerate(zip(stegos, secrets, targets), 1):
            recovered = predictor.model.decode(stego)
            for name, tensor in ((f"stego_{level}", stego), (f"secret_{level}_target", secret),
                                 (f"native_secret_{level}", recovered)):
                report["files"][name] = save_tensor(args.output, name, tensor)
            report["metrics"][f"native_s{level}"] = image_metrics(recovered, secret)[0]
            report["metrics"][f"stego{level}"] = image_metrics(stego, target)[0]
    else:
        carrier = read_carrier(args.stego)
        recovered = predictor.extract(carrier, args.carrier_level, args.levels,
                                      args.rd_mode, args.route, args.rd_input)
        args.output.mkdir(parents=True)
        report.update(carrier=str(args.stego.resolve()), carrier_sha256=sha256(args.stego),
                      carrier_level=args.carrier_level, levels=list(recovered), rd_mode=args.rd_mode,
                      route=args.route, rd_input=args.rd_input,
                      carrier_representation="raw_float32" if args.stego.suffix.lower() == ".npy" else "decoded_image",
                      native_y2_to_y1_operator=("bicubic" if args.rd_mode == "learned" and
                          predictor.rd_metadata.get("native_y2_to_y1_operator") == "bicubic" and
                          not predictor.rd_metadata.get("native_y2_supported", False) else args.rd_mode)
                          if args.carrier_level == 2 and 1 in recovered else "not_used")
        if args.rd_checkpoint:
            report["rd_checkpoint_sha256"] = sha256(args.rd_checkpoint)
        for level, tensor in recovered.items():
            name = f"recovered_secret_{level}"
            report["files"][name] = save_tensor(args.output, name, tensor)
    (args.output / "run.json").write_text(json.dumps(report, indent=2) + "\n")
    print(args.output.resolve())


if __name__ == "__main__":
    main()
