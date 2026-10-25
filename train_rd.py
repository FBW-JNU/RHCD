#!/usr/bin/env python
"""Train direct and stepwise learned RD with the main RHCD model frozen."""

import argparse
import csv
import json
import math
import os
import random
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from base.config import load_cfg_from_cfg_file, merge_cfg_from_list
from metrics.image_quality import image_metrics
from models.cross_level_downsample import HighResolutionCrossLevelDownsampler
from models.rhcd import prepare_batch, reference_generation
from train import make_datasets, runtime
from utils.checkpoint import load_model, sha256
from utils.rd_checkpoint import load_rd, rd_architecture, rd_protocol, save_rd


def gradient_l1(prediction, target):
    return (F.l1_loss(prediction[..., 1:, :] - prediction[..., :-1, :],
                      target[..., 1:, :] - target[..., :-1, :]) +
            F.l1_loss(prediction[..., :, 1:] - prediction[..., :, :-1],
                      target[..., :, 1:] - target[..., :, :-1])) / 2


def high_pass_l1(prediction, target):
    high_pass = lambda value: value - F.avg_pool2d(value, 3, stride=1, padding=1)
    return F.l1_loss(high_pass(prediction), high_pass(target))


def objective(output, main, stegos, secrets, cfg, native_output=None):
    predictions = [output["pred_2"], output["pred_1"], output["pred_chain_1"]]
    targets = [stegos[1], stegos[0], stegos[0]]
    # Main parameters/stats are frozen, but autograd through HCE must remain
    # enabled: these input gradients train the RD residual predictors.
    recovered = [main.decode(prediction) for prediction in predictions]
    loss = {"secret2": F.l1_loss(recovered[0], secrets[1]),
            "secret1": F.l1_loss(recovered[1], secrets[0]),
            "chain_secret1": F.l1_loss(recovered[2], secrets[0]),
            "image2": F.l1_loss(predictions[0], targets[0]),
            "image1": F.l1_loss(predictions[1], targets[1]),
            "chain_image1": F.l1_loss(predictions[2], targets[2]),
            "consistency": F.l1_loss(predictions[2], predictions[1].detach()),
            "gradient": torch.stack([gradient_l1(p, t) for p, t in zip(predictions, targets)]).mean(),
            "high_pass": torch.stack([high_pass_l1(p, t) for p, t in zip(predictions, targets)]).mean(),
            "residual": torch.stack([output[key].abs().mean() for key in
                                      ("residual_2", "residual_1", "residual_chain_1")]).mean()}
    if native_output is not None:
        native_recovered = main.decode(native_output["pred_1"])
        loss["native21_secret1"] = F.l1_loss(native_recovered, secrets[0])
        loss["native21_image1"] = F.l1_loss(native_output["pred_1"], stegos[0])
        loss["gradient"] = (3 * loss["gradient"] + gradient_l1(native_output["pred_1"], stegos[0])) / 4
        loss["high_pass"] = (3 * loss["high_pass"] + high_pass_l1(native_output["pred_1"], stegos[0])) / 4
        loss["residual"] = (3 * loss["residual"] + native_output["residual_1"].abs().mean()) / 4
        recovered.append(native_recovered)
    loss["total"] = sum(float(cfg["rd_loss_" + key]) * value for key, value in loss.items())
    return loss, recovered


@torch.no_grad()
def validate(rd, main, loader, cfg, device):
    rd.eval()
    main.eval()
    sums, count, rows = {}, 0, []
    for index, batch in enumerate(loader):
        if int(cfg.max_val_iters) > 0 and index >= int(cfg.max_val_iters):
            break
        base, secrets, targets = prepare_batch(batch, cfg, device)
        stegos = main(base, secrets)
        output = rd(stegos[2], main.scale)
        native_output = rd(stegos[1], main.scale, source_level=2)
        losses, recovered = objective(output, main, stegos, secrets, cfg, native_output=native_output)
        size = base.shape[0]
        count += size
        for key, value in losses.items():
            sums[key] = sums.get(key, 0.0) + float(value) * size
        batch_rows = [{"index": int(item)} for item in batch["sample_index"]]
        pairs = [("learned_y3_s2", recovered[0], secrets[1]),
                 ("learned_y3_s1", recovered[1], secrets[0]),
                 ("learned_y3_step_s1", recovered[2], secrets[0]),
                 ("learned_y2_s1", recovered[3], secrets[0]),
                 ("bicubic_y3_s2", main.decode(output["base_2"]), secrets[1]),
                 ("bicubic_y3_s1", main.decode(output["base_1"]), secrets[0])]
        for name, prediction, target in pairs:
            for row, metrics in zip(batch_rows, image_metrics(prediction, target)):
                for key, value in metrics.items():
                    row[name + "_" + key] = value
                    sums[name + "_" + key] = sums.get(name + "_" + key, 0.0) + value
        rows.extend(batch_rows)
    if not count:
        raise ValueError("Empty RD validation")
    return {**{key: value / count for key, value in sums.items()}, "validation_pairs": count}, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/three_round.yaml")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--stop-after-epoch", type=int, default=0,
                        help="Bound a run without changing the full cosine schedule")
    parser.add_argument("opts", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    cfg = merge_cfg_from_list(load_cfg_from_cfg_file(args.config), args.opts)
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("RD training uses one GPU; launch train.py for distributed main training")
    device, _, _ = runtime(cfg)
    main_hash = sha256(args.checkpoint)
    main_model, main_cfg, main_payload = load_model(args.checkpoint, device)
    if sha256(args.checkpoint) != main_hash:
        raise ValueError("Base checkpoint changed while loading")
    for key in ("rounds", "scale", "base_size"):
        if cfg[key] != main_cfg[key]:
            raise ValueError("Main checkpoint geometry differs: " + key)
    if reference_generation(cfg) != reference_generation(main_cfg):
        raise ValueError("Main checkpoint reference_generation differs")
    if main_model.rounds != 3 or int(cfg.rd_epochs) < 1:
        raise ValueError("Learned RD requires three tiers and rd_epochs >= 1")
    main_model.eval().requires_grad_(False)
    train_data, val_data, train_list, val_list = make_datasets(cfg)
    cfg.train_manifest_sha256, cfg.val_manifest_sha256 = sha256(train_list), sha256(val_list)
    generator = torch.Generator().manual_seed(int(cfg.manual_seed))
    loader = DataLoader(train_data, batch_size=int(cfg.batch_size), shuffle=True,
                        num_workers=int(cfg.workers), generator=generator, pin_memory=device.type == "cuda")
    val_loader = DataLoader(val_data, batch_size=int(cfg.batch_size_val), num_workers=int(cfg.workers))
    rd = HighResolutionCrossLevelDownsampler(**rd_architecture(cfg)).to(device)
    direct, chain = [], []
    for name, parameter in rd.named_parameters():
        (chain if name.startswith("y2_to_y1_") else direct).append(parameter)
    optimizer = torch.optim.AdamW([{"params": direct, "lr": float(cfg.rd_lr)},
                                   {"params": chain, "lr": float(cfg.rd_chain_lr)}],
                                  weight_decay=float(cfg.rd_weight_decay))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(cfg.rd_epochs),
                                                          eta_min=float(cfg.rd_min_lr))
    start, best = 0, math.inf
    if args.resume:
        restored, payload = load_rd(args.resume, args.checkpoint, device)
        if not payload.get("native_y2_supported", False):
            raise ValueError("RD resume requires the same native Y2 training protocol")
        if rd_architecture(cfg)["chain_input_clamp"] != restored.chain_input_clamp:
            raise ValueError("RD resume configuration differs: rd_stepwise_clamp")
        if reference_generation(cfg) != reference_generation(payload["config"]):
            raise ValueError("RD resume configuration differs: reference_generation")
        for key, value in dict(cfg).items():
            if key in ("rd_stepwise_clamp", "reference_generation"):
                continue
            if key.startswith("rd_") or key in ("train_manifest_sha256", "val_manifest_sha256", "manual_seed",
                    "train_crop_size", "val_crop_size", "batch_size", "batch_size_val", "hflip", "rotation",
                    "fixed_val_secret_seed", "max_train_iters", "max_val_iters", "scale", "rounds", "base_size"):
                if value != payload["config"][key]:
                    raise ValueError("RD resume configuration differs: " + key)
        rd.load_state_dict(restored.state_dict(), strict=True)
        optimizer.load_state_dict(payload["optimizer"])
        scheduler.load_state_dict(payload["scheduler"])
        start, best = int(payload["epoch"]), float(payload["best_metric"])
        state = payload["rng_state"]
        torch.set_rng_state(state["torch"])
        np.random.set_state(state["numpy"])
        random.setstate(state["python"])
        generator.set_state(state["data_generator"])
        if device.type == "cuda":
            torch.cuda.set_rng_state_all(state["cuda"])
        del restored
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "effective_config.json").write_text(json.dumps({"config": dict(cfg),
        "main_checkpoint": str(args.checkpoint.resolve()), "main_checkpoint_sha256": main_hash,
        "main_config": dict(main_cfg),
        "main_epoch": main_payload["epoch"], "main_texture_bias": main_model.current_texture_bias,
        "protocol": rd_protocol(rd),
        "parameters": sum(p.numel() for p in rd.parameters())}, indent=2) + "\n")
    if bool(cfg.eval_only):
        stats, _ = validate(rd, main_model, val_loader, cfg, device)
        (args.output / "eval_only.json").write_text(json.dumps(stats, indent=2) + "\n")
        return
    for epoch in range(start, int(cfg.rd_epochs)):
        train_data.set_epoch(epoch)
        rd.train()
        for step, batch in enumerate(loader):
            if int(cfg.max_train_iters) > 0 and step >= int(cfg.max_train_iters):
                break
            base, secrets, targets = prepare_batch(batch, cfg, device)
            with torch.no_grad():
                stegos = main_model(base, secrets)
            losses, _ = objective(rd(stegos[2], main_model.scale), main_model, stegos, secrets, cfg,
                                  native_output=rd(stegos[1], main_model.scale, source_level=2))
            if not torch.isfinite(losses["total"]):
                raise ValueError("Nonfinite RD loss")
            optimizer.zero_grad(set_to_none=True)
            losses["total"].backward()
            norm = torch.nn.utils.clip_grad_norm_(rd.parameters(), float(cfg.gradient_clip) or 1.0)
            if not torch.isfinite(norm):
                raise ValueError("Nonfinite RD gradient")
            optimizer.step()
            if step == 0 or (step + 1) % int(cfg.print_freq) == 0:
                print(json.dumps({"epoch": epoch + 1, "iteration": step + 1, "gradient_norm": float(norm),
                                  "losses": {key: float(value) for key, value in losses.items()}}), flush=True)
        scheduler.step()
        stats, rows = validate(rd, main_model, val_loader, cfg, device)
        selected = stats["total"] < best
        best = min(best, stats["total"])
        with (args.output / "validation.jsonl").open("a") as handle:
            handle.write(json.dumps({"epoch": epoch + 1, **stats}) + "\n")
        with (args.output / "validation_last.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        state = {"torch": torch.get_rng_state(), "numpy": np.random.get_state(), "python": random.getstate(),
                 "data_generator": generator.get_state(),
                 "cuda": torch.cuda.get_rng_state_all() if device.type == "cuda" else []}
        save_rd(args.output / "model/last.pth", rd, args.checkpoint, cfg, epoch + 1,
                optimizer, scheduler, best, state, base_checkpoint_sha256=main_hash, native_y2_supported=True)
        if selected:
            shutil.copyfile(args.output / "model/last.pth", args.output / "model/best.pth")
        print("VALIDATION", epoch + 1, json.dumps(stats), flush=True)
        if args.stop_after_epoch > 0 and epoch + 1 >= args.stop_after_epoch:
            break


if __name__ == "__main__":
    main()
