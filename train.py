#!/usr/bin/env python
"""End-to-end RHCD training with adjacent consistency."""

import argparse
import csv
import json
import math
import os
import random
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

from base.config import load_cfg_from_cfg_file, merge_cfg_from_list
from dataset.paper import RHCDDataset
from metrics.image_quality import image_metrics
from models.rhcd import RHCDSystem, prepare_batch, reference_generation, texture_bias, tier_sizes
from utils.checkpoint import capture_rng, load_model, save_model, sha256


def parse_config():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/three_round.yaml")
    parser.add_argument("opts", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    cfg = merge_cfg_from_list(load_cfg_from_cfg_file(args.config), args.opts)
    return cfg


def runtime(cfg):
    distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if cfg.device == "cuda":
        if not distributed and "CUDA_VISIBLE_DEVICES" not in os.environ:
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, cfg.train_gpu))
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; expose GPU devices or set device cpu for a bounded smoke")
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device(cfg.device)
    if distributed:
        dist.init_process_group("nccl" if device.type == "cuda" else "gloo")
    rank = dist.get_rank() if distributed else 0
    random.seed(int(cfg.manual_seed) + rank)
    np.random.seed(int(cfg.manual_seed) + rank)
    torch.manual_seed(int(cfg.manual_seed) + rank)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return device, distributed, rank


def make_datasets(cfg):
    train_list = cfg.train_list or str(Path(cfg.data_root) / "list/train.txt")
    val_list = cfg.val_list or str(Path(cfg.data_root) / "list/val.txt")
    train = RHCDDataset(train_list, cfg, training=True)
    val = RHCDDataset(val_list, cfg, training=False)
    if set(train.pool) & set(val.pool):
        raise ValueError("Training and validation source paths overlap")
    return train, val, train_list, val_list


def initialize(model, cfg):
    """Initialize from a model checkpoint."""
    if not cfg.weight:
        return None
    restored, _, payload = load_model(cfg.weight)
    if restored.architecture_version != model.architecture_version:
        raise ValueError("Initialization architecture differs from the selected configuration")
    model.load_state_dict(restored.state_dict(), strict=True)
    model.set_texture_bias(restored.current_texture_bias)
    model.set_secret_lambda(restored.secret_lambda)
    return {"path": str(Path(cfg.weight).resolve()), "sha256": sha256(cfg.weight),
            "epoch": payload["epoch"], "source_config": payload["config"]}


def hce_input_gradients(model, stegos):
    norms = {}
    parameters = list(model.hce.parameters())
    states = [parameter.requires_grad for parameter in parameters]
    try:
        for parameter in parameters:
            parameter.requires_grad_(False)
        with torch.enable_grad():
            for tier, stego in enumerate(stegos, 1):
                carrier = stego.detach().clone().requires_grad_(True)
                recovered = model.decode(carrier)
                generator = torch.Generator(device=recovered.device).manual_seed(131 + tier)
                probe = torch.randn(recovered.shape, generator=generator, device=recovered.device, dtype=recovered.dtype)
                gradient = torch.autograd.grad((recovered * probe).mean(), carrier)[0]
                norm = float(gradient.abs().max())
                if not math.isfinite(norm) or norm == 0:
                    raise ValueError(f"HCE input gradient must be finite and nonzero at tier {tier}")
                norms[f"hce_input_gradient_s{tier}"] = norm
    finally:
        for parameter, state in zip(parameters, states):
            parameter.requires_grad_(state)
    return norms


@torch.no_grad()
def validate(model, loader, cfg, device):
    model.eval()
    sums, count, rows, gradient_checks = {}, 0, [], {}
    for step, batch in enumerate(loader):
        if int(cfg.max_val_iters) > 0 and step >= int(cfg.max_val_iters):
            break
        base, secrets, targets = prepare_batch(batch, cfg, device)
        output = model(base, secrets, targets)
        if step == 0 and bool(getattr(cfg, "check_hce_input_gradients", True)):
            gradient_checks = hce_input_gradients(model, output["stegos"])
        size = base.shape[0]
        count += size
        for key, value in output["losses"].items():
            sums[key] = sums.get(key, 0.0) + float(value) * size
        pairs = []
        for level in range(model.rounds):
            pairs.extend([(f"stego{level + 1}", output["stegos"][level], targets[level]),
                          (f"native_s{level + 1}", output["native"][level], secrets[level])])
        for source in range(2, model.rounds + 1):
            for target in range(1, source):
                received = model.downgrade(output["stegos"][source - 1], source, target, bool(cfg.rd_train_clamp))
                pairs.append((f"y{source}_s{target}", model.decode(received), secrets[target - 1]))
        batch_rows = [{"index": int(index)} for index in batch["sample_index"]]
        for name, prediction, target in pairs:
            metrics = image_metrics(prediction, target)
            for row, item in zip(batch_rows, metrics):
                for key, value in item.items():
                    row[name + "_" + key] = value
                    sums[name + "_" + key] = sums.get(name + "_" + key, 0.0) + value
        rows.extend(batch_rows)
    if not count:
        raise ValueError("Validation produced no images")
    return {**{key: value / count for key, value in sums.items()}, "validation_pairs": count,
            **gradient_checks}, rows


def build_optimizer(model, cfg):
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    return torch.optim.Adam(parameters, lr=float(cfg.base_lr), betas=tuple(cfg.adam_betas),
                            weight_decay=float(cfg.weight_decay))


def main():
    cfg = parse_config()
    tier_sizes(cfg)
    if int(cfg.epochs) < 1 or int(cfg.batch_size) < 1:
        raise ValueError("epochs and batch_size must be positive")
    device, distributed, rank = runtime(cfg)
    train_data, val_data, train_list, val_list = make_datasets(cfg)
    cfg.train_manifest_sha256, cfg.val_manifest_sha256 = sha256(train_list), sha256(val_list)
    sampler = DistributedSampler(train_data, shuffle=True, seed=int(cfg.manual_seed)) if distributed else None
    generator = torch.Generator().manual_seed(int(cfg.manual_seed) + rank)
    train_loader = DataLoader(train_data, batch_size=int(cfg.batch_size), sampler=sampler,
                              shuffle=sampler is None, num_workers=int(cfg.workers),
                              pin_memory=device.type == "cuda", generator=generator)
    val_loader = DataLoader(val_data, batch_size=int(cfg.batch_size_val), shuffle=False,
                            num_workers=int(cfg.workers), pin_memory=device.type == "cuda")
    model = RHCDSystem(cfg).to(device)
    initialization = initialize(model, cfg)
    optimizer = build_optimizer(model, cfg)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(cfg.lr_step_size), gamma=float(cfg.lr_gamma))
    start, best = 0, math.inf
    if cfg.resume:
        restored, restored_cfg, payload = load_model(cfg.resume, device)
        if cfg.architecture_version != restored_cfg.architecture_version:
            raise ValueError("Resume configuration differs: architecture_version")
        for key, fallback in (("hce_fusion_type", cfg.fusion_type), ("hce_normalization", "original"),
                              ("hce_offset_limit", None)):
            if cfg.get(key, fallback) != restored_cfg.get(key, restored_cfg.fusion_type if key == "hce_fusion_type" else fallback):
                raise ValueError("Resume configuration differs: " + key)
        current_head_norm = cfg.get("hce_head_normalization", "group" if cfg.get("hce_normalization") == "group" else "instance")
        restored_head_norm = restored_cfg.get("hce_head_normalization", "group" if restored_cfg.get("hce_normalization") == "group" else "instance")
        if current_head_norm != restored_head_norm:
            raise ValueError("Resume configuration differs: hce_head_normalization")
        if cfg.get("select_after_epoch", 0) != restored_cfg.get("select_after_epoch", 0):
            raise ValueError("Resume configuration differs: select_after_epoch")
        if reference_generation(cfg) != reference_generation(restored_cfg):
            raise ValueError("Resume configuration differs: reference_generation")
        if bool(cfg.get("rd_stepwise_clamp", False)) != bool(restored_cfg.get("rd_stepwise_clamp", False)):
            raise ValueError("Resume configuration differs: rd_stepwise_clamp")
        for key in ("rounds", "scale", "base_size", "expert_definition", "scab_spatial_gating",
                    "n_feats", "n_resblocks", "hce_hidden_channels", "loss_img_weight",
                    "loss_secret_weight", "loss_consistency_weight", "rd_train_clamp",
                    "routing_texture_bias", "routing_hold_epochs", "routing_decay_end_epoch",
                    "adam_betas", "lr_step_size", "lr_gamma", "train_manifest_sha256", "val_manifest_sha256",
                    "base_lr", "weight_decay", "batch_size", "train_crop_size", "val_crop_size", "manual_seed",
                    "fixed_val_secret_seed", "hflip", "rotation", "detail_residual_scale", "secret_residual_weight",
                    "max_val_iters", "max_train_iters", "batch_size_val", "secret_residual_scale", "gradient_clip"):
            if cfg[key] != restored_cfg[key]:
                raise ValueError("Resume configuration differs: " + key)
        model.load_state_dict(restored.state_dict(), strict=True)
        model.set_texture_bias(restored.current_texture_bias)
        model.set_secret_lambda(restored.secret_lambda)
        if payload["optimizer"] is None or payload["scheduler"] is None:
            raise ValueError("Resume requires optimizer and scheduler states")
        optimizer.load_state_dict(payload["optimizer"])
        scheduler.load_state_dict(payload["scheduler"])
        start, best = int(payload["epoch"]), float(payload["best_metric"])
        states = payload.get("rank_rng_states")
        if states is not None and len(states) != (dist.get_world_size() if distributed else 1):
            raise ValueError("Resume requires the same distributed world size")
        if distributed and states is None:
            raise ValueError("Distributed resume requires per-rank RNG states")
        state = states[rank] if states is not None else payload["rng_state"]
        torch.set_rng_state(state["torch"])
        random.setstate(state["python"])
        np.random.set_state(state["numpy"])
        if state["data_generator"] is not None:
            generator.set_state(state["data_generator"])
        if device.type == "cuda" and state["cuda"]:
            torch.cuda.set_rng_state_all(state["cuda"])
        del restored
    output = Path(cfg.save_path)
    if rank == 0:
        output.mkdir(parents=True, exist_ok=True)
        record = {"config": dict(cfg), "architecture": model.architecture_version,
                  "parameters": sum(p.numel() for p in model.parameters()),
                  "runtime_state": {"texture_bias": model.current_texture_bias, "secret_lambda": model.secret_lambda},
                  "initialization": initialization, "train_list": str(Path(train_list).resolve()),
                  "train_list_sha256": sha256(train_list), "val_list": str(Path(val_list).resolve()),
                  "val_list_sha256": sha256(val_list), "training_images": len(train_data),
                  "validation_images": len(val_data), "geometry": model.sizes,
                  "optimizer": {"name": "Adam", "betas": list(optimizer.param_groups[0]["betas"])}}
        (output / "effective_config.json").write_text(json.dumps(record, indent=2) + "\n")
    core = model
    if distributed:
        model = DistributedDataParallel(model, device_ids=[device.index] if device.type == "cuda" else None,
                                        find_unused_parameters=True)
    if bool(cfg.eval_only):
        if rank == 0:
            stats, _ = validate(core, val_loader, cfg, device)
            (output / "eval_only.json").write_text(json.dumps(stats, indent=2) + "\n")
            print(json.dumps(stats), flush=True)
        if distributed:
            dist.barrier()
            dist.destroy_process_group()
        return
    for epoch in range(start, int(cfg.epochs)):
        train_data.set_epoch(epoch)
        if sampler is not None:
            sampler.set_epoch(epoch)
        model.train()
        core.set_texture_bias(texture_bias(epoch, cfg))
        core.set_secret_lambda(cfg.secret_residual_weight)
        begin = time.monotonic()
        for step, batch in enumerate(train_loader):
            if int(cfg.max_train_iters) > 0 and step >= int(cfg.max_train_iters):
                break
            base, secrets, targets = prepare_batch(batch, cfg, device)
            result = model(base, secrets, targets)
            total = result["losses"]["total"]
            if not torch.isfinite(total):
                raise ValueError("Nonfinite training loss")
            optimizer.zero_grad(set_to_none=True)
            total.backward()
            norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.grad is not None],
                                                  float(cfg.gradient_clip) if float(cfg.gradient_clip) > 0 else math.inf)
            if not torch.isfinite(norm):
                raise ValueError("Nonfinite optimizer gradient")
            optimizer.step()
            if rank == 0 and (step == 0 or (step + 1) % int(cfg.print_freq) == 0):
                print(json.dumps({"epoch": epoch + 1, "iteration": step + 1,
                                  "losses": {key: float(value) for key, value in result["losses"].items()},
                                  "texture_bias": core.current_texture_bias, "secret_lambda": core.secret_lambda,
                                  "gradient_norm": float(norm)}), flush=True)
        scheduler.step()
        if distributed:
            dist.barrier()
        selected = False
        if rank == 0:
            if (epoch + 1) % int(cfg.eval_freq) == 0:
                stats, rows = validate(core, val_loader, cfg, device)
                eligible = epoch + 1 >= int(getattr(cfg, "select_after_epoch", 0))
                selected = eligible and stats["total"] < best
                if eligible:
                    best = min(best, stats["total"])
                with (output / "validation.jsonl").open("a") as handle:
                    handle.write(json.dumps({"epoch": epoch + 1, "secret_lambda": core.secret_lambda,
                                             "texture_bias": core.current_texture_bias, **stats}) + "\n")
                with (output / "validation_last.csv").open("w", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                    writer.writeheader()
                    writer.writerows(rows)
                print("VALIDATION", epoch + 1, json.dumps(stats), flush=True)
        rank_states = None
        if distributed:
            rank_states = [None] * dist.get_world_size()
            dist.all_gather_object(rank_states, capture_rng(generator))
        if rank == 0:
            save_model(output / "model/last.pth", core, cfg, epoch + 1, optimizer, scheduler, best, generator,
                       rank_rng_states=rank_states)
            if selected:
                shutil.copyfile(output / "model/last.pth", output / "model/best.pth")
            print("EPOCH_SECONDS", epoch + 1, time.monotonic() - begin, flush=True)
        if distributed:
            dist.barrier()
    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
