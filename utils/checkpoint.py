"""RHCD model checkpoints and training states."""

import copy
import hashlib
import random
from pathlib import Path

import numpy as np
import torch

from base.config import CfgNode
from models.rhcd import ARCHITECTURE_VERSION, RHCDSystem


FORMAT = "rhcd_model_v1"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def capture_rng(data_generator=None):
    return {"torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else [],
            "python": random.getstate(), "numpy": np.random.get_state(),
            "data_generator": data_generator.get_state() if data_generator is not None else None}


def save_model(path, model, cfg, epoch, optimizer=None, scheduler=None, best_metric=None,
               data_generator=None, rank_rng_states=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    core = model.module if hasattr(model, "module") else model
    payload = {"format": FORMAT, "architecture_version": core.architecture_version,
               "config": copy.deepcopy(dict(cfg)), "epoch": int(epoch),
               "model_state": core.state_dict(),
               "runtime_state": {"texture_bias": core.current_texture_bias, "secret_lambda": core.secret_lambda},
               "best_metric": best_metric,
               "optimizer": optimizer.state_dict() if optimizer is not None else None,
               "scheduler": scheduler.state_dict() if scheduler is not None else None,
               "rng_state": capture_rng(data_generator),
               "rank_rng_states": rank_rng_states}
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_model(path, device="cpu"):
    payload = torch.load(path, map_location="cpu")
    return restore_model(payload, device)


def restore_model(payload, device="cpu"):
    if payload.get("format") != FORMAT or payload.get("architecture_version") != ARCHITECTURE_VERSION:
        raise ValueError("Expected an RHCD model checkpoint with the configured architecture")
    cfg = CfgNode(copy.deepcopy(payload["config"]))
    model = RHCDSystem(cfg)
    model.load_state_dict(payload["model_state"], strict=True)
    model.set_texture_bias(payload["runtime_state"]["texture_bias"])
    model.set_secret_lambda(payload["runtime_state"].get("secret_lambda", cfg.secret_residual_weight))
    return model.to(device), cfg, payload
