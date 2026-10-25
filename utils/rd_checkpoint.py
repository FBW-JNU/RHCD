"""Versioned RD weights and their compatible main-network checkpoint."""

from pathlib import Path

import torch

from models.cross_level_downsample import HighResolutionCrossLevelDownsampler
from utils.checkpoint import sha256


RD_FORMAT = "rhcd_rd_v1"
RD_PROTOCOL = "clamp_each_step_float_residual_unclipped_detached_chain_v3"
INITIAL_CLAMP_RD_PROTOCOL = "clamp_initial_float_residual_unclipped_detached_chain_v2"
LEGACY_RD_PROTOCOL = "clamp_y3_float_residual_unclipped_detached_chain_v1"


def rd_architecture(cfg):
    architecture = {key: cfg["rd_" + key] for key in
                    ("hidden_channels", "num_blocks", "residual_cap", "chain_residual_cap")}
    architecture["chain_input_clamp"] = bool(cfg.get("rd_stepwise_clamp", False))
    return architecture


def rd_protocol(model):
    return RD_PROTOCOL if model.chain_input_clamp else INITIAL_CLAMP_RD_PROTOCOL


def load_rd_operator(path, device="cpu"):
    payload = torch.load(path, map_location="cpu")
    if payload.get("format") != RD_FORMAT or payload.get("protocol") not in (
            RD_PROTOCOL, INITIAL_CLAMP_RD_PROTOCOL, LEGACY_RD_PROTOCOL):
        raise ValueError("Expected a versioned RHCD RD checkpoint")
    base_hash = payload.get("base_checkpoint_sha256", "")
    if not isinstance(base_hash, str) or len(base_hash) != 64 or any(character not in "0123456789abcdef" for character in base_hash):
        raise ValueError("RD weights require their main-checkpoint SHA256")
    architecture = dict(payload["architecture"])
    stepwise_clamp = payload["protocol"] == RD_PROTOCOL
    if "chain_input_clamp" in architecture and bool(architecture["chain_input_clamp"]) != stepwise_clamp:
        raise ValueError("RD architecture and stepwise protocol differ")
    architecture["chain_input_clamp"] = stepwise_clamp
    model = HighResolutionCrossLevelDownsampler(**architecture)
    model.load_state_dict(payload["state_dict"], strict=True)
    return model.to(device), payload


def load_rd(path, main_checkpoint=None, device="cpu", *, main_checkpoint_sha256=None):
    expected = sha256(main_checkpoint) if main_checkpoint is not None else main_checkpoint_sha256
    if expected is None:
        raise ValueError("RD loading requires its main-checkpoint path or SHA256")
    if main_checkpoint_sha256 is not None and main_checkpoint_sha256 != expected:
        raise ValueError("Main-checkpoint path and SHA256 differ")
    model, payload = load_rd_operator(path, device)
    if payload["base_checkpoint_sha256"] != expected:
        raise ValueError("RD checkpoint does not match the main-network checkpoint")
    return model, payload


def save_rd(path, model, base_checkpoint, cfg, epoch, optimizer, scheduler, best, rng_state,
            base_checkpoint_sha256=None, native_y2_supported=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    base_hash = sha256(base_checkpoint)
    if base_checkpoint_sha256 is not None and base_hash != base_checkpoint_sha256:
        raise ValueError("Base checkpoint changed during RD training")
    architecture = rd_architecture(cfg)
    if architecture["chain_input_clamp"] != model.chain_input_clamp:
        raise ValueError("RD configuration and stepwise protocol differ")
    payload = {"format": RD_FORMAT, "protocol": rd_protocol(model), "architecture": architecture,
               "native_y2_supported": bool(native_y2_supported),
               "state_dict": model.state_dict(), "base_checkpoint": str(Path(base_checkpoint).resolve()),
               "base_checkpoint_sha256": base_hash, "config": dict(cfg), "epoch": int(epoch),
               "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
               "best_metric": float(best), "rng_state": rng_state}
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)
