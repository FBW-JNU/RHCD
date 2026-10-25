"""Load receiver weights and tier geometry."""

import copy

import torch
from torch import nn

from base.config import CfgNode
from models.rhcd import ARCHITECTURE_VERSION, build_hce, tier_sizes


HCE_FORMAT = "rhcd_hce_v1"


class ContentExtractor(nn.Module):
    architecture_version = ARCHITECTURE_VERSION

    def __init__(self, cfg, runtime_state):
        super().__init__()
        self.hce = build_hce(cfg)
        self.rounds = int(cfg.rounds)
        self.scale = float(cfg.scale)
        self.sizes = tier_sizes(cfg)
        self.current_texture_bias = float(runtime_state["texture_bias"])
        self.secret_lambda = float(runtime_state["secret_lambda"])

    def decode(self, carrier):
        return self.hce(carrier.contiguous(), self.scale)

    def forward(self, carrier):
        return self.decode(carrier)


def load_hce(path, device="cpu"):
    payload = torch.load(path, map_location="cpu")
    if payload.get("format") != HCE_FORMAT or payload.get("architecture_version") != ARCHITECTURE_VERSION:
        raise ValueError("Expected versioned receiver weights")
    base_hash = payload.get("base_checkpoint_sha256", "")
    if not isinstance(base_hash, str) or len(base_hash) != 64 or any(character not in "0123456789abcdef" for character in base_hash):
        raise ValueError("Receiver weights require their main-checkpoint SHA256")
    cfg = CfgNode(copy.deepcopy(payload["config"]))
    if cfg.architecture_version != ARCHITECTURE_VERSION:
        raise ValueError("Receiver configuration and architecture version differ")
    model = ContentExtractor(cfg, payload["runtime_state"])
    model.hce.load_state_dict(payload["hce_state"], strict=True)
    return model.to(device), cfg, payload
