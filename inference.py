"""Progressive embedding and native/cross-tier content extraction."""

import copy

import torch

from base.config import CfgNode
from dataset.torch_bicubic import imresize
from models.rhcd import prepare_batch, tier_sizes
from test.checkpoint import load_checkpoint
from utils.checkpoint import sha256
from utils.hce_checkpoint import load_hce
from utils.rd_checkpoint import load_rd, load_rd_operator


def configure_inference():
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False


class ResolutionHierarchy:
    def representations(self, carrier, carrier_level, rd_mode="learned", route="direct", rd_input="clamp"):
        if not 1 <= carrier_level <= self.rounds:
            raise ValueError("Invalid carrier level")
        size = self.sizes[carrier_level]
        if carrier.shape[-2:] != (size, size):
            raise ValueError(f"Tier {carrier_level} requires {size}x{size} pixels")
        if rd_mode not in ("bicubic", "learned") or route not in ("direct", "stepwise"):
            raise ValueError("Unsupported RD mode or route")
        if rd_input not in ("raw", "clamp"):
            raise ValueError("rd_input must be raw or clamp")
        received = {carrier_level: carrier}
        if carrier_level == 1:
            return received
        if rd_mode == "learned":
            if self.rd is None:
                raise ValueError("--rd-checkpoint is required for learned RD")
            if rd_input != "clamp":
                raise ValueError("Learned RD uses its recorded input-clamp protocol")
            if carrier_level == 3:
                output = self.rd(carrier, self.scale)
                received[2] = output["pred_2"]
                received[1] = output["pred_1" if route == "direct" else "pred_chain_1"]
                return received
            if self.rd_metadata.get("native_y2_supported", False):
                received[1] = self.rd(carrier, self.scale, source_level=2)["pred_1"]
                return received
            if self.rd_metadata.get("native_y2_to_y1_operator") == "bicubic":
                source = carrier.clamp(0, 1).contiguous()
                received[1] = imresize(source, sizes=(self.sizes[1],) * 2).contiguous()
                return received
            raise ValueError("Native Y2->Y1 requires an RD checkpoint trained on that path")
        source = carrier.clamp(0, 1) if rd_input == "clamp" else carrier
        if route == "direct":
            for level in range(1, carrier_level):
                received[level] = imresize(source.contiguous(), sizes=(self.sizes[level],) * 2).contiguous()
        else:
            for level in range(carrier_level - 1, 0, -1):
                if rd_input == "clamp" and bool(self.cfg.get("rd_stepwise_clamp", False)):
                    source = source.clamp(0, 1)
                source = imresize(source.contiguous(), sizes=(self.sizes[level],) * 2).contiguous()
                received[level] = source
        return received


class ContentExtraction(ResolutionHierarchy):
    @torch.inference_mode()
    def extract(self, carrier, carrier_level, levels=None, rd_mode="learned", route="direct", rd_input="clamp"):
        if not 1 <= carrier_level <= self.rounds:
            raise ValueError("Invalid carrier level")
        levels = list(range(1, carrier_level + 1)) if levels is None else list(levels)
        if not levels or len(set(levels)) != len(levels) or any(level < 1 or level > carrier_level for level in levels):
            raise ValueError("Secret levels must be unique and <= carrier level")
        if levels == [carrier_level]:
            size = self.sizes[carrier_level]
            if carrier.shape[-2:] != (size, size):
                raise ValueError(f"Tier {carrier_level} requires {size}x{size} pixels")
            return {carrier_level: self.model.decode(carrier.to(self.device))}
        received = self.representations(carrier.to(self.device), carrier_level, rd_mode, route, rd_input)
        return {level: self.model.decode(received[level]) for level in levels}


class RHCD(ContentExtraction):
    def __init__(self, checkpoint, device="cpu", rd_checkpoint=None):
        configure_inference()
        self.device = torch.device(device)
        self.model, self.cfg, self.checkpoint = load_checkpoint(checkpoint, self.device)
        self.model.eval().requires_grad_(False)
        self.checkpoint_sha256 = sha256(checkpoint)
        self.base_checkpoint_sha256 = self.checkpoint_sha256
        self.rounds, self.scale, self.sizes = self.model.rounds, self.model.scale, self.model.sizes
        self.rd, self.rd_metadata = None, None
        if rd_checkpoint is not None:
            if self.rounds != 3:
                raise ValueError("Learned RD currently supports the three-tier geometry")
            self.rd, self.rd_metadata = load_rd(rd_checkpoint, checkpoint, self.device)
            self.rd.eval().requires_grad_(False)

    def prepare(self, sources):
        if len(sources) != self.rounds + 1:
            raise ValueError("Expected one cover and one secret per round")
        keys = ["img_gt", "img_sec"] + [f"img_sec_{i}" for i in range(2, self.rounds + 1)]
        return prepare_batch(dict(zip(keys, sources)), self.cfg, self.device)

    @torch.inference_mode()
    def embed(self, base, secrets):
        return self.model(base.to(self.device), [secret.to(self.device) for secret in secrets])


class RHCDReceiver(ContentExtraction):
    """Content extraction with receiver weights and optional RD."""

    def __init__(self, hce_checkpoint, device="cpu", rd_checkpoint=None):
        configure_inference()
        self.device = torch.device(device)
        self.model, self.cfg, self.checkpoint = load_hce(hce_checkpoint, self.device)
        self.model.eval().requires_grad_(False)
        self.checkpoint_sha256 = sha256(hce_checkpoint)
        self.base_checkpoint_sha256 = self.checkpoint["base_checkpoint_sha256"]
        self.rounds, self.scale, self.sizes = self.model.rounds, self.model.scale, self.model.sizes
        self.rd, self.rd_metadata = None, None
        if rd_checkpoint is not None:
            if self.rounds != 3:
                raise ValueError("Learned RD currently supports the three-tier geometry")
            self.rd, self.rd_metadata = load_rd(rd_checkpoint, device=self.device,
                                               main_checkpoint_sha256=self.base_checkpoint_sha256)
            self.rd.eval().requires_grad_(False)


class RHCDRelay(ResolutionHierarchy):
    """Generate lower-tier representations using the public RD weights."""

    def __init__(self, rd_checkpoint, device="cpu"):
        configure_inference()
        self.device = torch.device(device)
        self.rd, self.rd_metadata = load_rd_operator(rd_checkpoint, self.device)
        self.rd.eval().requires_grad_(False)
        self.cfg = CfgNode(copy.deepcopy(self.rd_metadata["config"]))
        self.rounds, self.scale, self.sizes = int(self.cfg.rounds), float(self.cfg.scale), tier_sizes(self.cfg)
        if self.rounds != 3:
            raise ValueError("Learned RD currently supports the three-tier geometry")
        self.checkpoint_sha256 = sha256(rd_checkpoint)
        self.base_checkpoint_sha256 = self.rd_metadata["base_checkpoint_sha256"]

    @torch.inference_mode()
    def downgrade(self, carrier, carrier_level, levels=None, route="direct"):
        levels = list(range(1, carrier_level)) if levels is None else list(levels)
        if not levels or len(set(levels)) != len(levels) or any(level < 1 or level >= carrier_level for level in levels):
            raise ValueError("Downgrade levels must be unique and below carrier level")
        received = self.representations(carrier.to(self.device), carrier_level, route=route)
        return {level: received[level] for level in levels}
