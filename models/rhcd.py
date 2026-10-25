"""RHCD progressive integration, extraction and adjacent consistency."""

import copy
import math

import torch
import torch.nn.functional as F
from torch import nn

from dataset.torch_bicubic import imresize
from models.arbedrs import EDRS
from models.arb import SourceAwareSAAdapt
from models.imp_subnet_DeepMIH import ImpMapBlock
from models.RevealNet import RevealNet


ARCHITECTURE_VERSION = "rhcd_paper_v1"


class GroupNorm2d(nn.GroupNorm):
    def forward(self, features):
        if features.device.type == "cpu":
            features = features.contiguous()
        return super().forward(features)


def set_group_normalization(module):
    for name, child in list(module.named_children()):
        if isinstance(child, (nn.BatchNorm2d, nn.InstanceNorm2d)):
            replacement = GroupNorm2d(math.gcd(8, child.num_features), child.num_features,
                                      eps=child.eps, affine=child.affine)
            if child.affine:
                with torch.no_grad():
                    replacement.weight.copy_(child.weight)
                    replacement.bias.copy_(child.bias)
            setattr(module, name, replacement)
        else:
            set_group_normalization(child)


def build_hce(cfg):
    hce_cfg = copy.deepcopy(cfg)
    hce_cfg.expert_definition = "paper"
    hce_cfg.fusion_type = getattr(cfg, "hce_fusion_type", cfg.fusion_type)
    hce_cfg.down_offset_limit = getattr(cfg, "hce_offset_limit", None)
    normalization = getattr(cfg, "hce_normalization", "original")
    head_normalization = getattr(cfg, "hce_head_normalization", "group" if normalization == "group" else "instance")
    if head_normalization not in ("none", "instance", "group"):
        raise ValueError("Unknown HCE head normalization")
    hce = RevealNet(3, 3, nhf=int(cfg.hce_hidden_channels), cfg=hce_cfg,
                    norm_layer=None if head_normalization == "none" else nn.InstanceNorm2d)
    if normalization == "group":
        set_group_normalization(hce.down_net)
    elif normalization != "original":
        raise ValueError("Unknown HCE normalization")
    if head_normalization == "group":
        for index in range(1, 6):
            norm = getattr(hce, f"norm{index}")
            setattr(hce, f"norm{index}", GroupNorm2d(math.gcd(8, norm.num_features),
                                                     norm.num_features, eps=norm.eps, affine=norm.affine))
    if head_normalization == "none":
        for index in range(1, 6):
            conv = getattr(hce, f"conv{index}")
            nn.init.kaiming_normal_(conv.weight, mode="fan_in", nonlinearity="relu")
            nn.init.zeros_(conv.bias)
        nn.init.xavier_uniform_(hce.conv6.weight)
        nn.init.zeros_(hce.conv6.bias)
    return hce


def tier_sizes(cfg):
    values = [int(cfg.base_size) * float(cfg.scale) ** index for index in range(int(cfg.rounds) + 1)]
    if int(cfg.rounds) not in (1, 2, 3, 4) or float(cfg.scale) <= 1:
        raise ValueError("Use 1–4 rounds and scale > 1")
    if any(abs(value - round(value)) > 1e-8 for value in values):
        raise ValueError("base_size * scale**tier must give exact integer dimensions")
    return [int(round(value)) for value in values]


def texture_bias(epoch, cfg):
    """epoch is zero-based; scheduled bias is additional to learned logits."""
    completed = int(epoch) + 1
    hold = int(cfg.routing_hold_epochs)
    end = int(cfg.routing_decay_end_epoch)
    initial = float(cfg.routing_texture_bias)
    if completed <= hold:
        return initial
    if end <= hold or completed >= end:
        return 0.0
    return initial * (end - completed) / (end - hold)


class RHCDSystem(nn.Module):
    """Progressive integration and content recovery."""

    architecture_version = ARCHITECTURE_VERSION

    def __init__(self, cfg):
        super().__init__()
        if cfg.architecture_version != ARCHITECTURE_VERSION:
            raise ValueError("Unsupported RHCD architecture version")
        if cfg.fusion_type != "source_moe" or cfg.expert_definition != "paper" or bool(cfg.scab_spatial_gating):
            raise ValueError("RHCD requires the configured SA-MoE and SCAB formulation")
        self.config = copy.deepcopy(dict(cfg))
        self.cfg = copy.deepcopy(cfg)
        if int(cfg.n_colors) != 3:
            raise ValueError("RHCD expects RGB source images")
        if int(cfg.n_resblocks) < 1 or int(cfg.K) < 1 or int(cfg.n_resblocks) % int(cfg.K):
            raise ValueError("n_resblocks must be a positive multiple of K")
        if int(cfg.n_feats) < 8 or int(cfg.n_feats) % 8:
            raise ValueError("n_feats must be a positive multiple of 8")
        self.rounds = int(cfg.rounds)
        self.scale = float(cfg.scale)
        self.sizes = tier_sizes(cfg)
        self.stages = nn.ModuleList()
        for index in range(self.rounds):
            stage_cfg = copy.deepcopy(cfg)
            stage_cfg.rescale = "up"
            self.stages.append(EDRS(stage_cfg, ifsec=1 if index == 0 else 2))
        self.importance = ImpMapBlock() if self.rounds > 1 else None
        self.hce = build_hce(cfg)
        self.set_secret_lambda(cfg.secret_residual_weight)
        self.current_texture_bias = 0.0

    def set_secret_lambda(self, value):
        value = float(value)
        if not math.isfinite(value) or value < 0:
            raise ValueError("Secret lambda must be finite and nonnegative")
        self.secret_lambda = value

    def set_texture_bias(self, value):
        value = float(value)
        if not math.isfinite(value):
            raise ValueError("Texture routing bias must be finite")
        for stage in self.stages:
            for module in stage.modules():
                if isinstance(module, SourceAwareSAAdapt):
                    module.external_logit_bias.zero_()
                    module.external_logit_bias[:, 1].fill_(value)
        self.current_texture_bias = value

    def forward(self, base, secrets, targets=None):
        if len(secrets) != self.rounds:
            raise ValueError("Exactly one secret is required per tier")
        if base.shape[-2:] != (self.sizes[0], self.sizes[0]):
            raise ValueError("Base image dimensions differ from the configured geometry")
        carrier, stegos = base, []
        for index, (stage, secret) in enumerate(zip(self.stages, secrets)):
            if carrier.shape != secret.shape:
                raise ValueError("Secret and current carrier dimensions must match at each tier")
            sources = [carrier, secret]
            if index:
                public_cover = imresize(base, sizes=carrier.shape[-2:]).contiguous()
                # IAL uses the public base cover, current secret and preceding carrier.
                importance = self.importance(public_cover, secret, carrier)
                sources.append(importance)
            size = self.sizes[index + 1]
            carrier = stage(torch.cat(sources, dim=1), self.scale, size, size,
                            secret_lambda=self.secret_lambda)
            stegos.append(carrier)
        if targets is not None:
            losses, native, cross = compute_losses(self, stegos, secrets, targets, self.cfg)
            return {"losses": losses, "stegos": stegos, "native": native, "cross": cross}
        return stegos

    def decode(self, carrier):
        return self.hce(carrier.contiguous(), self.scale)

    def downgrade(self, carrier, source_level, target_level, clamp=True):
        if not 1 <= target_level < source_level <= self.rounds:
            raise ValueError("Invalid downgrade levels")
        source = carrier.clamp(0, 1) if clamp else carrier
        return imresize(source.contiguous(), scale=self.scale ** (target_level - source_level)).contiguous()


def reference_generation(cfg):
    mode = cfg.get("reference_generation", "source")
    if mode not in ("source", "highest_tier"):
        raise ValueError("reference_generation must be source or highest_tier")
    return mode


def prepare_batch(batch, cfg, device):
    sizes = tier_sizes(cfg)
    sources = [batch["img_gt"]] + [batch["img_sec" if index == 0 else f"img_sec_{index + 1}"]
                                    for index in range(int(cfg.rounds))]
    sources = [source.to(device, non_blocking=True) for source in sources]
    resize = lambda tensor, size: imresize(tensor, sizes=(size, size)).contiguous()
    reference = resize(sources[0], sizes[-1]) if reference_generation(cfg) == "highest_tier" else sources[0]
    base = resize(reference, sizes[0])
    secrets = [resize(source, size) for source, size in zip(sources[1:], sizes[:-1])]
    targets = [resize(reference, size) for size in sizes[1:]]
    return base, secrets, targets


def compute_losses(model, stegos, secrets, targets, cfg):
    native = [model.decode(stego) for stego in stegos]
    image = torch.stack([F.l1_loss(prediction, target) for prediction, target in zip(stegos, targets)]).mean()
    secret = torch.stack([F.l1_loss(prediction, target) for prediction, target in zip(native, secrets)]).mean()
    representations, recoveries, cross = [], [], []
    for index in range(1, model.rounds):
        received = model.downgrade(stegos[index], index + 1, index, bool(cfg.rd_train_clamp))
        recovered = model.decode(received)
        representations.append(F.l1_loss(received, stegos[index - 1]))
        recoveries.append(F.l1_loss(recovered, secrets[index - 1]))
        cross.append(recovered)
    zero = image.new_zeros(())
    consistency_image = torch.stack(representations).mean() if representations else zero
    consistency_secret = torch.stack(recoveries).mean() if recoveries else zero
    consistency = consistency_image + consistency_secret
    total = float(cfg.loss_img_weight) * image + float(cfg.loss_secret_weight) * secret + \
        float(cfg.loss_consistency_weight) * consistency
    return {"total": total, "image": image, "secret": secret, "consistency": consistency,
            "consistency_image": consistency_image, "consistency_secret": consistency_secret}, native, cross
