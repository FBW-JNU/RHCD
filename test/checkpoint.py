"""Load model weights for embedding and extraction."""

import copy
import math

import torch
from torch import nn

from base.config import CfgNode
from models.arbedrs import EDRS
from models.RevealNet import RevealNet
from models.rhcd import tier_sizes
from utils.checkpoint import FORMAT, restore_model


class ConstantBias(nn.Module):
    def __init__(self, convolution):
        super().__init__()
        self.bias = nn.Parameter(convolution.bias.detach().clone())

    def forward(self, tensor):
        return self.bias.view(1, -1, 1, 1).expand(tensor.shape[0], -1, tensor.shape[2], tensor.shape[3])


class PrefixConv2d(nn.Conv2d):
    def forward(self, tensor):
        return super().forward(tensor[:, :self.in_channels].contiguous())


class ScaleBasisConv(nn.Module):
    def __init__(self, original, rank, anchor_scale):
        super().__init__()
        if original.bias or rank != 1:
            raise ValueError("Expected an unbiased rank-one scale basis")
        self.routing = copy.deepcopy(original.routing)
        self.stride, self.padding = original.stride, original.padding
        self.anchor_scale = float(anchor_scale)
        with torch.no_grad():
            coordinate = original.weight_pool.new_ones(1, 2) / self.anchor_scale
            routes = original.routing(coordinate).flatten()
            weight = (routes.view(-1, 1, 1, 1, 1) * original.weight_pool).sum(0)
            basis = (original.weight_pool[:rank] - weight).clone()
        self.anchor_weight = nn.Parameter(weight)
        self.basis = nn.Parameter(basis)
        self.register_buffer("anchor_routes", routes.detach().clone())
        projection = original.weight_pool.new_zeros(original.num_experts, rank)
        projection[0, 0] = 1
        self.register_buffer("projection", projection)

    def forward(self, tensor, scale):
        routes = self.routing(tensor.new_ones(1, 2) / scale).flatten()
        coefficients = (routes - self.anchor_routes) @ self.projection
        weight = self.anchor_weight + (coefficients.view(-1, 1, 1, 1, 1) * self.basis).sum(0)
        return torch.nn.functional.conv2d(tensor, weight, stride=self.stride, padding=self.padding)


def configure_modules(model):
    for module in list(model.modules()):
        if not isinstance(module, EDRS):
            continue
        if module.fixed_scale or not module.use_source_moe or module.source_stem is None:
            raise ValueError("Invalid pretrained module configuration")
        module.head = nn.Identity()
        if module.rescale == "up" and hasattr(module, "controlled_sample"):
            module.sa_sample = nn.Identity()
        for adapt in module.sa_adapt:
            if "secret" not in module.source_names:
                adapt.secret = ConstantBias(adapt.secret)
            adapt.modulate = ConstantBias(adapt.modulate)
            old = adapt.router[0]
            width = old.out_channels * (1 + len(module.source_names))
            adapted = PrefixConv2d(width, old.out_channels, 1, bias=True).to(old.weight)
            adapt.router[0] = adapted


class PretrainedNetwork(nn.Module):
    architecture_version = "rhcd_compact_v1"

    def __init__(self, payload):
        super().__init__()
        self.cfg = CfgNode(copy.deepcopy(payload["config"]))
        self.rounds, self.scale = int(self.cfg.rounds), float(self.cfg.scale)
        if self.rounds != 3:
            raise ValueError("Expected three-round weights")
        self.sizes = tier_sizes(self.cfg)
        stage_cfg = copy.deepcopy(self.cfg)
        stage_cfg.rescale = "up"
        stage_cfg.expert_definition = "legacy"
        stage_cfg.scab_spatial_gating = True
        self.hider = nn.Module()
        for name in ("up_net", "up_net_2", "up_net_3"):
            self.hider.add_module(name, EDRS(copy.deepcopy(stage_cfg), ifsec=1))
        self.reveals = nn.ModuleList([RevealNet(3, 3, cfg=copy.deepcopy(stage_cfg)) for _ in range(3)])
        configure_modules(self)
        for entry in payload["factorizations"]:
            if entry.get("kind") != "scale_basis":
                raise ValueError("Unsupported convolution representation")
            original = self.get_submodule(entry["name"])
            parent, _, leaf = entry["name"].rpartition(".")
            self.get_submodule(parent)._modules[leaf] = ScaleBasisConv(original, entry["rank"], entry["anchor_scale"])
        self.load_tensors(payload["model_state"])
        expected = payload.get("parameters")
        if expected is not None and sum(parameter.numel() for parameter in self.parameters()) != expected:
            raise ValueError("Checkpoint parameter count differs")
        state = payload["runtime_state"]
        self.secret_lambda = float(state["secret_lambda"])
        self.current_texture_bias = float(state["texture_bias"])
        if not math.isfinite(self.current_texture_bias) or not math.isfinite(self.secret_lambda) or self.secret_lambda < 0:
            raise ValueError("Invalid checkpoint runtime state")
        if float(self.cfg.secret_residual_weight) <= 0:
            raise ValueError("Invalid embedding coefficient")
        for stage in self.stages:
            for module in stage.sa_adapt:
                module.external_logit_bias[:, 1].fill_(self.current_texture_bias)

    @property
    def stages(self):
        return self.hider.up_net, self.hider.up_net_2, self.hider.up_net_3

    @torch.no_grad()
    def load_tensors(self, state):
        target = self.state_dict()
        if set(target) != set(state):
            raise ValueError("Checkpoint tensor keys differ from the model")
        for key, value in target.items():
            source = state[key]
            if value.shape != source.shape or value.dtype != source.dtype:
                raise ValueError("Checkpoint tensor shape/dtype differs: " + key)
            if not torch.isfinite(source).all():
                raise ValueError("Nonfinite checkpoint tensor: " + key)
            value.copy_(source)

    @torch.inference_mode()
    def forward(self, base, secrets):
        if len(secrets) != 3 or base.shape[-2:] != (self.sizes[0], self.sizes[0]):
            raise ValueError("Expected the configured base image and three secrets")
        ratio = self.secret_lambda / float(self.cfg.secret_residual_weight)
        carrier, stegos = base, []
        for index, (stage, secret, strength) in enumerate(zip(self.stages, secrets, self.cfg.stage_secret_lambdas)):
            if secret.shape != carrier.shape:
                raise ValueError("Secret and carrier dimensions differ")
            size = self.sizes[index + 1]
            carrier = stage(torch.cat([carrier, secret], dim=1), self.scale, size, size,
                            secret_lambda=float(strength) * ratio)
            stegos.append(carrier)
        return stegos

    @torch.inference_mode()
    def decode(self, carrier):
        shape = carrier.shape[-2:]
        if shape[0] != shape[1] or shape[0] not in self.sizes[1:]:
            raise ValueError("Carrier dimensions differ from the checkpoint geometry")
        return self.reveals[self.sizes[1:].index(shape[0])](carrier.contiguous(), self.scale)


def load_checkpoint(path, device="cpu"):
    payload = torch.load(path, map_location="cpu")
    if payload.get("format") == FORMAT and payload.get("architecture_version") == PretrainedNetwork.architecture_version:
        model = PretrainedNetwork(payload).eval().requires_grad_(False).to(device)
        return model, model.cfg, payload
    return restore_model(payload, device)
