import torch

from base.base_model import BaseModel
import torch.nn as nn
import torch.nn.functional as F
from models.common import MeanShift, default_conv, ResBlock, Upsampler, Downsampler
from models.arb import (
    ControlledResidualSCABUpsample,
    SA_adapt,
    SCAB_upsample,
    SCAB_downsample,
    SourceAwareSAAdapt,
)


def _zero_resblock_tail(block):
    for module in reversed(block.body):
        if isinstance(module, nn.Conv2d):
            nn.init.zeros_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
            break


class MultiSourceStem(nn.Module):
    def __init__(self, source_names, n_feats, kernel_size, res_blocks=2, res_scale=1):
        super(MultiSourceStem, self).__init__()
        self.source_names = tuple(source_names)
        self.convs = nn.ModuleDict({
            name: default_conv(3, n_feats, kernel_size, bias=False)
            for name in self.source_names
        })
        self.refine = nn.ModuleDict()
        for name in self.source_names:
            blocks = [
                ResBlock(default_conv, n_feats, kernel_size, res_scale=res_scale)
                for _ in range(res_blocks)
            ]
            for block in blocks:
                _zero_resblock_tail(block)
            self.refine[name] = nn.Sequential(*blocks)
        self.bias = nn.Parameter(torch.zeros(1, n_feats, 1, 1))

    def forward(self, sources):
        features = {}
        fused = self.bias
        for name in self.source_names:
            feature = self.convs[name](sources[name])
            feature = self.refine[name](feature)
            features[name] = feature
            fused = fused + feature
        return fused, features


class EDRS(BaseModel):
    def __init__(self, args=None, ifsec=None):
        super(EDRS, self).__init__()
        n_resblocks = args.n_resblocks
        self.n_resblocks = n_resblocks
        self.ifsec = ifsec
        n_feats = args.n_feats
        kernel_size = 3
        act = nn.ReLU(True)
        conv = default_conv
        rescale = args.rescale
        self.rescale = rescale
        self.fixed_scale = args.fixed_scale
        self.fusion_type = getattr(args, "fusion_type", "early")
        self.scab_spatial_gating = bool(getattr(args, "scab_spatial_gating", True))
        self.expert_definition = getattr(args, "expert_definition", "legacy")
        self.use_multi_source = self.fusion_type in (
            "single_source", "multi_source", "source_controlled", "source_moe", "source_moe_stage2_only")
        if self.fusion_type == "single_source" and ifsec is not None:
            raise ValueError("single_source is the image-only downsampling backbone")
        self.use_controlled_scab = self.fusion_type in (
            "source_controlled", "source_moe", "source_moe_stage2_only")
        self.use_source_moe = (
            self.fusion_type == "source_moe" or
            (self.fusion_type == "source_moe_stage2_only" and ifsec == 2)
        )
        self.default_secret_lambda = float(getattr(args, "secret_residual_weight", 0.05))

        self.sub_mean = MeanShift(args.rgb_range)
        self.add_mean = MeanShift(args.rgb_range, sign=1)

        # define head module
        if ifsec == 1:
            m_head = [conv(args.n_colors + 3, n_feats, kernel_size)]
            self.source_names = ("base", "secret")
        elif ifsec == 2:
            m_head = [conv(args.n_colors + 6, n_feats, kernel_size)]
            self.source_names = ("base", "secret", "control")
        else:
            m_head = [conv(args.n_colors, n_feats, kernel_size)]
            self.source_names = ("base",)

        # define body module
        m_body = [
            ResBlock(
                conv, n_feats, kernel_size, act=act, res_scale=args.res_scale
            ) for _ in range(n_resblocks)
        ]
        m_body.append(conv(n_feats, n_feats, kernel_size))

        # define tail module
        if not args.fixed_scale:
            m_tail = [
                None,
                conv(n_feats, args.n_colors, kernel_size)
            ] if rescale == 'up' else [
                None,
                conv(n_feats, args.n_colors, kernel_size)
            ]
        else:
            m_tail = [
                Upsampler(conv, args.scale, n_feats, act=False),
                conv(n_feats, args.n_colors, kernel_size)
            ] if rescale == 'up' else [
                Downsampler(conv, args.scale, n_feats, act=False),
                conv(n_feats, args.n_colors, kernel_size)
            ]
        self.head = nn.Sequential(*m_head)
        if self.use_multi_source:
            self.source_stem = MultiSourceStem(
                self.source_names,
                n_feats,
                kernel_size,
                res_blocks=int(getattr(args, "source_stem_blocks", 2)),
                res_scale=args.res_scale,
            )
        else:
            self.source_stem = None
        if self.use_controlled_scab and rescale == "up" and ifsec in (1, 2):
            self.secret_feature = conv(n_feats, n_feats, kernel_size)
            nn.init.zeros_(self.secret_feature.weight)
            if self.secret_feature.bias is not None:
                nn.init.zeros_(self.secret_feature.bias)
            self.control_gate = nn.Sequential(
                conv(n_feats * len(self.source_names), n_feats, kernel_size),
                nn.ReLU(True),
                conv(n_feats, 2, kernel_size),
            )
            embed_bias = float(getattr(args, "control_embed_bias", -2.0))
            nn.init.zeros_(self.control_gate[-1].weight)
            nn.init.constant_(self.control_gate[-1].bias[:1], 0.0)
            nn.init.constant_(self.control_gate[-1].bias[1:], embed_bias)
        else:
            self.secret_feature = None
            self.control_gate = None
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)

        if not args.fixed_scale:
            # For EDSR backbone, we set K=4
            self.K = args.K
            sa_adapt = []
            source_moe_router_init_bias = float(getattr(args, "source_moe_router_init_bias", 2.0))
            for i in range(self.n_resblocks // self.K):
                if self.use_source_moe:
                    sa_adapt.append(SourceAwareSAAdapt(
                        channels=args.n_feats,
                        num_experts=args.num_experts_SAconv,
                        router_init_bias=source_moe_router_init_bias,
                        texture_only=bool(getattr(args, "ablate_moe_texture_only", False)),
                        expert_definition=getattr(args, "expert_definition", "legacy"),
                        gating_mode=(getattr(args, "source_moe_gating", "spatial")
                                     if ifsec in (1, 2) else "spatial"),
                    ))
                else:
                    sa_adapt.append(SA_adapt(channels=args.n_feats, num_experts=args.num_experts_SAconv))
            self.sa_adapt = nn.Sequential(*sa_adapt)

            # Scale-aware Downsample and Upsample
            if rescale == 'up':
                if args.up_sampler == 'sampleB':
                    self.sa_sample = SCAB_upsample(channels=args.n_feats, num_experts=args.num_experts_CRM)
                    if self.use_controlled_scab and ifsec in (1, 2):
                        self.controlled_sample = ControlledResidualSCABUpsample(
                            channels=args.n_feats,
                            num_experts=args.num_experts_CRM,
                            detail_scale=float(getattr(args, "detail_residual_scale", 0.1)),
                            secret_scale=float(getattr(args, "secret_residual_scale", 1.0)),
                        )

            else:
                if args.down_sampler == 'sampleB':
                    self.sa_sample = SCAB_downsample(channels=args.n_feats, num_experts=args.num_experts_CRM,
                                                    offset_limit=getattr(args, "down_offset_limit", None))

    def _preprocess_sources(self, x):
        if x.shape[1] == 6:
            sources = {
                "base": self.sub_mean(x[:, 0:3, ...]),
                "secret": self.sub_mean(x[:, 3:6, ...]),
            }
            legacy = torch.cat([sources["secret"], sources["base"]], dim=1)
        elif x.shape[1] == 9:
            sources = {
                "base": self.sub_mean(x[:, 0:3, ...]),
                "secret": self.sub_mean(x[:, 3:6, ...]),
                "control": x[:, 6:9, ...],
            }
            legacy = torch.cat([sources["secret"], sources["base"], sources["control"]], dim=1)
        else:
            sources = {"base": self.sub_mean(x)}
            legacy = sources["base"]
        return sources, legacy

    def _get_control_gates(self, sources, source_features):
        if "control" in sources and sources["control"].shape[1] >= 2:
            return torch.sigmoid(sources["control"][:, 0:1, ...]), torch.sigmoid(sources["control"][:, 1:2, ...])
        gate_input = torch.cat([source_features[name] for name in self.source_names], dim=1)
        gates = torch.sigmoid(self.control_gate(gate_input))
        return gates[:, 0:1, ...], gates[:, 1:2, ...]

    def forward(self, x, scale, outH=None, outW=None, return_aux=False, secret_lambda=None):
        aux = {}
        moe_weights = []
        sources, legacy_x = self._preprocess_sources(x)
        if self.source_stem is not None:
            x, source_features = self.source_stem(sources)
            aux["source_features"] = source_features
        else:
            source_features = None
            x = self.head(legacy_x)

        # body
        if not self.fixed_scale:
            res = x
            moe_stds = []
            moe_entropies = []
            moe_mins = []
            moe_maxs = []
            moe_low_ratios = []
            moe_layer_mean_loss = []
            for i in range(self.n_resblocks):
                res = self.body[i](res)
                # scale modulation
                if (i + 1) % self.K == 0:
                    adapt = self.sa_adapt[(i + 1) // self.K - 1]
                    if self.use_source_moe:
                        if return_aux:
                            res, weights = adapt(res, scale, source_features=source_features, return_weights=True)
                            moe_layer_mean_loss.append(weights.mean(dim=(0, 2, 3)))
                            weights_for_stats = weights.detach()
                            moe_weights.append(weights_for_stats.mean(dim=(0, 2, 3)))
                            moe_stds.append(weights_for_stats.std(dim=(0, 2, 3), unbiased=False))
                            moe_entropies.append(
                                -(weights_for_stats.clamp_min(1e-8) *
                                  weights_for_stats.clamp_min(1e-8).log()).sum(dim=1).mean()
                            )
                            moe_mins.append(weights_for_stats.amin(dim=(0, 2, 3)))
                            moe_maxs.append(weights_for_stats.amax(dim=(0, 2, 3)))
                            moe_low_ratios.append((weights_for_stats < 0.01).float().mean(dim=(0, 2, 3)))
                        else:
                            res = adapt(res, scale, source_features=source_features, return_weights=False)
                    else:
                        res = adapt(res, scale)

            res = self.body[-1](res)
            res += x

            if self.rescale == 'up' and self.expert_definition == "paper" and self.use_source_moe:
                res, final_weights = self.sa_adapt[-1](
                    res, scale, source_features=source_features, return_weights=True)
                if return_aux:
                    aux["final_moe_weights"] = final_weights

            if self.rescale == 'up':
                if self.use_controlled_scab and source_features is not None and hasattr(self, "controlled_sample"):
                    if bool(getattr(self, "scab_spatial_gating", True)):
                        m_new, m_embed = self._get_control_gates(sources, source_features)
                    else:
                        m_new = torch.ones_like(source_features["base"][:, :1])
                        m_embed = torch.ones_like(m_new)
                    secret_feature = source_features["secret"] if self.expert_definition == "paper" else self.secret_feature(source_features["secret"])
                    res, scab_aux = self.controlled_sample(
                        source_features["base"],
                        res,
                        secret_feature,
                        m_new,
                        m_embed,
                        scale,
                        outH,
                        outW,
                        secret_lambda=self.default_secret_lambda if secret_lambda is None else secret_lambda,
                        return_aux=True,
                    )
                    if return_aux:
                        aux.update(scab_aux)
                        aux["m_new_lr"] = m_new
                        aux["m_embed_lr"] = m_embed
                else:
                    res = self.sa_sample(res, scale, outH, outW)
            else:
                res = self.sa_sample(res, scale)
            x = self.tail[1](res)
        else:
            res = self.body(x)
            res += x

            x = self.tail(res)

        x = self.add_mean(x)
        if return_aux and moe_weights:
            aux["moe_layer_mean_loss"] = torch.stack(moe_layer_mean_loss)
            aux["moe_layer_mean"] = torch.stack(moe_weights)
            aux["moe_layer_std"] = torch.stack(moe_stds)
            aux["moe_layer_entropy"] = torch.stack(moe_entropies)
            aux["moe_layer_min"] = torch.stack(moe_mins)
            aux["moe_layer_max"] = torch.stack(moe_maxs)
            aux["moe_layer_low_ratio"] = torch.stack(moe_low_ratios)
            aux["moe_weights"] = aux["moe_layer_mean"].mean(dim=0)
            aux["has_control_source"] = "control" in self.source_names
        if return_aux:
            return x, aux
        return x
