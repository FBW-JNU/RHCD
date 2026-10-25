import torch
import torch.nn as nn
import torch.nn.functional as F

from dataset.torch_bicubic import imresize


class CrossLevelResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(channels, channels, 3, 1, 1),
        )

    def forward(self, x):
        return x + self.body(x)


class HighResolutionCrossLevelDownsampler(nn.Module):
    """Full-resolution feature encoding for direct and stepwise RD paths.

    Three residual encoder/head pairs refine Y3->Y2, direct Y3->Y1 and
    progressive Y3->Y2->Y1 representations.
    """

    def __init__(self, hidden_channels=32, num_blocks=4, residual_cap=0.01,
                 chain_residual_cap=0.03, chain_input_clamp=True):
        super().__init__()
        self.residual_cap = float(residual_cap)
        self.chain_residual_cap = float(chain_residual_cap)
        self.chain_input_clamp = bool(chain_input_clamp)
        self.y3_to_y2_encoder = self._make_encoder(hidden_channels, num_blocks)
        self.y3_to_y1_encoder = self._make_encoder(hidden_channels, num_blocks)
        self.y2_to_y1_encoder = self._make_encoder(hidden_channels, num_blocks)
        self.y3_to_y2_head = self._make_head(hidden_channels)
        self.y3_to_y1_head = self._make_head(hidden_channels)
        self.y2_to_y1_head = self._make_head(hidden_channels)
        self._initialize_residual_heads()

    @staticmethod
    def _make_encoder(hidden_channels, num_blocks):
        layers = [
            nn.Conv2d(3, hidden_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
        ]
        for _ in range(int(num_blocks)):
            layers.append(CrossLevelResidualBlock(hidden_channels))
        return nn.Sequential(*layers)

    @staticmethod
    def _make_head(channels):
        return nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(channels, 3, 3, 1, 1),
        )

    def _initialize_residual_heads(self):
        for head in (
                self.y3_to_y2_head, self.y3_to_y1_head, self.y2_to_y1_head):
            nn.init.zeros_(head[-1].weight)
            nn.init.zeros_(head[-1].bias)

    @staticmethod
    def _predict(source, base, encoder, head, residual_cap):
        features = encoder(source)
        features = F.interpolate(
            features,
            size=base.shape[-2:],
            mode="bicubic",
            align_corners=False,
            antialias=True,
        )
        residual = residual_cap * torch.tanh(head(features))
        # Preserve floating-point bicubic values in the residual output.
        prediction = base + residual
        return prediction, residual

    def forward(self, stego_3, scale=1.5, source_level=3):
        if source_level == 2:
            source = stego_3.clamp(0.0, 1.0).contiguous()
            base = imresize(source, scale=1.0 / float(scale))
            prediction, residual = self._predict(source, base, self.y2_to_y1_encoder,
                                                  self.y2_to_y1_head, self.chain_residual_cap)
            return {"pred_1": prediction, "base_1": base, "residual_1": residual}
        if source_level != 3:
            raise ValueError("RD source_level must be 2 or 3")
        stego_3 = torch.clamp(stego_3, 0.0, 1.0).contiguous()
        base_2 = imresize(stego_3, scale=1.0 / float(scale))
        base_1 = imresize(stego_3, scale=1.0 / (float(scale) ** 2))

        pred_2, residual_2 = self._predict(
            stego_3, base_2, self.y3_to_y2_encoder, self.y3_to_y2_head,
            self.residual_cap)
        pred_1, residual_1 = self._predict(
            stego_3, base_1, self.y3_to_y1_encoder, self.y3_to_y1_head,
            self.residual_cap)

        # Detach the first prediction so the progressive objective updates
        # its dedicated Y2->Y1 encoder/head independently of the direct path.
        chain_source_2 = pred_2.detach().contiguous()
        if self.chain_input_clamp:
            chain_source_2 = chain_source_2.clamp(0.0, 1.0)
        chain_base_1 = imresize(chain_source_2, scale=1.0 / float(scale))
        pred_chain_1, residual_chain_1 = self._predict(
            chain_source_2, chain_base_1, self.y2_to_y1_encoder,
            self.y2_to_y1_head, self.chain_residual_cap)

        return {
            "pred_2": pred_2,
            "pred_1": pred_1,
            "pred_chain_1": pred_chain_1,
            "base_2": base_2,
            "base_1": base_1,
            "chain_base_1": chain_base_1,
            "residual_2": residual_2,
            "residual_1": residual_1,
            "residual_chain_1": residual_chain_1,
        }
