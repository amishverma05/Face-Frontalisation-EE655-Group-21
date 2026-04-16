"""
models/discriminator.py
───────────────────────
PatchGAN Discriminator for face frontalization.

A PatchGAN discriminator classifies overlapping 70×70 patches as
real or fake.  This gives better per-texture realism than a global
discriminator and is much lighter on VRAM.

Input  : cat(profile_img, frontal_img)  →  (B, 6, H, W)
Output : patch-level probability map    →  (B, 1, Ph, Pw)
"""

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=2, norm=True):
        super().__init__()
        layers = [nn.Conv2d(in_ch, out_ch, 4, stride=stride, padding=1, bias=not norm)]
        if norm:
            layers.append(nn.InstanceNorm2d(out_ch, affine=True))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class PatchDiscriminator(nn.Module):
    """
    70×70 PatchGAN discriminator.

    Parameters
    ----------
    in_ch : channels of ONE image (discriminator receives two images
            concatenated, so actual input channels = 2 * in_ch)
    ndf   : base number of discriminator filters
    """

    def __init__(self, in_ch: int = 3, ndf: int = 64):
        super().__init__()

        self.model = nn.Sequential(
            # No normalization on the first layer (standard practice)
            ConvBlock(in_ch * 2, ndf,     stride=2, norm=False),  # H/2
            ConvBlock(ndf,       ndf * 2, stride=2, norm=True),   # H/4
            ConvBlock(ndf * 2,   ndf * 4, stride=2, norm=True),   # H/8
            ConvBlock(ndf * 4,   ndf * 8, stride=1, norm=True),   # H/8  (stride 1)
            # Final layer → 1-channel patch map
            nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=1, padding=1),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0.0, 0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, profile: torch.Tensor,
                frontal: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        profile  : (B, 3, H, W)  input profile image
        frontal  : (B, 3, H, W)  real or generated frontal

        Returns
        -------
        (B, 1, Ph, Pw)  — patch logits (use LSGAN loss, no sigmoid needed)
        """
        x = torch.cat([profile, frontal], dim=1)
        return self.model(x)

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
