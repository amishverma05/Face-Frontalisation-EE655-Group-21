"""
models/generator.py
────────────────────
Pose-Conditioned FPN Encoder for face frontalization.

Architecture
------------
  Input: profile image (3×256×256) + pose angles (yaw, pitch)
       ↓
  ResNet-IR50 backbone → Feature maps at 4 FPN scales
       ↓
  Pose sincos embedding injected at each FPN scale (AdaIN-style)
       ↓
  Adapter blocks: Conv3×3 stride2 → ReLU → ConvTranspose → linear → 512-dim
       ↓
  W+ latent code [B, 18, 512]
       ↓
  Passed to frozen StyleGAN2 generator (in stylegan2_wrapper.py)

Note: The old U-Net Generator is preserved as `LegacyGenerator` for
reference but is NOT used in the new training pipeline.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

class SinCosEmbedding(nn.Module):
    """
    Sinusoidal + cosine embedding of a scalar angle.
    angle_deg → [sin(angle_rad), cos(angle_rad), sin(2*angle_rad), cos(2*angle_rad), ...]
    Returns a vector of length `dim`.
    """
    def __init__(self, dim: int = 64):
        super().__init__()
        assert dim % 2 == 0, "dim must be even"
        self.dim = dim
        # Frequency bands: 1, 2, 4, 8, ...
        # Frequency bands. Instead of 2^31 (which overflows FP16 to Inf),
        # we spread frequencies linearly or use bounded powers.
        # Max reasonable freq for face angles is ~ 2^8.
        # We spread k bands between 0 and 8.
        k = dim // 2
        freqs = 2.0 ** torch.linspace(0.0, 8.0, k)
        self.register_buffer('freqs', freqs)

    def forward(self, angle_deg: torch.Tensor) -> torch.Tensor:
        """
        angle_deg : (B,) in degrees
        returns   : (B, dim)
        """
        angle_rad = angle_deg * (math.pi / 180.0)          # (B,)
        angle_rad = angle_rad.unsqueeze(1)                   # (B, 1)
        phases = angle_rad * self.freqs.unsqueeze(0)         # (B, k)
        return torch.cat([phases.sin(), phases.cos()], dim=1)  # (B, dim)


class PoseEmbedding(nn.Module):
    """
    Encodes (yaw, pitch) → a single pose vector of `out_dim`.
    Uses sincos embedding for each angle then fuses them with an MLP.
    """
    def __init__(self, angle_dim: int = 64, out_dim: int = 256):
        super().__init__()
        self.sin_cos = SinCosEmbedding(angle_dim)
        self.mlp = nn.Sequential(
            nn.Linear(angle_dim * 2, out_dim),  # yaw + pitch each angle_dim
            nn.SiLU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, yaw: torch.Tensor, pitch: torch.Tensor) -> torch.Tensor:
        """yaw, pitch: (B,) in degrees → (B, out_dim)"""
        emb_yaw   = self.sin_cos(yaw)
        emb_pitch = self.sin_cos(pitch)
        return self.mlp(torch.cat([emb_yaw, emb_pitch], dim=1))


class AdaIN(nn.Module):
    """
    Adaptive Instance Normalization applied with a pose-derived affine transform.
    scale, bias come from a linear projection of the pose embedding vector.
    """
    def __init__(self, channels: int, pose_dim: int):
        super().__init__()
        self.norm   = nn.InstanceNorm2d(channels, affine=False)
        self.linear = nn.Linear(pose_dim, channels * 2)  # → scale, bias

    def forward(self, x: torch.Tensor, pose: torch.Tensor) -> torch.Tensor:
        """
        x    : (B, C, H, W)
        pose : (B, pose_dim)
        """
        params = self.linear(pose)                          # (B, 2C)
        scale, bias = params.chunk(2, dim=1)               # each (B, C)
        scale = scale.unsqueeze(-1).unsqueeze(-1) + 1.0    # init ~1
        bias  = bias.unsqueeze(-1).unsqueeze(-1)
        return scale * self.norm(x) + bias


class FPNAdapterBlock(nn.Module):
    """
    Adapter block that converts a FPN feature map → one style vector (512-dim).

    Pipeline per FPN level:
      Conv3×3 stride2 → ReLU → ConvTranspose2d (restore) → AdaIN(pose) → GlobalAvgPool → Linear → 512
    """
    def __init__(self, in_ch: int, pose_dim: int, style_dim: int = 512):
        super().__init__()
        mid_ch = min(in_ch, 256)
        self.down = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, 3, stride=2, padding=1, bias=False),
            nn.ReLU(inplace=True),
        )
        self.up = nn.ConvTranspose2d(mid_ch, mid_ch, 4, stride=2, padding=1, bias=False)
        self.adain = AdaIN(mid_ch, pose_dim)
        
        self.proj = nn.Linear(mid_ch, style_dim)
        # Initialize projection to very small weights to start close to w_avg
        # while preventing FP16 gradient underflow (0.0 causes dead FPN)
        nn.init.normal_(self.proj.weight, std=0.01)
        if self.proj.bias is not None:
            nn.init.zeros_(self.proj.bias)

    def forward(self, feat: torch.Tensor, pose: torch.Tensor) -> torch.Tensor:
        """Returns style vector (B, style_dim)."""
        x = self.down(feat)             # (B, mid_ch, H/2, W/2)
        x = self.up(x)                  # (B, mid_ch, H, W)   approx
        x = self.adain(x, pose)        # pose-injected
        x = x.mean(dim=[2, 3])         # (B, mid_ch)  global avg pool
        return self.proj(x)             # (B, 512)


# ─────────────────────────────────────────────────────────────────────────────
# Main Encoder
# ─────────────────────────────────────────────────────────────────────────────

class PoseConditionedEncoder(nn.Module):
    """
    Pose-Conditioned FPN Encoder.

    Uses a ResNet-50 backbone (ImageNet pretrained) as the FPN feature
    extractor.  At each of 4 FPN scales the pose embedding is injected
    via AdaIN inside an adapter block to produce a 512-dim style vector.
    All 18 style vectors are stacked to form the W+ latent code.

    Parameters
    ----------
    pose_dim   : dimension of the pose embedding MLP output  (default 256)
    style_dim  : StyleGAN2 W latent dimension                (default 512)
    n_styles   : number of W+ style vectors  (18 for 256px StyleGAN2)
    pretrained : use ImageNet-pretrained ResNet-50 backbone   (default True)
    """

    def __init__(
        self,
        pose_dim:   int  = 256,
        style_dim:  int  = 512,
        n_styles:   int  = 15,   # 256px StyleGAN2 uses 15 style inputs; w_avg is [1,15,512]
        pretrained: bool = True,
    ):
        super().__init__()
        self.n_styles  = n_styles
        self.style_dim = style_dim

        # ── Backbone (ResNet-50, ImageNet weights) ──────────────────────────
        weights = ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = resnet50(weights=weights)

        self.stem   = nn.Sequential(backbone.conv1, backbone.bn1,
                                    backbone.relu, backbone.maxpool)  # /4
        self.layer1 = backbone.layer1   # /4   64ch → 256ch
        self.layer2 = backbone.layer2   # /8   256ch → 512ch
        self.layer3 = backbone.layer3   # /16  512ch → 1024ch
        self.layer4 = backbone.layer4   # /32  1024ch → 2048ch

        # ── FPN lateral convs (reduce channel depth) ────────────────────────
        self.lat1 = nn.Conv2d(256,  256, 1, bias=False)
        self.lat2 = nn.Conv2d(512,  256, 1, bias=False)
        self.lat3 = nn.Conv2d(1024, 256, 1, bias=False)
        self.lat4 = nn.Conv2d(2048, 256, 1, bias=False)

        # ── Pose embedding ───────────────────────────────────────────────────
        self.pose_emb = PoseEmbedding(angle_dim=64, out_dim=pose_dim)

        # ── Adapter blocks (4 FPN levels × styles_per_level vectors) ────────
        # Distribute n_styles across 4 levels:  [5, 5, 4, 4] = 18
        self.styles_per_level = self._distribute_styles(n_styles, 4)
        self.adapters = nn.ModuleList()
        for n_s in self.styles_per_level:
            # Each level produces n_s style vectors
            level_adapters = nn.ModuleList([
                FPNAdapterBlock(256, pose_dim, style_dim) for _ in range(n_s)
            ])
            self.adapters.append(level_adapters)

        # ── Learnable W+ offset initialized to zero ─────────────────────────
        # (will be added to the frontal latent average loaded at inference)
        self.w_offset = nn.Parameter(torch.zeros(1, n_styles, style_dim))

        self._init_new_layers()

    @staticmethod
    def _distribute_styles(total: int, n_levels: int):
        """Distribute `total` styles evenly across `n_levels` levels."""
        base, rem = divmod(total, n_levels)
        return [base + (1 if i < rem else 0) for i in range(n_levels)]

    def _init_new_layers(self):
        """Initialize only the newly added layers (not the pretrained backbone)."""
        for m in [self.lat1, self.lat2, self.lat3, self.lat4]:
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
        for adapters in self.adapters:
            for adapter in adapters:
                for mm in adapter.modules():
                    if isinstance(mm, (nn.Conv2d, nn.ConvTranspose2d)):
                        nn.init.kaiming_normal_(mm.weight, mode='fan_out')
                    # NOTE: Skip nn.Linear here — FPNAdapterBlock.proj is already
                    # initialized to zeros in its constructor to anchor at w_avg.
                    # Xavier-reinitializing it here would break that guarantee.

        # Freeze early backbone layers to massively speed up training
        # ImageNet stem, layer1, and layer2 already extract basic edges/textures perfectly
        for block in [self.stem, self.layer1, self.layer2]:
            for param in block.parameters():
                param.requires_grad = False

    def train(self, mode: bool = True):
        # Override standard train mode to ensure frozen layers stay in eval mode!
        # If BatchNorm2d layers drop to train mode, their running stats will 
        # get completely destroyed by our tiny batch_size (4), ruining the backbone.
        super().train(mode)
        if mode:
            for block in [self.stem, self.layer1, self.layer2]:
                block.eval()
        return self

    def forward(
        self,
        x:     torch.Tensor,           # (B, 3, 256, 256)  profile image
        yaw:   torch.Tensor,           # (B,)  yaw in degrees
        pitch: torch.Tensor,           # (B,)  pitch in degrees
        w_avg: torch.Tensor = None,    # (1, 18, 512) or None
    ) -> torch.Tensor:
        """
        Returns W+ latent code of shape (B, 18, 512).
        If w_avg is provided, delta is added to w_avg (identity initialization).
        """
        # ── Pose embedding ───────────────────────────────────────────────────
        pose = self.pose_emb(yaw, pitch)   # (B, pose_dim)

        # ── Backbone forward ─────────────────────────────────────────────────
        # Normalize input from [-1,1] to ImageNet stats
        mean = torch.tensor([0.485, 0.456, 0.406],
                            device=x.device).view(1, 3, 1, 1)
        std  = torch.tensor([0.229, 0.224, 0.225],
                            device=x.device).view(1, 3, 1, 1)
        x_norm = ((x + 1.0) / 2.0 - mean) / std

        c0 = self.stem(x_norm)     # /4
        c1 = self.layer1(c0)       # /4,  256ch
        c2 = self.layer2(c1)       # /8,  512ch
        c3 = self.layer3(c2)       # /16, 1024ch
        c4 = self.layer4(c3)       # /32, 2048ch

        # ── FPN top-down pathway ─────────────────────────────────────────────
        p4 = self.lat4(c4)
        p3 = self.lat3(c3) + F.interpolate(p4, size=c3.shape[2:], mode='nearest')
        p2 = self.lat2(c2) + F.interpolate(p3, size=c2.shape[2:], mode='nearest')
        p1 = self.lat1(c1) + F.interpolate(p2, size=c1.shape[2:], mode='nearest')

        # ── Adapter blocks → style vectors ───────────────────────────────────
        fpn_feats = [p4, p3, p2, p1]
        style_vecs = []
        for level_feats, level_adapters in zip(fpn_feats, self.adapters):
            for adapter in level_adapters:
                style_vecs.append(adapter(level_feats, pose))   # (B, 512)

        # Stack → (B, 18, 512)
        w_delta = torch.stack(style_vecs, dim=1) + self.w_offset  # (B, 18, 512)

        # ── Add frontal latent average if provided ───────────────────────────
        if w_avg is not None:
            return w_avg.to(x.device) + w_delta
        return w_delta

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ─────────────────────────────────────────────────────────────────────────────
# Legacy U-Net Generator  (kept for reference / evaluation of old checkpoints)
# ─────────────────────────────────────────────────────────────────────────────

class ResBlock(nn.Module):
    """Residual block with InstanceNorm."""
    def __init__(self, channels: int, dropout: float = 0.0):
        super().__init__()
        layers = [
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, 3, 1, 0, bias=False),
            nn.InstanceNorm2d(channels, affine=True),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers += [
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, 3, 1, 0, bias=False),
            nn.InstanceNorm2d(channels, affine=True),
        ]
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.block(x)


class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(out_ch, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=False):
        super().__init__()
        layers = [
            nn.ConvTranspose2d(in_ch, out_ch, 4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(out_ch, affine=True),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout(0.5))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class Generator(nn.Module):
    """
    Legacy U-Net generator (kept for backward compatibility with old checkpoints).
    The new architecture uses PoseConditionedEncoder + StyleGAN2Generator.
    """

    def __init__(self, in_ch=3, out_ch=3, ngf=64, n_res=6):
        super().__init__()
        self.e1 = nn.Sequential(
            nn.Conv2d(in_ch, ngf, 4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.e2 = DownBlock(ngf,     ngf * 2)
        self.e3 = DownBlock(ngf * 2, ngf * 4)
        self.e4 = DownBlock(ngf * 4, ngf * 8)
        self.res = nn.Sequential(*[ResBlock(ngf * 8) for _ in range(n_res)])
        self.d4 = UpBlock(ngf * 8,     ngf * 4, dropout=True)
        self.d3 = UpBlock(ngf * 4 * 2, ngf * 2, dropout=True)
        self.d2 = UpBlock(ngf * 2 * 2, ngf,     dropout=False)
        self.d1 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 2, out_ch, 4, stride=2, padding=1),
            nn.Tanh(),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.normal_(m.weight, 0.0, 0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.InstanceNorm2d) and m.affine:
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(e1)
        e3 = self.e3(e2)
        e4 = self.e4(e3)
        b  = self.res(e4)
        d4 = self.d4(b)
        d3 = self.d3(torch.cat([d4, e3], dim=1))
        d2 = self.d2(torch.cat([d3, e2], dim=1))
        return self.d1(torch.cat([d2, e1], dim=1))

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
