"""
utils/metrics.py
────────────────
Evaluation metrics for face frontalization:

  - SSIM   : Structural Similarity Index (higher = better, max 1.0)
  - PSNR   : Peak Signal-to-Noise Ratio   (higher = better, dB)
  - LPIPS  : Learned Perceptual Image Patch Similarity (lower = better)
  - ID Score : Cosine similarity of FaceNet embeddings (higher = better)

All functions accept torch tensors in [-1, 1] range, shape (B, 3, H, W).
"""

import torch
import torch.nn.functional as F
import numpy as np
from skimage.metrics import structural_similarity as sk_ssim
from skimage.metrics import peak_signal_noise_ratio as sk_psnr


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _to_numpy_uint8(tensor: torch.Tensor) -> np.ndarray:
    """(B,3,H,W) in [-1,1] → (B,H,W,3) uint8 [0,255]."""
    t = tensor.detach().cpu().clamp(-1, 1)
    t = ((t + 1.0) / 2.0 * 255.0).to(torch.uint8)
    return t.permute(0, 2, 3, 1).numpy()


def _to_float01(tensor: torch.Tensor) -> np.ndarray:
    """(B,3,H,W) in [-1,1] → (B,H,W,3) float32 [0,1]."""
    t = tensor.detach().cpu().clamp(-1, 1)
    return ((t + 1.0) / 2.0).permute(0, 2, 3, 1).numpy().astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# SSIM
# ─────────────────────────────────────────────────────────────────────────────

def compute_ssim(generated: torch.Tensor, target: torch.Tensor) -> float:
    """
    Average SSIM across the batch.
    Both tensors: (B, 3, H, W), values in [-1, 1].
    """
    gen_np  = _to_float01(generated)
    tgt_np  = _to_float01(target)
    scores  = []
    for g, t in zip(gen_np, tgt_np):
        score = sk_ssim(g, t, data_range=1.0, channel_axis=2)
        scores.append(score)
    return float(np.mean(scores))


# ─────────────────────────────────────────────────────────────────────────────
# PSNR
# ─────────────────────────────────────────────────────────────────────────────

def compute_psnr(generated: torch.Tensor, target: torch.Tensor) -> float:
    """
    Average PSNR across the batch (dB).
    Both tensors: (B, 3, H, W), values in [-1, 1].
    """
    gen_np  = _to_float01(generated)
    tgt_np  = _to_float01(target)
    scores  = []
    for g, t in zip(gen_np, tgt_np):
        score = sk_psnr(t, g, data_range=1.0)
        scores.append(score)
    return float(np.mean(scores))


# ─────────────────────────────────────────────────────────────────────────────
# LPIPS
# ─────────────────────────────────────────────────────────────────────────────

class LPIPSMetric:
    """
    Wrapper around the `lpips` library.
    Loaded lazily to avoid startup overhead if not used.
    """
    def __init__(self, net: str = 'alex'):
        import lpips
        self._fn = lpips.LPIPS(net=net)
        self._fn.eval()
        print(f"[Metrics] LPIPS loaded (backbone: {net}).")

    @torch.no_grad()
    def __call__(self, generated: torch.Tensor,
                 target: torch.Tensor) -> float:
        """Both inputs: (B,3,H,W) in [-1,1]."""
        device = generated.device
        self._fn = self._fn.to(device)
        scores = self._fn(generated, target)          # (B,1,1,1)
        return float(scores.mean().item())


# ─────────────────────────────────────────────────────────────────────────────
# Identity / ID Score
# ─────────────────────────────────────────────────────────────────────────────

class IDScoreMetric:
    """
    Cosine similarity between FaceNet embeddings.
    Higher = more identity preserved.
    """
    def __init__(self):
        try:
            from facenet_pytorch import InceptionResnetV1
            self.model = InceptionResnetV1(pretrained='vggface2').eval()
            self.available = True
        except Exception as e:
            print(f"[Metrics] IDScore unavailable: {e}")
            self.available = False

    @torch.no_grad()
    def __call__(self, generated: torch.Tensor,
                 target: torch.Tensor) -> float:
        if not self.available:
            return float('nan')

        device = generated.device
        self.model = self.model.to(device)

        gen_r = F.interpolate(generated, size=(160, 160),
                              mode='bilinear', align_corners=False)
        tgt_r = F.interpolate(target,    size=(160, 160),
                              mode='bilinear', align_corners=False)

        gen_emb = self.model(gen_r)
        tgt_emb = self.model(tgt_r)

        cos = F.cosine_similarity(gen_emb, tgt_emb, dim=1)
        return float(cos.mean().item())


# ─────────────────────────────────────────────────────────────────────────────
# Metric Aggregator
# ─────────────────────────────────────────────────────────────────────────────

class MetricTracker:
    """
    Accumulates metric values across batches and returns epoch averages.

    Usage
    -----
    tracker = MetricTracker()
    for batch in loader:
        tracker.update(ssim=..., psnr=..., lpips=..., id_score=...)
    print(tracker.summary())
    tracker.reset()
    """

    def __init__(self):
        self._sums   = {}
        self._counts = {}

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if v != v:   # skip NaN
                continue
            self._sums[k]   = self._sums.get(k, 0.0) + float(v)
            self._counts[k] = self._counts.get(k, 0)  + 1

    def averages(self) -> dict:
        return {k: self._sums[k] / self._counts[k]
                for k in self._sums if self._counts[k] > 0}

    def summary(self) -> str:
        avgs = self.averages()
        parts = [f"{k}: {v:.4f}" for k, v in sorted(avgs.items())]
        return " | ".join(parts)

    def reset(self):
        self._sums.clear()
        self._counts.clear()
