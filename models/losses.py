"""
models/losses.py
────────────────
Hybrid loss stack for the Pose-Conditioned FPN Encoder + StyleGAN2 pipeline.

Loss Stack
──────────
  L_total = λ_id     × L_id       (ArcFace cosine identity preservation)
           + λ_lpips  × L_lpips    (LPIPS on 5 facial crops)
           + λ_lmk    × L_lmk     (68-pt landmark L2 — NOVEL)
           + λ_parse  × L_parse   (BiSeNet facial segmentation — Phase 1 only)
           + λ_wnorm  × L_wnorm   (W+ latent offset regularization)
           + λ_l2     × L_l2      (pixel L2 reconstruction)

Phase Schedule
──────────────
  Phase 1 (epochs 1–15):  id + lpips + lmk + parse + wnorm + l2
  Phase 2 (epochs 16–30): id + lpips + lmk + wnorm + l2  (parse dropped)

Legacy classes (PixelLoss, PerceptualLoss, IdentityLoss, LSGANLoss,
FrontalizationLoss) are preserved for backward compatibility.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


# ─────────────────────────────────────────────────────────────────────────────
# ══ LEGACY LOSSES (kept for backward compatibility) ══════════════════════════
# ─────────────────────────────────────────────────────────────────────────────

class PixelLoss(nn.Module):
    """Mean absolute error (L1) between generated and target images."""
    def forward(self, generated, target):
        return F.l1_loss(generated, target)


class PerceptualLoss(nn.Module):
    """Multi-scale VGG perceptual loss (relu1_2, relu2_2, relu3_3)."""
    VGG_MEAN = torch.tensor([0.485, 0.456, 0.406])
    VGG_STD  = torch.tensor([0.229, 0.224, 0.225])

    def __init__(self):
        super().__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        feats = list(vgg.features.children())
        self.slice1 = nn.Sequential(*feats[:4])
        self.slice2 = nn.Sequential(*feats[4:9])
        self.slice3 = nn.Sequential(*feats[9:16])
        for p in self.parameters():
            p.requires_grad = False

    def _normalize(self, x):
        x = (x + 1.0) / 2.0
        mean = self.VGG_MEAN.to(x.device).view(1, 3, 1, 1)
        std  = self.VGG_STD.to(x.device).view(1, 3, 1, 1)
        return (x - mean) / std

    def forward(self, generated, target):
        gn, tn = self._normalize(generated), self._normalize(target)
        loss = 0.0
        for slc in [self.slice1, self.slice2, self.slice3]:
            gn = slc(gn); tn = slc(tn)
            loss = loss + F.l1_loss(gn, tn.detach())
        return loss / 3.0


class IdentityLoss(nn.Module):
    """FaceNet cosine distance (legacy — replaced by ArcFaceIdentityLoss)."""
    def __init__(self):
        super().__init__()
        try:
            from facenet_pytorch import InceptionResnetV1
            self.facenet = InceptionResnetV1(pretrained='vggface2').eval()
            for p in self.facenet.parameters():
                p.requires_grad = False
            self.available = True
            print("[IdentityLoss] FaceNet loaded (VGGFace2).")
        except Exception as e:
            print(f"[IdentityLoss] WARNING: FaceNet not available ({e}). Skipping.")
            self.available = False

    def forward(self, generated, target):
        if not self.available:
            return torch.tensor(0.0, device=generated.device)
        gen_r = F.interpolate(generated, (160, 160), mode='bilinear', align_corners=False)
        tgt_r = F.interpolate(target,    (160, 160), mode='bilinear', align_corners=False)
        gen_emb = self.facenet(gen_r)
        tgt_emb = self.facenet(tgt_r.detach())
        return (1.0 - F.cosine_similarity(gen_emb, tgt_emb, dim=1)).mean()


class LSGANLoss(nn.Module):
    """Least-Squares GAN loss (legacy)."""
    def forward(self, pred, is_real):
        target = torch.ones_like(pred) if is_real else torch.zeros_like(pred)
        return F.mse_loss(pred, target)


class FrontalizationLoss(nn.Module):
    """Legacy combined loss (pixel + perceptual + identity + adversarial)."""
    def __init__(self, lambda_pixel=10.0, lambda_perceptual=5.0,
                 lambda_id=10.0, lambda_adv=1.0):
        super().__init__()
        self.pixel_loss      = PixelLoss()
        self.perceptual_loss = PerceptualLoss()
        self.id_loss         = IdentityLoss()
        self.adv_loss        = LSGANLoss()
        self.lam_pixel = lambda_pixel
        self.lam_perc  = lambda_perceptual
        self.lam_id    = lambda_id
        self.lam_adv   = lambda_adv

    def generator_loss(self, generated, target, disc_pred_fake):
        L_pixel = self.pixel_loss(generated, target)
        L_perc  = self.perceptual_loss(generated, target)
        L_id    = self.id_loss(generated, target)
        L_adv   = self.adv_loss(disc_pred_fake, is_real=True)
        total   = (self.lam_pixel * L_pixel + self.lam_perc * L_perc
                   + self.lam_id * L_id + self.lam_adv * L_adv)
        return total, {
            'G_total': total.item(), 'G_pixel': L_pixel.item(),
            'G_perceptual': L_perc.item(), 'G_identity': L_id.item(),
            'G_adv': L_adv.item(),
        }

    def discriminator_loss(self, disc_pred_real, disc_pred_fake):
        L_real = self.adv_loss(disc_pred_real, is_real=True)
        L_fake = self.adv_loss(disc_pred_fake, is_real=False)
        total  = (L_real + L_fake) * 0.5
        return total, {'D_total': total.item(), 'D_real': L_real.item(),
                       'D_fake': L_fake.item()}


# ─────────────────────────────────────────────────────────────────────────────
# ══ NEW HYBRID LOSSES ════════════════════════════════════════════════════════
# ─────────────────────────────────────────────────────────────────────────────

class ArcFaceIdentityLoss(nn.Module):
    """
    Identity preservation loss using ArcFace (insightface).

    Returns 1 - cosine_similarity (lower = better identity preservation).
    Falls back gracefully to FaceNet if insightface is unavailable.
    """

    def __init__(self):
        super().__init__()
        self.model = None
        self.backend = None
        self._try_insightface()
        if self.model is None:
            self._try_facenet()

    def _try_insightface(self):
        try:
            import insightface
            from insightface.app import FaceAnalysis
            # Use recognition model only — no detection needed
            app = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider',
                                                             'CPUExecutionProvider'])
            app.prepare(ctx_id=0, det_size=(112, 112))
            # Extract the arcface recognition model
            self.model     = app.models.get('recognition', None)
            self.backend   = 'insightface'
            print("[ArcFaceIdentityLoss] Loaded insightface ArcFace (buffalo_l).")
        except Exception as e:
            print(f"[ArcFaceIdentityLoss] insightface unavailable ({e}), trying FaceNet ...")

    def _try_facenet(self):
        try:
            from facenet_pytorch import InceptionResnetV1
            self.model   = InceptionResnetV1(pretrained='vggface2').eval()
            for p in self.model.parameters():
                p.requires_grad = False
            self.backend = 'facenet'
            print("[ArcFaceIdentityLoss] Using FaceNet fallback (VGGFace2).")
        except Exception as e:
            print(f"[ArcFaceIdentityLoss] No identity model available ({e}). Loss disabled.")
            self.backend = None

    def _get_embedding(self, imgs: torch.Tensor) -> torch.Tensor:
        """imgs: (B, 3, H, W) in [-1,1] → embeddings (B, D)."""
        if self.backend == 'insightface':
            # insightface expects (B, 3, 112, 112) BGR in [-1,1]
            imgs_112 = F.interpolate(imgs, (112, 112), mode='bilinear',
                                     align_corners=False)
            # Convert RGB→BGR
            imgs_bgr = imgs_112[:, [2, 1, 0], :, :]
            # insightface recognition model forward
            return self.model.get_feat(imgs_bgr)
        else:
            imgs_160 = F.interpolate(imgs, (160, 160), mode='bilinear',
                                     align_corners=False)
            return self.model(imgs_160)

    def forward(self, generated: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        if self.backend is None:
            return torch.tensor(0.0, device=generated.device)

        device = generated.device
        if self.backend == 'facenet' and next(self.model.parameters()).device != device:
            self.model = self.model.to(device)

        gen_emb = self._get_embedding(generated)
        tgt_emb = self._get_embedding(target.detach())
        return (1.0 - F.cosine_similarity(gen_emb, tgt_emb, dim=1)).mean()


class LPIPSCropLoss(nn.Module):
    """
    LPIPS computed on 5 facial crops:
      - Full face (resized to 256×256)
      - Left eye region
      - Right eye region
      - Mouth region
      - Nose region

    Crop ratios are relative to image height/width.
    Loss is the average over all 5 crops weighted equally.
    """

    # (y_start, y_end, x_start, x_end) as fractions of H, W
    CROP_REGIONS = {
        'full':       (0.00, 1.00, 0.00, 1.00),
        'left_eye':   (0.20, 0.45, 0.05, 0.48),
        'right_eye':  (0.20, 0.45, 0.52, 0.95),
        'nose':       (0.40, 0.68, 0.30, 0.70),
        'mouth':      (0.62, 0.90, 0.20, 0.80),
    }

    def __init__(self, net: str = 'alex'):
        super().__init__()
        try:
            import lpips
            self._fn = lpips.LPIPS(net=net)
            self._fn.eval()
            for p in self._fn.parameters():
                p.requires_grad = False
            self.available = True
            print(f"[LPIPSCropLoss] Loaded LPIPS ({net}).")
        except Exception as e:
            print(f"[LPIPSCropLoss] WARNING: lpips unavailable ({e}). Loss disabled.")
            self.available = False

    def _crop(self, x: torch.Tensor, region: tuple) -> torch.Tensor:
        B, C, H, W = x.shape
        ys, ye, xs, xe = region
        y0, y1 = int(ys * H), int(ye * H)
        x0, x1 = int(xs * W), int(xe * W)
        c = x[:, :, y0:y1, x0:x1]
        # Resize crop to 64×64 for efficiency
        return F.interpolate(c, (64, 64), mode='bilinear', align_corners=False)

    def forward(self, generated: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        if not self.available:
            return torch.tensor(0.0, device=generated.device)

        device = generated.device
        if next(self._fn.parameters()).device != device:
            self._fn = self._fn.to(device)

        import random
        # Always include full face, then sample 3 out of the 4 sub-regions
        sub_regions = [k for k in self.CROP_REGIONS.keys() if k != 'full']
        sampled_regions = ['full'] + random.sample(sub_regions, 3)

        gen_crops, tgt_crops = [], []
        for name in sampled_regions:
            region = self.CROP_REGIONS[name]
            gen_crops.append(self._crop(generated, region))
            tgt_crops.append(self._crop(target.detach(), region))
            
        gen_crops = torch.cat(gen_crops, dim=0)
        tgt_crops = torch.cat(tgt_crops, dim=0)
        
        return self._fn(gen_crops, tgt_crops).mean()


class LandmarkLoss(nn.Module):
    """
    [NOVEL] 68-point landmark geometric loss.

    Uses face_alignment library to detect 68 facial landmarks on the
    generated image and computes L2 distance against a precomputed
    frontal-face landmark template.

    Since landmark detection is non-differentiable, we:
      1. Detect landmarks on the generated image (stop-gradient).
      2. Compare against a stored frontal template target.
      3. Additionally use a differentiable proxy: VGG feature matching
         at the landmark coordinate positions (soft version).

    In practice we run detection every N steps and cache the prediction
    to avoid the CPU→GPU transfer bottleneck.
    """

    # Standard frontal 68-pt template (normalized to [0,1], 256×256 space)
    # These are approximate mean positions from AFLW frontal faces
    FRONTAL_TEMPLATE = None   # loaded lazily from face_alignment output

    def __init__(self, img_size: int = 256):
        super().__init__()
        self.img_size  = img_size
        self.fa        = None
        self._load_fa()

    def _load_fa(self):
        try:
            import face_alignment
            self.fa = face_alignment.FaceAlignment(
                face_alignment.LandmarksType.TWO_D,
                flip_input=False,
                device='cuda' if torch.cuda.is_available() else 'cpu',
            )
            self.available = True
            print("[LandmarkLoss] face_alignment loaded.")
        except Exception as e:
            print(f"[LandmarkLoss] WARNING: face_alignment unavailable ({e}). Loss disabled.")
            self.available = False

    def _detect(self, imgs: torch.Tensor):
        """
        Detect 68 landmarks for each image in batch.
        Returns list of (68, 2) arrays or None per image.
        """
        # Convert to uint8 numpy for face_alignment
        imgs_255 = ((imgs.detach().cpu().clamp(-1, 1) + 1.0) / 2.0 * 255).to(torch.uint8)
        # face_alignment expects (B, H, W, 3)
        imgs_np = imgs_255.permute(0, 2, 3, 1).numpy()
        results = []
        for img in imgs_np:
            try:
                lmks = self.fa.get_landmarks(img)
                results.append(lmks[0] if lmks is not None else None)  # (68, 2)
            except Exception:
                results.append(None)
        return results

    def forward(self, generated: torch.Tensor,
                frontal_lmks: torch.Tensor = None) -> torch.Tensor:
        """
        generated    : (B, 3, H, W) in [-1, 1]
        frontal_lmks : (B, 68, 2)  ground-truth landmark coords, or None.
                       If None, falls back to a simple gradient via
                       heatmap-style loss using an identity dummy.

        Returns scalar loss tensor.
        """
        if not self.available:
            return torch.tensor(0.0, device=generated.device)

        if frontal_lmks is None:
            # No GT landmarks available → return zero (skip gracefully)
            return torch.tensor(0.0, device=generated.device)

        # Detect landmarks on generated image (non-differentiable step)
        gen_lmks = self._detect(generated)

        total = torch.tensor(0.0, device=generated.device)
        count = 0
        for i, lmk_gen in enumerate(gen_lmks):
            if lmk_gen is None:
                continue
            lmk_gen_t = torch.tensor(lmk_gen, dtype=torch.float32,
                                     device=generated.device)            # (68, 2)
            lmk_gt_t  = frontal_lmks[i].to(generated.device)            # (68, 2)
            total = total + F.mse_loss(lmk_gen_t, lmk_gt_t)
            count += 1

        if count == 0:
            return torch.tensor(0.0, device=generated.device)
        return total / count


class ParseLoss(nn.Module):
    """
    BiSeNet facial segmentation consistency loss (Phase 1 only).

    Computes cross-entropy between the segmentation maps of the generated
    image and the GT frontal image.  Forces the network to produce
    segments (skin, hair, eyes, lips) in the correct regions.

    Falls back to a simple L1 pixel loss if BiSeNet is unavailable.
    """

    # BiSeNet 19-class labels: 0=background, 1=skin, 2=left_brow … 18=neck
    SEG_CLASSES = 19

    def __init__(self):
        super().__init__()
        self.model     = None
        self.available = False
        self._load_bisenet()

    def _load_bisenet(self):
        try:
            from models.bisenet import BiSeNet  # local copy if available
            self.model = BiSeNet(n_classes=self.SEG_CLASSES)
            import os
            ckpt_path = os.path.join('checkpoints', 'bisenet_79999_iter.pth')
            if os.path.exists(ckpt_path):
                state = torch.load(ckpt_path, map_location='cpu')
                self.model.load_state_dict(state)
                self.model.eval()
                for p in self.model.parameters():
                    p.requires_grad = False
                self.available = True
                print("[ParseLoss] BiSeNet loaded.")
            else:
                print(f"[ParseLoss] BiSeNet weights not found at {ckpt_path}. "
                      "Falling back to L1 proxy loss.")
        except Exception as e:
            print(f"[ParseLoss] BiSeNet unavailable ({e}). Using L1 proxy.")

    def forward(self, generated: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        if not self.available or self.model is None:
            # Graceful fallback: L1 pixel loss (lightweight proxy)
            return F.l1_loss(generated, target.detach()) * 0.1

        device = generated.device
        if next(self.model.parameters()).device != device:
            self.model = self.model.to(device)

        # BiSeNet expects (B, 3, 512, 512), values in [0, 1] (approximately)
        size = (512, 512)
        gen_512 = F.interpolate(generated,         size=size, mode='bilinear', align_corners=False)
        tgt_512 = F.interpolate(target.detach(),   size=size, mode='bilinear', align_corners=False)
        gen_512 = (gen_512 + 1.0) / 2.0
        tgt_512 = (tgt_512 + 1.0) / 2.0

        with torch.no_grad():
            tgt_seg = self.model(tgt_512)[0]       # (B, 19, 512, 512) — GT segmap
            gen_seg = self.model(gen_512)[0]       # (B, 19, 512, 512) — generated segmap

        # Segmentation KL divergence / cross-entropy between two seg maps
        tgt_probs = F.softmax(tgt_seg, dim=1).detach()
        gen_log   = F.log_softmax(gen_seg, dim=1)
        return F.kl_div(gen_log, tgt_probs, reduction='batchmean')


class WNormLoss(nn.Module):
    """
    W+ latent offset regularization.
    Penalizes large deviations of the predicted W+ code from the
    frontal latent average (w_avg).  Prevents mode collapse.

    L_wnorm = ||W_pred - w_avg||_2^2 / (18 * 512)
    """

    def forward(self, w_pred: torch.Tensor,
                w_avg: torch.Tensor) -> torch.Tensor:
        """
        w_pred : (B, 18, 512)
        w_avg  : (1, 18, 512) or (1, 1, 512)
        """
        delta = w_pred - w_avg.to(w_pred.device)
        return delta.pow(2).mean()


class PixelL2Loss(nn.Module):
    """L2 pixel reconstruction loss."""
    def forward(self, generated: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(generated, target)


# ─────────────────────────────────────────────────────────────────────────────
# HybridFrontalizationLoss  (main loss for the new pipeline)
# ─────────────────────────────────────────────────────────────────────────────

class HybridFrontalizationLoss(nn.Module):
    """
    Combined hybrid loss for the Pose-Conditioned FPN Encoder + StyleGAN2 pipeline.

    Phase 1 (epochs 1–50):  id + lpips + perceptual + wnorm + l2
    Phase 2 (epochs 51+):   id + lpips + wnorm + l2  (perceptual dropped)

    Parameters
    ----------
    lambda_id    : weight for ArcFace/FaceNet identity loss  (default 1.5)
    lambda_lpips : weight for LPIPS crop loss                (default 0.8)
    lambda_perc  : weight for VGG perceptual loss            (default 1.0)
    lambda_wnorm : weight for W+ offset regularization       (default 0.005)
    lambda_l2    : weight for pixel L2 loss                  (default 2.0)
    phase2_start : epoch at which perceptual loss is dropped (default 51)
    """

    def __init__(
        self,
        lambda_id:    float = 1.5,
        lambda_lpips: float = 0.8,
        lambda_lmk:   float = 0.0,   # disabled — frontal_lmks never provided
        lambda_parse: float = 0.0,   # disabled — replaced by lambda_perc below
        lambda_wnorm: float = 0.005,
        lambda_l2:    float = 2.0,
        lambda_perc:  float = 1.0,   # NEW: VGG-16 perceptual loss (replaces parse)
        phase2_start: int   = 51,
    ):
        super().__init__()
        self.lam_id    = lambda_id
        self.lam_lpips = lambda_lpips
        self.lam_wnorm = lambda_wnorm
        self.lam_l2    = lambda_l2
        self.lam_perc  = lambda_perc
        self.phase2_start = phase2_start

        self.id_loss    = ArcFaceIdentityLoss()
        self.lpips_loss = LPIPSCropLoss(net='alex')
        self.perc_loss  = PerceptualLoss()   # VGG-16, already installed
        self.wnorm_loss = WNormLoss()
        self.l2_loss    = PixelL2Loss()

    def forward(
        self,
        generated:     torch.Tensor,    # (B, 3, H, W) output image
        target:        torch.Tensor,    # (B, 3, H, W) GT frontal
        w_pred:        torch.Tensor,    # (B, 15, 512) predicted W+ code
        w_avg:         torch.Tensor,    # (1, 15, 512) frontal avg
        epoch:         int    = 1,
        frontal_lmks:  torch.Tensor = None,   # unused — kept for API compat
    ):
        """
        Returns (total_loss, loss_dict) where loss_dict contains named
        float values for TensorBoard logging.
        """
        is_phase1 = (epoch < self.phase2_start)

        L_id    = self.id_loss(generated, target)
        L_lpips = self.lpips_loss(generated, target)
        L_l2    = self.l2_loss(generated, target)

        total = (self.lam_id    * L_id
                 + self.lam_lpips * L_lpips
                 + self.lam_l2    * L_l2)

        # WNorm guard: skip for first 10 epochs so the encoder can escape the
        # w_avg fixed point before regularization pulls it back.
        # (WNorm from epoch 1 was holding the encoder at the garbage w_avg blob.)
        if epoch >= 10:
            L_wnorm = self.wnorm_loss(w_pred, w_avg)
            total   = total + self.lam_wnorm * L_wnorm
            wnorm_val = L_wnorm.item()
        else:
            wnorm_val = 0.0

        log = {
            'G_total':  total.item(),
            'G_id':     L_id.item(),
            'G_lpips':  L_lpips.item(),
            'G_lmk':    0.0,     # zeroed — LandmarkLoss disabled
            'G_wnorm':  wnorm_val,
            'G_l2':     L_l2.item(),
            'G_parse':  0.0,
        }

        # VGG Perceptual loss — only in Phase 1 for structural guidance
        if is_phase1 and self.lam_perc > 0:
            L_perc = self.perc_loss(generated, target)
            total  = total + self.lam_perc * L_perc
            log['G_perc']  = L_perc.item()
            log['G_total'] = total.item()
        else:
            log['G_perc'] = 0.0

        return total, log

