# EE655 Group Project — Technical Report
## Face Frontalization using Hybrid StyleGAN2 Encoder–Generator Architecture

---

## 1. Introduction

Face frontalization is the task of synthesizing a photorealistic, identity-preserving frontal view of a face from an arbitrary pose (profile, semi-profile, etc.). This project implements a **hybrid encoder–generator** approach: a trainable, pose-conditioned encoder predicts a W+ latent code that is decoded by a frozen, pre-trained **StyleGAN2** generator to produce the frontal face image.

The design of this system draws from three open-source repositories:

| Label | Repository | Core Contribution Used |
|-------|-----------|----------------------|
| **Ref-A** | [scaleway/frontalization](https://github.com/scaleway/frontalization) | GAN architecture philosophy, Encoder-Generator-Discriminator training loop, paired dataset structure |
| **Ref-B** | [NilouAp/StyleGAN_for_Face_Frontalization](https://github.com/NilouAp/StyleGAN_for_Face_Frontalization) | StyleGAN-based W+ encoding pipeline (pSp framework), LPIPS+ID+L2 loss stack, `start_from_latent_avg` anchoring, `w_norm_lambda` |
| **Ref-C** | [mhahmadi258/Face-frontalization-in-image-sequences-using-GAN-Inversion](https://github.com/mhahmadi258/Face-frontalization-in-image-sequences-using-GAN-Inversion) | E2Style-based GAN inversion, parse loss component, multi-scale encoder structure, transfer-learning on pretrained inversion model |

---

## 2. Overall Architecture

```
Profile Image (3×256×256)
        │
        ▼
┌───────────────────────────────┐
│  Pose-Conditioned FPN Encoder │  ← trainable (50M params)
│  ResNet-50 backbone           │
│  4-level FPN + AdaIN          │
│  Pose sinusoidal embedding    │
│  w_avg anchoring              │
└──────────────┬────────────────┘
               │  W+ latent code (B, 14, 512)
               ▼
┌───────────────────────────────┐
│  StyleGAN2 Generator          │  ← FROZEN (25M params)
│  Pre-trained on FFHQ-256      │
│  rosinality architecture      │
└──────────────┬────────────────┘
               │
               ▼
      Frontal Face (3×256×256)
```

---

## 3. Detailed Component Analysis and Source Attribution

### 3.1 Generator: Frozen Pre-trained StyleGAN2

**What we do:** We use the official rosinality StyleGAN2 generator (`stylegan2_model.py`) pre-trained on FFHQ-256 (`ffhq-256-config-e.pt`), with all weights frozen. The generator takes a W+ latent code of shape `(B, 14, 512)` and decodes it to a 256×256 RGB image.

**Source: Ref-B (NilouAp / pSp framework)**

NilouAp's repository (based on the pixel2style2pixel / pSp framework) introduces the core idea: instead of training a decoder from scratch, **freeze a powerful pre-trained StyleGAN2 generator** and only train an encoder to predict the W+ code. This is the single most important architectural choice in our project.

From NilouAp's training command:
```bash
python scripts/train.py \
  --encoder_type=GradualStyleEncoder \
  --start_from_latent_avg \      # ← key: anchor to w_avg
  ...
```

The FFHQ StyleGAN weights used (`rosinality` pre-trained) are also exactly referenced in NilouAp's model requirements.

**Our adaptation:** We adapted the rosinality codebase to work on Windows without C++/CUDA compilation by injecting pure-Python shims for `upfirdn2d` and `fused_leaky_relu` at module import time. The key remap translates checkpoint keys:
- `synthesis.conv1.*` → `conv1.*`
- `mapping.mapping.N.*` → `style.N.*`
- `conv.bias` stored as `(1,C,1,1)` → squeezed to `(C,)` for FusedLeakyReLU compatibility

---

### 3.2 Encoder: Pose-Conditioned FPN Encoder

**What we do:** A ResNet-50 backbone (ImageNet pretrained) extracts feature maps at 4 scales, which are connected via a Feature Pyramid Network (FPN). A pose embedding (sinusoidal encoding of yaw and pitch) is injected at each FPN level via Adaptive Instance Normalization (AdaIN). Adapter blocks convert each FPN feature map into a 512-dim style vector; 14 such vectors are stacked into the W+ code.

**Source: Ref-B (NilouAp / pSp) + Ref-C (mhahmadi258 / E2Style)**

**From Ref-B (pSp):**
The pSp framework introduces the `GradualStyleEncoder` — a feature pyramid encoder that produces **different style vectors for different resolution levels** of StyleGAN2. Each style vector controls different aspects of the output (coarse structure at low-res, fine details at high-res). This is the direct inspiration for our multi-level FPN adapter design.

```
pSp GradualStyleEncoder → Our PoseConditionedEncoder
 Encoder produces W+ [B, 18, 512]   →  We produce [B, 14, 512]
 Uses map2style modules at each level  →  We use FPNAdapterBlock
 No pose conditioning                 →  We add pose AdaIN injection
```

**From Ref-C (E2Style / mhahmadi258):**
E2Style extends pSp with a more efficient multi-stage encoder and the concept of **building from a pretrained StyleGAN inversion model** (transfer learning). This validates our approach of:
1. Starting from a pretrained ResNet-50 backbone (frozen early layers)
2. Using `start_from_latent_avg` to anchor the encoder output close to the mean frontal face initially

The E2Style architecture also inspired the idea of adapter/projection blocks at each FPN level rather than a single global bottleneck.

**Our original addition (not from any reference):**

1. **Pose Conditioning via AdaIN:** None of the three reference repositories incorporate explicit pose conditioning. In our architecture, the yaw and pitch angles extracted from the 300W-LP dataset are encoded using sinusoidal embeddings and injected into each FPN level via Adaptive Instance Normalization (`AdaIN` class in `generator.py`). This is crucial for frontalization (the encoder must know "how much rotation to undo").

2. **Sinusoidal Pose Embedding:** A custom `SinCosEmbedding` module encodes angles as `[sin(k·θ), cos(k·θ)]` for frequency bands `k = 2^0, ..., 2^8`. This provides a smooth, continuous representation of rotation angles.

3. **Learnable W+ offset `w_offset`:** A learnable `(1, n_styles, 512)` parameter initialized to zero allows the encoder to learn a global bias on top of the w_avg anchor without affecting the identity initialization.

---

### 3.3 Dataset: 300W-LP + AFLW2000-3D

**What we do:** We use the 300W-LP dataset (61,225 profile face images synthesized from 300W using 3DMM fitting) for training, organized as `(profile_image, yaw, pitch)` → `(frontal_GT)` pairs. Validation uses AFLW2000-3D.

**Source: Ref-A (Scaleway) + Original**

**From Ref-A (Scaleway/frontalization):**
Scaleway's approach establishes the **paired dataset structure** for face frontalization: for each subject, there is a set of profile images with a corresponding frontal target. Their `ExternalInputIterator` class traverses this structure to build `(profile, frontal)` pairs.

Our `FrontalizationDataset` in `data/dataset.py` mirrors this exact logic:
```python
# Scaleway's structure:
self.frontals = [...]           # GT frontal images per subject
profile_files = [[...], [...]]  # multiple profile images per subject

# Our structure (from dataset.py):
subject_map[subject_id] = {
    'frontal_paths': [...],
    'profile_paths': [...]
}
```

**Our adaptation:**
- We use **300W-LP** (a much larger and more diverse dataset) instead of CMU Multi-PIE
- Our pairs include **yaw/pitch metadata** (from the filename encoding in 300W-LP) to enable pose conditioning
- We implement a **subject-aware pairing strategy**: profile and frontal images are sampled from the same subject within each mini-batch, critical for identity learning
- We added a **subject-indexed cache** (`subject_map_cache.pkl`) to avoid re-scanning 61K files on every run

---

### 3.4 Discriminator: Patch-Level Adversarial Training

**What we do:** A `PatchDiscriminator` (`models/discriminator.py`) with 5 convolutional blocks (LeakyReLU, no BatchNorm in first layer) outputs a patch-level real/fake map. Training uses LSGAN (least-squares adversarial loss). A **20-epoch warmup** delays discriminator activation.

**Source: Ref-A (Scaleway/frontalization)**

Scaleway's repo provides the clearest model of a discriminator for face frontalization:
```python
# Scaleway's Discriminator D:
class D(nn.Module):
    self.main = nn.Sequential(
        nn.Conv2d(3, 16, 4, 2, 1),
        nn.LeakyReLU(0.2, inplace=True),
        ...
        nn.Conv2d(512, 1, 4, 2, 1, bias=False),
        nn.Sigmoid()    # Binary cross-entropy
    )
```

Our `PatchDiscriminator` follows this same Conv-LeakyReLU backbone philosophy but upgrades to:
- **PatchGAN output** (spatial map, not global scalar) for more granular feedback
- **LSGAN loss** instead of BCE for training stability
- **No sigmoid activation** (LSGAN uses raw logits)

**Our original addition:**
The **20-epoch discriminator warmup** (`disc_warmup_epoch: 20` in `config.yaml`) is not present in any reference. We introduced this after observing that early-epoch adversarial loss forced the encoder into collapsed "realistic but wrong identity" modes before it could learn basic reconstruction. During warmup: `lambda_adv = 0`.

---

### 3.5 Loss Function Stack

**What we do:** A composite loss is used with the following components:

```
L_total = λ_id·L_id + λ_lpips·L_lpips + λ_l2·L_l2 + λ_perc·L_perc
        + λ_wnorm·L_wnorm + λ_adv·L_adv
```

| Loss | λ | Phase | Source |
|------|---|-------|--------|
| `L_id` (FaceNet cosine similarity) | 2.0 | 1+2 | **Ref-B** |
| `L_lpips` (LPIPS AlexNet, center crop) | 1.5 | 1+2 | **Ref-B** |
| `L_l2` (pixel MSE) | 1.0 | 1+2 | **Ref-A** |
| `L_perc` (VGG perceptual) | 0.5 | 1+2 | **Ref-A, Ref-B** |
| `L_wnorm` (W+ distance from w_avg) | 0.001 | 1+2 | **Ref-B** |
| `L_adv` (LSGAN, PatchGAN) | 0.1 | 1+2 (after warmup) | **Ref-A** |

**From Ref-B (NilouAp / pSp):**

The pSp training command explicitly uses `--id_lambda`, `--lpips_lambda`, `--l2_lambda`, `--w_norm_lambda`. Our loss weights in `config.yaml` are directly scaled versions of these. The key insight from pSp is the **W-norm loss**:
```
L_wnorm = ||W+ - w_avg||₂
```
This regularizes the encoder to stay close to the mean frontal latent, preventing mode collapse. NilouAp uses `w_norm_lambda=0.005`; we tuned it down to `0.001` to reduce the "average face" pull as training progresses.

The **LPIPS crop loss** (computing LPIPS on a tight face crop rather than the full 256×256 image) is also directly from pSp/NilouAp's `lpips_lambda_crop` parameter.

**From Ref-A (Scaleway):**

Scaleway introduces the idea of a **pixel-level reconstruction loss** alongside GAN training:
```python
# From Scaleway main.py:
errG = GAN_factor * errG_GAN + L1_factor * errG_L1 + L2_factor * errG_L2
```
This validates our `lambda_l2` pixel MSE component, which provides a direct, unambiguous gradient toward the GT frontal image face structure.

**From Ref-C (mhahmadi258):**

E2Style's training command uses `--parse_lambda=1`, a **face parsing (segmentation) loss** that enforces region-wise consistency (skin, hair, eyes, mouth). We implemented this as `ParseLoss` in `models/losses.py`. However, it was **disabled** (`lambda_parse: 0.0`) in Phase 2 due to the difficulty of obtaining accurate face parse maps from the 300W-LP dataset without a pre-trained face parser. The infrastructure remains in the codebase for potential re-enablement.

---

### 3.6 Two-Phase Training Schedule

**What we do:**

- **Phase 1 (epochs 1–49):** Full loss stack active. Discriminator warmup (first 20 epochs no `L_adv`). Encoder learns to map profile→reasonable W+ code.
- **Phase 2 (epochs 50–100):** Parse loss dropped. Focus on identity sharpening.

**Source: Ref-A (Scaleway) + Original**

Scaleway demonstrated a two-phase approach:
> "We managed to produce the following results... we trained the GAN model for the first three epochs, then **set the GAN_factor to zero** and continued to train only the generator, optimizing the L1 and L2 losses, for two more epochs."

This validates the concept of **phased training** — changing the loss contribution over time — which we generalized into a configurable `phase2_start` epoch (50 in our setup) where `lambda_parse` is set to zero.

---

### 3.7 The `frontal_latent_avg` Initialization

**What we do:** Before training, `setup_stylegan.py` samples 500 random `z` vectors, maps them through StyleGAN2's mapping network, and averages the resulting W codes to obtain `w_avg` of shape `(1, 14, 512)`. During training, the encoder output `w_delta` is added to this `w_avg`:
```python
W+ = w_avg + w_delta
```

**Source: Ref-B (NilouAp / pSp)**

The pSp `--start_from_latent_avg` flag is the direct origin of this design. In pSp:
```python
# From pSp encoder:
if self.opts.start_from_latent_avg:
    return codes + self.latent_avg.repeat(...)
```

This initializes the encoder to output the average face for all inputs (random initialization of the projection layers ensures `w_delta ≈ 0` at epoch 0). Training then nudges the encoder toward identity-specific W+ codes. We compute the frontal-specific average from 500 z samples rather than a true face-image average, as we do not have a pre-trained encoder to invert images with.

---

### 3.8 Mixed Precision Training & Gradient Accumulation

**What we do:** Training uses PyTorch AMP (`torch.cuda.amp`) with `float16` compute and `float32` master weights. Effective batch size = 4 (physical) × 4 (accumulation steps) = **batch 16**, run on an RTX 4060 8GB.

**Source: Original contribution (not in any reference)**

None of the three reference repos target consumer laptop GPUs. Scaleway used Scaleway RENDER-S GPU instances; NilouAp used `--batch_size=8` on unspecified hardware; mhahmadi258 used `--batch_size=16`. Our AMP + gradient accumulation strategy was necessary to fit the hybrid pipeline (encoder + StyleGAN2 + discriminator + LPIPS + FaceNet) within 8GB VRAM.

---

## 4. Key Differences from Reference Implementations

| Aspect | Ref-A (Scaleway) | Ref-B (NilouAp) | Ref-C (mhahmadi258) | **Our Approach** |
|--------|-----------------|-----------------|---------------------|-----------------|
| Generator | Shallow CNN Encoder-Decoder (128px) | StyleGAN2 (frozen, 1024px) | StyleGAN2 (frozen) | StyleGAN2 (frozen, **256px**) |
| Encoder | Simple convolutional encoder | GradualStyleEncoder (pSp) | E2Style encoder | **ResNet-50 FPN + AdaIN pose** |
| Pose conditioning | None | None | None | **Yes — sinusoidal yaw/pitch embedding** |
| Dataset | CMU Multi-PIE | FFHQ | Custom synthetic video | **300W-LP + AFLW2000-3D** |
| Image resolution | 128×128 | 1024×1024 | 1024×1024 | **256×256** |
| Discriminator | Simple CNN + BCE | None (pSp has no disc.) | None | **PatchGAN + LSGAN + 20-ep warmup** |
| W-norm loss | No | Yes | Yes | **Yes (λ=0.001)** |
| Identity loss | No | ArcFace (IR-SE50) | ArcFace (IR-SE50) | **FaceNet/VGGFace2 fallback** |
| LPIPS | No | Yes (crop) | Yes | **Yes (full + crop)** |
| Hardware target | Cloud GPU | High-end GPU | High-end GPU | **RTX 4060 8GB laptop** |
| CUDA compilation | N/A | Required | Required | **Not required (pure PyTorch shims)** |

---

## 5. Architecture Diagrams

### 5.1 Encoder Detail

```
Profile Image (3×256×256)
       │
       ▼
   [ResNet-50 Backbone]
   ├── stem + layer1 → c1 (256ch, /4)   [FROZEN]
   ├── layer2         → c2 (512ch, /8)   [FROZEN]
   ├── layer3         → c3 (1024ch, /16) [trainable]
   └── layer4         → c4 (2048ch, /32) [trainable]
       │
       ▼
   [FPN Top-Down Pathway]
   p4 = lat4(c4)                          (256ch, /32)
   p3 = lat3(c3) + upsample(p4)           (256ch, /16)
   p2 = lat2(c2) + upsample(p3)           (256ch, /8)
   p1 = lat1(c1) + upsample(p2)           (256ch, /4)
       │
       ▼ (alongside)
   Pose Embedding: [yaw, pitch] → sincos → MLP → pose_vec (256-dim)
       │
       ▼
   [FPN Adapter Blocks] (14 total, distributed across 4 levels: 4,4,3,3)
   For each FPN level:
     Conv3×3↓ → ReLU → ConvTranspose2d↑ → AdaIN(pose_vec) → GlobalAvgPool → Linear(512)
       │
       ▼
   W+ delta: (B, 14, 512)
   + w_avg: (1, 14, 512)
   + w_offset: learnable (1, 14, 512)
       │
       ▼
   W+ code: (B, 14, 512)
```

### 5.2 Training Loop

```
For each epoch:
  ┌─ Discriminator step ─────────────────────────────────────────────┐
  │  real_frontal ──→ D ──→ D_loss_real                              │
  │  fake_frontal ──→ D ──→ D_loss_fake  (only if epoch > warmup)    │
  │  D_loss = D_loss_real + D_loss_fake  → optimizer_D.step()        │
  └──────────────────────────────────────────────────────────────────┘

  ┌─ Generator/Encoder step ─────────────────────────────────────────┐
  │  profile ──→ Encoder(pose) ──→ W+ ──→ StyleGAN2 ──→ fake_frontal │
  │                                                                   │
  │  L_total = λ_id·FaceNet(fake,real)                               │
  │          + λ_lpips·LPIPS(fake,real)                              │
  │          + λ_l2·MSE(fake,real)                                   │
  │          + λ_perc·VGG(fake,real)                                 │
  │          + λ_wnorm·||W+ - w_avg||                                │
  │          + λ_adv·LSGAN(D(fake))   [if epoch > warmup]           │
  │                                                                   │
  │  L_total → optimizer_E.step()                                    │
  └──────────────────────────────────────────────────────────────────┘
```

---

## 6. Training Configuration Summary

```yaml
# config.yaml — key parameters
training:
  epochs:          100
  batch_size:      4        # physical
  grad_accum:      4        # effective batch = 16
  lr_g:            2e-4
  disc_warmup:     20       # no adversarial loss for first 20 epochs
  phase2_start:    50       # drop parse loss

model:
  stylegan_ckpt:   ffhq-256-config-e.pt   # rosinality pre-trained
  n_styles:        14                      # log2(256)*2-2
  style_dim:       512

loss:
  lambda_id:       2.0
  lambda_lpips:    1.5
  lambda_l2:       1.0
  lambda_perc:     0.5
  lambda_wnorm:    0.001
  lambda_adv:      0.1
```

---

## 7. Quantitative Evaluation Metrics

The `evaluate.py` script computes the following metrics on the AFLW2000-3D test set:

| Metric | Definition | Target |
|--------|-----------|--------|
| **SSIM** | Structural Similarity (0–1, higher better) | > 0.55 |
| **PSNR** | Peak Signal-to-Noise Ratio (dB) | > 22 dB |
| **LPIPS** | Learned Perceptual Image Patch Similarity (lower better) | < 0.35 |
| **ID** | FaceNet cosine similarity to GT frontal | > 0.50 |

---

## 8. Summary of Inspirations by Repository

### Repository 1 (Ref-A): scaleway/frontalization
> **Key inspiration: GAN training paradigm, loss formulation, paired dataset structure**

- Established that face frontalization is best approached as a **supervised GAN** problem with a paired dataset
- Inspired our **Encoder + Discriminator** training loop design
- The `L1 + L2 + GAN` combined loss philosophy directly informed our `lambda_l2 + lambda_adv` components
- The `profile images → frontal target` dataset organization is directly followed in our `FrontalizationDataset`
- Their observation that GAN training sharpens fine details that pixel-loss alone cannot recover motivated keeping `lambda_adv = 0.1` even after warmup

### Repository 2 (Ref-B): NilouAp/StyleGAN_for_Face_Frontalization
> **Key inspiration: StyleGAN2 as frozen decoder, pSp encoder, W+ latent space, w_avg anchoring, identity+perceptual loss stack**

This is the **most architecturally influential** reference. Nearly every high-level design decision traces back here:
- **Freeze StyleGAN2**: instead of training a decoder, leverage FFHQ pre-training
- **W+ latent encoding**: the encoder predicts a 14-vector W+ code
- **`start_from_latent_avg`**: encoder output is `w_avg + w_delta` for identity initialization
- **Loss stack**: `id_lambda + lpips_lambda + l2_lambda + w_norm_lambda` is directly adopted
- **LPIPS crop**: perceptual loss on face-region crop for better identity convergence
- The use of **FaceNet / ArcFace** for identity loss

### Repository 3 (Ref-C): mhahmadi258/Face-frontalization-in-image-sequences-using-GAN-Inversion
> **Key inspiration: E2Style multi-scale encoder architecture, parse loss, transfer learning initialization**

- **Multi-scale encoder**: the concept of extracting style vectors at different feature resolutions (our FPN adapter blocks)
- **Parse loss** (`lambda_parse`): region-aware loss enforcing face structure consistency — implemented in our codebase as `ParseLoss` (currently disabled pending face parser integration)
- **Transfer learning**: starting from a pretrained image encoder and fine-tuning for frontalization, motivating our use of ImageNet-pretrained ResNet-50 backbone (layers 1-2 frozen)
- **Efficient adapter design**: using lightweight adapter blocks rather than training the full encoder end-to-end

---

## 9. Original Contributions (Not Present in Any Reference)

1. **Sinusoidal Pose Conditioning** — None of the three references condition on head pose. Our `SinCosEmbedding + PoseEmbedding + AdaIN` injection at each FPN level enables explicit pose-awareness.

2. **Windows GPU Deployment without CUDA Compilation** — We wrote pure-Python shims for `upfirdn2d` and `fused_leaky_relu` that make the rosinality CUDA-dependent ops portable to any platform (Windows/RTX 4060, no C++ compiler).

3. **Discriminator Warmup Strategy** — A 20-epoch warmup where `lambda_adv = 0` prevents adversarial collapse before the encoder develops basic reconstruction capability. Inspired by observing training dynamics, not from any reference paper.

4. **Corrected StyleGAN2 Key Remapping** — Building a production-grade checkpoint loader that correctly maps 5 distinct naming convention differences between the checkpoint and the rosinality model (`synthesis.*` prefix strip, `.bias`→`.activate.bias`, `.noise`→`.noise.weight`, `mapping.mapping.*`→`style.*`, 4D bias squeeze to 1D).

5. **Subject-Aware Dataset Pairing** — Our dataset pairs `(profile, frontal)` from the same subject within each mini-batch, using a cached subject map. This is stronger than random pairing used in Scaleway.

---

*Report generated for EE655 Group Project, April 2026.*
*Training: RTX 4060 Laptop GPU (8GB), dataset: 300W-LP (2500 train / 1000 val pairs)*
