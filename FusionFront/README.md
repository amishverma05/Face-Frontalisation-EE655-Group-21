# Face Frontalization — EE655 Group 21

> **Hybrid StyleGAN2 Encoder–Generator pipeline for identity-preserving face frontalization.**
> Converts arbitrary-pose face images into photorealistic frontal views using a frozen pre-trained StyleGAN2 generator and a trainable Pose-Conditioned FPN Encoder.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Dataset](#dataset)
- [Setup](#setup)
- [Training](#training)
- [Evaluation](#evaluation)
- [Inference](#inference)
- [Project Structure](#project-structure)
- [References](#references)

---

## Overview

Standard face recognition systems achieve best performance on frontal-facing images. This project implements a **face frontalization** system that synthesizes a frontal view from a profile or semi-profile image.

### Key Design Choices

| Component | Choice | Reason |
|-----------|--------|--------|
| Generator | Frozen StyleGAN2 (FFHQ-256) | Leverages state-of-the-art face synthesis without training a decoder from scratch |
| Encoder | ResNet-50 FPN + AdaIN pose injection | Multi-scale identity features + explicit yaw/pitch conditioning |
| Latent space | W+ (14 × 512 = 7168-dim) | Per-layer style control over coarse structure and fine details |
| Loss | ID + LPIPS + L2 + W-norm + Adversarial | Balances identity preservation, perceptual quality, and realism |

---

## Architecture

```
Profile Image (3 × 256 × 256)
        │
        ▼
┌──────────────────────────────────┐
│  Pose-Conditioned FPN Encoder    │  ← trainable (~50M params)
│  ResNet-50 backbone (frozen L1-2)│
│  4-level FPN + AdaIN pose inject │
│  Sinusoidal yaw/pitch embedding  │
│  w_avg anchoring (identity init) │
└──────────────┬───────────────────┘
               │  W+ latent (B, 14, 512)
               ▼
┌──────────────────────────────────┐
│  StyleGAN2 Generator             │  ← FROZEN (25M params, FFHQ-256)
│  rosinality architecture         │
│  Pure-PyTorch (no CUDA compile)  │
└──────────────┬───────────────────┘
               │
               ▼
      Frontal Face (3 × 256 × 256)
```

### Training Schedule

| Phase | Epochs | Active Losses |
|-------|--------|--------------|
| Warmup | 1–20 | ID + LPIPS + L2 + Perc + W-norm (no Discriminator) |
| Phase 1 | 1–49 | All losses + Adversarial (PatchGAN LSGAN) |
| Phase 2 | 50–100 | ID + LPIPS + L2 + W-norm + Adversarial (no Parse) |

---

## Dataset

We use the **300W-LP** dataset for training and **AFLW2000-3D** for evaluation.

### 300W-LP
- 61,225 images synthesized from 300W using 3DMM fitting
- Covers yaw angles from −90° to +90°
- Filenames encode yaw angle (e.g., `AFW_134212_1_0.jpg` → 0° yaw)

### AFLW2000-3D
- 2,000 in-the-wild face images with annotated 3D landmarks
- Used for SSIM / PSNR / LPIPS / ID evaluation

### Download

```bash
# Download and prepare datasets automatically:
python download_data.py
```

Or follow the manual instructions in [`data/DATASET.md`](data/DATASET.md).

**Expected structure after download:**
```
data/
├── 300W_LP/
│   ├── AFW/         ← subject folders
│   ├── HELEN/
│   ├── IBUG/
│   └── LFPW/
└── AFLW2000/
    └── *.jpg + *.mat
```

---

## Setup

### Prerequisites

- Python 3.9+
- NVIDIA GPU (tested on RTX 4060 8 GB)
- CUDA 11.8+

### Installation

```bash
# 1. Clone the repo
git clone https://github.com/amishverma05/Face-Frontalisation-EE655-Group-21.git
cd Face-Frontalisation-EE655-Group-21

# 2. Create virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac

# 3. Install dependencies
pip install -r requirements.txt
```

### StyleGAN2 Checkpoint

Download the pre-trained FFHQ-256 StyleGAN2 checkpoint:

```bash
# Automatic download (Google Drive):
python setup_stylegan.py
```

This will:
1. Download `ffhq-256-config-e.pt` to `checkpoints/`
2. Compute and save the frontal latent average `checkpoints/frontal_latent_avg.pt`

> **Note:** No C++ compiler or CUDA kernel compilation is required. The rosinality CUDA ops are replaced with pure-Python/PyTorch fallbacks that run on any GPU.

---

## Training

### Quick Start (full pipeline)

```bat
main.bat
```

This runs sanity checks → StyleGAN2 setup → training in sequence.

### Manual Steps

```bash
# Step 1: Sanity check (verifies GPU, models, losses work)
python sanity_check.py

# Step 2: Setup StyleGAN2 (downloads checkpoint + computes frontal w_avg)
python setup_stylegan.py

# Step 3: Train
python train.py --config config.yaml

# Step 4: Resume training from a checkpoint
python train.py --config config.yaml --resume checkpoints/best.pth
```

### Configuration

All hyperparameters are in [`config.yaml`](config.yaml):

```yaml
training:
  epochs:       100
  batch_size:   4          # physical (effective batch=16 with grad_accum=4)
  lr_g:         2.0e-4
  disc_warmup_epoch: 20    # discriminator starts after epoch 20
  phase2_start: 50         # drop parse loss at epoch 50

loss:
  lambda_id:    2.0        # FaceNet identity loss
  lambda_lpips: 1.5        # LPIPS perceptual loss
  lambda_l2:    1.0        # Pixel L2 loss
  lambda_perc:  0.5        # VGG perceptual loss
  lambda_wnorm: 0.001      # W+ regularization
  lambda_adv:   0.1        # Adversarial (LSGAN)
```

### Monitoring

Sample images are saved to `samples/` at the end of every epoch showing:
- Row 1: Profile input
- Row 2: Generated frontal
- Row 3: Ground truth frontal

---

## Evaluation

```bash
python evaluate.py --checkpoint checkpoints/best.pth --data_root data/AFLW2000
```

**Metrics computed:**

| Metric | Description |
|--------|-------------|
| SSIM | Structural Similarity (higher = better) |
| PSNR | Peak Signal-to-Noise Ratio in dB |
| LPIPS | Perceptual distance (lower = better) |
| ID | FaceNet cosine similarity to GT |

---

## Inference

```bash
python inference.py \
  --checkpoint checkpoints/best.pth \
  --input path/to/profile_image.jpg \
  --output output_frontal.jpg
```

For a directory of images:

```bash
python inference.py \
  --checkpoint checkpoints/best.pth \
  --input_dir path/to/profile_images/ \
  --output_dir results/
```

---

## Project Structure

```
.
├── config.yaml              # All hyperparameters
├── train.py                 # Main training loop
├── evaluate.py              # Evaluation on AFLW2000-3D
├── inference.py             # Single-image / batch inference
├── sanity_check.py          # Pre-training validation
├── setup_stylegan.py        # Download checkpoint + compute w_avg
├── main.bat                 # Full pipeline runner (Windows)
├── requirements.txt
│
├── models/
│   ├── generator.py         # PoseConditionedEncoder (ResNet-50 FPN)
│   ├── stylegan2_wrapper.py # StyleGAN2 loader + pure-Python CUDA shims
│   ├── discriminator.py     # PatchGAN discriminator
│   └── losses.py            # All loss functions (ID, LPIPS, W-norm, etc.)
│
├── data/
│   ├── dataset.py           # FrontalizationDataset (300W-LP + AFLW2000)
│   ├── DATASET.md           # Dataset download instructions
│   └── __init__.py
│
├── stylegan2_model.py       # Rosinality StyleGAN2 architecture
├── stylegan2_op/            # Pure-Python fallbacks for CUDA ops
├── utils/                   # Misc utilities
│
├── checkpoints/
│   └── frontal_latent_avg.pt  # Precomputed frontal W+ average (tracked)
│
└── data/
    └── DATASET.md           # Dataset setup guide
```

---

## References

This project draws inspiration from the following works:

1. **[scaleway/frontalization](https://github.com/scaleway/frontalization)**
   — GAN training loop, paired profile↔frontal dataset structure, pixel reconstruction loss

2. **[NilouAp/StyleGAN_for_Face_Frontalization](https://github.com/NilouAp/StyleGAN_for_Face_Frontalization)**
   — Based on the [pSp framework](https://arxiv.org/abs/2008.00951); frozen StyleGAN2 decoder, W+ encoding, `start_from_latent_avg`, identity + LPIPS + W-norm loss stack

3. **[mhahmadi258/Face-frontalization-in-image-sequences-using-GAN-Inversion](https://github.com/mhahmadi258/Face-frontalization-in-image-sequences-using-GAN-Inversion)**
   — Based on [E2Style](https://github.com/wty-ustc/e2style); multi-scale encoder design, parse loss concept, transfer-learning initialization

4. **[rosinality/stylegan2-pytorch](https://github.com/rosinality/stylegan2-pytorch)**
   — StyleGAN2 PyTorch implementation used as the frozen decoder

5. **Richardson et al., "Encoding in Style: a StyleGAN Encoder for Image-to-Image Translation" (CVPR 2021)**
   — pSp paper: W+ GAN inversion for image-to-image translation

### Dataset Papers

- **300W-LP** — Zhu et al., "Face Alignment Across Large Poses: A 3D Solution" (CVPR 2016)
- **AFLW2000-3D** — Zhu et al., same paper as above

---

## Team — EE655 Group 21

| Name | Contribution |
|------|-------------|
| Amish Verma | Repo management, dataset pipeline |
| Contributor | Architecture design, training, debugging |

---

## License

This project is for academic purposes (EE655 course project). The StyleGAN2 checkpoint is subject to the [NVIDIA license](https://github.com/NVlabs/stylegan2/blob/master/LICENSE.txt).
