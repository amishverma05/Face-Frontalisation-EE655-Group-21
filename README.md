# FusionFront: Dual-Pipeline Facial Image Restoration & Frontalization

**EE655 Group Project — Group 21**

FusionFront is a dual-pipeline deep learning framework designed to address two challenging problems in facial image processing:

1. **Blind Face Super-Resolution (Deblurring & Restoration):** Recovering high-frequency, photorealistic details from severely degraded or pixelated inputs.
2. **Pose-Conditioned Face Frontalization:** Transforming extreme profile images (up to 90° yaw) into a canonical frontal view while preserving identity embeddings.

---

## 🔹 Part 1: Image Enhancement & Blind Super-Resolution (Deblurring)

### 1.1 Problem Setting

Real-world images often suffer from unknown degradation kernels, motion blur, and aggressive downsampling. This module reconstructs high-quality (HQ) images from low-resolution (LR) inputs under blind degradation.

---

### 1.2 Model Architectures & Behavior

#### **SRResNet (Pixel-Level Reconstruction)**

* Optimizes strict **L1 (MAE)** loss
* Learns deterministic LR → HR mapping
* Produces **structurally accurate but smooth outputs**
* Avoids hallucination due to pixel averaging

#### **Real-ESRGAN (Perceptual Super-Resolution)**

* Uses **GAN + perceptual (VGG-based) loss**
* U-Net discriminator with spectral normalization
* Produces **sharp, visually realistic outputs**
* Hallucinates high-frequency details (pores, hair, edges)

---

### 1.3 Quantitative Evaluation

| Metric    | SRResNet   | Real-ESRGAN | Better   |
| --------- | ---------- | ----------- | -------- |
| PSNR (dB) | **35.59**  | 29.35       | SRResNet |
| SSIM      | **0.9883** | 0.8424      | SRResNet |
| L1 Loss   | **0.0127** | 5.4104      | SRResNet |

---

### 1.4 Insight

* **SRResNet:** mathematically optimal reconstruction
* **Real-ESRGAN:** perceptually superior (human vision)

👉 Trade-off: **accuracy vs realism**

---

## 🔹 Part 2: Hybrid Face Frontalization (FusionFront)

### 2.1 Motivation

Standard encoder-decoder architectures fail under extreme pose due to missing geometry. We instead leverage **latent-space manipulation with a frozen StyleGAN2 generator**.

---

### 2.2 Architecture Breakdown

#### 🔸 Phase 1: Encoder Trio

1. **Geometry Encoder (Trainable)**

   * Extracts 68 landmark heatmaps
   * Encodes spatial structure → **512-D vector (f_geo)**

2. **Identity Encoder (Frozen ArcFace)**

   * Removes pose/lighting
   * Produces identity embedding → **512-D vector (f_id)**

3. **pSp Encoder (ResNet50 + FPN)**

   * Multi-scale feature extraction
   * Outputs latent codes:

   ```
   W_coarse ∈ ℝ^{B × 18 × 512}
   ```

---

#### 🔸 Phase 2: Hybrid Latent Refinement

We refine latent codes using **Multi-Head Cross-Attention (MHA)**:

* Query → `W_coarse`
* Key/Value → `[f_geo, f_id]`

👉 Mechanism:

* Aligns identity with geometry
* Learns pose correction in latent space

---

#### 🔸 Gated Residual Refinement

* Learns latent offset:

  ```
  ΔW
  ```
* Uses gating parameter:

  ```
  γ ≈ 0 (initially)
  ```
* Final latent:

  ```
  W_ref = W_coarse + γΔW
  ```

👉 Ensures **stable training + gradual refinement**

---

#### 🔸 Phase 3: StyleGAN2 Generator

* Pretrained FFHQ-256 model
* **Fully frozen (110/110 layers)**
* Guarantees:

  * photorealism
  * no GAN collapse
  * stable outputs

---

### 2.3 Loss Function Stack

| Loss               | Weight | Role                 |
| ------------------ | ------ | -------------------- |
| Identity (ArcFace) | 4.0    | preserve identity    |
| LPIPS              | 0.5    | perceptual structure |
| L1                 | 0.1    | pixel consistency    |

👉 Prevents:

* identity drift
* structural distortion
* trivial copying

---

## 🔹 Part 3: Integrated Pipeline & Demo

FusionFront combines both modules into a **single inference pipeline**:

```
Profile Image
   ↓
Frontalization (FusionFront)
   ↓
Enhancement (Deblurring / GFPGAN)
   ↓
High-quality Frontal Output
```

---

### 3.1 Setup

```bash
pip install -r requirements.txt
pip install gfpgan facexlib basicsr
```

---

### 3.2 Required Checkpoints

Place in `checkpoints/`:

* `best.pth`
* `ffhq-256-config-e.pt`
* `frontal_latent_avg.pt`
* `GFPGANv1.4.pth`

---

### 3.3 Run Demo

```bash
python demo.py
```

Open:

```
http://localhost:7860
```

---

### 3.4 Features

* Dual output: raw vs enhanced
* Real-time frontalization
* Interactive gallery
* Visualization of internal architecture

---

## 🔹 Project Structure

```
.
├── FusionFront/        # Frontalization module
├── Deblurring/         # Super-resolution module
├── checkpoints/
├── models/
├── data/
├── utils/
├── README.md
```

---

## 🔹 Key Contribution

FusionFront integrates:

* **Geometric reasoning (landmarks)**
* **Identity preservation (ArcFace)**
* **Latent-space refinement (MHA + gating)**
* **Photorealistic generation (StyleGAN2)**

with a complementary **Deblurring pipeline**, forming a complete facial restoration system.

---
