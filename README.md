# FusionFront: Dual-Pipeline Facial Image Restoration & Frontalization

**EE655 Group Project — Group 21**

FusionFront is an advanced dual-pipeline computer vision framework designed to solve two challenging problems in facial image processing:

1. **Blind Face Super-Resolution (Deblurring & Restoration):** Reconstructing high-frequency, photorealistic details from severely degraded, low-resolution, or pixelated inputs.
2. **Pose-Conditioned Face Frontalization:** Transforming extreme profile images (up to 90° yaw) into a normalized frontal view while preserving identity.

---

## 🔹 Part 1: Image Enhancement & Blind Super-Resolution (Deblurring)

### 1.1 Overview

Real-world images often suffer from unknown degradation, blur, and downsampling. This module focuses on reconstructing high-quality images from degraded inputs.

### 1.2 Models Used

#### SRResNet (Pixel-Level Reconstruction)

* Minimizes pixel-wise loss (L1)
* Produces smooth but accurate outputs
* Avoids hallucination

#### Real-ESRGAN (Perceptual Enhancement)

* Uses GAN + perceptual loss
* Produces sharper, visually realistic outputs
* Hallucinates high-frequency details

---

### 1.3 Quantitative Results

| Metric  | SRResNet   | Real-ESRGAN | Better   |
| ------- | ---------- | ----------- | -------- |
| PSNR    | **35.59**  | 29.35       | SRResNet |
| SSIM    | **0.9883** | 0.8424      | SRResNet |
| L1 Loss | **0.0127** | 5.4104      | SRResNet |

---

### 1.4 Conclusion

* **SRResNet:** Better for accuracy (metrics)
* **Real-ESRGAN:** Better for visual realism

---

## 🔹 Part 2: Hybrid Face Frontalization (FusionFront)

### 2.1 Architecture Overview

#### Encoder Components

* **Geometry Encoder:** extracts facial landmarks → 512D
* **Identity Encoder (ArcFace):** preserves identity → 512D
* **pSp Encoder:** generates W+ latent codes

---

### 2.2 Latent Refinement

* Multi-head cross-attention
* Combines identity + geometry
* Produces refined latent vector

---

### 2.3 Generator

* Frozen StyleGAN2 (FFHQ-256)
* Ensures photorealistic output
* Prevents GAN collapse

---

### 2.4 Loss Functions

| Loss     | Purpose               |
| -------- | --------------------- |
| Identity | preserve identity     |
| LPIPS    | perceptual similarity |
| L1       | pixel alignment       |

---

## 🔹 Part 3: Interactive Demo

### Features

* Frontalization + enhancement pipeline
* Side-by-side comparison
* Dynamic gallery support

---

### Run the App

```bash
pip install -r requirements.txt
pip install gfpgan facexlib basicsr
python demo.py
```

Access at:

```
http://localhost:7860
```

---

## 🔹 Project Structure

```
.
├── FusionFront/        # Frontalization pipeline
├── Deblurring/         # Super-resolution & restoration
├── checkpoints/
├── models/
├── data/
├── utils/
├── README.md
```

---

## 🔹 Final Insight

FusionFront combines:

* **Geometric understanding**
* **Identity preservation**
* **Generative modeling**

with a complementary **Deblurring module**, creating a complete facial restoration pipeline.

---
