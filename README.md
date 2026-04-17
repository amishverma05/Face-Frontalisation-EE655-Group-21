# Fusion Front: Dual-Pipeline Facial Image Restoration & Frontalization 🎭✨

**EE655 Group Project — Group 21**

Fusion Front is an advanced, dual-pipeline computer vision framework developed to tackle two notoriously difficult challenges in facial image processing: 
1. **Blind Face Super-Resolution (Pixerization):** Reconstructing high-frequency, photorealistic structural details from severely degraded, low-resolution pixelated inputs.
2. **Pose-Conditioned Face Frontalization:** Rotating extreme profile images (up to 90° yaw) into a normalized, perfectly aligned forward-facing viewpoint while strictly mathematically preserving the subject's identity embeddings.

---

## 🚀 Part 1: Image Enhancement & Blind Super-Resolution (Pixerization)

### 1.1 Overview and Objective
Real-world images often suffer from unknown degradation kernels, motion blur, and severe downsampling. The objective of this sub-project is to reconstruct high-quality (HQ) images from pixelated low-resolution (LR) input streams. To analyze the behavioral differences in deep generative super-resolution, two distinct model architectures—**SRResNet** and **Real-ESRGAN**—were fine-tuned and evaluated on a paired face dataset.

### 1.2 Model Architectures & Loss Objectives

#### SRResNet (Pixel-Level Reconstruction)
SRResNet is engineered to learn a direct deterministic mapping between degraded and ground-truth images. 
* **Objective:** It minimizes strict pixel-wise reconstruction errors (L1 / Mean Absolute Error).
* **Behavior:** Because it mathematically averages the possible pixel distributions to minimize Euclidean loss, it inherently avoids hallucinating details. While mathematically safer, this leads to oversmoothed, slightly blurred images that lack perceptual crispness.

#### Real-ESRGAN (Perceptual Enhancement)
Real-ESRGAN is explicitly designed for **blind super-resolution** using a U-Net discriminator with spectral normalization.
* **Objective:** Focuses heavily on generating visually pleasing, photorealistic outputs by leveraging Adversarial (GAN) and Perceptual (VGG-style) learning parameters instead of strict pixel matching.
* **Behavior:** It violently forces the discriminator to extract "sharpness", which aggressively hallucinates high-frequency facial textures such as skin pores, eyelashes, and hair strands.

### 1.3 Quantitative Evaluation

Models were evaluated across structural similarity index measure (SSIM) and Peak Signal-to-Noise Ratio (PSNR). 

| Metric | SRResNet | Real-ESRGAN | Better Model |
| :--- | :--- | :--- | :--- |
| **PSNR (dB)** | **35.59** | 29.35 | SRResNet |
| **SSIM** | **0.9883** | 0.8424 | SRResNet |
| **Loss / L1** | **0.0127** | 5.4104 | SRResNet |

*Note: The superior PSNR and SSIM values of SRResNet mathematically confirm exceptional reconstruction accuracy and structural similarity to the exact Ground Truth bounding boxes.*

### 1.4 Qualitative Analysis & Conclusion

While mathematically superior, SRResNet outputs are observed to be exceedingly smooth. **Real-ESRGAN** introduces a trade-off: it suffers from higher point-for-point error, yet produces visually stunning, sharper images due to perceptual enhancement. 

**Conclusion:** 
* SRResNet mathematically dominates exact-match benchmarks. 
* Real-ESRGAN dominates human perceptual evaluation by hallucinating lost real-world textures.

---

## 🧠 Part 2: Hybrid Face Frontalization (StyleGAN2)

Generating frontal faces from severe side-profiles generally fails using standard U-Net techniques due to missing geometric data (e.g., the hidden side of the face). We migrated our pipeline to a **Hybrid Latent Inversion Architecture** utilizing the tremendous generative prios of a pre-trained StyleGAN2 network.

### 2.1 Deep Architecture Breakdown

Our architecture is split into three core phases: Conditional Extraction, Latent Refinement, and Generator Decoding.

#### Phase 1: The Encoder Trio
1. **Geometry Encoder (Trainable):** 
   Extracts 68 precise facial landmark heatmaps from the profile image, mapping out jawlines, nose bridges, and eye spacing. This is flattened into a highly dense `512-D` geometric tensor (`f_geo`).
2. **Identity Encoder (Frozen ArcFace):** 
   Utilizes an IR-SE50 Deep Neural Network pre-trained on millions of faces. It strips away lighting and pose data to return a purely mathematical `512-D` embedding of the user's facial identity (`f_id`).
3. **pSp Encoder (Trainable - ResNet50 FPN):** 
   A Feature Pyramid Network that extracts basic structural hierarchies at 3 distinct scales (coarse head structure, mid-level features, fine textures) and maps them into 18 initial style vectors -> `W_coarse [B, 18, 512]`.

#### Phase 2: Hybrid Latent Refinement via Cross-Attention
We cannot directly feed `W_coarse` to the generator because it was extracted from a sideways profile. 
* **Multi-Head Cross-Attention (MHA):** We pass `W_coarse` as the **Query**. We concatenate `[f_geo, f_id]` into a `[B, 2, 512]` matrix which serves as the **Key/Value**. 
* **Mechanism:** The network literally "attends" to the mathematical representation of the person's identity and merges it with the structural geometry of a frontal pose. 
* A gated residual MLP network initializes at `γ≈0` and continuously learns the specific latent delta mapping (`ΔW`). 

#### Phase 3: StyleGAN2 Synthesis Network
* We pass this refined latent tensor (`W_ref`) into an FFHQ-256 pre-trained **StyleGAN2** model. 
* **110/110 parameters are entirely frozen**. Because we do not allow the generator to update itself over the 100 training epochs, it acts as a safeguard. It guarantees that the system's output will *always* physically resemble a highly detailed photorealistic human face, completely avoiding "blob-like" checkerboard artifacts typical in GAN collapses.

### 2.2 Objective Loss Stack Configuration
The model is constrained mathematically to avoid auto-encoding background features:
* **Identity Loss (λ=4.0):** The highest weighted penalty matrix. Uses cosine similarity to strictly punish the network if the generated frontal face does not structurally match the ArcFace embedding of the input profile.
* **LPIPS Loss (λ=0.5):** AlexNet intermediate feature perceptual similarity prevents structural disfigurations.
* **L1 Loss (λ=0.1):** Minimized pixel-wise reconstruction to prevent the GAN from defaulting to simply mirroring the input image to get a high L1 score. 

---

## 💻 3. Implementation & Interactive Demo Interface

This framework includes an interactive UI combining both projects into one pipeline: it takes sideways profile inputs, Frontalizes them via the hybrid StyleGAN2 pipeline, and immediately cascades the result into a Real-ESRGAN/GFPGAN perceptual enhancement layer to synthesize high-frequency detail.

### 3.1 Setup Instructions

Ensure your Python environment runs PyTorch >= 2.0 with CUDA support.

1. **Install Virtual Environment Dependencies:**
   ```bash
   pip install -r requirements.txt
   pip install gfpgan facexlib basicsr  # Required for Super-Resolution cascade
   ```

2. **Network Checkpoints Location:**
   Place the heavily serialized models inside the `checkpoints/` root folder:
   * `best.pth` (Trained Encoder Weights)
   * `ffhq-256-config-e.pt` (Rosinality StyleGAN2 Snapshot)
   * `frontal_latent_avg.pt` 
   * `GFPGANv1.4.pth` (Face restoration enhancement weights)

### 3.2 Running the Application Workspace

Launch the integrated UI server locally:
```bash
.\venv\Scripts\python demo.py
```
*The web dashboard is universally mapped to `http://localhost:7860`.*

### 3.3 Dashboard Features
* **Live Dual-View Frontalization:** Provide explicit bounding angles (Yaw 0°-90° and Pitch). Generates the pure "Raw" Frontalization, and simultaneously computes the "Enhanced High-Frequency" Super-Res image side-by-side. 
* **Dynamic Gallery Integration:** Populate `assets/examples/` with `.jpg` images, and the dashboard gallery will auto-digest them for rapid presentation clicking.
* **CSS/HTML Architecture Inspection:** Tab 2 features a completely interactive frontend diagram covering the entire mathematical flow. Hover over any block (Geometry Encoder, MHA Refiner, Gated Residuals) to observe internal PyTorch tensor dimensions and data routes.
