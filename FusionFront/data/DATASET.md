# Dataset Setup Guide

## What We Use

| Split | Dataset | # Images | Size |
|-------|---------|----------|------|
| Train | 300W-LP (non-Flip, yaw ≥ 15°) | ~22,000 pairs | ~2.7 GB |
| Val   | 300W-LP (5% held-out)         | ~1,100 pairs  | included |
| Test  | AFLW2000-3D                   | 2,000 images  | ~200 MB |

## Download Instructions

### Step 1: 300W-LP (Training Data)

**Official site (requires registration):**
http://www.cbsr.ia.ac.cn/users/xiangyuzhu/projects/3DDFA/main.htm

**Google Drive mirror (no registration):**
https://drive.google.com/file/d/0B7OEHD3T4eCkVGs0TkhUWFN6N1k

Download `300W-LP.zip` → extract here as `data/300W_LP/`

### Step 2: AFLW2000-3D (Evaluation Data)

**Same official site or:**
https://drive.google.com/file/d/1Nf4Z-XxJCx4bMEXt0MXCY5G8_E9TnC2

Download `AFLW2000-3D.zip` → extract here as `data/AFLW2000/`

### Quick download via script:
```bash
# Activate venv first!
.\venv\Scripts\activate

python download_data.py
```

---

## Expected Folder Structure After Extraction

```
data/
├── 300W_LP/
│   ├── AFW/
│   │   ├── AFW_134212_1_0.jpg      ← frontal  (yaw ≈ 0°)
│   │   ├── AFW_134212_1_0.mat      ← pose mat file
│   │   ├── AFW_134212_1_15.jpg     ← profile  (yaw ≈ 15°)
│   │   ├── AFW_134212_1_15.mat
│   │   ├── AFW_134212_1_30.jpg
│   │   ├── ...
│   ├── AFW_Flip/                   ← mirrored (we skip during training)
│   ├── HELEN/
│   ├── HELEN_Flip/
│   ├── IBUG/
│   ├── IBUG_Flip/
│   ├── LFPW/
│   └── LFPW_Flip/
│
└── AFLW2000/
    ├── image00002.jpg
    ├── image00002.mat
    ├── image00004.jpg
    ├── ...

```

## File Naming Convention (300W-LP)

```
AFW_134212_1_45.jpg
│   │        │  │
│   │        │  └── Yaw angle in degrees (0 = frontal)
│   │        └───── Subject index within source (AFW)
│   └────────────── Subject ID
└────────────────── Source dataset (AFW, HELEN, IBUG, LFPW)
```

The `.mat` file alongside each image contains:
```
Pose_Para: [pitch, yaw, roll, tx, ty, tz]   ← in RADIANS
roi:       [x, y, w, h]                       ← face bounding box
```

## How We Create Training Pairs

For each subject, we:
1. Find the image with smallest |yaw| as the **frontal reference**  
2. Pair every image with |yaw| ≥ 15° as a **profile input**
3. Return `(profile_img, frontal_img)` tensor pairs

```
Subject AFW_134212_1:
  _0.jpg  (yaw=0°)   → FRONTAL (target)
  _15.jpg (yaw=15°)  ┐
  _30.jpg (yaw=30°)  ├─ PROFILE inputs paired with the frontal above
  _45.jpg (yaw=45°)  │
  _60.jpg (yaw=60°)  ┘
```

## Pose Distribution in 300W-LP

```
yaw  0°–15°   frontal zone    → used as GT target only
yaw 15°–30°   mild profile    → ~35% of training pairs
yaw 30°–60°   medium profile  → ~40% of training pairs
yaw 60°–90°   extreme profile → ~25% of training pairs
```

## Validate Your Download

```bash
python download_data.py --validate-only
```

Expected output:
```
[data/300W_LP]   ~61,000 images | ~61,000 mat files  →  ✅ OK
[data/AFLW2000]  ~2,000  images | ~2,000  mat files  →  ✅ OK
```
