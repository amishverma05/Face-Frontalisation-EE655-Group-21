"""
data/dataset.py
───────────────
300W-LP Dataset loader for face frontalization.

Dataset structure expected:
  300W_LP/
    AFW/        ← AFW_<id>_<n>_<yaw>.jpg  +  .mat  files
    HELEN/
    IBUG/
    LFPW/
    (AFW_Flip/, HELEN_Flip/ ... optional mirrored versions)

Each .mat file contains:
  - Pose_Para: [pitch, yaw, roll, tx, ty, tz]
  - roi:       [x, y, w, h]  bounding box

For evaluation: AFLW2000/ directory with same format.
"""

import os
import glob
import random
from collections import defaultdict

import numpy as np
import scipy.io as sio
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def read_pose(mat_path):
    """Return (pitch, yaw, roll) in degrees from a 300W-LP .mat file."""
    try:
        mat = sio.loadmat(mat_path)
        pose = mat['Pose_Para'][0]          # [pitch, yaw, roll, tx, ty, tz]
        pitch, yaw, roll = pose[0], pose[1], pose[2]
        return float(np.degrees(pitch)), float(np.degrees(yaw)), float(np.degrees(roll))
    except Exception:
        return None


def read_pose_full(mat_path):
    """Return (pitch, yaw, roll) tuple — alias kept for clarity."""
    return read_pose(mat_path)   # (pitch_deg, yaw_deg, roll_deg)


def parse_subject_id(filename):
    """
    Extract a stable subject key from the filename.
    e.g.  AFW_134212_1_45.jpg  →  AFW_134212_1
    Strategy: strip the last numeric token (pose angle).
    """
    base = os.path.splitext(os.path.basename(filename))[0]
    parts = base.rsplit('_', 1)
    # parts[-1] should be the yaw index; parts[0] is the id
    return parts[0] if len(parts) == 2 and parts[1].lstrip('-').isdigit() else base


# ---------------------------------------------------------------------------
# Main Dataset
# ---------------------------------------------------------------------------

class FrontalizationDataset(Dataset):
    """
    Returns (profile_img, frontal_img) tensors for supervised frontalization.

    Strategy
    --------
    1. Scan all images under `root`.
    2. Group by subject id (filename prefix before last `_<angle>`).
    3. For each subject, the image with the smallest |yaw| is the
       'frontal reference'.  All images with |yaw| >= min_yaw are
       'profile' images.
    4. Each sample = (profile_tensor, frontal_tensor).
    """

    def __init__(
        self,
        root: str,
        img_size: int = 128,
        min_yaw: float = 15.0,
        max_yaw: float = 90.0,
        augment: bool = True,
        split: str = 'train',          # 'train' | 'val'
        val_fraction: float = 0.05,
        seed: int = 42,
    ):
        super().__init__()
        self.root      = root
        self.img_size  = img_size
        self.min_yaw   = min_yaw
        self.max_yaw   = max_yaw
        self.augment   = augment

        self.transform = self._build_transform(augment, img_size)

        # Collect all jpg/png images (skip _Flip variants during pairing)
        all_imgs = sorted(glob.glob(os.path.join(root, '**', '*.jpg'), recursive=True))
        all_imgs += sorted(glob.glob(os.path.join(root, '**', '*.png'), recursive=True))

        # Build subject → {frontal_path, profile_paths[]}
        self.pairs = self._build_pairs(all_imgs, min_yaw, max_yaw)

        # Reproducible train/val split
        random.seed(seed)
        random.shuffle(self.pairs)
        n_val = max(1, int(len(self.pairs) * val_fraction))
        if split == 'val':
            self.pairs = self.pairs[:n_val]
            if len(self.pairs) > 1000: self.pairs = self.pairs[:1000] # Cap val mapping
        else:
            self.pairs = self.pairs[n_val:]
            if len(self.pairs) > 2500: self.pairs = self.pairs[:2500] # Cap train subset for 4-minute virtual epochs

        print(f"[Dataset] {split:5s} | {len(self.pairs):,} pairs | root: {root}")

    # ------------------------------------------------------------------
    def _build_pairs(self, all_imgs, min_yaw, max_yaw):
        """Group images by subject and create (profile, frontal) pairs."""
        import pickle
        cache_file = os.path.join(self.root, "subject_map_cache.pkl")
        
        if os.path.exists(cache_file):
            print(f"[Dataset] Loading cached index from {cache_file} (instant) ...")
            with open(cache_file, 'rb') as f:
                subject_map = pickle.load(f)
        else:
            print(f"[Dataset] Parsing ~122,000 .mat files... (This takes ~5-10 minutes ONCE) ...")
            subject_map = defaultdict(list)   # subject_id → [(img_path, abs_yaw, pitch)]
            import tqdm
            for img_path in tqdm.tqdm(all_imgs, desc="Indexing Dataset"):
                mat_path = os.path.splitext(img_path)[0] + '.mat'
                if not os.path.exists(mat_path):
                    continue
                pose = read_pose(mat_path)
                if pose is None:
                    continue
                pitch, yaw, _ = pose
                subject_id = parse_subject_id(img_path)
                subject_map[subject_id].append((img_path, abs(yaw), pitch))
            
            with open(cache_file, 'wb') as f:
                pickle.dump(subject_map, f)


        pairs = []
        for sid, entries in subject_map.items():
            if len(entries) < 2:
                continue
            # Handle both old cache format (2-tuple) and new (3-tuple)
            if len(entries[0]) == 2:
                # Old format: (img_path, abs_yaw) — upgrade to 3-tuple with pitch=0
                entries = [(e[0], e[1], 0.0) for e in entries]
            # Frontal = entry with smallest |yaw|
            entries.sort(key=lambda x: x[1])
            frontal_path, frontal_yaw, _ = entries[0]
            if frontal_yaw > min_yaw:
                continue                  # no near-frontal image for this subject

            for img_path, yaw, pitch in entries[1:]:
                if min_yaw <= yaw <= max_yaw:
                    pairs.append((img_path, frontal_path, yaw, pitch))

        return pairs

    # ------------------------------------------------------------------
    def _build_transform(self, augment, img_size):
        # Removed PIL ColorJitter which brutally slows down CPU when num_workers=0
        ops = [T.Resize((img_size, img_size)), T.ToTensor(),
               T.Normalize([0.5]*3, [0.5]*3)]
        return T.Compose(ops)

    # ------------------------------------------------------------------
    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        entry = self.pairs[idx]
        # Handle old 3-tuple format gracefully
        if len(entry) == 3:
            profile_path, frontal_path, yaw = entry
            pitch = 0.0
        else:
            profile_path, frontal_path, yaw, pitch = entry

        profile_img  = Image.open(profile_path).convert('RGB')
        frontal_img  = Image.open(frontal_path).convert('RGB')

        # ── Landmark-Based Square Face Crop ──────────
        # Fixes distortion and zoom inconsistencies by calculating tight bounds
        # directly from the 68 facial landmarks instead of the volatile 'roi' parameter.
        def crop_squared(img, mat_path, margin=0.6):
            try:
                mat = sio.loadmat(mat_path)
                if 'pt2d' in mat:
                    pts = mat['pt2d']  # (2, 68)
                    x_pts, y_pts = pts[0, :], pts[1, :]
                    min_x, max_x = float(x_pts.min()), float(x_pts.max())
                    min_y, max_y = float(y_pts.min()), float(y_pts.max())
                    x, y = min_x, min_y
                    w, h = max_x - min_x, max_y - min_y
                elif 'roi' in mat: # Fallback
                    r = mat['roi'][0]  # [x, y, w, h]
                    x, y, w, h = float(r[0]), float(r[1]), float(r[2]), float(r[3])
                else:
                    return img
                    
                cx, cy = x + (w / 2.0), y + (h / 2.0)
                side = max(w, h) * (1.0 + margin)
                half = side / 2.0
                
                left, top = int(cx - half), int(cy - half)
                right, bottom = int(cx + half), int(cy + half)
                
                # PIL automatically pads out-of-bounds coordinates with black
                return img.crop((left, top, right, bottom))
            except Exception:
                pass
            return img

        p_mat_path = os.path.splitext(profile_path)[0] + '.mat'
        f_mat_path = os.path.splitext(frontal_path)[0] + '.mat'

        profile_img = crop_squared(profile_img, p_mat_path)
        frontal_img = crop_squared(frontal_img, f_mat_path)


        profile_t = self.transform(profile_img)
        frontal_t = self.transform(frontal_img)

        return {
            'profile':       profile_t,
            'frontal':       frontal_t,
            'yaw':           torch.tensor(yaw,   dtype=torch.float32),
            'pitch':         torch.tensor(pitch, dtype=torch.float32),
            'profile_path':  profile_path,
            'frontal_path':  frontal_path,
        }


# ---------------------------------------------------------------------------
# Evaluation Dataset  (AFLW2000-3D)
# ---------------------------------------------------------------------------

class AFLW2000Dataset(Dataset):
    """
    Evaluation-only dataset from AFLW2000-3D.
    Returns single face images with their pose label.
    We treat images with |yaw| >= min_yaw as the 'hard' test set.
    """

    def __init__(self, root: str, img_size: int = 128, min_yaw: float = 0.0):
        self.img_size = img_size

        self.transform = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize([0.5]*3, [0.5]*3),
        ])

        all_imgs = sorted(glob.glob(os.path.join(root, '*.jpg')))
        all_imgs += sorted(glob.glob(os.path.join(root, '*.png')))

        self.samples = []
        for img_path in all_imgs:
            mat_path = os.path.splitext(img_path)[0] + '.mat'
            if not os.path.exists(mat_path):
                continue
            pose = read_pose(mat_path)
            if pose is None:
                continue
            _, yaw, _ = pose
            if abs(yaw) >= min_yaw:
                self.samples.append((img_path, abs(yaw)))

        print(f"[Dataset] AFLW2000 eval | {len(self.samples):,} images | min_yaw={min_yaw}°")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, yaw = self.samples[idx]
        img = Image.open(img_path).convert('RGB')
        return {
            'image': self.transform(img),
            'yaw':   torch.tensor(yaw, dtype=torch.float32),
            'path':  img_path,
        }
