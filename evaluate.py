"""
evaluate.py
────────────
Full evaluation script for a trained face frontalization model.

Supports both:
  • Hybrid pipeline (PoseConditionedEncoder + frozen StyleGAN2)
  • Legacy U-Net baseline

Usage
─────
  # Hybrid pipeline checkpoint:
  python evaluate.py --checkpoint checkpoints/best.pth

  # Legacy U-Net checkpoint:
  python evaluate.py --checkpoint checkpoints/best.pth --legacy

Outputs
───────
  results/
    ├── metrics.csv        ← per-image SSIM, PSNR, LPIPS, ID-score
    ├── metrics_summary.txt
    ├── visualizations/    ← side-by-side strips (profile | generated)
    └── grids/             ← NxN comparison grids
"""

import os
import argparse
import csv
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms as T
from tqdm import tqdm
from PIL import Image
import numpy as np

from models  import (Generator, PoseConditionedEncoder, StyleGAN2Generator)
from data    import AFLW2000Dataset, FrontalizationDataset
from utils   import (compute_ssim, compute_psnr,
                     LPIPSMetric, IDScoreMetric, MetricTracker,
                     save_sample_grid, save_comparison_strip)


# ─────────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────────

def load_model(ckpt_path: str, device):
    """
    Load a trained model from checkpoint.

    Supports both:
      • Hybrid pipeline: checkpoint has 'encoder' key
      • Legacy U-Net:    checkpoint has 'G' key

    Returns
    -------
    (infer_fn, pipeline_type)
    infer_fn : callable(batch) -> tensor  (given a batch dict, returns generated images)
    pipeline_type : 'hybrid' | 'legacy'
    """
    import yaml
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg  = ckpt.get('cfg', {})

    if 'encoder' in ckpt:
        # ─ Hybrid pipeline ───────────────────────────────────────────
        print(f"[Evaluate] Hybrid pipeline checkpoint: epoch {ckpt.get('epoch', '?')}")
        mc = cfg.get('model', {})
        encoder = PoseConditionedEncoder(
            pose_dim=mc.get('pose_dim', 256),
            style_dim=mc.get('style_dim', 512),
            n_styles=mc.get('n_styles', 18),
            pretrained=False,
        ).to(device)
        encoder.load_state_dict(ckpt['encoder'])
        encoder.eval()

        # Load StyleGAN2 checkpoint path from saved cfg
        stylegan_ckpt = mc.get('stylegan_ckpt', 'checkpoints/ffhq-256-config-e.pt')
        G_frozen = StyleGAN2Generator(
            ckpt_path=stylegan_ckpt if os.path.exists(stylegan_ckpt) else None,
            style_dim=mc.get('style_dim', 512),
        ).to(device)
        G_frozen.eval()

        w_avg = ckpt.get('w_avg', torch.zeros(1, mc.get('n_styles', 15),
                                              mc.get('style_dim', 512)))
        w_avg = w_avg.to(device).detach()

        @torch.no_grad()
        def infer_hybrid(batch):
            profile = batch['profile'].to(device)
            yaw     = batch['yaw'].to(device)
            pitch   = batch.get('pitch', torch.zeros_like(yaw)).to(device)
            w_pred  = encoder(profile, yaw, pitch, w_avg=w_avg)
            return G_frozen(w_pred, input_is_latent=True)

        print(f"[Evaluate] Encoder params: {encoder.count_params():,}")
        return infer_hybrid, 'hybrid'

    else:
        # ─ Legacy U-Net pipeline ──────────────────────────────────────
        print(f"[Evaluate] Legacy U-Net checkpoint: epoch {ckpt.get('epoch', '?')}")
        mc = cfg.get('model', {'ngf': 64, 'num_resblocks': 6})
        G = Generator(ngf=mc.get('ngf', 64),
                      n_res=mc.get('num_resblocks', 6)).to(device)
        G.load_state_dict(ckpt['G'])
        G.eval()

        @torch.no_grad()
        def infer_legacy(batch):
            return G(batch['profile'].to(device))

        print(f"[Evaluate] Generator params: {G.count_params():,}")
        return infer_legacy, 'legacy'


def tensor_to_pil(t: torch.Tensor) -> Image.Image:
    t = t.detach().cpu().clamp(-1, 1)
    t = ((t + 1.0) / 2.0 * 255).to(torch.uint8)
    return Image.fromarray(t.permute(1, 2, 0).numpy())


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation on paired dataset  (300W-LP val split)
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_paired(infer_fn, data_root: str, img_size: int,
                    device, results_dir: str, args):
    dataset = FrontalizationDataset(
        root=data_root, img_size=img_size,
        min_yaw=args.min_yaw, augment=False,
        split='val', val_fraction=args.val_fraction)

    loader = DataLoader(dataset, batch_size=args.batch_size,
                        shuffle=False, num_workers=0, pin_memory=True)

    lpips_fn = LPIPSMetric()
    id_fn    = IDScoreMetric()
    tracker  = MetricTracker()
    rows     = []

    vis_dir  = os.path.join(results_dir, 'visualizations')
    grid_dir = os.path.join(results_dir, 'grids')
    os.makedirs(vis_dir, exist_ok=True)
    os.makedirs(grid_dir, exist_ok=True)

    grid_profiles, grid_generated, grid_frontals = [], [], []

    for i, batch in enumerate(tqdm(loader, desc='Evaluating', ncols=90)):
        profile  = batch['profile'].to(device)
        frontal  = batch['frontal'].to(device)
        yaw      = batch['yaw']

        generated = infer_fn(batch)

        ssim_v  = compute_ssim(generated, frontal)
        psnr_v  = compute_psnr(generated, frontal)
        lpips_v = lpips_fn(generated, frontal)
        id_v    = id_fn(generated, frontal)

        tracker.update(ssim=ssim_v, psnr=psnr_v, lpips=lpips_v, id_score=id_v)

        for j in range(profile.shape[0]):
            rows.append({
                'profile_path': batch['profile_path'][j],
                'yaw_deg':      round(float(yaw[j]), 1),
                'ssim':         round(ssim_v, 4),
                'psnr':         round(psnr_v, 2),
                'lpips':        round(lpips_v, 4),
                'id_score':     round(id_v, 4),
            })

        # Accumulate for grid
        if i < 4:
            grid_profiles.append(profile)
            grid_generated.append(generated)
            grid_frontals.append(frontal)

    # Save sample grids
    if grid_profiles:
        gp = torch.cat(grid_profiles)[:32]
        gg = torch.cat(grid_generated)[:32]
        gf = torch.cat(grid_frontals)[:32]
        save_sample_grid(gp, gg, gf,
                         os.path.join(grid_dir, 'eval_grid.jpg'),
                         max_images=8, title='Evaluation (Profile | Generated | GT)')

    # Save CSV
    csv_path = os.path.join(results_dir, 'metrics.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    # Summary
    avgs = tracker.averages()
    summary = {
        'SSIM':     round(avgs.get('ssim', 0), 4),
        'PSNR':     round(avgs.get('psnr', 0), 2),
        'LPIPS':    round(avgs.get('lpips', 0), 4),
        'ID Score': round(avgs.get('id_score', 0), 4),
        'N_images': len(rows),
    }

    summary_path = os.path.join(results_dir, 'metrics_summary.txt')
    with open(summary_path, 'w') as f:
        f.write("Face Frontalization — Baseline Evaluation Results\n")
        f.write("=" * 52 + "\n")
        for k, v in summary.items():
            f.write(f"  {k:<12}: {v}\n")

    print(f"\n{'='*52}")
    print(f"  Evaluation Results")
    print(f"{'='*52}")
    for k, v in summary.items():
        print(f"  {k:<12}: {v}")
    print(f"{'='*52}")
    print(f"  Saved → {results_dir}\n")

    return summary


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Evaluate Face Frontalization Model')
    parser.add_argument('--checkpoint',   required=True,               type=str)
    parser.add_argument('--data_root',    default='data/300W_LP',     type=str)
    parser.add_argument('--img_size',     default=256,                 type=int)
    parser.add_argument('--batch_size',   default=4,                   type=int)
    parser.add_argument('--min_yaw',      default=15.0,                type=float)
    parser.add_argument('--val_fraction', default=0.1,                 type=float)
    parser.add_argument('--results_dir',  default='results/hybrid',    type=str)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.results_dir, exist_ok=True)

    infer_fn, pipeline = load_model(args.checkpoint, device)
    print(f"[Evaluate] Pipeline: {pipeline}")
    evaluate_paired(infer_fn, args.data_root, args.img_size, device, args.results_dir, args)


if __name__ == '__main__':
    main()
