"""
utils/visualization.py
───────────────────────
Helpers for saving result images during training and evaluation.
"""

import os
import math
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Force non-interactive backend — prevents Tkinter crash on Windows
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from torchvision.utils import make_grid
from PIL import Image


def tensor_to_pil(t: torch.Tensor) -> Image.Image:
    """Convert (3,H,W) tensor in [-1,1] to PIL RGB image."""
    t = t.detach().cpu().clamp(-1, 1)
    t = ((t + 1.0) / 2.0 * 255).to(torch.uint8)
    return Image.fromarray(t.permute(1, 2, 0).numpy())


def save_sample_grid(profiles: torch.Tensor,
                     generated: torch.Tensor,
                     targets:   torch.Tensor,
                     save_path: str,
                     max_images: int = 8,
                     title: str = ''):
    """
    Save a 3-row grid:  top=profile | middle=generated | bottom=GT frontal

    Parameters
    ----------
    profiles, generated, targets : (B,3,H,W) tensors in [-1,1]
    save_path : full path including filename
    max_images: how many columns to show
    """
    n = min(max_images, profiles.shape[0])
    rows = torch.cat([
        profiles[:n],    # row 1: input profile
        generated[:n],   # row 2: generated frontal
        targets[:n],     # row 3: ground truth frontal
    ], dim=0)           # (3n, 3, H, W)

    grid = make_grid(
        (rows.clamp(-1, 1) + 1.0) / 2.0,   # → [0,1]
        nrow=n,
        padding=2,
        normalize=False,
    )                                         # (3, 3H+pad, nW+pad)

    grid_np = grid.permute(1, 2, 0).cpu().numpy()

    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    fig, ax = plt.subplots(figsize=(n * 2, 7))
    ax.imshow(grid_np)
    ax.axis('off')

    # Row labels
    h = grid_np.shape[0]
    row_h = h // 3
    for i, label in enumerate(['Profile (Input)', 'Generated Frontal', 'GT Frontal']):
        ax.text(-5, int(row_h * i + row_h * 0.5),
                label, va='center', ha='right',
                fontsize=9, fontweight='bold', rotation=90,
                transform=ax.transData)

    if title:
        ax.set_title(title, fontsize=11)

    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close()


def plot_training_curves(log_dict: dict, save_path: str):
    """
    Plot generator and discriminator losses over training steps.

    Parameters
    ----------
    log_dict  : {metric_name: [values]}
    save_path : path to save the figure
    """
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)

    g_keys = [k for k in log_dict if k.startswith('G_')]
    d_keys = [k for k in log_dict if k.startswith('D_')]
    m_keys = [k for k in log_dict if k in ('ssim', 'psnr', 'lpips', 'id_score')]

    n_plots = int(bool(g_keys)) + int(bool(d_keys)) + int(bool(m_keys))
    if n_plots == 0:
        return

    fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 4))
    if n_plots == 1:
        axes = [axes]

    ax_idx = 0
    if g_keys:
        ax = axes[ax_idx]; ax_idx += 1
        for k in g_keys:
            ax.plot(log_dict[k], label=k)
        ax.set_title('Generator Losses'); ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    if d_keys:
        ax = axes[ax_idx]; ax_idx += 1
        for k in d_keys:
            ax.plot(log_dict[k], label=k, linestyle='--')
        ax.set_title('Discriminator Losses'); ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    if m_keys:
        ax = axes[ax_idx]; ax_idx += 1
        for k in m_keys:
            ax.plot(log_dict[k], label=k)
        ax.set_title('Eval Metrics'); ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=110, bbox_inches='tight')
    plt.close()


def save_comparison_strip(profile_path: str,
                           generated: torch.Tensor,
                           output_path: str,
                           gt_path:  str = None):
    """
    Save a single-sample comparison strip for a demo / paper figure.
    [Profile] → [Generated Frontal] → [GT (if available)]
    """
    profile_img = Image.open(profile_path).convert('RGB')
    gen_img     = tensor_to_pil(generated.squeeze(0))

    images = [profile_img, gen_img]
    labels = ['Profile (Input)', 'Generated Frontal']
    if gt_path and os.path.exists(gt_path):
        images.append(Image.open(gt_path).convert('RGB'))
        labels.append('Ground Truth')

    n = len(images)
    w, h = images[0].size
    fig, axes = plt.subplots(1, n, figsize=(n * 3, 3.5))
    if n == 1:
        axes = [axes]

    for ax, img, lbl in zip(axes, images, labels):
        ax.imshow(img)
        ax.set_title(lbl, fontsize=10, fontweight='bold')
        ax.axis('off')

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
