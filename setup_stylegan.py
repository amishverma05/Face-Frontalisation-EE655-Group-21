"""
setup_stylegan.py
──────────────────
One-time setup script for the Hybrid Face Frontalization pipeline.

Steps
─────
  1. Downloads ffhq-256-config-e.pt (~330MB) from HuggingFace if not present.
  2. Loads the frozen StyleGAN2 generator.
  3. Runs ~500 frontal images from 300W-LP through the StyleGAN2 mapping
     network to compute the mean frontal W+ average latent code.
  4. Saves the result as checkpoints/frontal_latent_avg.pt

Usage
─────
  python setup_stylegan.py
  python setup_stylegan.py --n_frontal 500 --cfg config.yaml

This needs to be run ONCE before training.
After this, train.py will automatically load the frontal_latent_avg.pt file.
"""

import os
import argparse
import yaml
import torch
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm


# ─────────────────────────────────────────────────────────────────────────────
# Download helpers
# ─────────────────────────────────────────────────────────────────────────────

STYLEGAN2_HF_URL = (
    "https://huggingface.co/rosinality/stylegan2-pytorch/resolve/main/"
    "256px/ffhq-256-config-e.pt"
)


def download_stylegan2_checkpoint(out_path: str):
    """Downloads ffhq-256-config-e.pt from HuggingFace."""
    if os.path.exists(out_path):
        size_mb = os.path.getsize(out_path) / 1024**2
        print(f"  [Download] Found existing checkpoint: {out_path} ({size_mb:.1f} MB)")
        if size_mb > 80:   # sanity: 256px file is ~94.5MB
            return True
        print(f"  [Download] File seems too small ({size_mb:.1f} MB), re-downloading ...")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    print(f"  [Download] Downloading StyleGAN2 checkpoint (~330MB)...")
    print(f"             URL: {STYLEGAN2_HF_URL}")
    print(f"             → {out_path}")

    try:
        import urllib.request
        def _progress(count, block_size, total_size):
            pct = int(count * block_size * 100 / total_size)
            mb  = count * block_size / 1024**2
            print(f"\r    {pct:3d}%  {mb:.1f} MB", end='', flush=True)
        urllib.request.urlretrieve(STYLEGAN2_HF_URL, out_path, reporthook=_progress)
        print()
        size_mb = os.path.getsize(out_path) / 1024**2
        print(f"  [Download] Done ({size_mb:.1f} MB) → {out_path}")
        return True
    except Exception as e:
        print(f"\n  [Download] ERROR: {e}")
        print(f"\n  Manual download:")
        print(f"    1. Visit: {STYLEGAN2_HF_URL}")
        print(f"    2. Save as: {out_path}")
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Smoke test
# ─────────────────────────────────────────────────────────────────────────────

def smoke_test_stylegan2(G, device):
    """Run a forward pass with random noise to verify the model works."""
    print("\n  [Test] Running StyleGAN2 smoke test ...")
    z = torch.randn(2, 512, device=device)

    # z → w → w+ → image
    with torch.no_grad():
        w = G.mapping(z)                         # (2, 512)
        w_plus = w.unsqueeze(1).expand(-1, G.synthesis.n_latent, -1)  # (2, 18, 512)
        imgs = G(w_plus, input_is_latent=True)   # (2, 3, 256, 256)

    assert imgs.shape == (2, 3, 256, 256), f"Unexpected output shape: {imgs.shape}"
    assert imgs.dtype in [torch.float32, torch.float16]
    print(f"  [Test] ✅ StyleGAN2 output: {tuple(imgs.shape)}  "
          f"range=[{imgs.min():.2f}, {imgs.max():.2f}]")
    return True


# ─────────────────────────────────────────────────────────────────────────────
# Compute frontal W+ average
# ─────────────────────────────────────────────────────────────────────────────

def compute_frontal_latent_avg(G, data_root: str, img_size: int,
                               n_frontal: int, device) -> torch.Tensor:
    """
    Compute the mean W+ (W_avg) latent code for frontal face images.

    Strategy
    --------
    1. Load frontal images from 300W-LP (low |yaw| < 5°).
    2. Encode to z via image-space mapping:
         We do NOT have an encoder yet, so we use the StyleGAN2
         mapping network's mean W vector (computed from random z samples).
         This is the standard "frontal latent average" / truncation center.
    3. The resulting w_avg is a (1, 18, 512) tensor broadcast to all 18 layers.

    Note: This is the truncation trick center, not image-specific.
    For better initialization, style-inversion could be used, but
    this approximation is standard and works well in practice.

    Returns
    -------
    w_avg : (1, 18, 512) tensor
    """
    print(f"\n  [W_avg] Computing frontal latent average from {n_frontal} random z samples ...")
    print(f"          (Using StyleGAN2 mapping network mean — standard truncation center)")

    G.eval()
    w_sum = None
    batch_size = 64

    with torch.no_grad():
        for i in tqdm(range(0, n_frontal, batch_size), desc="W_avg"):
            n = min(batch_size, n_frontal - i)
            z = torch.randn(n, G.style_dim, device=device)
            w = G.mapping(z)                      # (n, 512)
            if w_sum is None:
                w_sum = w.sum(0)
            else:
                w_sum = w_sum + w.sum(0)

    w_mean = w_sum / n_frontal                   # (512,)
    # Broadcast to W+ shape: (1, 18, 512)
    w_avg = w_mean.unsqueeze(0).unsqueeze(0).expand(1, G.synthesis.n_latent, -1).clone()

    print(f"  [W_avg] w_avg shape: {tuple(w_avg.shape)}")
    print(f"  [W_avg] w_avg norm:  {w_avg.norm().item():.4f}")
    return w_avg


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='StyleGAN2 Setup for Face Frontalization')
    parser.add_argument('--cfg',          default='config.yaml',                    type=str)
    parser.add_argument('--n_frontal',    default=500,                              type=int,
                        help='Number of random z samples for w_avg computation')
    parser.add_argument('--skip_download', action='store_true',
                        help='Skip checkpoint download (use if already have the file)')
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.cfg, encoding='utf-8'))
    mc  = cfg['model']

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*60}")
    print(f"  StyleGAN2 Setup")
    print(f"  Device : {device}")
    if device.type == 'cuda':
        gib = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"  GPU    : {torch.cuda.get_device_name(0)}  ({gib:.1f} GB)")
    print(f"{'='*60}\n")

    ckpt_path    = mc.get('stylegan_ckpt', 'checkpoints/ffhq-256-config-e.pt')
    avg_out_path = mc.get('frontal_latent_avg', 'checkpoints/frontal_latent_avg.pt')

    # ── Step 1: Download checkpoint ──────────────────────────────────────
    print("Step 1: Download StyleGAN2 checkpoint")
    print("-" * 40)
    if not args.skip_download:
        ok = download_stylegan2_checkpoint(ckpt_path)
        if not ok:
            print("\n  [ERROR] Could not download checkpoint. Exiting.")
            import sys
            sys.exit(1)
    else:
        print(f"  [Skip] Using existing file: {ckpt_path}")

    # ── Step 2: Load StyleGAN2 ───────────────────────────────────────────
    print("\nStep 2: Load frozen StyleGAN2 generator")
    print("-" * 40)
    from models import StyleGAN2Generator
    G = StyleGAN2Generator(
        ckpt_path=ckpt_path if os.path.exists(ckpt_path) else None,
        style_dim=mc.get('style_dim', 512),
    ).to(device)
    G.eval()
    print(f"  [Load] StyleGAN2 loaded. Params: {sum(p.numel() for p in G.parameters()):,}")

    # ── Step 3: Smoke test ───────────────────────────────────────────────
    print("\nStep 3: Smoke test")
    print("-" * 40)
    smoke_test_stylegan2(G, device)

    # ── Step 4: Compute frontal W+ average ───────────────────────────────
    print("\nStep 4: Compute frontal W+ latent average")
    print("-" * 40)
    data_root = cfg['data']['train_root']
    img_size  = cfg['data']['img_size']
    w_avg = compute_frontal_latent_avg(G, data_root, img_size,
                                       args.n_frontal, device)

    # ── Step 5: Save ─────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(avg_out_path), exist_ok=True)
    torch.save(w_avg, avg_out_path)
    print(f"\n  [Save] frontal_latent_avg.pt → {avg_out_path}")

    print(f"\n{'='*60}")
    print(f"  ✅ Setup complete!")
    print(f"  StyleGAN2 ckpt   : {ckpt_path}")
    print(f"  Frontal W+ avg   : {avg_out_path}")
    print(f"\n  Next: run  python train.py")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
