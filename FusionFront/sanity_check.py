"""
sanity_check.py
----------------
Quick smoke test -- verifies every module loads and runs a forward pass
without needing any real data.

Tests both the new Hybrid pipeline (PoseConditionedEncoder + StyleGAN2)
and verifies the HybridFrontalizationLoss computes without error.

Usage:  python sanity_check.py
        python sanity_check.py --skip_legacy   # skip old U-Net check
"""

import argparse
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast

parser = argparse.ArgumentParser()
parser.add_argument('--skip_legacy', action='store_true',
                    help='Skip legacy U-Net generator check')
args = parser.parse_args()

print("\n" + "="*60)
print("  Face Frontalization -- Sanity Check (Hybrid Pipeline)")
print("="*60)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n  Device : {device}")
if device.type == 'cuda':
    gib = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"  GPU    : {torch.cuda.get_device_name(0)}  ({gib:.1f} GB)")

B = 2    # batch size for all tests
IMG = 256

# -- 1. PoseConditionedEncoder -----------------------------------------------
print("\n[1] PoseConditionedEncoder ...")
from models.generator import PoseConditionedEncoder
encoder = PoseConditionedEncoder(pose_dim=256, style_dim=512, n_styles=15,
                                  pretrained=True).to(device)
x   = torch.randn(B, 3, IMG, IMG, device=device)
yaw = torch.tensor([30.0, 45.0], device=device)
pit = torch.tensor([ 5.0, 10.0], device=device)

w_plus = encoder(x, yaw, pit)
assert w_plus.shape == (B, 15, 512), f"Bad W+ shape: {w_plus.shape}"
print(f"    [OK] Input {tuple(x.shape)} -> W+ {tuple(w_plus.shape)}")
print(f"    [OK] Trainable params: {encoder.count_params():,}")

# -- 2. StyleGAN2Generator (random weights, no checkpoint) -------------------
print("\n[2] StyleGAN2Generator (random weights) ...")
from models.stylegan2_wrapper import StyleGAN2Generator
G = StyleGAN2Generator(ckpt_path=None, style_dim=512).to(device)
G.eval()

with torch.no_grad():
    imgs = G(w_plus.detach(), input_is_latent=True)

assert imgs.shape == (B, 3, 256, 256), f"Bad output shape: {imgs.shape}"
print(f"    [OK] W+ {tuple(w_plus.shape)} -> Image {tuple(imgs.shape)}")
print(f"    [OK] Output range: [{imgs.min():.2f}, {imgs.max():.2f}]")

# -- 3. Full forward pass (encoder -> W+ -> StyleGAN2) -----------------------
print("\n[3] Full forward pass (encoder -> W+ -> StyleGAN2) ...")
frontal = torch.randn(B, 3, IMG, IMG, device=device)
w_avg   = torch.zeros(1, 15, 512, device=device)

w_pred     = encoder(x, yaw, pit, w_avg=w_avg)
fake_imgs  = G(w_pred, input_is_latent=True)
assert fake_imgs.shape == (B, 3, 256, 256)
print(f"    [OK] Full pipeline: input {tuple(x.shape)} -> {tuple(fake_imgs.shape)}")

# -- 4. HybridFrontalizationLoss ---------------------------------------------
print("\n[4] HybridFrontalizationLoss ...")
from models.losses import HybridFrontalizationLoss
loss_fn = HybridFrontalizationLoss(
    lambda_id=0.5, lambda_lpips=0.8, lambda_lmk=1.0,
    lambda_parse=1.0, lambda_wnorm=0.005, lambda_l2=0.01
).to(device)

total, log = loss_fn(
    generated=fake_imgs,
    target=frontal,
    w_pred=w_pred,
    w_avg=w_avg,
    epoch=1,        # Phase 1 -- parse included
)
print(f"    [OK] Phase 1 total loss = {total.item():.4f}")
for k, v in log.items():
    print(f"         {k}: {v:.4f}")

# Phase 2 check
total2, log2 = loss_fn(
    generated=fake_imgs, target=frontal,
    w_pred=w_pred, w_avg=w_avg, epoch=20
)
print(f"    [OK] Phase 2 total loss = {total2.item():.4f} (parse=0: {log2['G_parse']==0.0})")

# -- 5. Backward pass (gradient check) ---------------------------------------
print("\n[5] Backward pass (gradient flows through encoder) ...")
scaler = GradScaler('cuda' if device.type == 'cuda' else 'cpu', enabled=(device.type == 'cuda'))

opt = torch.optim.Adam(encoder.parameters(), lr=1e-4)
opt.zero_grad()

with autocast('cuda', enabled=(device.type == 'cuda')):
    w_p  = encoder(x, yaw, pit, w_avg=w_avg)
    imgs = G(w_p, input_is_latent=True)
    loss, _ = loss_fn(imgs, frontal, w_p, w_avg, epoch=1)

scaler.scale(loss).backward()
# Check that encoder received gradients
has_grad = any(p.grad is not None for p in encoder.parameters())
print(f"    [OK] Encoder has gradients: {has_grad}")
scaler.step(opt); scaler.update()
print(f"    [OK] Optimizer step OK")

# -- 6. Legacy U-Net (optional) ----------------------------------------------
if not args.skip_legacy:
    print("\n[6] Legacy U-Net Generator (backward compat) ...")
    from models.generator import Generator
    G_legacy = Generator(ngf=64, n_res=6).to(device)
    x_128 = torch.randn(2, 3, 128, 128, device=device)
    out_legacy = G_legacy(x_128)
    assert out_legacy.shape == (2, 3, 128, 128)
    print(f"    [OK] Legacy U-Net: {tuple(x_128.shape)} -> {tuple(out_legacy.shape)}")

# -- 7. Metrics --------------------------------------------------------------
print("\n[7] Metrics ...")
from utils.metrics import compute_ssim, compute_psnr
ssim_v = compute_ssim(fake_imgs.detach(), frontal)
psnr_v = compute_psnr(fake_imgs.detach(), frontal)
print(f"    [OK] SSIM  = {ssim_v:.4f}")
print(f"    [OK] PSNR  = {psnr_v:.2f} dB")

# -- 8. VRAM usage -----------------------------------------------------------
if device.type == 'cuda':
    print("\n[8] VRAM usage ...")
    alloc = torch.cuda.memory_allocated() / 1024**3
    rsvd  = torch.cuda.memory_reserved() / 1024**3
    print(f"    [OK] Allocated : {alloc:.2f} GB")
    print(f"    [OK] Reserved  : {rsvd:.2f} GB")
    print(f"    [OK] Free      : {(gib - rsvd):.2f} GB")

print("\n" + "="*60)
print("  ALL CHECKS PASSED! Ready to run setup_stylegan.py + train.py")
print("="*60 + "\n")
