"""
train.py
─────────
Main training script for the Hybrid Face Frontalization pipeline.

Architecture: Pose-Conditioned FPN Encoder → W+ latent → Frozen StyleGAN2

Usage
─────
  python train.py                          # use config.yaml defaults
  python train.py --cfg config.yaml
  python train.py --cfg config.yaml --epochs 30 --batch_size 4
  python train.py --resume checkpoints/epoch_015.pth

Training Phases
───────────────
  Phase 1 (epochs 1–15):  id + lpips + lmk + parse + wnorm + l2
  Phase 2 (epochs 16–30): id + lpips + lmk + wnorm + l2  (parse dropped)

Features
────────
  • Mixed-precision (FP16) training via torch.cuda.amp
  • Gradient accumulation (simulates effective batch of 16)
  • Linear LR decay after `decay_epoch`
  • Checkpoint save / resume
  • TensorBoard logging
  • Sample grid saved every N steps
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")
import time
import argparse
import yaml
import random
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from data    import FrontalizationDataset
from models  import (PoseConditionedEncoder, StyleGAN2Generator,
                     HybridFrontalizationLoss, PatchDiscriminator, LSGANLoss)
from utils   import (save_sample_grid, plot_training_curves,
                     compute_ssim, compute_psnr,
                     LPIPSMetric, IDScoreMetric, MetricTracker)


# ─────────────────────────────────────────────────────────────────────────────
# Config loading
# ─────────────────────────────────────────────────────────────────────────────

def load_config(path: str) -> dict:
    with open(path, encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    return cfg


def merge_cli(cfg: dict, args) -> dict:
    """Allow CLI overrides of config values."""
    if args.epochs:
        cfg['training']['epochs'] = args.epochs
    if args.batch_size:
        cfg['training']['batch_size'] = args.batch_size
    if args.img_size:
        cfg['data']['img_size'] = args.img_size
    if args.lr_g:
        cfg['training']['lr_g'] = args.lr_g
    return cfg


# ─────────────────────────────────────────────────────────────────────────────
# Seeding
# ─────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ─────────────────────────────────────────────────────────────────────────────
# LR scheduler  (constant → linear decay to 0)
# ─────────────────────────────────────────────────────────────────────────────

def get_scheduler(optimizer, total_epochs: int, decay_epoch: int,
                  warmup_epochs: int = 5):
    """
    Linear warmup for `warmup_epochs` epochs, then constant until `decay_epoch`,
    then linear decay to 0.  Warmup prevents the encoder from spiking into the
    blob attractor during the first few steps when gradients are noisiest.
    """
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs          # linear warmup
        if epoch < decay_epoch:
            return 1.0                                   # constant
        return max(0.0,
                   1.0 - (epoch - decay_epoch) /
                         (total_epochs - decay_epoch))  # linear decay
    return LambdaLR(optimizer, lr_lambda=lr_lambda)


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint helpers
# ─────────────────────────────────────────────────────────────────────────────

def save_checkpoint(state: dict, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)
    print(f"  [Checkpoint] Saved → {path}")


def load_checkpoint(path: str, encoder, opt_E, device, D=None, opt_D=None):
    ckpt = torch.load(path, map_location=device)
    encoder.load_state_dict(ckpt['encoder'])
    opt_E.load_state_dict(ckpt['opt_E'])
    if D is not None and 'discriminator' in ckpt:
        D.load_state_dict(ckpt['discriminator'])
        print("  [Checkpoint] Discriminator weights restored.")
    if opt_D is not None and 'opt_D' in ckpt:
        opt_D.load_state_dict(ckpt['opt_D'])
    start_epoch = ckpt.get('epoch', 0) + 1
    print(f"  [Checkpoint] Resumed from epoch {ckpt.get('epoch', 0)} → {path}")
    return start_epoch


# ─────────────────────────────────────────────────────────────────────────────
# Load frontal W+ latent average
# ─────────────────────────────────────────────────────────────────────────────

def load_w_avg(path: str, device, style_dim: int = 512, n_styles: int = 15):
    """
    Load precomputed frontal W+ average from setup_stylegan.py.
    Falls back to zeros if file doesn't exist yet.
    """
    if os.path.exists(path):
        w_avg = torch.load(path, map_location=device)
        print(f"  [Latent] Loaded frontal W+ avg from {path}  shape={tuple(w_avg.shape)}")
    else:
        print(f"  [Latent] WARNING: {path} not found. Using zero init.")
        print(f"           Run: python setup_stylegan.py  (one time, ~2 min)")
        w_avg = torch.zeros(1, n_styles, style_dim, device=device)
    return w_avg


# ─────────────────────────────────────────────────────────────────────────────
# Training epoch
# ─────────────────────────────────────────────────────────────────────────────

def train_epoch(encoder, G_frozen, D, loss_fn, opt_E, opt_D, loader,
                scaler, device, cfg, epoch,
                w_avg, writer, global_step, sample_dir):

    encoder.train()
    D.train()
    acc_steps       = cfg['training']['gradient_accumulation_steps']
    lam_adv         = cfg['loss'].get('lambda_adv', 0.1)
    disc_warmup_ep  = cfg['training'].get('disc_warmup_epoch', 20)
    use_disc        = (epoch >= disc_warmup_ep)   # discriminator gate
    adv_loss        = LSGANLoss()
    log             = defaultdict(float)
    pbar            = tqdm(loader, desc=f"Epoch {epoch:03d}", leave=False,
                           dynamic_ncols=True, miniters=5, mininterval=1.0)

    opt_E.zero_grad(set_to_none=True)

    for step, batch in enumerate(pbar):
        profile = batch['profile'].to(device, non_blocking=True)
        frontal = batch['frontal'].to(device, non_blocking=True)
        yaw     = batch['yaw'].to(device, non_blocking=True)
        pitch   = batch['pitch'].to(device, non_blocking=True)

        # ── Encoder forward (FP16) ───────────────────────────────────────
        with autocast('cuda', enabled=cfg['training']['mixed_precision']):
            w_pred       = encoder(profile, yaw, pitch, w_avg=w_avg)        # (B, 15, 512)
            
        # StyleGAN2 demodulation overflows in native FP16 (causing black NaN blobs).
        # We MUST evaluate the generator in FP32.
        fake_frontal = G_frozen(w_pred.float(), input_is_latent=True)       # (B, 3, 256, 256)

        # ── Discriminator step (skipped during warmup for stability) ────
        if use_disc:
            opt_D.zero_grad()
            pred_real = D(profile, frontal)
            pred_fake = D(profile, fake_frontal.detach())
            d_loss = 0.5 * (adv_loss(pred_real, True) + adv_loss(pred_fake, False))
            d_loss.backward()
            opt_D.step()
        else:
            d_loss = torch.tensor(0.0, device=device)

        # ── Generator / Encoder step ─────────────────────────────────────
        with autocast('cuda', enabled=cfg['training']['mixed_precision']):
            # Reconstruction + identity losses
            loss, loss_dict = loss_fn(
                generated=fake_frontal,
                target=frontal,
                w_pred=w_pred,
                w_avg=w_avg,
                epoch=epoch,
                frontal_lmks=None,
            )
            # Adversarial loss for generator (only after disc warmup)
            if use_disc:
                g_adv = adv_loss(D(profile, fake_frontal), True)
                loss  = loss + lam_adv * g_adv
                loss_dict['G_adv'] = g_adv.item()
            else:
                loss_dict['G_adv'] = 0.0
            loss_dict['D_loss']  = d_loss.item()
            loss_dict['G_total'] = loss.item()

            loss = loss / acc_steps

        scaler.scale(loss).backward()

        if (step + 1) % acc_steps == 0:
            scaler.unscale_(opt_E)
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=1.0)
            scaler.step(opt_E)
            scaler.update()
            opt_E.zero_grad(set_to_none=True)

        # ── Logging ──────────────────────────────────────────────────────
        for k, v in loss_dict.items():
            log[k] += v

        # Save sample grid every ~10% of epoch
        if (step + 1) % max(1, len(loader) // 10) == 0:
            with torch.no_grad():
                encoder.eval()
                with autocast('cuda', enabled=cfg['training']['mixed_precision']):
                    w_s = encoder(profile[:8], yaw[:8], pitch[:8], w_avg=w_avg)
                # Generate sample in FP32 to avoid NaN blobs
                fake_sample = G_frozen(w_s.float(), input_is_latent=True)
                encoder.train()
            path = os.path.join(sample_dir, f'epoch{epoch:03d}_step{step+1}.jpg')
            save_sample_grid(profile[:8], fake_sample, frontal[:8], path,
                             title=f'Epoch {epoch}, step {step+1}')

        global_step += 1
        phase_str = 'P1' if epoch < cfg['training'].get('phase2_start', 16) else 'P2'
        disc_str  = 'D-ON' if use_disc else f'D-off({disc_warmup_ep-epoch}ep)'
        pbar.set_postfix({'loss': f"{loss_dict.get('G_total', 0):.3f}",
                          'D':    f"{d_loss.item():.3f}",
                          'disc': disc_str,
                          'phase': phase_str})

    # Average over steps
    n   = len(loader)
    avg = {k: v / n for k, v in log.items()}

    for k, v in avg.items():
        writer.add_scalar(f'train/{k}', v, epoch)

    return avg, global_step


# ─────────────────────────────────────────────────────────────────────────────
# Validation epoch
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def validate(encoder, G_frozen, loader, device, cfg,
             w_avg, lpips_fn, id_fn, writer, epoch):
    encoder.eval()
    tracker = MetricTracker()

    for batch in tqdm(loader, desc='  Val', leave=False,
                      dynamic_ncols=True, miniters=5, mininterval=1.0):
        profile = batch['profile'].to(device)
        frontal = batch['frontal'].to(device)
        yaw     = batch['yaw'].to(device)
        pitch   = batch['pitch'].to(device)

        w_pred      = encoder(profile, yaw, pitch, w_avg=w_avg)
        gen         = G_frozen(w_pred, input_is_latent=True)

        ssim_v  = compute_ssim(gen, frontal)
        psnr_v  = compute_psnr(gen, frontal)
        lpips_v = lpips_fn(gen, frontal)
        id_v    = id_fn(gen, frontal)

        tracker.update(ssim=ssim_v, psnr=psnr_v, lpips=lpips_v, id_score=id_v)

    avgs = tracker.averages()
    for k, v in avgs.items():
        writer.add_scalar(f'val/{k}', v, epoch)

    encoder.train()
    return avgs


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Hybrid Face Frontalization Training')
    parser.add_argument('--cfg',        default='config.yaml',  type=str)
    parser.add_argument('--epochs',     default=None,           type=int)
    parser.add_argument('--batch_size', default=None,           type=int)
    parser.add_argument('--img_size',   default=None,           type=int)
    parser.add_argument('--lr_g',       default=None,           type=float)
    parser.add_argument('--resume',     default=None,           type=str,
                        help='Path to checkpoint to resume from')
    args = parser.parse_args()

    cfg = load_config(args.cfg)
    cfg = merge_cli(cfg, args)
    set_seed(42)

    # ── Device ──────────────────────────────────────────────────────────
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True  # Accelerate convolutions
    
    print(f"\n{'='*60}")
    print(f"  Face Frontalization — Hybrid Training")
    print(f"  Device : {device}")
    if device.type == 'cuda':
        total_vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"  GPU    : {torch.cuda.get_device_name(0)}  ({total_vram:.1f} GB)")
    print(f"  Config : {args.cfg}")
    print(f"{'='*60}\n")

    # ── Directories ─────────────────────────────────────────────────────
    ckpt_dir   = cfg['paths']['checkpoint_dir']
    log_dir    = cfg['paths']['log_dir']
    sample_dir = cfg['paths']['sample_dir']
    for d in [ckpt_dir, log_dir, sample_dir]:
        os.makedirs(d, exist_ok=True)

    # ── Data ────────────────────────────────────────────────────────────
    dc = cfg['data']
    dataset_train = FrontalizationDataset(
        root=dc['train_root'], img_size=dc['img_size'],
        min_yaw=dc['min_yaw'], max_yaw=dc['max_yaw'],
        augment=True, split='train', val_fraction=dc['val_split'])

    dataset_val = FrontalizationDataset(
        root=dc['train_root'], img_size=dc['img_size'],
        min_yaw=dc['min_yaw'], max_yaw=dc['max_yaw'],
        augment=False, split='val', val_fraction=dc['val_split'])

    tc = cfg['training']
    loader_train = DataLoader(
        dataset_train, batch_size=tc['batch_size'],
        shuffle=True, num_workers=tc['num_workers'],
        pin_memory=True, drop_last=True)

    loader_val = DataLoader(
        dataset_val, batch_size=tc['batch_size'],
        shuffle=False, num_workers=tc['num_workers'],
        pin_memory=True)

    # ── Models ──────────────────────────────────────────────────────────
    mc = cfg['model']

    # Frozen StyleGAN2 Generator
    stylegan_ckpt = mc.get('stylegan_ckpt', None)
    G_frozen = StyleGAN2Generator(
        ckpt_path=stylegan_ckpt if (stylegan_ckpt and os.path.exists(stylegan_ckpt)) else None,
        style_dim=mc.get('style_dim', 512),
    ).to(device)
    G_frozen.eval()
    print(f"  StyleGAN2 params (frozen) : {sum(p.numel() for p in G_frozen.parameters()):,}")

    # Trainable Pose-Conditioned FPN Encoder
    encoder = PoseConditionedEncoder(
        pose_dim=mc.get('pose_dim', 256),
        style_dim=mc.get('style_dim', 512),
        n_styles=mc.get('n_styles', 18),
        pretrained=True,
    ).to(device)
    print(f"  Encoder params (trainable): {encoder.count_params():,}\n")

    # PatchGAN Discriminator (profile+frontal → patch logits)
    D = PatchDiscriminator(in_ch=3, ndf=64).to(device)
    print(f"  Discriminator params      : {D.count_params():,}\n")

    # ── Frontal W+ average ───────────────────────────────────────────────
    w_avg = load_w_avg(
        mc.get('frontal_latent_avg', 'checkpoints/frontal_latent_avg.pt'),
        device,
        style_dim=mc.get('style_dim', 512),
        n_styles=mc.get('n_styles', 18),
    )
    # Keep w_avg as non-parameter constant
    w_avg = w_avg.detach()

    # ── Losses ──────────────────────────────────────────────────────────
    lc      = cfg['loss']
    loss_fn = HybridFrontalizationLoss(
        lambda_id    = lc.get('lambda_id',    1.5),
        lambda_lpips = lc.get('lambda_lpips', 0.8),
        lambda_perc  = lc.get('lambda_perc',  1.0),
        lambda_lmk   = lc.get('lambda_lmk',   0.0),
        lambda_parse = lc.get('lambda_parse', 0.0),
        lambda_wnorm = lc.get('lambda_wnorm', 0.005),
        lambda_l2    = lc.get('lambda_l2',    2.0),
        phase2_start = tc.get('phase2_start', 51),
    ).to(device)

    # ── Optimizer (encoder only — G_frozen has no grads) ────────────────
    opt_E = Adam(encoder.parameters(), lr=tc['lr_g'],
                 betas=(tc['beta1'], tc['beta2']))
    opt_D = Adam(D.parameters(), lr=tc['lr_g'] * 0.5,
                 betas=(0.5, 0.999))          # standard GAN discriminator betas
    sched_E = get_scheduler(opt_E, tc['epochs'], tc['decay_epoch'], warmup_epochs=5)
    sched_D = get_scheduler(opt_D, tc['epochs'], tc['decay_epoch'], warmup_epochs=5)
    scaler  = GradScaler('cuda', enabled=tc['mixed_precision'])

    # ── Resume ──────────────────────────────────────────────────────────
    start_epoch = 1
    if args.resume:
        start_epoch = load_checkpoint(args.resume, encoder, opt_E, device,
                                      D=D, opt_D=opt_D)

    # ── Logger + Metrics ────────────────────────────────────────────────
    writer   = SummaryWriter(log_dir=log_dir)
    lpips_fn = LPIPSMetric(net='alex')
    id_fn    = IDScoreMetric()

    # ── Training Loop ────────────────────────────────────────────────────
    global_step = 0
    best_ssim   = 0.0
    epoch_log   = defaultdict(list)

    print(f"  Starting hybrid training for {tc['epochs']} epochs ...")
    print(f"  Phase 1: epochs 1–{tc.get('phase2_start',16)-1}  "
          f"(id+lpips+lmk+parse+wnorm+l2)")
    print(f"  Phase 2: epochs {tc.get('phase2_start',16)}–{tc['epochs']}  "
          f"(id+lpips+lmk+wnorm+l2)\n")

    for epoch in range(start_epoch, tc['epochs'] + 1):
        t0 = time.time()

        # Train
        train_losses, global_step = train_epoch(
            encoder, G_frozen, D, loss_fn, opt_E, opt_D, loader_train,
            scaler, device, cfg, epoch,
            w_avg, writer, global_step, sample_dir)

        # Validate every 5 epochs (or first 3 epochs)
        if epoch <= 3 or epoch % 5 == 0:
            val_metrics = validate(
                encoder, G_frozen, loader_val, device, cfg,
                w_avg, lpips_fn, id_fn, writer, epoch)
        else:
            val_metrics = {}

        # LR step
        sched_E.step()
        sched_D.step()

        # Log epoch summary
        elapsed = time.time() - t0
        phase = 'P1' if epoch < tc.get('phase2_start', 16) else 'P2'
        print(f"  Epoch {epoch:03d}/{tc['epochs']:03d} [{phase}]  "
              f"| Loss={train_losses.get('G_total', 0):.3f} "
              f"| SSIM={val_metrics.get('ssim', 0):.4f} "
              f"| LPIPS={val_metrics.get('lpips', 0):.4f} "
              f"| ID={val_metrics.get('id_score', 0):.4f} "
              f"| {elapsed:.0f}s")

        for k, v in {**train_losses, **val_metrics}.items():
            epoch_log[k].append(v)

        # Save checkpoints
        state = {
            'epoch':         epoch,
            'encoder':       encoder.state_dict(),
            'discriminator': D.state_dict(),
            'opt_E':         opt_E.state_dict(),
            'opt_D':         opt_D.state_dict(),
            'cfg':           cfg,
            'w_avg':         w_avg,
        }
        if epoch % 5 == 0 or epoch == tc['epochs']:
            save_checkpoint(state, os.path.join(ckpt_dir, f'epoch_{epoch:03d}.pth'))

        # Save best checkpoint
        if val_metrics.get('ssim', 0) > best_ssim:
            best_ssim = val_metrics['ssim']
            save_checkpoint(state, os.path.join(ckpt_dir, 'best.pth'))
            print(f"  ★ New best SSIM: {best_ssim:.4f}")

    # Final plots
    plot_training_curves(epoch_log, os.path.join(log_dir, 'training_curves.png'))
    writer.close()

    print(f"\n{'='*60}")
    print(f"  Training complete!  Best SSIM = {best_ssim:.4f}")
    print(f"  Checkpoints : {ckpt_dir}")
    print(f"  Samples     : {sample_dir}")
    print(f"  Logs        : {log_dir}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
