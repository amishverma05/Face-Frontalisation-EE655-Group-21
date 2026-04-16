"""
inference.py
─────────────
Run a trained frontalization model on a single image or a folder.
No dataset annotations required — just raw face images.

Usage
─────
  # Single image
  python inference.py --checkpoint checkpoints/best.pth --input face.jpg

  # Folder of images
  python inference.py --checkpoint checkpoints/best.pth --input images/ --output results/
"""

import os
import sys
import argparse
import glob
from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision import transforms as T
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from models import Generator


# ─────────────────────────────────────────────────────────────────────────────
# Image I/O
# ─────────────────────────────────────────────────────────────────────────────

TRANSFORM = T.Compose([
    T.Resize((128, 128)),
    T.ToTensor(),
    T.Normalize([0.5]*3, [0.5]*3),
])


def load_image(path: str, img_size: int = 128) -> torch.Tensor:
    """Load a PIL image and return (1, 3, H, W) tensor in [-1, 1]."""
    tf = T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize([0.5]*3, [0.5]*3),
    ])
    img = Image.open(path).convert('RGB')
    return tf(img).unsqueeze(0)   # (1,3,H,W)


def tensor_to_pil(t: torch.Tensor) -> Image.Image:
    t = t.squeeze(0).detach().cpu().clamp(-1, 1)
    t = ((t + 1.0) / 2.0 * 255).to(torch.uint8)
    return Image.fromarray(t.permute(1, 2, 0).numpy())


# ─────────────────────────────────────────────────────────────────────────────
# Load model
# ─────────────────────────────────────────────────────────────────────────────

def load_model(ckpt_path: str, device) -> tuple:
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg  = ckpt.get('cfg', {})
    mc   = cfg.get('model', {'ngf': 64, 'num_resblocks': 6})
    dc   = cfg.get('data',  {'img_size': 128})

    G = Generator(ngf=mc['ngf'], n_res=mc['num_resblocks']).to(device)
    G.load_state_dict(ckpt['G'])
    G.eval()
    return G, dc.get('img_size', 128)


# ─────────────────────────────────────────────────────────────────────────────
# Single image inference
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def frontalize(G, image_path: str, device, img_size: int = 128) -> Image.Image:
    tensor = load_image(image_path, img_size).to(device)
    output = G(tensor)
    return tensor_to_pil(output)


def save_side_by_side(input_path: str, output_img: Image.Image,
                      save_path: str, title: str = ''):
    input_img = Image.open(input_path).convert('RGB')
    fig, axes = plt.subplots(1, 2, figsize=(7, 3.5))

    axes[0].imshow(input_img)
    axes[0].set_title('Input (Profile)', fontweight='bold', fontsize=11)
    axes[0].axis('off')

    axes[1].imshow(output_img)
    axes[1].set_title('Generated Frontal', fontweight='bold', fontsize=11)
    axes[1].axis('off')

    if title:
        fig.suptitle(title, fontsize=12, y=1.01)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved → {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Face Frontalization Inference')
    parser.add_argument('--checkpoint', required=True,           type=str,
                        help='Path to trained checkpoint (.pth)')
    parser.add_argument('--input',      required=True,           type=str,
                        help='Input image path or folder')
    parser.add_argument('--output',     default='results/inference', type=str,
                        help='Output directory')
    parser.add_argument('--img_size',   default=None,            type=int,
                        help='Override image size (default: from checkpoint)')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[Inference] Device: {device}")

    G, img_size = load_model(args.checkpoint, device)
    if args.img_size:
        img_size = args.img_size

    print(f"[Inference] Image size: {img_size}×{img_size}")

    os.makedirs(args.output, exist_ok=True)

    # Collect input images
    input_path = Path(args.input)
    if input_path.is_file():
        image_paths = [str(input_path)]
    elif input_path.is_dir():
        image_paths = (glob.glob(str(input_path / '*.jpg')) +
                       glob.glob(str(input_path / '*.png')) +
                       glob.glob(str(input_path / '*.jpeg')))
    else:
        print(f"ERROR: input not found: {args.input}")
        sys.exit(1)

    print(f"[Inference] Processing {len(image_paths)} image(s) ...\n")

    for img_path in image_paths:
        stem     = Path(img_path).stem
        out_name = f"{stem}_frontal.jpg"
        out_comp = f"{stem}_comparison.jpg"

        # Run model
        frontal_img = frontalize(G, img_path, device, img_size)

        # Save frontal only
        frontal_img.save(os.path.join(args.output, out_name))

        # Save side-by-side comparison
        save_side_by_side(
            img_path, frontal_img,
            os.path.join(args.output, out_comp),
            title=f'Face Frontalization — {stem}')

    print(f"\n[Done] Results saved to: {args.output}")


if __name__ == '__main__':
    main()
