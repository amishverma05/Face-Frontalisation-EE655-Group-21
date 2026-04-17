import os
import sys
import argparse
from pathlib import Path

import torch
from torchvision import transforms as T
from PIL import Image
import matplotlib.pyplot as plt

# Import the new hybrid architecture components
from models.generator import PoseConditionedEncoder
from models.stylegan2_wrapper import StyleGAN2Generator


def load_image(path: str, img_size: int = 256) -> torch.Tensor:
    """Load a PIL image and return (1, 3, H, W) tensor in [-1, 1]."""
    tf = T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize([0.5]*3, [0.5]*3),
    ])
    img = Image.open(path).convert('RGB')
    return tf(img).unsqueeze(0)


def tensor_to_pil(t: torch.Tensor) -> Image.Image:
    t = t.squeeze(0).detach().cpu().clamp(-1, 1)
    t = ((t + 1.0) / 2.0 * 255).to(torch.uint8)
    return Image.fromarray(t.permute(1, 2, 0).numpy())


def load_hybrid_model(ckpt_path: str, stylegan_ckpt: str, device):
    print("[Inference] Loading Hybrid Architecture...")
    
    # 1. Load the trainable Encoder
    ckpt = torch.load(ckpt_path, map_location=device)
    encoder = PoseConditionedEncoder(n_styles=14, style_dim=512).to(device)
    encoder.load_state_dict(ckpt['encoder'])
    encoder.eval()

    # 2. Load the frozen StyleGAN2 Generator
    stylegan = StyleGAN2Generator(stylegan_ckpt).to(device)
    stylegan.eval()

    # 3. Load the average face mathematical anchor
    avg_path = os.path.join(os.path.dirname(ckpt_path), 'frontal_latent_avg.pt')
    if os.path.exists(avg_path):
        w_avg = torch.load(avg_path, map_location=device)
        # w_avg is already (1, 14, 512)
    else:
        print(f"[Warning] w_avg not found at {avg_path}. Using zero anchor.")
        w_avg = torch.zeros(1, 14, 512, device=device)

    return encoder, stylegan, w_avg


@torch.no_grad()
def frontalize(encoder, stylegan, w_avg, image_path: str, yaw_deg: float, pitch_deg: float, device) -> Image.Image:
    tensor = load_image(image_path, 256).to(device)
    yaw_t = torch.tensor([yaw_deg], dtype=torch.float32, device=device)
    pitch_t = torch.tensor([pitch_deg], dtype=torch.float32, device=device)

    # 1. Extract geometric identity map
    w_pred = encoder(tensor, yaw_t, pitch_t, w_avg=w_avg)
    
    # 2. Render photorealistic face
    fake_img = stylegan(w_pred, input_is_latent=True)
    return tensor_to_pil(fake_img)


def save_side_by_side(input_path: str, output_img: Image.Image, save_path: str, yaw: float, pitch: float):
    input_img = Image.open(input_path).convert('RGB')
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))

    axes[0].imshow(input_img)
    axes[0].set_title(f'Input (Yaw: {yaw}°, Pitch: {pitch}°)', fontweight='bold', fontsize=11)
    axes[0].axis('off')

    axes[1].imshow(output_img)
    axes[1].set_title('Synthesized Frontal Matrix', fontweight='bold', fontsize=11)
    axes[1].axis('off')

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved High-Res Result -> {save_path}")


def main():
    parser = argparse.ArgumentParser(description='Hybrid Face Frontalization Inference')
    parser.add_argument('--image',      required=True, type=str, help='Path to your custom face image')
    parser.add_argument('--checkpoint', default='checkpoints/best.pth', type=str, help='Path to your output.pth')
    parser.add_argument('--stylegan',   default='checkpoints/ffhq-256-config-e.pt', type=str, help='Base StyleGAN weights')
    parser.add_argument('--yaw',        default=45.0, type=float, help='Estimated angle of the profile (e.g. 45 or 60)')
    parser.add_argument('--pitch',      default=0.0, type=float, help='Estimated vertical tilt of the face (e.g. 10 or -10)')
    parser.add_argument('--output',     default='results/inference', type=str, help='Output directory')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if not os.path.exists(args.checkpoint):
        print(f"ERROR: Checkpoint not found at {args.checkpoint}. You must complete at least 1 training epoch.")
        sys.exit(1)

    encoder, stylegan, w_avg = load_hybrid_model(args.checkpoint, args.stylegan, device)
    
    out_name = f"{Path(args.image).stem}_frontalized.jpg"
    out_path = os.path.join(args.output, out_name)

    print(f"\n[Inference] Mapping Face Data from: {args.image}")
    frontal_img = frontalize(encoder, stylegan, w_avg, args.image, args.yaw, args.pitch, device)
    
    save_side_by_side(args.image, frontal_img, out_path, args.yaw, args.pitch)


if __name__ == '__main__':
    main()
