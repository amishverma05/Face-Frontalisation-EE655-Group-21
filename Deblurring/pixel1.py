from google.colab import drive
drive.mount('/content/drive')

import os
ROOT = '/content/drive/MyDrive/datasets'
TRAIN_INPUT_DIR  = os.path.join(ROOT, 'train/input')
TRAIN_TARGET_DIR = os.path.join(ROOT, 'train/target')
TEST_INPUT_DIR   = os.path.join(ROOT, 'test/input')
TEST_TARGET_DIR  = os.path.join(ROOT, 'test/target')
CHECKPOINT_DIR = '/content/drive/MyDrive/realesrgan_ckpt'
RESULTS_DIR    = '/content/drive/MyDrive/realesrgan_results'
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# %cd /content
!git clone https://github.com/xinntao/Real-ESRGAN.git
# %cd Real-ESRGAN
!pip install basicsr facexlib gfpgan -q
!pip install -r requirements.txt -q
!python setup.py develop -q

!mkdir -p experiments/pretrained_models
!wget -q https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth \
    -O experiments/pretrained_models/RealESRGAN_x4plus.pth
!wget -q https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.3/RealESRGAN_x4plus_netD.pth \
    -O experiments/pretrained_models/RealESRGAN_x4plus_netD.pth

META_FILE = '/content/meta.txt'
with open(META_FILE, 'w') as f:
    for name in sorted(os.listdir(TRAIN_INPUT_DIR)):
        inp = os.path.join(TRAIN_INPUT_DIR, name)
        tgt = os.path.join(TRAIN_TARGET_DIR, name)
        if os.path.exists(tgt):
            f.write(f"{tgt}, {inp}\n")

config = """name: finetune
model_type: RealESRGANModel
scale: 1
num_gpu: 1

datasets:
  train:
    name: train
    type: RealESRGANPairedDataset
    dataroot_gt: /content/drive/MyDrive/datasets/train/target
    dataroot_lq: /content/drive/MyDrive/datasets/train/input
    meta_info: /content/meta.txt
    io_backend:
      type: disk
    gt_size: 128
    use_hflip: true
    use_rot: true
    batch_size_per_gpu: 2
    num_worker_per_gpu: 2
    use_shuffle: true
    dataset_enlarge_ratio: 1
  val:
    name: val
    type: PairedImageDataset
    dataroot_gt: /content/drive/MyDrive/datasets/test/target
    dataroot_lq: /content/drive/MyDrive/datasets/test/input
    io_backend:
      type: disk

network_g:
  type: RRDBNet
  num_in_ch: 3
  num_out_ch: 3
  num_feat: 64
  num_block: 23
  num_grow_ch: 32
  scale: 1

network_d:
  type: UNetDiscriminatorSN
  num_in_ch: 3
  num_feat: 64

path:
  pretrain_network_g: /content/Real-ESRGAN/experiments/pretrained_models/RealESRGAN_x4plus.pth
  pretrain_network_d: /content/Real-ESRGAN/experiments/pretrained_models/RealESRGAN_x4plus_netD.pth
  param_key_g: params_ema
  strict_load_g: false

train:
  total_iter: 5000
  warmup_iter: -1
  optim_g:
    type: Adam
    lr: 0.0001
  optim_d:
    type: Adam
    lr: 0.0001
  scheduler:
    type: MultiStepLR
    milestones: [2500, 4000]
    gamma: 0.5
  pixel_opt:
    type: L1Loss
    loss_weight: 1.0
  gan_opt:
    type: GANLoss
    gan_type: vanilla
    loss_weight: 0.1
    real_label_val: 1.0
    fake_label_val: 0.0
  net_d_iters: 1
  net_d_init_iters: 0
"""
with open('/content/config.yml', 'w') as f:
    f.write(config)

!rm -rf /content/Real-ESRGAN/experiments/finetune

import glob
file = glob.glob('/usr/local/lib/python*/dist-packages/basicsr/data/degradations.py')[0]
with open(file, 'r') as f:
    content = f.read()
content = content.replace(
    'from torchvision.transforms.functional_tensor import rgb_to_grayscale',
    'from torchvision.transforms.functional import rgb_to_grayscale'
)
with open(file, 'w') as f:
    f.write(content)

# %cd /content/Real-ESRGAN
!python realesrgan/train.py -opt /content/config.yml

import shutil
shutil.copy(
    '/content/Real-ESRGAN/experiments/finetune/models/net_g_latest.pth',
    '/content/drive/MyDrive/realesrgan_final_model.pth'
)

import cv2
import torch
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer

model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=1)
upsampler = RealESRGANer(
    scale=1,
    model_path="/content/drive/MyDrive/realesrgan_final_model.pth",
    model=model,
    tile=256,
    tile_pad=10,
    pre_pad=0,
    half=False
)

output_dir = "/content/drive/MyDrive/test_results"
os.makedirs(output_dir, exist_ok=True)
for img_name in os.listdir(TEST_INPUT_DIR):
    img = cv2.imread(os.path.join(TEST_INPUT_DIR, img_name), cv2.IMREAD_COLOR)
    if img is None:
        continue
    output, _ = upsampler.enhance(img, outscale=1)
    cv2.imwrite(os.path.join(output_dir, img_name), output)

from tqdm import tqdm
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

RESULT_DIR = "/content/drive/MyDrive/test_results"
TARGET_DIR = "/content/drive/MyDrive/datasets/test/target"
EXTENSIONS = ('.png', '.jpg', '.jpeg')
psnr_list, ssim_list, l1_list = [], [], []

for fname in tqdm([f for f in os.listdir(RESULT_DIR) if f.lower().endswith(EXTENSIONS)], desc="Evaluating"):
    target_path = os.path.join(TARGET_DIR, fname)
    if not os.path.exists(target_path):
        continue
    result = cv2.cvtColor(cv2.imread(os.path.join(RESULT_DIR, fname)), cv2.COLOR_BGR2RGB)
    target = cv2.cvtColor(cv2.imread(target_path), cv2.COLOR_BGR2RGB)
    if result.shape != target.shape:
        result = cv2.resize(result, (target.shape[1], target.shape[0]))
    l1_list.append(np.mean(np.abs(result.astype(np.float32) - target.astype(np.float32))))
    psnr_list.append(psnr(target, result, data_range=255))
    ssim_list.append(ssim(target, result, data_range=255, channel_axis=2))
    if len(psnr_list) % 10 == 0:
        print(f"[{len(psnr_list)}] PSNR: {np.mean(psnr_list):.2f}  SSIM: {np.mean(ssim_list):.4f}  L1: {np.mean(l1_list):.2f}")

print(f"\nImages: {len(psnr_list)}")
print(f"PSNR : {np.mean(psnr_list):.2f} dB")
print(f"SSIM : {np.mean(ssim_list):.4f}")
print(f"L1   : {np.mean(l1_list):.4f}")