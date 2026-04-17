!pip install torch torchvision tqdm scikit-image

import os
import glob
import shutil
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from google.colab import drive

drive.mount('/content/drive')

TRAIN_INPUT  = "/content/drive/MyDrive/datasets/train/input"
TRAIN_TARGET = "/content/drive/MyDrive/datasets/train/target"
TEST_INPUT   = "/content/drive/MyDrive/datasets/test/input"
TEST_TARGET  = "/content/drive/MyDrive/datasets/test/target"

class SRDataset(Dataset):
    def __init__(self, input_dir, target_dir):
        self.input_paths  = sorted(glob.glob(os.path.join(input_dir, "*")))
        self.target_paths = sorted(glob.glob(os.path.join(target_dir, "*")))
        self.transform = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor()])
    def __len__(self):
        return len(self.input_paths)
    def __getitem__(self, idx):
        inp = self.transform(Image.open(self.input_paths[idx]).convert("RGB"))
        tgt = self.transform(Image.open(self.target_paths[idx]).convert("RGB"))
        return inp, tgt

class FastSRCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 5, 1, 2), nn.ReLU(),
            nn.Conv2d(32, 16, 3, 1, 1), nn.ReLU(),
            nn.Conv2d(16, 3, 3, 1, 1)
        )
    def forward(self, x):
        return self.net(x)

train_loader = DataLoader(SRDataset(TRAIN_INPUT, TRAIN_TARGET), batch_size=4, shuffle=True)
test_loader  = DataLoader(SRDataset(TEST_INPUT, TEST_TARGET), batch_size=1, shuffle=False)

device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model     = FastSRCNN().to(device)
criterion = nn.L1Loss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

EPOCHS = 2
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{EPOCHS}]")
    for inp, tgt in loop:
        inp, tgt = inp.to(device), tgt.to(device)
        out = model(inp)
        loss = criterion(out, tgt)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        loop.set_postfix(loss=loss.item())
    print(f"Epoch {epoch+1} Loss: {total_loss/len(train_loader):.4f}")

torch.save(model.state_dict(), "/content/drive/MyDrive/fastsrcnn.pth")

model.eval()
os.makedirs("/content/results", exist_ok=True)
to_pil = transforms.ToPILImage()
total_psnr = total_ssim = total_loss = 0

with torch.no_grad():
    for i, (inp, tgt) in enumerate(tqdm(test_loader)):
        inp, tgt = inp.to(device), tgt.to(device)
        out  = model(inp)
        total_loss += criterion(out, tgt).item()
        out_img = out.squeeze(0).cpu().clamp(0, 1).permute(1, 2, 0).numpy()
        tgt_img = tgt.squeeze(0).cpu().permute(1, 2, 0).numpy()
        total_psnr += peak_signal_noise_ratio(tgt_img, out_img, data_range=1.0)
        total_ssim += structural_similarity(tgt_img, out_img, channel_axis=2, data_range=1.0)
        to_pil(out.squeeze(0).cpu().clamp(0, 1)).save(f"/content/results/{i}.png")

n = len(test_loader)
print(f"Loss : {total_loss/n:.4f}")
print(f"PSNR : {total_psnr/n:.2f}")
print(f"SSIM : {total_ssim/n:.4f}")

drive_results = "/content/drive/MyDrive/results"
os.makedirs(drive_results, exist_ok=True)
for i, file in enumerate(sorted(glob.glob("/content/results/*.png"))):
    shutil.move(file, os.path.join(drive_results, f"img_{i:03d}.png"))