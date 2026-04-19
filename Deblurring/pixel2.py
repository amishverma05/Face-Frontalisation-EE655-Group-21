# 🔹 Install
!pip install torch torchvision tqdm scikit-image

import os, glob, shutil
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from google.colab import drive

drive.mount('/content/drive')

# 🔹 Paths
TRAIN_INPUT  = "/content/drive/MyDrive/datasets/train/input"
TRAIN_TARGET = "/content/drive/MyDrive/datasets/train/target"
TEST_INPUT   = "/content/drive/MyDrive/datasets/test/input"
TEST_TARGET  = "/content/drive/MyDrive/datasets/test/target"

# 🔹 Dataset
class SRDataset(Dataset):
    def __init__(self, input_dir, target_dir):
        self.input_paths  = sorted(glob.glob(os.path.join(input_dir, "*")))
        self.target_paths = sorted(glob.glob(os.path.join(target_dir, "*")))
        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor()
        ])
    def __len__(self):
        return len(self.input_paths)
    def __getitem__(self, idx):
        inp = self.transform(Image.open(self.input_paths[idx]).convert("RGB"))
        tgt = self.transform(Image.open(self.target_paths[idx]).convert("RGB"))
        return inp, tgt

# 🔹 Residual Block
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.BatchNorm2d(channels)
        )
    def forward(self, x):
        return x + self.block(x)

# 🔹 SRResNet-like Model
class SRResNet(nn.Module):
    def __init__(self, num_blocks=8):
        super().__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, 9, 1, 4),
            nn.ReLU()
        )
        
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(64) for _ in range(num_blocks)]
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64)
        )
        
        self.conv3 = nn.Conv2d(64, 3, 9, 1, 4)

    def forward(self, x):
        out1 = self.conv1(x)
        res  = self.res_blocks(out1)
        out2 = self.conv2(res)
        out  = out1 + out2
        return self.conv3(out)

# 🔹 Load data
train_loader = DataLoader(SRDataset(TRAIN_INPUT, TRAIN_TARGET), batch_size=4, shuffle=True)
test_loader  = DataLoader(SRDataset(TEST_INPUT, TEST_TARGET), batch_size=1, shuffle=False)

# 🔹 Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SRResNet().to(device)

# 🔹 (Optional) Load pretrained weights
# model.load_state_dict(torch.load("path_to_pretrained.pth"), strict=False)

criterion = nn.L1Loss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# 🔹 Training
EPOCHS = 5
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    
    for inp, tgt in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
        inp, tgt = inp.to(device), tgt.to(device)
        
        out = model(inp)
        loss = criterion(out, tgt)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    print(f"Epoch {epoch+1} Loss: {total_loss/len(train_loader):.4f}")

# 🔹 Save model
torch.save(model.state_dict(), "/content/drive/MyDrive/srresnet_finetuned.pth")

# 🔹 Evaluation
model.eval()
total_psnr = total_ssim = total_loss = 0

with torch.no_grad():
    for inp, tgt in tqdm(test_loader):
        inp, tgt = inp.to(device), tgt.to(device)
        out = model(inp)
        
        total_loss += criterion(out, tgt).item()
        
        out_img = out.squeeze(0).cpu().clamp(0,1).permute(1,2,0).numpy()
        tgt_img = tgt.squeeze(0).cpu().permute(1,2,0).numpy()
        
        total_psnr += peak_signal_noise_ratio(tgt_img, out_img, data_range=1.0)
        total_ssim += structural_similarity(tgt_img, out_img, channel_axis=2, data_range=1.0)

n = len(test_loader)
print(f"Loss : {total_loss/n:.4f}")
print(f"PSNR : {total_psnr/n:.2f}")
print(f"SSIM : {total_ssim/n:.4f}")
