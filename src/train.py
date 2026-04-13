import sys, os
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm

# Path
IMG_DIR = 'dataset/oxford-iiit-pet/images'
MASK_DIR = 'dataset/oxford-iiit-pet/annotations/trimaps'
TRAIN_TXT = 'dataset/oxford-iiit-pet/train.txt'
VAL_TXT = 'dataset/oxford-iiit-pet/val.txt'
TEST_TXT = ['dataset/oxford-iiit-pet/test_unet.txt' , 'dataset/oxford-iiit-pet/test_res_unet.txt']
OUT_DIR = 'pred_masks'
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs('saved_models', exist_ok=True)

BATCH_SIZE = 32
IMG_SIZE = 128
train_ids = get_ids(TRAIN_TXT)
val_ids = get_ids(VAL_TXT)
train_dataset = OxfordPetDataset(IMG_DIR, MASK_DIR, train_ids, img_size=IMG_SIZE)
val_dataset = OxfordPetDataset(IMG_DIR, MASK_DIR, val_ids, img_size=IMG_SIZE)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Training loop for UNet
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = UNet().to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=3e-4)
criterion = nn.BCEWithLogitsLoss()

def dice_torch(preds, targets, eps=1e-7):
    preds = torch.sigmoid(preds)
    preds = (preds > 0.5).float()
    targets = targets.float()
    inter = (preds * targets).sum(dim=(2,3))
    union = preds.sum(dim=(2,3)) + targets.sum(dim=(2,3))
    dice = (2*inter + eps) / (union + eps)
    return dice.mean().item()

best_val = 0
for epoch in range(1, 31):  # 30 epochs
    model.train()
    epoch_loss = 0
    for imgs, masks in tqdm(train_loader):
        imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
        optimizer.zero_grad()
        logits = model(imgs)
        loss = criterion(logits, masks)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()*imgs.size(0)
    epoch_loss /= len(train_loader.dataset)
    model.eval()
    val_dice = 0
    with torch.no_grad():
        for imgs, masks in val_loader:
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            logits = model(imgs)
            val_dice += dice_torch(logits, masks)*imgs.size(0)
    val_dice /= len(val_loader.dataset)
    print(f"Epoch {epoch} | Loss: {epoch_loss:.3f} | Val Dice: {val_dice:.3f}")
    if val_dice > best_val:
        best_val = val_dice
        torch.save(model.state_dict(), 'saved_models/unet_best.pth')


# Training loop for ResNet34UNet

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ResNet34UNet().to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=3e-4)
criterion = nn.BCEWithLogitsLoss()

def dice_torch(preds, targets, eps=1e-7):
    preds = torch.sigmoid(preds)
    preds = (preds > 0.5).float()
    targets = targets.float()
    inter = (preds * targets).sum(dim=(2,3))
    union = preds.sum(dim=(2,3)) + targets.sum(dim=(2,3))
    dice = (2*inter + eps) / (union + eps)
    return dice.mean().item()

best_val = 0
for epoch in range(1, 41):  # 40 epochs
    model.train()
    epoch_loss = 0
    for imgs, masks in tqdm(train_loader):
        imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
        optimizer.zero_grad()
        logits = model(imgs)
        loss = criterion(logits, masks)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()*imgs.size(0)
    epoch_loss /= len(train_loader.dataset)
    model.eval()
    val_dice = 0
    with torch.no_grad():
        for imgs, masks in val_loader:
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            logits = model(imgs)
            val_dice += dice_torch(logits, masks)*imgs.size(0)
    val_dice /= len(val_loader.dataset)
    print(f"Epoch {epoch} | Loss: {epoch_loss:.3f} | Val Dice: {val_dice:.3f}")
    if val_dice > best_val:
        best_val = val_dice
        torch.save(model.state_dict(), 'saved_models/resnet34_unet_best.pth')
