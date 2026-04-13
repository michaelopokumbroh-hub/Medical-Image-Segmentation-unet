# Evaluating the trained model on the validation set.

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.models.unet import UNet
from src.models.resnet34_unet import ResNet34UNet
from src.oxford_pet import OxfordPetDataset, get_ids

IMG_DIR = 'dataset/oxford-iiit-pet/images'
MASK_DIR = 'dataset/oxford-iiit-pet/annotations/trimaps'
VAL_TXT = 'dataset/oxford-iiit-pet/val.txt'
IMG_SIZE = 128
BATCH_SIZE = 32

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def dice_torch(logits, mask, threshold=0.5, eps=1e-7):
    pred = torch.sigmoid(logits)
    pred = (pred > threshold).float()
    mask = mask.float()
    
    intersection = (pred * mask).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + mask.sum(dim=(2, 3))
    dice = (2 * intersection + eps) / (union + eps)
    return dice.mean().item()

def evaluate(model, checkpoint_path, batch_size=BATCH_SIZE):
    if not os.path.exists(checkpoint_path):
        print(f"Error: {checkpoint_path} not found.")
        return
        
    print(f"Evaluating: {checkpoint_path}")
    val_ids = get_ids(VAL_TXT)
    val_dataset = OxfordPetDataset(IMG_DIR, MASK_DIR, val_ids, img_size=IMG_SIZE)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    model.eval()
    
    criterion = torch.nn.BCEWithLogitsLoss()
    avg_loss = 0
    avg_dice = 0
    n = 0
    
    with torch.no_grad():
        for imgs, masks in tqdm(val_loader):
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            logits = model(imgs)
            
            loss = criterion(logits, masks)
            dice = dice_torch(logits, masks)
            
            avg_loss += loss.item() * imgs.size(0)
            avg_dice += dice * imgs.size(0)
            n += imgs.size(0)
            
    avg_loss /= n
    avg_dice /= n
    print(f"Validation BCE Loss: {avg_loss:.4f}")
    print(f"Validation Dice Score: {avg_dice:.4f}")
    return avg_loss, avg_dice

if __name__ == "__main__":
    # For UNet
    print("\n==== UNet results ====")
    unet_model = UNet().to(DEVICE)
    evaluate(unet_model, 'saved_models/unet_best.pth')

    # For ResNet34UNet
    print("\n==== ResNet34+UNet results ====")
    resnet_model = ResNet34UNet().to(DEVICE)
    evaluate(resnet_model, 'saved_models/resnet34_unet_best.pth')
