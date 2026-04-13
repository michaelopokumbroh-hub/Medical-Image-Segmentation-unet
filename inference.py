import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import sys
from PIL import Image
import numpy as np
from skimage.transform import resize
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.models.unet import UNet
from src.models.resnet34_unet import ResNet34UNet

def rle_encode(mask):
    pixels = mask.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    encoded = " ".join(str(x) for x in runs)
    return encoded

def get_ids(txt_path):
    with open(txt_path) as f:
        return [l.strip() for l in f]

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IMG_SIZE = 128 
IMG_DIR = 'dataset/oxford-iiit-pet/images'


# Inference for UNet
def execute_unet():
    TEST_TXT = 'dataset/oxford-iiit-pet/test_unet.txt'
    MODEL_PATH = 'saved_models/unet_best.pth'
    OUT_MASK_DIR = 'pred_masks_unet'
    os.makedirs(OUT_MASK_DIR, exist_ok=True)

    test_ids = get_ids(TEST_TXT)
    model = UNet().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    submission_lines = ["image_id,encoded_mask"]

    print("Starting UNet Inference")
    for img_id in tqdm(test_ids):
        orig_img = Image.open(os.path.join(IMG_DIR, img_id + '.jpg')).convert('RGB')
        orig_W, orig_H = orig_img.size
        img = orig_img.resize((IMG_SIZE, IMG_SIZE))
        img = np.array(img).astype(np.float32) / 255.0
        img = torch.tensor(img.transpose((2,0,1)), dtype=torch.float32).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            logit = model(img)
            pred_mask = (torch.sigmoid(logit)[0,0].cpu().numpy() > 0.5).astype(np.uint8)
        
        up_mask = resize(pred_mask, (orig_H, orig_W), order=0, preserve_range=True, anti_aliasing=False).astype(np.uint8)
        Image.fromarray(up_mask*255).save(os.path.join(OUT_MASK_DIR, img_id + '.png'))
        rle = rle_encode(up_mask)
        submission_lines.append(f"{img_id},{rle}")

    with open("submission_unet.csv", "w") as f:
        f.write('\n'.join(submission_lines))


# Inference for ResNet34UNet
def execute_resnet():
    TEST_TXT = 'dataset/oxford-iiit-pet/test_res_unet.txt'
    MODEL_PATH = 'saved_models/resnet34_unet_best.pth'
    OUT_MASK_DIR = 'pred_masks_resnet34_unet'
    os.makedirs(OUT_MASK_DIR, exist_ok=True)

    test_ids = get_ids(TEST_TXT)
    model = ResNet34UNet().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    submission_lines = ["image_id,encoded_mask"]

    print("Starting ResNet34-UNet Inference")
    for img_id in tqdm(test_ids):
        orig_img = Image.open(os.path.join(IMG_DIR, img_id + '.jpg')).convert('RGB')
        orig_W, orig_H = orig_img.size
        img = orig_img.resize((IMG_SIZE, IMG_SIZE))
        img = np.array(img).astype(np.float32) / 255.0
        img = torch.tensor(img.transpose((2,0,1)), dtype=torch.float32).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            logit = model(img)
            pred_mask = (torch.sigmoid(logit)[0,0].cpu().numpy() > 0.5).astype(np.uint8)
        
        up_mask = resize(pred_mask, (orig_H, orig_W), order=0, preserve_range=True, anti_aliasing=False).astype(np.uint8)
        Image.fromarray(up_mask*255).save(os.path.join(OUT_MASK_DIR, img_id + '.png'))
        rle = rle_encode(up_mask)
        submission_lines.append(f"{img_id},{rle}")

    with open("submission_resnet34_unet.csv", "w") as f:
        f.write('\n'.join(submission_lines))

if __name__ == "__main__":
    execute_unet()
    execute_resnet()
    print("Done! All submissions generated successfully.")
