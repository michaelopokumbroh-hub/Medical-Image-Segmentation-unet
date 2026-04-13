# Oxford-iiit Pet Dataset Loading and Preprocessing.

import os
import numpy as np       
import torch
from PIL import Image
from torch.utils.data import Dataset
from src.utils import mask_trimap_to_binary

class OxfordPetDataset(Dataset):
    def __init__(self, img_dir, mask_dir, ids, img_size=128, is_test=False):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.ids = ids
        self.img_size = img_size
        self.is_test = is_test

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        img = Image.open(os.path.join(self.img_dir, img_id + '.jpg')).convert('RGB')
        img = img.resize((self.img_size, self.img_size))
        img = np.array(img).astype(np.float32) / 255.0
        img = img.transpose((2,0,1))
        if self.is_test:
            return torch.tensor(img, dtype=torch.float32), img_id
        mask = Image.open(os.path.join(self.mask_dir, img_id + '.png')).resize((self.img_size, self.img_size), Image.NEAREST)
        mask = mask_trimap_to_binary(np.array(mask))
        return torch.tensor(img, dtype=torch.float32), torch.tensor(mask, dtype=torch.float32).unsqueeze(0)
    
    def __len__(self):
        return len(self.ids)

def get_ids(txt_path):
    with open(txt_path) as f:
        return [l.strip() for l in f]
