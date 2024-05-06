import random
import torch
import numpy as np
import sys 
from glob import glob 
from os import path 
import torch.nn as nn

def setup_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

class binary_erosion(object):
    def __init__ (self):
        self.conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, bias=False, padding=1)
        self.conv.weight = torch.nn.Parameter(torch.tensor([[0.0, 1.0, 0.0], [1.0, 1.0, 1.0], [0.0, 1.0, 0.0]]).unsqueeze(0).unsqueeze(0))
    def __call__(self, mask, iterations):
        mask = mask.unsqueeze(0)
        print(f'mask shape: {mask.shape}')
        
        for _ in range(iterations):
            mask = self.conv(mask.float()) >= 2.0
        return mask
    def __repr__(self):
        return "binary erosion..."
    
class Add_Legs(object):
    def __init__(self, leg_directory, p=0.9):
        self.p = p
        self.leg_files = glob(path.join(leg_directory, '*.pt'))
        self.erosion = binary_erosion()

        if len(self.leg_files) == 0:
            raise ValueError("No legs found.")
        print(f"{len(self.leg_files)} leg masks found at {leg_directory}.")
    def __call__(self, img):
        if torch.rand(1) > self.p: return img
        legs = torch.load(random.choice(self.leg_files))
        mask = legs[3,...].unsqueeze(0)
        mask = mask == 0.0
        legs = legs[:3,...] / 255.0
        mask = self.erosion(mask, 6).squeeze(0)
        print(f'Mask: {mask.shape}')
        print(f'Image: {img.shape}')
        print(f'Legs: {legs.shape}')
        return img
        # return img * mask.unsqueeze(0) + legs.unsqueeze(0)
    def __repr__(self):
        return "adding random legs augmentation"