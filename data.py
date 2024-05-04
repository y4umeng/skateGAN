import os, sys
import numpy as np
import torch
# from torchvision import transforms
from torch.utils.data import Dataset
from glob import glob
from os import path
import random

class skate_data(Dataset):
    def __init__(self, data_path, device, transform):
        self.transform = transform
        self.device = device
        self.data = []
        if len(self.data) != len(self.labels): raise ValueError("The number of validation or testing images and labels do not match.")
        print(f"{len(self.data)} files found at {data_path}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        pass