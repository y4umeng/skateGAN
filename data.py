import numpy as np
from torch.utils.data import Dataset
from glob import glob
from os import path
from PIL import Image
import torch.nn as nn
import torch
import csv

class skate_data(Dataset):
    def __init__(self, data_path, label_csv_path, device, transform):
        self.transform = transform
        self.device = device
        self.files = glob(path.join(data_path, "*.jpg"))
        self.labels = {}
        with open(label_csv_path, 'r') as data:
            count = 0
            for line in csv.reader(data):
                print(count)
                if count == 0: continue # skip first line
                self.labels[line[0]] = torch.tensor([line[1], line[2], line[3]])
                print(f'{line[0]} : {self.labels[line[0]]}')
                count += 1
                if count == 10: break
                print(count)

        print(f"{len(self.files)} files found at {data_path}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        pass

class skate_data_pretrain(Dataset):
    def __init__(self, data_paths, device, transform=nn.Identity()):
        self.transform = transform
        self.device = device
        self.files = []
        for p in data_paths:
          self.files += glob(path.join(p, '*.jpg'))
          print(f"{len(self.files)} files found at {p}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        return self.transform(Image.open(self.files[idx])).to(self.device)