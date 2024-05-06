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
        files = glob(path.join(data_path, "*.jpg"))
        frame_ids = [f.split('/')[-1].split('.')[0] for f in files]
        labels = {}
        with open(label_csv_path, 'r') as data:
            count = 0
            for line in csv.reader(data):
                # skip first line
                if count == 0: 
                    count += 1
                    continue
                # previous dist normalizing: round((float(line[1].strip()) - 0.5)*247.5)
                labels[line[0]] = torch.tensor([float(line[1]), float(line[2]) % 360.0, float(line[3]) % 180.0])

        self.files = []
        self.labels = []
        self.ids = []
        for file, id in zip(files, frame_ids):
            if id.isnumeric() and id in labels:
                self.files.append(file)
                self.labels.append(labels[id])
                self.ids.append(int(id))

        print(f"{len(self.files)} valid files found at {data_path}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        labels = self.labels[idx]
        return self.transform(Image.open(self.files[idx])), labels[0], labels[1], labels[2], self.ids[idx]

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
   