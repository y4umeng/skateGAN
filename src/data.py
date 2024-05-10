from torch.utils.data import Dataset
from glob import glob
from os import path
from PIL import Image
import torch.nn as nn
import torch
import torchvision
import random
import csv
from utils import add_background_image, Add_Legs
from torchvision.transforms import ToTensor, Compose, Normalize
import matplotlib.pyplot as plt

class skate_data_synth_test(Dataset):
    '''
    Dataset for loading jpg synthetic files with ground truth labels in csv form
    '''
    def __init__(self, data_path, label_csv_path, transform):
        self.transform = transform

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
    
class skate_data_synth(Dataset):
    '''
    Dataset for loading synthetic data still in .pt form, along with ground truth labels in csv.
    Requires directory of background jpgs to choose at random
    '''
    def __init__(self, image_path, background_path, label_path, transform):
        self.transform = transform
        self.backgrounds = glob(path.join(background_path, '*.jpg'))

        files = glob(path.join(image_path, "*.pt"))
        frame_ids = [f.split('/')[-1].split('.')[0] for f in files]
        labels = {}
        with open(label_path, 'r') as data:
            count = 0
            for line in csv.reader(data):
                # skip first line
                if count == 0: 
                    count += 1
                    continue
                labels[line[0]] = torch.tensor([float(line[1]), float(line[2]) % 360.0, float(line[3]) % 180.0])

        self.images = []
        self.labels = []
        self.ids = []
        for file, id in zip(files, frame_ids):
            if id.isnumeric() and id in labels:
                self.images.append(file)
                self.labels.append(labels[id])
                self.ids.append(torch.tensor([int(id)]))

        print(f"{len(self.images)} valid files found at {image_path}")
        print(f'{len(self.backgrounds)} found at {background_path}')
    def __len__(self):
        return len(self.images)
    def __getitem__(self, idx):
        background = torchvision.transforms.functional.pil_to_tensor(Image.open(random.choice(self.backgrounds)))
        img = torch.load(self.images[idx], map_location='cpu').permute(2, 0, 1)
        img = add_background_image(img[:3,...], img[3,...], background)
        labels = self.labels[idx]
        return self.transform(img), labels[0], labels[1], labels[2], self.ids[idx]

class skate_data_pretrain(Dataset):
    '''
    Simple dataset for the real world data in jpg form. No labels.
    '''
    def __init__(self, data_paths, transform):
        self.transform = transform
        self.files = []
        for p in data_paths:
            self.files += glob(path.join(p, '*.jpg'))
            print(f"{len(self.files)} files found at {p}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        return self.transform(Image.open(self.files[idx]))
    
class skate_data_synth_pretrain(Dataset):
    '''
    Dataset for loading synthetic data without labels for pretraining MAE on image reconstruction
    '''
    def __init__(self, image_path, background_path, transform):
        self.transform = transform
        self.images = glob(path.join(image_path, '*.pt'))
        self.backgrounds = glob(path.join(background_path, '*.jpg'))
        print(f'{len(self.images)} found at {image_path}')
        print(f'{len(self.backgrounds)} found at {background_path}')
    def __len__(self):
        return len(self.images)
    def __getitem__(self, idx):
        background = torchvision.transforms.functional.pil_to_tensor(Image.open(random.choice(self.backgrounds)))
        img = torch.load(self.images[idx], map_location='cpu').permute(2, 0, 1)
        img = add_background_image(img[:3,...], img[3,...], background)
        return self.transform(img)

class skate_data_combined(Dataset):
    '''
    Used in pretraining to load both real and synthetic data in one Dataset instance
    '''
    def __init__(self, real_dataset, synth_dataset, transform):
        self.transform = transform
        self.real = real_dataset
        self.synth = synth_dataset
    def __len__(self):
        return len(self.real) + len(self.synth)
    def __getitem__(self, idx):
        if idx < len(self.real): return self.transform(self.real[idx])
        return self.transform(self.synth[idx-len(self.real)])