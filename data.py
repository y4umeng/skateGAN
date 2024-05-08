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
    def __init__(self, data_paths, transform=nn.Identity()):
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
    def __init__(self, image_path, background_path, transform=nn.Identity()):
        self.transform = transform
        self.images = glob(path.join(image_path, '*.pt'))
        self.backgrounds = glob(path.join(background_path, '*.jpg'))
    def __len__(self):
        return len(self.images)
    def __getitem__(self, idx):
        background = torchvision.transforms.functional.pil_to_tensor(Image.open(random.choice(self.backgrounds)))
        img = torch.load(self.images[idx], map_location='cpu').permute(2, 0, 1)
        img = add_background_image(img[:3,...], img[3,...], background)
        return self.transform(img)

class skate_data_combined(Dataset):
    def __init__(self, real_dataset, synth_dataset, transform=nn.Identity()):
        self.transform = transform
        self.real = real_dataset
        self.synth = synth_dataset
    def __len__(self):
        return len(self.real) + len(self.synth)
    def __getitem__(self, idx):
        if idx < len(self.real): return self.transform(self.real[idx])
        return self.transform(self.synth[idx-len(self.real)])

if __name__ == '__main__':
    device = 'cpu'
    transform = Compose([Add_Legs('data/leg_masks128', p=1.0), 
                         Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])])
    inv_normalize = Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225], std=[1/0.229, 1/0.224, 1/0.255])
    
    train_dataset = skate_data_synthetic('data/synthetic_frames128', 'data/backgrounds128', transform=transform)
    dataloader = torch.utils.data.DataLoader(train_dataset, 1, shuffle=True, num_workers=1)
    for img, labels in dataloader:
        print(img.max())
        print(img.min())
        img = inv_normalize(img)
        print(img.max())
        print(img.min())
        plt.imshow(img.squeeze().permute(1, 2, 0))
        plt.show()
        break