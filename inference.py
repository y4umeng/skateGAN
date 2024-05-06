import argparse
import math
import torch
# from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import ToTensor, Compose, Normalize
from tqdm import tqdm

from model import *
from utils import *
from glob import glob
from os import path
from PIL import Image

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--model_path', type=str, default='')
    parser.add_argument('--clip_id', type=str, default='0')
    
    args = parser.parse_args()

    setup_seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_devices = torch.cuda.device_count() 
    print(f"Device: {device}")
    print(f"Num GPUs: {num_devices}")

    transform = Compose([ToTensor(), Normalize(0.5, 0.5)])
    
    files = glob('data/batb1k/frames/*.jpg')
    frame_paths = {}
    print(f'Num files: {len(files)}')
    for f in files:
        splt = f.split('/')[-1].split('.')[0].split('_')
        if args.clip_id == splt[0]:
            frame_paths[int(splt[-1])] = f
    
    print(f"Num frames: {len(frame_paths)}")

    frames = torch.zeros((len(frame_paths), 3, 32, 32))
    for fp in frame_paths:
        frame = transform(Image.open(frame_paths[fp]))
        frames[fp,...] = frame
        print(fp, frame.shape)

    frames = frames.to(device)
    print(f"Frames shape: {frames.shape}")

    # model = torch.load(args.model_path, map_location=device)
    # # model = model.to(device)

    # model.eval()
    # with torch.no_grad():
    #     all_data = []
    #     dist_preds, elev_preds, azim_preds = model(frames)
    #     all_data.append([id[0], dist_preds[0], elev_preds[0], azim_preds[0]])
    #     print(all_data[-1])
       

