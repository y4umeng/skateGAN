import argparse
import math
import torch
# from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import ToTensor, Compose, Normalize
from tqdm import tqdm
from moviepy.editor import ImageSequenceClip
import numpy as np

from model import *
from utils import *
from glob import glob
from os import path
from PIL import Image

def get_poses(model_path, clip_id, device):
    '''
    Generates rotation estimation given a clip_id (essentially file name).
    '''
    transform = Compose([ToTensor(), Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])])
    real_frames = get_clip_frames(clip_id, transform=transform).to(device)

    model = torch.load(model_path, map_location=device).module
    model = model.to(device)

    model.eval()
    with torch.no_grad():
        dist_preds, elev_preds, azim_preds = model(real_frames)
        elev_preds = torch.round(elev_preds)
        azim_preds = torch.round(azim_preds)
        all_data = torch.stack((dist_preds, elev_preds, azim_preds), dim=1)
        print(f"Final data shape: {all_data.shape}")
        torch.save(all_data.cpu(), f"inference/clip{clip_id}_pred128.pt")
    return

def pt_to_gif(path, clip_id):
    '''
    Converts image frames in .pt form to a .gif for presentation
    '''
    frames = torch.load(path)
    print(f"Frames: {frames.shape}, Max: {frames.max()}")
    clip = ImageSequenceClip(list(frames), fps=5)
    clip.write_gif(f'inference/clip{clip_id}_128.gif', fps=5)
    print(f'Saved GIF to {path}')

# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--seed', type=int, default=888)
#     parser.add_argument('--model_path', type=str, default='checkpoints/skatemae128.pt')
#     parser.add_argument('--clip_id', type=str, default='0')
#     args = parser.parse_args()
#     setup_seed(args.seed)
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
#     num_devices = torch.cuda.device_count() 
#     print(f"Device: {device}")
#     print(f"Num GPUs: {num_devices}")
#     pt_to_gif('inference/clip43_FinalFrames128.pt', 43)
    # get_poses(args.model_path, args.clip_id, device)
    # # generate_gif(preds, args.clip_id, real_frames)
    # print(f"Done with clip {args.clip_id}")