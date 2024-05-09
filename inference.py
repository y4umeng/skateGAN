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
from gen_synth_data import pose_generator

def get_poses(model_path, clip_id, device):
    transform = Compose([ToTensor(), Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])])
    real_frames = glob(f'data/batb1k/frames/{clip_id}_*.jpg')
    real_frame_paths = {}
    print(f'Num real frames: {len(real_frames)}')
    for f in real_frames:
        splt = f.split('/')[-1].split('.')[0].split('_')
        real_frame_paths[int(splt[-1])] = f
    
    print(f"Num frames: {len(real_frame_paths)}")
    # print(sorted(frame_paths.keys()))

    real_frames = torch.zeros((max(real_frame_paths.keys())+1, 3, 128, 128))
    for frame_id in range(real_frames.shape[0]):
        frame = transform(Image.open(real_frame_paths[frame_id]))
        real_frames[frame_id,...] = frame
        # print(fp, frame.shape)
    print(f"Frames shape: {real_frames.shape}")
    real_frames = real_frames.to(device)

    model = torch.load(model_path, map_location=device)
    model = model.to(device)

    model.eval()
    with torch.no_grad():
        dist_preds, elev_preds, azim_preds = model(real_frames)
        all_data = torch.stack((dist_preds, elev_preds, azim_preds), dim=1)
        print(f"Final data shape: {all_data.shape}")
        torch.save(all_data.cpu(), f"inference/clip{clip_id}_pred128.pt")
    return all_data, real_frames

def generate_gif(preds, clip_id, real_frames):
    pg = pose_generator('data/board_model/skateboard.obj', 128, 1, device)
    synth_frames = torch.zeros_like(real_frames)

    for i in range(synth_frames.shape[0]):
        frame = pg(preds[i,0], preds[i,1], preds[i,2])
        print(f"Synth frame: {frame.shape}, {frame.max()}")
        synth_frames[i,...] = frame

    gif_frames = torch.cat((real_frames, synth_frames), dim=-1)
    print(f'GIF frames shape: {gif_frames.shape}')

    real_gif = gif_frames.permute(0, 2, 3, 1).numpy() * 255
    clip = ImageSequenceClip(list(real_gif), fps=5)
    clip.write_gif(f'inference/clip{clip_id}_128.gif', fps=5)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=888)
    parser.add_argument('--model_path', type=str, default='checkpoints/skatemae128.pt')
    parser.add_argument('--clip_id', type=str, default='0')
    args = parser.parse_args()
    setup_seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_devices = torch.cuda.device_count() 
    print(f"Device: {device}")
    print(f"Num GPUs: {num_devices}")

    preds, real_frames = get_poses(args.model_path, args.clip_id, device)
    generate_gif(preds, args.clip_id, real_frames)
    print(f"Done with clip {args.clip_id}")