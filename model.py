import torch
import numpy as np
import torch.nn as nn
from mae import *
from utils import pose_generator

class skateGAN(torch.nn.Module):
    def __init__(self, obj_path, batch_size, device, img_size=64, patch_size=4) -> None:
        super().__init__()
        self.encoder = MAE_Encoder(image_size=img_size, patch_size=patch_size)
        self.decoder = MAE_Decoder()
        self.camera_position_heads = [ViT_Classifier(self.encoder, 180)] * 3
        self.pose_image_generator = pose_generator(obj_path, img_size, batch_size, device)
    def forward(self, curr_imgs, prev_imgs):
        encoded_curr = self.encoder(curr_imgs)
        encoded_prev = self.encoder(prev_imgs)
        cam_pos = [head(encoded_curr) for head in self.camera_position_heads]
        generated_poses = pose_generator(cam_pos[0], cam_pos[1], cam_pos[2])
        encoded_gen = self.encoder(generated_poses)
        predicted_imgs = self.decoder(encoded_prev, encoded_gen)
        return predicted_imgs, cam_pos