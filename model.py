import torch
import numpy as np
import torch.nn as nn
from mae import *
from utils import *

class skateMAE(torch.nn.Module):
    def __init__(self, encoder : MAE_Encoder, embed_dim : int) -> None:
        super().__init__()
        self.cls_token = encoder.cls_token
        self.pos_embedding = encoder.pos_embedding
        self.patchify = encoder.patchify
        self.transformer = encoder.transformer
        self.layer_norm = encoder.layer_norm
        self.heads = [nn.Sequential(nn.Linear(self.pos_embedding.shape[-1], embed_dim), nn.Linear(embed_dim, 1))] * 3
    def forward(self, img):
        patches = self.patchify(img)
        patches = rearrange(patches, 'b c h w -> (h w) b c')
        patches = patches + self.pos_embedding
        patches = torch.cat([self.cls_token.expand(-1, patches.shape[1], -1), patches], dim=0)
        patches = rearrange(patches, 't b c -> b t c')
        features = self.layer_norm(self.transformer(patches))
        features = rearrange(features, 'b t c -> t b c')
        return tuple([head(features[0]) for head in self.heads])

# class skateGAN(torch.nn.Module):
#     def __init__(self, obj_path, batch_size, device, img_size=64, patch_size=4) -> None:
#         super().__init__()
#         self.encoder = MAE_Encoder(image_size=img_size, patch_size=patch_size)
#         self.decoder = MAE_Decoder()
#         self.camera_position_heads = [ViT_Classifier(self.encoder, 180)] * 3
#         self.pose_image_generator = pose_generator(obj_path, img_size, batch_size, device)
#     def forward(self, curr_imgs, prev_imgs):
#         encoded_curr = self.encoder(curr_imgs)
#         encoded_prev = self.encoder(prev_imgs)
#         cam_pos = [head(encoded_curr) for head in self.camera_position_heads]
#         generated_poses = pose_generator(cam_pos[0], cam_pos[1], cam_pos[2])
#         encoded_gen = self.encoder(generated_poses)
#         predicted_imgs = self.decoder(encoded_prev, encoded_gen)
#         return predicted_imgs, cam_pos