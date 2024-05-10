import torch
import numpy as np
import torch.nn as nn
from mae import *
from utils import *

class skateMAE(torch.nn.Module):
    '''
    SkateMAE model as described in paper. Modified from 
    https://github.com/IcarusWizard/MAE/blob/main/model.py
    '''
    def __init__(self, encoder : MAE_Encoder, embed_dim : int) -> None:
        super().__init__()
        self.cls_token = encoder.cls_token
        self.pos_embedding = encoder.pos_embedding
        self.patchify = encoder.patchify
        self.transformer = encoder.transformer
        self.layer_norm = encoder.layer_norm
        self.dist_head = nn.Sequential(nn.Linear(self.pos_embedding.shape[-1], embed_dim), nn.Linear(embed_dim, 1))
        self.elev_head = nn.Sequential(nn.Linear(self.pos_embedding.shape[-1], embed_dim), nn.Linear(embed_dim, 1))
        self.azim_head = nn.Sequential(nn.Linear(self.pos_embedding.shape[-1], embed_dim), nn.Linear(embed_dim, 1)) 
        self.heads = [self.dist_head, self.elev_head, self.azim_head]
    def forward(self, img):
        patches = self.patchify(img)
        patches = rearrange(patches, 'b c h w -> (h w) b c')
        patches = patches + self.pos_embedding
        patches = torch.cat([self.cls_token.expand(-1, patches.shape[1], -1), patches], dim=0)
        patches = rearrange(patches, 't b c -> b t c')
        features = self.layer_norm(self.transformer(patches))
        features = rearrange(features, 'b t c -> t b c')
        dist_pred = self.dist_head(features[0])
        elev_pred = self.elev_head(features[0]) * 360
        azim_pred = self.azim_head(features[0]) * 180
        return dist_pred, elev_pred, azim_pred, #features[0]

class skateMAE_Video(torch.nn.Module):
    '''
    In progress
    '''
    def __init__(self, encoder : MAE_Encoder, embed_dim : int) -> None:
        super().__init__()
        self.mae = skateMAE(encoder, embed_dim)
        self.dist_head = nn.Sequential(nn.Linear(self.mae.pos_embedding.shape[-1] * 2, embed_dim), nn.Linear(embed_dim, 1))
        self.elev_head = nn.Sequential(nn.Linear(self.mae.pos_embedding.shape[-1] * 2, embed_dim), nn.Linear(embed_dim, 1))
        self.azim_head = nn.Sequential(nn.Linear(self.mae.pos_embedding.shape[-1] * 2, embed_dim), nn.Linear(embed_dim, 1))
    def forward(self, imgs, prev_imgs):
        _, _, _, curr_features =  self.mae(imgs)
        _, _, _, prev_features = self.mae(prev_imgs)
        features = torch.cat((curr_features, prev_features), dim=-1)
        dist_pred = self.dist_head(features[0])
        elev_pred = self.elev_head(features[0]) * 360
        azim_pred = self.azim_head(features[0]) * 180
        return dist_pred, elev_pred, azim_pred 