import random
import torch
import numpy as np
import argparse
from glob import glob 
from os import path 
import torch.nn as nn
import matplotlib.pyplot as plt
import time
from pytorch3d.io import load_obj

# datastructures
from pytorch3d.structures import Meshes

# rendering components
from pytorch3d.renderer import (
    FoVPerspectiveCameras, look_at_view_transform, SoftPhongShader,
    RasterizationSettings, MeshRenderer, MeshRasterizer, BlendParams,
    TexturesAtlas, PointLights
)

def setup_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

class binary_erosion(object):
    def __init__ (self):
        self.conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, bias=False, padding=1)
        self.conv.weight = torch.nn.Parameter(torch.tensor([[0.0, 1.0, 0.0], [1.0, 1.0, 1.0], [0.0, 1.0, 0.0]]).unsqueeze(0).unsqueeze(0))
    def __call__(self, mask, iterations):
        mask = mask.unsqueeze(0)
        for _ in range(iterations):
            mask = self.conv(mask.float()) >= 2.0
        return mask
    def __repr__(self):
        return "binary erosion..."
    
class Add_Legs(object):
    def __init__(self, leg_directory, p=0.9):
        self.p = p
        self.leg_files = glob(path.join(leg_directory, '*.pt'))
        self.erosion = binary_erosion()

        if len(self.leg_files) == 0:
            raise ValueError("No legs found.")
        print(f"{len(self.leg_files)} leg masks found at {leg_directory}.")
    def __call__(self, img):
        if torch.rand(1) > self.p: return img
        legs = torch.load(random.choice(self.leg_files))
        mask = legs[3,...].unsqueeze(0)
        mask = mask == 0.0
        legs = legs[:3,...] / 255.0
        mask = self.erosion(mask, 6).squeeze(0)
        return img * mask + legs
    def __repr__(self):
        return "adding random legs augmentation"
    
class pose_generator(torch.nn.Module):
    def __init__(self, obj_path, img_shape, batch_size, device) -> None:
        super().__init__()
        # Get vertices, faces, and auxiliary information:
        verts, faces, aux = load_obj(
            obj_path,
            device=device,
            load_textures=True,
            create_texture_atlas=True,
            texture_atlas_size=4,
            texture_wrap="repeat")

        # Create a textures object
        atlas = aux.texture_atlas

        # Initialize the mesh with vertices, faces, and textures.
        # Created Meshes object
        mesh = Meshes(
            verts=[verts],
            faces=[faces.verts_idx],
            
            textures=TexturesAtlas(atlas=[atlas]),)
        meshes = mesh.extend(batch_size)
        print('We have {0} vertices and {1} faces.'.format(verts.shape[0], faces.verts_idx.shape[0]))

        # To blend the 100 faces we set a few parameters which control the opacity and the sharpness of edges.
        white = (1.0, 1.0, 1.0)
        blend_params = BlendParams(sigma=1e-8, gamma=1e-4, background_color=white)

        # Here we set the output image to be of size 256 x 256 based on config.json
        raster_settings = RasterizationSettings(
            image_size = img_shape,
            blur_radius = 0.0,
            faces_per_pixel = 100,
            bin_size=0
        )

        self.lights = PointLights(device=device, location=[[0.0, 0.0, -3.0]])

        R, T = look_at_view_transform(0, 0, 0)
        cameras = FoVPerspectiveCameras(device=device, R=R, T=T)

        # Create a Phong renderer by composing a rasterizer and a shader. The textured Phong shader will
        # interpolate the texture uv coordinates for each vertex, sample from a texture image and
        # apply the Phong lighting model

        self.renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=cameras,
                raster_settings=raster_settings,
                
            ),
            shader=SoftPhongShader(
                device=device,
                cameras=cameras,
                blend_params=blend_params,
                lights=self.lights
            )
        )

        self.meshes = meshes
        self.device = device

    def forward(self, dist, elev, azim):
        # camera positions: N x 3 tensor, (dist, elev, azim)
        R, T = look_at_view_transform(dist=dist, elev=elev, azim=azim)
        cameras = FoVPerspectiveCameras(znear=-100, zfar=100, device=self.device, R=R, T=T)

        # image = self.renderer(meshes_world=self.meshes.clone(), R=R, T=T)
        images = self.renderer(self.meshes, cameras=cameras, lights=self.lights)
        return images[..., :3], torch.ceil(images[..., 3])
    
def image_grid(
    images,
    rows=None,
    cols=None,
    fill: bool = True,
    show_axes: bool = False,
    rgb: bool = True,
):
    """
    A util function for plotting a grid of images.

    Args:
        images: (N, H, W, 4) array of RGBA images
        rows: number of rows in the grid
        cols: number of columns in the grid
        fill: boolean indicating if the space between images should be filled
        show_axes: boolean indicating if the axes of the plots should be visible
        rgb: boolean, If True, only RGB channels are plotted.
            If False, only the alpha channel is plotted.

    Returns:
        None
    """
    if (rows is None) != (cols is None):
        raise ValueError("Specify either both rows and cols or neither.")

    if rows is None:
        rows = len(images)
        cols = 1

    gridspec_kw = {"wspace": 0.0, "hspace": 0.0} if fill else {}
    fig, axarr = plt.subplots(rows, cols, gridspec_kw=gridspec_kw, figsize=(15, 9))
    bleed = 0
    fig.subplots_adjust(left=bleed, bottom=bleed, right=(1 - bleed), top=(1 - bleed))

    for ax, im in zip(axarr.ravel(), images):
        if rgb:
            # only render RGB channels
            ax.imshow(im[..., :3])
        else:
            # only render Alpha channel
            ax.imshow(im[..., 3])
        if not show_axes:
            ax.set_axis_off()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    device = args.device
    print(f"Device: {device}")
    batch_size = 64
    setup_seed(8)
    pg = pose_generator('/home/ywongar/skateGAN/data/board_model/skateboard.obj', 128, batch_size, device)
    start = time.time()
    dist = torch.rand(batch_size) * 0.4 + 0.5
    elev = torch.rand(batch_size) * 360
    azim = torch.rand(batch_size) * 180
    # print(dist.shape)
    images, _ = pg(dist.to(device), elev.to(device), azim.to(device))
    print(f"Images: {images.shape}")
    print(f'Time: {time.time() - start}')
    # image_grid(images.cpu())
    torch.save(images.cpu(), 'util_images.pt')
    print("Saved...")
