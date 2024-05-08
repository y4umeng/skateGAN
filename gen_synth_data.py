import torch
import argparse
from glob import glob 
from os import path 
import torch.nn as nn
import matplotlib.pyplot as plt
import csv
import torchvision
from utils import setup_seed, Add_Legs
from data import skate_data_synth
from tqdm import tqdm
# from pytorch3d.io import load_obj

# # datastructures
# from pytorch3d.structures import Meshes

# # rendering components
# from pytorch3d.renderer import (
#     FoVPerspectiveCameras, look_at_view_transform, SoftPhongShader,
#     RasterizationSettings, MeshRenderer, MeshRasterizer, BlendParams,
#     TexturesAtlas, PointLights
# )

# class pose_generator(torch.nn.Module):
#     def __init__(self, obj_path, img_shape, batch_size, device) -> None:
#         super().__init__()
#         # Get vertices, faces, and auxiliary information:
#         verts, faces, aux = load_obj(
#             obj_path,
#             device=device,
#             load_textures=True,
#             create_texture_atlas=True,
#             texture_atlas_size=4,
#             texture_wrap="repeat")

#         # Create a textures object
#         atlas = aux.texture_atlas

#         # Initialize the mesh with vertices, faces, and textures.
#         # Created Meshes object
#         mesh = Meshes(
#             verts=[verts],
#             faces=[faces.verts_idx],
            
#             textures=TexturesAtlas(atlas=[atlas]),)
#         meshes = mesh.extend(batch_size)
#         print('We have {0} vertices and {1} faces.'.format(verts.shape[0], faces.verts_idx.shape[0]))

#         # To blend the 100 faces we set a few parameters which control the opacity and the sharpness of edges.
#         white = (1.0, 1.0, 1.0)
#         blend_params = BlendParams(sigma=1e-8, gamma=1e-4, background_color=white)

#         # Here we set the output image to be of size 256 x 256 based on config.json
#         raster_settings = RasterizationSettings(
#             image_size = img_shape,
#             blur_radius = 0.0,
#             faces_per_pixel = 100,
#             bin_size=0
#         )

#         self.lights = PointLights(device=device, location=[[0.0, 0.0, -3.0]])

#         R, T = look_at_view_transform(0, 0, 0)
#         cameras = FoVPerspectiveCameras(device=device, R=R, T=T)

#         # Create a Phong renderer by composing a rasterizer and a shader. The textured Phong shader will
#         # interpolate the texture uv coordinates for each vertex, sample from a texture image and
#         # apply the Phong lighting model

#         self.renderer = MeshRenderer(
#             rasterizer=MeshRasterizer(
#                 cameras=cameras,
#                 raster_settings=raster_settings,
                
#             ),
#             shader=SoftPhongShader(
#                 device=device,
#                 cameras=cameras,
#                 blend_params=blend_params,
#                 lights=self.lights
#             )
#         )

#         self.meshes = meshes
#         self.device = device

#     def forward(self, dist, elev, azim):
#         # camera positions: N x 3 tensor, (dist, elev, azim)
#         R, T = look_at_view_transform(dist=dist, elev=elev, azim=azim)
#         cameras = FoVPerspectiveCameras(znear=-100, zfar=100, device=self.device, R=R, T=T)

#         # image = self.renderer(meshes_world=self.meshes.clone(), R=R, T=T)
#         images = self.renderer(self.meshes, cameras=cameras, lights=self.lights)
#         return images[..., :3], torch.ceil(images[..., 3])
    
# def image_grid(
#     images,
#     rows=None,
#     cols=None,
#     fill: bool = True,
#     show_axes: bool = False,
#     rgb: bool = True,
# ):
#     """
#     A util function for plotting a grid of images.

#     Args:
#         images: (N, H, W, 4) array of RGBA images
#         rows: number of rows in the grid
#         cols: number of columns in the grid
#         fill: boolean indicating if the space between images should be filled
#         show_axes: boolean indicating if the axes of the plots should be visible
#         rgb: boolean, If True, only RGB channels are plotted.
#             If False, only the alpha channel is plotted.

#     Returns:
#         None
#     """
#     if (rows is None) != (cols is None):
#         raise ValueError("Specify either both rows and cols or neither.")

#     if rows is None:
#         rows = len(images)
#         cols = 1

#     gridspec_kw = {"wspace": 0.0, "hspace": 0.0} if fill else {}
#     fig, axarr = plt.subplots(rows, cols, gridspec_kw=gridspec_kw, figsize=(15, 9))
#     bleed = 0
#     fig.subplots_adjust(left=bleed, bottom=bleed, right=(1 - bleed), top=(1 - bleed))

#     for ax, im in zip(axarr.ravel(), images):
#         if rgb:
#             # only render RGB channels
#             ax.imshow(im[..., :3])
#         else:
#             # only render Alpha channel
#             ax.imshow(im[..., 3])
#         if not show_axes:
#             ax.set_axis_off()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    device = args.device
    print(f"Device: {device}")
    synth_frames_path = 'data/batb1k/test_synthetic_frames128'
    csv_path = 'data/batb1k/test_synthetic_poses128.csv'
    transform = Add_Legs('data/batb1k/leg_masks128')
    dataset = skate_data_synth(synth_frames_path, 'data/batb1k/backgrounds128', csv_path, transform=transform)
    dl = torch.utils.data.DataLoader(dataset, 1, shuffle=False, num_workers=2)

    for img, _, _, _, id in tqdm(iter(dl)):
        print(img.shape)
        # torch.save(img, path.join(synth_frames_path, f'{id}.jpg')) 

    # batch_size = 8
    # setup_seed(8)
    # pg = pose_generator('/home/ywongar/skateGAN/data/board_model/skateboard.obj', 128, batch_size, device)
    # csv_path = 'data/batb1k/test_synthetic_poses128.csv'
    # synth_frames_path = 'data/batb1k/test_synthetic_frames128'
    
    # if not path.isfile(csv_path):
    #     fields = ['synthetic_frame_id', 'dist', 'elev', 'azim']
    #     with open(csv_path, 'w', newline='') as file:
    #         writer = csv.DictWriter(file, fieldnames = fields)
    #         writer.writeheader()

    # frame_id = 400
    # for _ in range(100):
    #     dist = torch.rand(batch_size) * 0.4 + 0.4
    #     elev = torch.round(torch.rand(batch_size) * 360)
    #     azim = torch.round(torch.rand(batch_size) * 180)
    #     images, alphas = pg(dist.to(device), elev.to(device), azim.to(device))
    #     images = torch.cat((images, alphas.unsqueeze(-1)), dim=-1)

    #     for i in range(batch_size):
    #         torch.save(images[i,...], path.join(synth_frames_path, f'{frame_id}.pt'))
    #         with open(csv_path, 'a', newline='') as pose_csv:
    #             pose_csv.write(f'{frame_id},{dist[i]},{elev[i]},{azim[i]}\n')
    #         frame_id += 1
    # # print(f"Images: {images.shape}")
    # # print(f'Time: {time.time() - start}')
    # # image_grid(images.cpu())
    # print(f"Saved... {frame_id}")