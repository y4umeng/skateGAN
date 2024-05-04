import random
import torch
import numpy as np
import torch.nn as nn
# io utils
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
        blend_params = BlendParams(sigma=1e-4, gamma=1e-4, background_color=white)

        # Here we set the output image to be of size 256 x 256 based on config.json
        raster_settings = RasterizationSettings(
            image_size = img_shape,
            blur_radius = 0.0,
            faces_per_pixel = 100,
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
                raster_settings=raster_settings
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
        cameras = FoVPerspectiveCameras(device=self.device, R=R, T=T)

        # image = self.renderer(meshes_world=self.meshes.clone(), R=R, T=T) 
        images = self.renderer(self.meshes, cameras=cameras, lights=self.lights)
        return images