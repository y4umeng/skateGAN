import torch
import sys
import os
if __name__ == '__main__':
    need_pytorch3d=False
    try:
        import pytorch3d
    except ModuleNotFoundError:
        need_pytorch3d=True
    if need_pytorch3d:
        pyt_version_str=torch.__version__.split("+")[0].replace(".", "")
        version_str="".join([
            f"py3{sys.version_info.minor}_cu",
            torch.version.cuda.replace(".",""),
            f"_pyt{pyt_version_str}"
        ])
        # !pip install fvcore iopath
        if sys.platform.startswith("linux"):
            print("Trying to install wheel for PyTorch3D")
            os.system(f'pip install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/{version_str}/download.html')
            pip_list = 'pip freeze'
            need_pytorch3d = not any(i.startswith("pytorch3d==") for  i in pip_list)
        if need_pytorch3d:
            print(f"failed to find/install wheel for {version_str}")
    if need_pytorch3d:
        print("Installing PyTorch3D from source")
        !pip install ninja
        !pip install 'git+https://github.com/facebookresearch/pytorch3d.git@stable'