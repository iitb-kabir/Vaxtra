# Vaxtra

Vaxtra is a lightweight pipeline built around the [SMPL-X](https://smpl-x.is.tue.mpg.de/) parametric 3D body model. It supports 3D mesh generation, pose estimation, and visual processing using PyTorch and SMPL-X tools.

---

## Environment Setup

This project requires Python 3.9 and uses PyTorch with SMPL-X, OpenCV, Trimesh, and other tools. Follow one of the setup options below:

### Using Conda (Recommended)

1. Make sure Conda is installed.
2. Create the environment:

```bash
conda env create -f smplx_env.yml
conda activate smplx_env
```
## After setting up the environment, run the main script:
```bash
python smpl_x.py

