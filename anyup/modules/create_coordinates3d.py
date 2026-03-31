import torch

def create_coordinates_3d(t, h, w, start=0.0, end=1.0, device=None, dtype=None):
    z = torch.linspace(start, end, t, device=device, dtype=dtype)
    x = torch.linspace(start, end, h, device=device, dtype=dtype)
    y = torch.linspace(start, end, w, device=device, dtype=dtype)
    zz, xx, yy = torch.meshgrid(z, x, y, indexing="ij")
    return torch.stack((zz, xx, yy), -1).view(1, t * h * w, 3)