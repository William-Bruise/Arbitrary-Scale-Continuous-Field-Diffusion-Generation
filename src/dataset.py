import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms


def make_mnist_dataset(root: str = "./data", train: bool = True) -> Dataset:
    transform = transforms.ToTensor()
    return datasets.MNIST(root=root, train=train, download=True, transform=transform)


def pixel_grid_to_continuous(h: int, w: int, device=None):
    ys = torch.linspace(0.0, 1.0, h, device=device)
    xs = torch.linspace(0.0, 1.0, w, device=device)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    coords = torch.stack([xx, yy], dim=-1)
    return coords
