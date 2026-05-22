import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms


def make_dataset(name: str = "mnist", root: str = "./data", train: bool = True) -> Dataset:
    name = name.lower()
    if name == "mnist":
        tf = transforms.ToTensor()
        return datasets.MNIST(root=root, train=train, download=True, transform=tf)
    if name == "fashionmnist":
        tf = transforms.ToTensor()
        return datasets.FashionMNIST(root=root, train=train, download=True, transform=tf)
    if name == "kmnist":
        tf = transforms.ToTensor()
        return datasets.KMNIST(root=root, train=train, download=True, transform=tf)
    if name == "cifar10":
        # convert to grayscale 32x32 so current single-channel continuous field pipeline remains unchanged
        tf = transforms.Compose([transforms.Grayscale(num_output_channels=1), transforms.ToTensor()])
        return datasets.CIFAR10(root=root, train=train, download=True, transform=tf)
    raise ValueError(f"Unsupported dataset: {name}. Choose from [mnist, fashionmnist, kmnist, cifar10]")


def make_mnist_dataset(root: str = "./data", train: bool = True) -> Dataset:
    # backward compatibility
    return make_dataset("mnist", root=root, train=train)


def pixel_grid_to_continuous(h: int, w: int, device=None):
    ys = torch.linspace(0.0, 1.0, h, device=device)
    xs = torch.linspace(0.0, 1.0, w, device=device)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    coords = torch.stack([xx, yy], dim=-1)
    return coords
