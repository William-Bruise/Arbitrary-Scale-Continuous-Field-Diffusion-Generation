import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms


def make_dataset(name: str = "mnist", root: str = "./data", train: bool = True, image_size: int = 32) -> Dataset:
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
        tf = transforms.ToTensor()  # keep RGB
        return datasets.CIFAR10(root=root, train=train, download=True, transform=tf)
    if name == "celeba":
        tf = transforms.Compose([transforms.CenterCrop(178), transforms.Resize((image_size, image_size)), transforms.ToTensor()])
        split = "train" if train else "valid"
        return datasets.CelebA(root=root, split=split, download=True, transform=tf)
    if name == "stl10":
        tf = transforms.Compose([transforms.Resize((image_size, image_size)), transforms.ToTensor()])
        split = "unlabeled" if train else "test"
        return datasets.STL10(root=root, split=split, download=True, transform=tf)
    raise ValueError("Unsupported dataset: choose from [mnist, fashionmnist, kmnist, cifar10, celeba, stl10]")


def make_mnist_dataset(root: str = "./data", train: bool = True) -> Dataset:
    return make_dataset("mnist", root=root, train=train)


def pixel_grid_to_continuous(h: int, w: int, device=None):
    ys = torch.linspace(0.0, 1.0, h, device=device)
    xs = torch.linspace(0.0, 1.0, w, device=device)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    coords = torch.stack([xx, yy], dim=-1)
    return coords
