from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class ImageFolderDataset(Dataset):
    def __init__(self, root: str, split: str = "train", image_size: int = 128):
        self.root = Path(root) / split
        if not self.root.exists():
            raise FileNotFoundError(f"Missing split folder: {self.root}")
        self.files = sorted([p for p in self.root.rglob('*') if p.suffix.lower() in {'.png','.jpg','.jpeg','.webp'}])
        if not self.files:
            raise RuntimeError(f"No image files found in {self.root}")
        self.tf = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5]),
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        im = Image.open(self.files[idx]).convert('RGB')
        return self.tf(im)
