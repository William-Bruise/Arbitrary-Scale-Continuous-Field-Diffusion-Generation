import argparse
import os
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import resnet18

from .continuous_field import ContinuousGaussianField
from .diffusion import DDPMCoefficients
from .model import LatentUNetDenoiser


def load_generator(ckpt_path: str, device: str):
    ckpt = torch.load(ckpt_path, map_location=device)
    channels = ckpt.get("channels", 1)
    field = ContinuousGaussianField(
        num_basis=ckpt["num_basis"],
        sigma=ckpt.get("sigma", 0.08),
        channels=channels,
        device=device,
    ).to(device)
    denoiser = LatentUNetDenoiser(
        k=ckpt["num_basis"],
        channels=channels,
        base=ckpt.get("unet_base", 64),
    ).to(device)
    denoiser.load_state_dict(ckpt["model"])
    denoiser.eval()
    diff = DDPMCoefficients(timesteps=ckpt["timesteps"], device=device)
    return ckpt, field, denoiser, diff


def sample_images(field, denoiser, diff, ckpt, n, h, w, device):
    ch = ckpt.get("channels", 1)
    k = ckpt["num_basis"]
    with torch.no_grad():
        coeff = diff.sample(denoiser, batch_size=n, k=ch * k, device=device)
        if ckpt.get("normalize_coeffs", False):
            mean = ckpt["coeff_mean"].to(device).unsqueeze(0)
            std = ckpt["coeff_std"].to(device).unsqueeze(0)
            coeff = coeff * std + mean
        coeff = coeff.reshape(n, ch, k)
        img = field.render(coeff, h, w)
    return img


def save_sample_grid(images: torch.Tensor, out_path: str, n_show: int = 16, title: str = ""):
    imgs = images[:n_show].detach().cpu()  # [N,C,H,W]
    n_show = imgs.shape[0]
    cols = int(n_show ** 0.5)
    rows = (n_show + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
    if rows == 1 and cols == 1:
        axes = [[axes]]
    elif rows == 1:
        axes = [axes]
    elif cols == 1:
        axes = [[ax] for ax in axes]

    idx = 0
    for r in range(rows):
        for c in range(cols):
            ax = axes[r][c]
            if idx < n_show:
                im = imgs[idx]
                if im.shape[0] == 1:
                    ax.imshow(im[0], cmap="gray", vmin=0, vmax=1)
                else:
                    ax.imshow(im.permute(1, 2, 0).clamp(0, 1))
            ax.axis("off")
            idx += 1
    if title:
        fig.suptitle(title)
    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def build_dataset_pair(dataset: str, root: str, image_size: int, channels: int):
    dname = dataset.lower()
    if dname in ["mnist", "fashionmnist", "kmnist"]:
        tf = transforms.ToTensor()
        if dname == "mnist":
            return (
                datasets.MNIST(root=root, train=True, download=True, transform=tf),
                datasets.MNIST(root=root, train=False, download=True, transform=tf),
            )
        if dname == "fashionmnist":
            return (
                datasets.FashionMNIST(root=root, train=True, download=True, transform=tf),
                datasets.FashionMNIST(root=root, train=False, download=True, transform=tf),
            )
        return (
            datasets.KMNIST(root=root, train=True, download=True, transform=tf),
            datasets.KMNIST(root=root, train=False, download=True, transform=tf),
        )

    if dname == "cifar10":
        tf = transforms.Compose([transforms.Resize((image_size, image_size)), transforms.ToTensor()])
        return (
            datasets.CIFAR10(root=root, train=True, download=True, transform=tf),
            datasets.CIFAR10(root=root, train=False, download=True, transform=tf),
        )

    if dname == "stl10":
        tf = transforms.Compose([transforms.Resize((image_size, image_size)), transforms.ToTensor()])
        return (
            datasets.STL10(root=root, split="train", download=True, transform=tf),
            datasets.STL10(root=root, split="test", download=True, transform=tf),
        )

    if dname == "celeba":
        tf = transforms.Compose([transforms.CenterCrop(178), transforms.Resize((image_size, image_size)), transforms.ToTensor()])
        return (
            datasets.CelebA(root=root, split="train", download=True, transform=tf),
            datasets.CelebA(root=root, split="valid", download=True, transform=tf),
        )

    raise ValueError(f"Unsupported dataset for eval classifier: {dataset}")


def train_classifier(device: str, epochs: int, batch_size: int, dataset: str, root: str, image_size: int, channels: int):
    train_ds, test_ds = build_dataset_pair(dataset, root, image_size, channels)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    model = resnet18(num_classes=10)
    model.conv1 = torch.nn.Conv2d(channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = torch.nn.Identity()
    model.to(device)

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(epochs):
        model.train()
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()

    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in test_dl:
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.numel()
    return model, correct / max(total, 1)


def classifier_metrics(clf, images):
    with torch.no_grad():
        logits = clf(images)
        probs = F.softmax(logits, dim=1)
        conf, pred = probs.max(dim=1)
    hist = torch.bincount(pred.cpu(), minlength=10).float()
    hist = hist / hist.sum().clamp_min(1e-8)
    return {
        "mean_confidence": conf.mean().item(),
        "label_entropy": (-(hist * (hist + 1e-8).log()).sum()).item(),
        "label_hist": hist.tolist(),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--dataset", type=str, default="mnist")
    p.add_argument("--data-root", type=str, default="./data")
    p.add_argument("--num-samples", type=int, default=1024)
    p.add_argument("--clf-epochs", type=int, default=2)
    p.add_argument("--outdir", type=str, default="runs/eval")
    p.add_argument("--train-size", type=int, default=28)
    p.add_argument("--ood-sizes", type=str, default="42,56,70,84,112")
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    ckpt, field, denoiser, diff = load_generator(args.ckpt, args.device)
    channels = ckpt.get("channels", 1)
    image_size = ckpt.get("image_size", args.train_size)

    clf, acc = train_classifier(
        args.device,
        epochs=args.clf_epochs,
        batch_size=256,
        dataset=args.dataset,
        root=args.data_root,
        image_size=image_size,
        channels=channels,
    )
    print(f"[eval] classifier test acc={acc:.4f}")

    sizes = [int(x.strip()) for x in args.ood_sizes.split(",") if x.strip()]
    all_sizes = [args.train_size] + [s for s in sizes if s != args.train_size]

    rows = []
    for s in all_sizes:
        imgs = sample_images(field, denoiser, diff, ckpt, args.num_samples, s, s, args.device)
        save_sample_grid(imgs, os.path.join(args.outdir, f"samples_size_{s}.png"), n_show=16, title=f"Generated samples @ {s}x{s}")
        if s != image_size:
            imgs_for_clf = F.interpolate(imgs, size=(image_size, image_size), mode="bilinear", align_corners=False)
        else:
            imgs_for_clf = imgs
        m = classifier_metrics(clf, imgs_for_clf)
        row = {"size": s, "mean_confidence": m["mean_confidence"], "label_entropy": m["label_entropy"], "label_hist": m["label_hist"]}
        rows.append(row)
        print(f"[eval] size={s} conf={row['mean_confidence']:.4f} entropy={row['label_entropy']:.4f}")

    out_path = os.path.join(args.outdir, "metrics.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"classifier_test_acc={acc:.6f}\n")
        for r in rows:
            f.write(f"size={r['size']} mean_confidence={r['mean_confidence']:.6f} label_entropy={r['label_entropy']:.6f}\n")
            f.write("label_hist=" + ",".join([f"{x:.6f}" for x in r["label_hist"]]) + "\n")

    print("[eval] saved metrics:", out_path)
    print("[eval] saved sample grids:", os.path.join(args.outdir, "samples_size_*.png"))


if __name__ == "__main__":
    main()
