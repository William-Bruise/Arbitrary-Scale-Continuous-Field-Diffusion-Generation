import argparse
import os
import torch
import matplotlib.pyplot as plt

from .continuous_field import ContinuousGaussianField
from .model import CoeffDenoiser
from .diffusion import DDPMCoefficients


def save_multires(field, coeff, out_path, sizes=(28, 42, 56, 84)):
    imgs = [field.render(coeff, s, s)[0, 0].detach().cpu() for s in sizes]
    fig, axes = plt.subplots(1, len(sizes), figsize=(3 * len(sizes), 3))
    for ax, im, s in zip(axes, imgs, sizes):
        ax.imshow(im, cmap="gray", vmin=0, vmax=1)
        ax.set_title(f"{s}x{s}")
        ax.axis("off")
    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--outdir", type=str, default="runs/sample")
    p.add_argument("--device", type=str, default="cpu")
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    ckpt = torch.load(args.ckpt, map_location=args.device)

    field = ContinuousGaussianField(num_basis=ckpt["num_basis"], device=args.device).to(args.device)
    model = CoeffDenoiser(k=ckpt["num_basis"]).to(args.device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    diff = DDPMCoefficients(timesteps=ckpt["timesteps"], device=args.device)
    with torch.no_grad():
        coeff = diff.sample(model, batch_size=1, k=ckpt["num_basis"], device=args.device)
        if ckpt.get("normalize_coeffs", False):
            mean = ckpt["coeff_mean"].to(args.device).unsqueeze(0)
            std = ckpt["coeff_std"].to(args.device).unsqueeze(0)
            coeff = coeff * std + mean
    save_multires(field, coeff, f"{args.outdir}/sample_multires.png")
    print("saved:", f"{args.outdir}/sample_multires.png")


if __name__ == "__main__":
    main()
