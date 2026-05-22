import argparse
import os
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from .dataset import make_mnist_dataset
from .continuous_field import ContinuousGaussianField, fit_coeffs_to_image
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
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--timesteps", type=int, default=100)
    p.add_argument("--num-basis", type=int, default=64)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--outdir", type=str, default="runs/exp")
    p.add_argument("--sample-every", type=int, default=1)
    args = p.parse_args()

    os.makedirs(f"{args.outdir}/samples", exist_ok=True)
    os.makedirs(f"{args.outdir}/checkpoints", exist_ok=True)
    os.makedirs(f"{args.outdir}/logs", exist_ok=True)

    ds = make_mnist_dataset(train=True)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)

    field = ContinuousGaussianField(num_basis=args.num_basis, device=args.device).to(args.device)
    model = CoeffDenoiser(k=args.num_basis).to(args.device)
    diff = DDPMCoefficients(timesteps=args.timesteps, device=args.device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    print("[shape] centers:", field.centers.shape)

    step = 0
    it = iter(dl)
    log_path = f"{args.outdir}/logs/train_log.txt"
    while step < args.steps:
        try:
            x, _ = next(it)
        except StopIteration:
            it = iter(dl)
            x, _ = next(it)

        x = x.to(args.device)
        coeffs = fit_coeffs_to_image(field, x)
        b, k = coeffs.shape
        t = torch.randint(0, args.timesteps, (b,), device=args.device)
        noise = torch.randn_like(coeffs)
        x_t = diff.q_sample(coeffs, t, noise)

        pred = model(x_t, t)
        loss_ddpm = ((pred - noise) ** 2).mean()
        smooth = (coeffs[:, 1:] - coeffs[:, :-1]).pow(2).mean()
        loss = loss_ddpm + 1e-4 * smooth

        opt.zero_grad()
        loss.backward()
        opt.step()

        step += 1
        msg = f"step={step} loss={loss.item():.6f} coeffs={tuple(coeffs.shape)} x_t={tuple(x_t.shape)}"
        print(msg)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(msg + "\n")

        if step % args.sample_every == 0:
            with torch.no_grad():
                sample_coeff = diff.sample(model, batch_size=1, k=args.num_basis, device=args.device)
                save_multires(field, sample_coeff, f"{args.outdir}/samples/step_{step}_multires.png")

            ckpt = {
                "model": model.state_dict(),
                "steps": step,
                "num_basis": args.num_basis,
                "timesteps": args.timesteps,
            }
            torch.save(ckpt, f"{args.outdir}/checkpoints/step_{step}.pt")


if __name__ == "__main__":
    main()
