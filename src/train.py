import argparse
import os
import math
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


def grid_tv_smooth(coeffs: torch.Tensor) -> torch.Tensor:
    # coeffs: [B, K], K should be perfect square
    b, k = coeffs.shape
    side = int(math.sqrt(k))
    assert side * side == k, "num_basis must be a perfect square for 2D TV smooth"
    grid = coeffs.reshape(b, side, side)
    dh = (grid[:, 1:, :] - grid[:, :-1, :]).pow(2).mean()
    dw = (grid[:, :, 1:] - grid[:, :, :-1]).pow(2).mean()
    return dh + dw


def estimate_coeff_stats(field, dl, device: str, max_batches: int = 100):
    """Estimate dataset-level coeff mean/std for normalization."""
    sums = None
    sq_sums = None
    count = 0
    with torch.no_grad():
        for bi, (x, _) in enumerate(dl):
            if bi >= max_batches:
                break
            x = x.to(device)
            coeffs = fit_coeffs_to_image(field, x)  # [B,K]
            if sums is None:
                sums = coeffs.sum(dim=0)
                sq_sums = (coeffs**2).sum(dim=0)
            else:
                sums += coeffs.sum(dim=0)
                sq_sums += (coeffs**2).sum(dim=0)
            count += coeffs.shape[0]
    mean = sums / max(count, 1)
    var = sq_sums / max(count, 1) - mean**2
    std = torch.sqrt(var.clamp_min(1e-8))
    return mean, std


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=10, help="full training epochs")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--timesteps", type=int, default=200)
    p.add_argument("--num-basis", type=int, default=144)
    p.add_argument("--sigma", type=float, default=0.08, help="gaussian basis width for continuous field")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--outdir", type=str, default="runs/full_train")
    p.add_argument("--sample-every", type=int, default=500, help="sample every N optimization steps")
    p.add_argument("--ckpt-every", type=int, default=1000, help="checkpoint every N optimization steps")
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--hidden", type=int, default=512, help="denoiser hidden dim")
    p.add_argument("--depth", type=int, default=4, help="denoiser depth (hidden blocks)")
    p.add_argument("--smooth-weight", type=float, default=1e-5, help="weight for 2D TV smooth prior on coeff grid")
    p.add_argument("--normalize-coeffs", action="store_true", help="z-score normalize coefficients before diffusion")
    p.add_argument("--stats-batches", type=int, default=100, help="batches used to estimate coeff mean/std")
    args = p.parse_args()

    os.makedirs(f"{args.outdir}/samples", exist_ok=True)
    os.makedirs(f"{args.outdir}/checkpoints", exist_ok=True)
    os.makedirs(f"{args.outdir}/logs", exist_ok=True)

    ds = make_mnist_dataset(train=True)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True)

    field = ContinuousGaussianField(num_basis=args.num_basis, sigma=args.sigma, device=args.device).to(args.device)
    model = CoeffDenoiser(k=args.num_basis, hidden=args.hidden, depth=args.depth).to(args.device)
    diff = DDPMCoefficients(timesteps=args.timesteps, device=args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    coeff_mean = torch.zeros(args.num_basis, device=args.device)
    coeff_std = torch.ones(args.num_basis, device=args.device)
    if args.normalize_coeffs:
        coeff_mean, coeff_std = estimate_coeff_stats(field, dl, args.device, max_batches=args.stats_batches)
        print(f"[coeff-stats] estimated from <= {args.stats_batches} batches")
        print(f"[coeff-stats] mean(abs)={coeff_mean.abs().mean().item():.4f} std(mean)={coeff_std.mean().item():.4f}")

    print("[shape] centers:", field.centers.shape)
    print(f"[train] dataset={len(ds)} batch_size={args.batch_size} steps_per_epoch={len(dl)} num_basis={args.num_basis} sigma={args.sigma} hidden={args.hidden} depth={args.depth}")

    global_step = 0
    log_path = f"{args.outdir}/logs/train_log.txt"

    for epoch in range(1, args.epochs + 1):
        for x, _ in dl:
            x = x.to(args.device)
            coeffs = fit_coeffs_to_image(field, x)
            coeffs_train = (coeffs - coeff_mean.unsqueeze(0)) / coeff_std.unsqueeze(0) if args.normalize_coeffs else coeffs

            b, _ = coeffs_train.shape
            t = torch.randint(0, args.timesteps, (b,), device=args.device)
            noise = torch.randn_like(coeffs_train)
            x_t = diff.q_sample(coeffs_train, t, noise)

            pred = model(x_t, t)
            loss_ddpm = ((pred - noise) ** 2).mean()
            smooth = grid_tv_smooth(coeffs)
            loss = loss_ddpm + args.smooth_weight * smooth

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            global_step += 1
            if global_step % 50 == 0 or global_step == 1:
                msg = (
                    f"epoch={epoch}/{args.epochs} step={global_step} "
                    f"loss={loss.item():.6f} ddpm={loss_ddpm.item():.6f} smooth={smooth.item():.6f} "
                    f"coeffs={tuple(coeffs_train.shape)} x_t={tuple(x_t.shape)}"
                )
                print(msg)
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(msg + "\n")

            if global_step % args.sample_every == 0:
                with torch.no_grad():
                    sample_coeff = diff.sample(model, batch_size=1, k=args.num_basis, device=args.device)
                    if args.normalize_coeffs:
                        sample_coeff = sample_coeff * coeff_std.unsqueeze(0) + coeff_mean.unsqueeze(0)
                    save_multires(field, sample_coeff, f"{args.outdir}/samples/step_{global_step}_multires.png")

            if global_step % args.ckpt_every == 0:
                ckpt = {
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "steps": global_step,
                    "num_basis": args.num_basis,
                    "sigma": args.sigma,
                    "hidden": args.hidden,
                    "depth": args.depth,
                    "timesteps": args.timesteps,
                    "normalize_coeffs": args.normalize_coeffs,
                    "coeff_mean": coeff_mean.detach().cpu(),
                    "coeff_std": coeff_std.detach().cpu(),
                }
                torch.save(ckpt, f"{args.outdir}/checkpoints/step_{global_step}.pt")

    ckpt = {
        "model": model.state_dict(),
        "epoch": args.epochs,
        "steps": global_step,
        "num_basis": args.num_basis,
                    "sigma": args.sigma,
                    "hidden": args.hidden,
                    "depth": args.depth,
        "timesteps": args.timesteps,
        "normalize_coeffs": args.normalize_coeffs,
        "coeff_mean": coeff_mean.detach().cpu(),
        "coeff_std": coeff_std.detach().cpu(),
    }
    torch.save(ckpt, f"{args.outdir}/checkpoints/final.pt")

    with torch.no_grad():
        sample_coeff = diff.sample(model, batch_size=1, k=args.num_basis, device=args.device)
        if args.normalize_coeffs:
            sample_coeff = sample_coeff * coeff_std.unsqueeze(0) + coeff_mean.unsqueeze(0)
        save_multires(field, sample_coeff, f"{args.outdir}/samples/final_multires.png")

    print(f"[done] total_steps={global_step} saved_final={args.outdir}/checkpoints/final.pt")


if __name__ == "__main__":
    main()
