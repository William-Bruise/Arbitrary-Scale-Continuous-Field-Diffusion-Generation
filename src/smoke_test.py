import torch
from .continuous_field import ContinuousGaussianField
from .diffusion import DDPMCoefficients
from .model import CoeffDenoiser


def main():
    device = "cpu"
    k = 64
    field = ContinuousGaussianField(num_basis=k, device=device)
    coeff = torch.randn(2, k)
    im28 = field.render(coeff, 28, 28)
    im56 = field.render(coeff, 56, 56)
    print("[smoke] render 28:", im28.shape)
    print("[smoke] render 56:", im56.shape)

    model = CoeffDenoiser(k=k)
    diff = DDPMCoefficients(timesteps=10, device=device)
    t = torch.randint(0, 10, (2,))
    noise = torch.randn_like(coeff)
    xt = diff.q_sample(coeff, t, noise)
    pred = model(xt, t)
    print("[smoke] xt:", xt.shape, "pred:", pred.shape)


if __name__ == "__main__":
    main()
