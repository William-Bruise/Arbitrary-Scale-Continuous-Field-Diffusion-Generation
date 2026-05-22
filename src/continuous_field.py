import torch
import torch.nn as nn


class ContinuousGaussianField(nn.Module):
    def __init__(self, num_basis: int = 64, sigma: float = 0.12, channels: int = 1, device=None):
        super().__init__()
        side = int(num_basis ** 0.5)
        assert side * side == num_basis, "num_basis must be a perfect square"
        xs = torch.linspace(0.0, 1.0, side)
        ys = torch.linspace(0.0, 1.0, side)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        centers = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=-1)
        self.register_buffer("centers", centers.to(device))
        self.register_buffer("sigma", torch.tensor(sigma, device=device))
        self.channels = channels

    def basis(self, coords: torch.Tensor) -> torch.Tensor:
        if coords.dim() == 2:
            coords = coords.unsqueeze(0)
        diff = coords.unsqueeze(2) - self.centers.unsqueeze(0).unsqueeze(0)
        sq = (diff**2).sum(dim=-1)
        return torch.exp(-0.5 * sq / (self.sigma**2 + 1e-8))

    def query(self, coeffs: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        # coeffs: [B,C,K]
        if coords.dim() == 2:
            coords = coords.unsqueeze(0).expand(coeffs.shape[0], -1, -1)
        phi = self.basis(coords)  # [B,N,K]
        return torch.einsum("bnk,bck->bcn", phi, coeffs)

    def render(self, coeffs: torch.Tensor, h: int, w: int) -> torch.Tensor:
        ys = torch.linspace(0.0, 1.0, h, device=coeffs.device)
        xs = torch.linspace(0.0, 1.0, w, device=coeffs.device)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        coords = torch.stack([xx, yy], dim=-1).reshape(-1, 2)
        values = self.query(coeffs, coords)
        return torch.sigmoid(values.reshape(coeffs.shape[0], coeffs.shape[1], h, w))


def fit_coeffs_to_image(field: ContinuousGaussianField, image: torch.Tensor, reg: float = 1e-4):
    b, c, h, w = image.shape
    ys = torch.linspace(0.0, 1.0, h, device=image.device)
    xs = torch.linspace(0.0, 1.0, w, device=image.device)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    coords = torch.stack([xx, yy], dim=-1).reshape(-1, 2)

    phi = field.basis(coords).squeeze(0)
    n, k = phi.shape
    gram = phi.T @ phi + reg * torch.eye(k, device=image.device)
    pinv = torch.linalg.inv(gram) @ phi.T

    target = torch.logit(image.clamp(1e-4, 1 - 1e-4)).reshape(b, c, n)
    coeffs = torch.einsum("kn,bcn->bck", pinv, target)
    return coeffs
