import torch
import torch.nn as nn


class ContinuousGaussianField(nn.Module):
    """Fixed centers/scales, variable coefficients define one image field."""

    def __init__(self, num_basis: int = 64, sigma: float = 0.12, device=None):
        super().__init__()
        side = int(num_basis ** 0.5)
        assert side * side == num_basis, "num_basis must be a perfect square"
        xs = torch.linspace(0.0, 1.0, side)
        ys = torch.linspace(0.0, 1.0, side)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        centers = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=-1)
        self.register_buffer("centers", centers.to(device))
        self.register_buffer("sigma", torch.tensor(sigma, device=device))

    def basis(self, coords: torch.Tensor) -> torch.Tensor:
        # coords: [B,N,2] or [N,2]
        if coords.dim() == 2:
            coords = coords.unsqueeze(0)
        diff = coords.unsqueeze(2) - self.centers.unsqueeze(0).unsqueeze(0)  # [B,N,K,2]
        sq = (diff**2).sum(dim=-1)
        phi = torch.exp(-0.5 * sq / (self.sigma**2 + 1e-8))
        return phi

    def query(self, coeffs: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        # coeffs: [B,K], coords:[N,2] or [B,N,2]
        if coords.dim() == 2:
            coords = coords.unsqueeze(0).expand(coeffs.shape[0], -1, -1)
        phi = self.basis(coords)  # [B,N,K]
        values = torch.einsum("bnk,bk->bn", phi, coeffs)
        return values

    def render(self, coeffs: torch.Tensor, h: int, w: int) -> torch.Tensor:
        ys = torch.linspace(0.0, 1.0, h, device=coeffs.device)
        xs = torch.linspace(0.0, 1.0, w, device=coeffs.device)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        coords = torch.stack([xx, yy], dim=-1).reshape(-1, 2)
        values = self.query(coeffs, coords)
        img = values.reshape(coeffs.shape[0], 1, h, w)
        img = torch.sigmoid(img)
        return img


def fit_coeffs_to_image(field: ContinuousGaussianField, image: torch.Tensor, reg: float = 1e-4):
    """Closed-form ridge fit of coefficients for each image.
    image: [B,1,H,W]
    returns coeffs [B,K]
    """
    b, _, h, w = image.shape
    ys = torch.linspace(0.0, 1.0, h, device=image.device)
    xs = torch.linspace(0.0, 1.0, w, device=image.device)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    coords = torch.stack([xx, yy], dim=-1).reshape(-1, 2)

    phi = field.basis(coords).squeeze(0)  # [N,K]
    n, k = phi.shape
    gram = phi.T @ phi + reg * torch.eye(k, device=image.device)
    gram_inv = torch.linalg.inv(gram)
    pinv = gram_inv @ phi.T  # [K,N]

    target = torch.logit(image.clamp(1e-4, 1 - 1e-4)).reshape(b, n)
    coeffs = torch.einsum("kn,bn->bk", pinv, target)
    return coeffs
