import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ContinuousGaussianField(nn.Module):
    """
    Hybrid continuous field:
    - First gaussian_channels are rendered by anisotropic rotated Gaussian basis.
    - Remaining channels are rendered by bicubic upsampling from a coarse feature grid.
    """

    def __init__(
        self,
        num_basis: int = 64,
        sigma: float = 0.12,
        channels: int = 1,
        gaussian_channels: int | None = None,
        coarse_size: int | None = None,
        device=None,
    ):
        super().__init__()
        side = int(num_basis ** 0.5)
        assert side * side == num_basis, "num_basis must be a perfect square"
        xs = torch.linspace(0.0, 1.0, side)
        ys = torch.linspace(0.0, 1.0, side)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        centers = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=-1)
        self.register_buffer("centers", centers.to(device))
        self.channels = channels
        self.num_basis = num_basis
        self.gaussian_channels = channels if gaussian_channels is None else max(0, min(channels, gaussian_channels))
        self.coarse_size = side if coarse_size is None else coarse_size

        # Per-basis anisotropic + rotation parameters (fixed in this prototype but expressive)
        self.register_buffer("sigma_x", torch.full((num_basis,), float(sigma), device=device))
        self.register_buffer("sigma_y", torch.full((num_basis,), float(sigma), device=device))
        self.register_buffer("theta", torch.zeros(num_basis, device=device))

    def basis(self, coords: torch.Tensor) -> torch.Tensor:
        # coords: [B,N,2] or [N,2]
        if coords.dim() == 2:
            coords = coords.unsqueeze(0)
        diff = coords.unsqueeze(2) - self.centers.unsqueeze(0).unsqueeze(0)  # [B,N,K,2]

        dx, dy = diff[..., 0], diff[..., 1]
        ct = torch.cos(self.theta).view(1, 1, -1)
        st = torch.sin(self.theta).view(1, 1, -1)
        x_rot = ct * dx + st * dy
        y_rot = -st * dx + ct * dy

        sx2 = (self.sigma_x.view(1, 1, -1) ** 2 + 1e-8)
        sy2 = (self.sigma_y.view(1, 1, -1) ** 2 + 1e-8)
        q = (x_rot**2) / sx2 + (y_rot**2) / sy2
        return torch.exp(-0.5 * q)

    def query(self, coeffs: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        # coeffs: [B,C,K] for gaussian channels only
        if coords.dim() == 2:
            coords = coords.unsqueeze(0).expand(coeffs.shape[0], -1, -1)
        phi = self.basis(coords)
        return torch.einsum("bnk,bck->bcn", phi, coeffs)

    def render(self, coeffs: torch.Tensor, h: int, w: int) -> torch.Tensor:
        # coeffs: [B,C,K]
        b, c, k = coeffs.shape
        assert k == self.num_basis
        out = []

        # Gaussian-rendered channels
        if self.gaussian_channels > 0:
            ys = torch.linspace(0.0, 1.0, h, device=coeffs.device)
            xs = torch.linspace(0.0, 1.0, w, device=coeffs.device)
            yy, xx = torch.meshgrid(ys, xs, indexing="ij")
            coords = torch.stack([xx, yy], dim=-1).reshape(-1, 2)
            gvals = self.query(coeffs[:, : self.gaussian_channels, :], coords)
            out.append(gvals.reshape(b, self.gaussian_channels, h, w))

        # Bicubic-rendered channels (coarse grid -> target size)
        if c > self.gaussian_channels:
            rem = coeffs[:, self.gaussian_channels :, :]
            side = int(math.sqrt(k))
            rem_grid = rem.reshape(b, c - self.gaussian_channels, side, side)
            bic = F.interpolate(rem_grid, size=(h, w), mode="bicubic", align_corners=False)
            out.append(bic)

        img = torch.cat(out, dim=1) if len(out) > 1 else out[0]
        return torch.sigmoid(img)


def fit_coeffs_to_image(field: ContinuousGaussianField, image: torch.Tensor, reg: float = 1e-4):
    b, c, h, w = image.shape
    ys = torch.linspace(0.0, 1.0, h, device=image.device)
    xs = torch.linspace(0.0, 1.0, w, device=image.device)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    coords = torch.stack([xx, yy], dim=-1).reshape(-1, 2)

    phi = field.basis(coords).squeeze(0)  # [N,K]
    n, k = phi.shape
    gram = phi.T @ phi + reg * torch.eye(k, device=image.device)
    pinv = torch.linalg.inv(gram) @ phi.T

    target = torch.logit(image.clamp(1e-4, 1 - 1e-4)).reshape(b, c, n)
    coeffs = torch.einsum("kn,bcn->bck", pinv, target)
    return coeffs
