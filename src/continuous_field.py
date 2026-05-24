import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ContinuousGaussianField(nn.Module):
    """
    Continuous Gaussian field inspired by GaussianSR's 2D Gaussian parameterization.

    - Basis centers are laid on a regular grid in normalized space [-1, 1].
    - Each basis has trainable anisotropic covariance (sigma_x, sigma_y, rho).
    - Rendering stays fully continuous: the same coefficients can be queried/rendered at any resolution.
    - For stability and efficiency, non-gaussian channels can optionally be rendered by bicubic interpolation.
    """

    def __init__(
        self,
        num_basis: int = 64,
        sigma: float = 0.12,
        channels: int = 1,
        gaussian_channels: int | None = None,
        coarse_size: int | None = None,
        trainable_basis: bool = True,
        normalize_basis: bool = True,
        device=None,
    ):
        super().__init__()
        side = int(num_basis ** 0.5)
        assert side * side == num_basis, "num_basis must be a perfect square"

        xs = torch.linspace(-1.0, 1.0, side)
        ys = torch.linspace(-1.0, 1.0, side)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        centers = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=-1)
        self.register_buffer("centers", centers.to(device))

        self.channels = channels
        self.num_basis = num_basis
        self.gaussian_channels = channels if gaussian_channels is None else max(0, min(channels, gaussian_channels))
        self.coarse_size = side if coarse_size is None else coarse_size
        self.normalize_basis = normalize_basis

        sx = torch.full((num_basis,), float(sigma), device=device)
        sy = torch.full((num_basis,), float(sigma), device=device)
        rho = torch.zeros(num_basis, device=device)

        if trainable_basis:
            self.log_sigma_x = nn.Parameter(torch.log(sx.clamp_min(1e-5)))
            self.log_sigma_y = nn.Parameter(torch.log(sy.clamp_min(1e-5)))
            self.atanh_rho = nn.Parameter(torch.atanh(rho.clamp(-0.99, 0.99)))
        else:
            self.register_buffer("log_sigma_x", torch.log(sx.clamp_min(1e-5)))
            self.register_buffer("log_sigma_y", torch.log(sy.clamp_min(1e-5)))
            self.register_buffer("atanh_rho", torch.atanh(rho.clamp(-0.99, 0.99)))

    def _cov_params(self):
        sigma_x = self.log_sigma_x.exp().clamp_min(1e-4)
        sigma_y = self.log_sigma_y.exp().clamp_min(1e-4)
        rho = torch.tanh(self.atanh_rho).clamp(-0.99, 0.99)
        return sigma_x, sigma_y, rho

    def basis(self, coords: torch.Tensor) -> torch.Tensor:
        # coords: [B,N,2] or [N,2], expected in [-1, 1]
        if coords.dim() == 2:
            coords = coords.unsqueeze(0)
        diff = coords.unsqueeze(2) - self.centers.unsqueeze(0).unsqueeze(0)  # [B,N,K,2]
        dx, dy = diff[..., 0], diff[..., 1]

        sigma_x, sigma_y, rho = self._cov_params()
        sx2 = (sigma_x.view(1, 1, -1) ** 2 + 1e-8)
        sy2 = (sigma_y.view(1, 1, -1) ** 2 + 1e-8)
        r = rho.view(1, 1, -1)
        one_minus_r2 = (1.0 - r * r + 1e-8)

        maha = (dx * dx / sx2 + dy * dy / sy2 - 2.0 * r * dx * dy / (sigma_x.view(1, 1, -1) * sigma_y.view(1, 1, -1) + 1e-8)) / one_minus_r2
        return torch.exp(-0.5 * maha)

    def query(self, coeffs: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        if coords.dim() == 2:
            coords = coords.unsqueeze(0).expand(coeffs.shape[0], -1, -1)
        phi = self.basis(coords)
        if self.normalize_basis:
            phi = phi / (phi.sum(dim=-1, keepdim=True).clamp_min(1e-6))
        return torch.einsum("bnk,bck->bcn", phi, coeffs)

    def render(self, coeffs: torch.Tensor, h: int, w: int) -> torch.Tensor:
        b, c, k = coeffs.shape
        assert k == self.num_basis
        out = []

        if self.gaussian_channels > 0:
            ys = torch.linspace(-1.0, 1.0, h, device=coeffs.device)
            xs = torch.linspace(-1.0, 1.0, w, device=coeffs.device)
            yy, xx = torch.meshgrid(ys, xs, indexing="ij")
            coords = torch.stack([xx, yy], dim=-1).reshape(-1, 2)
            gvals = self.query(coeffs[:, : self.gaussian_channels, :], coords)
            out.append(gvals.reshape(b, self.gaussian_channels, h, w))

        if c > self.gaussian_channels:
            rem = coeffs[:, self.gaussian_channels :, :]
            side = int(math.sqrt(k))
            rem_grid = rem.reshape(b, c - self.gaussian_channels, side, side)
            out.append(F.interpolate(rem_grid, size=(h, w), mode="bicubic", align_corners=False))

        img = torch.cat(out, dim=1) if len(out) > 1 else out[0]
        return torch.sigmoid(img)


def fit_coeffs_to_image(field: ContinuousGaussianField, image: torch.Tensor, reg: float = 1e-4):
    b, c, h, w = image.shape
    ys = torch.linspace(-1.0, 1.0, h, device=image.device)
    xs = torch.linspace(-1.0, 1.0, w, device=image.device)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    coords = torch.stack([xx, yy], dim=-1).reshape(-1, 2)

    phi = field.basis(coords).squeeze(0)  # [N,K]
    n, k = phi.shape
    gram = phi.T @ phi + reg * torch.eye(k, device=image.device)
    pinv = torch.linalg.inv(gram) @ phi.T

    target = torch.logit(image.clamp(1e-4, 1 - 1e-4)).reshape(b, c, n)
    return torch.einsum("kn,bcn->bck", pinv, target)
