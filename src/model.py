import math
import torch
import torch.nn as nn


class TimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(0, half, device=t.device) / max(half, 1))
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
        if self.dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=1)
        return emb


class ResBlock(nn.Module):
    def __init__(self, channels: int, tdim: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.tproj = nn.Linear(tdim, channels)
        self.norm2 = nn.GroupNorm(8, channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(self.act(self.norm1(x)))
        h = h + self.tproj(temb).unsqueeze(-1).unsqueeze(-1)
        h = self.conv2(self.act(self.norm2(h)))
        return x + h


class LatentUNetDenoiser(nn.Module):
    """UNet-like denoiser over coefficient grid latent [B, S, S]."""

    def __init__(self, k: int, base: int = 64, tdim: int = 128):
        super().__init__()
        side = int(k ** 0.5)
        assert side * side == k, "num_basis must be perfect square for latent UNet"
        self.side = side
        self.k = k

        self.temb = nn.Sequential(TimeEmbedding(tdim), nn.Linear(tdim, tdim), nn.SiLU(), nn.Linear(tdim, tdim))

        self.in_conv = nn.Conv2d(1, base, 3, padding=1)
        self.down1 = ResBlock(base, tdim)
        self.downsample = nn.Conv2d(base, base * 2, 4, stride=2, padding=1)

        self.mid1 = ResBlock(base * 2, tdim)
        self.mid2 = ResBlock(base * 2, tdim)

        self.upsample = nn.ConvTranspose2d(base * 2, base, 4, stride=2, padding=1)
        self.up1 = ResBlock(base * 2, tdim)
        self.up_reduce = nn.Conv2d(base * 2, base, 1)
        self.out_norm = nn.GroupNorm(8, base)
        self.out_conv = nn.Conv2d(base, 1, 3, padding=1)
        self.act = nn.SiLU()

    def forward(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        b = x_t.shape[0]
        x = x_t.reshape(b, 1, self.side, self.side)
        temb = self.temb(t)

        x0 = self.in_conv(x)
        x1 = self.down1(x0, temb)
        x2 = self.downsample(x1)

        xm = self.mid1(x2, temb)
        xm = self.mid2(xm, temb)

        xu = self.upsample(xm)
        if xu.shape[-2:] != x1.shape[-2:]:
            xu = torch.nn.functional.interpolate(xu, size=x1.shape[-2:], mode="nearest")
        xu = torch.cat([xu, x1], dim=1)
        xu = self.up1(xu, temb)
        xu = self.up_reduce(xu)

        out = self.out_conv(self.act(self.out_norm(xu)))
        return out.reshape(b, self.k)


# backward-compat alias
CoeffDenoiser = LatentUNetDenoiser
