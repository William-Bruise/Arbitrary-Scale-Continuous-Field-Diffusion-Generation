import math
import torch
import torch.nn as nn
import torch.nn.functional as F


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
        g = min(8, channels)
        self.norm1 = nn.GroupNorm(g, channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.tproj = nn.Linear(tdim, channels)
        self.norm2 = nn.GroupNorm(g, channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(self.act(self.norm1(x)))
        h = h + self.tproj(temb).unsqueeze(-1).unsqueeze(-1)
        h = self.conv2(self.act(self.norm2(h)))
        return x + h


class LatentUNetDenoiser(nn.Module):
    def __init__(self, k: int, channels: int = 1, base: int = 64, levels: int = 2, resblocks_per_level: int = 2, tdim: int = 128):
        super().__init__()
        side = int(k ** 0.5)
        assert side * side == k
        self.side, self.k, self.channels = side, k, channels
        self.levels = levels
        self.resblocks_per_level = resblocks_per_level

        self.temb = nn.Sequential(TimeEmbedding(tdim), nn.Linear(tdim, tdim), nn.SiLU(), nn.Linear(tdim, tdim))

        chs = [base * (2 ** i) for i in range(levels)]
        self.in_conv = nn.Conv2d(channels, chs[0], 3, padding=1)

        self.down_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        for i in range(levels):
            blocks = nn.ModuleList([ResBlock(chs[i], tdim) for _ in range(resblocks_per_level)])
            self.down_blocks.append(blocks)
            if i < levels - 1:
                self.downsamples.append(nn.Conv2d(chs[i], chs[i + 1], 4, stride=2, padding=1))

        self.mid_blocks = nn.ModuleList([ResBlock(chs[-1], tdim), ResBlock(chs[-1], tdim)])

        self.upsamples = nn.ModuleList()
        self.up_reduce = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        for i in range(levels - 1, 0, -1):
            self.upsamples.append(nn.ConvTranspose2d(chs[i], chs[i - 1], 4, stride=2, padding=1))
            self.up_reduce.append(nn.Conv2d(chs[i - 1] * 2, chs[i - 1], 1))
            self.up_blocks.append(nn.ModuleList([ResBlock(chs[i - 1], tdim) for _ in range(resblocks_per_level)]))

        g = min(8, chs[0])
        self.out_norm = nn.GroupNorm(g, chs[0])
        self.out_conv = nn.Conv2d(chs[0], channels, 3, padding=1)
        self.act = nn.SiLU()

    def forward(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        b = x_t.shape[0]
        x = x_t.reshape(b, self.channels, self.side, self.side)
        temb = self.temb(t)

        x = self.in_conv(x)
        skips = []
        for i, blocks in enumerate(self.down_blocks):
            for blk in blocks:
                x = blk(x, temb)
            skips.append(x)
            if i < len(self.downsamples):
                x = self.downsamples[i](x)

        for blk in self.mid_blocks:
            x = blk(x, temb)

        for i in range(len(self.upsamples)):
            x = self.upsamples[i](x)
            skip = skips[-(i + 2)]
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode='nearest')
            x = torch.cat([x, skip], dim=1)
            x = self.up_reduce[i](x)
            for blk in self.up_blocks[i]:
                x = blk(x, temb)

        out = self.out_conv(self.act(self.out_norm(x)))
        return out.reshape(b, self.channels * self.k)


CoeffDenoiser = LatentUNetDenoiser
