import math
import torch
import torch.nn as nn


class TimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(0, half, device=t.device) / half)
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
        if self.dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=1)
        return emb


class CoeffDenoiser(nn.Module):
    def __init__(self, k: int, hidden: int = 256, tdim: int = 64):
        super().__init__()
        self.temb = TimeEmbedding(tdim)
        self.net = nn.Sequential(
            nn.Linear(k + tdim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, k),
        )

    def forward(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        emb = self.temb(t)
        h = torch.cat([x_t, emb], dim=1)
        return self.net(h)
