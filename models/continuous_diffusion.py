import math
import torch
import torch.nn as nn


def sinusoidal(t, dim):
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / max(half,1))
    x = t[:,None].float() * freqs[None,:]
    emb = torch.cat([x.sin(), x.cos()], dim=-1)
    return emb if dim % 2 == 0 else torch.cat([emb, torch.zeros_like(emb[:,:1])], dim=-1)


class CoordMLP(nn.Module):
    def __init__(self, in_dim, hidden, out_dim, depth):
        super().__init__()
        layers=[]
        d=in_dim
        for _ in range(depth-1):
            layers += [nn.Linear(d, hidden), nn.SiLU()]
            d=hidden
        layers += [nn.Linear(d, out_dim)]
        self.net=nn.Sequential(*layers)

    def forward(self,x):
        return self.net(x)


class ContinuousFieldDenoiser(nn.Module):
    def __init__(self, global_latent_dim=256, hidden_dim=256, depth=6):
        super().__init__()
        self.global_encoder = nn.Sequential(
            nn.Conv2d(3,64,4,2,1), nn.SiLU(),
            nn.Conv2d(64,128,4,2,1), nn.SiLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(128,global_latent_dim)
        )
        self.coord = CoordMLP(in_dim=2+3+global_latent_dim+128, hidden=hidden_dim, out_dim=3, depth=depth)
        self.time_dim = 128

    def encode_global(self, x0):
        return self.global_encoder(x0)

    def forward(self, x_t_vals, coords, t, global_latent):
        b,n,_ = x_t_vals.shape
        temb = sinusoidal(t, self.time_dim)[:,None,:].expand(b,n,-1)
        g = global_latent[:,None,:].expand(b,n,-1)
        inp = torch.cat([coords, x_t_vals, temb, g], dim=-1)
        return self.coord(inp)
