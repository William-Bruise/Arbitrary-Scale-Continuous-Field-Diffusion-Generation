import torch


class DDPMCoefficients:
    def __init__(self, timesteps: int = 100, beta_start: float = 1e-4, beta_end: float = 2e-2, device="cpu"):
        self.timesteps = timesteps
        self.betas = torch.linspace(beta_start, beta_end, timesteps, device=device)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
        self.alpha_bars_prev = torch.cat([torch.ones(1, device=device), self.alpha_bars[:-1]], dim=0)

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor):
        a_bar = self.alpha_bars[t].unsqueeze(1)
        return torch.sqrt(a_bar) * x0 + torch.sqrt(1 - a_bar) * noise

    @torch.no_grad()
    def p_sample(self, model, x_t: torch.Tensor, t_scalar: int):
        t = torch.full((x_t.shape[0],), t_scalar, device=x_t.device, dtype=torch.long)
        beta_t = self.betas[t].unsqueeze(1)
        alpha_t = self.alphas[t].unsqueeze(1)
        a_bar_t = self.alpha_bars[t].unsqueeze(1)

        eps_theta = model(x_t, t)
        mean = (1 / torch.sqrt(alpha_t)) * (x_t - (beta_t / torch.sqrt(1 - a_bar_t)) * eps_theta)
        if t_scalar > 0:
            z = torch.randn_like(x_t)
            a_bar_prev = self.alpha_bars_prev[t].unsqueeze(1)
            posterior_var = beta_t * (1 - a_bar_prev) / (1 - a_bar_t)
            sigma = torch.sqrt(posterior_var.clamp_min(1e-20))
            return mean + sigma * z
        return mean

    @torch.no_grad()
    def sample(self, model, batch_size: int, k: int, device):
        x = torch.randn(batch_size, k, device=device)
        for ts in reversed(range(self.timesteps)):
            x = self.p_sample(model, x, ts)
        return x
