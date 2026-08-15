"""Same Gaussian diffusion process as mini-diffusion, vendored for the
inference path this service actually uses (DDIM sampling from a trained
checkpoint) — see github.com/Shivam101s/mini-diffusion for the training code.
"""

import torch
import torch.nn.functional as F


def linear_beta_schedule(timesteps: int, beta_start: float = 1e-4, beta_end: float = 0.02):
    return torch.linspace(beta_start, beta_end, timesteps)


class GaussianDiffusion:
    def __init__(self, timesteps: int = 1000, device: str = "cpu"):
        self.timesteps = timesteps
        self.device = device

        betas = linear_beta_schedule(timesteps).to(device)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.betas = betas
        self.alphas_cumprod = alphas_cumprod

    @torch.no_grad()
    def ddim_sample(self, model, shape, device, ddim_steps: int = 50, eta: float = 0.0):
        step_indices = torch.linspace(0, self.timesteps - 1, ddim_steps, device=device).long()
        step_indices = torch.flip(step_indices, dims=(0,))

        x = torch.randn(shape, device=device)
        for idx, i in enumerate(step_indices):
            t = torch.full((shape[0],), int(i), device=device, dtype=torch.long)
            predicted_noise = model(x, t)

            alpha_cumprod_t = self.alphas_cumprod[i]
            if idx + 1 < len(step_indices):
                alpha_cumprod_prev = self.alphas_cumprod[step_indices[idx + 1]]
            else:
                alpha_cumprod_prev = torch.tensor(1.0, device=device)

            pred_x0 = (x - torch.sqrt(1 - alpha_cumprod_t) * predicted_noise) / torch.sqrt(
                alpha_cumprod_t
            )
            sigma = eta * torch.sqrt(
                (1 - alpha_cumprod_prev)
                / (1 - alpha_cumprod_t)
                * (1 - alpha_cumprod_t / alpha_cumprod_prev)
            )
            dir_xt = torch.sqrt(1 - alpha_cumprod_prev - sigma ** 2) * predicted_noise
            noise = torch.randn_like(x) if eta > 0 else 0.0
            x = torch.sqrt(alpha_cumprod_prev) * pred_x0 + dir_xt + sigma * noise
        return x
