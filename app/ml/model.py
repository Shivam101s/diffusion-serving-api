"""Same UNet architecture as github.com/Shivam101s/mini-diffusion, vendored
here so this service has no runtime dependency on that repo's training code
— only the inference-time model definition it needs to load a checkpoint.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000) * torch.arange(half, device=t.device, dtype=torch.float32) / half
    )
    args = t.float()[:, None] * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = F.pad(embedding, (0, 1))
    return embedding


class ResidualBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_emb_dim: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.time_proj = nn.Linear(time_emb_dim, out_ch)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time_proj(F.silu(t_emb))[:, :, None, None]
        h = self.conv2(self.dropout(F.silu(self.norm2(h))))
        return h + self.skip(x)


class SelfAttention(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)
        self.scale = channels ** -0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        qkv = self.qkv(self.norm(x)).reshape(b, 3, c, h * w)
        q, k, v = qkv.unbind(1)
        attn = torch.softmax((q.transpose(1, 2) @ k) * self.scale, dim=-1)
        out = (v @ attn.transpose(1, 2)).reshape(b, c, h, w)
        return x + self.proj(out)


class Downsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.op = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x)


class Upsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.op = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(F.interpolate(x, scale_factor=2, mode="nearest"))


class UNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 64,
        channel_mults=(1, 2, 2, 2),
        num_res_blocks: int = 2,
        attn_resolutions=(16, 8),
        image_size: int = 32,
    ):
        super().__init__()
        time_emb_dim = base_channels * 4
        self.time_embed_dim = base_channels
        self.time_mlp = nn.Sequential(
            nn.Linear(base_channels, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        self.in_conv = nn.Conv2d(in_channels, base_channels, 3, padding=1)

        self.down_blocks = nn.ModuleList()
        channels = [base_channels]
        now_ch = base_channels
        res = image_size
        for i, mult in enumerate(channel_mults):
            out_ch = base_channels * mult
            for _ in range(num_res_blocks):
                self.down_blocks.append(ResidualBlock(now_ch, out_ch, time_emb_dim))
                now_ch = out_ch
                if res in attn_resolutions:
                    self.down_blocks.append(SelfAttention(now_ch))
                channels.append(now_ch)
            if i != len(channel_mults) - 1:
                self.down_blocks.append(Downsample(now_ch))
                channels.append(now_ch)
                res //= 2

        self.mid_block1 = ResidualBlock(now_ch, now_ch, time_emb_dim)
        self.mid_attn = SelfAttention(now_ch)
        self.mid_block2 = ResidualBlock(now_ch, now_ch, time_emb_dim)

        self.up_blocks = nn.ModuleList()
        for i, mult in reversed(list(enumerate(channel_mults))):
            out_ch = base_channels * mult
            for _ in range(num_res_blocks + 1):
                self.up_blocks.append(ResidualBlock(now_ch + channels.pop(), out_ch, time_emb_dim))
                now_ch = out_ch
                if res in attn_resolutions:
                    self.up_blocks.append(SelfAttention(now_ch))
            if i != 0:
                self.up_blocks.append(Upsample(now_ch))
                res *= 2

        self.out_norm = nn.GroupNorm(8, now_ch)
        self.out_conv = nn.Conv2d(now_ch, in_channels, 3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_mlp(timestep_embedding(t, self.time_embed_dim))

        h = self.in_conv(x)
        skips = [h]
        for layer in self.down_blocks:
            if isinstance(layer, ResidualBlock):
                h = layer(h, t_emb)
                skips.append(h)
            elif isinstance(layer, SelfAttention):
                h = layer(h)
                skips[-1] = h
            else:  # Downsample
                h = layer(h)
                skips.append(h)

        h = self.mid_block1(h, t_emb)
        h = self.mid_attn(h)
        h = self.mid_block2(h, t_emb)

        for layer in self.up_blocks:
            if isinstance(layer, ResidualBlock):
                h = layer(torch.cat([h, skips.pop()], dim=1), t_emb)
            elif isinstance(layer, SelfAttention):
                h = layer(h)
            else:  # Upsample
                h = layer(h)

        return self.out_conv(F.silu(self.out_norm(h)))
