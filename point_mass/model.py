"""Spec 7 reference: noncausal action transformer. D = Euler solver plus fixed action adapter."""
import math
import torch
import torch.nn as nn


def sinusoid(x, dim):
    half = dim // 2
    freq = torch.exp(torch.arange(half, device=x.device, dtype=x.dtype) * (-math.log(10000.0) / max(half - 1, 1)))
    a = x[..., None] * freq
    return torch.cat([a.sin(), a.cos()], dim=-1)


class FMPolicy(nn.Module):
    def __init__(self, obs_dim=10, act_dim=2, H=16, width=256, layers=4, heads=4, ff=1024):
        super().__init__()
        self.H, self.act_dim, self.width = H, act_dim, width
        self.obs_mlp = nn.Sequential(nn.Linear(obs_dim, width), nn.SiLU(), nn.Linear(width, width))
        self.time_mlp = nn.Sequential(nn.Linear(width, width), nn.SiLU(), nn.Linear(width, width))
        self.in_proj = nn.Linear(act_dim, width)
        self.register_buffer("rel_time", sinusoid(torch.arange(H, dtype=torch.float32), width))
        layer = nn.TransformerEncoderLayer(width, heads, ff, dropout=0.0, activation="gelu",
                                           batch_first=True, norm_first=True)
        self.blocks = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.out = nn.Linear(width, act_dim)
        self.register_buffer("obs_mean", torch.zeros(obs_dim))
        self.register_buffer("obs_std", torch.ones(obs_dim))
        self.register_buffer("act_lo", -torch.ones(act_dim))
        self.register_buffer("act_hi", torch.ones(act_dim))
        self.version = "unset"

    def velocity(self, x, t, obs):
        o = (obs - self.obs_mean) / self.obs_std
        cond = self.obs_mlp(o) + self.time_mlp(sinusoid(t * 1000.0, self.width))
        tok = self.in_proj(x) + self.rel_time
        h = self.blocks(torch.cat([cond[:, None], tok], dim=1))
        return self.out(h[:, 1:])

    def solve(self, z, obs, steps=10):
        """Noise-to-action Euler map in normalized coordinates. Deterministic given (z, obs)."""
        x = z
        for i in range(steps):
            t = torch.full((z.shape[0],), i / steps, device=z.device, dtype=z.dtype)
            x = x + self.velocity(x, t, obs) / steps
        return x

    def adapter(self, x):
        """Fixed postprocessing B: normalized to native command units, clipped to limits."""
        return (x.clamp(-1, 1) + 1) / 2 * (self.act_hi - self.act_lo) + self.act_lo

    @torch.no_grad()
    def decode(self, z, obs, steps=10):
        return self.adapter(self.solve(z, obs, steps))
