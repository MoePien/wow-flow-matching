import math
import torch
from torch import nn


def time_embedding(t, dim):
    t = t.view(-1)
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000) * torch.arange(0, half, device=t.device).float() / max(half - 1, 1)
    )
    args = t[:, None] * freqs[None]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


class LocalInfoMLP(nn.Module):
    """Baseline velocity field for fixed-size clouds x in R^{N x d}.

    Each point is described by its sorted distances to all points together with the points in that
    order (local information), and by the mean and upper-triangular covariance of the cloud (global
    information). A sinusoidal time embedding is appended, the features pass through an MLP and one
    self-attention block, and a linear layer returns the velocity in R^d.
    """

    def __init__(self, in_dim, n_points, time_dim=32, hidden_dim=256, depth=4, heads=4):
        super().__init__()
        self.in_dim = in_dim
        self.n_points = n_points
        self.time_dim = time_dim

        input_dim = n_points * (in_dim + 1) + in_dim + (in_dim * (in_dim + 1)) // 2 + time_dim
        layers = []
        dim = input_dim
        for _ in range(max(depth, 1)):
            layers.append(nn.Linear(dim, hidden_dim))
            layers.append(nn.GELU())
            dim = hidden_dim
        layers.append(nn.Linear(dim, hidden_dim))
        self.mlp = nn.Sequential(*layers)
        self.token_proj = nn.Linear(hidden_dim + in_dim + 1, hidden_dim)
        self.set_blocks = nn.ModuleList([SetTransformerBlock(hidden_dim, heads)])
        self.out_proj = nn.Linear(hidden_dim, in_dim)

    def forward(self, x, t):
        """x: (N, d), t: (1,)."""
        if x.shape[0] != self.n_points:
            raise RuntimeError(f"LocalInfoMLP requires n_points={self.n_points}, got {x.shape[0]}")

        dists = torch.cdist(x, x, p=2)
        dists_sorted, idx = torch.sort(dists, dim=1)
        local = torch.cat([dists_sorted.unsqueeze(-1), x[idx]], dim=2)
        local = local.reshape(local.shape[0], -1)

        mean = x.mean(dim=0)
        centered = x - mean
        cov = centered.t().mm(centered) / max(x.shape[0] - 1, 1)
        tri = torch.triu_indices(self.in_dim, self.in_dim, device=x.device)
        g = torch.cat([mean, cov[tri[0], tri[1]]], dim=0)
        g = g.unsqueeze(0).expand(x.shape[0], -1)

        if t.dim() == 0:
            t = t[None]
        te = time_embedding(t, self.time_dim).expand(x.shape[0], -1)

        out = self.mlp(torch.cat([local, g, te], dim=1))
        t_scalar = t.view(-1, 1).expand(x.shape[0], 1)
        tokens = self.token_proj(torch.cat([out, x, t_scalar], dim=1)).unsqueeze(0)
        for blk in self.set_blocks:
            tokens = blk(tokens)
        return self.out_proj(tokens.squeeze(0))


class SetTransformerBlock(nn.Module):
    def __init__(self, dim, heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=0.0, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim),
        )

    def forward(self, x):
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


def build_model(cfg):
    m = cfg["model"]
    return LocalInfoMLP(
        in_dim=m["in_dim"],
        n_points=cfg["dataset"]["n_points"],
        time_dim=m["time_dim"],
        hidden_dim=m["hidden_dim"],
        depth=m["depth"],
    )


@torch.no_grad()
def euler_sample(model, x0, steps):
    x = x0
    for k in range(steps):
        t = torch.tensor([k / steps], device=x.device)
        x = x + (1.0 / steps) * model(x, t)
    return x


@torch.no_grad()
def euler_trajectory(model, x0, steps):
    x = x0
    traj = [x0]
    for k in range(steps):
        t = torch.tensor([k / steps], device=x.device)
        x = x + (1.0 / steps) * model(x, t)
        traj.append(x)
    return torch.stack(traj, dim=0)
