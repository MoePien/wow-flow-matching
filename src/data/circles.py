import numpy as np
import torch
from torch.utils.data import Dataset


class CircleClouds(Dataset):
    """Target metameasure of Sec. 4.2: n_points on a circle of radius `radius` centred at (h, y_offset).

    The centre h ~ Unif[-h_range, h_range] is fixed per index; the points on the circle are redrawn
    on every access.
    """

    def __init__(self, n_samples, n_points, h_range, y_offset, radius, seed=0):
        self.n_samples = n_samples
        self.n_points = n_points
        self.h_range = h_range
        self.y_offset = y_offset
        self.radius = radius
        self.seed = seed

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        rng = np.random.default_rng(self.seed + idx)
        rng.integers(self.n_points, self.n_points + 1)  # point-count draw, kept for identical random streams
        if rng.random() < 0.5:
            h = float(rng.uniform(-self.h_range, 0.0))
        else:
            h = float(rng.uniform(0.0, self.h_range))
        pts = sample_circle(h, self.y_offset, self.n_points, self.radius, np.random.default_rng())
        return pts, int(idx)


def sample_circle(h, y, n_points, radius, rng):
    """n_points at (h, y) + radius * eps / |eps| with eps ~ N(0, I_2)."""
    eps = rng.normal(size=(n_points, 2)).astype(np.float32)
    norms = np.linalg.norm(eps, axis=1, keepdims=True).clip(min=1e-8)
    pts = eps / norms * radius + np.array([h, y], dtype=np.float32)
    return torch.from_numpy(pts)


def sample_source(n_points, h_range, radius, device):
    """Source cloud of Sec. 4.2: circle of radius `radius` centred at (h, 0), h ~ Unif[-h_range, h_range]."""
    h = torch.empty(1).uniform_(-h_range, h_range).item()
    return sample_circle(h, 0.0, n_points, radius, np.random.default_rng()).to(device)
