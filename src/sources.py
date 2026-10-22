"""Source metameasures: barycentric noise, pure noise, circles (Sec. 4.2) and the NSOT source."""
import math
import torch

from src.data import circles


def apply_hybrid_beta(base, center, scale, beta):
    """NSOT hybrid source x0' = center + sqrt(1 - beta) (x0 - center) + sqrt(beta) scale eps."""
    if not beta or beta <= 0:
        return base
    fresh = torch.randn_like(base) * scale
    return center + math.sqrt(1.0 - beta) * (base - center) + math.sqrt(beta) * fresh


class Source:
    """Draws a source cloud for a given target; `aligned` tells whether it is in barycenter order."""

    def __init__(self, cfg, device, barycenter=None, nsot=None):
        self.cfg = cfg["source"]
        self.type = self.cfg["type"]
        self.dataset_cfg = cfg["dataset"]
        self.in_dim = cfg["model"]["in_dim"]
        self.lazy_linear = cfg["ot"]["inner"] == "llw"
        self.device = device
        self.barycenter = barycenter
        self.nsot = nsot

    def sample(self, target, target_idx, generation=False):
        device = self.device
        n = target.shape[0]
        if self.type == "barycentric":
            noise = torch.randn_like(self.barycenter) * self.cfg["noise_std"]
            noise = noise * (0.5 + torch.rand_like(noise))
            return self.barycenter.to(device) + noise.to(device), self.lazy_linear
        if self.type == "pure_noise":
            low, high = self.cfg["sigma_low"], self.cfg["sigma_high"]
            sigma = low + (high - low) * torch.rand(1).item()
            return torch.randn(n, self.in_dim, device=device) * sigma, False
        if self.type == "circles":
            ds = self.dataset_cfg
            return circles.sample_source(n, ds["h_range"], ds["source_radius"], device), False
        if self.type == "nsot":
            if generation:
                return self.nsot.sample_source(n, device), True
            base = self.nsot.matched_source(int(target_idx), device)
            beta = self.cfg.get("hybrid_beta", 0.0)
            if beta:
                mean = torch.as_tensor(self.nsot.source_mean, dtype=torch.float32, device=device)
                std = torch.as_tensor(self.nsot.source_std, dtype=torch.float32, device=device)
                base = apply_hybrid_beta(base, mean, std, beta)
            return base, True
        raise ValueError(f"Unknown source type: {self.type}")


def load_barycenter(cfg, train_dataset=None):
    """Barycenter rho of the training clouds, loaded from (or computed and cached at) cfg['barycenter_path']."""
    from src.ot.inner import load_or_compute_barycenter

    return load_or_compute_barycenter(
        cfg["barycenter_path"], train_dataset, cfg["dataset"]["n_points"], n_clouds=8, seed=cfg["seed"]
    )


def needs_barycenter(cfg):
    return cfg["source"]["type"] == "barycentric" or cfg["ot"]["inner"] == "llw"

