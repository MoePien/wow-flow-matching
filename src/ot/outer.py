"""Outer transport plans OP between a batch of source clouds and a batch of target clouds.

`outer_cost_matrix` returns the B x B cost between all source/target pairs; `outer_plan` solves the
discrete OT problem on it. Training then reorders the targets with `sample_from_plan`.
"""
import numpy as np
import ot
import torch


def sliced_wasserstein_distance(X, Y, num_projections):
    d = X.shape[1]
    dirs = torch.randn((num_projections, d), device=X.device)
    dirs = dirs / (dirs.norm(dim=1, keepdim=True) + 1e-8)
    proj_x, _ = torch.sort(X @ dirs.T, dim=0)
    proj_y, _ = torch.sort(Y @ dirs.T, dim=0)
    return ((proj_x - proj_y) ** 2).mean()


def _sqrtm_psd(A):
    A = 0.5 * (A + A.T)
    evals, evecs = torch.linalg.eigh(A)
    evals = evals.clamp_min(0.0).sqrt()
    return (evecs * evals.unsqueeze(0)) @ evecs.T


def _fit_gaussian(X, eps=1e-6):
    X = X.detach().double()
    mean = X.mean(dim=0)
    centered = X - mean
    cov = centered.T @ centered / max(X.shape[0], 1)
    cov = 0.5 * (cov + cov.T)
    cov = cov + eps * torch.eye(cov.shape[0], device=X.device, dtype=X.dtype)
    return mean, cov


def bures_wasserstein_distance(X, Y, eps=1e-6):
    """Squared W2 between the Gaussians fitted to X and Y (closed form)."""
    mean_x, cov_x = _fit_gaussian(X, eps=eps)
    mean_y, cov_y = _fit_gaussian(Y, eps=eps)
    sqrt_cov_x = _sqrtm_psd(cov_x)
    sqrt_middle = _sqrtm_psd(sqrt_cov_x @ cov_y @ sqrt_cov_x)
    mean_term = ((mean_x - mean_y) ** 2).sum()
    cov_term = torch.trace(cov_x + cov_y - 2.0 * sqrt_middle)
    dist = (mean_term + cov_term).clamp_min(0.0)
    if not torch.isfinite(dist):
        raise RuntimeError("Bures-Wasserstein outer cost produced a non-finite distance.")
    return dist.to(dtype=X.dtype)


def outer_cost_matrix(bases, targets, target_indices, outer, llw=None, sliced_projections=8):
    """outer in {w, sw, llw, bw, euclidean}.

    w: exact W2^2 between clouds. sw: sliced W2^2. llw: squared Euclidean distance after ordering each
    target by its barycenter alignment (sources are already in barycenter order). bw: Bures-Wasserstein
    distance of fitted Gaussians. euclidean: mean squared difference of the point arrays in stored order.
    """
    B = len(bases)
    device = bases[0].device
    if outer == "llw":
        base_aligned = torch.stack(bases, dim=0)
        target_aligned = torch.stack([llw.align_target(t, idx) for t, idx in zip(targets, target_indices)], dim=0)
        diff = base_aligned[:, None, :, :] - target_aligned[None, :, :, :]
        return (diff ** 2).mean(dim=(2, 3))
    C = torch.zeros((B, B), device=device)
    for i in range(B):
        for j in range(B):
            if outer == "w":
                xa = bases[i].detach().cpu().numpy()
                xb = targets[j].detach().cpu().numpy()
                wa = np.ones((xa.shape[0],)) / xa.shape[0]
                wb = np.ones((xb.shape[0],)) / xb.shape[0]
                cost = ot.dist(xa, xb, metric="euclidean") ** 2
                C[i, j] = torch.tensor(ot.emd2(wa, wb, cost), device=device)
            elif outer == "sw":
                C[i, j] = sliced_wasserstein_distance(bases[i], targets[j], sliced_projections)
            elif outer == "bw":
                C[i, j] = bures_wasserstein_distance(bases[i], targets[j])
            elif outer == "euclidean":
                C[i, j] = ((bases[i] - targets[j]) ** 2).mean()
            else:
                raise ValueError(f"Unknown outer plan: {outer}")
    return C


def outer_plan(C, method="emd", reg=0.05):
    Cn = C.detach().cpu().numpy()
    B = Cn.shape[0]
    a = np.ones((B,)) / B
    if method == "emd":
        P = ot.emd(a, a, Cn)
    elif method == "sinkhorn":
        Cn = Cn / (Cn.max() + 1e-8)
        P = ot.sinkhorn(a, a, Cn, reg)
    else:
        raise ValueError(f"Unknown OT method: {method}")
    return torch.from_numpy(P).to(C.device)
