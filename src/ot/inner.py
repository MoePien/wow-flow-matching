"""Inner transport plans IP between a source cloud X and a target cloud Y (both N x d).

Every plan returns an N x N matrix P; training draws one target index per source point from the rows
of P (`sample_from_plan`).
"""
import os
import numpy as np
import ot
import torch
from scipy.optimize import linear_sum_assignment


def sample_from_plan(P):
    row_sums = P.sum(dim=1, keepdim=True).clamp_min(1e-12)
    probs = P / row_sums
    idx = []
    for i in range(probs.shape[0]):
        idx.append(torch.multinomial(probs[i], 1).squeeze(0))
    return torch.stack(idx, dim=0)


class IndependentPlan:
    """IP_ind: a uniformly random permutation per call."""

    def compute_plan(self, X, Y, **_):
        n = X.shape[0]
        perm = torch.randperm(n, device=X.device)
        P = torch.zeros((n, n), device=X.device)
        P[torch.arange(n, device=X.device), perm] = 1.0 / n
        return P


class WassersteinPlan:
    """IP_W: exact OT (ot.emd) or entropic OT (ot.sinkhorn on the max-normalized cost)."""

    def __init__(self, method="emd", reg=0.05):
        self.method = method
        self.reg = reg

    def compute_plan(self, X, Y, **_):
        Xn = X.detach().cpu().numpy()
        Yn = Y.detach().cpu().numpy()
        n, m = Xn.shape[0], Yn.shape[0]
        a = np.ones((n,)) / n
        b = np.ones((m,)) / m
        C = ot.dist(Xn, Yn, metric="euclidean") ** 2
        if self.method == "emd":
            P = ot.emd(a, b, C)
        elif self.method == "sinkhorn":
            C = C / (C.max() + 1e-8)
            P = ot.sinkhorn(a, b, C, self.reg)
        else:
            raise ValueError(f"Unknown OT method: {self.method}")
        return torch.from_numpy(P).to(X.device)


class SlicedPlan:
    """IP_SW: average of the K one-dimensional (sorting) plans along random directions theta ~ Unif(S^{d-1})."""

    def __init__(self, num_projections):
        self.num_projections = num_projections

    def compute_plan(self, X, Y, **_):
        n, d = X.shape
        K = self.num_projections
        directions = torch.randn(K, d, device=X.device)
        directions = directions / (directions.norm(dim=1, keepdim=True) + 1e-8)
        x_idx = torch.argsort(X @ directions.T, dim=0)
        y_idx = torch.argsort(Y @ directions.T, dim=0)
        perm = torch.empty_like(x_idx)
        perm.scatter_(0, x_idx, y_idx)
        rows = torch.arange(n, device=X.device)[:, None].expand(n, K).reshape(-1)
        cols = perm.reshape(-1)
        vals = torch.full((n * K,), 1.0 / n, device=X.device)
        P = torch.zeros((n, n), device=X.device)
        P.index_put_((rows, cols), vals, accumulate=True)
        return P / float(K)


class LazyLinearPlan:
    """IP_LLW: plans through a fixed reference (barycenter) rho of the training clouds.

    Every training cloud nu_i is matched to rho once (`alignments[i]` lists, for each barycenter point,
    the index of its partner in nu_i). A barycentric-noise source sample keeps the point order of rho
    (`x_aligned=True`), so its plan to nu_i is read off the stored alignment without solving OT.
    """

    def __init__(self, barycenter, cache_path=None):
        self.barycenter = barycenter
        self.n = barycenter.shape[0]
        self.cache_path = cache_path
        self.alignments = None
        self._alignments_torch = None

    def fit(self, dataset):
        if self.cache_path and os.path.exists(self.cache_path):
            align = np.load(self.cache_path)
            if align.shape == (len(dataset), self.n):
                self.alignments = align
                print(f"LLW: loaded alignments from {self.cache_path}")
                return
        from tqdm import tqdm

        align = np.zeros((len(dataset), self.n), dtype=np.int64)
        for i in tqdm(range(len(dataset)), desc="LLW alignments", unit="cloud"):
            x = dataset[i][0]
            align[i] = invert_perm(perm_to_reference(x, self.barycenter), self.n)
        self.alignments = align
        if self.cache_path:
            os.makedirs(os.path.dirname(self.cache_path) or ".", exist_ok=True)
            np.save(self.cache_path, align)
            print(f"LLW: saved alignments to {self.cache_path}")

    def alignment(self, idx, device):
        if self._alignments_torch is None:
            self._alignments_torch = torch.from_numpy(self.alignments)
        if device.type == "cuda" and self._alignments_torch.device != device:
            self._alignments_torch = self._alignments_torch.to(device)
        row = self._alignments_torch[idx]
        return row if row.device == device else row.to(device)

    def align_target(self, Y, idx):
        return Y[self.alignment(int(idx), Y.device)]

    def perm_from_index(self, y_idx, device):
        x_by_b = torch.arange(self.n, device=device)
        y_by_b = self.alignment(int(y_idx), device)
        perm = torch.empty_like(x_by_b, device=device)
        perm[x_by_b] = y_by_b
        return perm

    def compute_plan(self, X, Y, y_idx=None, x_aligned=False, **_):
        if x_aligned:
            x_by_b = torch.arange(self.n, device=X.device)
        else:
            x_by_b = invert_perm(perm_to_reference(X, self.barycenter), self.n).to(X.device)
        if y_idx is not None:
            y_by_b = self.alignment(int(y_idx), Y.device)
        else:
            y_by_b = invert_perm(perm_to_reference(Y, self.barycenter), self.n).to(Y.device)
        perm = torch.empty_like(x_by_b, device=X.device)
        perm[x_by_b] = y_by_b
        P = torch.zeros((self.n, self.n), device=X.device)
        P[torch.arange(self.n, device=X.device), perm] = 1.0 / self.n
        return P


def perm_to_reference(X, B):
    """Permutation p with X[i] -> B[p[i]] from the exact OT plan between two uniform N-point clouds."""
    Xn = X.detach().cpu().numpy()
    Bn = B.detach().cpu().numpy()
    n = Xn.shape[0]
    a = np.ones((n,)) / n
    C = ot.dist(Xn, Bn, metric="euclidean") ** 2
    P = torch.from_numpy(ot.emd(a, a, C))
    perm = torch.argmax(P, dim=1)
    if perm.unique().numel() != perm.numel():
        row, col = linear_sum_assignment((-P).numpy())
        perm = torch.zeros_like(torch.from_numpy(row))
        perm[row] = torch.from_numpy(col)
    return perm


def invert_perm(perm, n):
    inv = torch.empty((n,), dtype=torch.long, device=perm.device)
    inv[perm] = torch.arange(n, device=perm.device)
    return inv


def load_or_compute_barycenter(path, dataset, n_points, n_clouds=8, seed=0):
    """Free-support W2 barycenter of `n_clouds` random training clouds, cached at `path`."""
    if os.path.exists(path):
        return torch.from_numpy(np.load(path))
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(dataset), size=n_clouds, replace=False)
    clouds = [dataset[i][0].numpy() for i in idx]
    weights = [np.ones((n_points,)) / n_points for _ in clouds]
    B = ot.lp.free_support_barycenter(clouds, weights, np.mean(np.stack(clouds, axis=0), axis=0), numItermax=100)
    B = B + np.random.default_rng(0).normal(scale=1e-4, size=B.shape)
    B = B.astype(np.float32)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    np.save(path, B)
    return torch.from_numpy(B)
