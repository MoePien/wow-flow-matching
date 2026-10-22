"""Not-so-optimal transport (NSOT) baseline of Hui et al., "Not-So-Optimal Transport Flows for 3D
Point Cloud Generation" (ICLR 2025), with a standard Gaussian source.

Offline, for every training cloud nu_i with its N fixed points we build a data superset of size M
(the N points plus M - N extra points drawn from the same digit histogram) and a superset of M i.i.d.
N(0, I_d) source points, solve exact OT between the two supersets, and store for each of the N target
points the source point it is matched to. During training the source of cloud i is this matched noise,
so the inner plan is the identity. At generation time the source is plain N(0, I_d) noise.
"""
import os
import numpy as np
import ot
import torch

from src.data.mnist import normalize_hist, sample_points_from_hist

_W = {}


def _match(idx):
    dataset, n, m, dim, seed = _W["dataset"], _W["n"], _W["m"], _W["dim"], _W["seed"]
    y = dataset[idx][0].detach().cpu().numpy().astype(np.float64)
    rng = np.random.default_rng(np.int64(seed) + np.int64(idx))
    if m > n:
        hist = normalize_hist(dataset.get_image(idx))
        extra = sample_points_from_hist(hist, m - n, rng).detach().cpu().numpy().astype(np.float64)
        x1 = np.concatenate([y, extra], axis=0)
    else:
        x1 = y
    rng_noise = np.random.default_rng(np.int64(seed) + np.int64(idx) + np.int64(1_000_003))
    x0 = _W["mean"][None, :] + _W["std"][None, :] * rng_noise.standard_normal((x1.shape[0], dim))

    M = x1.shape[0]
    a = np.full(M, 1.0 / M)
    plan = ot.emd(a, a, ot.dist(x0, x1, metric="sqeuclidean"), numItermax=1_000_000, numThreads=1)
    matched_idx = np.argmax(plan[:, :n], axis=0)
    return idx, x0[matched_idx].astype(np.float32)


class NSOTCoupling:
    def __init__(self, dim, cache_dir, superset_size=1024, seed=0, num_workers=0):
        self.dim = dim
        self.cache_dir = cache_dir
        self.superset_size = int(superset_size)
        self.seed = int(seed)
        self.num_workers = num_workers
        self.source_mean = np.zeros(dim, dtype=np.float64)
        self.source_std = np.ones(dim, dtype=np.float64)
        self.matched_noise = None
        self._matched_torch = None

    def cache_path(self, dataset_name, n, length):
        return os.path.join(
            self.cache_dir,
            f"nsot_{dataset_name}_M{self.superset_size}_N{n}_d{self.dim}_L{length}_s{self.seed}.npy",
        )

    def fit(self, dataset, dataset_name):
        n = int(dataset[0][0].shape[0])
        length = len(dataset)
        m = max(self.superset_size, n)
        path = self.cache_path(dataset_name, n, length)
        if os.path.exists(path):
            matched = np.load(path)
            if matched.shape == (length, n, self.dim):
                self.matched_noise = matched
                print(f"NSOT: loaded matched-source cache from {path}")
                return

        import multiprocessing as mp
        from tqdm import tqdm

        print(f"NSOT: superset OT (M={m}, N={n}) for {length} clouds")
        _W.clear()
        _W.update(dataset=dataset, n=n, m=m, dim=self.dim, seed=self.seed, mean=self.source_mean, std=self.source_std)
        matched = np.zeros((length, n, self.dim), dtype=np.float32)
        workers = self.num_workers or os.cpu_count() or 1
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        if workers <= 1:
            for i in tqdm(range(length), desc="NSOT precompute", unit="cloud"):
                matched[i] = _match(i)[1]
        else:
            with mp.get_context("fork").Pool(processes=workers) as pool:
                for i, z in tqdm(pool.imap_unordered(_match, range(length), chunksize=64),
                                 total=length, desc="NSOT precompute", unit="cloud"):
                    matched[i] = z
        _W.clear()
        os.makedirs(self.cache_dir, exist_ok=True)
        np.save(path, matched)
        self.matched_noise = matched
        print(f"NSOT: saved matched-source cache to {path}")

    def matched_source(self, idx, device):
        if self._matched_torch is None:
            self._matched_torch = torch.from_numpy(self.matched_noise)
        return self._matched_torch[int(idx)].to(device)

    def sample_source(self, n, device):
        mean = torch.as_tensor(self.source_mean, dtype=torch.float32, device=device)
        std = torch.as_tensor(self.source_std, dtype=torch.float32, device=device)
        return mean + std * torch.randn(int(n), mean.shape[0], device=device)

    def compute_plan(self, X, Y, **_):
        n = X.shape[0]
        P = torch.zeros((n, n), device=X.device)
        P[torch.arange(n, device=X.device), torch.arange(n, device=X.device)] = 1.0 / n
        return P
