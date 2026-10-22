import numpy as np
import torch
from torch.utils.data import Dataset


class MNISTPointClouds(Dataset):
    """MNIST digits as point clouds of n_points samples from the normalized pixel histogram on [0,1]^2.

    Item idx is drawn with np.random.default_rng(seed + idx), so every cloud is fixed across epochs.
    """

    def __init__(self, root, split="train", n_samples=0, n_points=64, seed=0):
        from torchvision.datasets import MNIST
        from torchvision.transforms import ToTensor

        self.n_points = n_points
        self.seed = seed
        self.ds = MNIST(root, train=split == "train", download=True, transform=ToTensor())
        self.indices = np.arange(len(self.ds))
        if n_samples and len(self.indices) > n_samples:
            rng = np.random.default_rng(seed)
            rng.shuffle(self.indices)
            self.indices = self.indices[:n_samples]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        rng = np.random.default_rng(self.seed + idx)
        hist = normalize_hist(self.get_image(idx))
        return sample_points_from_hist(hist, self.n_points, rng), int(idx)

    def get_image(self, idx):
        img, _ = self.ds[self.indices[idx]]
        return img.squeeze(0)


def normalize_hist(img):
    x = img.detach().cpu().numpy().astype(np.float32)
    x = np.clip(x, 0, None)
    s = x.sum()
    if s <= 0:
        x = np.ones_like(x) / x.size
    else:
        x = x / s
    return torch.from_numpy(x)


def sample_points_from_hist(hist, n_points, rng):
    probs = hist.reshape(-1).detach().cpu().numpy()
    idx = rng.choice(len(probs), size=n_points, replace=True, p=probs)
    h, w = hist.shape
    pts = np.stack([idx // w, idx % w], axis=1).astype(np.float32)
    pts[:, 0] /= float(h - 1)
    pts[:, 1] /= float(w - 1)
    return torch.from_numpy(pts)
