import random
import numpy as np
import torch
import yaml

from src.data.circles import CircleClouds
from src.data.mnist import MNISTPointClouds


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_dataset(cfg, split="train", seed=None):
    ds = cfg["dataset"]
    seed = cfg["seed"] if seed is None else seed
    if ds["name"] == "mnist":
        return MNISTPointClouds(ds["root"], split=split, n_samples=ds["n_samples"], n_points=ds["n_points"], seed=seed)
    if ds["name"] == "circles":
        return CircleClouds(
            n_samples=ds["n_samples"],
            n_points=ds["n_points"],
            h_range=ds["h_range"],
            y_offset=ds["y_offset"],
            radius=ds["target_radius"],
            seed=seed,
        )
    raise ValueError(f"Unknown dataset: {ds['name']}")
