"""Builds the NSOT matched-source cache for an MNIST config on CPU before training.

    python scripts/precompute_nsot.py --config configs/mnist/mnist_nsot.yaml --num_workers 8
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.common import build_dataset, load_config  # noqa: E402
from src.ot.nsot import NSOTCoupling  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--num_workers", type=int, default=0, help="0: all cores")
    args = parser.parse_args()

    cfg = load_config(args.config)
    src = cfg["source"]
    NSOTCoupling(
        dim=cfg["model"]["in_dim"],
        cache_dir=src["cache_dir"],
        superset_size=src["superset_size"],
        seed=cfg["seed"],
        num_workers=args.num_workers,
    ).fit(build_dataset(cfg, split="train"), cfg["dataset"]["name"])


if __name__ == "__main__":
    main()
