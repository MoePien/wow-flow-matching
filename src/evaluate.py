"""Chamfer-NNA and OT-NNA (Sec. 4.1) of a trained MNIST run.

n_gen generated clouds are compared with n_gen clouds from the MNIST test split. Each repeat r uses
seed + r for the test-set point sampling and the source noise.

    python -m src.evaluate --run_dir runs/mnist_llw_llw --n_gen 512 --steps 5 25 125 --repeats 5
"""
import argparse
import os
import statistics

import numpy as np
import ot
import torch
from torch.utils.data import DataLoader

from src.common import build_dataset, load_config, set_seed
from src.model import build_model, euler_sample
from src.ot.nsot import NSOTCoupling
from src.sources import Source, load_barycenter, needs_barycenter


def chamfer_distance(a, b):
    d = torch.cdist(a, b, p=2) ** 2
    return d.min(dim=1).values.mean() + d.min(dim=0).values.mean()


def ot_distance(a, b):
    xa = a.detach().cpu().numpy()
    xb = b.detach().cpu().numpy()
    wa = np.ones((xa.shape[0],)) / xa.shape[0]
    wb = np.ones((xb.shape[0],)) / xb.shape[0]
    return ot.emd2(wa, wb, ot.dist(xa, xb, metric="euclidean") ** 2)


def nn_accuracy(reals, fakes, dist_fn):
    """1-NN two-sample accuracy on the union of real and generated clouds (0.5 is best, 1.0 worst)."""
    all_sets = reals + fakes
    labels = torch.tensor([0] * len(reals) + [1] * len(fakes))
    n = len(all_sets)
    dist = torch.zeros((n, n), dtype=torch.float32)
    for i in range(n):
        for j in range(i + 1, n):
            d = float(dist_fn(all_sets[i], all_sets[j]))
            dist[i, j] = d
            dist[j, i] = d
    dist += torch.eye(n) * 1e9
    pred = labels[dist.argmin(dim=1)]
    return (pred == labels).float().mean().item()


def evaluate_once(cfg, ckpt_path, n_gen, steps, seed, device):
    set_seed(seed)
    dataset = build_dataset(cfg, split="test", seed=seed)
    loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0, collate_fn=list, drop_last=True)
    model = build_model(cfg).to(device)
    barycenter = load_barycenter(cfg) if needs_barycenter(cfg) else None
    nsot = None
    if cfg["source"]["type"] == "nsot":
        nsot = NSOTCoupling(dim=cfg["model"]["in_dim"], cache_dir=cfg["source"]["cache_dir"])
    source = Source(cfg, device, barycenter=barycenter, nsot=nsot)
    model.load_state_dict(torch.load(ckpt_path, map_location=device)["model"])
    model.eval()

    reals, gens = [], []
    loader_iter = iter(loader)
    for _ in range(n_gen):
        try:
            item = next(loader_iter)[0]
        except StopIteration:
            loader_iter = iter(loader)
            item = next(loader_iter)[0]
        target, target_idx = item
        target = target.to(device)
        base, _ = source.sample(target, target_idx, generation=True)
        reals.append(target)
        gens.append(euler_sample(model, base, steps))
    return nn_accuracy(reals, gens, chamfer_distance), nn_accuracy(reals, gens, ot_distance)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--ckpt", default=None, help="default: <run_dir>/checkpoints/model_final.pt")
    parser.add_argument("--n_gen", type=int, default=512)
    parser.add_argument("--steps", type=int, nargs="+", default=[5, 25, 125])
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()

    cfg = load_config(os.path.join(args.run_dir, "config.yaml"))
    ckpt = args.ckpt or os.path.join(args.run_dir, "checkpoints", "model_final.pt")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    lines = [f"run={args.run_dir} n_gen={args.n_gen} repeats={args.repeats}"]
    for steps in args.steps:
        chamfer, otnna = [], []
        for r in range(args.repeats):
            c, o = evaluate_once(cfg, ckpt, args.n_gen, steps, cfg["seed"] + r, device)
            chamfer.append(c)
            otnna.append(o)
            print(f"steps={steps} repeat={r} Chamfer_NNA={c} OT_NNA={o}", flush=True)
        std = statistics.pstdev if len(chamfer) > 1 else (lambda v: 0.0)
        lines.append(
            f"steps={steps}  Chamfer_NNA: mean={statistics.mean(chamfer):.6f} std={std(chamfer):.6f}"
            f"  OT_NNA: mean={statistics.mean(otnna):.6f} std={std(otnna):.6f}"
        )
        print(lines[-1], flush=True)
    with open(os.path.join(args.run_dir, "eval_nna.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
