"""Generalized Wasserstein flow matching on point clouds.

For a batch of B source clouds x^b and target clouds y^b the outer plan OP pairs sources with targets,
the inner plan IP matches the N points within each pair, and the velocity field V_theta is regressed
onto x' - x along the straight interpolation (1 - t) x + t x' (Eq. 29).

    python -m src.train --config configs/mnist/mnist_llw_llw.yaml
"""
import argparse
import copy
import logging
import os
import time

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.common import build_dataset, load_config, set_seed
from src.data.circles import sample_circle
from src.model import build_model, euler_sample, euler_trajectory
from src.ot.inner import IndependentPlan, LazyLinearPlan, SlicedPlan, WassersteinPlan, sample_from_plan
from src.ot.nsot import NSOTCoupling
from src.ot.outer import outer_cost_matrix, outer_plan
from src.sources import Source, load_barycenter, needs_barycenter
from src.viz import save_circle_trajectories, save_coupling_grid, save_mnist_grid

CIRCLE_H_VALUES = (-20, -10, 10, 20)


def llw_cache_path(cfg, length):
    ds = cfg["dataset"]
    return os.path.join(cfg["ot"]["cache_dir"], f"llw_alignments_{ds['name']}_N{ds['n_points']}_L{length}_s{cfg['seed']}.npy")


def build_inner(cfg, dataset, barycenter):
    ot_cfg = cfg["ot"]
    inner = ot_cfg["inner"]
    if inner == "ind":
        return IndependentPlan()
    if inner == "w":
        return WassersteinPlan(method=ot_cfg.get("method", "emd"), reg=ot_cfg.get("reg", 0.05))
    if inner == "sw":
        return SlicedPlan(ot_cfg["sliced_projections"])
    if inner == "llw":
        plan = LazyLinearPlan(barycenter, cache_path=llw_cache_path(cfg, len(dataset)))
        plan.fit(dataset)
        return plan
    if inner == "nsot":
        src = cfg["source"]
        plan = NSOTCoupling(
            dim=cfg["model"]["in_dim"],
            cache_dir=src["cache_dir"],
            superset_size=src["superset_size"],
            seed=cfg["seed"],
            num_workers=src.get("num_workers", 0),
        )
        plan.fit(dataset, cfg["dataset"]["name"])
        return plan
    raise ValueError(f"Unknown inner plan: {inner}")


def wow_loss(model, inner, outer, llw, cfg, bases, aligned, targets, target_indices, device):
    if outer != "ind":
        C = outer_cost_matrix(bases, targets, target_indices, outer, llw, cfg["ot"]["sliced_projections"])
        perm = sample_from_plan(outer_plan(C, method=cfg["ot"]["method"], reg=cfg["ot"]["reg"]))
        targets = [targets[i] for i in perm.tolist()]
        target_indices = [target_indices[i] for i in perm.tolist()]

    loss_sum = 0.0
    for base, target, target_idx, x_aligned in zip(bases, targets, target_indices, aligned):
        t = torch.rand(1, device=device)
        P = inner.compute_plan(base, target, y_idx=target_idx, x_aligned=x_aligned)
        T_hat = target[sample_from_plan(P)]
        X_t = (1.0 - t) * base + t * T_hat
        loss_sum = loss_sum + ((model(X_t, t) - (T_hat - base)) ** 2).mean()
    return loss_sum / max(len(targets), 1)


def visualize_mnist(epoch_dir, cfg, model, loader, source, inner, device):
    tr = cfg["train"]
    reals, gens, noises, couplings = [], [], [], []
    viz_iter = iter(loader)
    for _ in range(tr["vis_per_epoch"]):
        try:
            batch = next(viz_iter)
        except StopIteration:
            viz_iter = iter(loader)
            batch = next(viz_iter)
        target, target_idx = batch[0]
        target = target.to(device)
        base, x_aligned = source.sample(target, target_idx, generation=True)
        gen = euler_sample(model, base, tr["viz_sample_steps"])
        reals.append(target.cpu().numpy())
        gens.append(gen.cpu().numpy())
        noises.append(base.cpu().numpy())
        if len(couplings) < tr["viz_couplings"]:
            if isinstance(inner, LazyLinearPlan) and x_aligned:
                perm = inner.perm_from_index(target_idx, device)
            else:
                P = inner.compute_plan(base, target, y_idx=target_idx, x_aligned=x_aligned)
                sample_from_plan(P)  # extra draw keeps the random stream of the original runs
                perm = sample_from_plan(P)
            couplings.append((base.cpu().numpy(), target.cpu().numpy(), perm.cpu().numpy()))
    blur = tr.get("viz_blur_sigma", 0.6)
    save_mnist_grid(os.path.join(epoch_dir, "grid_real.png"), reals, "Real", normalize=False, blur_sigma=blur)
    save_mnist_grid(os.path.join(epoch_dir, "grid_gen.png"), gens, "Generated", blur_sigma=blur)
    save_mnist_grid(os.path.join(epoch_dir, "grid_noise.png"), noises, "Input Noise", blur_sigma=blur)
    save_coupling_grid(os.path.join(epoch_dir, "inner_coupling.png"), couplings)


def visualize_circles(epoch_dir, cfg, model, device):
    ds = cfg["dataset"]
    steps = cfg["train"]["viz_sample_steps"]
    trajs = []
    for h in CIRCLE_H_VALUES:
        src = sample_circle(h, 0.0, ds["n_points"], ds["source_radius"], np.random.default_rng()).to(device)
        trajs.append(euler_trajectory(model, src, steps).cpu().numpy())
    save_circle_trajectories(os.path.join(epoch_dir, "trajectories.png"), trajs, CIRCLE_H_VALUES, steps_to_show=steps)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run_dir", default=None, help="default: runs/<config name>")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tr = cfg["train"]

    run_dir = args.run_dir or os.path.join("runs", os.path.splitext(os.path.basename(args.config))[0])
    os.makedirs(os.path.join(run_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(run_dir, "viz"), exist_ok=True)
    with open(os.path.join(run_dir, "config.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f)
    logger = logging.getLogger("train")
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(os.path.join(run_dir, "train.log"))
    fh.setFormatter(logging.Formatter("%(asctime)s | %(message)s"))
    logger.addHandler(fh)

    dataset = build_dataset(cfg, split="train")
    loader = DataLoader(dataset, batch_size=tr["batch_size"], shuffle=True, num_workers=0, collate_fn=list, drop_last=True)
    model = build_model(cfg).to(device)

    barycenter = load_barycenter(cfg, dataset) if needs_barycenter(cfg) else None
    inner = build_inner(cfg, dataset, barycenter)
    outer = cfg["ot"]["outer"]
    llw = inner if isinstance(inner, LazyLinearPlan) else None
    if outer == "llw" and llw is None:
        llw = LazyLinearPlan(barycenter, cache_path=llw_cache_path(cfg, len(dataset)))
        llw.fit(dataset)
    source = Source(cfg, device, barycenter=barycenter, nsot=inner if isinstance(inner, NSOTCoupling) else None)
    if barycenter is not None and cfg["dataset"]["name"] == "mnist":
        save_mnist_grid(os.path.join(run_dir, "viz", "barycenter.png"), [barycenter.numpy()], "Barycenter")

    optimizer = torch.optim.Adam(model.parameters(), lr=tr["lr"])
    ema_model = None
    if tr.get("use_ema", False):
        ema_model = copy.deepcopy(model)
        ema_model.eval()
        for p in ema_model.parameters():
            p.requires_grad_(False)

    step = 0
    t0 = time.time()
    steps_per_epoch = len(dataset) // tr["batch_size"]
    for epoch in tqdm(range(tr["epochs"]), desc="epochs", unit="epoch"):
        loader_iter = iter(loader)
        pbar = tqdm(total=steps_per_epoch, desc=f"epoch {epoch}", unit="batch", leave=False)
        for batch_idx in range(steps_per_epoch):
            batch = next(loader_iter)
            targets = [x.to(device) for x, _ in batch]
            target_indices = [idx for _, idx in batch]
            bases, aligned = [], []
            for target, target_idx in zip(targets, target_indices):
                base, x_aligned = source.sample(target, target_idx)
                bases.append(base)
                aligned.append(x_aligned)
            loss = wow_loss(model, inner, outer, llw, cfg, bases, aligned, targets, target_indices, device)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            if ema_model is not None:
                decay = tr.get("ema_decay", 0.999)
                with torch.no_grad():
                    for p, ema_p in zip(model.parameters(), ema_model.parameters()):
                        ema_p.mul_(decay).add_(p, alpha=1.0 - decay)

            if step % tr["log_every"] == 0:
                logger.info(
                    f"epoch {epoch} | batch {batch_idx} | global_step {step} | loss {loss.item():.6f} | {time.time() - t0:.1f}s"
                )
                t0 = time.time()
                pbar.set_postfix(loss=f"{loss.item():.6f}")
            step += 1
            pbar.update(1)
        pbar.close()

        if (epoch + 1) % tr["vis_every_epochs"] == 0:
            epoch_dir = os.path.join(run_dir, "viz", f"epoch_{epoch:04d}")
            if cfg["dataset"]["name"] == "circles":
                visualize_circles(epoch_dir, cfg, model, device)
            else:
                visualize_mnist(epoch_dir, cfg, model, loader, source, inner, device)

    ckpt_dir = os.path.join(run_dir, "checkpoints")
    if ema_model is not None:
        torch.save({"model": model.state_dict(), "cfg": cfg}, os.path.join(ckpt_dir, "model_final_raw.pt"))
    weights = ema_model.state_dict() if ema_model is not None else model.state_dict()
    torch.save({"model": weights, "cfg": cfg}, os.path.join(ckpt_dir, "model_final.pt"))
    logger.info(f"Saved checkpoint to {os.path.join(ckpt_dir, 'model_final.pt')}")
    print(f"RUN_DIR={run_dir}", flush=True)


if __name__ == "__main__":
    main()
