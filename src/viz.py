import os
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch


def points_to_hist(points, size):
    """Bilinear splatting of points in [0,1]^2 onto a size x size grid, normalized to sum one."""
    hist = torch.zeros((size, size), device=points.device, dtype=points.dtype)
    r = points[:, 0] * (size - 1)
    c = points[:, 1] * (size - 1)
    r0 = torch.floor(r).long()
    c0 = torch.floor(c).long()
    r1 = r0 + 1
    c1 = c0 + 1
    dr = (r - r0.float()).clamp(0, 1)
    dc = (c - c0.float()).clamp(0, 1)

    def add(rr, cc, w):
        mask = (rr >= 0) & (rr < size) & (cc >= 0) & (cc < size)
        if mask.any():
            hist.index_put_((rr[mask], cc[mask]), w[mask], accumulate=True)

    add(r0, c0, (1 - dr) * (1 - dc))
    add(r0, c1, (1 - dr) * dc)
    add(r1, c0, dr * (1 - dc))
    add(r1, c1, dr * dc)
    return hist / (hist.sum() + 1e-8)


def save_mnist_grid(path, clouds, title, normalize=True, blur_sigma=0.6):
    from scipy.ndimage import gaussian_filter

    os.makedirs(os.path.dirname(path), exist_ok=True)
    imgs = [points_to_hist(torch.as_tensor(c), 28).numpy() for c in clouds]
    n = len(imgs)
    cols = min(8, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = np.atleast_2d(axes)
    for i in range(rows):
        for j in range(cols):
            idx = i * cols + j
            ax = axes[i, j]
            ax.axis("off")
            if idx >= n:
                continue
            img = imgs[idx]
            if normalize and img.max() > 0:
                img = img / img.max()
            if blur_sigma and blur_sigma > 0:
                img = gaussian_filter(img, sigma=blur_sigma)
            ax.imshow(img, cmap="gray")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_coupling_grid(path, pairs, cols=4, size=2.2):
    """Source points, target points and the sampled inner plan (gray lines) for a few pairs."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    n = len(pairs)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(size * cols, size * rows))
    axes = np.atleast_2d(axes)
    for i in range(rows):
        for j in range(cols):
            idx = i * cols + j
            ax = axes[i, j]
            ax.set_aspect("equal")
            ax.axis("off")
            if idx >= n:
                continue
            X, Y, perm = pairs[idx]
            ax.scatter(X[:, 0], X[:, 1], s=4, alpha=0.6)
            ax.scatter(Y[:, 0], Y[:, 1], s=4, alpha=0.6)
            for k, t in enumerate(perm):
                ax.plot([X[k, 0], Y[t, 0]], [X[k, 1], Y[t, 1]], color="gray", alpha=0.3, linewidth=0.5)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_circle_trajectories(path, trajs, h_values, size=5, steps_to_show=200):
    """Fig. 1: particle paths for source circles centred at (h, 0); saves `path` and a `_no_legend` copy.

    trajs: list of arrays (T, N, 2). Circles mark t=0, squares t=1/2, triangles t=1.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
    fig, ax = plt.subplots(1, 1, figsize=(size, size))
    ax.set_aspect("equal")
    ax.axis("off")
    for traj, color, h in zip(trajs, colors, h_values):
        traj = np.asarray(traj)
        T, N, _ = traj.shape
        t_idx = np.linspace(0, T - 1, min(steps_to_show, T)).round().astype(int)
        for p in range(N):
            ax.plot(traj[t_idx, p, 0], traj[t_idx, p, 1], color=color, alpha=0.15, linewidth=0.8)
        ax.scatter(traj[0, :, 0], traj[0, :, 1], s=8, color=color, alpha=0.3, marker="o")
        ax.scatter(traj[T // 2, :, 0], traj[T // 2, :, 1], s=8, color=color, alpha=0.6, marker="s")
        ax.scatter(traj[-1, :, 0], traj[-1, :, 1], s=8, color=color, alpha=0.9, marker="^", label=f"$h={h}$")
    fig.tight_layout()
    stem, ext = os.path.splitext(path)
    fig.savefig(f"{stem}_no_legend{ext}", dpi=150)
    ax.legend(loc="upper center", fontsize=14, framealpha=0.8)
    fig.savefig(path, dpi=150)
    plt.close(fig)
