"""Comparison of the 51 synergies with the DIEM (replica of Fig. 9 and Fig. 10).

DIEM (Dimension-Insensitive Euclidean Metric, Tessari, Yao, Hogan, 2024)
compares the Euclidean distance between two vectors with the distance
between random vectors of the same size and range:

    DIEM = (v_max - v_min) / var * (distance - center)

where center and var are the median and the variance of the distances
between random vectors (n components, uniform in [v_min, v_max]),
estimated by simulation as in the official code (github.com/ftessari23/DIEM).

DIEM = 0: as far apart as two random vectors. Negative: more similar than
random. Positive: more different than random.

Range choice: the paper does not report it. Here the forces are divided by
the largest muscle force among the 51 synergies, so every component is in
[0, 1] (v_min = 0, v_max = 1). The DIEM of the pairs does not change with
this scaling; the reference lines then match the official code. The result
depends on the choice of the maximum (see README).

Run after min_effort.py (it needs synergies_51.npy):
    python diem.py
"""

import os

import matplotlib
matplotlib.use("Agg")  # save files only, no windows
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import null_space

MUSCLES = ["sf", "se", "bf", "be", "ef", "ee", "wf", "we", "fwf", "fwe"]
OUT_DIR = "figures"


def diem_stats(n, v_min, v_max, n_samples=100_000, seed=0):
    """Center and variance of the distance between random vectors.

    Args:
        n: number of components of each vector.
        v_min, v_max: range of each component.
        n_samples: number of random pairs.
        seed: seed of the random generator.

    Returns:
        center: median distance between two random vectors.
        var: variance of that distance.
        distances: the n_samples random distances, array (n_samples,).
        orthogonal: median distance between a random vector and a unit
            vector perpendicular to it (reference used in the official code).
    """
    rng = np.random.default_rng(seed)
    a = rng.uniform(v_min, v_max, size=(n_samples, n))
    b = rng.uniform(v_min, v_max, size=(n_samples, n))
    distances = np.linalg.norm(a - b, axis=1)
    ortho = [np.linalg.norm(vec - null_space(vec[None, :])[:, 0]) for vec in a[:5000]]
    return np.median(distances), np.var(distances), distances, np.median(ortho)


def diem(distance, center, var, v_min, v_max):
    """DIEM of one or more Euclidean distances (same units as v_min, v_max)."""
    return (v_max - v_min) * (distance - center) / var


def pairwise_diem(vectors, center, var, v_min, v_max):
    """DIEM of every pair of rows of `vectors`.

    Args:
        vectors: array (k, n), one vector per row.

    Returns:
        values: DIEM of the k*(k-1)/2 pairs, array.
        pairs: the (i, j) indices of each pair, i < j, array (k*(k-1)/2, 2).
    """
    i, j = np.triu_indices(len(vectors), k=1)          # every pair once
    distances = np.linalg.norm(vectors[i] - vectors[j], axis=1)
    return diem(distances, center, var, v_min, v_max), np.column_stack([i, j])


def plot_histogram(values, ref, path):
    """Replica of Fig. 9: DIEM distribution with reference lines."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for k, alpha in [(3, 0.10), (2, 0.15), (1, 0.22)]:   # 1, 2, 3 standard deviations
        ax.axvspan(-k * ref["std"], k * ref["std"], color="tab:red", alpha=alpha, lw=0)
    ax.hist(values, bins=40, color="tab:blue", alpha=0.85)
    for x, name in [(ref["identical"], "identical"), (0, "random"),
                    (ref["orthogonal"], "orthogonal")]:
        ax.axvline(x, color="0.2", linestyle="--", linewidth=1)
        ax.text(x, 1.01, name, transform=ax.get_xaxis_transform(), ha="center", fontsize=9)
    ax.set_xlabel("DIEM")
    ax.set_ylabel("Number of pairs")
    ax.set_title(f"DIEM between the {ref['k']} synergies ({len(values)} pairs)\n"
                 f"min {values.min():.1f}, median {np.median(values):.1f}, "
                 f"max {values.max():.1f}  (red bands: 1, 2, 3 SD of random pairs)",
                 fontsize=10, pad=18)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_examples(synergies, values, pairs, path):
    """Replica of Fig. 10: most similar pair, a pair near the mode, most different pair."""
    counts, edges = np.histogram(values, bins=40)
    mode = (edges[np.argmax(counts)] + edges[np.argmax(counts) + 1]) / 2
    chosen = [("Most similar", np.argmin(values)),
              ("Near the mode", np.argmin(np.abs(values - mode))),
              ("Most different", np.argmax(values))]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
    y = np.arange(len(MUSCLES))
    for ax, (title, k) in zip(axes, chosen):
        i, j = pairs[k]
        ax.barh(y - 0.2, synergies[i], 0.4, label=f"synergy {i + 1}")
        ax.barh(y + 0.2, synergies[j], 0.4, label=f"synergy {j + 1}")
        ax.set_title(f"{title}: DIEM = {values[k]:.2f}", fontsize=10)
        ax.set_xlabel("Muscle force [N]")
        ax.legend(fontsize=8)
        ax.grid(True, axis="x", alpha=0.3)
    axes[0].set_yticks(y, MUSCLES)
    axes[0].invert_yaxis()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    forces = np.load("synergies_51.npy")               # (51, 10) in N, last row = minimum effort
    k, n = forces.shape
    f_max = forces.max()
    synergies = forces / f_max                          # every component in [0, 1]
    v_min, v_max = 0.0, 1.0

    center, var, random_distances, ortho = diem_stats(n, v_min, v_max)
    values, pairs = pairwise_diem(synergies, center, var, v_min, v_max)

    ref = {
        "k": k,
        "identical": diem(0.0, center, var, v_min, v_max),        # distance 0
        "orthogonal": diem(ortho, center, var, v_min, v_max),
        "std": np.std(diem(random_distances, center, var, v_min, v_max)),
    }

    os.makedirs(OUT_DIR, exist_ok=True)
    plot_histogram(values, ref, os.path.join(OUT_DIR, "fig9_diem.png"))
    plot_examples(forces, values, pairs, os.path.join(OUT_DIR, "fig10_examples.png"))

    print(f"Forces divided by {f_max:.0f} N (largest muscle force), n = {n}")
    print(f"Random pairs: center {center:.3f}, SD {np.sqrt(var):.3f} (scaled units)")
    print(f"References: identical {ref['identical']:.2f}, random 0, "
          f"orthogonal {ref['orthogonal']:.2f}, 1 SD = {ref['std']:.2f}")
    print(f"{len(values)} pairs: min {values.min():.2f}, median {np.median(values):.2f}, "
          f"max {values.max():.2f}  (paper: -20.52, mode -12.50, +6.09)")
    print(f"Pairs more similar than random: {np.mean(values < 0):.0%}")
    print("Saved fig9_diem.png and fig10_examples.png in figures/")
