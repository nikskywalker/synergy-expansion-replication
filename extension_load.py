"""Extension: synergies under load (not part of the paper).

The numerical example has no upper limit on muscle forces. Here every muscle
gets the same maximum force F_MAX (a hypothetical value), and two things are
computed as the required hand force grows:

    1. the range of forces each muscle can take while the task is still
       performed exactly (linear programming: min and max of each force);
    2. the DIEM between 50 random-search solutions at each force level,
       with the reference range [0, F_MAX].

Saves figures/extension_feasible.png and figures/extension_diem_load.png.
    python extension_load.py
"""

import os

import matplotlib
matplotlib.use("Agg")  # save files only, no windows
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linprog

from diem import diem, diem_stats, pairwise_diem
from min_effort import TAU_REQUIRED
from muscles import Jm

F_MAX = 300.0                       # N, hypothetical, same for every muscle
MUSCLES = ["sf", "se", "bf", "be", "ef", "ee", "wf", "we", "fwf", "fwe"]
BOUNDS = [(0, F_MAX)] * 10
OUT_DIR = "figures"


def max_hand_force():
    """Largest hand force in the task direction reachable with forces in [0, F_MAX]."""
    # unknowns: 10 muscle forces and the scale s of the 10 N task; maximize s
    cost = np.zeros(11)
    cost[-1] = -1
    a_eq = np.hstack([Jm.T, -TAU_REQUIRED[:, None]])
    result = linprog(cost, A_eq=a_eq, b_eq=np.zeros(3), bounds=BOUNDS + [(0, None)])
    return 10 * result.x[-1]


def feasible_ranges(hand_force):
    """Min and max force of each muscle that still produce the task exactly."""
    tau = TAU_REQUIRED * hand_force / 10
    low, high = np.zeros(10), np.zeros(10)
    for i in range(10):
        e = np.zeros(10)
        e[i] = 1
        low[i] = linprog(e, A_eq=Jm.T, b_eq=tau, bounds=BOUNDS).fun
        high[i] = -linprog(-e, A_eq=Jm.T, b_eq=tau, bounds=BOUNDS).fun
    return low, high


def bounded_search(tau, seed, threshold=0.01, n_candidates=5, step=25, max_iter=30_000):
    """Same random search as search.run_search, with forces clipped to [0, F_MAX]."""
    rng = np.random.default_rng(seed)
    Fm = np.zeros(10)
    for _ in range(max_iter):
        candidates = np.clip(Fm + rng.uniform(-step, step, (n_candidates, 10)), 0, F_MAX)
        errors = np.sum((candidates @ Jm - tau) ** 2, axis=1)   # Jm^T Fm for every candidate
        best = np.argmin(errors)
        Fm = candidates[best]
        if errors[best] <= threshold:
            return Fm
    return None                     # did not converge


def plot_feasible(f_top, path):
    levels = [(10, "#9ecae1", "10 N (task of the paper)"), (round(f_top * 0.7), "#4292c6", None),
              (f_top, "#08306b", None)]
    levels[1] = (levels[1][0], levels[1][1], f"{levels[1][0]} N")
    levels[2] = (levels[2][0], levels[2][1], f"{f_top:.0f} N (maximum)")
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(14, 5.6), gridspec_kw={"width_ratios": [1.5, 1]})
    y = np.arange(10)
    for k, (force, color, label) in enumerate(levels):
        low, high = feasible_ranges(force)
        for i in range(10):
            yy = y[i] + (k - 1) * 0.25
            if high[i] - low[i] < 3:          # no freedom left: a single value
                ax.plot(low[i], yy, "o", color=color, ms=7)
            else:
                ax.plot([low[i], high[i]], [yy, yy], color=color, lw=5, solid_capstyle="butt")
        ax.plot([], [], color=color, lw=5, label=label)
    ax.set_yticks(y, MUSCLES)
    ax.invert_yaxis()
    ax.set_xlim(-5, F_MAX + 5)
    ax.set_xlabel("Feasible muscle force [N]")
    ax.grid(axis="x", alpha=0.3)
    ax.legend(title="Required hand force (dot = a single possible value)", loc="upper center",
              bbox_to_anchor=(0.5, -0.12), ncol=3, fontsize=9, title_fontsize=9, frameon=False)
    ax.set_title("Range of forces each muscle can take\nwhile the task is still performed exactly",
                 fontsize=11)

    forces = np.linspace(0.5, f_top, 50)
    freedom = [np.sum(np.subtract(*feasible_ranges(f)[::-1])) / (10 * F_MAX) for f in forces]
    ax2.plot(forces, 100 * np.array(freedom), color="#08306b", lw=2.5)
    ax2.axvline(10, color="0.5", ls=":", lw=1)
    ax2.text(10.5, 8, "task of\nthe paper", fontsize=9, color="0.4")
    ax2.set_xlabel("Required hand force [N]")
    ax2.set_ylabel("Freedom left [% of the full range]")
    ax2.set_ylim(0, 105)
    ax2.grid(alpha=0.3)
    ax2.set_title("Total freedom of the 10 muscles", fontsize=11)
    for a in (ax, ax2):
        for side in ["top", "right"]:
            a.spines[side].set_visible(False)
    fig.suptitle(f"With a maximum force per muscle ({F_MAX:.0f} N, hypothetical), "
                 "the solution space shrinks as the task gets harder", fontsize=12.5)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_diem_load(results, path):
    center, var, _, _ = diem_stats(10, 0.0, 1.0)
    identical = diem(0.0, center, var, 0.0, 1.0)
    levels = sorted(results)
    colors = plt.cm.Blues(np.linspace(0.35, 0.95, len(levels)))
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={"width_ratios": [1.25, 1]})
    bins = np.linspace(-22, 16, 60)
    for k, (force, color) in enumerate(zip(levels, colors)):
        counts, _ = np.histogram(results[force], bins=bins)
        counts = counts / counts.max() * 0.85
        ax.fill_between(bins[:-1], k, k + counts, step="post", color=color, alpha=0.9)
        ax.text(15.5, k + 0.25, f"{force:.0f} N", fontsize=10, ha="right")
    for x, name in [(0, "random"), (identical, "identical")]:
        ax.axvline(x, color="0.3", ls="--", lw=1)
        ax.text(x + 0.3, len(levels) - 0.05, name, fontsize=9.5)
    ax.set_yticks([])
    ax.set_xlabel("DIEM between pairs of solutions")
    ax.set_xlim(-22, 16)
    ax.set_title("DIEM of 50 solutions at each required hand force\n"
                 "(bottom: easy task, top: close to the maximum)", fontsize=11)
    for side in ["top", "right", "left"]:
        ax.spines[side].set_visible(False)

    med = [np.median(results[f]) for f in levels]
    p5 = [np.percentile(results[f], 5) for f in levels]
    p95 = [np.percentile(results[f], 95) for f in levels]
    ax2.fill_between(levels, p5, p95, color="#9ecae1", alpha=0.6, label="5th–95th percentile")
    ax2.plot(levels, med, "o-", color="#08306b", lw=2.5, label="median")
    for y, name in [(0, "random"), (identical, "identical")]:
        ax2.axhline(y, color="0.3", ls="--", lw=1)
        ax2.text(levels[0], y + 0.6, name, fontsize=9.5)
    ax2.set_xlabel("Required hand force [N]")
    ax2.set_ylabel("DIEM")
    ax2.set_ylim(-23, 16)
    ax2.grid(alpha=0.3)
    ax2.legend(frameon=False, loc="upper right")
    ax2.set_title("The harder the task, the more similar the solutions", fontsize=11)
    for side in ["top", "right"]:
        ax2.spines[side].set_visible(False)
    fig.suptitle(f"Random search with a maximum of {F_MAX:.0f} N per muscle "
                 f"(DIEM range [0, {F_MAX:.0f} N])", fontsize=12.5)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    f_top = max_hand_force()
    print(f"Maximum hand force with {F_MAX:.0f} N per muscle: {f_top:.1f} N")
    plot_feasible(f_top, os.path.join(OUT_DIR, "extension_feasible.png"))

    center, var, _, _ = diem_stats(10, 0.0, 1.0)
    levels = [5, 10, 20, round(f_top * 0.75), round(f_top * 0.92), np.floor(f_top)]
    results = {}
    for force in levels:
        tau = TAU_REQUIRED * force / 10
        solutions = [bounded_search(tau, seed) for seed in range(50)]
        solutions = np.array([s for s in solutions if s is not None])
        values, _ = pairwise_diem(solutions / F_MAX, center, var, 0.0, 1.0)
        results[force] = values
        print(f"{force:5.1f} N: {len(solutions)} runs, DIEM median {np.median(values):6.1f}, "
              f"max {values.max():5.1f}, more similar than random {np.mean(values < 0):.0%}")
    plot_diem_load(results, os.path.join(OUT_DIR, "extension_diem_load.png"))
    print("Saved extension_feasible.png and extension_diem_load.png in figures/")
