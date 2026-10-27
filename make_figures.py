"""Figures for the README.

Generates, in the folder figures/:
    task.png       the arm in the task posture with the required hand force
    solutions.png  the muscle forces of the most different solutions found
    learning.gif   the random search in action for three runs

Run after search.py (it needs synergies.npy):
    python make_figures.py
"""

import os

import matplotlib
matplotlib.use("Agg")  # save files only, no windows
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np

from kinematics import forward_kinematics, force_to_torque
from muscles import Jm, muscle_force_to_torque
from search import compute_error


# Task, same as in search.py: 10 N in +x at the hand, posture from Fig. 7.
F = np.array([10, 0])
Q = np.deg2rad([30, 45, 30])
L = (0.35, 0.30, 0.19)
TAU_REQUIRED = force_to_torque(F, *Q, *L)

MUSCLES = ["sf", "se", "bf", "be", "ef", "ee", "wf", "we", "fwf", "fwe"]
OUT_DIR = "figures"


def draw_ground(ax, point, size=0.05):
    """Fixed-support symbol under a point: triangle, base line and hatching."""
    x, y = point
    triangle = plt.Polygon([[x, y], [x - size / 2, y - size], [x + size / 2, y - size]],
                           closed=True, facecolor="0.85", edgecolor="0.25",
                           linewidth=1.5, zorder=0)
    ax.add_patch(triangle)
    base_y = y - size
    ax.plot([x - size, x + size], [base_y, base_y], color="0.25", linewidth=1.5)
    for xi in np.linspace(x - size, x + size, 7):
        ax.plot([xi, xi - size / 3], [base_y, base_y - size / 3],
                color="0.25", linewidth=1)


def plot_task(path):
    """Arm in the task posture, fixed at the shoulder, required force at the hand."""
    elbow, wrist, hand = forward_kinematics(*Q, *L)
    shoulder = np.array([0.0, 0.0])
    points = np.array([shoulder, elbow, wrist, hand])

    fig, ax = plt.subplots(figsize=(6, 6))
    draw_ground(ax, shoulder)
    ax.plot(points[:, 0], points[:, 1], "-", color="0.25", linewidth=5,
            solid_capstyle="round", zorder=1)
    ax.plot(points[:, 0], points[:, 1], "o", color="white",
            markeredgecolor="0.25", markersize=11, markeredgewidth=2, zorder=2)
    for point, name in zip(points, ["Shoulder (S)", "Elbow (E)", "Wrist (W)", "Hand (EE)"]):
        ax.annotate(name, point, textcoords="offset points", xytext=(-10, 12),
                    ha="right", fontsize=10)

    arrow_scale = 0.02  # m of arrow per N of force
    tip = hand + arrow_scale * F
    ax.annotate("", xy=tip, xytext=hand,
                arrowprops=dict(arrowstyle="-|>", color="tab:red", lw=2.5,
                                mutation_scale=20))
    ax.text(*(hand + arrow_scale * F / 2 + np.array([0, 0.03])), "F = 10 N",
            color="tab:red", ha="center", fontsize=11)

    x_min = min(points[:, 0].min(), tip[0]) - 0.15
    x_max = max(points[:, 0].max(), tip[0]) + 0.15
    y_min = min(points[:, 1].min(), tip[1]) - 0.1
    y_max = max(points[:, 1].max(), tip[1]) + 0.15
    ax.set_aspect("equal")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.grid(True, alpha=0.3)
    ax.set_title("Task: push with 10 N at the hand, posture fixed\n"
                 "3 joints, 10 muscles")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def most_different(solutions, n):
    """Indices of n solutions far apart from each other (farthest-point sampling)."""
    dist = np.linalg.norm(solutions[:, None, :] - solutions[None, :, :], axis=2)
    i, j = np.unravel_index(np.argmax(dist), dist.shape)
    chosen = [int(i), int(j)]
    while len(chosen) < n:
        closest = dist[:, chosen].min(axis=1)
        chosen.append(int(np.argmax(closest)))
    return chosen


MUSCLE_GROUPS = {  # colour by the joint the muscle acts on
    "shoulder": (["sf", "se"], "tab:blue"),
    "shoulder + elbow (bi-articular)": (["bf", "be"], "tab:purple"),
    "elbow": (["ef", "ee"], "tab:orange"),
    "wrist": (["wf", "we", "fwf", "fwe"], "tab:green"),
}


def plot_solutions(synergies, runs, path):
    """One row per solution: muscle forces on the left (all different),
    joint torques they produce on the right (all the same)."""
    colors = []
    for name in MUSCLES:
        for muscles, color in MUSCLE_GROUPS.values():
            if name in muscles:
                colors.append(color)

    n = len(runs)
    fig, axes = plt.subplots(n, 2, figsize=(11, 1.9 * n + 1.2),
                             gridspec_kw={"width_ratios": [3, 1]},
                             sharex="col")
    f_max = 1.08 * synergies[runs].max()
    t_lim = 1.25 * np.abs(TAU_REQUIRED).max()
    x_m = np.arange(len(MUSCLES))
    x_t = np.arange(3)

    for row, run in enumerate(runs):
        Fm = synergies[run]
        tau = muscle_force_to_torque(Fm)
        ax_f, ax_t = axes[row]

        ax_f.bar(x_m, Fm, color=colors)
        ax_f.set_ylim(0, f_max)
        ax_f.set_ylabel(f"Solution {row + 1}\n[N]")
        ax_f.text(1.0, 1.03, f"total muscle force {Fm.sum():.0f} N",
                  transform=ax_f.transAxes, ha="right", va="bottom", fontsize=9)
        ax_f.grid(True, axis="y", alpha=0.3)

        ax_t.bar(x_t, tau, color="0.55", width=0.6)
        ax_t.scatter(x_t, TAU_REQUIRED, marker="_", s=900, color="tab:red",
                     linewidths=2.5, zorder=3)
        ax_t.axhline(0, color="0.25", linewidth=0.8)
        ax_t.set_ylim(-t_lim, 0.3 * t_lim)
        ax_t.set_ylabel("[N m]")
        ax_t.grid(True, axis="y", alpha=0.3)

    axes[0, 0].set_title("Muscle forces: all different", pad=18)
    axes[0, 1].set_title("Joint torques: all the same\n(grey = produced, red = required)",
                         fontsize=10, pad=8)
    axes[-1, 0].set_xticks(x_m, MUSCLES)
    axes[-1, 0].set_xlabel("Muscle")
    axes[-1, 1].set_xticks(x_t, ["shoulder", "elbow", "wrist"])

    handles = [plt.Rectangle((0, 0), 1, 1, color=color)
               for _, color in MUSCLE_GROUPS.values()]
    fig.legend(handles, list(MUSCLE_GROUPS), loc="lower center", ncol=4,
               fontsize=9, frameon=False)
    fig.suptitle(f"{n} of the {len(synergies)} solutions found by the random search: "
                 "different muscle patterns, same task", fontsize=12)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run_search_traced(seed, threshold=0.01, n_candidates=5, step=25):
    """Same algorithm and random numbers as search.run_search, but it also
    records the muscle forces at every iteration (iteration 0 = all zero).

    Returns:
        forces: array (n_iterations + 1, 10)
        errors: array (n_iterations + 1,)
    """
    rng = np.random.default_rng(seed)
    Fm = np.zeros(10)
    forces = [Fm]
    errors = [compute_error(Fm, TAU_REQUIRED)]
    best_e = np.inf

    while best_e > threshold:
        best_e = np.inf
        best_Fm = None
        for _ in range(n_candidates):
            Fm_c = np.maximum(Fm + rng.uniform(-step, +step, size=10), 0)
            e = compute_error(Fm_c, TAU_REQUIRED)
            if e < best_e:
                best_e = e
                best_Fm = Fm_c
        Fm = best_Fm
        forces.append(Fm)
        errors.append(best_e)

    return np.array(forces), np.array(errors)


def make_gif(runs, path, fps=8):
    """Animated bars: three learners start from the same point and end apart."""
    traces = [run_search_traced(seed) for seed in runs]
    n_max = max(len(errors) for _, errors in traces)

    # Every iteration at the start, where most of the change happens,
    # then fewer and fewer frames towards the end of the longest run.
    frames = np.unique(np.concatenate([
        np.arange(0, 40),
        np.round(np.geomspace(40, n_max - 1, 50)),
    ])).astype(int)
    frames = np.concatenate([frames, np.full(2 * fps, n_max - 1)])  # hold the end

    y_max = 1.1 * max(forces.max() for forces, _ in traces)
    x = np.arange(len(MUSCLES))

    fig, axes = plt.subplots(1, len(runs), figsize=(12, 4), sharey=True)
    bars, labels = [], []
    for ax, run in zip(axes, runs):
        bars.append(ax.bar(x, np.zeros(len(MUSCLES)), color="tab:blue"))
        labels.append(ax.text(0.5, 0.95, "", transform=ax.transAxes,
                              ha="center", va="top", fontsize=10))
        ax.set_xticks(x, MUSCLES, fontsize=8)
        ax.set_ylim(0, y_max)
        ax.set_title(f"Run {run}")
        ax.grid(True, axis="y", alpha=0.3)
    axes[0].set_ylabel("Muscle force [N]")
    title = fig.suptitle("")

    def update(frame):
        title.set_text("Random search: same start, same task, different solutions"
                       f"   (iteration {frame})")
        for (forces, errors), run_bars, label in zip(traces, bars, labels):
            k = min(frame, len(errors) - 1)  # finished runs stay on their solution
            for bar, height in zip(run_bars, forces[k]):
                bar.set_height(height)
            done = "  (done)" if k == len(errors) - 1 else ""
            label.set_text(f"error {errors[k]:.3f}{done}")
        return []

    fig.tight_layout()
    animation = FuncAnimation(fig, update, frames=frames, blit=False)
    animation.save(path, writer=PillowWriter(fps=fps), dpi=80)
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    synergies = np.load("synergies.npy")

    # Sanity check: the saved solutions really solve the task.
    torques = synergies @ Jm  # same as Jm.T @ Fm for every row
    assert np.allclose(torques[0], muscle_force_to_torque(synergies[0]))
    assert all(compute_error(Fm, TAU_REQUIRED) <= 0.01 for Fm in synergies)

    runs = most_different(synergies, 4)

    plot_task(os.path.join(OUT_DIR, "task.png"))
    plot_solutions(synergies, runs, os.path.join(OUT_DIR, "solutions.png"))
    make_gif(runs[:3], os.path.join(OUT_DIR, "learning.gif"))
    print(f"Saved task.png, solutions.png and learning.gif in {OUT_DIR}/ "
          f"(runs shown: {runs})")