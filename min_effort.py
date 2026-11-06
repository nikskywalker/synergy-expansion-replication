"""Minimum-effort muscle forces: the reference solution of the paper.

Among all muscle forces Fm that produce exactly the required joint torques
(Jm^T Fm = tau), the minimum-effort solution is the one with the smallest
effort sum(Fm_i^2). The paper computes it with a constrained Moore-Penrose
inverse, so that all muscle forces are positive (muscles can only pull).

Two steps:
    1. Plain pseudoinverse: exact and minimum effort, but ignores the sign,
       so some muscles end up "pushing" (negative forces). Shown as a check.
    2. Same problem with the constraint Fm >= 0, solved numerically (SLSQP).

The constrained solution is added to the 50 solutions of the random search
as the 51st synergy (synergies_51.npy), the input for the DIEM comparison.

Run after search.py (it needs synergies.npy):
    python min_effort.py
"""

import numpy as np
from scipy.optimize import minimize

from kinematics import force_to_torque
from muscles import Jm
from search import compute_error


# Task, same as in search.py: 10 N in +x at the hand, posture from Fig. 7.
F = np.array([10, 0])
Q = np.deg2rad([30, 45, 30])
L = (0.35, 0.30, 0.20)
TAU_REQUIRED = force_to_torque(F, *Q, *L)

MUSCLES = ["sf", "se", "bf", "be", "ef", "ee", "wf", "we", "fwf", "fwe"]


def effort(Fm):
    """Muscle effort: sum of the squared muscle forces, in N^2."""
    return np.sum(Fm**2)


def pseudoinverse_solution(tau_required):
    """Exact, minimum-effort muscle forces, without the sign constraint.

    Args:
        tau_required: joint torques required by the task, array of shape (3,).

    Returns:
        Muscle forces in N, array of shape (10,). Some can be negative.
    """
    return np.linalg.pinv(Jm.T) @ tau_required


def min_effort_solution(tau_required):
    """Exact, minimum-effort muscle forces with all forces >= 0.

    Solves: minimize effort(Fm)
            subject to Jm^T Fm - tau_required = 0   (the task, exactly)
            and        Fm >= 0                       (muscles only pull)

    Args:
        tau_required: joint torques required by the task, array of shape (3,).

    Returns:
        Muscle forces in N, array of shape (10,), all >= 0.
    """
    task = {"type": "eq", "fun": lambda Fm: Jm.T @ Fm - tau_required}
    bounds = [(0, None)] * 10          # each force: at least 0, no maximum
    result = minimize(effort, x0=np.zeros(10), method="SLSQP",
                      bounds=bounds, constraints=task)
    if not result.success:
        raise RuntimeError(f"Optimization failed: {result.message}")
    return result.x


def print_solution(name, Fm):
    """Print muscle forces, torque error and effort of one solution."""
    print(f"\n{name}")
    for muscle, force in zip(MUSCLES, Fm):
        print(f"  {muscle:>4}: {force:8.2f} N")
    print(f"  torque error: {compute_error(Fm, TAU_REQUIRED):.2e} N^2 m^2")
    print(f"  effort:       {effort(Fm):.1f} N^2")


if __name__ == "__main__":
    Fm_pinv = pseudoinverse_solution(TAU_REQUIRED)
    Fm_min = min_effort_solution(TAU_REQUIRED)
    print_solution("Pseudoinverse (no sign constraint)", Fm_pinv)
    print_solution("Minimum effort with Fm >= 0 (the reference)", Fm_min)

    # The 51st synergy: the 50 random-search solutions plus the reference.
    synergies = np.load("synergies.npy")                      # shape (50, 10)
    synergies_51 = np.vstack([synergies, Fm_min])            # shape (51, 10)
    np.save("synergies_51.npy", synergies_51)

    # Good-enough solutions accept a torque error up to 0.01, so some of them
    # can need less effort than the exact minimum-effort solution.
    efforts = np.array([effort(Fm) for Fm in synergies])
    n_lower = np.sum(efforts < effort(Fm_min))
    print(f"\nRandom-search solutions with lower effort than the reference: "
          f"{n_lower} of {len(synergies)}")
    print(f"Saved synergies_51.npy, shape {synergies_51.shape}")
