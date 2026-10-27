import numpy as np
import matplotlib.pyplot as plt
from kinematics import force_to_torque
from muscles import muscle_force_to_torque


def compute_error(Fm_c, tau_required):
    """Squared torque error of a candidate muscle force vector (Eq. 16-17).

    Args:
        Fm_c: candidate muscle forces in N, array of shape (10,).
        tau_required: joint torques required by the task, array of shape (3,).

    Returns:
        e = sum over the 3 joints of (tau_prod - tau_required)^2, a float.
    """
    tau_prod = muscle_force_to_torque(Fm_c)
    delta_tau = tau_prod - tau_required
    error = np.sum(delta_tau**2)
    return error


def run_search(seed, tau_required, threshold=0.01, n_candidates=5, step=25):
    """One run of the random search for muscle forces (Eq. 18).

    Starts from relaxed muscles (Fm = 0). At each iteration builds
    n_candidates candidates (Fm + uniform random increment in [-step, step] N
    on every muscle, negative forces set to 0) and keeps the one with the
    lowest error, even if worse than the current Fm (as Fig. 8 suggests).
    Stops when the error drops to the threshold or below.

    Args:
        seed: seed of the random generator (one seed = one "subject").
        tau_required: joint torques required by the task, array of shape (3,).
        threshold: stopping error in N^2 m^2 (paper: 1e-2).
        n_candidates: candidates per iteration (paper: 5).
        step: maximum random increment per muscle in N (paper: 25).

    Returns:
        Fm: final muscle forces (one selectable synergy), array of shape (10,).
        history: list with the error after each iteration.
    """
    rng = np.random.default_rng(seed)
    Fm = np.zeros(10)
    best_e = np.inf
    history = []

    while best_e > threshold:
        best_e = np.inf
        best_Fm = None

        for i in range(n_candidates):
            delta_Fm = rng.uniform(-step, +step, size=10)
            Fm_c = Fm + delta_Fm
            Fm_c = np.maximum(Fm_c, 0)

            e = compute_error(Fm_c, tau_required)

            if e < best_e:
                best_e = e
                best_Fm = Fm_c

        Fm = best_Fm
        history.append(best_e)

    return Fm, history


if __name__ == "__main__":
    # Task: F = 10 N in +x at the end-effector, arm configuration from
    # Fig. 7 (estimated). Required joint torques are unique: tau = J^T F.
    F = np.array([10, 0])
    tau_required = force_to_torque(F, np.deg2rad(30), np.deg2rad(45), np.deg2rad(30),
                                   0.35, 0.30, 0.19)
    initial_error = compute_error(np.zeros(10), tau_required)

    # 50 runs, one per seed, all from Fm = 0 (Fig. 8)
    n_runs = 50
    synergies = []
    histories = []
    for seed in range(n_runs):
        Fm, history = run_search(seed, tau_required)
        synergies.append(Fm)
        histories.append(history)

    synergies = np.array(synergies)          # shape (50, 10): one synergy per row
    np.save("synergies.npy", synergies)      # input for stage 5 (DIEM)

    n_iterations = [len(h) for h in histories]
    print(f"Initial error: {initial_error:.2f}")
    print(f"Iterations: min {min(n_iterations)}, max {max(n_iterations)}, "
          f"median {int(np.median(n_iterations))}")
    print(f"Final errors all <= 0.01: {all(h[-1] <= 0.01 for h in histories)}")

    # Plot like Fig. 8: error vs iterations, one line per run. Each curve
    # starts from the initial error (Fm = 0) at iteration 0.
    # Left: all iterations. Right: first 45 iterations, same range as Fig. 8.
    fig, (ax_all, ax_zoom) = plt.subplots(1, 2, figsize=(11, 4))
    for history in histories:
        errors = [initial_error] + history
        ax_all.plot(range(len(errors)), errors, linewidth=0.8)
        ax_zoom.plot(range(len(errors)), errors, linewidth=0.8)
    ax_all.set_title("All iterations")
    ax_zoom.set_title("First 45 iterations (range of Fig. 8)")
    ax_zoom.set_xlim(0, 45)
    for ax in (ax_all, ax_zoom):
        ax.set_xlabel("Iterations")
        ax.set_ylabel("Error [N$^2$ m$^2$]")
    fig.suptitle("Random search, 50 runs (replica of Fig. 8)")
    fig.tight_layout()
    fig.savefig("fig8_replica.png", dpi=150)
    plt.show()