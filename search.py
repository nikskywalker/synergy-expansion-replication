import numpy as np
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


if __name__ == "__main__":
    # Task: F = 10 N in +x at the end-effector, arm configuration from
    # Fig. 7 (estimated). Required joint torques are unique: tau = J^T F.
    F = np.array([10, 0])
    tau_required = force_to_torque(F, np.deg2rad(30), np.deg2rad(45), np.deg2rad(15),
                                   0.35, 0.30, 0.19)

    # Random search (Eq. 18): start from relaxed muscles (Fm = 0). At each
    # iteration build 5 candidates (Fm + random increment in [-25, 25] N on
    # every muscle, negative forces set to 0) and keep the one with the
    # lowest error, even if worse than the current Fm (as Fig. 8 suggests).
    # Stop when the error drops below 1e-2 N^2 m^2.
    rng = np.random.default_rng(1)
    Fm = np.zeros(10)
    best_e = np.inf

    while best_e > 0.01:
        best_e = np.inf
        best_Fm = None

        for i in range(5):
            delta_Fm = rng.uniform(-25, +25, size=10)
            Fm_c = Fm + delta_Fm
            Fm_c = np.maximum(Fm_c, 0)

            e = compute_error(Fm_c, tau_required)

            if e < best_e:
                best_e = e
                best_Fm = Fm_c

        Fm = best_Fm
        print(best_e)

    # TODO: repeat with 50 different seeds