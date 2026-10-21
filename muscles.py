import numpy as np


# Muscle Jacobian Jm (10 x 3): rows = muscles, columns = joints
# (shoulder, elbow, wrist). Muscle order as in the paper (Eq. 15):
# sf, se, bf, be, ef, ee, wf, we, fwf, fwe.
#
# Each entry is d(muscle length)/d(joint angle). With a pulley model
# (muscle wrapped on a pulley of radius r at the joint) this equals the
# moment arm r, constant with the angle. The arm posture is fixed anyway,
# so Jm is a constant matrix.
#
# Sign convention: + for flexors, - for extensors, i.e. the sign of the
# torque the muscle produces on that joint, so that tau = Jm^T @ Fm.
# Mono-articular muscles have one nonzero entry, bi-articular (bf, be) two.
#
# Moment arm values are NOT given in the paper or SI Appendix: they are
# our choice (shoulder 0.04 m, elbow 0.03 m, wrist 0.02 m).

Jm = np.array([
    [+0.04,   0,        0   ], # sf
    [-0.04,   0,        0   ], # se
    [+0.04,  +0.03,     0   ], # bf
    [-0.04,  -0.03,     0   ], # be
    [0,      +0.03,     0   ], # ef
    [0,      -0.03,     0   ], # ee
    [0,       0,       +0.02], # wf
    [0,       0,       -0.02], # we
    [0,       0,       +0.02], # fwf
    [0,       0,       -0.02]  # fwe
    ])

def muscle_force_to_torque(Fm):
    """Joint torques produced by the muscle forces.

    Computes tau = Jm^T @ Fm (statics, same reasoning as tau = J^T F).

    Args:
        Fm: muscle forces in N, array of shape (10,), order
            sf, se, bf, be, ef, ee, wf, we, fwf, fwe.

    Returns:
        Joint torques [tau_s, tau_e, tau_w] in Nm, array of shape (3,).
    """
    tau_prod = Jm.T@Fm
    return tau_prod

