import numpy as np


def forward_kinematics(q1, q2, q3, l1=1, l2=1, l3=1):
    """Forward kinematics of a planar 3-link arm.

    Angles q1, q2, q3 in radians, relative (q2 measured from link 1 and 
    q3 measured from link 2); l1, l2, l3 link lengths. Shoulder at the origin.
    Returns (elbow, wrist, end_effector): joint positions as numpy arrays [x, y].
    """
    link1 = np.array([l1*np.cos(q1), l1*np.sin(q1)])
    link2 = np.array([l2*np.cos(q1+q2), l2*np.sin(q1+q2)])
    link3 = np.array([l3*np.cos(q1+q2+q3), l3*np.sin(q1+q2+q3)])

    shoulder = np.array([0, 0])
    elbow = np.array([shoulder[0] + link1[0], shoulder[1] + link1[1]])
    wrist = np.array([elbow[0] + link2[0], elbow[1] + link2[1]])    
    end_effector = np.array([wrist[0] + link3[0], wrist[1] + link3[1]])

    return elbow, wrist, end_effector


def jacobian(q1, q2, q3, l1=1, l2=1, l3=1):
    """Analytic 2x3 Jacobian of the end-effector position w.r.t. the joint angles.

    Angles q1, q2, q3 in radians; l1, l2, l3 link lengths.
    Rows: hand x and y. Columns: derivatives w.r.t. q1, q2 and q3.
    """    
    j11 = -l1*np.sin(q1) - l2*np.sin(q1+q2) - l3*np.sin(q1+q2+q3)
    j12 = -l2*np.sin(q1+q2) - l3*np.sin(q1+q2+q3)
    j13 = -l3*np.sin(q1+q2+q3)
    j21 = l1*np.cos(q1) + l2*np.cos(q1+q2) + l3*np.cos(q1+q2+q3)
    j22 = l2*np.cos(q1+q2) + l3*np.cos(q1+q2+q3)
    j23 = l3*np.cos(q1+q2+q3)

    return np.array([[j11, j12, j13], [j21, j22, j23]])


def numerical_jacobian(q1, q2, q3, l1=1, l2=1, l3=1, h=1e-6):
    """Numerical 2x3 Jacobian of the end-effector position, by central differences.

    Angles q1, q2, q3 in radians; l1, l2, l3 link lengths; h angular step.
    Each column = (ee with joint +h - ee with joint -h) / (2h).
    Used to check the analytic Jacobian.
    """
    _, _, ee_q1_plus = forward_kinematics(q1+h, q2, q3, l1, l2, l3)
    _, _, ee_q1_minus = forward_kinematics(q1-h, q2, q3, l1, l2, l3)

    col1 = (ee_q1_plus - ee_q1_minus)/(2*h)

    _, _, ee_q2_plus = forward_kinematics(q1, q2+h, q3, l1, l2, l3)
    _, _, ee_q2_minus = forward_kinematics(q1, q2-h, q3, l1, l2, l3)

    col2 = (ee_q2_plus - ee_q2_minus)/(2*h)

    _, _, ee_q3_plus = forward_kinematics(q1, q2, q3+h, l1, l2, l3)
    _, _, ee_q3_minus = forward_kinematics(q1, q2, q3-h, l1, l2, l3)

    col3 = (ee_q3_plus - ee_q3_minus)/(2*h)

    return np.column_stack([col1, col2, col3])


def force_to_torque(F, q1, q2, q3, l1=1, l2=1, l3=1):
    """Joint torques needed to exert a force F at the end-effector (statics).

    Computes tau = J^T F, with J the 2x3 Jacobian at the given configuration.
    F: force at the end-effector [Fx, Fy] in N. Angles q1, q2, q3 in radians;
    l1, l2, l3 link lengths. Returns the torques [tau_s, tau_e, tau_w].
    """
    J = jacobian(q1, q2, q3, l1, l2, l3)
    tau = J.T @ F
    return tau










    
