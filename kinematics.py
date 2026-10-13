import numpy as np

def forward_kinematics(q1, q2, l1=1, l2=1):
    """Forward kinematics of a planar 2-link arm.

    Angles q1, q2 in radians, relative (q2 measured from link 1);
    l1, l2 link lengths. Shoulder at the origin.
    Returns (elbow, hand): joint positions as numpy arrays [x, y].
    """
    link1_x = l1*np.cos(q1)
    link1_y = l1*np.sin(q1)
    link1 = np.array([link1_x, link1_y])

    shoulder = np.array([0, 0])
    elbow = shoulder + link1

    link2_x = l2*np.cos(q1+q2)
    link2_y = l2*np.sin(q1+q2)
    link2 = np.array([link2_x, link2_y])
    
    hand_x = elbow[0] + link2[0]
    hand_y = elbow[1] + link2[1]

    hand = np.array([hand_x, hand_y])

    return elbow, hand


def jacobian(q1, q2, l1=1, l2=1):
    """Analytic 2x2 Jacobian of the hand position w.r.t. the joint angles.

    Angles q1, q2 in radians; l1, l2 link lengths.
    Rows: hand x and y. Columns: derivatives w.r.t. q1 and q2.
    """    
    j11 = -l1*np.sin(q1) - l2*np.sin(q1+q2)
    j12 = -l2*np.sin(q1+q2)
    j21 = l1*np.cos(q1) + l2*np.cos(q1+q2)
    j22 = l2*np.cos(q1+q2)

    return np.array([[j11, j12], [j21, j22]])


def numerical_jacobian(q1, q2, l1=1, l2=1, h=1e-6):
    """Numerical 2x2 Jacobian of the hand position, by central differences.

    Angles q1, q2 in radians; l1, l2 link lengths; h angular step.
    Each column = (hand with joint +h - hand with joint -h) / (2h).
    Used to check the analytic Jacobian.
    """
    _, hand_q1_plus = forward_kinematics(q1+h, q2, l1, l2)
    _, hand_q1_minus = forward_kinematics(q1-h, q2, l1, l2)

    col1 = (hand_q1_plus - hand_q1_minus)/(2*h)

    _, hand_q2_plus = forward_kinematics(q1, q2+h, l1, l2)
    _, hand_q2_minus = forward_kinematics(q1, q2-h, l1, l2)

    col2 = (hand_q2_plus - hand_q2_minus)/(2*h)

    return np.column_stack([col1, col2])



    


    
