import numpy as np
import matplotlib.pyplot as plt
from kinematics import forward_kinematics

def plot_arm(q1, q2, q3, l1=1, l2=1, l3=1):
    """Plot the planar 3-link arm in its current configuration.

    Angles q1, q2, q3 in radians, relative; l1, l2, l3 link lengths.
    Draws the chain shoulder -> elbow -> wrist -> end-effector,
    with markers on the joints, equal axis scaling and a grid.
    """
    elbow, wrist, end_effector = forward_kinematics(q1, q2, q3, l1, l2, l3)
    shoulder = np.array([0, 0])

    x = [shoulder[0], elbow[0], wrist[0], end_effector[0]]
    y = [shoulder[1], elbow[1], wrist[1], end_effector[1]]

    plt.plot(x, y, 'o-')
    plt.axis('equal')
    plt.grid(True)
    plt.show()

