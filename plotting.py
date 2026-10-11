import matplotlib.pyplot as plt
from kinematics import forward_kinematics
import numpy as np

def plot_arm(q1, q2, l1=1, l2=1):
    elbow, hand = forward_kinematics(q1, q2, l1, l2)
    shoulder = np.array([0, 0])

    x = [shoulder[0], elbow[0], hand[0]]
    y = [shoulder[1], elbow[1], hand[1]]

    plt.plot(x, y, 'o-')
    plt.axis('equal')
    plt.grid(True)
    plt.show()

