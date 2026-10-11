import numpy as np

def forward_kinematics(q1, q2, l1=1, l2=1):

    elbow_x = l1*np.cos(q1)
    elbow_y = l1*np.sin(q1)
    elbow = np.array([elbow_x, elbow_y])

    hand_x = elbow[0] + l2*np.cos(q1+q2)
    hand_y = elbow[1] + l2*np.sin(q1+q2)
    hand = np.array([hand_x, hand_y])

    return elbow, hand