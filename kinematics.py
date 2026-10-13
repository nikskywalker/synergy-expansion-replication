import numpy as np

def forward_kinematics(q1, q2, l1=1, l2=1):

    link1_x = l1*np.cos(q1)
    link1_y = l1*np.sin(q1)
    link1 = np.array([link1_x, link1_y])

    link2_x = l2*np.cos(q1+q2)
    link2_y = l2*np.sin(q1+q2)
    link2 = np.array([link2_x, link2_y])
    
    hand_x = link1[0] + link2[0]
    hand_y = link1[1] + link2[1]


    hand = np.array([hand_x, hand_y])

    return link1, hand


def jacobian(q1, q2, l1=1, l2=1):
    """Jacobiana 2x2 della posizione della mano rispetto agli angoli dei giunti.

Angoli q1, q2 in radianti; l1, l2 lunghezze dei link.
Righe: x e y della mano. Colonne: derivate rispetto a q1 e q2.
"""
    j11 = -l1*np.sin(q1) - l2*np.sin(q1+q2)
    j12 = -l2*np.sin(q1+q2)
    j21 = l1*np.cos(q1) + l2*np.cos(q1+q2)
    j22 = l2*np.cos(q1+q2)

    return np.array([[j11, j12], [j21, j22]])
