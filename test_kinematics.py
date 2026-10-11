import numpy as np
from kinematics import forward_kinematics as fk

elbow, hand = fk(np.deg2rad(0), np.deg2rad(90))

expected_elbow = np.array([1., 0.]) 
expected_hand = np.array([1., 1.])

assert np.allclose(elbow, expected_elbow)
assert np.allclose(hand, expected_hand)

print('Test passed.')

