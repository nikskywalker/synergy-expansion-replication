import numpy as np
from kinematics import forward_kinematics, jacobian

elbow, hand = forward_kinematics(np.deg2rad(0), np.deg2rad(90))
jac1 = jacobian(np.deg2rad(0), np.deg2rad(90))
jac2 = jacobian(np.deg2rad(90), np.deg2rad(0))

expected_elbow = np.array([1., 0.]) 
expected_hand = np.array([1., 1.])
expected_jac1 = np.array([[-1, -1], 
                          [1, 0]])
expected_jac2 = np.array([[-2, -1],
                          [0, 0]])

assert np.allclose(elbow, expected_elbow)
assert np.allclose(hand, expected_hand)
assert np.allclose(jac1, expected_jac1)
assert np.allclose(jac2, expected_jac2)

print('Test passed.')



