import numpy as np
from kinematics import forward_kinematics, jacobian, numerical_jacobian

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

rng = np.random.default_rng(21)

for i in range(100):
    q1 = rng.uniform(-np.pi, +np.pi)
    q2 = rng.uniform(-np.pi, +np.pi)
    l1 = rng.uniform(0.5, 2)
    l2 = rng.uniform(0.5, 2)

    an_jac = jacobian(q1, q2, l1, l2)
    num_jac = numerical_jacobian(q1, q2, l1, l2)

    assert np.allclose(an_jac, num_jac)


print('Test passed.')



