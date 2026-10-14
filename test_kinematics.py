import numpy as np
from kinematics import forward_kinematics, jacobian, numerical_jacobian

# Test1: 0°, 90°, 0° and l1=l2=l3=1
elbow1, wrist1, end_effector1 = forward_kinematics(np.deg2rad(0), np.deg2rad(90), np.deg2rad(0))
jac1 = jacobian(np.deg2rad(0), np.deg2rad(90), np.deg2rad(0))

expected_elbow1 = np.array([1., 0.])
expected_wrist1 = np.array([1., 1.]) 
expected_end_effector1 = np.array([1., 2.])

expected_jac1 = np.array([[-2., -2., -1.], [1., 0., 0.]])


assert np.allclose(elbow1, expected_elbow1)
assert np.allclose(wrist1, expected_wrist1)
assert np.allclose(end_effector1, expected_end_effector1)
assert np.allclose(jac1, expected_jac1)


# Test2: 0°, 90°, 45° and l1=l2=1 and l3=0
elbow2, wrist2, end_effector2 = forward_kinematics(np.deg2rad(0), np.deg2rad(90), np.deg2rad(45), l3=0)
jac2 = jacobian(np.deg2rad(0), np.deg2rad(90), np.deg2rad(45), l3=0)

expected_elbow2 = np.array([1., 0.])
expected_wrist2 = np.array([1., 1.]) 
expected_end_effector2 = np.array([1., 1.])

expected_jac2 = np.array([[-1, -1, 0],
                          [1, 0, 0]])

assert np.allclose(elbow2, expected_elbow2)
assert np.allclose(wrist2, expected_wrist2)
assert np.allclose(end_effector2, expected_end_effector2)
assert np.allclose(jac2, expected_jac2)


# Test3: analytic vs numerical Jacobian on 100 random configurations
# (random angles in [-pi, pi], random lengths in [0.5, 2])
rng = np.random.default_rng(21)

for i in range(100):
    q1 = rng.uniform(-np.pi, +np.pi)
    q2 = rng.uniform(-np.pi, +np.pi)
    q3 = rng.uniform(-np.pi, +np.pi)
    l1 = rng.uniform(0.5, 2)
    l2 = rng.uniform(0.5, 2)
    l3 = rng.uniform(0.5, 2)

    an_jac = jacobian(q1, q2, q3, l1, l2, l3)
    num_jac = numerical_jacobian(q1, q2, q3, l1, l2, l3)

    assert np.allclose(an_jac, num_jac)


print('Test passed.')



